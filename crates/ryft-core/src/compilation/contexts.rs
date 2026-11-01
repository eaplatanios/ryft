//! Compilation domain contracts and the shared cache of compiled programs.
//!
//! [`CompilationDomain`] is the boundary between backend-neutral compilation machinery and a concrete backend, and its
//! optional extensions add transactional external state ([`StatefulCompilationDomain`]), compiled program caching
//! ([`CompilationCacheDomain`]), and compiled program analysis ([`AnalyzableCompilationDomain`]).
//! [`CompilationContext`] is the cache of compiled programs that caching backends share across compilations, which
//! coordinates same key compilations and resolves misses through the optional persistent and exchange tiers.

// TODO(eaplatanios): Review this module.

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
use crate::compilation::function::{
    CallRequest, CompileRequest, CompiledFunction, ExecutableFunction, LoweredFunction, LoweringRequest,
    ReferenceExecution, StageRequest, StagedFunction,
};
use crate::contexts::{Domain, StagingContext};
use crate::parameters::Parameterized;
use crate::programs::{ProgramError, Type};
use crate::tracing::DomainTracingContext;

/// [`CompilationDomain`]s are [`Domain`]s that support lowering, compiling, and executing staged programs.
/// Compilation is deliberately split into four semantic stages:
///
///   1. [`Self::stage`] traces a closure using options fixed before tracing.
///   2. [`Self::lower`] transforms the staged artifact into [`Self::LoweredProgram`].
///   3. [`Self::compile`] transforms the lowered artifact into [`Self::CompiledProgram`].
///   4. [`Self::call`] executes an [`ExecutableFunction`] against structured runtime inputs.
///
/// The domain is the active service for every transition, while staged, lowered, compiled, and executable artifacts
/// remain passive values. Each function owns its complete backend-specific transition, including any validation,
/// caching, or runtime conversion policy that the backend chooses to apply.
pub trait CompilationDomain: Domain<Constant: CaptureConstant> + Clone {
    /// Backend-defined key for one input specialization of a retained JIT dispatcher.
    type DispatchKey: Clone + Eq + Hash;

    /// Backend-specific lowered program representation produced by [`Self::lower`].
    type LoweredProgram;

    /// Backend-specific compiled program representation produced by [`Self::compile`].
    type CompiledProgram;

    /// Backend-specific compilation options type. Meshes, sharding and layout overrides, donation declarations,
    /// compiler flags, etc., are all represented as part of this type.
    type Options;

    /// Backend-specific error type. Staging and call boundary errors flow through it as [`ProgramError`]s, which is
    /// why it requires [`From<ProgramError>`].
    type Error: std::error::Error + From<ProgramError>;

    /// Derives the effective staged input types and their dispatch key from one runtime input signature.
    ///
    /// Most domains return the input types unchanged and use them as the key. Backends may instead apply a semantics
    /// preserving dispatch policy such as bounded shape bucketing. The returned key must compare equal exactly when the
    /// returned effective types are interchangeable for tracing, lowering, compilation, and calls.
    ///
    /// The effective types are returned as a shared slice so that key construction on the hot path of retained
    /// dispatch never clones the type vector deeply. Domains whose key embeds the effective types (e.g., exact
    /// dispatch) share one allocation between the key and the returned signature, and the caller clones individual
    /// types only on a specialization miss, when it prepares the staging request.
    fn dispatch_signature(
        &self,
        input_types: Vec<Self::Type>,
        options: &Self::Options,
    ) -> Result<(Self::DispatchKey, Arc<[Self::Type]>), Self::Error>;

    /// Traces a fallible function with explicit runtime captures and their symbolic references.
    ///
    /// This is the only staging primitive. Implementations apply all options that affect tracing to the effective
    /// input and output signatures and retain those same options in the returned artifact.
    fn stage<Request>(
        &self,
        request: Request,
    ) -> Result<StagedFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Request: StageRequest<Self>,
        Self: Sized,
        DomainTracingContext<Self, Self::Value>:
            StagingContext<Type = Self::Type, Constant = Self::Constant, Operation = Self::Operation>;

    /// Performs the complete backend-specific lowering transition.
    ///
    /// The returned lowered artifact must be self-describing. Every option, compiler, target, and topology property
    /// that affects compilation must be folded into [`Self::LoweredProgram`] during this transition, because caching
    /// backends derive compiled program identity from the lowered program alone.
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

/// Optional execution capability for compiled functions that bind external mutable references.
///
/// Implementations must keep hidden state results out of [`CallRequest::RuntimeOutput`] and preserve their reference
/// transaction guarantees even when the public output has no leaves or fails reconstruction.
pub trait StatefulCompilationDomain: CompilationDomain {
    /// Performs a stateful structured execution transition that returns a completion bearing result.
    fn call_statefully_async<Request>(
        &self,
        request: Request,
    ) -> ReferenceExecution<Request::RuntimeOutput, Self::Error>
    where
        Self: Sized,
        Request: CallRequest<Self>;

    /// Executes a stateful call and waits for the whole invocation to complete before returning.
    #[inline]
    fn call_statefully<Request>(&self, request: Request) -> Result<Request::RuntimeOutput, Self::Error>
    where
        Self: Sized,
        Request: CallRequest<Self>,
    {
        self.call_statefully_async(request).r#await()
    }
}

/// Optional capability for caching backend-compiled programs in a [`CompilationContext`].
///
/// Cache identity is derived from the lowered program alone. [`CompilationDomain::lower`] folds every option, compiler,
/// target, and topology property that affects compilation into [`CompilationDomain::LoweredProgram`], so key equality
/// always means that compiled programs are interchangeable.
pub trait CompilationCacheDomain: CompilationDomain {
    /// Exact structural cache key for interchangeable compiled programs.
    type CacheKey: Clone + Eq + Hash + Send + Sync + 'static;

    /// Constructs the exact cache key for `program`.
    fn compilation_key(&self, program: &Self::LoweredProgram) -> Result<Self::CacheKey, Self::Error>;

    /// Returns stable canonical bytes for persistent cache identity, or [`None`] when this domain does not support
    /// persistent executable caching.
    ///
    /// These bytes must remain stable across processes that may share the cache and must cover the complete
    /// computation, options, compiler and backend versions, and target topology. Core hashes them only to obtain safe
    /// filenames and does not add any missing semantic state.
    #[inline]
    fn persistent_cache_key(&self, _key: &Self::CacheKey) -> Option<Vec<u8>> {
        None
    }

    /// Serializes a compiled program for persistent caching and artifact exchange. `Ok(None)` means unsupported.
    #[inline]
    fn serialize_program(&self, _program: &Self::CompiledProgram) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(None)
    }

    /// Deserializes a persistent cache or artifact exchange payload. `Ok(None)` means unsupported or incompatible.
    #[inline]
    fn deserialize_program(&self, _bytes: &[u8]) -> Result<Option<Self::CompiledProgram>, Self::Error> {
        Ok(None)
    }
}

/// Optional capability for inspecting a compiled program without recompiling it.
///
/// Analysis is deliberately separate from [`CompilationDomain`] because not every backend or runtime plugin can
/// expose compiler cost and memory information. Implementations should cache immutable backend results when querying
/// them is expensive. Calling this function must never trigger compilation.
pub trait AnalyzableCompilationDomain: CompilationDomain {
    /// Backend-owned typed analysis report.
    type Analysis;

    /// Analyzes the compiled program carried by an [`ExecutableFunction`] without recompiling it.
    fn analyze<Input: Parameterized<Self::Type>, Output: Parameterized<Self::Type>>(
        &self,
        executable_function: &ExecutableFunction<Self, Input, Output>,
    ) -> Result<Self::Analysis, Self::Error>
    where
        Self: Sized;
}

/// Default capacity of the in-memory tier of a [`CompilationContext`]. Use [`CompilationContext::with_capacity`] when
/// a workload needs a different bound.
const DEFAULT_CACHE_CAPACITY: usize = 8192;

/// Default number of structured compilation events retained for diagnostics.
const DEFAULT_EVENT_CAPACITY: usize = 0;

/// Cache tier or lifecycle stage that produced a [`CompilationEvent`].
///
/// Events are emitted only by [`CompilationContext`], so the variants cover exactly the tiers that the shared cache of
/// compiled programs coordinates.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompilationCacheLevel {
    /// In-memory cache of compiled programs, local to the process.
    Memory,

    /// Persistent cache of serialized programs.
    Persistent,

    /// Distributed exchange of serialized programs.
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

/// Stable explanation for a compilation cache miss or failure.
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
    /// Lifecycle stage or cache tier that emitted this event.
    pub level: CompilationCacheLevel,

    /// Result of the operation.
    pub outcome: CompilationCacheOutcome,

    /// Host duration spent in this operation.
    pub duration: Duration,

    /// Stable miss or failure explanation, when applicable.
    pub miss_reason: Option<CompilationMissReason>,
}

/// Snapshot of the activity of one [`CompilationContext`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CompilationCacheStatistics {
    /// Lookups served directly by the in-memory LRU cache.
    pub memory_hits: u64,

    /// Lookups restored from a validated persistent cache entry.
    pub persistent_hits: u64,

    /// Lookups that became the single producer for a missing key.
    pub misses: u64,

    /// Backend producer closures invoked after every cache tier missed.
    pub compilations: u64,

    /// Same key lookups that waited for an in-flight producer.
    pub waits: u64,

    /// Persistent read, write, serialization, or deserialization failures degraded to misses.
    pub persistent_errors: u64,

    /// Lookups restored from a distributed artifact exchange.
    pub exchange_hits: u64,

    /// Follower lookups that waited for a distributed artifact.
    pub exchange_waits: u64,

    /// Compiled artifacts successfully published by the exchange leader.
    pub exchange_publishes: u64,

    /// Distributed exchange, serialization, or deserialization failures.
    pub exchange_errors: u64,

    /// Distributed waits that completed without an artifact before their deadline.
    pub exchange_timeouts: u64,

    /// Distributed misses or failures that fell back to local compilation.
    pub exchange_fallbacks: u64,

    /// Total host duration spent in persistent lookups and deserialization.
    pub persistent_lookup_duration: Duration,

    /// Total host duration spent in in-memory lookups and same key waits.
    pub memory_lookup_duration: Duration,

    /// Total host duration spent in exchange preflights, waits, receptions, and deserialization.
    pub exchange_duration: Duration,

    /// Total host duration spent in backend producer closures.
    pub compilation_duration: Duration,
}

/// Adds `duration` to `counter`, which accumulates nanoseconds and saturates at [`u64::MAX`].
pub(super) fn accumulate_duration(counter: &AtomicU64, duration: Duration) {
    let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
    counter
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |total| Some(total.saturating_add(nanoseconds)))
        .unwrap();
}

/// Lock-free counters underlying [`CompilationCacheStatistics`]. Durations accumulate in nanoseconds.
#[derive(Default)]
struct CompilationCacheStatisticsAccumulator {
    /// Counter underlying [`CompilationCacheStatistics::memory_hits`].
    memory_hits: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::persistent_hits`].
    persistent_hits: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::misses`].
    misses: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::compilations`].
    compilations: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::waits`].
    waits: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::persistent_errors`].
    persistent_errors: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::exchange_hits`].
    exchange_hits: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::exchange_waits`].
    exchange_waits: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::exchange_publishes`].
    exchange_publishes: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::exchange_errors`].
    exchange_errors: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::exchange_timeouts`].
    exchange_timeouts: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::exchange_fallbacks`].
    exchange_fallbacks: AtomicU64,

    /// Nanoseconds underlying [`CompilationCacheStatistics::persistent_lookup_duration`].
    persistent_lookup_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`CompilationCacheStatistics::memory_lookup_duration`].
    memory_lookup_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`CompilationCacheStatistics::exchange_duration`].
    exchange_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`CompilationCacheStatistics::compilation_duration`].
    compilation_nanoseconds: AtomicU64,
}

impl CompilationCacheStatisticsAccumulator {
    /// Returns a snapshot of the counters. The counters are read independently, so a snapshot taken during concurrent
    /// compilations need not correspond to a single point in time.
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
            persistent_lookup_duration: Duration::from_nanos(
                self.persistent_lookup_nanoseconds.load(Ordering::Relaxed),
            ),
            memory_lookup_duration: Duration::from_nanos(self.memory_lookup_nanoseconds.load(Ordering::Relaxed)),
            exchange_duration: Duration::from_nanos(self.exchange_nanoseconds.load(Ordering::Relaxed)),
            compilation_duration: Duration::from_nanos(self.compilation_nanoseconds.load(Ordering::Relaxed)),
        }
    }

    /// Resets every counter to zero.
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
        self.persistent_lookup_nanoseconds.store(0, Ordering::Relaxed);
        self.memory_lookup_nanoseconds.store(0, Ordering::Relaxed);
        self.exchange_nanoseconds.store(0, Ordering::Relaxed);
        self.compilation_nanoseconds.store(0, Ordering::Relaxed);
    }
}

/// State of one in-flight compilation that same key waiters observe.
enum InFlightState<P> {
    /// The producer has not finished yet.
    Pending,

    /// The producer published this program.
    Ready(Arc<P>),

    /// The producer failed, so waiters must retry through a new producer election.
    Failed,
}

/// Coordination point between the producer of one cache key and the waiters for that key.
struct InFlightCompilation<P> {
    /// Current state of the compilation.
    state: Mutex<InFlightState<P>>,

    /// Condition variable notified when `state` leaves [`InFlightState::Pending`].
    ready: Condvar,
}

impl<P> InFlightCompilation<P> {
    /// Creates a pending in-flight compilation.
    fn new() -> Self {
        Self { state: Mutex::new(InFlightState::Pending), ready: Condvar::new() }
    }

    /// Blocks until the producer finishes and returns its program, or [`None`] when the producer failed.
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

    /// Publishes the terminal `state` of this compilation and wakes every waiter.
    fn finish(&self, state: InFlightState<P>) {
        *self.state.lock().expect("in-flight compilation mutex should not be poisoned") = state;
        self.ready.notify_all();
    }
}

/// Artifact exchange that participates in one compilation, together with the persistent key that addresses the
/// exchanged artifact.
#[derive(Copy, Clone)]
struct ActiveExchange<'o> {
    /// Exchange transport.
    exchange: &'o dyn CompilationArtifactExchange,

    /// Stable persistent key bytes that address the exchanged artifact.
    key: &'o [u8],

    /// Follower deadline of the configured exchange policy.
    timeout: Duration,
}

impl ActiveExchange<'_> {
    /// Returns whether this process is the exchange leader, which compiles and publishes artifacts.
    fn is_leader(self) -> bool {
        self.exchange.process_index() == 0
    }
}

/// Process-local cache of compiled programs, generic over the [`CompilationCacheDomain`] backend.
///
/// Construct one [`CompilationContext`] per backend handle at program start and reuse it across compilation requests
/// and any backend-specific helpers that look up entries in the cache.
///
/// The cache is keyed by the structural [`CompilationCacheDomain::CacheKey`] of the domain. Equality of keys, rather
/// than equality of their hashes, guarantees that distinct programs never collide. On a hit, the cached program is
/// returned without invoking the producer closure. On a miss, the producer runs and its result is inserted.
///
/// The in-memory tier uses LRU eviction with a default capacity of `8192` entries, which
/// [`CompilationContext::with_capacity`] overrides.
///
/// An optional [`DiskCache`] tier is configured via [`CompilationContext::with_disk_cache`]. When present, the disk
/// tier is consulted between the in-memory tier and the producer closure. The cache uses
/// [`CompilationCacheDomain::serialize_program`] and [`CompilationCacheDomain::deserialize_program`] to persist and
/// restore programs, and any error from either function is treated as a cache miss for that entry. An optional
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
/// Misses for the same key coordinate one backend producer, while requests for different keys remain concurrent.
/// Every successful source publishes to memory, while persistent writes are best effort and exchange publication is
/// controlled by the configured policy.
#[cfg_attr(doc, ryft_macros::mermaid)]
pub struct CompilationContext<C: CompilationCacheDomain> {
    /// In-memory LRU cache keyed by the structural [`CompilationCacheDomain::CacheKey`] of the domain.
    programs: Mutex<LruCache<C::CacheKey, Arc<C::CompiledProgram>>>,

    /// Producer coordination for each key. Entries exist only while a cache miss is being restored or compiled, so
    /// unrelated keys never wait for the backend work of one another.
    in_flight: Mutex<HashMap<C::CacheKey, Arc<InFlightCompilation<C::CompiledProgram>>>>,

    /// Optional disk-backed persistent tier.
    disk_cache: Option<DiskCache>,

    /// Optional exchange for sharing serialized programs across processes.
    artifact_exchange: Option<Arc<dyn CompilationArtifactExchange>>,

    /// Distributed exchange behavior, which is disabled by default to preserve local compilation semantics.
    artifact_exchange_policy: CompilationArtifactExchangePolicy,

    /// Lock-free counters used for verification and operational observability.
    statistics: CompilationCacheStatisticsAccumulator,

    /// Bounded history of structured events used for diagnostics without a logging dependency.
    recent_events: Mutex<VecDeque<CompilationEvent>>,

    /// Maximum number of entries retained in `recent_events`, where zero disables retention.
    event_capacity: usize,

    /// Optional reporting hook invoked after an event is retained.
    event_reporter: Option<Arc<dyn Fn(&CompilationEvent) + Send + Sync>>,
}

/// Elected producer for one missing cache key.
///
/// Dropping a producer without calling [`Self::finish`] (e.g., when compilation fails or panics) removes its in-flight
/// marker and wakes its waiters, which then retry through a new producer election.
struct InFlightProducer<'o, C: CompilationCacheDomain> {
    /// Compilation context that elected this producer.
    context: &'o CompilationContext<C>,

    /// Cache key owned by this producer, which is taken when the producer finishes.
    cache_key: Option<C::CacheKey>,

    /// In-flight state shared with waiters, which is taken when the producer finishes.
    in_flight: Option<Arc<InFlightCompilation<C::CompiledProgram>>>,
}

impl<'o, C: CompilationCacheDomain> InFlightProducer<'o, C> {
    /// Creates the producer for `cache_key`.
    fn new(
        context: &'o CompilationContext<C>,
        cache_key: C::CacheKey,
        in_flight: Arc<InFlightCompilation<C::CompiledProgram>>,
    ) -> Self {
        Self { context, cache_key: Some(cache_key), in_flight: Some(in_flight) }
    }

    /// Returns the cache key owned by this producer.
    fn cache_key(&self) -> &C::CacheKey {
        self.cache_key.as_ref().unwrap()
    }

    /// Publishes `program` to the in-memory tier and to every waiter, and returns the shared program.
    fn finish(mut self, program: C::CompiledProgram) -> Arc<C::CompiledProgram> {
        let cache_key = self.cache_key.take().unwrap();
        let in_flight = self.in_flight.take().unwrap();
        self.context.finish_success(cache_key, in_flight, program)
    }
}

impl<C: CompilationCacheDomain> Drop for InFlightProducer<'_, C> {
    fn drop(&mut self) {
        // A producer that never finished releases its waiters so that they can elect a new producer.
        if let (Some(cache_key), Some(in_flight)) = (self.cache_key.take(), self.in_flight.take()) {
            self.context.finish_failure(&cache_key, in_flight);
        }
    }
}

impl<C: CompilationCacheDomain> CompilationContext<C> {
    /// Creates a [`CompilationContext`] with the default cache capacity and no persistent or exchange tiers.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CACHE_CAPACITY)
    }

    /// Creates a [`CompilationContext`] whose in-memory tier retains at most `capacity` compiled programs. A zero
    /// `capacity` is clamped to one entry.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            programs: Mutex::new(LruCache::new(capacity)),
            in_flight: Mutex::new(HashMap::new()),
            disk_cache: None,
            artifact_exchange: None,
            artifact_exchange_policy: CompilationArtifactExchangePolicy::Disabled,
            statistics: CompilationCacheStatisticsAccumulator::default(),
            recent_events: Mutex::new(VecDeque::with_capacity(DEFAULT_EVENT_CAPACITY)),
            event_capacity: DEFAULT_EVENT_CAPACITY,
            event_reporter: None,
        }
    }

    /// Attaches a [`DiskCache`] rooted at `directory` as a persistent tier behind the in-memory tier of this context.
    /// On a memory miss, [`Self::get_or_compile`] consults the disk before invoking the supplied producer. Domains that
    /// explicitly support persistence can then serialize sufficiently expensive compilations so that future processes
    /// restart warm.
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] only when the directory itself cannot be opened or created. Later persistent read,
    /// write, or codec errors are recorded in [`Self::statistics`] and degrade to the in-memory path.
    pub fn with_disk_cache(mut self, directory: impl Into<PathBuf>) -> std::io::Result<Self> {
        self.disk_cache = Some(DiskCache::open(directory)?);
        Ok(self)
    }

    /// Attaches an already configured [`DiskCache`], including any custom capacity or write thresholds.
    #[inline]
    pub fn with_configured_disk_cache(mut self, disk_cache: DiskCache) -> Self {
        self.disk_cache = Some(disk_cache);
        self
    }

    /// Attaches the [`DiskCache`] configured by the [`DiskCache::ENV_VAR`] environment variable, if it is set, and
    /// leaves this context unchanged otherwise.
    ///
    /// # Errors
    ///
    /// Returns invalid configuration and filesystem failures reported by [`DiskCache::from_env`].
    pub fn with_disk_cache_from_env(mut self) -> std::io::Result<Self> {
        if let Some(disk_cache) = DiskCache::from_env()? {
            self.disk_cache = Some(disk_cache);
        }
        Ok(self)
    }

    /// Attaches a distributed exchange of compiled artifacts that is used according to `policy`.
    ///
    /// The exchange is used only when the domain supplies stable bytes through
    /// [`CompilationCacheDomain::persistent_cache_key`] and can serialize and deserialize compiled programs. Process
    /// zero compiles and publishes, while follower processes receive and restore. An exchange with a single process
    /// is ignored.
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
    /// The hook runs without holding any internal cache or event buffer lock. It should return promptly so that it
    /// does not add latency to compilations.
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
    /// This is mostly useful for telemetry and for tests that confirm that repeated compilations of the same
    /// structural signature reuse the cached program instead of recompiling it.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.programs.lock().expect("compilation cache mutex should not be poisoned").len()
    }

    /// Returns a lock-free snapshot of the activity counters of this context.
    #[inline]
    pub fn statistics(&self) -> CompilationCacheStatistics {
        self.statistics.snapshot()
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

    /// Removes every entry from the in-memory tier.
    #[inline]
    pub fn clear_cache(&self) {
        self.programs.lock().expect("compilation cache mutex should not be poisoned").clear();
    }

    /// Resets every activity counter without changing cached programs.
    #[inline]
    pub fn clear_statistics(&self) {
        self.statistics.clear();
    }

    /// Removes every retained structured event without changing cache entries or statistics.
    #[inline]
    pub fn clear_events(&self) {
        self.recent_events.lock().expect("compilation event mutex should not be poisoned").clear();
    }

    /// Performs the complete cached compilation transition for `request`.
    ///
    /// This function owns the orchestration that caching backends share. It derives the exact structural key through
    /// [`CompilationCacheDomain::compilation_key`], resolves the shared cache tiers through [`Self::get_or_compile`]
    /// with `produce` as the backend producer, validates that the compiled output types refine the lowered
    /// declaration, and assembles the compiled artifact. Backends that need a different policy can call
    /// [`Self::get_or_compile`] directly.
    ///
    /// # Parameters
    ///
    ///   - `domain`: Compilation domain performing the transition.
    ///   - `request`: Lowered artifact consumed by the transition.
    ///   - `produce`: Backend compilation of the lowered program, invoked only when every cache tier misses.
    ///   - `output_types`: Effective flat output types of a compiled program.
    pub fn compile_request<Request, Produce, OutputTypes>(
        &self,
        domain: &C,
        request: Request,
        produce: Produce,
        output_types: OutputTypes,
    ) -> Result<CompiledFunction<C, Request::Input, Request::Output>, C::Error>
    where
        Request: CompileRequest<C>,
        Produce: FnOnce(&C::LoweredProgram) -> Result<C::CompiledProgram, C::Error>,
        OutputTypes: FnOnce(&C::CompiledProgram) -> Vec<C::Type>,
    {
        let cache_key = domain.compilation_key(request.lowered().lowered_program())?;
        let program = self.get_or_compile(domain, cache_key, || produce(request.lowered().lowered_program()))?;
        let compiled_output_types = output_types(program.as_ref());
        let declared_output_types = request.lowered().output_types();
        if compiled_output_types.len() != declared_output_types.len() {
            return Err(ProgramError::InvalidOutputCount {
                expected: declared_output_types.len(),
                actual: compiled_output_types.len(),
            }
            .into());
        }
        for (declared, actual) in declared_output_types.iter().zip(&compiled_output_types) {
            if !declared.is_refined_by(actual) {
                return Err(ProgramError::InvalidArgument {
                    message: format!("output type {actual} does not refine declared type {declared}"),
                }
                .into());
            }
        }
        Ok(request.into_compiled(program, compiled_output_types))
    }

    /// Returns a shared cached program for `cache_key`, restoring or producing it when absent.
    ///
    /// On a hit, the entry is moved to the most recently used position. On a miss, the new entry is inserted, and the
    /// least recently used entry is evicted if the cache is at capacity.
    ///
    /// Misses for the same key are single flight. Exactly one caller restores or compiles the program while the rest
    /// wait for its result, and different keys may compile concurrently. When a producer fails, its error is returned
    /// only to that caller, and waiters retry through a new producer election.
    ///
    /// Persistent caching is consulted only when both a [`DiskCache`] is attached and
    /// [`CompilationCacheDomain::persistent_cache_key`] returns stable key bytes. Persistent I/O and codec failures are
    /// counted and degraded to misses.
    pub fn get_or_compile<F: FnOnce() -> Result<C::CompiledProgram, C::Error>>(
        &self,
        domain: &C,
        cache_key: C::CacheKey,
        produce: F,
    ) -> Result<Arc<C::CompiledProgram>, C::Error> {
        loop {
            let lookup_start = Instant::now();
            if let Some(program) = self.cached_program(&cache_key) {
                self.statistics.memory_hits.fetch_add(1, Ordering::Relaxed);
                self.record_tier(
                    CompilationCacheLevel::Memory,
                    CompilationCacheOutcome::Hit,
                    None,
                    lookup_start.elapsed(),
                    &self.statistics.memory_lookup_nanoseconds,
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
                        &self.statistics.memory_lookup_nanoseconds,
                    );
                    return Ok(program);
                }
                continue;
            }

            // A prior producer can insert the program after the first lookup above and remove its in-flight marker
            // before this caller elects itself. Checking again after the election closes that narrow handoff window,
            // which would otherwise trigger a duplicate compilation.
            if let Some(program) = self.cached_program(&cache_key) {
                self.in_flight.lock().expect("in-flight cache mutex should not be poisoned").remove(&cache_key);
                in_flight.finish(InFlightState::Ready(Arc::clone(&program)));
                self.statistics.memory_hits.fetch_add(1, Ordering::Relaxed);
                self.record_tier(
                    CompilationCacheLevel::Memory,
                    CompilationCacheOutcome::Hit,
                    None,
                    lookup_start.elapsed(),
                    &self.statistics.memory_lookup_nanoseconds,
                );
                return Ok(program);
            }

            self.statistics.misses.fetch_add(1, Ordering::Relaxed);
            self.record_tier(
                CompilationCacheLevel::Memory,
                CompilationCacheOutcome::Miss,
                Some(CompilationMissReason::NotFound),
                lookup_start.elapsed(),
                &self.statistics.memory_lookup_nanoseconds,
            );
            return self.restore_or_compile(domain, InFlightProducer::new(self, cache_key, in_flight), produce);
        }
    }

    /// Returns the program cached in the in-memory tier for `cache_key`, marking it as most recently used.
    fn cached_program(&self, cache_key: &C::CacheKey) -> Option<Arc<C::CompiledProgram>> {
        self.programs
            .lock()
            .expect("compilation cache mutex should not be poisoned")
            .get(cache_key)
            .map(Arc::clone)
    }

    /// Resolves a memory miss through the persistent tier, the artifact exchange, and finally backend compilation,
    /// publishing the result through `producer`.
    fn restore_or_compile<F: FnOnce() -> Result<C::CompiledProgram, C::Error>>(
        &self,
        domain: &C,
        producer: InFlightProducer<'_, C>,
        produce: F,
    ) -> Result<Arc<C::CompiledProgram>, C::Error> {
        let configured_exchange = self
            .artifact_exchange
            .as_deref()
            .zip(self.artifact_exchange_policy.timeout())
            .filter(|(exchange, _)| exchange.process_count() != 1);
        let persistent_key = (self.disk_cache.is_some() || configured_exchange.is_some())
            .then(|| domain.persistent_cache_key(producer.cache_key()))
            .flatten();
        let persistent = self
            .disk_cache
            .as_ref()
            .zip(persistent_key.as_deref())
            .map(|(disk_cache, key)| (disk_cache, CacheDigest::from_bytes(key)));

        let exchange = match (configured_exchange, persistent_key.as_deref()) {
            (None, _) => None,
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
            (Some((exchange, timeout)), Some(key)) => {
                let exchange = ActiveExchange { exchange, key, timeout };
                self.preflight_exchange(exchange)?.then_some(exchange)
            }
        };

        if let Some((disk_cache, digest)) = &persistent
            && let Some((program, bytes)) = self.restore_from_disk(domain, disk_cache, digest)
        {
            if let Some(exchange) = exchange
                && exchange.is_leader()
            {
                self.publish_to_exchange(exchange, bytes.as_slice())?;
            }
            return Ok(producer.finish(program));
        }

        if let Some(exchange) = exchange
            && !exchange.is_leader()
            && let Some(program) = self.receive_from_exchange(domain, exchange)?
        {
            return Ok(producer.finish(program));
        }

        self.compile_and_publish(domain, producer, persistent, exchange, produce)
    }

    /// Preflights `exchange` and returns whether it participates in this compilation.
    ///
    /// A preflight failure is fatal when the policy requires sharing. Otherwise, it is recorded and the exchange is
    /// deactivated for this compilation.
    fn preflight_exchange(&self, exchange: ActiveExchange<'_>) -> Result<bool, C::Error> {
        let preflight_start = Instant::now();
        let Err(error) = exchange.exchange.preflight(exchange.key, exchange.timeout) else {
            return Ok(true);
        };
        let (counter, miss_reason) = match &error {
            CompilationExchangeError::TimedOut => (&self.statistics.exchange_timeouts, CompilationMissReason::TimedOut),
            CompilationExchangeError::Incompatible { .. } => {
                (&self.statistics.exchange_errors, CompilationMissReason::Incompatible)
            }
            CompilationExchangeError::Failed { .. } => {
                (&self.statistics.exchange_errors, CompilationMissReason::ReadFailed)
            }
        };
        self.record_exchange_miss(
            counter,
            CompilationCacheOutcome::Failed,
            miss_reason,
            preflight_start,
            error.to_string().as_str(),
        )?;
        Ok(false)
    }

    /// Restores a compiled program from the persistent tier, returning its serialized payload alongside it so that an
    /// exchange leader can publish it without serializing it again. Every outcome other than a hit is recorded and
    /// degraded to [`None`].
    fn restore_from_disk(
        &self,
        domain: &C,
        disk_cache: &DiskCache,
        digest: &CacheDigest,
    ) -> Option<(C::CompiledProgram, Vec<u8>)> {
        let lookup_start = Instant::now();
        let (outcome, miss_reason, restored) = match disk_cache.get(digest) {
            Ok(Some(bytes)) => match domain.deserialize_program(bytes.as_slice()) {
                Ok(Some(program)) => (CompilationCacheOutcome::Hit, None, Some((program, bytes))),
                Ok(None) => (CompilationCacheOutcome::Miss, Some(CompilationMissReason::Incompatible), None),
                Err(_) => (CompilationCacheOutcome::Failed, Some(CompilationMissReason::DeserializationFailed), None),
            },
            Ok(None) => (CompilationCacheOutcome::Miss, Some(CompilationMissReason::NotFound), None),
            Err(_) => (CompilationCacheOutcome::Failed, Some(CompilationMissReason::ReadFailed), None),
        };
        match outcome {
            CompilationCacheOutcome::Hit => {
                self.statistics.persistent_hits.fetch_add(1, Ordering::Relaxed);
            }
            CompilationCacheOutcome::Failed => {
                self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
            }
            CompilationCacheOutcome::Miss
            | CompilationCacheOutcome::Wait
            | CompilationCacheOutcome::Succeeded
            | CompilationCacheOutcome::Skipped => {}
        }
        self.record_tier(
            CompilationCacheLevel::Persistent,
            outcome,
            miss_reason,
            lookup_start.elapsed(),
            &self.statistics.persistent_lookup_nanoseconds,
        );
        restored
    }

    /// Waits on a follower process for the artifact that the exchange leader publishes.
    ///
    /// Returns the restored program on success, and [`None`] when the policy permits falling back to local compilation
    /// after a miss, timeout, or failure.
    fn receive_from_exchange(
        &self,
        domain: &C,
        exchange: ActiveExchange<'_>,
    ) -> Result<Option<C::CompiledProgram>, C::Error> {
        self.statistics.exchange_waits.fetch_add(1, Ordering::Relaxed);
        let exchange_start = Instant::now();
        let bytes = match exchange.exchange.receive(exchange.key, exchange.timeout) {
            Ok(Some(bytes)) => bytes,
            Ok(None) | Err(CompilationExchangeError::TimedOut) => {
                self.record_exchange_miss(
                    &self.statistics.exchange_timeouts,
                    CompilationCacheOutcome::Miss,
                    CompilationMissReason::TimedOut,
                    exchange_start,
                    "timed out waiting for a compilation artifact",
                )?;
                return Ok(None);
            }
            Err(CompilationExchangeError::Incompatible { message }) => {
                self.record_exchange_miss(
                    &self.statistics.exchange_errors,
                    CompilationCacheOutcome::Failed,
                    CompilationMissReason::Incompatible,
                    exchange_start,
                    message.as_str(),
                )?;
                return Ok(None);
            }
            Err(CompilationExchangeError::Failed { message }) => {
                self.record_exchange_miss(
                    &self.statistics.exchange_errors,
                    CompilationCacheOutcome::Failed,
                    CompilationMissReason::ReadFailed,
                    exchange_start,
                    message.as_str(),
                )?;
                return Ok(None);
            }
        };
        match domain.deserialize_program(bytes.as_slice()) {
            Ok(Some(program)) => {
                self.statistics.exchange_hits.fetch_add(1, Ordering::Relaxed);
                self.record_tier(
                    CompilationCacheLevel::Exchange,
                    CompilationCacheOutcome::Hit,
                    None,
                    exchange_start.elapsed(),
                    &self.statistics.exchange_nanoseconds,
                );
                Ok(Some(program))
            }
            Ok(None) => {
                self.record_exchange_miss(
                    &self.statistics.exchange_errors,
                    CompilationCacheOutcome::Miss,
                    CompilationMissReason::Incompatible,
                    exchange_start,
                    "received compilation artifact is incompatible",
                )?;
                Ok(None)
            }
            Err(error) => {
                // The domain error is more informative than a generic exchange requirement error, so it is returned
                // directly when the policy does not permit a fallback.
                self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                self.record_tier(
                    CompilationCacheLevel::Exchange,
                    CompilationCacheOutcome::Failed,
                    Some(CompilationMissReason::DeserializationFailed),
                    exchange_start.elapsed(),
                    &self.statistics.exchange_nanoseconds,
                );
                if !self.artifact_exchange_policy.permits_local_fallback() {
                    return Err(error);
                }
                self.statistics.exchange_fallbacks.fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    /// Publishes the serialized `artifact` through the exchange leader.
    ///
    /// Publication failures are fatal only when the policy requires sharing. Publication durations deliberately stay
    /// out of the exchange duration counter, which tracks time spent obtaining artifacts.
    fn publish_to_exchange(&self, exchange: ActiveExchange<'_>, artifact: &[u8]) -> Result<(), C::Error> {
        let publish_start = Instant::now();
        match exchange.exchange.publish(exchange.key, artifact) {
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

    /// Publishes a terminal failure through the exchange leader when it cannot publish an artifact, so that followers
    /// stop waiting for it instead of running into their deadline.
    ///
    /// `reason` explains why no artifact is published. Failures to publish the failure itself are recorded but never
    /// replace the original failure.
    fn publish_failure_to_exchange(&self, exchange: ActiveExchange<'_>, reason: CompilationMissReason, message: &str) {
        let publish_start = Instant::now();
        let outcome = match exchange.exchange.publish_failure(exchange.key, message) {
            Ok(()) => CompilationCacheOutcome::Succeeded,
            Err(_) => {
                self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                CompilationCacheOutcome::Failed
            }
        };
        self.record_event(CompilationEvent {
            level: CompilationCacheLevel::Exchange,
            outcome,
            duration: publish_start.elapsed(),
            miss_reason: Some(reason),
        });
    }

    /// Invokes the backend producer and publishes its program to every configured tier.
    ///
    /// The persistent tier is written only by the exchange leader (or by every process when no exchange is active),
    /// and only when the compilation meets the write thresholds of the [`DiskCache`]. The exchange leader publishes
    /// the serialized program, or a terminal failure whenever it cannot, so that followers stop waiting.
    fn compile_and_publish<F: FnOnce() -> Result<C::CompiledProgram, C::Error>>(
        &self,
        domain: &C,
        producer: InFlightProducer<'_, C>,
        persistent: Option<(&DiskCache, CacheDigest)>,
        exchange: Option<ActiveExchange<'_>>,
        produce: F,
    ) -> Result<Arc<C::CompiledProgram>, C::Error> {
        let leader_exchange = exchange.filter(|exchange| exchange.is_leader());
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
                    &self.statistics.compilation_nanoseconds,
                );
                if let Some(exchange) = leader_exchange {
                    self.publish_failure_to_exchange(
                        exchange,
                        CompilationMissReason::ProducerFailed,
                        error.to_string().as_str(),
                    );
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
            &self.statistics.compilation_nanoseconds,
        );

        let persistent = persistent.filter(|(disk_cache, _)| {
            exchange.is_none_or(ActiveExchange::is_leader) && disk_cache.should_serialize(compile_duration)
        });
        if persistent.is_none() && leader_exchange.is_none() {
            return Ok(producer.finish(program));
        }

        let serialization_start = Instant::now();
        match domain.serialize_program(&program) {
            Ok(Some(bytes)) => {
                if let Some((disk_cache, digest)) = &persistent
                    && disk_cache.should_persist(compile_duration, bytes.len())
                    && disk_cache.put(digest, bytes.as_slice()).is_err()
                {
                    self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
                    self.record_event(CompilationEvent {
                        level: CompilationCacheLevel::Persistent,
                        outcome: CompilationCacheOutcome::Failed,
                        duration: serialization_start.elapsed(),
                        miss_reason: Some(CompilationMissReason::WriteFailed),
                    });
                }
                if let Some(exchange) = leader_exchange {
                    self.publish_to_exchange(exchange, bytes.as_slice())?;
                }
            }
            Ok(None) => {
                let duration = serialization_start.elapsed();
                if persistent.is_some() {
                    self.record_event(CompilationEvent {
                        level: CompilationCacheLevel::Persistent,
                        outcome: CompilationCacheOutcome::Skipped,
                        duration,
                        miss_reason: Some(CompilationMissReason::Unsupported),
                    });
                }
                if let Some(exchange) = leader_exchange {
                    let message = "domain does not support compiled program serialization";
                    self.record_event(CompilationEvent {
                        level: CompilationCacheLevel::Exchange,
                        outcome: CompilationCacheOutcome::Skipped,
                        duration,
                        miss_reason: Some(CompilationMissReason::Unsupported),
                    });
                    self.publish_failure_to_exchange(exchange, CompilationMissReason::Unsupported, message);
                    self.require_exchange_or_record_fallback(message)?;
                }
            }
            Err(error) => {
                let duration = serialization_start.elapsed();
                if persistent.is_some() {
                    self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
                    self.record_event(CompilationEvent {
                        level: CompilationCacheLevel::Persistent,
                        outcome: CompilationCacheOutcome::Failed,
                        duration,
                        miss_reason: Some(CompilationMissReason::WriteFailed),
                    });
                }
                if let Some(exchange) = leader_exchange {
                    self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                    self.record_event(CompilationEvent {
                        level: CompilationCacheLevel::Exchange,
                        outcome: CompilationCacheOutcome::Failed,
                        duration,
                        miss_reason: Some(CompilationMissReason::WriteFailed),
                    });
                    self.publish_failure_to_exchange(
                        exchange,
                        CompilationMissReason::WriteFailed,
                        error.to_string().as_str(),
                    );
                    if !self.artifact_exchange_policy.permits_local_fallback() {
                        return Err(error);
                    }
                    self.statistics.exchange_fallbacks.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        Ok(producer.finish(program))
    }

    /// Records an exchange operation that could not supply an artifact and then applies the fallback policy.
    ///
    /// # Parameters
    ///
    ///   - `counter`: Statistics counter that classifies the miss (e.g., timeouts or errors).
    ///   - `outcome`: Outcome of the exchange operation.
    ///   - `miss_reason`: Stable explanation of the miss.
    ///   - `start`: Instant at which the exchange operation started.
    ///   - `message`: Description of the miss used in the error returned when the policy requires sharing.
    fn record_exchange_miss(
        &self,
        counter: &AtomicU64,
        outcome: CompilationCacheOutcome,
        miss_reason: CompilationMissReason,
        start: Instant,
        message: &str,
    ) -> Result<(), C::Error> {
        counter.fetch_add(1, Ordering::Relaxed);
        self.record_tier(
            CompilationCacheLevel::Exchange,
            outcome,
            Some(miss_reason),
            start.elapsed(),
            &self.statistics.exchange_nanoseconds,
        );
        self.require_exchange_or_record_fallback(message)
    }

    /// Returns an error describing `message` when the exchange policy requires sharing, and otherwise counts a fallback
    /// to local compilation.
    fn require_exchange_or_record_fallback(&self, message: &str) -> Result<(), C::Error> {
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

    /// Accumulates `duration` into `duration_counter` and records one structured event for the operation.
    fn record_tier(
        &self,
        level: CompilationCacheLevel,
        outcome: CompilationCacheOutcome,
        miss_reason: Option<CompilationMissReason>,
        duration: Duration,
        duration_counter: &AtomicU64,
    ) {
        accumulate_duration(duration_counter, duration);
        self.record_event(CompilationEvent { level, outcome, duration, miss_reason });
    }

    /// Retains `event` when event retention is enabled and then reports it to the installed reporter, if any. The
    /// reporter runs after the event buffer lock is released.
    fn record_event(&self, event: CompilationEvent) {
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

    /// Publishes the `program` produced for `cache_key` to the in-memory tier and to the waiters of `in_flight`.
    fn finish_success(
        &self,
        cache_key: C::CacheKey,
        in_flight: Arc<InFlightCompilation<C::CompiledProgram>>,
        program: C::CompiledProgram,
    ) -> Arc<C::CompiledProgram> {
        let program = Arc::new(program);
        self.programs
            .lock()
            .expect("compilation cache mutex should not be poisoned")
            .put(cache_key.clone(), Arc::clone(&program));
        self.in_flight.lock().expect("in-flight cache mutex should not be poisoned").remove(&cache_key);
        in_flight.finish(InFlightState::Ready(Arc::clone(&program)));
        program
    }

    /// Removes the in-flight marker of a failed producer for `cache_key` and releases its waiters.
    fn finish_failure(&self, cache_key: &C::CacheKey, in_flight: Arc<InFlightCompilation<C::CompiledProgram>>) {
        self.in_flight.lock().expect("in-flight cache mutex should not be poisoned").remove(cache_key);
        in_flight.finish(InFlightState::Failed);
    }
}

impl<C: CompilationCacheDomain> Default for CompilationContext<C> {
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

    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::captures::CaptureReference;
    use crate::compilation::function::{CompilationTracer, stage_function};
    use crate::contexts::Domain;
    use crate::programs::ProgramError;

    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct TestCompiledProgram(usize);

    #[derive(Clone, Default)]
    struct TestDomain {
        persistent: bool,
        serialization_fails: bool,
    }

    impl TestDomain {
        fn persistent() -> Self {
            Self { persistent: true, serialization_fails: false }
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
            if self.serialization_fails {
                return Err(ProgramError::MalformedProgram("expected serialization failure".to_string()));
            }
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
        artifacts: Mutex<HashMap<Vec<u8>, Result<Vec<u8>, String>>>,
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
            self.state.artifacts.lock().unwrap().insert(key.to_vec(), Ok(artifact.to_vec()));
            self.state.ready.notify_all();
            Ok(())
        }

        fn publish_failure(&self, key: &[u8], message: &str) -> Result<(), CompilationExchangeError> {
            self.state.artifacts.lock().unwrap().insert(key.to_vec(), Err(message.to_string()));
            self.state.ready.notify_all();
            Ok(())
        }

        fn receive(&self, key: &[u8], timeout: Duration) -> Result<Option<Vec<u8>>, CompilationExchangeError> {
            if self.state.receive_fails.load(Ordering::Relaxed) {
                return Err(CompilationExchangeError::Failed { message: "expected receive failure".to_string() });
            }
            let artifacts = self.state.artifacts.lock().unwrap();
            let (artifacts, _) = self
                .state
                .ready
                .wait_timeout_while(artifacts, timeout, |artifacts| !artifacts.contains_key(key))
                .unwrap();
            match artifacts.get(key).cloned() {
                None => Ok(None),
                Some(Ok(artifact)) => Ok(Some(artifact)),
                Some(Err(message)) => Err(CompilationExchangeError::Failed { message }),
            }
        }
    }

    #[test]
    fn test_compilation_context_with_capacity() {
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

        // The second key evicts the first one, which is then compiled again.
        assert_eq!(producer_calls.load(Ordering::Relaxed), 3);
        assert_eq!(context.cache_size(), 1);

        // A zero capacity is clamped to one entry.
        let context = CompilationContext::<TestDomain>::with_capacity(0);
        context.get_or_compile(&domain, 1, || Ok(TestCompiledProgram(1))).unwrap();
        assert_eq!(context.cache_size(), 1);
    }

    #[test]
    fn test_compilation_context_with_configured_disk_cache() {
        let directory = tempfile::tempdir().unwrap();
        let domain = TestDomain::persistent();
        let disk_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let first_context = CompilationContext::<TestDomain>::new().with_configured_disk_cache(disk_cache);
        first_context.get_or_compile(&domain, 3, || Ok(TestCompiledProgram(17))).unwrap();
        assert_eq!(first_context.disk_cache().unwrap().directory(), directory.path());

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
    fn test_compilation_context_with_configured_disk_cache_unsupported_domain() {
        let directory = tempfile::tempdir().unwrap();
        let context = CompilationContext::<TestDomain>::new()
            .with_configured_disk_cache(DiskCache::open(directory.path()).unwrap());
        context.get_or_compile(&TestDomain::default(), 1, || Ok(TestCompiledProgram(1))).unwrap();

        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(context.statistics().persistent_errors, 0);
    }

    #[test]
    fn test_compilation_context_with_configured_disk_cache_default_write_thresholds() {
        let directory = tempfile::tempdir().unwrap();
        let context = CompilationContext::<TestDomain>::new()
            .with_configured_disk_cache(DiskCache::open(directory.path()).unwrap());
        context.get_or_compile(&TestDomain::persistent(), 1, || Ok(TestCompiledProgram(1))).unwrap();

        // A fast compilation is below the default minimum compilation duration, so it is never serialized.
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[test]
    fn test_compilation_context_with_configured_disk_cache_serialization_failure() {
        let directory = tempfile::tempdir().unwrap();
        let disk_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let context = CompilationContext::<TestDomain>::new()
            .with_configured_disk_cache(disk_cache)
            .with_event_capacity(8);
        let domain = TestDomain { persistent: true, serialization_fails: true };

        // Serialization failures degrade to an in-memory result and are recorded as persistent write failures.
        let program = context.get_or_compile(&domain, 1, || Ok(TestCompiledProgram(3))).unwrap();
        assert_eq!(program.0, 3);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(context.statistics().persistent_errors, 1);
        let last_event = context.recent_events().pop().unwrap();
        assert_eq!(last_event.level, CompilationCacheLevel::Persistent);
        assert_eq!(last_event.outcome, CompilationCacheOutcome::Failed);
        assert_eq!(last_event.miss_reason, Some(CompilationMissReason::WriteFailed));
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange() {
        let state = Arc::new(TestExchangeState::default());
        let policy = CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(5) };
        let follower_context = CompilationContext::<TestDomain>::new()
            .with_artifact_exchange(Arc::new(TestExchange::new(1, Arc::clone(&state))), policy);
        assert_eq!(follower_context.artifact_exchange().unwrap().process_index(), 1);
        assert_eq!(follower_context.artifact_exchange_policy(), policy);
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
    fn test_compilation_context_with_artifact_exchange_publishes_leader_persistent_hit() {
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
                    panic!("follower must restore the persistent artifact of the leader")
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
    fn test_compilation_context_with_artifact_exchange_publishes_leader_compilation_failure() {
        let state = Arc::new(TestExchangeState::default());
        let policy = CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(30) };
        let leader_context = CompilationContext::<TestDomain>::new()
            .with_artifact_exchange(Arc::new(TestExchange::new(0, Arc::clone(&state))), policy);
        let producer_error = ProgramError::MalformedProgram("expected test failure".to_string());
        assert_eq!(
            leader_context.get_or_compile(&TestDomain::persistent(), 9, || Err(producer_error.clone())).err(),
            Some(producer_error.clone()),
        );

        // The follower observes the terminal failure published by the leader instead of waiting for its deadline.
        let follower_context = CompilationContext::<TestDomain>::new()
            .with_artifact_exchange(Arc::new(TestExchange::new(1, state)), policy);
        let start = Instant::now();
        assert_eq!(
            follower_context.get_or_compile(&TestDomain::persistent(), 9, || Ok(TestCompiledProgram(59))).err(),
            Some(ProgramError::InvalidArgument {
                message: format!("compilation artifact exchange requirement was not satisfied: {producer_error}"),
            }),
        );
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(follower_context.statistics().exchange_errors, 1);
        assert_eq!(follower_context.statistics().compilations, 0);
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_publishes_leader_serialization_failure() {
        let state = Arc::new(TestExchangeState::default());
        let leader_context = CompilationContext::<TestDomain>::new().with_artifact_exchange(
            Arc::new(TestExchange::new(0, Arc::clone(&state))),
            CompilationArtifactExchangePolicy::PreferSharing {
                timeout: Duration::from_secs(30),
                fallback_to_local_compile: true,
            },
        );
        let domain = TestDomain { persistent: true, serialization_fails: true };

        // The leader keeps its program when the policy permits a fallback, but still releases its followers.
        let program = leader_context.get_or_compile(&domain, 10, || Ok(TestCompiledProgram(61))).unwrap();
        assert_eq!(program.0, 61);
        assert_eq!(leader_context.statistics().exchange_errors, 1);
        assert_eq!(leader_context.statistics().exchange_fallbacks, 1);
        assert_eq!(leader_context.statistics().exchange_publishes, 0);
        assert_eq!(
            state.artifacts.lock().unwrap().get([10u8].as_slice()),
            Some(&Err(ProgramError::MalformedProgram("expected serialization failure".to_string()).to_string())),
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_receive_failure_fallback() {
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
    fn test_compilation_context_with_artifact_exchange_rejects_preflight_disagreement() {
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

        assert_eq!(
            result.err(),
            Some(ProgramError::InvalidArgument {
                message: "compilation artifact exchange requirement was not satisfied: \
                    compilation artifact exchange is incompatible: expected preflight incompatibility"
                    .to_string(),
            }),
        );
        assert_eq!(producer_calls.load(Ordering::Relaxed), 0);
        assert_eq!(context.statistics().exchange_errors, 1);
        assert_eq!(context.statistics().compilations, 0);
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_rejects_unsupported_domain_without_waiting() {
        let context = CompilationContext::<TestDomain>::new().with_artifact_exchange(
            Arc::new(TestExchange::new(1, Arc::new(TestExchangeState::default()))),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(30) },
        );
        let start = Instant::now();

        let result = context.get_or_compile(&TestDomain::default(), 4, || Ok(TestCompiledProgram(35)));

        assert_eq!(
            result.err(),
            Some(ProgramError::InvalidArgument {
                message: "compilation artifact exchange requirement was not satisfied: \
                    domain does not provide a persistent compilation key"
                    .to_string(),
            }),
        );
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(context.statistics().compilations, 0);
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_required_timeout() {
        let context = CompilationContext::<TestDomain>::new().with_artifact_exchange(
            Arc::new(TestExchange::new(1, Arc::new(TestExchangeState::default()))),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_millis(1) },
        );
        let producer_calls = AtomicUsize::new(0);

        let result = context.get_or_compile(&TestDomain::persistent(), 4, || {
            producer_calls.fetch_add(1, Ordering::Relaxed);
            Ok(TestCompiledProgram(37))
        });

        assert_eq!(
            result.err(),
            Some(ProgramError::InvalidArgument {
                message: "compilation artifact exchange requirement was not satisfied: \
                    timed out waiting for a compilation artifact"
                    .to_string(),
            }),
        );
        assert_eq!(producer_calls.load(Ordering::Relaxed), 0);
        assert_eq!(context.statistics().exchange_timeouts, 1);
        assert_eq!(context.statistics().exchange_fallbacks, 0);
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_permitted_timeout_fallback() {
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
    fn test_compilation_context_with_event_capacity() {
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(2);
        let domain = TestDomain::default();
        context.get_or_compile(&domain, 1, || Ok(TestCompiledProgram(43))).unwrap();
        context.get_or_compile(&domain, 1, || Ok(TestCompiledProgram(47))).unwrap();

        // The memory miss of the first lookup is evicted by the two later events.
        let events = context.recent_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].level, CompilationCacheLevel::Backend);
        assert_eq!(events[0].outcome, CompilationCacheOutcome::Succeeded);
        assert_eq!(events[1].level, CompilationCacheLevel::Memory);
        assert_eq!(events[1].outcome, CompilationCacheOutcome::Hit);

        context.clear_events();
        assert_eq!(context.recent_events(), Vec::new());
    }

    #[test]
    fn test_compilation_context_with_event_reporter() {
        let reported = Arc::new(Mutex::new(Vec::new()));
        let reporter_events = Arc::clone(&reported);
        let context =
            CompilationContext::<TestDomain>::new().with_event_reporter(Arc::new(move |event: &CompilationEvent| {
                reporter_events.lock().unwrap().push(*event)
            }));
        context.get_or_compile(&TestDomain::default(), 1, || Ok(TestCompiledProgram(43))).unwrap();

        // The reporter observes every event even though retention is disabled by default.
        assert_eq!(context.recent_events(), Vec::new());
        let reported = reported.lock().unwrap();
        assert_eq!(
            reported.iter().map(|event| (event.level, event.outcome)).collect::<Vec<_>>(),
            vec![
                (CompilationCacheLevel::Memory, CompilationCacheOutcome::Miss),
                (CompilationCacheLevel::Backend, CompilationCacheOutcome::Succeeded),
            ],
        );
    }

    #[test]
    fn test_compilation_context_clear_cache() {
        let context = CompilationContext::<TestDomain>::new();
        let domain = TestDomain::default();
        context.get_or_compile(&domain, 1, || Ok(TestCompiledProgram(1))).unwrap();
        assert_eq!(context.cache_size(), 1);

        context.clear_cache();
        assert_eq!(context.cache_size(), 0);
        assert_eq!(context.statistics().compilations, 1);

        context.clear_statistics();
        assert_eq!(context.statistics(), CompilationCacheStatistics::default());
    }

    #[test]
    fn test_compilation_context_compile_request() {
        let context = CompilationContext::<TestDomain>::new();
        let domain = TestDomain::default();
        let staged: StagedFunction<TestDomain, ArrayType, ArrayType> =
            stage_function(&domain, |input: CompilationTracer<TestDomain>| input, ArrayType::scalar(DataType::F32), ())
                .unwrap();
        let lowered = domain.lower(staged).unwrap();

        let compiled = context
            .compile_request(
                &domain,
                lowered.clone(),
                |_| Ok(TestCompiledProgram(5)),
                |_| vec![ArrayType::scalar(DataType::F32)],
            )
            .unwrap();
        assert_eq!(compiled.compiled_program(), &TestCompiledProgram(5));
        assert_eq!(compiled.output_types(), &[ArrayType::scalar(DataType::F32)]);

        // The compiled output types must match the lowered declaration in count and refine it.
        assert!(matches!(
            context.compile_request(&domain, lowered.clone(), |_| Ok(TestCompiledProgram(5)), |_| Vec::new()),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        ));
        assert!(matches!(
            context.compile_request(&domain, lowered, |_| Ok(TestCompiledProgram(5)), |_| {
                vec![ArrayType::scalar(DataType::I64)]
            }),
            Err(ProgramError::InvalidArgument { message })
                if message == "output type i64[] does not refine declared type f32[]",
        ));
    }

    #[test]
    fn test_compilation_context_get_or_compile() {
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
    fn test_compilation_context_get_or_compile_producer_error() {
        let context = CompilationContext::<TestDomain>::new();
        let domain = TestDomain::default();

        // Producer errors are returned to their caller and never cached.
        let error = ProgramError::MalformedProgram("expected test failure".to_string());
        assert_eq!(context.get_or_compile(&domain, 1, || Err(error.clone())).err(), Some(error));
        let program = context.get_or_compile(&domain, 1, || Ok(TestCompiledProgram(9))).unwrap();

        assert_eq!(program.0, 9);
        assert_eq!(context.statistics().compilations, 2);
    }

    #[test]
    fn test_compilation_context_get_or_compile_producer_panic() {
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

        // The panicking producer released its in-flight state, so the next lookup elects a new producer.
        let program = context.get_or_compile(&domain, 1, || Ok(TestCompiledProgram(13))).unwrap();
        assert_eq!(program.0, 13);
    }

    #[test]
    fn test_compilation_context_get_or_compile_single_flight() {
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
    fn test_compilation_context_get_or_compile_unrelated_keys_concurrently() {
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

        // Both producers must enter while neither has been released, so unrelated keys never serialize.
        let first = entered_receiver.recv_timeout(Duration::from_secs(5)).unwrap();
        let second = entered_receiver.recv_timeout(Duration::from_secs(5));
        release.store(true, Ordering::Release);
        for thread in threads {
            thread.join().unwrap();
        }

        assert_ne!(first, second.unwrap());
        assert_eq!(context.statistics().compilations, 2);
    }
}
