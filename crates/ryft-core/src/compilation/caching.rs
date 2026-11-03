use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use lru::LruCache;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::compilation::domains::CompilationCacheDomain;
use crate::compilation::exchange::{
    CompilationArtifactExchange, CompilationArtifactExchangePolicy, CompilationExchangeError,
};
use crate::compilation::functions::accumulate_duration;
use crate::programs::ProgramError;

/// Represents errors reported by [`CompilationStep`] outcomes. Each variant classifies a failure during cache access,
/// artifact exchange, or backend compilation. Message payloads capture snapshots of the underlying errors' display
/// text. Compilation APIs preserve the original error types when propagating failures to callers. Lookup misses and
/// skipped operations use [`CompilationMissReason`] and [`CompilationSkipReason`] instead.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CompilationStepError {
    #[error("required compilation artifact operation is unsupported")]
    Unsupported,

    #[error("compilation artifact or exchange compatibility requirements were not satisfied: {message}")]
    Incompatible { message: String },

    #[error("compilation artifact exchange timed out")]
    TimedOut,

    #[error("failed to read or receive a compilation artifact: {message}")]
    ReadFailed { message: String },

    #[error("compilation artifact exchange coordination failed: {message}")]
    ExchangeFailed { message: String },

    #[error("failed to serialize a compiled program: {message}")]
    SerializationFailed { message: String },

    #[error("failed to deserialize a compiled program: {message}")]
    DeserializationFailed { message: String },

    #[error("failed to write or publish a compilation artifact or failure notification: {message}")]
    WriteFailed { message: String },

    #[error("backend compilation failed: {message}")]
    CompilationFailed { message: String },
}

/// Storage tier consulted by a [`CompilationStep::CacheLookup`]. Distributed artifact exchange and backend
/// compilation are separate operations represented by their own [`CompilationStep`] variants.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompilationCacheTier {
    /// In-memory cache of compiled programs, local to the process.
    Memory,

    /// Persistent cache of serialized programs.
    Persistent,
}

/// Reason a lookup found no reusable program without failing.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompilationMissReason {
    /// No entry existed for the requested key.
    NotFound,

    /// An artifact was found but could not be reused by the requesting program or runtime.
    Incompatible,

    /// The configured deadline elapsed without a reusable artifact.
    TimedOut,
}

/// Reason a compilation step was skipped because it is unsupported or does not meet its write thresholds.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompilationSkipReason {
    /// The domain or configured service does not support the requested step.
    Unsupported,

    /// The compilation was faster, or its serialized program smaller, than the write thresholds of the persistent
    /// cache require.
    BelowWriteThreshold,
}

/// Result of looking up a reusable program in a cache or receiving one through an artifact exchange.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompilationLookupOutcome {
    /// A reusable entry was found.
    Hit,

    /// No reusable entry was found.
    Miss(CompilationMissReason),

    /// The lookup or deserialization failed.
    Failed(CompilationStepError),
}

/// Completion result of a backend compilation, persistent storage, or artifact exchange step.
/// Reuse results belong to [`CompilationLookupOutcome`] instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompilationStepOutcome {
    /// The step completed successfully.
    Succeeded,

    /// The step failed.
    Failed(CompilationStepError),

    /// The step was unsupported or below its configured write thresholds.
    Skipped(CompilationSkipReason),
}

/// Step reported by a [`CompilationEvent`], together with its lookup or completion result. A cache hit describes
/// artifact reuse, while successful compilation or publication describes completed work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompilationStep {
    /// Looked up a program in a local cache.
    CacheLookup {
        /// Storage tier that was consulted.
        tier: CompilationCacheTier,

        /// Whether the lookup found a reusable program.
        outcome: CompilationLookupOutcome,
    },

    /// Waited for an in-flight producer, which shared its program. Waits whose producer fails retry without emitting
    /// this event.
    MemoryWait {
        /// Whether the shared program can serve the waiting request.
        outcome: CompilationLookupOutcome,
    },

    /// Validated compatibility before using a distributed artifact exchange.
    ExchangeCompatibilityValidation {
        /// Completion result of the compatibility validation.
        outcome: CompilationStepOutcome,
    },

    /// Attempted to receive and restore an artifact published by the exchange leader.
    ExchangeReceive {
        /// Whether a compatible program was received and deserialized.
        outcome: CompilationLookupOutcome,
    },

    /// Prepared or published a serialized artifact to followers. Preparation can fail or be unsupported during
    /// serialization.
    ExchangePublish {
        /// Completion result of preparing or publishing the artifact.
        outcome: CompilationStepOutcome,
    },

    /// Notified followers that the leader cannot provide an artifact. Successful delivery does not imply that the
    /// original compilation or serialization succeeded.
    ExchangeFailureNotification {
        /// Original failure that prevents the leader from providing an artifact, including required serialization
        /// that the domain does not support.
        cause: CompilationStepError,

        /// Completion result of delivering the failure notification.
        outcome: CompilationStepOutcome,
    },

    /// Prepared or wrote a program to persistent storage. Preparation includes serialization. Exchange followers never
    /// write to persistent storage, so they emit no such events.
    PersistentStore {
        /// Completion result of preparing or writing the artifact.
        outcome: CompilationStepOutcome,
    },

    /// Invoked the backend producer to compile a program after reuse attempts were exhausted.
    BackendCompilation {
        /// Completion result of the producer invocation.
        outcome: CompilationStepOutcome,
    },
}

/// Structured diagnostic event for one [`CompilationStep`] of a [`CompilationContext`]. Contexts retain recent
/// events when configured via [`CompilationContext::with_event_capacity`] and report every event to the hook
/// installed via [`CompilationContext::with_event_reporter`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilationEvent {
    /// Step being reported and its result.
    pub step: CompilationStep,

    /// Host duration spent in this step.
    pub duration: Duration,
}

/// Snapshot of the activity of one [`CompilationContext`] across its cache lookups, artifact exchange, and backend
/// compilation. The statistics accumulate from construction or from the latest [`CompilationContext::clear_statistics`]
/// call and are unaffected by [`CompilationContext::clear_cache`]. Each field is read independently, so a snapshot
/// taken during concurrent compilations need not correspond to a single point in time. Every attempt of a lookup counts
/// as exactly one of [`Self::memory_hits`], [`Self::waits`], and [`Self::misses`], and a waiter whose producer fails
/// retries and counts again. A memory hit or wait whose program [`CompilationCacheDomain::validate_cached_program`]
/// rejects still counts, although it fails the compilation. Durations are host durations summed over all threads,
/// so concurrent compilations can make them exceed the elapsed wall-clock time.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CompilationCacheStatistics {
    /// Lookups served directly by the in-memory LRU cache.
    pub memory_hits: u64,

    /// Same key lookups that waited for an in-flight producer.
    pub waits: u64,

    /// Lookups that became the single producer for a missing key.
    pub misses: u64,

    /// Lookups restored from a validated persistent cache entry.
    pub persistent_hits: u64,

    /// Persistent read, write, serialization, and deserialization failures, none of which fails a compilation.
    pub persistent_errors: u64,

    /// Lookups restored from a distributed artifact exchange.
    pub exchange_hits: u64,

    /// Follower lookups that waited for a distributed artifact.
    pub exchange_waits: u64,

    /// Compiled artifacts successfully published by the exchange leader.
    pub exchange_publishes: u64,

    /// Distributed exchange, serialization, or deserialization failures.
    pub exchange_errors: u64,

    /// Exchange compatibility validations and follower waits that reached their deadline or ended without an artifact.
    pub exchange_timeouts: u64,

    /// Distributed misses or failures that fell back to local compilation.
    pub exchange_fallbacks: u64,

    /// Uncached backend compilations invoked when no reusable program was available.
    pub compilations: u64,

    /// Total host duration spent in in-memory lookups and same key waits.
    pub memory_lookup_duration: Duration,

    /// Total host duration spent in persistent lookups and deserialization.
    pub persistent_lookup_duration: Duration,

    /// Total host duration spent in exchange compatibility validation, waits, receptions, and deserialization.
    pub exchange_duration: Duration,

    /// Total host duration spent in uncached backend compilations.
    pub compilation_duration: Duration,
}

/// Lock-free counters underlying [`CompilationCacheStatistics`]. Durations accumulate in nanoseconds.
#[derive(Default)]
struct CompilationCacheStatisticsAccumulator {
    /// Counter underlying [`CompilationCacheStatistics::memory_hits`].
    memory_hits: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::waits`].
    waits: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::misses`].
    misses: AtomicU64,

    /// Counter underlying [`CompilationCacheStatistics::persistent_hits`].
    persistent_hits: AtomicU64,

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

    /// Counter underlying [`CompilationCacheStatistics::compilations`].
    compilations: AtomicU64,

    /// Nanoseconds underlying [`CompilationCacheStatistics::memory_lookup_duration`].
    memory_lookup_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`CompilationCacheStatistics::persistent_lookup_duration`].
    persistent_lookup_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`CompilationCacheStatistics::exchange_duration`].
    exchange_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`CompilationCacheStatistics::compilation_duration`].
    compilation_nanoseconds: AtomicU64,
}

impl CompilationCacheStatisticsAccumulator {
    /// Returns a snapshot of the counters (refer to [`CompilationCacheStatistics`] for its consistency guarantees).
    fn snapshot(&self) -> CompilationCacheStatistics {
        CompilationCacheStatistics {
            memory_hits: self.memory_hits.load(Ordering::Relaxed),
            waits: self.waits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            persistent_hits: self.persistent_hits.load(Ordering::Relaxed),
            persistent_errors: self.persistent_errors.load(Ordering::Relaxed),
            exchange_hits: self.exchange_hits.load(Ordering::Relaxed),
            exchange_waits: self.exchange_waits.load(Ordering::Relaxed),
            exchange_publishes: self.exchange_publishes.load(Ordering::Relaxed),
            exchange_errors: self.exchange_errors.load(Ordering::Relaxed),
            exchange_timeouts: self.exchange_timeouts.load(Ordering::Relaxed),
            exchange_fallbacks: self.exchange_fallbacks.load(Ordering::Relaxed),
            compilations: self.compilations.load(Ordering::Relaxed),
            memory_lookup_duration: Duration::from_nanos(self.memory_lookup_nanoseconds.load(Ordering::Relaxed)),
            persistent_lookup_duration: Duration::from_nanos(
                self.persistent_lookup_nanoseconds.load(Ordering::Relaxed),
            ),
            exchange_duration: Duration::from_nanos(self.exchange_nanoseconds.load(Ordering::Relaxed)),
            compilation_duration: Duration::from_nanos(self.compilation_nanoseconds.load(Ordering::Relaxed)),
        }
    }

    /// Resets every counter to zero.
    fn clear(&self) {
        self.memory_hits.store(0, Ordering::Relaxed);
        self.waits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.persistent_hits.store(0, Ordering::Relaxed);
        self.persistent_errors.store(0, Ordering::Relaxed);
        self.exchange_hits.store(0, Ordering::Relaxed);
        self.exchange_waits.store(0, Ordering::Relaxed);
        self.exchange_publishes.store(0, Ordering::Relaxed);
        self.exchange_errors.store(0, Ordering::Relaxed);
        self.exchange_timeouts.store(0, Ordering::Relaxed);
        self.exchange_fallbacks.store(0, Ordering::Relaxed);
        self.compilations.store(0, Ordering::Relaxed);
        self.memory_lookup_nanoseconds.store(0, Ordering::Relaxed);
        self.persistent_lookup_nanoseconds.store(0, Ordering::Relaxed);
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

    /// The producer failed, so waiters must retry, and the first of them to miss again becomes the new producer.
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

/// Producer of the in-flight compilation for one missing cache key. The first caller to miss on a key becomes
/// its producer, while later callers for the same key wait for its result. Dropping a producer without calling
/// [`Self::finish`] (e.g., when compilation fails or panics) removes its in-flight marker and wakes its waiters,
/// which then retry, and the first of them to miss again becomes the new producer.
struct InFlightProducer<'o, C: CompilationCacheDomain> {
    /// Compilation context in which this producer registered its in-flight compilation.
    context: &'o CompilationContext<'o, C>,

    /// Cache key owned by this producer, which is taken when the producer finishes.
    cache_key: Option<C::CacheKey>,

    /// In-flight state shared with waiters, which is taken when the producer finishes.
    in_flight: Option<Arc<InFlightCompilation<C::CompiledProgram>>>,
}

impl<'o, C: CompilationCacheDomain> InFlightProducer<'o, C> {
    /// Creates the producer for `cache_key`.
    fn new(
        context: &'o CompilationContext<'o, C>,
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
        let program = Arc::new(program);

        // Publish to memory before releasing the producer slot so a racing lookup can find this program.
        // Release each lock before acquiring the next one or notifying waiters.
        self.context
            .programs
            .lock()
            .expect("compilation cache mutex should not be poisoned")
            .put(cache_key.clone(), Arc::clone(&program));
        self.context
            .in_flight
            .lock()
            .expect("in-flight cache mutex should not be poisoned")
            .remove(&cache_key);
        in_flight.finish(InFlightState::Ready(Arc::clone(&program)));
        program
    }
}

impl<C: CompilationCacheDomain> Drop for InFlightProducer<'_, C> {
    fn drop(&mut self) {
        // A producer that never finished releases its waiters so that one of them can become the new producer.
        if let (Some(cache_key), Some(in_flight)) = (self.cache_key.take(), self.in_flight.take()) {
            // Remove the marker before waking waiters so their retries can elect a new producer.
            self.context
                .in_flight
                .lock()
                .expect("in-flight cache mutex should not be poisoned")
                .remove(&cache_key);
            in_flight.finish(InFlightState::Failed);
        }
    }
}

/// Artifact exchange that participates in one compilation, together with the persistent key that addresses the
/// exchanged artifact.
#[derive(Copy, Clone)]
struct ActiveExchange<'o> {
    /// Exchange transport.
    exchange: &'o dyn CompilationArtifactExchange,

    /// Persistent key bytes identifying the artifact to exchange.
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

/// Default capacity of the in-memory cache tier of a [`CompilationContext`]. Use [`CompilationContext::with_capacity`]
/// when a workload needs a different bound.
const DEFAULT_CACHE_CAPACITY: usize = 8192;

/// Default number of structured compilation events retained for diagnostics, which disables retention until
/// [`CompilationContext::with_event_capacity`] enables it.
const DEFAULT_EVENT_CAPACITY: usize = 0;

/// Process-local cache of compiled programs, generic over the [`CompilationCacheDomain`] backend. Construct one
/// [`CompilationContext`] per backend handle at program start and reuse it across compilations and any backend-specific
/// helpers that look up entries in the cache.
///
/// The cache is keyed by the structural [`CompilationCacheDomain::CacheKey`] of the domain. Equality of keys, rather
/// than equality of their hashes, guarantees that distinct programs never collide. On a hit, the cached program is
/// returned without compiling anything. On a miss, persistent storage or artifact exchange restores the program, or
/// [`CompilationCacheDomain::compile_uncached`] compiles it, and the result is inserted. Every program that a request
/// does not compile itself must pass [`CompilationCacheDomain::validate_cached_program`] for that request.
///
/// The in-memory tier uses LRU eviction with a default capacity of `8192` entries, which
/// [`CompilationContext::with_capacity`] overrides.
///
/// An optional [`DiskCache`] tier is attached via [`CompilationContext::with_disk_cache`] or
/// [`CompilationContext::with_disk_cache_from_env`]. When present, the disk tier is consulted between the
/// in-memory tier and backend compilation. The cache uses [`CompilationCacheDomain::serialize_program`] and
/// [`CompilationCacheDomain::deserialize_program`] to persist and restore programs, and it records failures of either
/// function and of the disk itself without failing the compilation. An optional [`CompilationArtifactExchange`] may
/// provide a serialized program from the leader after local cache misses, according to the configured
/// [`CompilationArtifactExchangePolicy`].
///
/// The attached artifact exchange and event reporter may borrow data that lives for `'o` (e.g., a backend client),
/// so the context cannot outlive that data. A context whose exchange and reporter own all of their state can use
/// `'static`.
///
/// # Cache Resolution
///
/// ```text
/// ┌──────────────────────────┐
/// │ Program and Cache Key    │
/// └─────────────┬────────────┘
///               ▼
/// ┌──────────────────────────┐  hit   ┌───────────────────────┐ accepted ┌────────────────────────┐
/// │ 1. In-Memory LRU         ├───────▶│ Validate the Program  ├─────────▶│ `Arc<CompiledProgram>` │◀┐
/// └─────────────┬────────────┘        └───────────────────┬───┘          └────────────────────────┘ │
///               │ miss    ▲ producer failed         ▲     └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐              │
///               ▼         └───────────────┐         │ shared program                 ▼ rejected     │
/// ┌──────────────────────────┐ waiter ┌───┴─────────┴─────────┐          ┌────────────────────────┐ │
/// │ 2. In-Flight Compilation ├───────▶│ Wait for the Producer │          │ Validation Error       │ │
/// └─────────────┬────────────┘        └───────────────────────┘          └────────────────────────┘ │
///               │ producer                                                                          │
///               ▼                                                                                   │
/// ┌──────────────────────────┐  hit   ┌───────────────────────┐          ┌────────────────────────┐ │
/// │ 3. Disk Cache            ├─────┬─▶│ Publish the Program   ├─────────▶│ Insert into the LRU    ├─┘
/// └─────────────┬────────────┘     │  └─────┬───────────┬─────┘          └────────────────────────┘
///               │ miss or skipped  │        ┆           └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐
///               ▼                  │        ▼ best effort                            ▼ per policy
/// ┌──────────────────────────┐ hit │  ┌───────────────────────┐          ┌────────────────────────┐
/// │ 4. Artifact Exchange     ├─────┤  │ Store in Disk Cache   │          │ Publish to Exchange    │
/// └─────────────┬────────────┘     │  └───────────────────────┘          └────────────────────────┘
///               │ miss or skipped  │
///               ▼                  │
/// ┌──────────────────────────┐     │
/// │ 5. Backend Compilation   ├─────┘
/// └──────────────────────────┘
/// ```
///
/// Misses for the same key coordinate one backend producer, while requests for different keys remain concurrent.
/// The disk cache and the artifact exchange are skipped when they are not configured or when the domain supplies no
/// persistent key. Programs that they restore are validated like memory hits, and rejected ones count as misses.
/// Every successful source publishes to memory, while persistent writes are best effort and exchange publication is
/// controlled by the configured policy.
///
/// # Example
///
/// Backends own their context (e.g., one per XLA session), so users configure caching and artifact exchange when they
/// construct a backend handle. The following function configures both for any caching domain. It bounds the in-memory
/// tier, persists only compilations that take at least ten seconds in a disk cache capped at 10 GiB, and prefers
/// artifacts that the exchange leader publishes while still allowing followers to compile locally after a miss.
/// It also retains the most recent events for diagnostics.
///
/// ```rust
/// # use std::path::Path;
/// # use std::sync::Arc;
/// # use std::time::Duration;
/// # use ryft_core::{
/// #     CompilationArtifactExchange, CompilationArtifactExchangePolicy, CompilationCacheDomain, CompilationContext,
/// #     DiskCache,
/// # };
/// fn compilation_context<'o, C: CompilationCacheDomain>(
///     directory: &Path,
///     exchange: Arc<dyn 'o + CompilationArtifactExchange>,
/// ) -> std::io::Result<CompilationContext<'o, C>> {
///     let disk_cache = DiskCache::with_max_bytes(directory, 10 << 30)?
///         .with_write_thresholds(Duration::from_secs(10), 0);
///     let policy = CompilationArtifactExchangePolicy::PreferSharing {
///         timeout: Duration::from_secs(60),
///         fallback_to_local_compile: true,
///     };
///     Ok(CompilationContext::with_capacity(1024)
///         .with_disk_cache(disk_cache)
///         .with_artifact_exchange(exchange, policy)
///         .with_event_capacity(64))
/// }
/// ```
///
/// Contexts never read the environment on their own. Use [`CompilationContext::with_disk_cache_from_env`] to opt into
/// the disk cache that the [`DiskCache::ENV_VAR`] and [`DiskCache::MAX_BYTES_ENV_VAR`] environment variables configure.
pub struct CompilationContext<'o, C: CompilationCacheDomain> {
    /// In-memory LRU cache keyed by the structural [`CompilationCacheDomain::CacheKey`] of the domain.
    programs: Mutex<LruCache<C::CacheKey, Arc<C::CompiledProgram>>>,

    /// Producer coordination for each key. Entries exist only while a cache miss is being restored or compiled,
    /// so unrelated keys never wait for the backend work of one another.
    in_flight: Mutex<HashMap<C::CacheKey, Arc<InFlightCompilation<C::CompiledProgram>>>>,

    /// Optional disk-backed persistent tier.
    disk_cache: Option<DiskCache>,

    /// Optional exchange for sharing serialized programs across processes.
    artifact_exchange: Option<Arc<dyn 'o + CompilationArtifactExchange>>,

    /// Distributed exchange behavior, which is disabled by default to preserve local compilation semantics.
    artifact_exchange_policy: CompilationArtifactExchangePolicy,

    /// Lock-free counters used for verification and operational observability.
    statistics: CompilationCacheStatisticsAccumulator,

    /// Bounded history of structured events used for diagnostics without a logging dependency.
    recent_events: Mutex<VecDeque<CompilationEvent>>,

    /// Maximum number of entries retained in `recent_events`, where zero disables retention.
    event_capacity: usize,

    /// Optional reporting hook invoked after an event is retained.
    event_reporter: Option<Arc<dyn 'o + Fn(&CompilationEvent) + Send + Sync>>,
}

impl<'o, C: CompilationCacheDomain> CompilationContext<'o, C> {
    /// Creates a [`CompilationContext`] with the default cache capacity and no persistent cache or artifact exchange.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CACHE_CAPACITY)
    }

    /// Creates a [`CompilationContext`] whose in-memory tier retains at most `capacity` compiled programs.
    /// Note that a `capacity` of zero is clamped to one entry.
    #[inline]
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

    /// Attaches `disk_cache` as a persistent tier behind the in-memory tier of this context, with its size cap and
    /// write thresholds. On a memory miss, [`Self::compile`] consults the disk before compiling the program. Domains
    /// that explicitly support persistence can then serialize sufficiently expensive compilations so that future
    /// processes restart warm. Persistent read, write, and codec errors are recorded in [`Self::statistics`] and
    /// degrade to the in-memory path.
    #[inline]
    pub fn with_disk_cache(mut self, disk_cache: DiskCache) -> Self {
        self.disk_cache = Some(disk_cache);
        self
    }

    /// Attaches the [`DiskCache`] configured by the [`DiskCache::ENV_VAR`] environment variable, if it is set,
    /// and leaves this context unchanged otherwise.
    ///
    /// # Errors
    ///
    /// Returns invalid configuration and filesystem failures reported by [`DiskCache::from_env`].
    #[inline]
    pub fn with_disk_cache_from_env(mut self) -> std::io::Result<Self> {
        if let Some(disk_cache) = DiskCache::from_env()? {
            self.disk_cache = Some(disk_cache);
        }
        Ok(self)
    }

    /// Attaches a distributed exchange of compiled artifacts that is used according to `policy`. The exchange is used
    /// only when the domain supplies a key through [`CompilationCacheDomain::persistent_cache_key`] and can serialize
    /// and deserialize compiled programs. Participating processes must produce identical key bytes for the same
    /// compilation. Process zero compiles and publishes, while follower processes receive and restore. An exchange
    /// with a single process is ignored.
    #[inline]
    pub fn with_artifact_exchange(
        mut self,
        exchange: Arc<dyn 'o + CompilationArtifactExchange>,
        policy: CompilationArtifactExchangePolicy,
    ) -> Self {
        self.artifact_exchange = Some(exchange);
        self.artifact_exchange_policy = policy;
        self
    }

    /// Configures the number of structured events retained by [`Self::recent_events`]. A capacity of zero disables
    /// retention while leaving an installed reporter active. Events retained before this call are discarded.
    #[inline]
    pub fn with_event_capacity(mut self, capacity: usize) -> Self {
        self.event_capacity = capacity;
        self.recent_events = Mutex::new(VecDeque::with_capacity(capacity));
        self
    }

    /// Installs a reporting hook invoked for every structured compilation event. The hook runs without holding any
    /// internal cache or event buffer lock, and it should return promptly so that it does not add latency to
    /// compilations.
    #[inline]
    pub fn with_event_reporter(mut self, reporter: Arc<dyn 'o + Fn(&CompilationEvent) + Send + Sync>) -> Self {
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
    pub fn artifact_exchange(&self) -> Option<&(dyn 'o + CompilationArtifactExchange)> {
        self.artifact_exchange.as_deref()
    }

    /// Returns the configured distributed artifact exchange policy.
    #[inline]
    pub fn artifact_exchange_policy(&self) -> CompilationArtifactExchangePolicy {
        self.artifact_exchange_policy
    }

    /// Returns the number of compiled programs currently cached in the in-memory tier. This is mostly useful for
    /// telemetry and for tests that confirm that repeated compilations of one cache key reuse the cached program
    /// instead of recompiling it.
    #[inline]
    pub fn program_count(&self) -> usize {
        self.programs.lock().expect("compilation cache mutex should not be poisoned").len()
    }

    /// Returns a lock-free snapshot of the activity counters of this context.
    #[inline]
    pub fn statistics(&self) -> CompilationCacheStatistics {
        self.statistics.snapshot()
    }

    /// Returns the retained structured events in emission order.
    #[inline]
    pub fn recent_events(&self) -> Vec<CompilationEvent> {
        self.recent_events
            .lock()
            .expect("compilation event mutex should not be poisoned")
            .iter()
            .cloned()
            .collect()
    }

    /// Removes every entry from the in-memory tier. Persistent entries, in-flight compilations, statistics,
    /// and retained events are unaffected.
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

    /// Compiles `program` through the shared cache and returns the shared compiled program. This function derives the
    /// exact structural key of `program` through [`CompilationCacheDomain::compilation_key`] and resolves it through
    /// the in-memory tier, the persistent tier, and the artifact exchange, in that order, before it falls back to
    /// [`CompilationCacheDomain::compile_uncached`]. Every program that it does not compile for this request must pass
    /// [`CompilationCacheDomain::validate_cached_program`] first.
    ///
    /// On a hit, the entry is moved to the most recently used position. On a miss, the new entry is inserted, and the
    /// least recently used entry is evicted if the cache is at capacity.
    ///
    /// Misses for the same key are deduplicated. The first caller to miss becomes the producer and restores or compiles
    /// the program, while the rest wait for its result, and different keys may compile concurrently. When a producer
    /// fails, its error is returned only to that caller, and waiters retry, so the first of them to miss again becomes
    /// the new producer.
    ///
    /// Persistent caching is consulted only when both a [`DiskCache`] is attached and
    /// [`CompilationCacheDomain::persistent_cache_key`] returns key bytes for the requesting compilation. Persistent
    /// I/O and codec failures, as well as persistent artifacts that the domain rejects, are counted and degraded to
    /// misses.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`CompilationCacheDomain::compilation_key`] and of
    /// [`CompilationCacheDomain::compile_uncached`] when this caller compiles the program. It also returns the error of
    /// [`CompilationCacheDomain::validate_cached_program`] when the domain rejects a program from the in-memory tier or
    /// from another producer. When the artifact exchange policy requires sharing, it also returns the exchange failures
    /// that prevent this process from obtaining or publishing an artifact, including deserialization errors of
    /// received artifacts. Persistent tier failures never fail this function.
    pub fn compile(&self, domain: &C, program: &C::LoweredProgram) -> Result<Arc<C::CompiledProgram>, C::Error> {
        // Derive the exact key once so every retry after a failed producer refers to the same request.
        let cache_key = domain.compilation_key(program)?;
        loop {
            let lookup_start = Instant::now();

            // Lookup refreshes the LRU position. Clone the handle and release the cache lock before validation and
            // event callbacks, which may access this context again.
            let cached = self
                .programs
                .lock()
                .expect("compilation cache mutex should not be poisoned")
                .get(&cache_key)
                .map(Arc::clone);
            if let Some(cached) = cached {
                self.statistics.memory_hits.fetch_add(1, Ordering::Relaxed);
                return self.accept_memory_hit(domain, program, cached, lookup_start);
            }

            // Elect one producer per key while holding only the in-flight map lock. Waiting, backend calls, and event
            // reporters run after releasing it, so unrelated keys can progress and callbacks cannot retain this lock.
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
                let Some(shared) = in_flight.wait() else {
                    // A failed or panicking producer clears its marker and wakes waiters. Retry the lookup and
                    // election instead of sharing that caller's failure, because another attempt may succeed.
                    accumulate_duration(wait_start.elapsed(), &self.statistics.memory_lookup_nanoseconds);
                    continue;
                };

                // The producer compiled for its own request. Equal keys must also serve this caller's request, and
                // rejecting an in-memory result is an error rather than permission to compile a conflicting entry.
                let validation = domain.validate_cached_program(program, &shared);
                let outcome = match &validation {
                    Ok(()) => CompilationLookupOutcome::Hit,
                    Err(error) => CompilationLookupOutcome::Failed(CompilationStepError::Incompatible {
                        message: error.to_string(),
                    }),
                };

                self.record_duration(
                    CompilationStep::MemoryWait { outcome },
                    wait_start.elapsed(),
                    &self.statistics.memory_lookup_nanoseconds,
                );

                return validation.map(|()| shared);
            }

            // A prior producer can insert the program after the first lookup above and remove its in-flight marker
            // before this caller registers as the new producer. Checking again after registering closes that narrow
            // handoff window, which would otherwise trigger a duplicate compilation. The waiters of this producer
            // validate the program against their own requests. Release the cache lock before waking them or reporting.
            let cached = self
                .programs
                .lock()
                .expect("compilation cache mutex should not be poisoned")
                .get(&cache_key)
                .map(Arc::clone);
            if let Some(cached) = cached {
                self.in_flight.lock().expect("in-flight cache mutex should not be poisoned").remove(&cache_key);
                in_flight.finish(InFlightState::Ready(Arc::clone(&cached)));
                self.statistics.memory_hits.fetch_add(1, Ordering::Relaxed);
                return self.accept_memory_hit(domain, program, cached, lookup_start);
            }

            // Establish cleanup before reporting the miss, since reporters may panic. Until finish publishes a
            // program, dropping this guard removes the marker and wakes waiters on every error or unwind path.
            let producer = InFlightProducer::new(self, cache_key, in_flight);
            self.statistics.misses.fetch_add(1, Ordering::Relaxed);
            self.record_duration(
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                lookup_start.elapsed(),
                &self.statistics.memory_lookup_nanoseconds,
            );

            // A disabled policy has no deadline and a single process has no peers, so neither activates the exchange.
            // Derive the persistent key only when the disk tier or exchange can use it.
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
                .map(|(disk_cache, key)| (disk_cache, DiskCacheDigest::from_key(key)));

            // The persistent key addresses artifacts across processes, so the exchange cannot participate without it.
            // Validate peer compatibility before disk restoration, including when this process has a disk hit.
            let exchange = match (configured_exchange, persistent_key.as_deref()) {
                (None, _) => None,
                (Some(_), None) => {
                    self.record_event(CompilationEvent {
                        step: CompilationStep::ExchangeCompatibilityValidation {
                            outcome: CompilationStepOutcome::Skipped(CompilationSkipReason::Unsupported),
                        },
                        duration: Duration::ZERO,
                    });
                    self.require_exchange_or_record_fallback("domain does not provide a persistent compilation key")?;
                    None
                }
                (Some((exchange, timeout)), Some(key)) => {
                    let exchange = ActiveExchange { exchange, key, timeout };
                    self.validate_exchange_compatibility(exchange)?.then_some(exchange)
                }
            };

            // A leader that restores a program from disk still publishes it, because its followers wait for that
            // artifact. Restore validates it against the request and treats rejected artifacts as misses.
            if let Some((disk_cache, digest)) = &persistent
                && let Some((restored, bytes)) = self.restore_from_disk(domain, program, disk_cache, digest)
            {
                if let Some(exchange) = exchange
                    && exchange.is_leader()
                {
                    self.publish_to_exchange(exchange, bytes.as_slice())?;
                }
                return Ok(producer.finish(restored));
            }

            // Followers try the leader after their disk tier. The receive helper enforces whether a miss may fall
            // back to local compilation. Leaders proceed directly to compilation when disk restoration misses.
            if let Some(exchange) = exchange
                && !exchange.is_leader()
                && let Some(received) = self.receive_from_exchange(domain, program, exchange)?
            {
                return Ok(producer.finish(received));
            }

            return self.compile_and_publish(domain, program, producer, persistent, exchange);
        }
    }

    /// Validates the `cached` in-memory program against the requesting `program` and records the lookup that started
    /// at `lookup_start`. A rejection fails the compilation, because equal exact keys must mean interchangeable
    /// programs.
    fn accept_memory_hit(
        &self,
        domain: &C,
        program: &C::LoweredProgram,
        cached: Arc<C::CompiledProgram>,
        lookup_start: Instant,
    ) -> Result<Arc<C::CompiledProgram>, C::Error> {
        let validation = domain.validate_cached_program(program, &cached);
        let outcome = match &validation {
            Ok(()) => CompilationLookupOutcome::Hit,
            Err(error) => {
                CompilationLookupOutcome::Failed(CompilationStepError::Incompatible { message: error.to_string() })
            }
        };
        self.record_duration(
            CompilationStep::CacheLookup { tier: CompilationCacheTier::Memory, outcome },
            lookup_start.elapsed(),
            &self.statistics.memory_lookup_nanoseconds,
        );
        validation.map(|()| cached)
    }

    /// Validates compatibility with `exchange` and returns whether it participates in this compilation. A validation
    /// failure is fatal when the policy requires sharing. Otherwise, it is recorded and the exchange is deactivated for
    /// this compilation.
    fn validate_exchange_compatibility(&self, exchange: ActiveExchange<'_>) -> Result<bool, C::Error> {
        let compatibility_start = Instant::now();
        let Err(error) = exchange.exchange.validate_compatibility(exchange.key, exchange.timeout) else {
            self.record_duration(
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                compatibility_start.elapsed(),
                &self.statistics.exchange_nanoseconds,
            );
            return Ok(true);
        };
        let message = error.to_string();
        let (counter, reason) = match &error {
            CompilationExchangeError::TimedOut => (&self.statistics.exchange_timeouts, CompilationStepError::TimedOut),
            CompilationExchangeError::Incompatible { .. } => {
                (&self.statistics.exchange_errors, CompilationStepError::Incompatible { message: message.clone() })
            }
            CompilationExchangeError::Failed { .. } => {
                (&self.statistics.exchange_errors, CompilationStepError::ExchangeFailed { message: message.clone() })
            }
        };
        self.record_exchange_fallback(
            counter,
            CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Failed(reason) },
            compatibility_start,
            message.as_str(),
        )?;
        Ok(false)
    }

    /// Restores a compiled program from the persistent tier, returning its serialized payload alongside it so that
    /// an exchange leader can publish it without serializing it again. Every outcome other than a hit, including a
    /// restored program that the domain rejects for the requesting `program`, is recorded and degraded to [`None`].
    fn restore_from_disk(
        &self,
        domain: &C,
        program: &C::LoweredProgram,
        disk_cache: &DiskCache,
        digest: &DiskCacheDigest,
    ) -> Option<(C::CompiledProgram, Vec<u8>)> {
        let lookup_start = Instant::now();
        let (outcome, restored) = match disk_cache.get(digest) {
            Ok(Some(bytes)) => match domain.deserialize_program(bytes.as_slice()) {
                Ok(Some(restored)) if domain.validate_cached_program(program, &restored).is_err() => {
                    (CompilationLookupOutcome::Miss(CompilationMissReason::Incompatible), None)
                }
                Ok(Some(restored)) => (CompilationLookupOutcome::Hit, Some((restored, bytes))),
                Ok(None) => (CompilationLookupOutcome::Miss(CompilationMissReason::Incompatible), None),
                Err(error) => (
                    CompilationLookupOutcome::Failed(CompilationStepError::DeserializationFailed {
                        message: error.to_string(),
                    }),
                    None,
                ),
            },
            Ok(None) => (CompilationLookupOutcome::Miss(CompilationMissReason::NotFound), None),
            Err(error) => (
                CompilationLookupOutcome::Failed(CompilationStepError::ReadFailed { message: error.to_string() }),
                None,
            ),
        };
        match outcome {
            CompilationLookupOutcome::Hit => {
                self.statistics.persistent_hits.fetch_add(1, Ordering::Relaxed);
            }
            CompilationLookupOutcome::Failed(_) => {
                self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
            }
            CompilationLookupOutcome::Miss(_) => {}
        }
        self.record_duration(
            CompilationStep::CacheLookup { tier: CompilationCacheTier::Persistent, outcome },
            lookup_start.elapsed(),
            &self.statistics.persistent_lookup_nanoseconds,
        );
        restored
    }

    /// Waits on a follower process for the artifact that the exchange leader publishes. Returns the restored program
    /// on success, and [`None`] when the policy permits falling back to local compilation after a miss, timeout, or
    /// failure. A received program that the domain rejects for the requesting `program` is an incompatible miss.
    fn receive_from_exchange(
        &self,
        domain: &C,
        program: &C::LoweredProgram,
        exchange: ActiveExchange<'_>,
    ) -> Result<Option<C::CompiledProgram>, C::Error> {
        self.statistics.exchange_waits.fetch_add(1, Ordering::Relaxed);
        let exchange_start = Instant::now();
        let bytes = match exchange.exchange.receive(exchange.key, exchange.timeout) {
            Ok(Some(bytes)) => bytes,
            Ok(None) | Err(CompilationExchangeError::TimedOut) => {
                self.record_exchange_fallback(
                    &self.statistics.exchange_timeouts,
                    CompilationStep::ExchangeReceive {
                        outcome: CompilationLookupOutcome::Miss(CompilationMissReason::TimedOut),
                    },
                    exchange_start,
                    "timed out waiting for a compilation artifact",
                )?;
                return Ok(None);
            }
            Err(ref error @ CompilationExchangeError::Incompatible { ref message }) => {
                self.record_exchange_fallback(
                    &self.statistics.exchange_errors,
                    CompilationStep::ExchangeReceive {
                        outcome: CompilationLookupOutcome::Failed(CompilationStepError::Incompatible {
                            message: error.to_string(),
                        }),
                    },
                    exchange_start,
                    message.as_str(),
                )?;
                return Ok(None);
            }
            Err(ref error @ CompilationExchangeError::Failed { ref message }) => {
                self.record_exchange_fallback(
                    &self.statistics.exchange_errors,
                    CompilationStep::ExchangeReceive {
                        outcome: CompilationLookupOutcome::Failed(CompilationStepError::ReadFailed {
                            message: error.to_string(),
                        }),
                    },
                    exchange_start,
                    message.as_str(),
                )?;
                return Ok(None);
            }
        };
        match domain.deserialize_program(bytes.as_slice()) {
            Ok(Some(received)) => {
                if let Err(error) = domain.validate_cached_program(program, &received) {
                    self.record_duration(
                        CompilationStep::ExchangeReceive {
                            outcome: CompilationLookupOutcome::Miss(CompilationMissReason::Incompatible),
                        },
                        exchange_start.elapsed(),
                        &self.statistics.exchange_nanoseconds,
                    );
                    self.require_exchange_or_record_fallback(
                        format!("received compilation artifact is incompatible: {error}").as_str(),
                    )?;
                    return Ok(None);
                }
                self.statistics.exchange_hits.fetch_add(1, Ordering::Relaxed);
                self.record_duration(
                    CompilationStep::ExchangeReceive { outcome: CompilationLookupOutcome::Hit },
                    exchange_start.elapsed(),
                    &self.statistics.exchange_nanoseconds,
                );
                Ok(Some(received))
            }
            Ok(None) => {
                self.record_duration(
                    CompilationStep::ExchangeReceive {
                        outcome: CompilationLookupOutcome::Miss(CompilationMissReason::Incompatible),
                    },
                    exchange_start.elapsed(),
                    &self.statistics.exchange_nanoseconds,
                );
                self.require_exchange_or_record_fallback("received compilation artifact is incompatible")?;
                Ok(None)
            }
            Err(error) => {
                // The domain error is more informative than a generic exchange requirement error, so it is returned
                // directly when the policy does not permit a fallback.
                self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                self.record_duration(
                    CompilationStep::ExchangeReceive {
                        outcome: CompilationLookupOutcome::Failed(CompilationStepError::DeserializationFailed {
                            message: error.to_string(),
                        }),
                    },
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

    /// Publishes the serialized `artifact` through the exchange leader. Publication failures are fatal only when the
    /// policy requires sharing. Publication durations deliberately stay out of the exchange duration counter, which
    /// tracks time spent obtaining artifacts.
    fn publish_to_exchange(&self, exchange: ActiveExchange<'_>, artifact: &[u8]) -> Result<(), C::Error> {
        let publish_start = Instant::now();
        match exchange.exchange.publish(exchange.key, artifact) {
            Ok(()) => {
                self.statistics.exchange_publishes.fetch_add(1, Ordering::Relaxed);
                self.record_event(CompilationEvent {
                    step: CompilationStep::ExchangePublish { outcome: CompilationStepOutcome::Succeeded },
                    duration: publish_start.elapsed(),
                });
                Ok(())
            }
            Err(error) => {
                self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                self.record_event(CompilationEvent {
                    step: CompilationStep::ExchangePublish {
                        outcome: CompilationStepOutcome::Failed(CompilationStepError::WriteFailed {
                            message: error.to_string(),
                        }),
                    },
                    duration: publish_start.elapsed(),
                });
                self.require_exchange_or_record_fallback(error.to_string().as_str())
            }
        }
    }

    /// Publishes a terminal failure through the exchange leader when it cannot publish an artifact, so that followers
    /// stop waiting for it instead of running into their deadline. `cause` classifies the failure in the recorded
    /// event, and `message` describes it to the followers. Failures to publish the failure itself are recorded but
    /// never replace the original failure.
    fn publish_failure_to_exchange(&self, exchange: ActiveExchange<'_>, cause: CompilationStepError, message: &str) {
        let publish_start = Instant::now();
        let outcome = match exchange.exchange.publish_failure(exchange.key, message) {
            Ok(()) => CompilationStepOutcome::Succeeded,
            Err(error) => {
                self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                CompilationStepOutcome::Failed(CompilationStepError::WriteFailed { message: error.to_string() })
            }
        };
        self.record_event(CompilationEvent {
            step: CompilationStep::ExchangeFailureNotification { cause, outcome },
            duration: publish_start.elapsed(),
        });
    }

    /// Compiles `program` through `domain` and publishes it to configured caches and the artifact exchange. The
    /// persistent tier is written only by the exchange leader (or by every process when no exchange is active), and
    /// only when the compilation meets the write thresholds of the [`DiskCache`]. The exchange leader publishes the
    /// serialized program, or a terminal failure whenever it cannot, so that followers stop waiting.
    fn compile_and_publish(
        &self,
        domain: &C,
        program: &C::LoweredProgram,
        producer: InFlightProducer<'_, C>,
        persistent: Option<(&DiskCache, DiskCacheDigest)>,
        exchange: Option<ActiveExchange<'_>>,
    ) -> Result<Arc<C::CompiledProgram>, C::Error> {
        let leader_exchange = exchange.filter(|exchange| exchange.is_leader());
        self.statistics.compilations.fetch_add(1, Ordering::Relaxed);
        let compilation_start = Instant::now();
        let program = match domain.compile_uncached(program) {
            Ok(program) => program,
            Err(error) => {
                let message = error.to_string();
                let cause = CompilationStepError::CompilationFailed { message: message.clone() };
                self.record_duration(
                    CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Failed(cause.clone()) },
                    compilation_start.elapsed(),
                    &self.statistics.compilation_nanoseconds,
                );
                if let Some(exchange) = leader_exchange {
                    self.publish_failure_to_exchange(exchange, cause, message.as_str());
                }
                return Err(error);
            }
        };
        let compilation_duration = compilation_start.elapsed();
        self.record_duration(
            CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
            compilation_duration,
            &self.statistics.compilation_nanoseconds,
        );

        // Serialization is skipped entirely when neither the persistent tier nor the exchange would consume its bytes.
        let persistent = match persistent.filter(|_| exchange.is_none_or(ActiveExchange::is_leader)) {
            Some((disk_cache, _)) if !disk_cache.should_serialize(compilation_duration) => {
                self.record_event(CompilationEvent {
                    step: CompilationStep::PersistentStore {
                        outcome: CompilationStepOutcome::Skipped(CompilationSkipReason::BelowWriteThreshold),
                    },
                    duration: Duration::ZERO,
                });
                None
            }
            persistent => persistent,
        };
        if persistent.is_none() && leader_exchange.is_none() {
            return Ok(producer.finish(program));
        }

        let serialization_start = Instant::now();
        match domain.serialize_program(&program) {
            Ok(Some(bytes)) => {
                if let Some((disk_cache, digest)) = &persistent {
                    let outcome = if !disk_cache.should_persist(compilation_duration, bytes.len()) {
                        CompilationStepOutcome::Skipped(CompilationSkipReason::BelowWriteThreshold)
                    } else {
                        match disk_cache.put(digest, bytes.as_slice()) {
                            Ok(()) => CompilationStepOutcome::Succeeded,
                            Err(error) => {
                                self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
                                CompilationStepOutcome::Failed(CompilationStepError::WriteFailed {
                                    message: error.to_string(),
                                })
                            }
                        }
                    };
                    self.record_event(CompilationEvent {
                        step: CompilationStep::PersistentStore { outcome },
                        duration: serialization_start.elapsed(),
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
                        step: CompilationStep::PersistentStore {
                            outcome: CompilationStepOutcome::Skipped(CompilationSkipReason::Unsupported),
                        },
                        duration,
                    });
                }
                if let Some(exchange) = leader_exchange {
                    let message = "domain does not support compiled program serialization";
                    self.record_event(CompilationEvent {
                        step: CompilationStep::ExchangePublish {
                            outcome: CompilationStepOutcome::Skipped(CompilationSkipReason::Unsupported),
                        },
                        duration,
                    });
                    self.publish_failure_to_exchange(exchange, CompilationStepError::Unsupported, message);
                    self.require_exchange_or_record_fallback(message)?;
                }
            }
            Err(error) => {
                let duration = serialization_start.elapsed();
                let message = error.to_string();
                let cause = CompilationStepError::SerializationFailed { message: message.clone() };
                if persistent.is_some() {
                    self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
                    self.record_event(CompilationEvent {
                        step: CompilationStep::PersistentStore {
                            outcome: CompilationStepOutcome::Failed(cause.clone()),
                        },
                        duration,
                    });
                }
                if let Some(exchange) = leader_exchange {
                    self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                    self.record_event(CompilationEvent {
                        step: CompilationStep::ExchangePublish {
                            outcome: CompilationStepOutcome::Failed(cause.clone()),
                        },
                        duration,
                    });
                    self.publish_failure_to_exchange(exchange, cause, message.as_str());
                    if !self.artifact_exchange_policy.permits_local_fallback() {
                        return Err(error);
                    }
                    self.statistics.exchange_fallbacks.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        Ok(producer.finish(program))
    }

    /// Records an exchange step that could not supply an artifact and then applies the fallback policy.
    ///
    /// # Parameters
    ///
    ///   - `counter`: Statistics counter that classifies the miss (e.g., timeouts or errors).
    ///   - `step`: Exchange step and its lookup or completion result.
    ///   - `start`: Instant at which the exchange step started.
    ///   - `message`: Description of the miss used in the error returned when the policy requires sharing.
    fn record_exchange_fallback(
        &self,
        counter: &AtomicU64,
        step: CompilationStep,
        start: Instant,
        message: &str,
    ) -> Result<(), C::Error> {
        counter.fetch_add(1, Ordering::Relaxed);
        self.record_duration(step, start.elapsed(), &self.statistics.exchange_nanoseconds);
        self.require_exchange_or_record_fallback(message)
    }

    /// Returns an error describing `message` when the exchange policy requires sharing,
    /// and otherwise counts a fallback to local compilation.
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

    /// Accumulates `duration` into `duration_counter` and records one structured event for the step.
    fn record_duration(&self, step: CompilationStep, duration: Duration, duration_counter: &AtomicU64) {
        accumulate_duration(duration, duration_counter);
        self.record_event(CompilationEvent { step, duration });
    }

    /// Retains `event` when event retention is enabled and then reports it to the installed reporter, if any.
    /// The reporter runs after the event buffer lock is released.
    fn record_event(&self, event: CompilationEvent) {
        if self.event_capacity > 0 {
            let mut events = self.recent_events.lock().expect("compilation event mutex should not be poisoned");
            if events.len() == self.event_capacity {
                events.pop_front();
            }
            events.push_back(event.clone());
        }
        if let Some(reporter) = self.event_reporter.as_ref() {
            reporter(&event);
        }
    }
}

impl<C: CompilationCacheDomain> Default for CompilationContext<'_, C> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Magic bytes that open every uncompressed entry envelope, which identify files that this cache wrote apart from
/// other gzip compressed files in its directory. The envelope header has a fixed layout, so the magic value has a fixed
/// width, and eight bytes is the conventional width of a file signature (e.g., that of PNG files) that makes accidental
/// matches negligible. `RYFTCACHE` is therefore truncated to `RYFTCACH`. Changes to the envelope layout increment
/// [`ENTRY_FORMAT_VERSION`] instead of changing this value.
const ENTRY_MAGIC: &[u8; 8] = b"RYFTCACH";

/// Version of the entry envelope format.
const ENTRY_FORMAT_VERSION: u32 = 1;

/// Length of a SHA-256 digest, in bytes.
const SHA256_LENGTH: usize = 32;

/// Length of the entry envelope header (i.e., magic, version, key digest, payload length, and payload checksum).
const ENTRY_HEADER_LENGTH: usize = ENTRY_MAGIC.len() + 4 + SHA256_LENGTH + 8 + SHA256_LENGTH;

/// Maximum size of an auxiliary payload, in bytes.
const MAXIMUM_AUXILIARY_ENTRY_SIZE: usize = 64 * 1024 * 1024;

/// Filename extension of serialized executable entries.
const EXECUTABLE_EXTENSION: &str = "executable";

/// Filename extension of auxiliary entries.
const AUXILIARY_EXTENSION: &str = "metadata";

/// Maximum number of temporary filenames that a single write tries before giving up.
const MAXIMUM_TEMPORARY_FILE_ATTEMPTS: usize = 256;

/// Default minimum compilation duration required before a newly compiled program is persisted.
const DEFAULT_MINIMUM_COMPILATION_DURATION: Duration = Duration::from_secs(1);

/// Default minimum serialized payload size required before a newly compiled program is persisted.
const DEFAULT_MINIMUM_ENTRY_SIZE: usize = 0;

/// Directory-backed persistent cache of serialized compiled programs. A [`DiskCache`] persists serialized compiled
/// programs across processes, similar to JAX's
/// [persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html). When attached
/// to a [`CompilationContext`], it serves as a second tier below the in-memory LRU cache. That context manages
/// executable persistence through the serialization hooks of [`CompilationCacheDomain`], restoring programs on memory
/// misses and persisting new compilations that meet the write thresholds configured via
/// [`Self::with_write_thresholds`].
///
/// Entries are stored gzip compressed as `<directory>/<sha256>.executable`. A backend opts into this tier by providing
/// canonical bytes for the complete persistent cache key through [`CompilationCacheDomain::persistent_cache_key`].
/// For the same compilation, processes sharing the cache must produce identical key bytes, and the filename is their
/// SHA-256 digest. The compressed entry is a versioned envelope that repeats the key digest and records the payload
/// length and SHA-256 checksum, all of which are validated before backend deserialization. Backends use the public
/// [`Self::get_auxiliary`] and [`Self::put_auxiliary`] functions to persist data outside the executable lifecycle
/// (e.g., tuning results), stored as `<directory>/<sha256>.metadata` entries with the same envelope.
///
/// Writes are atomic because they use the standard pattern of writing a temporary file and renaming it. Filesystem and
/// validation failures are returned to the [`CompilationContext`], which records each failure and degrades it to a
/// cache miss.
///
/// When constructed via [`DiskCache::with_max_bytes`] or with the [`DiskCache::MAX_BYTES_ENV_VAR`] environment variable
/// set, the cache evicts the least recently written entries after each write to keep its footprint on disk under the
/// configured cap. Caches opened via [`DiskCache::open`] have no cap.
///
/// Wire one into a [`CompilationContext`] via [`CompilationContext::with_disk_cache`] or
/// [`CompilationContext::with_disk_cache_from_env`].
pub struct DiskCache {
    /// Filesystem directory holding the cached entries.
    directory: PathBuf,

    /// Counter that produces candidate suffixes for temporary filenames. Exclusive file creation resolves collisions
    /// with independently opened caches and with stale files left by an earlier process that had the same process
    /// identifier.
    write_counter: AtomicU64,

    /// Optional cap on the size of the cached entries on disk, in bytes. [`None`] disables eviction entirely. When set,
    /// the cache evicts the least recently written entries after each successful write to keep the total under the
    /// cap, always preserving the entry that was just written.
    max_bytes: Option<u64>,

    /// Minimum compilation duration required before a newly compiled program is persisted.
    minimum_compilation_duration: Duration,

    /// Minimum serialized backend payload size required before it is persisted.
    minimum_entry_size: usize,
}

impl DiskCache {
    /// Environment variable that [`Self::from_env`] reads for the cache directory.
    pub const ENV_VAR: &str = "RYFT_COMPILATION_CACHE_DIR";

    /// Environment variable that [`Self::from_env`] reads for the optional cap on the size of the cache on disk,
    /// in bytes. It must hold an unsigned 64-bit integer when set, and eviction is disabled when it is absent.
    pub const MAX_BYTES_ENV_VAR: &str = "RYFT_COMPILATION_CACHE_MAX_BYTES";

    /// Opens a [`DiskCache`] rooted at `directory`, creating the directory and any missing parents if necessary.
    /// The resulting cache has no size cap. Use [`Self::with_max_bytes`] to bound its footprint on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if `directory` already exists as a non-directory or cannot be created.
    #[inline]
    pub fn open(directory: impl Into<PathBuf>) -> std::io::Result<Self> {
        Self::open_with_max_bytes(directory, None)
    }

    /// Opens a [`DiskCache`] rooted at `directory` whose entries are evicted, least recently written first, after each
    /// write that brings the size of the cache on disk above `max_bytes`.
    ///
    /// The most recently written entry is always preserved, even if it alone exceeds `max_bytes`, so that callers
    /// always observe their own writes on the immediately following read.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::open`].
    #[inline]
    pub fn with_max_bytes<D: Into<PathBuf>>(directory: D, max_bytes: u64) -> std::io::Result<Self> {
        Self::open_with_max_bytes(directory, Some(max_bytes))
    }

    /// Opens the [`DiskCache`] configured by the [`RYFT_COMPILATION_CACHE_DIR`](Self::ENV_VAR) environment
    /// variable. Returns `Ok(Some(cache))` when the variable is set and `Ok(None)` when it is unset. When
    /// [`RYFT_COMPILATION_CACHE_MAX_BYTES`](Self::MAX_BYTES_ENV_VAR) is also set, the resulting cache is
    /// capped as if it were opened via [`Self::with_max_bytes`].
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::ErrorKind::InvalidInput`] error when the size cap variable does not hold an unsigned
    /// 64-bit integer, and the errors of [`Self::open`] when the directory cannot be opened.
    pub fn from_env() -> std::io::Result<Option<Self>> {
        let Some(directory) = std::env::var_os(Self::ENV_VAR) else {
            return Ok(None);
        };
        let max_bytes = match std::env::var(Self::MAX_BYTES_ENV_VAR) {
            Ok(value) => Some(value.parse::<u64>().map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid `{}` value `{}`: {}", Self::MAX_BYTES_ENV_VAR, value, error),
                )
            })?),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid `{}` value: {}", Self::MAX_BYTES_ENV_VAR, error),
                ));
            }
        };
        Self::open_with_max_bytes(directory, max_bytes).map(Some)
    }

    /// Configures the minimum compilation duration and serialized payload size required before a newly compiled
    /// program is written. Reads are unaffected, so an existing validated entry is always eligible for reuse.
    #[inline]
    pub fn with_write_thresholds(mut self, minimum_compilation_duration: Duration, minimum_entry_size: usize) -> Self {
        self.minimum_compilation_duration = minimum_compilation_duration;
        self.minimum_entry_size = minimum_entry_size;
        self
    }

    /// Returns the directory backing this cache.
    #[inline]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Returns the configured size cap, if any.
    #[inline]
    pub fn max_bytes(&self) -> Option<u64> {
        self.max_bytes
    }

    /// Returns the total size, in bytes, of every executable and auxiliary entry in the cache directory, which is
    /// zero for a missing or empty directory.
    ///
    /// # Errors
    ///
    /// Returns the errors of reading the directory and the metadata of its entries, other than a missing directory.
    pub fn cache_size_bytes(&self) -> std::io::Result<u64> {
        let entries = match fs::read_dir(&self.directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error),
        };
        let mut total_bytes = 0u64;
        for entry in entries {
            let entry = entry?;
            if Self::is_entry_path(&entry.path()) {
                total_bytes = total_bytes.saturating_add(entry.metadata()?.len());
            }
        }
        Ok(total_bytes)
    }

    /// Reads a checksummed auxiliary payload that a backend associated with `key` in `namespace`. Auxiliary entries
    /// share the directory of this cache but use a different filename extension than executables. Reads are bounded to
    /// 64 MiB of uncompressed payload so that corrupted compressed inputs cannot grow memory without limit. Missing
    /// entries return `Ok(None)`, while validation and I/O failures are returned explicitly.
    pub fn get_auxiliary(&self, namespace: &str, key: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
        let digest = Self::auxiliary_digest(namespace, key);
        Self::read_entry(&self.auxiliary_entry_path(&digest), &digest, Some(MAXIMUM_AUXILIARY_ENTRY_SIZE))
    }

    /// Atomically writes a checksummed auxiliary payload that a backend associates with `key` in `namespace`. Payloads
    /// larger than 64 MiB are rejected. Readers observe either the prior complete entry or the new complete entry
    /// across crashes and concurrent processes.
    pub fn put_auxiliary(&self, namespace: &str, key: &[u8], data: &[u8]) -> std::io::Result<()> {
        if data.len() > MAXIMUM_AUXILIARY_ENTRY_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "persistent auxiliary payload exceeds the 64 MiB limit",
            ));
        }
        let digest = Self::auxiliary_digest(namespace, key);
        self.write_entry(&self.auxiliary_entry_path(&digest), &digest, AUXILIARY_EXTENSION, data)
    }

    /// Opens a [`DiskCache`] rooted at `directory` with an optional size cap and the default write thresholds.
    fn open_with_max_bytes<D: Into<PathBuf>>(directory: D, max_bytes: Option<u64>) -> std::io::Result<Self> {
        let directory = directory.into();
        fs::create_dir_all(&directory)?;
        Ok(Self {
            directory,
            write_counter: AtomicU64::new(0),
            max_bytes,
            minimum_compilation_duration: DEFAULT_MINIMUM_COMPILATION_DURATION,
            minimum_entry_size: DEFAULT_MINIMUM_ENTRY_SIZE,
        })
    }

    /// Reads and validates the cached serialized payload for `digest`. A missing entry is a cache miss, while
    /// filesystem, decompression, envelope, and checksum failures are returned explicitly.
    fn get(&self, digest: &DiskCacheDigest) -> std::io::Result<Option<Vec<u8>>> {
        Self::read_entry(&self.entry_path(digest), digest, None)
    }

    /// Atomically writes `data` to the cache under `digest`, so that a partial write or process crash can never leave
    /// a half-written entry visible to other readers. After a successful write, evicts the least recently written
    /// entries when a size cap is configured. Returns any write, synchronization, rename, cleanup, or eviction failure
    /// explicitly. The compilation context records such failures before degrading to an in-memory result.
    fn put(&self, digest: &DiskCacheDigest, data: &[u8]) -> std::io::Result<()> {
        self.write_entry(&self.entry_path(digest), digest, EXECUTABLE_EXTENSION, data)
    }

    /// Returns whether a compilation that took `compilation_duration` may be persisted, which is checked before backend
    /// serialization so that cheap compilations are never serialized.
    #[inline]
    fn should_serialize(&self, compilation_duration: Duration) -> bool {
        compilation_duration >= self.minimum_compilation_duration
    }

    /// Returns whether a newly compiled program that took `compilation_duration` and serialized to `entry_size` bytes
    /// meets the write policy of this cache.
    #[inline]
    fn should_persist(&self, compilation_duration: Duration, entry_size: usize) -> bool {
        self.should_serialize(compilation_duration) && entry_size >= self.minimum_entry_size
    }

    /// Returns the filesystem path of the executable entry for `digest`.
    fn entry_path(&self, digest: &DiskCacheDigest) -> PathBuf {
        self.directory.join(format!("{}.{}", digest.to_hex(), EXECUTABLE_EXTENSION))
    }

    /// Returns the filesystem path of the auxiliary entry for `digest`.
    fn auxiliary_entry_path(&self, digest: &DiskCacheDigest) -> PathBuf {
        self.directory.join(format!("{}.{}", digest.to_hex(), AUXILIARY_EXTENSION))
    }

    /// Returns whether `path` names an executable or auxiliary entry, as opposed to a temporary or unrelated file.
    fn is_entry_path(path: &Path) -> bool {
        path.extension()
            .is_some_and(|extension| extension == EXECUTABLE_EXTENSION || extension == AUXILIARY_EXTENSION)
    }

    /// Returns the digest that addresses the auxiliary entry for `key` in `namespace`. The digest is domain separated
    /// from executable digests and hashes both components independently, so that no pair of namespaces and keys
    /// can collide by concatenation.
    fn auxiliary_digest(namespace: &str, key: &[u8]) -> DiskCacheDigest {
        let mut hasher = Sha256::new();
        hasher.update(b"RYFT-AUXILIARY\0");
        hasher.update(Sha256::digest(namespace.as_bytes()));
        hasher.update(Sha256::digest(key));
        DiskCacheDigest { bytes: hasher.finalize().into() }
    }

    /// Reads and validates the entry at `path` for `digest`, returning `Ok(None)` when it does not exist.
    ///
    /// # Parameters
    ///
    ///   - `path`: Filesystem path of the entry.
    ///   - `digest`: Digest that the entry envelope must repeat.
    ///   - `maximum_payload_size`: Optional bound on the uncompressed payload size, which also bounds the memory that
    ///     decompression may allocate.
    fn read_entry(
        path: &Path,
        digest: &DiskCacheDigest,
        maximum_payload_size: Option<usize>,
    ) -> std::io::Result<Option<Vec<u8>>> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let mut envelope = Vec::new();
        match maximum_payload_size {
            None => {
                GzDecoder::new(file).read_to_end(&mut envelope)?;
            }
            Some(maximum_payload_size) => {
                let maximum_envelope_size = ENTRY_HEADER_LENGTH + maximum_payload_size;
                GzDecoder::new(file).take(maximum_envelope_size as u64 + 1).read_to_end(&mut envelope)?;
                if envelope.len() > maximum_envelope_size {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "persistent cache payload exceeds the {} MiB limit",
                            maximum_payload_size / (1024 * 1024),
                        ),
                    ));
                }
            }
        }

        // Validate the complete header before slicing it, so the fixed-width slice conversions below cannot fail.
        if envelope.len() < ENTRY_HEADER_LENGTH {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent cache entry is shorter than its header",
            ));
        }

        if &envelope[..ENTRY_MAGIC.len()] != ENTRY_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent cache entry has an invalid magic value",
            ));
        }

        let mut offset = ENTRY_MAGIC.len();
        let version = u32::from_le_bytes(envelope[offset..offset + 4].try_into().unwrap());
        offset += 4;
        if version != ENTRY_FORMAT_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("persistent cache entry uses unsupported format version {version}",),
            ));
        }

        // Reject an artifact addressed by another key before interpreting its payload.
        let stored_digest = &envelope[offset..offset + SHA256_LENGTH];
        offset += SHA256_LENGTH;
        if stored_digest != digest.as_bytes() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent cache entry key digest does not match its requested key",
            ));
        }

        let payload_length = u64::from_le_bytes(envelope[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let payload_length = usize::try_from(payload_length).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent cache payload length does not fit in memory",
            )
        })?;

        let stored_checksum = &envelope[offset..offset + SHA256_LENGTH];
        offset += SHA256_LENGTH;
        let payload = &envelope[offset..];

        // Require an exact payload length, including rejecting trailing bytes, before checking its integrity.
        if payload.len() != payload_length {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("persistent cache payload declares {} bytes but contains {}", payload_length, payload.len()),
            ));
        }

        if Sha256::digest(payload).as_slice() != stored_checksum {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent cache payload checksum does not match",
            ));
        }

        Ok(Some(payload.to_vec()))
    }

    /// Atomically writes `data` as the entry at `path` for `digest` and then evicts older entries when a size cap is
    /// configured. The entry is first written gzip-compressed to an exclusively created temporary file, synchronized,
    /// and then renamed over `path`, so that readers observe either the prior complete entry or the new complete entry.
    /// The temporary file is removed when any step before the rename fails.
    fn write_entry(&self, path: &Path, digest: &DiskCacheDigest, extension: &str, data: &[u8]) -> std::io::Result<()> {
        let (temporary_path, temporary_file) = self.temporary_entry(digest, extension)?;
        let write_result = (|| -> std::io::Result<()> {
            // Fast compression bounds the host overhead added after an already expensive compilation.
            let mut encoder = GzEncoder::new(temporary_file, Compression::fast());

            // Keep the versioned header in wire order and append the payload. The digest binds the entry to its
            // requested key, while the checksum detects payload corruption.
            let payload_checksum = Sha256::digest(data);
            let payload_length = u64::try_from(data.len()).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "persistent cache payload length does not fit in `u64`",
                )
            })?;

            encoder.write_all(ENTRY_MAGIC)?;
            encoder.write_all(&ENTRY_FORMAT_VERSION.to_le_bytes())?;
            encoder.write_all(digest.as_bytes())?;
            encoder.write_all(&payload_length.to_le_bytes())?;
            encoder.write_all(payload_checksum.as_slice())?;
            encoder.write_all(data)?;
            encoder.finish()?.sync_all()
        })();

        if let Err(error) = write_result.and_then(|()| fs::rename(&temporary_path, path)) {
            return Self::remove_temporary_entry(&temporary_path, error);
        }

        // Evict only after the atomic rename succeeds. Count completed executable and auxiliary entries,
        // leaving temporary files alone because another process may still be writing them.
        if let Some(max_bytes) = self.max_bytes {
            let mut total_bytes = 0u64;
            let mut candidates = Vec::new();
            for entry in fs::read_dir(&self.directory)? {
                let entry = entry?;
                let entry_path = entry.path();
                if !Self::is_entry_path(&entry_path) {
                    continue;
                }
                let metadata = entry.metadata()?;
                total_bytes = total_bytes.saturating_add(metadata.len());
                if entry_path != path {
                    candidates.push((entry_path, metadata.modified()?, metadata.len()));
                }
            }

            if total_bytes <= max_bytes {
                return Ok(());
            }

            // Evict by write time rather than read activity. This eviction pass keeps the just-published entry even
            // when it alone exceeds the cap.
            candidates.sort_by_key(|(_, modified, _)| *modified);
            let mut bytes_to_evict = total_bytes - max_bytes;
            for (entry_path, _, size) in candidates {
                if bytes_to_evict == 0 {
                    break;
                }

                match fs::remove_file(&entry_path) {
                    Ok(()) => {
                        bytes_to_evict = bytes_to_evict.saturating_sub(size);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        // Another cache instance may already have removed this candidate.
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            }
        }

        Ok(())
    }

    /// Exclusively creates a temporary file for an entry without truncating the active write of another cache instance.
    /// Existing files, including symbolic links and files abandoned by an earlier process with the same process
    /// identifier, are skipped, while all other creation errors are returned without deleting anything.
    fn temporary_entry(&self, digest: &DiskCacheDigest, extension: &str) -> std::io::Result<(PathBuf, File)> {
        for _ in 0..MAXIMUM_TEMPORARY_FILE_ATTEMPTS {
            let counter = self.write_counter.fetch_add(1, Ordering::Relaxed);
            let path =
                self.directory.join(format!("{}.{}.tmp.{}.{}", digest.to_hex(), extension, process::id(), counter));
            match File::options().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok((path, file)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("could not reserve a temporary cache file after {MAXIMUM_TEMPORARY_FILE_ATTEMPTS} attempts"),
        ))
    }

    /// Removes the temporary file at `path` after a failed write and returns `error`, extended with the cleanup failure
    /// if the file could not be removed.
    fn remove_temporary_entry(path: &Path, error: std::io::Error) -> std::io::Result<()> {
        match fs::remove_file(path) {
            Ok(()) => Err(error),
            Err(cleanup_error) if cleanup_error.kind() == std::io::ErrorKind::NotFound => Err(error),
            Err(cleanup_error) => Err(std::io::Error::new(
                error.kind(),
                format!("{error}, and failed to clean up the temporary cache entry: {cleanup_error}"),
            )),
        }
    }
}

/// SHA-256 digest identifying a [`DiskCache`] entry in its filename and serialized envelope. Executable entries hash
/// the domain's persistent cache key, while auxiliary entries hash their namespace and key.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct DiskCacheDigest {
    /// Raw digest bytes.
    bytes: [u8; SHA256_LENGTH],
}

impl DiskCacheDigest {
    /// Hashes the canonical persistent key bytes provided by a compilation domain.
    #[inline]
    fn from_key(key: &[u8]) -> Self {
        Self { bytes: Sha256::digest(key).into() }
    }

    /// Returns the raw digest bytes.
    #[inline]
    fn as_bytes(&self) -> &[u8; SHA256_LENGTH] {
        &self.bytes
    }

    /// Returns the lowercase hexadecimal encoding of this digest, which is suitable for filenames.
    #[inline]
    fn to_hex(&self) -> String {
        self.bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

// TODO(eaplatanios): Review from here onwards.

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::fs::FileTimes;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Barrier, Condvar, Mutex, mpsc};
    use std::thread;
    use std::time::{Duration, SystemTime};

    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::captures::CaptureReference;
    use crate::compilation::domains::{CompilationDomain, Compile};
    use crate::compilation::functions::{CompilationTracer, StagedFunction, StagedProgram};
    use crate::contexts::Domain;
    use crate::programs::{ProgramError, TypedProgram};

    use super::*;

    /// Lowered program of [`TestDomain`], which holds only its output types.
    struct TestLoweredProgram(Vec<ArrayType>);

    impl TypedProgram for TestLoweredProgram {
        type Type = ArrayType;

        fn input_count(&self) -> usize {
            0
        }

        fn input_types(&self) -> Cow<'_, [ArrayType]> {
            Cow::Borrowed(&[])
        }

        fn output_count(&self) -> usize {
            self.0.len()
        }

        fn output_types(&self) -> Cow<'_, [ArrayType]> {
            Cow::Borrowed(&self.0)
        }
    }

    /// Compiled program of [`TestDomain`], which holds only its number of outputs.
    #[derive(Debug, PartialEq, Eq)]
    struct TestCompiledProgram(usize);

    impl TypedProgram for TestCompiledProgram {
        type Type = ArrayType;

        fn input_count(&self) -> usize {
            0
        }

        fn input_types(&self) -> Cow<'_, [ArrayType]> {
            Cow::Borrowed(&[])
        }

        fn output_count(&self) -> usize {
            self.0
        }

        fn output_types(&self) -> Cow<'_, [ArrayType]> {
            Cow::Owned(vec![ArrayType::scalar(DataType::F32); self.0])
        }
    }

    #[derive(Clone, Default)]
    struct TestDomain {
        persistent: bool,
        serialization_fails: bool,
        serialization_unsupported: bool,
        deserialization_incompatible: bool,

        /// Synthetic key used by fixtures that exercise particular cache entries.
        cache_key: u8,

        /// Accepts synthetic fixture artifacts independently of the lowered output count.
        accept_all_programs: bool,

        /// One-shot backend compiler shared by clones of this fixture domain.
        compiler: Option<Arc<Mutex<Option<Box<dyn FnOnce() -> Result<TestCompiledProgram, ProgramError> + Send>>>>>,
    }

    impl TestDomain {
        fn persistent() -> Self {
            Self { persistent: true, ..Self::default() }
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
        type LoweredProgram = TestLoweredProgram;
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

        fn lower(&self, staged: &StagedProgram<Self>) -> Result<Self::LoweredProgram, Self::Error> {
            Ok(TestLoweredProgram(staged.output_types().to_vec()))
        }

        fn compile(&self, program: &Self::LoweredProgram) -> Result<Arc<Self::CompiledProgram>, Self::Error> {
            Ok(Arc::new(TestCompiledProgram(program.0.len())))
        }

        fn execute(&self, _program: &Self::CompiledProgram, _inputs: Vec<Array>) -> Result<Vec<Array>, Self::Error> {
            Ok(Vec::new())
        }
    }

    impl CompilationCacheDomain for TestDomain {
        type CacheKey = u8;

        fn compilation_key(&self, _program: &TestLoweredProgram) -> Result<u8, ProgramError> {
            Ok(self.cache_key)
        }

        fn compile_uncached(&self, program: &TestLoweredProgram) -> Result<TestCompiledProgram, ProgramError> {
            if let Some(compiler) = &self.compiler {
                // Release the fixture mutex before invoking callbacks that can block or panic.
                let compiler = compiler.lock().unwrap().take().unwrap();
                return compiler();
            }
            Ok(TestCompiledProgram(program.0.len()))
        }

        // Direct compilation tests require matching output counts. Synthetic cache fixtures accept every artifact.
        fn validate_cached_program(
            &self,
            program: &TestLoweredProgram,
            compiled: &TestCompiledProgram,
        ) -> Result<(), ProgramError> {
            if self.accept_all_programs || compiled.0 == program.0.len() {
                Ok(())
            } else {
                Err(ProgramError::InvalidArgument {
                    message: format!("cached program {} cannot serve {} outputs", compiled.0, program.0.len()),
                })
            }
        }

        fn persistent_cache_key(&self, key: &u8) -> Option<Vec<u8>> {
            self.persistent.then(|| vec![*key])
        }

        fn serialize_program(&self, program: &Self::CompiledProgram) -> Result<Option<Vec<u8>>, Self::Error> {
            if self.serialization_fails {
                return Err(ProgramError::MalformedProgram("expected serialization failure".to_string()));
            }
            if self.serialization_unsupported {
                return Ok(None);
            }
            Ok(self.persistent.then(|| program.0.to_le_bytes().to_vec()))
        }

        fn deserialize_program(&self, bytes: &[u8]) -> Result<Option<Self::CompiledProgram>, Self::Error> {
            if !self.persistent || self.deserialization_incompatible {
                return Ok(None);
            }
            let bytes: [u8; size_of::<usize>()] = bytes.try_into().map_err(|_| {
                ProgramError::MalformedProgram("test persistent payload has the wrong length".to_string())
            })?;
            Ok(Some(TestCompiledProgram(usize::from_le_bytes(bytes))))
        }
    }

    /// Compiles a synthetic cache request through [`CompilationContext::compile`] with a one-shot backend compiler.
    /// Direct calls keep the fixture's output-count validation enabled.
    fn compile_with(
        context: &CompilationContext<'_, TestDomain>,
        domain: &TestDomain,
        cache_key: u8,
        compiler: impl 'static + Send + FnOnce() -> Result<TestCompiledProgram, ProgramError>,
    ) -> Result<Arc<TestCompiledProgram>, ProgramError> {
        let domain = TestDomain {
            cache_key,
            accept_all_programs: true,
            compiler: Some(Arc::new(Mutex::new(Some(Box::new(compiler))))),
            ..domain.clone()
        };
        context.compile(&domain, &TestLoweredProgram(Vec::new()))
    }

    #[derive(Default)]
    struct TestExchangeState {
        artifacts: Mutex<HashMap<Vec<u8>, Result<Vec<u8>, String>>>,
        ready: Condvar,
        compatibility_error: Mutex<Option<CompilationExchangeError>>,
        receive_fails: AtomicBool,
        receive_incompatible: AtomicBool,
        publish_fails: AtomicBool,
        failure_publish_fails: AtomicBool,
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

        fn validate_compatibility(&self, _key: &[u8], _timeout: Duration) -> Result<(), CompilationExchangeError> {
            match self.state.compatibility_error.lock().unwrap().as_ref() {
                Some(error) => Err(error.clone()),
                None => Ok(()),
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
            if self.state.failure_publish_fails.load(Ordering::Relaxed) {
                return Err(CompilationExchangeError::Failed {
                    message: "expected failure notification publication failure".to_string(),
                });
            }
            self.state.artifacts.lock().unwrap().insert(key.to_vec(), Err(message.to_string()));
            self.state.ready.notify_all();
            Ok(())
        }

        fn receive(&self, key: &[u8], timeout: Duration) -> Result<Option<Vec<u8>>, CompilationExchangeError> {
            if self.state.receive_fails.load(Ordering::Relaxed) {
                return Err(CompilationExchangeError::Failed { message: "expected receive failure".to_string() });
            }
            if self.state.receive_incompatible.load(Ordering::Relaxed) {
                return Err(CompilationExchangeError::Incompatible {
                    message: "expected receive incompatibility".to_string(),
                });
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

    /// Exchange that borrows another exchange, which contexts accept for as long as the borrowed exchange lives.
    struct BorrowedExchange<'o>(&'o TestExchange);

    impl CompilationArtifactExchange for BorrowedExchange<'_> {
        fn process_index(&self) -> usize {
            self.0.process_index()
        }

        fn process_count(&self) -> usize {
            self.0.process_count()
        }

        fn validate_compatibility(&self, key: &[u8], timeout: Duration) -> Result<(), CompilationExchangeError> {
            self.0.validate_compatibility(key, timeout)
        }

        fn publish(&self, key: &[u8], artifact: &[u8]) -> Result<(), CompilationExchangeError> {
            self.0.publish(key, artifact)
        }

        fn publish_failure(&self, key: &[u8], message: &str) -> Result<(), CompilationExchangeError> {
            self.0.publish_failure(key, message)
        }

        fn receive(&self, key: &[u8], timeout: Duration) -> Result<Option<Vec<u8>>, CompilationExchangeError> {
            self.0.receive(key, timeout)
        }
    }

    /// Returns the steps retained by `context`, preserving their reported order.
    fn event_steps(context: &CompilationContext<'_, TestDomain>) -> Vec<CompilationStep> {
        context.recent_events().into_iter().map(|event| event.step).collect()
    }

    /// Returns the uncompressed envelope that the production cache writes for `payload` under `digest`.
    fn encode_envelope(digest: &DiskCacheDigest, payload: &[u8]) -> Vec<u8> {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap();
        cache.put(digest, payload).unwrap();
        let file = File::open(cache.entry_path(digest)).unwrap();
        let mut envelope = Vec::new();
        GzDecoder::new(file).read_to_end(&mut envelope).unwrap();
        envelope
    }

    /// Writes `envelope` verbatim as a gzip entry so read tests can inject malformed envelope bytes.
    fn write_compressed_envelope(path: &Path, envelope: &[u8]) {
        let file = File::create(path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(envelope).unwrap();
        encoder.finish().unwrap();
    }

    #[test]
    fn test_compilation_event_error_display() {
        for (error, message) in [
            (CompilationStepError::Unsupported, "required compilation artifact operation is unsupported"),
            (
                CompilationStepError::Incompatible { message: "compiler versions differ".to_string() },
                "compilation artifact or exchange compatibility requirements were not satisfied: \
                 compiler versions differ",
            ),
            (CompilationStepError::TimedOut, "compilation artifact exchange timed out"),
            (
                CompilationStepError::ReadFailed { message: "connection reset".to_string() },
                "failed to read or receive a compilation artifact: connection reset",
            ),
            (
                CompilationStepError::ExchangeFailed { message: "leader disconnected".to_string() },
                "compilation artifact exchange coordination failed: leader disconnected",
            ),
            (
                CompilationStepError::DeserializationFailed { message: "invalid executable header".to_string() },
                "failed to deserialize a compiled program: invalid executable header",
            ),
            (
                CompilationStepError::SerializationFailed { message: "executable serialization failed".to_string() },
                "failed to serialize a compiled program: executable serialization failed",
            ),
            (
                CompilationStepError::WriteFailed { message: "disk quota exceeded".to_string() },
                "failed to write or publish a compilation artifact or failure notification: disk quota exceeded",
            ),
            (
                CompilationStepError::CompilationFailed { message: "unsupported operation".to_string() },
                "backend compilation failed: unsupported operation",
            ),
        ] {
            let diagnostic: &dyn std::error::Error = &error;
            assert_eq!(diagnostic.to_string(), message);
            assert!(diagnostic.source().is_none());
        }
    }

    #[test]
    fn test_compilation_context_with_capacity() {
        let context = CompilationContext::<TestDomain>::with_capacity(1);
        let domain = TestDomain::default();
        let producer_calls = Arc::new(AtomicUsize::new(0));
        for key in [1, 2, 1] {
            compile_with(&context, &domain, key, {
                let producer_calls = Arc::clone(&producer_calls);
                move || {
                    producer_calls.fetch_add(1, Ordering::Relaxed);
                    Ok(TestCompiledProgram(key as usize))
                }
            })
            .unwrap();
        }

        // The second key evicts the first one, which is then compiled again.
        assert_eq!(producer_calls.load(Ordering::Relaxed), 3);
        assert_eq!(context.program_count(), 1);

        // A zero capacity is clamped to one entry.
        let context = CompilationContext::<TestDomain>::with_capacity(0);
        compile_with(&context, &domain, 1, move || Ok(TestCompiledProgram(1))).unwrap();
        assert_eq!(context.program_count(), 1);
    }

    #[test]
    fn test_compilation_context_with_disk_cache() {
        let directory = tempfile::tempdir().unwrap();
        let domain = TestDomain::persistent();
        let disk_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let first_context = CompilationContext::<TestDomain>::new().with_event_capacity(16).with_disk_cache(disk_cache);
        compile_with(&first_context, &domain, 3, move || Ok(TestCompiledProgram(17))).unwrap();
        assert_eq!(first_context.disk_cache().unwrap().directory(), directory.path());

        let disk_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let second_context =
            CompilationContext::<TestDomain>::new().with_event_capacity(16).with_disk_cache(disk_cache);
        let restored =
            compile_with(&second_context, &domain, 3, move || -> Result<TestCompiledProgram, ProgramError> {
                panic!("persistent hit must not invoke producer")
            })
            .unwrap();

        assert_eq!(restored.0, 17);
        assert_eq!(second_context.statistics().persistent_hits, 1);
        assert_eq!(second_context.statistics().compilations, 0);

        assert_eq!(
            event_steps(&first_context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Persistent,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::PersistentStore { outcome: CompilationStepOutcome::Succeeded },
            ],
        );
        assert_eq!(
            event_steps(&second_context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Persistent,
                    outcome: CompilationLookupOutcome::Hit,
                },
            ],
        );
        assert_eq!(second_context.statistics().persistent_lookup_duration, second_context.recent_events()[1].duration);
    }

    #[test]
    fn test_compilation_context_with_disk_cache_lookup_failures() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::MAX, 0);
        cache.put(&DiskCacheDigest::from_key(&[1]), b"invalid payload").unwrap();
        let context = CompilationContext::<TestDomain>::new().with_disk_cache(cache).with_event_capacity(8);
        let program =
            compile_with(&context, &TestDomain::persistent(), 1, move || Ok(TestCompiledProgram(17))).unwrap();
        assert_eq!(program.0, 17);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Persistent,
                    outcome: CompilationLookupOutcome::Failed(CompilationStepError::DeserializationFailed {
                        message: "encountered malformed program: test persistent payload has the wrong length"
                            .to_string()
                    }),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::PersistentStore {
                    outcome: CompilationStepOutcome::Skipped(CompilationSkipReason::BelowWriteThreshold),
                },
            ],
        );
        assert_eq!(context.statistics().persistent_errors, 1);

        // An incompatible artifact is a completed lookup miss, rather than a failed lookup.
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::MAX, 0);
        cache.put(&DiskCacheDigest::from_key(&[2]), &19usize.to_le_bytes()).unwrap();
        let context = CompilationContext::<TestDomain>::new().with_disk_cache(cache).with_event_capacity(8);
        let domain = TestDomain { persistent: true, deserialization_incompatible: true, ..TestDomain::default() };
        assert_eq!(compile_with(&context, &domain, 2, move || Ok(TestCompiledProgram(23))).unwrap().0, 23);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Persistent,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::Incompatible),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::PersistentStore {
                    outcome: CompilationStepOutcome::Skipped(CompilationSkipReason::BelowWriteThreshold),
                },
            ],
        );
        assert_eq!(context.statistics().persistent_errors, 0);

        // A directory at the artifact path deterministically makes the read fail without filesystem permissions.
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::MAX, 0);
        fs::create_dir(cache.entry_path(&DiskCacheDigest::from_key(&[3]))).unwrap();
        let read_message = cache.get(&DiskCacheDigest::from_key(&[3])).unwrap_err().to_string();
        let context = CompilationContext::<TestDomain>::new().with_disk_cache(cache).with_event_capacity(8);
        let program =
            compile_with(&context, &TestDomain::persistent(), 3, move || Ok(TestCompiledProgram(29))).unwrap();
        assert_eq!(program.0, 29);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Persistent,
                    outcome: CompilationLookupOutcome::Failed(CompilationStepError::ReadFailed {
                        message: read_message
                    }),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::PersistentStore {
                    outcome: CompilationStepOutcome::Skipped(CompilationSkipReason::BelowWriteThreshold),
                },
            ],
        );
        assert_eq!(context.statistics().persistent_errors, 1);
    }

    #[test]
    fn test_compilation_context_with_disk_cache_unsupported_domain() {
        let directory = tempfile::tempdir().unwrap();
        let context =
            CompilationContext::<TestDomain>::new().with_disk_cache(DiskCache::open(directory.path()).unwrap());
        compile_with(&context, &TestDomain::default(), 1, move || Ok(TestCompiledProgram(1))).unwrap();

        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(context.statistics().persistent_errors, 0);
    }

    #[test]
    fn test_compilation_context_with_disk_cache_default_write_thresholds() {
        let directory = tempfile::tempdir().unwrap();
        let context =
            CompilationContext::<TestDomain>::new().with_disk_cache(DiskCache::open(directory.path()).unwrap());
        compile_with(&context, &TestDomain::persistent(), 1, move || Ok(TestCompiledProgram(1))).unwrap();

        // A fast compilation is below the default minimum compilation duration, so it is never serialized.
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[test]
    fn test_compilation_context_with_disk_cache_below_minimum_entry_size() {
        // A serialized program that is smaller than the minimum entry size is skipped after serialization.
        let directory = tempfile::tempdir().unwrap();
        let disk_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, usize::MAX);
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(8).with_disk_cache(disk_cache);
        compile_with(&context, &TestDomain::persistent(), 1, move || Ok(TestCompiledProgram(1))).unwrap();
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(context.statistics().persistent_errors, 0);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Persistent,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::PersistentStore {
                    outcome: CompilationStepOutcome::Skipped(CompilationSkipReason::BelowWriteThreshold),
                },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_disk_cache_serialization_failure() {
        let directory = tempfile::tempdir().unwrap();
        let disk_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let context = CompilationContext::<TestDomain>::new().with_disk_cache(disk_cache).with_event_capacity(8);
        let domain = TestDomain { persistent: true, serialization_fails: true, ..TestDomain::default() };

        // Serialization failures degrade to an in-memory result and are recorded separately from persistent write
        // failures.
        let program = compile_with(&context, &domain, 1, move || Ok(TestCompiledProgram(3))).unwrap();
        assert_eq!(program.0, 3);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(context.statistics().persistent_errors, 1);
        let last_event = context.recent_events().pop().unwrap();
        assert_eq!(
            last_event.step,
            CompilationStep::PersistentStore {
                outcome: CompilationStepOutcome::Failed(CompilationStepError::SerializationFailed {
                    message: "encountered malformed program: expected serialization failure".to_string()
                }),
            },
        );
    }

    #[test]
    fn test_compilation_context_with_disk_cache_write_failure() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let entry_path = cache.entry_path(&DiskCacheDigest::from_key(&[1]));
        let context = CompilationContext::<TestDomain>::new().with_disk_cache(cache).with_event_capacity(8);

        // Create the obstruction after the lookup, so only the store fails and compilation still succeeds.
        let program = compile_with(&context, &TestDomain::persistent(), 1, {
            let entry_path = entry_path.clone();
            move || {
                fs::create_dir(&entry_path).unwrap();
                Ok(TestCompiledProgram(3))
            }
        })
        .unwrap();
        assert_eq!(program.0, 3);
        assert_eq!(context.statistics().persistent_errors, 1);
        let write_message = context
            .disk_cache()
            .unwrap()
            .put(&DiskCacheDigest::from_key(&[1]), &3usize.to_le_bytes())
            .unwrap_err()
            .to_string();
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Persistent,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::PersistentStore {
                    outcome: CompilationStepOutcome::Failed(CompilationStepError::WriteFailed {
                        message: write_message
                    }),
                },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange() {
        let state = Arc::new(TestExchangeState::default());
        let policy = CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(5) };
        let follower_context = CompilationContext::<TestDomain>::new()
            .with_event_capacity(16)
            .with_artifact_exchange(Arc::new(TestExchange::new(1, Arc::clone(&state))), policy);
        assert_eq!(follower_context.artifact_exchange().unwrap().process_index(), 1);
        assert_eq!(follower_context.artifact_exchange_policy(), policy);
        let follower = thread::spawn(move || {
            let program = compile_with(
                &follower_context,
                &TestDomain::persistent(),
                7,
                move || -> Result<TestCompiledProgram, ProgramError> {
                    panic!("follower must restore the leader artifact")
                },
            )
            .unwrap();
            (program.0, follower_context.statistics(), follower_context.recent_events())
        });

        let leader_context = CompilationContext::<TestDomain>::new()
            .with_event_capacity(16)
            .with_artifact_exchange(Arc::new(TestExchange::new(0, Arc::clone(&state))), policy);
        let leader =
            compile_with(&leader_context, &TestDomain::persistent(), 7, move || Ok(TestCompiledProgram(29))).unwrap();
        let (follower_value, follower_statistics, follower_events) = follower.join().unwrap();

        assert_eq!(leader.0, 29);
        assert_eq!(follower_value, 29);
        assert_eq!(leader_context.statistics().compilations, 1);
        assert_eq!(leader_context.statistics().exchange_publishes, 1);
        assert_eq!(follower_statistics.compilations, 0);
        assert_eq!(follower_statistics.exchange_hits, 1);
        assert_eq!(follower_statistics.exchange_waits, 1);

        assert_eq!(
            event_steps(&leader_context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangePublish { outcome: CompilationStepOutcome::Succeeded },
            ],
        );
        assert_eq!(
            follower_events.iter().map(|event| event.step.clone()).collect::<Vec<_>>(),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangeReceive { outcome: CompilationLookupOutcome::Hit },
            ],
        );
        // The exchange duration covers compatibility validation and receptions, but never publication.
        assert_eq!(leader_context.statistics().exchange_duration, leader_context.recent_events()[1].duration);
        assert_eq!(follower_statistics.exchange_duration, follower_events[1].duration + follower_events[2].duration);
    }

    #[test]
    fn test_compilation_context_with_borrowed_artifact_exchange() {
        let state = Arc::new(TestExchangeState::default());
        let leader_exchange = TestExchange::new(0, Arc::clone(&state));
        let follower_exchange = TestExchange::new(1, state);
        let policy = CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(5) };

        // The leader publishes before the follower looks up the key, so both contexts can borrow exchanges that live on
        // this stack frame instead of owning them.
        let leader_context = CompilationContext::<TestDomain>::new()
            .with_artifact_exchange(Arc::new(BorrowedExchange(&leader_exchange)), policy);
        let leader =
            compile_with(&leader_context, &TestDomain::persistent(), 7, move || Ok(TestCompiledProgram(29))).unwrap();
        let follower_context = CompilationContext::<TestDomain>::new()
            .with_artifact_exchange(Arc::new(BorrowedExchange(&follower_exchange)), policy);
        let follower = compile_with(
            &follower_context,
            &TestDomain::persistent(),
            7,
            move || -> Result<TestCompiledProgram, ProgramError> {
                panic!("follower must restore the leader artifact")
            },
        )
        .unwrap();

        assert_eq!(leader.0, 29);
        assert_eq!(follower.0, 29);
        assert_eq!(leader_context.statistics().exchange_publishes, 1);
        assert_eq!(follower_context.statistics().exchange_hits, 1);
        assert_eq!(follower_context.statistics().compilations, 0);
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_publishes_leader_persistent_hit() {
        let directory = tempfile::tempdir().unwrap();
        let domain = TestDomain::persistent();
        let seed_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        compile_with(
            &CompilationContext::<TestDomain>::new().with_event_capacity(16).with_disk_cache(seed_cache),
            &domain,
            8,
            move || Ok(TestCompiledProgram(53)),
        )
        .unwrap();

        let state = Arc::new(TestExchangeState::default());
        let policy = CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(5) };
        let follower_context = CompilationContext::<TestDomain>::new()
            .with_event_capacity(16)
            .with_artifact_exchange(Arc::new(TestExchange::new(1, Arc::clone(&state))), policy);
        let follower = thread::spawn(move || {
            compile_with(
                &follower_context,
                &TestDomain::persistent(),
                8,
                move || -> Result<TestCompiledProgram, ProgramError> {
                    panic!("follower must restore the persistent artifact of the leader")
                },
            )
            .unwrap()
            .0
        });

        let leader_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let leader_context = CompilationContext::<TestDomain>::new()
            .with_event_capacity(16)
            .with_disk_cache(leader_cache)
            .with_artifact_exchange(Arc::new(TestExchange::new(0, state)), policy);
        let leader = compile_with(&leader_context, &domain, 8, move || -> Result<TestCompiledProgram, ProgramError> {
            panic!("leader must restore its persistent artifact")
        })
        .unwrap();

        assert_eq!(leader.0, 53);
        assert_eq!(follower.join().unwrap(), 53);
        assert_eq!(leader_context.statistics().persistent_hits, 1);
        assert_eq!(leader_context.statistics().exchange_publishes, 1);
        assert_eq!(leader_context.statistics().compilations, 0);

        assert_eq!(
            event_steps(&leader_context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Persistent,
                    outcome: CompilationLookupOutcome::Hit,
                },
                CompilationStep::ExchangePublish { outcome: CompilationStepOutcome::Succeeded },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_publishes_leader_compilation_failure() {
        let state = Arc::new(TestExchangeState::default());
        let policy = CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(30) };
        let leader_context = CompilationContext::<TestDomain>::new()
            .with_event_capacity(16)
            .with_artifact_exchange(Arc::new(TestExchange::new(0, Arc::clone(&state))), policy);
        let producer_error = ProgramError::MalformedProgram("expected test failure".to_string());
        assert_eq!(
            compile_with(&leader_context, &TestDomain::persistent(), 9, {
                let producer_error = producer_error.clone();
                move || Err(producer_error.clone())
            })
            .err(),
            Some(producer_error.clone()),
        );

        // The follower observes the terminal failure published by the leader instead of waiting for its deadline.
        let follower_context = CompilationContext::<TestDomain>::new()
            .with_event_capacity(16)
            .with_artifact_exchange(Arc::new(TestExchange::new(1, state)), policy);
        let start = Instant::now();
        assert_eq!(
            compile_with(&follower_context, &TestDomain::persistent(), 9, move || Ok(TestCompiledProgram(59))).err(),
            Some(ProgramError::InvalidArgument {
                message: format!("compilation artifact exchange requirement was not satisfied: {producer_error}"),
            }),
        );
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(follower_context.statistics().exchange_errors, 1);
        assert_eq!(follower_context.statistics().compilations, 0);

        assert_eq!(
            event_steps(&leader_context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::BackendCompilation {
                    outcome: CompilationStepOutcome::Failed(CompilationStepError::CompilationFailed {
                        message: "encountered malformed program: expected test failure".to_string()
                    }),
                },
                CompilationStep::ExchangeFailureNotification {
                    cause: CompilationStepError::CompilationFailed {
                        message: "encountered malformed program: expected test failure".to_string()
                    },
                    outcome: CompilationStepOutcome::Succeeded,
                },
            ],
        );
        assert_eq!(
            event_steps(&follower_context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangeReceive {
                    outcome: CompilationLookupOutcome::Failed(CompilationStepError::ReadFailed {
                        message:
                            "compilation artifact exchange failed: encountered malformed program: expected test failure"
                                .to_string()
                    }),
                },
            ],
        );
        assert_eq!(leader_context.statistics().compilation_duration, leader_context.recent_events()[2].duration);
        assert_eq!(leader_context.statistics().exchange_duration, leader_context.recent_events()[1].duration);
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_failure_notification_failure() {
        let state = Arc::new(TestExchangeState::default());
        state.failure_publish_fails.store(true, Ordering::Relaxed);
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(8).with_artifact_exchange(
            Arc::new(TestExchange::new(0, state)),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(1) },
        );
        let producer_error = ProgramError::MalformedProgram("expected producer failure".to_string());

        assert_eq!(
            compile_with(&context, &TestDomain::persistent(), 1, {
                let producer_error = producer_error.clone();
                move || Err(producer_error.clone())
            })
            .err(),
            Some(producer_error),
        );
        assert_eq!(context.statistics().exchange_errors, 1);
        assert_eq!(context.statistics().compilations, 1);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::BackendCompilation {
                    outcome: CompilationStepOutcome::Failed(CompilationStepError::CompilationFailed {
                        message: "encountered malformed program: expected producer failure".to_string()
                    }),
                },
                CompilationStep::ExchangeFailureNotification {
                    cause: CompilationStepError::CompilationFailed {
                        message: "encountered malformed program: expected producer failure".to_string()
                    },
                    outcome: CompilationStepOutcome::Failed(CompilationStepError::WriteFailed {
                        message:
                            "compilation artifact exchange failed: expected failure notification publication failure"
                                .to_string()
                    }),
                },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_publishes_leader_serialization_failure() {
        let state = Arc::new(TestExchangeState::default());
        let leader_context = CompilationContext::<TestDomain>::new().with_event_capacity(16).with_artifact_exchange(
            Arc::new(TestExchange::new(0, Arc::clone(&state))),
            CompilationArtifactExchangePolicy::PreferSharing {
                timeout: Duration::from_secs(30),
                fallback_to_local_compile: true,
            },
        );
        let domain = TestDomain { persistent: true, serialization_fails: true, ..TestDomain::default() };

        // The leader keeps its program when the policy permits a fallback, but still releases its followers.
        let program = compile_with(&leader_context, &domain, 10, move || Ok(TestCompiledProgram(61))).unwrap();
        assert_eq!(program.0, 61);
        assert_eq!(leader_context.statistics().exchange_errors, 1);
        assert_eq!(leader_context.statistics().exchange_fallbacks, 1);
        assert_eq!(leader_context.statistics().exchange_publishes, 0);
        assert_eq!(
            state.artifacts.lock().unwrap().get([10u8].as_slice()),
            Some(&Err(ProgramError::MalformedProgram("expected serialization failure".to_string()).to_string())),
        );

        assert_eq!(
            event_steps(&leader_context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangePublish {
                    outcome: CompilationStepOutcome::Failed(CompilationStepError::SerializationFailed {
                        message: "encountered malformed program: expected serialization failure".to_string()
                    }),
                },
                CompilationStep::ExchangeFailureNotification {
                    cause: CompilationStepError::SerializationFailed {
                        message: "encountered malformed program: expected serialization failure".to_string()
                    },
                    outcome: CompilationStepOutcome::Succeeded,
                },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_serialization_failure_required() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let state = Arc::new(TestExchangeState::default());
        let context = CompilationContext::<TestDomain>::new()
            .with_disk_cache(cache)
            .with_event_capacity(8)
            .with_artifact_exchange(
                Arc::new(TestExchange::new(0, Arc::clone(&state))),
                CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(1) },
            );
        let domain = TestDomain { persistent: true, serialization_fails: true, ..TestDomain::default() };
        let error = ProgramError::MalformedProgram("expected serialization failure".to_string());
        let cause = CompilationStepError::SerializationFailed { message: error.to_string() };

        // All three destinations retain the diagnostic while the caller receives the original domain error.
        assert_eq!(compile_with(&context, &domain, 1, move || Ok(TestCompiledProgram(3))).err(), Some(error.clone()));
        assert_eq!(context.program_count(), 0);
        assert_eq!(context.statistics().persistent_errors, 1);
        assert_eq!(context.statistics().exchange_errors, 1);
        assert_eq!(context.statistics().exchange_fallbacks, 0);
        assert_eq!(context.statistics().compilations, 1);
        assert_eq!(state.artifacts.lock().unwrap().get([1u8].as_slice()), Some(&Err(error.to_string())));
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Persistent,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::PersistentStore { outcome: CompilationStepOutcome::Failed(cause.clone()) },
                CompilationStep::ExchangePublish { outcome: CompilationStepOutcome::Failed(cause.clone()) },
                CompilationStep::ExchangeFailureNotification { cause, outcome: CompilationStepOutcome::Succeeded },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_serialization_unsupported() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let state = Arc::new(TestExchangeState::default());
        let context = CompilationContext::<TestDomain>::new()
            .with_disk_cache(cache)
            .with_event_capacity(8)
            .with_artifact_exchange(
                Arc::new(TestExchange::new(0, state)),
                CompilationArtifactExchangePolicy::PreferSharing {
                    timeout: Duration::from_secs(1),
                    fallback_to_local_compile: true,
                },
            );
        let domain = TestDomain { persistent: true, serialization_unsupported: true, ..TestDomain::default() };

        assert_eq!(compile_with(&context, &domain, 1, move || Ok(TestCompiledProgram(3))).unwrap().0, 3);
        assert_eq!(context.statistics().persistent_errors, 0);
        assert_eq!(context.statistics().exchange_errors, 0);
        assert_eq!(context.statistics().exchange_fallbacks, 1);
        assert_eq!(context.statistics().exchange_publishes, 0);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Persistent,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::PersistentStore {
                    outcome: CompilationStepOutcome::Skipped(CompilationSkipReason::Unsupported),
                },
                CompilationStep::ExchangePublish {
                    outcome: CompilationStepOutcome::Skipped(CompilationSkipReason::Unsupported),
                },
                CompilationStep::ExchangeFailureNotification {
                    cause: CompilationStepError::Unsupported,
                    outcome: CompilationStepOutcome::Succeeded,
                },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_receive_failure_fallback() {
        let state = Arc::new(TestExchangeState::default());
        state.receive_fails.store(true, Ordering::Relaxed);
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(16).with_artifact_exchange(
            Arc::new(TestExchange::new(1, state)),
            CompilationArtifactExchangePolicy::PreferSharing {
                timeout: Duration::from_millis(10),
                fallback_to_local_compile: true,
            },
        );

        let program =
            compile_with(&context, &TestDomain::persistent(), 2, move || Ok(TestCompiledProgram(31))).unwrap();

        assert_eq!(program.0, 31);
        assert_eq!(context.statistics().exchange_errors, 1);
        assert_eq!(context.statistics().exchange_fallbacks, 1);
        assert_eq!(context.statistics().compilations, 1);

        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangeReceive {
                    outcome: CompilationLookupOutcome::Failed(CompilationStepError::ReadFailed {
                        message: "compilation artifact exchange failed: expected receive failure".to_string()
                    }),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
            ],
        );
        assert_eq!(
            context.statistics().exchange_duration,
            context.recent_events()[1].duration + context.recent_events()[2].duration,
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_receive_incompatible() {
        let state = Arc::new(TestExchangeState::default());
        state.receive_incompatible.store(true, Ordering::Relaxed);
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(8).with_artifact_exchange(
            Arc::new(TestExchange::new(1, state)),
            CompilationArtifactExchangePolicy::PreferSharing {
                timeout: Duration::from_secs(1),
                fallback_to_local_compile: true,
            },
        );
        assert_eq!(
            compile_with(&context, &TestDomain::persistent(), 1, move || Ok(TestCompiledProgram(3))).unwrap().0,
            3
        );
        assert_eq!(context.statistics().exchange_errors, 1);
        assert_eq!(context.statistics().exchange_fallbacks, 1);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangeReceive {
                    outcome: CompilationLookupOutcome::Failed(CompilationStepError::Incompatible {
                        message: "compilation artifact exchange is incompatible: expected receive incompatibility"
                            .to_string()
                    }),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
            ],
        );

        // Receiving bytes succeeds, but rejecting their backend compatibility is a lookup miss.
        let state = Arc::new(TestExchangeState::default());
        state.artifacts.lock().unwrap().insert(vec![1], Ok(5usize.to_le_bytes().to_vec()));
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(8).with_artifact_exchange(
            Arc::new(TestExchange::new(1, state)),
            CompilationArtifactExchangePolicy::PreferSharing {
                timeout: Duration::from_secs(1),
                fallback_to_local_compile: true,
            },
        );
        let domain = TestDomain { persistent: true, deserialization_incompatible: true, ..TestDomain::default() };
        assert_eq!(compile_with(&context, &domain, 1, move || Ok(TestCompiledProgram(7))).unwrap().0, 7);
        assert_eq!(context.statistics().exchange_errors, 0);
        assert_eq!(context.statistics().exchange_fallbacks, 1);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangeReceive {
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::Incompatible),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_deserialization_failure() {
        let state = Arc::new(TestExchangeState::default());
        state.artifacts.lock().unwrap().insert(vec![1], Ok(vec![0]));
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(8).with_artifact_exchange(
            Arc::new(TestExchange::new(1, state)),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(1) },
        );
        assert_eq!(
            compile_with(
                &context,
                &TestDomain::persistent(),
                1,
                move || -> Result<TestCompiledProgram, ProgramError> {
                    panic!("required sharing must not compile after deserialization fails")
                }
            )
            .err(),
            Some(ProgramError::MalformedProgram("test persistent payload has the wrong length".to_string())),
        );
        assert_eq!(context.statistics().exchange_errors, 1);
        assert_eq!(context.statistics().exchange_fallbacks, 0);
        assert_eq!(context.statistics().compilations, 0);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangeReceive {
                    outcome: CompilationLookupOutcome::Failed(CompilationStepError::DeserializationFailed {
                        message: "encountered malformed program: test persistent payload has the wrong length"
                            .to_string()
                    }),
                },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_publish_failure() {
        let state = Arc::new(TestExchangeState::default());
        state.publish_fails.store(true, Ordering::Relaxed);
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(8).with_artifact_exchange(
            Arc::new(TestExchange::new(0, state)),
            CompilationArtifactExchangePolicy::PreferSharing {
                timeout: Duration::from_secs(1),
                fallback_to_local_compile: true,
            },
        );
        assert_eq!(
            compile_with(&context, &TestDomain::persistent(), 1, move || Ok(TestCompiledProgram(3))).unwrap().0,
            3
        );
        assert_eq!(context.statistics().exchange_errors, 1);
        assert_eq!(context.statistics().exchange_fallbacks, 1);
        assert_eq!(context.statistics().exchange_publishes, 0);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangePublish {
                    outcome: CompilationStepOutcome::Failed(CompilationStepError::WriteFailed {
                        message: "compilation artifact exchange failed: expected publish failure".to_string()
                    }),
                },
            ],
        );
        // Exchange duration includes compatibility validation and excludes the failed publication.
        assert_eq!(context.statistics().exchange_duration, context.recent_events()[1].duration);
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_rejects_compatibility_disagreement() {
        let state = Arc::new(TestExchangeState::default());
        *state.compatibility_error.lock().unwrap() =
            Some(CompilationExchangeError::Incompatible { message: "expected exchange incompatibility".to_string() });
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(16).with_artifact_exchange(
            Arc::new(TestExchange::new(0, state)),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(1) },
        );
        let producer_calls = Arc::new(AtomicUsize::new(0));

        let result = compile_with(&context, &TestDomain::persistent(), 3, {
            let producer_calls = Arc::clone(&producer_calls);
            move || {
                producer_calls.fetch_add(1, Ordering::Relaxed);
                Ok(TestCompiledProgram(33))
            }
        });

        assert_eq!(
            result.err(),
            Some(ProgramError::InvalidArgument {
                message: "compilation artifact exchange requirement was not satisfied: \
                    compilation artifact exchange is incompatible: expected exchange incompatibility"
                    .to_string(),
            }),
        );
        assert_eq!(producer_calls.load(Ordering::Relaxed), 0);
        assert_eq!(context.statistics().exchange_errors, 1);
        assert_eq!(context.statistics().compilations, 0);

        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation {
                    outcome: CompilationStepOutcome::Failed(CompilationStepError::Incompatible {
                        message: "compilation artifact exchange is incompatible: expected exchange incompatibility"
                            .to_string()
                    }),
                },
            ],
        );
        assert_eq!(context.statistics().exchange_duration, context.recent_events()[1].duration);
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_compatibility_validation_failure() {
        let state = Arc::new(TestExchangeState::default());
        *state.compatibility_error.lock().unwrap() =
            Some(CompilationExchangeError::Failed { message: "expected coordination failure".to_string() });
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(8).with_artifact_exchange(
            Arc::new(TestExchange::new(0, state)),
            CompilationArtifactExchangePolicy::PreferSharing {
                timeout: Duration::from_secs(1),
                fallback_to_local_compile: true,
            },
        );

        assert_eq!(
            compile_with(&context, &TestDomain::persistent(), 1, move || Ok(TestCompiledProgram(3))).unwrap().0,
            3
        );
        assert_eq!(context.statistics().exchange_errors, 1);
        assert_eq!(context.statistics().exchange_fallbacks, 1);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation {
                    outcome: CompilationStepOutcome::Failed(CompilationStepError::ExchangeFailed {
                        message: "compilation artifact exchange failed: expected coordination failure".to_string(),
                    }),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_rejects_unsupported_domain_without_waiting() {
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(16).with_artifact_exchange(
            Arc::new(TestExchange::new(1, Arc::new(TestExchangeState::default()))),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(30) },
        );
        let start = Instant::now();

        let result = compile_with(&context, &TestDomain::default(), 4, move || Ok(TestCompiledProgram(35)));

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

        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation {
                    outcome: CompilationStepOutcome::Skipped(CompilationSkipReason::Unsupported),
                },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_required_timeout() {
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(16).with_artifact_exchange(
            Arc::new(TestExchange::new(1, Arc::new(TestExchangeState::default()))),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_millis(1) },
        );
        let producer_calls = Arc::new(AtomicUsize::new(0));

        let result = compile_with(&context, &TestDomain::persistent(), 4, {
            let producer_calls = Arc::clone(&producer_calls);
            move || {
                producer_calls.fetch_add(1, Ordering::Relaxed);
                Ok(TestCompiledProgram(37))
            }
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

        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangeReceive {
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::TimedOut),
                },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_artifact_exchange_permitted_timeout_fallback() {
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(16).with_artifact_exchange(
            Arc::new(TestExchange::new(1, Arc::new(TestExchangeState::default()))),
            CompilationArtifactExchangePolicy::PreferSharing {
                timeout: Duration::from_millis(1),
                fallback_to_local_compile: true,
            },
        );

        let program =
            compile_with(&context, &TestDomain::persistent(), 5, move || Ok(TestCompiledProgram(41))).unwrap();

        assert_eq!(program.0, 41);
        assert_eq!(context.statistics().exchange_timeouts, 1);
        assert_eq!(context.statistics().exchange_fallbacks, 1);
        assert_eq!(context.statistics().compilations, 1);

        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangeReceive {
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::TimedOut),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_event_capacity() {
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(2);
        let domain = TestDomain::default();
        compile_with(&context, &domain, 1, move || Ok(TestCompiledProgram(43))).unwrap();
        compile_with(&context, &domain, 1, move || Ok(TestCompiledProgram(47))).unwrap();

        // The memory miss of the first lookup is evicted by the two later events.
        let events = context.recent_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].step, CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded });
        assert_eq!(
            events[1].step,
            CompilationStep::CacheLookup { tier: CompilationCacheTier::Memory, outcome: CompilationLookupOutcome::Hit },
        );

        context.clear_events();
        assert_eq!(context.recent_events(), Vec::new());
    }

    #[test]
    fn test_compilation_context_with_event_reporter() {
        // The reporter borrows `reported`, which outlives the context.
        let reported = Mutex::new(Vec::new());
        let context = CompilationContext::<TestDomain>::new()
            .with_event_reporter(Arc::new(|event: &CompilationEvent| reported.lock().unwrap().push(event.clone())));
        compile_with(&context, &TestDomain::default(), 1, move || Ok(TestCompiledProgram(43))).unwrap();

        // The reporter observes every event even though retention is disabled by default.
        assert_eq!(context.recent_events(), Vec::new());
        let reported = reported.lock().unwrap();
        assert_eq!(
            reported.iter().map(|event| event.step.clone()).collect::<Vec<_>>(),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
            ],
        );
    }

    #[test]
    fn test_compilation_context_with_event_reporter_failure_diagnostics() {
        let reported = Arc::new(Mutex::new(Vec::new()));
        let reporter_events = Arc::clone(&reported);
        let context = CompilationContext::<TestDomain>::new()
            .with_event_capacity(2)
            .with_event_reporter(Arc::new(move |event| reporter_events.lock().unwrap().push(event.clone())));
        let error = ProgramError::MalformedProgram("expected reported producer failure".to_string());

        assert_eq!(
            compile_with(&context, &TestDomain::default(), 1, {
                let error = error.clone();
                move || Err(error.clone())
            })
            .err(),
            Some(error)
        );
        let retained = context.recent_events();
        assert_eq!(*reported.lock().unwrap(), retained);
        assert_eq!(
            retained.iter().map(|event| event.step.clone()).collect::<Vec<_>>(),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::BackendCompilation {
                    outcome: CompilationStepOutcome::Failed(CompilationStepError::CompilationFailed {
                        message: "encountered malformed program: expected reported producer failure".to_string(),
                    }),
                },
            ],
        );

        context.clear_events();
        assert_eq!(context.recent_events(), Vec::new());
        assert_eq!(*reported.lock().unwrap(), retained);
    }

    #[test]
    fn test_compilation_context_clear_cache() {
        let context = CompilationContext::<TestDomain>::new();
        let domain = TestDomain::default();
        compile_with(&context, &domain, 1, move || Ok(TestCompiledProgram(1))).unwrap();
        assert_eq!(context.program_count(), 1);

        context.clear_cache();
        assert_eq!(context.program_count(), 0);
        assert_eq!(context.statistics().compilations, 1);

        context.clear_statistics();
        assert_eq!(context.statistics(), CompilationCacheStatistics::default());
    }

    #[test]
    fn test_compilation_context_compile() {
        let context = CompilationContext::<TestDomain>::new();
        let domain = TestDomain::default();
        let staged: StagedFunction<TestDomain, ArrayType, ArrayType> = domain
            .stage(|input: CompilationTracer<TestDomain>| input, ArrayType::scalar(DataType::F32), ())
            .unwrap();
        let lowered = staged.lower(&domain).unwrap();

        // Only the first request compiles, and the second request is served by the in-memory tier.
        let first = context.compile(&domain, lowered.lowered_program()).unwrap();
        let second = context.compile(&domain, lowered.lowered_program()).unwrap();
        assert_eq!(first.as_ref(), &TestCompiledProgram(1));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(context.statistics().compilations, 1);
    }

    #[test]
    fn test_compilation_context_compile_rejects_incompatible_memory_hits() {
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(8);
        let domain = TestDomain::default();
        compile_with(&context, &domain, 0, move || Ok(TestCompiledProgram(2))).unwrap();
        context.clear_events();

        // Equal exact keys must mean interchangeable programs, so a rejected in-memory program fails the compilation.
        assert_eq!(
            context.compile(&domain, &TestLoweredProgram(vec![ArrayType::scalar(DataType::F32)])),
            Err(ProgramError::InvalidArgument { message: "cached program 2 cannot serve 1 outputs".to_string() }),
        );
        assert_eq!(
            context
                .compile(&domain, &TestLoweredProgram(vec![ArrayType::scalar(DataType::F32); 2]))
                .unwrap()
                .as_ref(),
            &TestCompiledProgram(2),
        );
        assert_eq!(context.statistics().memory_hits, 2);
        assert_eq!(context.statistics().compilations, 1);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Failed(CompilationStepError::Incompatible {
                        message: "cached program 2 cannot serve 1 outputs".to_string()
                    }),
                },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Hit,
                },
            ],
        );
    }

    #[test]
    fn test_compilation_context_compile_rejects_incompatible_shared_programs() {
        let context = Arc::new(CompilationContext::<TestDomain>::new().with_event_capacity(8));
        let release = Arc::new(AtomicBool::new(false));
        let producer = {
            let context = Arc::clone(&context);
            let release = Arc::clone(&release);
            thread::spawn(move || {
                compile_with(&context, &TestDomain::default(), 0, {
                    let release = Arc::clone(&release);
                    move || {
                        while !release.load(Ordering::Acquire) {
                            thread::yield_now();
                        }
                        Ok(TestCompiledProgram(2))
                    }
                })
                .unwrap()
            })
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while context.statistics().misses == 0 && Instant::now() < deadline {
            thread::yield_now();
        }
        let waiter = {
            let context = Arc::clone(&context);
            thread::spawn(move || {
                context.compile(&TestDomain::default(), &TestLoweredProgram(vec![ArrayType::scalar(DataType::F32)]))
            })
        };
        while context.statistics().waits == 0 && Instant::now() < deadline {
            thread::yield_now();
        }
        release.store(true, Ordering::Release);

        // The producer compiled the shared program for its own request, so the waiter validates it like a hit.
        assert_eq!(producer.join().unwrap().as_ref(), &TestCompiledProgram(2));
        assert_eq!(
            waiter.join().unwrap(),
            Err(ProgramError::InvalidArgument { message: "cached program 2 cannot serve 1 outputs".to_string() }),
        );
        assert_eq!(context.statistics().waits, 1);
        assert_eq!(
            event_steps(&context).last(),
            Some(&CompilationStep::MemoryWait {
                outcome: CompilationLookupOutcome::Failed(CompilationStepError::Incompatible {
                    message: "cached program 2 cannot serve 1 outputs".to_string()
                }),
            }),
        );
    }

    #[test]
    fn test_compilation_context_compile_recompiles_incompatible_persistent_artifacts() {
        let directory = tempfile::tempdir().unwrap();
        let domain = TestDomain::persistent();
        let disk_cache = || DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        compile_with(&CompilationContext::<TestDomain>::new().with_disk_cache(disk_cache()), &domain, 0, move || {
            Ok(TestCompiledProgram(2))
        })
        .unwrap();

        // A rejected persistent artifact is an incompatible miss, and the fresh compilation replaces it.
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(8).with_disk_cache(disk_cache());
        let lowered = TestLoweredProgram(vec![ArrayType::scalar(DataType::F32)]);
        assert_eq!(context.compile(&domain, &lowered).unwrap().as_ref(), &TestCompiledProgram(1));
        assert_eq!(context.statistics().persistent_hits, 0);
        assert_eq!(context.statistics().persistent_errors, 0);
        assert_eq!(context.statistics().compilations, 1);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Persistent,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::Incompatible),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::PersistentStore { outcome: CompilationStepOutcome::Succeeded },
            ],
        );

        let context = CompilationContext::<TestDomain>::new().with_disk_cache(disk_cache());
        assert_eq!(context.compile(&domain, &lowered).unwrap().as_ref(), &TestCompiledProgram(1));
        assert_eq!(context.statistics().persistent_hits, 1);
        assert_eq!(context.statistics().compilations, 0);
    }

    #[test]
    fn test_compilation_context_compile_falls_back_from_incompatible_exchange_artifacts() {
        let follower_context = |policy| {
            let state = Arc::new(TestExchangeState::default());
            state.artifacts.lock().unwrap().insert(vec![0], Ok(2usize.to_le_bytes().to_vec()));
            CompilationContext::<TestDomain>::new()
                .with_event_capacity(8)
                .with_artifact_exchange(Arc::new(TestExchange::new(1, state)), policy)
        };
        let lowered = TestLoweredProgram(vec![ArrayType::scalar(DataType::F32)]);

        // A rejected received artifact is an incompatible miss, so a follower that may fall back compiles locally.
        let context = follower_context(CompilationArtifactExchangePolicy::PreferSharing {
            timeout: Duration::from_secs(1),
            fallback_to_local_compile: true,
        });
        assert_eq!(context.compile(&TestDomain::persistent(), &lowered).unwrap().as_ref(), &TestCompiledProgram(1));
        assert_eq!(context.statistics().exchange_hits, 0);
        assert_eq!(context.statistics().exchange_errors, 0);
        assert_eq!(context.statistics().exchange_fallbacks, 1);
        assert_eq!(context.statistics().compilations, 1);
        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::ExchangeCompatibilityValidation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::ExchangeReceive {
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::Incompatible),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
            ],
        );

        // A follower that requires sharing fails instead.
        let context =
            follower_context(CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(1) });
        assert_eq!(
            context.compile(&TestDomain::persistent(), &lowered),
            Err(ProgramError::InvalidArgument {
                message: "compilation artifact exchange requirement was not satisfied: received compilation artifact \
                    is incompatible: cached program 2 cannot serve 1 outputs"
                    .to_string(),
            }),
        );
        assert_eq!(context.statistics().compilations, 0);
    }

    #[test]
    fn test_compilation_context_compile_publishes_fresh_artifacts_after_incompatible_persistent_artifacts() {
        let directory = tempfile::tempdir().unwrap();
        let domain = TestDomain::persistent();
        let disk_cache = || DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        compile_with(&CompilationContext::<TestDomain>::new().with_disk_cache(disk_cache()), &domain, 0, move || {
            Ok(TestCompiledProgram(2))
        })
        .unwrap();

        // A leader never publishes a persistent artifact that the domain rejects, so followers receive the fresh one.
        let state = Arc::new(TestExchangeState::default());
        let context = CompilationContext::<TestDomain>::new().with_disk_cache(disk_cache()).with_artifact_exchange(
            Arc::new(TestExchange::new(0, Arc::clone(&state))),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(1) },
        );
        assert_eq!(
            context
                .compile(&domain, &TestLoweredProgram(vec![ArrayType::scalar(DataType::F32)]))
                .unwrap()
                .as_ref(),
            &TestCompiledProgram(1),
        );
        assert_eq!(context.statistics().exchange_publishes, 1);
        assert_eq!(state.artifacts.lock().unwrap().get(&vec![0]), Some(&Ok(1usize.to_le_bytes().to_vec())));
    }

    #[test]
    fn test_compilation_context_compile_reuses_memory_entries() {
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(16);
        let domain = TestDomain::default();
        let producer_calls = Arc::new(AtomicUsize::new(0));

        let first = compile_with(&context, &domain, 1, {
            let producer_calls = Arc::clone(&producer_calls);
            move || {
                producer_calls.fetch_add(1, Ordering::Relaxed);
                Ok(TestCompiledProgram(7))
            }
        })
        .unwrap();
        let second = compile_with(&context, &domain, 1, {
            let producer_calls = Arc::clone(&producer_calls);
            move || {
                producer_calls.fetch_add(1, Ordering::Relaxed);
                Ok(TestCompiledProgram(8))
            }
        })
        .unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.0, 7);
        assert_eq!(producer_calls.load(Ordering::Relaxed), 1);
        let statistics = context.statistics();
        assert_eq!(statistics.memory_hits, 1);
        assert_eq!(statistics.misses, 1);
        assert_eq!(statistics.compilations, 1);

        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Hit,
                },
            ],
        );
    }

    #[test]
    fn test_compilation_context_compile_retries_after_miss_reporter_panic() {
        let panic_on_miss = AtomicBool::new(true);
        let context =
            CompilationContext::<TestDomain>::new().with_event_reporter(Arc::new(|event: &CompilationEvent| {
                if matches!(
                    &event.step,
                    CompilationStep::CacheLookup {
                        tier: CompilationCacheTier::Memory,
                        outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                    },
                ) && panic_on_miss.swap(false, Ordering::Relaxed)
                {
                    panic!("expected memory miss reporter panic");
                }
            }));
        let domain = TestDomain::default();
        let lowered = TestLoweredProgram(vec![ArrayType::scalar(DataType::F32)]);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| context.compile(&domain, &lowered)));
        assert!(panic.is_err());

        // Check cleanup before retrying so a leaked marker fails directly instead of making the test wait forever.
        assert!(context.in_flight.lock().unwrap().is_empty());
        assert_eq!(context.program_count(), 0);
        assert_eq!(context.compile(&domain, &lowered).unwrap().as_ref(), &TestCompiledProgram(1));
        assert_eq!(context.statistics().compilations, 1);
    }

    #[test]
    fn test_compilation_context_compile_retries_after_backend_error() {
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(16);
        let domain = TestDomain::default();

        // Producer errors are returned to their caller and never cached.
        let error = ProgramError::MalformedProgram("expected test failure".to_string());
        assert_eq!(
            compile_with(&context, &domain, 1, {
                let error = error.clone();
                move || Err(error.clone())
            })
            .err(),
            Some(error)
        );
        let program = compile_with(&context, &domain, 1, move || Ok(TestCompiledProgram(9))).unwrap();

        assert_eq!(program.0, 9);
        assert_eq!(context.statistics().compilations, 2);

        assert_eq!(
            event_steps(&context),
            vec![
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::BackendCompilation {
                    outcome: CompilationStepOutcome::Failed(CompilationStepError::CompilationFailed {
                        message: "encountered malformed program: expected test failure".to_string()
                    }),
                },
                CompilationStep::CacheLookup {
                    tier: CompilationCacheTier::Memory,
                    outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
                },
                CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
            ],
        );
    }

    #[test]
    fn test_compilation_context_compile_retries_after_backend_panic() {
        let context = CompilationContext::<TestDomain>::new();
        let domain = TestDomain::default();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compile_with(&context, &domain, 1, move || -> Result<TestCompiledProgram, ProgramError> {
                panic!("expected producer panic")
            })
            .unwrap()
        }));
        assert!(panic.is_err());

        // The panicking producer released its in-flight state, so the next lookup becomes the new producer.
        let program = compile_with(&context, &domain, 1, move || Ok(TestCompiledProgram(13))).unwrap();
        assert_eq!(program.0, 13);
    }

    #[test]
    fn test_compilation_context_compile_deduplicates_same_key_misses() {
        const THREAD_COUNT: usize = 8;

        let context = Arc::new(CompilationContext::<TestDomain>::new().with_event_capacity(16));
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
                compile_with(&context, &TestDomain::default(), 1, {
                    let producer_calls = Arc::clone(&producer_calls);
                    let release = Arc::clone(&release);
                    move || {
                        producer_calls.fetch_add(1, Ordering::Relaxed);
                        while !release.load(Ordering::Acquire) {
                            thread::yield_now();
                        }
                        Ok(TestCompiledProgram(11))
                    }
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

        let mut expected = vec![
            CompilationStep::CacheLookup {
                tier: CompilationCacheTier::Memory,
                outcome: CompilationLookupOutcome::Miss(CompilationMissReason::NotFound),
            },
            CompilationStep::BackendCompilation { outcome: CompilationStepOutcome::Succeeded },
        ];
        expected.extend(std::iter::repeat_n(
            CompilationStep::MemoryWait { outcome: CompilationLookupOutcome::Hit },
            THREAD_COUNT - 1,
        ));
        assert_eq!(event_steps(&context), expected);
        let lookup_duration = context
            .recent_events()
            .iter()
            .filter(|event| !matches!(event.step, CompilationStep::BackendCompilation { .. }))
            .map(|event| event.duration)
            .sum::<Duration>();
        assert_eq!(context.statistics().memory_lookup_duration, lookup_duration);
    }

    #[test]
    fn test_compilation_context_compile_unrelated_keys_concurrently() {
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
                compile_with(&context, &TestDomain::default(), key, {
                    let release = Arc::clone(&release);
                    move || {
                        entered_sender.send(key).unwrap();
                        while !release.load(Ordering::Acquire) {
                            thread::yield_now();
                        }
                        Ok(TestCompiledProgram(key as usize))
                    }
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

    #[test]
    fn test_disk_cache_open() {
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("nested").join("cache");
        let cache = DiskCache::open(&nested).unwrap();
        assert_eq!(cache.directory(), nested.as_path());
        assert_eq!(cache.max_bytes(), None);
        assert_eq!(cache.cache_size_bytes().unwrap(), 0);

        // A path that already exists as a file cannot back a cache.
        let file = directory.path().join("file");
        fs::write(&file, b"not a directory").unwrap();
        assert!(DiskCache::open(&file).is_err());
    }

    #[test]
    fn test_disk_cache_with_max_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::with_max_bytes(directory.path(), 64).unwrap();
        assert_eq!(cache.max_bytes(), Some(64));

        // The most recently written entry is preserved even when it alone exceeds the cap.
        let digest = DiskCacheDigest::from_key(b"oversized");
        let payload = vec![0u8; 16 * 1024];
        cache.put(&digest, &payload).unwrap();
        assert_eq!(cache.get(&digest).unwrap(), Some(payload));
    }

    #[test]
    fn test_disk_cache_with_write_thresholds() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::from_secs(2), 128);

        assert!(!cache.should_serialize(Duration::from_secs(1)));
        assert!(cache.should_serialize(Duration::from_secs(2)));
        assert!(!cache.should_persist(Duration::from_secs(1), 256));
        assert!(!cache.should_persist(Duration::from_secs(3), 64));
        assert!(cache.should_persist(Duration::from_secs(2), 128));
    }

    #[test]
    fn test_disk_cache_get_auxiliary() {
        let directory = tempfile::tempdir().unwrap();
        DiskCache::open(directory.path())
            .unwrap()
            .put_auxiliary("adaptive-profile", b"baseline-and-policy", b"aggregated-profile")
            .unwrap();

        let reopened = DiskCache::open(directory.path()).unwrap();
        assert_eq!(
            reopened.get_auxiliary("adaptive-profile", b"baseline-and-policy").unwrap(),
            Some(b"aggregated-profile".to_vec()),
        );
        assert_eq!(reopened.get_auxiliary("different-namespace", b"baseline-and-policy").unwrap(), None);
    }

    #[test]
    fn test_disk_cache_get_auxiliary_rejects_corruption() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap();
        let key = b"baseline-and-policy";
        cache.put_auxiliary("adaptive-profile", key, b"aggregated-profile").unwrap();
        let path = cache.auxiliary_entry_path(&DiskCache::auxiliary_digest("adaptive-profile", key));
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0xff;
        fs::write(path, bytes).unwrap();

        // Corrupting the gzip trailer makes decompression fail before the envelope is validated.
        let error = cache.get_auxiliary("adaptive-profile", key).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_disk_cache_put_auxiliary() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap();
        let error = cache.put_auxiliary("adaptive-profile", b"key", &vec![0u8; MAXIMUM_AUXILIARY_ENTRY_SIZE + 1]);
        let error = error.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(error.to_string(), "persistent auxiliary payload exceeds the 64 MiB limit");
        assert_eq!(cache.cache_size_bytes().unwrap(), 0);
    }

    #[test]
    fn test_disk_cache_put_auxiliary_evicts_to_capacity() {
        let directory = tempfile::tempdir().unwrap();
        let uncapped = DiskCache::open(directory.path()).unwrap();
        uncapped.put_auxiliary("adaptive-profile", b"first", &[1; 256]).unwrap();
        let first_path = uncapped.auxiliary_entry_path(&DiskCache::auxiliary_digest("adaptive-profile", b"first"));
        let first_size = fs::metadata(&first_path).unwrap().len();

        let cache = DiskCache::with_max_bytes(directory.path(), first_size).unwrap();
        cache.put_auxiliary("adaptive-profile", b"second", &[2; 256]).unwrap();

        assert_eq!(cache.get_auxiliary("adaptive-profile", b"first").unwrap(), None);
        assert_eq!(cache.get_auxiliary("adaptive-profile", b"second").unwrap(), Some(vec![2; 256]));
        let second_path = cache.auxiliary_entry_path(&DiskCache::auxiliary_digest("adaptive-profile", b"second"));
        assert_eq!(cache.cache_size_bytes().unwrap(), fs::metadata(second_path).unwrap().len());
    }

    #[test]
    fn test_disk_cache_get() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap();
        let digest = DiskCacheDigest::from_key(b"stable-key");
        let data = b"hello compiled bytes".to_vec();

        assert_eq!(cache.get(&digest).unwrap(), None);
        cache.put(&digest, &data).unwrap();
        assert_eq!(cache.get(&digest).unwrap(), Some(data));
        assert_eq!(cache.get(&DiskCacheDigest::from_key(b"missing")).unwrap(), None);
    }

    #[test]
    fn test_disk_cache_put_compresses_entries() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap();
        let digest = DiskCacheDigest::from_key(b"compressible");
        let payload = vec![0u8; 64 * 1024];
        cache.put(&digest, &payload).unwrap();

        let size_on_disk = fs::metadata(cache.entry_path(&digest)).unwrap().len();
        assert!(
            size_on_disk < payload.len() as u64,
            "expected gzip to compress 64 KiB of zeros below the raw size, got {size_on_disk} bytes",
        );
        assert_eq!(cache.cache_size_bytes().unwrap(), size_on_disk);
        assert_eq!(cache.get(&digest).unwrap(), Some(payload));
    }

    #[test]
    fn test_disk_cache_put_independent_instances() {
        let directory = tempfile::tempdir().unwrap();
        let first = DiskCache::open(directory.path()).unwrap();
        let second = DiskCache::open(directory.path()).unwrap();
        let digest = DiskCacheDigest::from_key(b"same-key");
        let barrier = Barrier::new(2);
        let first_payload = vec![17u8; 32 * 1024];
        let second_payload = vec![93u8; 48 * 1024];
        std::thread::scope(|scope| {
            let left = scope.spawn(|| {
                barrier.wait();
                first.put(&digest, &first_payload).unwrap();
                first.put_auxiliary("tuning", b"same-key", &first_payload).unwrap();
            });
            let right = scope.spawn(|| {
                barrier.wait();
                second.put(&digest, &second_payload).unwrap();
                second.put_auxiliary("tuning", b"same-key", &second_payload).unwrap();
            });
            left.join().unwrap();
            right.join().unwrap();
        });
        let executable = first.get(&digest).unwrap().unwrap();
        let auxiliary = second.get_auxiliary("tuning", b"same-key").unwrap().unwrap();
        assert!(executable == first_payload || executable == second_payload);
        assert!(auxiliary == first_payload || auxiliary == second_payload);

        // Atomic publication leaves only the two complete envelope files, with no temporary write debris.
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
    }

    #[test]
    fn test_disk_cache_put_evicts_to_capacity() {
        let directory = tempfile::tempdir().unwrap();
        let uncapped = DiskCache::open(directory.path()).unwrap();
        let payloads: Vec<Vec<u8>> =
            (0..3).map(|entry| (0..256).map(|byte| (entry * 31 + byte) as u8).collect()).collect();
        let digests: Vec<DiskCacheDigest> = (0..3).map(|index| DiskCacheDigest::from_key(&[index])).collect();
        for index in 0..3 {
            uncapped.put(&digests[index], &payloads[index]).unwrap();
            let file = File::options().write(true).open(uncapped.entry_path(&digests[index])).unwrap();
            let modified = SystemTime::UNIX_EPOCH + Duration::from_secs(index as u64 + 1);
            file.set_times(FileTimes::new().set_modified(modified)).unwrap();
        }

        let sizes = digests
            .iter()
            .map(|digest| fs::metadata(uncapped.entry_path(digest)).unwrap().len())
            .collect::<Vec<_>>();
        let capacity = sizes[1] + sizes[2];
        let cache = DiskCache::with_max_bytes(directory.path(), capacity).unwrap();

        // Rewriting the newest entry triggers eviction through the production write path and protects that entry.
        cache.put(&digests[2], &payloads[2]).unwrap();
        assert_eq!(cache.get(&digests[0]).unwrap(), None);
        assert_eq!(cache.get(&digests[1]).unwrap(), Some(payloads[1].clone()));
        assert_eq!(cache.get(&digests[2]).unwrap(), Some(payloads[2].clone()));
        assert!(cache.cache_size_bytes().unwrap() <= capacity);
    }

    #[test]
    fn test_disk_cache_read_entry() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("entry");
        let digest = DiskCacheDigest::from_key(b"key");
        let envelope = encode_envelope(&digest, b"compiled");
        assert_eq!(envelope.len(), ENTRY_HEADER_LENGTH + b"compiled".len());

        write_compressed_envelope(&path, &envelope);
        assert_eq!(DiskCache::read_entry(&path, &digest, None).unwrap(), Some(b"compiled".to_vec()));
    }

    #[test]
    fn test_disk_cache_read_entry_rejects_malformed_envelopes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("entry");
        let digest = DiskCacheDigest::from_key(b"key");
        let envelope = encode_envelope(&digest, b"compiled");

        write_compressed_envelope(&path, &envelope[..ENTRY_HEADER_LENGTH - 1]);
        let error = DiskCache::read_entry(&path, &digest, None).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache entry is shorter than its header");

        let mut invalid_magic = envelope.clone();
        invalid_magic[0] ^= 0xff;
        write_compressed_envelope(&path, &invalid_magic);
        let error = DiskCache::read_entry(&path, &digest, None).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache entry has an invalid magic value");

        let mut unsupported_version = envelope.clone();
        unsupported_version[ENTRY_MAGIC.len()..ENTRY_MAGIC.len() + 4].copy_from_slice(&2u32.to_le_bytes());
        write_compressed_envelope(&path, &unsupported_version);
        let error = DiskCache::read_entry(&path, &digest, None).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache entry uses unsupported format version 2");

        let other_key = encode_envelope(&DiskCacheDigest::from_key(b"other"), b"compiled");
        write_compressed_envelope(&path, &other_key);
        let error = DiskCache::read_entry(&path, &digest, None).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache entry key digest does not match its requested key");

        write_compressed_envelope(&path, &envelope[..envelope.len() - 1]);
        let error = DiskCache::read_entry(&path, &digest, None).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache payload declares 8 bytes but contains 7");

        let mut corrupted_payload = envelope.clone();
        *corrupted_payload.last_mut().unwrap() ^= 0xff;
        write_compressed_envelope(&path, &corrupted_payload);
        let error = DiskCache::read_entry(&path, &digest, None).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache payload checksum does not match");
    }

    #[test]
    fn test_disk_cache_temporary_entry_independent_instances() {
        let directory = tempfile::tempdir().unwrap();
        let first = DiskCache::open(directory.path()).unwrap();
        let second = DiskCache::open(directory.path()).unwrap();
        let digest = DiskCacheDigest::from_key(b"same-key");
        let barrier = Barrier::new(2);

        // Both caches begin with counter zero. Holding both files open together prevents either writer from reusing a
        // name after the other publishes or cleans up, which deterministically exercises independent instances.
        let paths = std::thread::scope(|scope| {
            let left = scope.spawn(|| {
                let (path, mut file) = first.temporary_entry(&digest, AUXILIARY_EXTENSION).unwrap();
                barrier.wait();
                file.write_all(b"first writer").unwrap();
                path
            });
            let right = scope.spawn(|| {
                let (path, mut file) = second.temporary_entry(&digest, AUXILIARY_EXTENSION).unwrap();
                barrier.wait();
                file.write_all(b"second writer").unwrap();
                path
            });
            (left.join().unwrap(), right.join().unwrap())
        });
        assert_ne!(paths.0, paths.1);
        assert_eq!(fs::read(&paths.0).unwrap(), b"first writer");
        assert_eq!(fs::read(&paths.1).unwrap(), b"second writer");

        // A reopened instance must also skip abandoned files instead of truncating their contents.
        let reopened = DiskCache::open(directory.path()).unwrap();
        let (path, file) = reopened.temporary_entry(&digest, AUXILIARY_EXTENSION).unwrap();
        assert_ne!(path, paths.0);
        assert_ne!(path, paths.1);
        drop(file);
        assert_eq!(fs::read(&paths.0).unwrap(), b"first writer");
        assert_eq!(fs::read(&paths.1).unwrap(), b"second writer");
    }

    #[test]
    fn test_disk_cache_digest_from_key() {
        let digest = DiskCacheDigest::from_key(b"abc");
        assert_eq!(digest.to_hex(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(digest, DiskCacheDigest::from_key(b"abc"));
        assert_ne!(digest, DiskCacheDigest::from_key(b"abd"));
    }
}
