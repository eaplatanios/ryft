//! Backend-neutral, process-local caching primitive for retained _specializations_.
//!
//! A specialization is the artifact a front end produces for one exact configuration of a retained callable: a traced
//! [`Program`](crate::Program), a lowered program, a compiled function, or a transformed program.
//! [`SpecializationCache`] retains those artifacts under a **single level** contract, mapping one key to one artifact.
//! It deliberately has no tiers, no fallback lifecycles, and no timing policy. Consumers keep their own phase-specific
//! statistics and decide what a key and an artifact means.
//!
//! # Reuse Contract
//!
//! Reuse is defined by key _equality_, never by a hash alone, so `Key: Clone + Eq + Hash` and the retained map is a
//! bounded Least Recently Used (LRU) cache. Artifacts must be `Clone`, which in practice means large structural
//! artifacts should be held behind cheap reference-counted handles rather than deep-cloned on every hit.
//!
//! # Production and Reentrancy
//!
//! Production runs outside every lock. [`SpecializationCache::lookup`] takes the locks only long enough to consult
//! the retained map and to register an in-flight marker, then hands back a [`SpecializationCacheProducer`] Resource
//! Acquisition Is Initialization (RAII) guard; the caller runs arbitrary code (e.g., tracing, lowering, compilation)
//! while the cache is fully available to other threads. [`SpecializationCacheProducer::insert`] publishes the artifact,
//! and dropping the producer without inserting (because production failed or panicked) clears the marker and caches
//! nothing, so the next call retries from scratch.
//!
//! In-flight markers are tracked per `(ThreadId, Key)` pair, which gives exactly three behaviors:
//!
//! - **Same Thread, Same Key:** Rejected immediately with [`ReentrantSpecializationError`]. Recursive production
//!   of the specialization currently being produced cannot terminate, so it is an error rather than a wait.
//! - **Same Thread, Different Key:** Proceeds. Nested production of a _different_ specialization is legitimate.
//! - **Different Thread, Same Key:** Proceeds. Both threads produce, and both inserts are idempotent with the last
//!   one winning.
//!
//! Cross-thread duplicate production is deliberate. Front-end work (i.e., tracing and lowering) is cheap relative
//! to backend compilation, duplicate results are interchangeable by construction because the key defines
//! interchangeability, and the expensive backend compile is already single-flighted by
//! [`CompilationContext`](crate::CompilationContext). This primitive therefore **never blocks and never waits**,
//! and it must not be "improved" into blocking single-flight: waiting coordination belongs exclusively to
//! [`CompilationContext`](crate::CompilationContext), which additionally owns persistence, distributed
//! artifact exchange, and an event model that this primitive intentionally lacks.
//!
//! # Thread Safety
//!
//! Thread safety is conditional and structural. The retained map and the in-flight set are behind [`Mutex`], and the
//! statistics are [`AtomicU64`] counters, so [`SpecializationCache`] is `Send` and `Sync` exactly when `Key` and `A`
//! are, with no `unsafe impl` anywhere. Thread-confined consumers whose artifacts are not `Send` still work
//! single-threaded; they simply do not get `Send`.

// TODO(eaplatanios): Review from here onwards.

use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::ThreadId;

use lru::LruCache;
use thiserror::Error;

/// Error returned when a thread requests production of a specialization key that the *same* thread is already
/// producing.
///
/// Recursive production of the specialization currently in flight cannot terminate, so [`SpecializationCache`]
/// rejects it immediately instead of waiting. Producing a different key from within an active producer is allowed,
/// and so is producing the same key concurrently from a different thread.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Error)]
#[error("recursive request for a specialization that is already being produced on this thread")]
pub struct ReentrantSpecializationError;

/// Error returned by the convenience [`SpecializationCache::get_or_try_insert_with`] entry point, layering a
/// caller-defined production error `E` over the cache's own [`ReentrantSpecializationError`] rejection.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum SpecializationCacheError<E: Debug + Display> {
    /// The same thread is already producing this key.
    #[error(transparent)]
    Reentrant(#[from] ReentrantSpecializationError),

    /// The caller's production closure failed. Nothing was cached and the next call retries.
    #[error("{0}")]
    Production(E),
}

/// Cache key identifying one specialization of a retained function.
///
/// Two calls are interchangeable — that is, they stage the same program — exactly when all three components agree:
///
///   - `static_parameters`: the host values the traced closure may branch on, read, or embed as literals. Unequal
///     static parameters can stage arbitrarily different programs, so they must separate specializations. Static
///     parameters must be `Clone + Debug + Eq + Hash`; runtime arrays and other backend values should remain dynamic
///     inputs rather than becoming static parameters merely because they provide identity equality.
///   - `input_structure`: the parameter-structure shape of the dynamic input, that is, its
///     [`Parameterized::ParameterStructure`](crate::parameters::Parameterized::ParameterStructure). Tracing rebuilds
///     the closure's argument from this structure, so a closure may legitimately branch on container arity. Keying on
///     the structure rather than on the flattened leaves also distinguishes inputs that differ only in *empty*
///     substructure, which flat leaf paths and flat leaf types cannot see.
///   - `dispatch`: the domain-normalized abstract signature of the flattened dynamic input, such as element data
///     types and shapes. Programs are staged against abstract types, so unequal signatures stage unequal programs.
///
/// Everything else that affects staging — the closure itself, its captures, the domain, and fixed options — is
/// implicit in the cache's owner, because a cache is scoped to exactly one retained callable. That is why this key
/// carries no fragile function-pointer identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionSpecializationKey<Structure, Dispatch, Static> {
    /// Host values declared static for this specialization.
    static_parameters: Static,

    /// Parameter structure of the dynamic input.
    input_structure: Structure,

    /// Domain-normalized abstract dispatch signature of the flattened dynamic input.
    dispatch: Dispatch,
}

impl<Structure, Dispatch, Static> FunctionSpecializationKey<Structure, Dispatch, Static> {
    /// Creates a specialization key from its three components.
    ///
    /// # Parameters
    ///   - `static_parameters`: Host values declared static for this specialization.
    ///   - `input_structure`: Parameter structure of the dynamic input.
    ///   - `dispatch`: Domain-normalized abstract dispatch signature of the flattened dynamic input.
    #[inline]
    pub fn new(static_parameters: Static, input_structure: Structure, dispatch: Dispatch) -> Self {
        Self { static_parameters, input_structure, dispatch }
    }

    /// Returns the host values declared static for this specialization.
    #[inline]
    pub fn static_parameters(&self) -> &Static {
        &self.static_parameters
    }

    /// Returns the parameter structure of the dynamic input.
    #[inline]
    pub fn input_structure(&self) -> &Structure {
        &self.input_structure
    }

    /// Returns the domain-normalized abstract dispatch signature of the flattened dynamic input.
    #[inline]
    pub fn dispatch(&self) -> &Dispatch {
        &self.dispatch
    }
}

/// Snapshot of one [`SpecializationCache`]'s activity since construction or since the last
/// [`SpecializationCache::clear_statistics`] call.
///
/// Every counter saturates rather than wrapping. Note that `misses` counts lookups that found no retained artifact,
/// including lookups subsequently rejected with [`ReentrantSpecializationError`], and that `productions + failures` need
/// not equal `misses` while producers are still in flight.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SpecializationCacheStatistics {
    /// Lookups served by a retained artifact.
    pub hits: u64,

    /// Lookups that found no retained artifact.
    pub misses: u64,

    /// Artifacts published through [`SpecializationCacheProducer::insert`].
    pub productions: u64,

    /// Producers dropped without inserting, because production failed or panicked.
    pub failures: u64,

    /// Retained artifacts dropped by the bounded LRU to make room for a new one. Entries removed by
    /// [`SpecializationCache::clear`] or [`SpecializationCache::invalidate_where`] are not evictions.
    pub evictions: u64,
}

/// Atomic counters backing [`SpecializationCacheStatistics`].
///
/// All updates use [`Ordering::Relaxed`]: the counters are diagnostic, they order nothing, and the retained map's
/// [`Mutex`] already provides the synchronization that correctness depends on.
#[derive(Debug, Default)]
struct SpecializationCacheStatisticsState {
    /// Lookups served by a retained artifact.
    hits: AtomicU64,

    /// Lookups that found no retained artifact.
    misses: AtomicU64,

    /// Artifacts published through a producer.
    productions: AtomicU64,

    /// Producers dropped without inserting.
    failures: AtomicU64,

    /// Retained artifacts dropped by the bounded LRU.
    evictions: AtomicU64,
}

impl SpecializationCacheStatisticsState {
    /// Adds one to `counter`, saturating instead of wrapping.
    fn increment(counter: &AtomicU64) {
        // The update closure always returns `Some`, so this update can never be rejected.
        counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| Some(count.saturating_add(1)))
            .unwrap();
    }

    /// Returns a consistent-enough snapshot of the counters. Counters are read independently, so a snapshot taken
    /// while other threads are active reflects a plausible interleaving rather than a single instant.
    fn snapshot(&self) -> SpecializationCacheStatistics {
        SpecializationCacheStatistics {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            productions: self.productions.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    /// Resets every counter to zero.
    fn clear(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.productions.store(0, Ordering::Relaxed);
        self.failures.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
    }
}

/// Bounded, process-local cache mapping one specialization key to one retained artifact.
///
/// Refer to the [module documentation](self) for the reuse, production, reentrancy, and thread-safety contracts.
#[derive(Debug)]
pub struct SpecializationCache<K: Clone + Eq + Hash, A: Clone> {
    /// Retained artifacts in least-recently-used order.
    entries: Mutex<LruCache<K, A>>,

    /// Keys currently being produced, paired with the thread that is producing them.
    in_flight: Mutex<HashSet<(ThreadId, K)>>,

    /// Diagnostic counters.
    statistics: SpecializationCacheStatisticsState,
}

/// Outcome of a [`SpecializationCache::lookup`].
#[derive(Debug)]
pub enum SpecializationCacheLookup<'a, K: Clone + Eq + Hash, A: Clone> {
    /// A retained artifact was found and its recency was refreshed.
    Hit(A),

    /// No artifact was retained. The producer guard holds the in-flight marker until it inserts or is dropped.
    Miss(SpecializationCacheProducer<'a, K, A>),
}

/// RAII guard authorizing production of one specialization.
///
/// The guard holds the `(ThreadId, K)` in-flight marker for its key. Calling [`Self::insert`] publishes an artifact
/// and releases the marker; dropping the guard without inserting releases the marker, counts a failure, and caches
/// nothing. Because the guard borrows the cache rather than holding a lock, production runs with the cache fully
/// available to other threads.
#[derive(Debug)]
pub struct SpecializationCacheProducer<'a, K: Clone + Eq + Hash, A: Clone> {
    /// Cache this producer publishes to.
    cache: &'a SpecializationCache<K, A>,

    /// Thread that registered the in-flight marker, so the marker is released with the key it was registered under.
    thread: ThreadId,

    /// Key being produced. Taken by [`Self::insert`] so that [`Drop`] does not double-release the marker.
    key: Option<K>,
}

impl<K: Clone + Eq + Hash, A: Clone> SpecializationCache<K, A> {
    /// Creates an empty cache retaining at most `capacity` artifacts. A `capacity` of zero is clamped to one, because
    /// a cache that can retain nothing would turn every producer into wasted work.
    pub fn new(capacity: usize) -> Self {
        let capacity = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            entries: Mutex::new(LruCache::new(capacity)),
            in_flight: Mutex::new(HashSet::new()),
            statistics: SpecializationCacheStatisticsState::default(),
        }
    }

    /// Looks `key` up, returning either the retained artifact or a producer authorized to make one.
    ///
    /// A hit refreshes the key's recency, counts a hit, and returns a clone of the artifact. A miss counts a miss —
    /// including when the lookup is then rejected as reentrant — registers a `(current thread, key)` in-flight marker,
    /// and returns a [`SpecializationCacheProducer`]. No lock is held once this method returns.
    ///
    /// # Parameters
    ///   - `key`: Specialization to look up. It is consumed because a miss retains it as the producer's key.
    pub fn lookup(&self, key: K) -> Result<SpecializationCacheLookup<'_, K, A>, ReentrantSpecializationError> {
        if let Some(artifact) = self.entries.lock().expect("specialization cache mutex is poisoned").get(&key).cloned()
        {
            SpecializationCacheStatisticsState::increment(&self.statistics.hits);
            return Ok(SpecializationCacheLookup::Hit(artifact));
        }
        SpecializationCacheStatisticsState::increment(&self.statistics.misses);
        let thread = std::thread::current().id();
        let registered =
            self.in_flight.lock().expect("specialization cache mutex is poisoned").insert((thread, key.clone()));
        if !registered {
            return Err(ReentrantSpecializationError);
        }
        Ok(SpecializationCacheLookup::Miss(SpecializationCacheProducer { cache: self, thread, key: Some(key) }))
    }

    /// Returns the artifact retained for `key`, producing and retaining one with `produce` on a miss.
    ///
    /// This is the convenience form of [`Self::lookup`]. Callers that need to time the lookup separately from
    /// production, or that need the producer's key, should use [`Self::lookup`] directly. Production errors are
    /// propagated and nothing is cached, so a later call retries.
    ///
    /// # Parameters
    ///   - `key`: Specialization to look up.
    ///   - `produce`: Runs on a miss, outside every cache lock, to build the artifact.
    pub fn get_or_try_insert_with<E, F>(&self, key: K, produce: F) -> Result<A, SpecializationCacheError<E>>
    where
        E: Debug + Display,
        F: FnOnce() -> Result<A, E>,
    {
        match self.lookup(key)? {
            SpecializationCacheLookup::Hit(artifact) => Ok(artifact),
            SpecializationCacheLookup::Miss(producer) => match produce() {
                Ok(artifact) => Ok(producer.insert(artifact)),
                Err(error) => Err(SpecializationCacheError::Production(error)),
            },
        }
    }

    /// Returns the number of retained artifacts.
    pub fn len(&self) -> usize {
        self.entries.lock().expect("specialization cache mutex is poisoned").len()
    }

    /// Returns whether no artifact is retained.
    pub fn is_empty(&self) -> bool {
        self.entries.lock().expect("specialization cache mutex is poisoned").is_empty()
    }

    /// Returns the maximum number of artifacts this cache retains.
    pub fn capacity(&self) -> usize {
        self.entries.lock().expect("specialization cache mutex is poisoned").cap().get()
    }

    /// Returns the retained keys from most to least recently used.
    pub fn keys(&self) -> Vec<K> {
        self.entries
            .lock()
            .expect("specialization cache mutex is poisoned")
            .iter()
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Removes every retained artifact, leaving the statistics untouched.
    ///
    /// Clearing does not cancel producers that are already in flight. A producer that succeeds afterwards still
    /// inserts its artifact, because it was authorized before the clear and its key still identifies an
    /// interchangeable specialization.
    pub fn clear(&self) {
        self.entries.lock().expect("specialization cache mutex is poisoned").clear();
    }

    /// Removes every retained artifact whose key satisfies `predicate`, returning how many were removed.
    ///
    /// Like [`Self::clear`], this does not cancel producers that are already in flight.
    ///
    /// # Parameters
    ///   - `predicate`: Selects the keys to remove. It is called once per retained key, from most to least recently
    ///     used.
    pub fn invalidate_where(&self, mut predicate: impl FnMut(&K) -> bool) -> usize {
        let mut entries = self.entries.lock().expect("specialization cache mutex is poisoned");
        let removed = entries.iter().map(|(key, _)| key).filter(|key| predicate(key)).cloned().collect::<Vec<_>>();
        for key in &removed {
            entries.pop(key);
        }
        removed.len()
    }

    /// Returns a snapshot of this cache's activity counters.
    pub fn statistics(&self) -> SpecializationCacheStatistics {
        self.statistics.snapshot()
    }

    /// Resets this cache's activity counters, leaving the retained artifacts untouched.
    pub fn clear_statistics(&self) {
        self.statistics.clear();
    }
}

impl<K: Clone + Eq + Hash, A: Clone> SpecializationCacheProducer<'_, K, A> {
    /// Returns the key this producer is authorized to produce.
    #[inline]
    pub fn key(&self) -> &K {
        self.key.as_ref().unwrap()
    }

    /// Publishes `artifact` for this producer's key, releases the in-flight marker, and returns the artifact.
    ///
    /// Inserting is idempotent: if another thread produced the same key concurrently, the later insert replaces the
    /// earlier one. Both artifacts are interchangeable by the cache's reuse contract, so last-one-wins is safe.
    pub fn insert(mut self, artifact: A) -> A {
        let key = self.key.take().unwrap();
        {
            let mut entries = self.cache.entries.lock().expect("specialization cache mutex is poisoned");
            if let Some((replaced_key, _)) = entries.push(key.clone(), artifact.clone())
                && replaced_key != key
            {
                SpecializationCacheStatisticsState::increment(&self.cache.statistics.evictions);
            }
        }
        SpecializationCacheStatisticsState::increment(&self.cache.statistics.productions);
        self.cache
            .in_flight
            .lock()
            .expect("specialization cache mutex is poisoned")
            .remove(&(self.thread, key));
        artifact
    }
}

impl<K: Clone + Eq + Hash, A: Clone> Drop for SpecializationCacheProducer<'_, K, A> {
    fn drop(&mut self) {
        // `insert` takes the key, so this only runs when production failed or unwound. Releasing the marker here is
        // what makes failed and panicking production retryable instead of permanently reentrant.
        if let Some(key) = self.key.take() {
            self.cache
                .in_flight
                .lock()
                .expect("specialization cache mutex is poisoned")
                .remove(&(self.thread, key));
            SpecializationCacheStatisticsState::increment(&self.cache.statistics.failures);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::hash::Hasher;
    use std::panic::AssertUnwindSafe;
    use std::sync::Barrier;
    use std::thread;

    use pretty_assertions::assert_eq;

    use super::*;

    /// Extracts the producer from a lookup that is expected to miss.
    fn expect_producer<K: Clone + Eq + Hash, A: Clone>(
        lookup: Result<SpecializationCacheLookup<'_, K, A>, ReentrantSpecializationError>,
    ) -> SpecializationCacheProducer<'_, K, A> {
        match lookup {
            Ok(SpecializationCacheLookup::Miss(producer)) => producer,
            Ok(SpecializationCacheLookup::Hit(_)) => panic!("expected a miss but the lookup hit"),
            Err(error) => panic!("expected a miss but the lookup failed: {error}"),
        }
    }

    /// Extracts the artifact from a lookup that is expected to hit.
    fn expect_hit<K: Clone + Eq + Hash, A: Clone>(
        lookup: Result<SpecializationCacheLookup<'_, K, A>, ReentrantSpecializationError>,
    ) -> A {
        match lookup {
            Ok(SpecializationCacheLookup::Hit(artifact)) => artifact,
            Ok(SpecializationCacheLookup::Miss(_)) => panic!("expected a hit but the lookup missed"),
            Err(error) => panic!("expected a hit but the lookup failed: {error}"),
        }
    }

    #[test]
    fn test_reuse_requires_key_equality_not_hash_equality() {
        /// Key whose `Hash` implementation collides for every value while `Eq` still distinguishes values, pinning
        /// that reuse is decided by equality alone.
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct CollidingKey(u32);

        impl Hash for CollidingKey {
            fn hash<H: Hasher>(&self, state: &mut H) {
                state.write_u8(0);
            }
        }

        let cache = SpecializationCache::<CollidingKey, &'static str>::new(4);
        expect_producer(cache.lookup(CollidingKey(1))).insert("one");
        expect_producer(cache.lookup(CollidingKey(2))).insert("two");

        // Colliding hashes must not alias: each key keeps its own artifact.
        assert_eq!(expect_hit(cache.lookup(CollidingKey(1))), "one");
        assert_eq!(expect_hit(cache.lookup(CollidingKey(2))), "two");
        assert_eq!(cache.len(), 2);

        // An equal key reuses, and an unequal key with the same hash still misses.
        expect_producer(cache.lookup(CollidingKey(3)));
        assert_eq!(cache.statistics().misses, 3);
    }

    #[test]
    fn test_bounded_lru_recency_eviction_and_capacity_clamping() {
        let cache = SpecializationCache::<u32, &'static str>::new(2);
        assert_eq!(cache.capacity(), 2);
        assert!(cache.is_empty());

        expect_producer(cache.lookup(1)).insert("one");
        expect_producer(cache.lookup(2)).insert("two");
        assert_eq!(cache.keys(), vec![2, 1]);

        // A hit refreshes recency, so the *other* key becomes the eviction candidate.
        assert_eq!(expect_hit(cache.lookup(1)), "one");
        assert_eq!(cache.keys(), vec![1, 2]);
        expect_producer(cache.lookup(3)).insert("three");
        assert_eq!(cache.keys(), vec![3, 1]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.statistics().evictions, 1);

        // Zero capacity is clamped to one so that a producer's work is never discarded immediately.
        let clamped = SpecializationCache::<u32, &'static str>::new(0);
        assert_eq!(clamped.capacity(), 1);
        expect_producer(clamped.lookup(1)).insert("one");
        expect_producer(clamped.lookup(2)).insert("two");
        assert_eq!(clamped.keys(), vec![2]);
        assert_eq!(clamped.statistics().evictions, 1);
    }

    #[test]
    fn test_statistics_counting_and_clearing() {
        let cache = SpecializationCache::<u32, &'static str>::new(1);
        assert_eq!(cache.statistics(), SpecializationCacheStatistics::default());

        expect_producer(cache.lookup(1)).insert("one");
        let _ = expect_hit(cache.lookup(1));
        drop(expect_producer(cache.lookup(2)));
        expect_producer(cache.lookup(2)).insert("two");
        assert_eq!(
            cache.statistics(),
            SpecializationCacheStatistics { hits: 1, misses: 3, productions: 2, failures: 1, evictions: 1 },
        );

        // Clearing statistics leaves the retained artifacts alone.
        cache.clear_statistics();
        assert_eq!(cache.statistics(), SpecializationCacheStatistics::default());
        assert_eq!(expect_hit(cache.lookup(2)), "two");
        assert_eq!(cache.statistics(), SpecializationCacheStatistics { hits: 1, ..Default::default() });
    }

    #[test]
    fn test_failed_production_caches_nothing_and_retries() {
        let cache = SpecializationCache::<u32, &'static str>::new(4);

        // Dropping a producer without inserting must leave no entry and no in-flight marker.
        drop(expect_producer(cache.lookup(1)));
        assert!(cache.is_empty());
        assert_eq!(cache.statistics().failures, 1);

        // The convenience entry point propagates production errors without caching them.
        let failed = cache.get_or_try_insert_with(1, || Err::<&'static str, _>("production failed"));
        assert!(matches!(failed, Err(SpecializationCacheError::Production(message)) if message == "production failed"));
        assert!(cache.is_empty());

        // A later attempt for the same key retries from scratch and succeeds.
        assert_eq!(cache.get_or_try_insert_with::<&'static str, _>(1, || Ok("one")), Ok("one"));
        assert_eq!(cache.get_or_try_insert_with::<&'static str, _>(1, || panic!("must not reproduce")), Ok("one"));
        assert_eq!(
            cache.statistics(),
            SpecializationCacheStatistics { hits: 1, misses: 3, productions: 1, failures: 2, evictions: 0 },
        );
    }

    #[test]
    fn test_panicking_production_unwinds_cleanly() {
        let cache = SpecializationCache::<u32, &'static str>::new(4);
        let panicked = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let producer = expect_producer(cache.lookup(1));
            assert_eq!(producer.key(), &1);
            panic!("production panicked");
        }));
        assert!(panicked.is_err());

        // The unwound producer's `Drop` cleared the in-flight marker and cached nothing.
        assert!(cache.is_empty());
        assert_eq!(cache.statistics().failures, 1);
        expect_producer(cache.lookup(1)).insert("one");
        assert_eq!(expect_hit(cache.lookup(1)), "one");
    }

    #[test]
    fn test_same_thread_same_key_reentrancy_is_rejected() {
        let cache = SpecializationCache::<u32, &'static str>::new(4);
        let producer = expect_producer(cache.lookup(1));

        // Recursively producing the in-flight key cannot terminate, so it is rejected rather than awaited.
        assert!(matches!(cache.lookup(1), Err(ReentrantSpecializationError)));
        assert_eq!(
            ReentrantSpecializationError.to_string(),
            "recursive request for a specialization that is already being produced on this thread",
        );

        // Nested production of a *different* key is legitimate and proceeds.
        expect_producer(cache.lookup(2)).insert("two");
        producer.insert("one");
        assert_eq!(expect_hit(cache.lookup(1)), "one");
        assert_eq!(expect_hit(cache.lookup(2)), "two");

        // The marker is released once production completes, so the same key can be produced again.
        cache.clear();
        expect_producer(cache.lookup(1)).insert("one again");
        assert_eq!(expect_hit(cache.lookup(1)), "one again");
    }

    #[test]
    fn test_clear_and_selective_invalidation() {
        let cache = SpecializationCache::<u32, &'static str>::new(8);
        expect_producer(cache.lookup(1)).insert("one");
        expect_producer(cache.lookup(2)).insert("two");
        expect_producer(cache.lookup(3)).insert("three");

        // Selective invalidation reports how many retained keys it removed.
        assert_eq!(cache.invalidate_where(|key| key % 2 == 1), 2);
        assert_eq!(cache.keys(), vec![2]);
        assert_eq!(cache.invalidate_where(|key| *key == 99), 0);

        // Removal is not eviction, so the eviction counter stays untouched.
        assert_eq!(cache.statistics().evictions, 0);

        // Clearing does not cancel an active producer, which may still publish afterwards.
        let producer = expect_producer(cache.lookup(4));
        cache.clear();
        assert!(cache.is_empty());
        producer.insert("four");
        assert_eq!(cache.keys(), vec![4]);
    }

    #[test]
    fn test_thread_safety_is_structural_and_conditional() {
        fn assert_send_and_sync<T: Send + Sync>() {}

        // Thread safety derives from the `Mutex`/`AtomicU64` state; there are no `unsafe impl`s in this module.
        assert_send_and_sync::<SpecializationCache<u32, &'static str>>();
        assert_send_and_sync::<SpecializationCacheStatistics>();
        assert_send_and_sync::<ReentrantSpecializationError>();
    }

    #[test]
    fn test_concurrent_cold_misses_both_produce_one_consistent_entry() {
        let cache = SpecializationCache::<u32, String>::new(4);
        let barrier = Barrier::new(2);
        thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    // The barrier pins that both lookups miss before either producer publishes, so this exercises
                    // genuine cross-thread duplicate production rather than a hit after a race.
                    let producer = expect_producer(cache.lookup(1));
                    barrier.wait();
                    producer.insert("compiled".to_string());
                });
            }
        });

        // Duplicate production is allowed across threads and inserts are idempotent, so one entry remains. The second
        // insert replaces the first rather than evicting anything, which the zero eviction count below pins.
        assert_eq!(cache.len(), 1);
        assert_eq!(expect_hit(cache.lookup(1)), "compiled".to_string());
        assert_eq!(
            cache.statistics(),
            SpecializationCacheStatistics { hits: 1, misses: 2, productions: 2, failures: 0, evictions: 0 },
        );
    }

    #[test]
    fn test_concurrent_hits_on_a_warm_cache() {
        let cache = SpecializationCache::<u32, String>::new(4);
        expect_producer(cache.lookup(1)).insert("compiled".to_string());
        cache.clear_statistics();

        let barrier = Barrier::new(4);
        thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    barrier.wait();
                    assert_eq!(expect_hit(cache.lookup(1)), "compiled".to_string());
                });
            }
        });

        assert_eq!(cache.statistics(), SpecializationCacheStatistics { hits: 4, ..Default::default() });
    }

    #[test]
    fn test_function_specialization_key() {
        let key = FunctionSpecializationKey::new(("training", 4), vec![(), ()], "f32[2,3]");
        assert_eq!(key.static_parameters(), &("training", 4));
        assert_eq!(key.input_structure(), &vec![(), ()]);
        assert_eq!(key.dispatch(), &"f32[2,3]");

        // Every component participates in equality, and equal keys reuse one cache entry.
        assert_eq!(key, FunctionSpecializationKey::new(("training", 4), vec![(), ()], "f32[2,3]"));
        assert_ne!(key, FunctionSpecializationKey::new(("inference", 4), vec![(), ()], "f32[2,3]"));
        assert_ne!(key, FunctionSpecializationKey::new(("training", 4), vec![()], "f32[2,3]"));
        assert_ne!(key, FunctionSpecializationKey::new(("training", 4), vec![(), ()], "f32[4,3]"));

        let cache = SpecializationCache::new(4);
        expect_producer(cache.lookup(key.clone())).insert("staged");
        assert_eq!(expect_hit(cache.lookup(key)), "staged");
    }
}
