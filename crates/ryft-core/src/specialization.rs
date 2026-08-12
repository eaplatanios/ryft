//! Backend-neutral, process-local caching primitive for retained _specializations_.
//!
//! A specialization is the artifact a front end produces for one exact configuration of a retained callable:
//! a traced [`Program`](crate::Program), a lowered program, a compiled function, or a transformed program.
//! [`SpecializationCache`] retains those artifacts under a **single level** contract, mapping one key to one artifact.
//! It deliberately has no tiers, no fallback lifecycles, and no timing policy. Consumers keep their own phase-specific
//! statistics and decide what a key and an artifact means.
//!
//! # API Structure
//!
//! [`SpecializationCache`] owns retained artifacts, in-flight markers, and private atomic counters.
//! [`SpecializationCacheEntry`] and [`SpecializationCacheProducer`] model the entry-production protocol, while
//! [`SpecializationCacheStatistics`] is an ordinary value snapshot of the live counters. Refer to the
//! [`SpecializationCache`] documentation for more information on these relationships.
//! [`TransformCache`](crate::programs::transforms::TransformCache) lets a typed transform descriptor select this same
//! cache without wrapping or changing its entry protocol.
//!
//! # Reuse Contract
//!
//! Reuse is defined by key _equality_, never by a hash alone, so `Key: Clone + Eq + Hash` and the retained map is a
//! bounded Least Recently Used (LRU) cache. Artifacts must be `Clone`, which in practice means large structural
//! artifacts should be held behind cheap reference-counted handles rather than deep-cloned on every hit.
//!
//! # Production and Reentrancy
//!
//! Production runs outside every lock. [`SpecializationCache::try_entry`] takes the locks only long enough to consult
//! the retained map and to register an in-flight marker, then hands back a [`SpecializationCacheProducer`] Resource
//! Acquisition Is Initialization (RAII) guard; the caller runs arbitrary code (e.g., tracing, lowering, compilation)
//! while the cache is fully available to other threads. [`SpecializationCacheProducer::insert`] publishes the artifact,
//! and dropping the producer without inserting (because production failed or panicked) clears the marker and caches
//! nothing, so the next call retries from scratch.
//!
//! In-flight markers are tracked per `(ThreadId, Key)` pair, which gives exactly three behaviors:
//!
//!   - **Same Thread, Same Key:** Rejected immediately with [`ReentrantSpecializationError`]. Recursive production
//!     of the specialization currently being produced cannot terminate, so it is an error rather than a wait.
//!   - **Same Thread, Different Key:** Proceeds. Nested production of a _different_ specialization is legitimate.
//!   - **Different Thread, Same Key:** Proceeds. Both threads produce, and both inserts are idempotent with the last
//!     one winning.
//!
//! Because that marker names the thread that requested the entry, a [`SpecializationCacheProducer`] is deliberately
//! neither `Send` nor `Sync`: production must complete on the thread that started it, and the compiler enforces that
//! instead of a runtime check.
//!
//! Caller-supplied code never runs while a cache lock is held. Production runs after [`SpecializationCache::try_entry`]
//! returns, [`SpecializationCache::invalidate_entries_if`] evaluates its predicate against a snapshot of the retained
//! keys, and keys and artifacts removed by insertion, invalidation, or [`SpecializationCache::clear`] are dropped after
//! the lock is released, so a predicate or a destructor may reenter the cache freely. The sole exceptions are the `Key`
//! and `Artifact` operations the cache itself performs under a lock: `Key::clone` and `Artifact::clone` while a hit is
//! being served or a key snapshot is taken, and `Key::hash` and `Key::eq` on every retained-map consultation and every
//! in-flight marker registration and release. None of those four may reenter the cache.
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
//! statistics are [`AtomicU64`] counters, so [`SpecializationCache`] is `Send` and `Sync` exactly when `Key` and
//! `Artifact` are, with no `unsafe impl` anywhere. Thread-confined consumers whose artifacts are not `Send` still
//! work single-threaded; they simply do not get `Send`.

use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::ThreadId;

use lru::LruCache;
use thiserror::Error;

/// Error returned by the convenience [`SpecializationCache::get_or_try_insert_with`] entry point, layering a
/// caller-defined production error `E` over the cache's own [`ReentrantSpecializationError`] rejection.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum SpecializationCacheError<E: Debug + Display> {
    /// Error returned if the same thread is already producing this key.
    #[error(transparent)]
    Reentrant(#[from] ReentrantSpecializationError),

    /// Error returned if the caller's production closure failed. Nothing was cached and the next call retries.
    #[error("{0}")]
    Production(E),
}

/// Error returned when a thread requests production of a specialization key that the _same_ thread is
/// already producing. Recursive production of the specialization currently in flight cannot terminate, and so
/// [`SpecializationCache`] rejects it immediately instead of waiting. Producing a different key from within an active
/// producer is allowed, and so is producing the same key concurrently from a different thread.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Error)]
#[error("recursive request for a specialization that is already being produced on this thread")]
pub struct ReentrantSpecializationError;

/// Snapshot of a [`SpecializationCache`]'s activity since construction or since the last
/// [`SpecializationCache::clear_statistics`] call. Every counter saturates rather than wrapping. Note
/// that `misses` counts lookups that found no retained artifact, including requests subsequently rejected with
/// [`ReentrantSpecializationError`], and that `productions + abandoned_productions` need not equal `misses`
/// while producers are still in flight.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SpecializationCacheStatistics {
    /// Lookups served by a retained artifact.
    pub hits: u64,

    /// Lookups that found no retained artifact.
    pub misses: u64,

    /// Artifacts published through [`SpecializationCacheProducer::insert`].
    pub productions: u64,

    /// Producers dropped without publishing an artifact, including failed, panicking, or deliberately abandoned work.
    pub abandoned_productions: u64,

    /// Retained artifacts dropped by the bounded Least Recently Used (LRU) cache to make room for a new one. Entries
    /// removed by [`SpecializationCache::clear`] or [`SpecializationCache::invalidate_entries_if`] are not evictions.
    pub evictions: u64,
}

/// Internal accumulator of independent atomic counters backing [`SpecializationCacheStatistics`]. All updates use
/// [`Ordering::Relaxed`] as the counters are diagnostic, they order nothing, and the retained map's [`Mutex`] already
/// provides the synchronization that correctness depends on.
#[derive(Debug, Default)]
struct SpecializationCacheStatisticsAccumulator {
    /// Lookups served by a retained artifact.
    hits: AtomicU64,

    /// Lookups that found no retained artifact.
    misses: AtomicU64,

    /// Artifacts published through a producer.
    productions: AtomicU64,

    /// Producers dropped without publishing an artifact.
    abandoned_productions: AtomicU64,

    /// Retained artifacts dropped by the bounded Least Recently Used (LRU) cache.
    evictions: AtomicU64,
}

impl SpecializationCacheStatisticsAccumulator {
    /// Records a lookup served by a retained artifact.
    fn increment_hits(&self) {
        self.hits
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| Some(count.saturating_add(1)))
            .unwrap();
    }

    /// Records a lookup that found no retained artifact.
    fn increment_misses(&self) {
        self.misses
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| Some(count.saturating_add(1)))
            .unwrap();
    }

    /// Records an artifact published through a producer.
    fn increment_productions(&self) {
        self.productions
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| Some(count.saturating_add(1)))
            .unwrap();
    }

    /// Records a producer dropped without publishing an artifact.
    fn increment_abandoned_productions(&self) {
        self.abandoned_productions
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| Some(count.saturating_add(1)))
            .unwrap();
    }

    /// Records an artifact evicted by the bounded Least Recently Used (LRU) cache.
    fn increment_evictions(&self) {
        self.evictions
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| Some(count.saturating_add(1)))
            .unwrap();
    }

    /// Returns a [`SpecializationCacheStatistics`] of the counters underlying this
    /// [`SpecializationCacheStatisticsAccumulator`]. The counters are read independently, so a snapshot
    /// taken while other threads are active reflects a plausible interleaving rather than a single instant.
    fn snapshot(&self) -> SpecializationCacheStatistics {
        SpecializationCacheStatistics {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            productions: self.productions.load(Ordering::Relaxed),
            abandoned_productions: self.abandoned_productions.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    /// Resets all the underlying counters of this [`SpecializationCacheStatisticsAccumulator`] to zero.
    fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.productions.store(0, Ordering::Relaxed);
        self.abandoned_productions.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
    }
}

impl<Key: Clone + Eq + Hash, Artifact: Clone> Debug for SpecializationCache<Key, Artifact> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SpecializationCache")
            .field("len", &self.len())
            .field("capacity", &self.capacity())
            .field("statistics", &self.statistics.snapshot())
            .finish_non_exhaustive()
    }
}

/// Resource Acquisition Is Initialization (RAII) guard authorizing production of a specialization. This guard holds
/// the `(ThreadId, Key)` in-flight marker for its key. Calling [`Self::insert`] publishes an artifact and releases
/// the marker. Dropping the guard without inserting releases the marker, counts an abandoned production, and caches
/// nothing. Because the guard borrows the cache rather than holding a lock, production runs with the cache fully
/// available to other threads.
///
/// A producer is thread-affine and therefore neither `Send` nor `Sync`, which the [`PhantomData`] marker it holds
/// enforces at compile time. The in-flight marker is keyed by the thread that requested the entry, so production must
/// complete on that thread. If a producer was moved elsewhere, the originating thread would remain barred from the key
/// it is no longer producing, while the receiving thread could register the same key a second time and recursively
/// produce it (i.e., the nonterminating recursion that [`ReentrantSpecializationError`] exists to reject).
#[derive(Debug)]
pub struct SpecializationCacheProducer<'c, Key: Clone + Eq + Hash, Artifact: Clone> {
    /// [`SpecializationCache`] that this producer publishes to.
    cache: &'c SpecializationCache<Key, Artifact>,

    /// [`ThreadId`] of the thread that registered the in-flight marker, so that the marker is released with the key
    /// it was registered under.
    thread: ThreadId,

    /// Key being produced. Taken by [`Self::insert`] so that [`Drop`] does not double-release the marker.
    key: Option<Key>,

    /// [`PhantomData`] marker pinning this guard to the thread that registered the in-flight marker. A raw pointer is
    /// neither `Send` nor `Sync`, and so neither is this guard.
    marker: PhantomData<*mut ()>,
}

impl<Key: Clone + Eq + Hash, Artifact: Clone> SpecializationCacheProducer<'_, Key, Artifact> {
    /// Returns the key this [`SpecializationCacheProducer`] is authorized to produce.
    #[inline]
    pub fn key(&self) -> &Key {
        self.key.as_ref().unwrap()
    }

    /// Publishes `artifact` for this [`SpecializationCacheProducer`]'s key, releases the in-flight marker, and returns
    /// the artifact. Inserting is _idempotent_ meaning that if another thread produced the same key concurrently, the
    /// later insert replaces the earlier one. Both artifacts are interchangeable by the [`SpecializationCache`]'s reuse
    /// contract, so last-one-wins is safe.
    ///
    /// The entry this insertion displaces, if any, is dropped only after the artifact is published, the counters are
    /// updated, and the in-flight marker is released, so a displaced destructor that panics leaves the cache exactly
    /// as a successful insertion would.
    pub fn insert(mut self, artifact: Artifact) -> Artifact {
        let key = self.key.take().unwrap();

        // The displaced pair is bound here so that it outlives the guard below and its destructors run unlocked, which
        // is what lets an artifact reenter the cache while being dropped. It is deliberately dropped last, after the
        // counters are updated and the in-flight marker is released, so that a destructor that panics can neither
        // strand the marker (which would reject every later lookup of this key on this thread as reentrant) nor skip a
        // counter update.
        let displaced = {
            let mut entries = self.cache.entries.lock().expect("specialization cache mutex is poisoned");
            entries.push(key.clone(), artifact.clone())
        };
        if let Some((displaced_key, _)) = &displaced
            && *displaced_key != key
        {
            self.cache.statistics.increment_evictions();
        }
        self.cache.statistics.increment_productions();

        // Both the marker offered for removal and the one recovered from the set are bound outside the guard,
        // so that a `Key` destructor may reenter the cache as well.
        let marker = (self.thread, key);
        let removed = {
            let mut in_flight = self.cache.in_flight.lock().expect("specialization cache mutex is poisoned");
            in_flight.take(&marker)
        };

        // Reverse declaration-order dropping would currently produce this same order, but we keep it explicit here.
        // These values contain caller-defined keys or artifacts whose destructors may reenter or panic, so they must
        // run only after publication, counter updates, marker removal, and both mutex guards have completed.
        drop(removed);
        drop(marker);
        drop(displaced);

        artifact
    }
}

impl<Key: Clone + Eq + Hash, Artifact: Clone> Drop for SpecializationCacheProducer<'_, Key, Artifact> {
    fn drop(&mut self) {
        // `insert` takes the key, so this only runs when production failed or unwound. Releasing the marker here is
        // what makes failed and panicking production retryable instead of permanently reentrant.
        if let Some(key) = self.key.take() {
            // Both the marker offered for removal and the one recovered from the set are bound outside the guard,
            // so that a `Key` destructor may reenter the cache.
            let marker = (self.thread, key);
            let removed = {
                let mut in_flight = self.cache.in_flight.lock().expect("specialization cache mutex is poisoned");
                in_flight.take(&marker)
            };
            self.cache.statistics.increment_abandoned_productions();
            drop(removed);
            drop(marker);
        }
    }
}

/// Represents a [`SpecializationCache`] entry returned by [`SpecializationCache::try_entry`].
#[derive(Debug)]
pub enum SpecializationCacheEntry<'c, Key: Clone + Eq + Hash, Artifact: Clone> {
    /// A retained artifact was found and its recency was refreshed.
    Occupied(Artifact),

    /// No artifact was retained. The producer guard holds the in-flight marker until it inserts a new entry
    /// or is dropped.
    Vacant(SpecializationCacheProducer<'c, Key, Artifact>),
}

/// Bounded, process-local cache mapping owner-defined keys to retained artifacts. Refer to the documentation of
/// [this module](self) for information on the reuse, production, reentrancy, and thread-safety contracts.
///
/// # Type Relationships
///
/// ```mermaid
/// %%{init: {"themeCSS": ".nodeLabel code, .edgeLabel code { white-space: nowrap !important; }"}}%%
/// flowchart LR
///   owner["Retained Callable or Transform Owner"] -->|"owns"| cache["&lt;code&gt;SpecializationCache&lt;/code&gt;"]
///   key["Owner-Defined &lt;code&gt;Key&lt;/code&gt;"] --> cache
///   caller["Caller"] -->|"&lt;code&gt;try_entry(Key)&lt;/code&gt;"| cache
///   cache -->|"&lt;code&gt;Occupied&lt;/code&gt;"| artifact["Cloned &lt;code&gt;Artifact&lt;/code&gt;"]
///   cache -->|"&lt;code&gt;Vacant&lt;/code&gt;"| producer["&lt;code&gt;SpecializationCacheProducer&lt;/code&gt;"]
///   cache -->|"same thread and &lt;code&gt;Key&lt;/code&gt;"| reentrant["&lt;code&gt;ReentrantSpecializationError&lt;/code&gt;"]
///   producer -->|"&lt;code&gt;insert(Artifact)&lt;/code&gt;"| cache
///   producer -->|"drop without insert"| retry["Marker cleared; next request retries"]
///   cache --> counters["Private atomic counters"]
///   counters -->|"&lt;code&gt;statistics()&lt;/code&gt;"| snapshot["&lt;code&gt;SpecializationCacheStatistics&lt;/code&gt;"]
/// ```
///
/// [`Self::try_entry`] represents the low-level entry API for this cache. An occupied entry returns a cloned artifact.
/// A vacant entry returns a thread-affine producer that authorizes publication. [`Self::get_or_try_insert_with`] wraps
/// that protocol when callers do not need to separate entry resolution from production. Statistics are exposed as
/// ordinary value snapshots, while private atomics accumulate events without participating in cache correctness.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct SpecializationCache<Key: Clone + Eq + Hash, Artifact: Clone> {
    /// Retained artifacts in Least-Recently-Used (LRU) order.
    entries: Mutex<LruCache<Key, Artifact>>,

    /// Keys currently being produced, paired with the [`ThreadId`] of the thread that is producing them.
    in_flight: Mutex<HashSet<(ThreadId, Key)>>,

    /// [`SpecializationCacheStatisticsAccumulator`] that contains diagnostic counters.
    statistics: SpecializationCacheStatisticsAccumulator,
}

impl<Key: Clone + Eq + Hash, Artifact: Clone> SpecializationCache<Key, Artifact> {
    /// Creates an empty [`SpecializationCache`] retaining at most `capacity` artifacts. A `capacity` of zero is clamped
    /// to one, because a cache that can retain nothing would turn every producer into wasted work.
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(LruCache::new(NonZeroUsize::new(capacity.max(1)).unwrap())),
            in_flight: Mutex::new(HashSet::new()),
            statistics: SpecializationCacheStatisticsAccumulator::default(),
        }
    }

    /// Returns the number of retained artifacts in this [`SpecializationCache`].
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.lock().expect("specialization cache mutex is poisoned").len()
    }

    /// Returns `true` if no artifact is retained in this [`SpecializationCache`].
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.lock().expect("specialization cache mutex is poisoned").is_empty()
    }

    /// Returns the maximum number of artifacts this [`SpecializationCache`] can retain.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.entries.lock().expect("specialization cache mutex is poisoned").cap().get()
    }

    /// Returns the keys retained in this [`SpecializationCache`] from most to least recently used.
    #[inline]
    pub fn keys(&self) -> Vec<Key> {
        self.entries
            .lock()
            .expect("specialization cache mutex is poisoned")
            .iter()
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Returns the cache entry for `key`, containing either the retained artifact or a [`SpecializationCacheProducer`]
    /// authorized to make one. A hit refreshes the key's recency, counts a hit, and returns a clone of the artifact.
    /// A miss counts a miss (including when the request is then rejected as reentrant) registers a
    /// `(current thread, key)` in-flight marker, and returns a producer. No lock is held once this function returns.
    pub fn try_entry(
        &self,
        key: Key,
    ) -> Result<SpecializationCacheEntry<'_, Key, Artifact>, ReentrantSpecializationError> {
        if let Some(artifact) = self.entries.lock().expect("specialization cache mutex is poisoned").get(&key).cloned()
        {
            self.statistics.increment_hits();
            return Ok(SpecializationCacheEntry::Occupied(artifact));
        }

        self.statistics.increment_misses();
        let thread = std::thread::current().id();

        // Registering with `replace` rather than `insert` hands the equal marker that is already in flight back out
        // of the guard, so that a rejected duplicate registration drops its `Key` after the lock is released instead
        // of under it. The two markers are interchangeable by definition, since they compare equal.
        let already_in_flight = {
            let mut in_flight = self.in_flight.lock().expect("specialization cache mutex is poisoned");
            in_flight.replace((thread, key.clone()))
        };
        if already_in_flight.is_some() {
            return Err(ReentrantSpecializationError);
        }

        Ok(SpecializationCacheEntry::Vacant(SpecializationCacheProducer {
            cache: self,
            thread,
            key: Some(key),
            marker: PhantomData,
        }))
    }

    /// Returns the artifact retained for `key`, producing and retaining one with `produce_fn` on a miss. This is the
    /// convenience form of [`Self::try_entry`]. Callers that need to time entry resolution separately from production,
    /// or that need the producer's key, should use [`Self::try_entry`] directly. Production errors are propagated and
    /// nothing is cached, so a later call retries.
    #[inline]
    pub fn get_or_try_insert_with<E: Debug + Display, F: FnOnce() -> Result<Artifact, E>>(
        &self,
        key: Key,
        produce_fn: F,
    ) -> Result<Artifact, SpecializationCacheError<E>> {
        match self.try_entry(key)? {
            SpecializationCacheEntry::Occupied(artifact) => Ok(artifact),
            SpecializationCacheEntry::Vacant(producer) => match produce_fn() {
                Ok(artifact) => Ok(producer.insert(artifact)),
                Err(error) => Err(SpecializationCacheError::Production(error)),
            },
        }
    }

    /// Removes every retained artifact from this [`SpecializationCache`], leaving its [`SpecializationCacheStatistics`]
    /// untouched. Note that clearing the cache does not cancel producers that are already in flight. A producer that
    /// succeeds afterwards still inserts its artifact, because it was authorized before the clear and its key still
    /// identifies an interchangeable specialization.
    pub fn clear(&self) {
        // Swapping an empty map of the same capacity in under the lock and dropping the removed one afterwards keeps
        // every key and artifact destructor outside the lock so that a destructor may reenter the cache.
        let removed = {
            let mut entries = self.entries.lock().expect("specialization cache mutex is poisoned");
            let capacity = entries.cap();
            std::mem::replace(&mut *entries, LruCache::new(capacity))
        };
        drop(removed);
    }

    /// Removes every retained artifact whose key satisfies `predicate_fn` from this [`SpecializationCache`], returning
    /// the number of artifacts that were removed. The predicate runs outside every cache lock, against a snapshot of
    /// the keys retained when this call started, and so it may reenter the cache. If it panics, no removal pass begins
    /// and no cache lock is poisoned. Keys retained after the snapshot are not offered to the predicate. Removal is by
    /// key equality rather than entry generation, so if another thread replaces a selected key with an equal key
    /// between the snapshot and removal, that current equal entry is removed. Like [`Self::clear`], this does not
    /// cancel producers already in flight.
    pub fn invalidate_entries_if(&self, mut predicate_fn: impl FnMut(&Key) -> bool) -> usize {
        let selected = self.keys().into_iter().filter(|key| predicate_fn(key)).collect::<Vec<_>>();
        let removed = {
            let mut entries = self.entries.lock().expect("specialization cache mutex is poisoned");
            selected.iter().filter_map(|key| entries.pop_entry(key)).collect::<Vec<_>>()
        };

        // The removed pairs are dropped after the guard above is released, and so their destructors may reenter.
        let count = removed.len();
        drop(removed);
        count
    }

    /// Returns a snapshot of this [`SpecializationCache`]'s [`SpecializationCacheStatistics`].
    #[inline]
    pub fn statistics(&self) -> SpecializationCacheStatistics {
        self.statistics.snapshot()
    }

    /// Resets this [`SpecializationCache`]'s [`SpecializationCacheStatistics`], leaving the retained artifacts
    /// untouched. The relaxed atomic counters stored in the underlying statistics reset independently. When other
    /// threads are active, events may fall on either side of the reset for each counter. Exact interval accounting
    /// therefore requires callers to quiesce cache activity.
    #[inline]
    pub fn clear_statistics(&self) {
        self.statistics.reset();
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::{Debug, Formatter};
    use std::hash::{Hash, Hasher};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Barrier, OnceLock};
    use std::thread;

    use pretty_assertions::assert_eq;

    use super::*;

    /// Extracts the producer from an entry that is expected to be vacant.
    fn expect_vacant_producer<Key: Clone + Eq + Hash, Artifact: Clone>(
        entry: Result<SpecializationCacheEntry<'_, Key, Artifact>, ReentrantSpecializationError>,
    ) -> SpecializationCacheProducer<'_, Key, Artifact> {
        match entry {
            Ok(SpecializationCacheEntry::Vacant(producer)) => producer,
            Ok(SpecializationCacheEntry::Occupied(_)) => panic!("expected a vacant entry but it was occupied"),
            Err(error) => panic!("expected a vacant entry but the request failed: {error}"),
        }
    }

    /// Extracts the artifact from an entry that is expected to be occupied.
    fn expect_occupied_artifact<Key: Clone + Eq + Hash, Artifact: Clone>(
        entry: Result<SpecializationCacheEntry<'_, Key, Artifact>, ReentrantSpecializationError>,
    ) -> Artifact {
        match entry {
            Ok(SpecializationCacheEntry::Occupied(artifact)) => artifact,
            Ok(SpecializationCacheEntry::Vacant(_)) => panic!("expected an occupied entry but it was vacant"),
            Err(error) => panic!("expected an occupied entry but the request failed: {error}"),
        }
    }

    #[test]
    fn test_specialization_cache_errors() {
        let reentrant_error = ReentrantSpecializationError;
        assert_eq!(
            reentrant_error.to_string(),
            "recursive request for a specialization that is already being produced on this thread",
        );
        assert_eq!(format!("{reentrant_error:?}"), "ReentrantSpecializationError");

        let cache_error = SpecializationCacheError::<&str>::from(reentrant_error);
        assert_eq!(
            cache_error.to_string(),
            "recursive request for a specialization that is already being produced on this thread",
        );
        assert_eq!(format!("{cache_error:?}"), "Reentrant(ReentrantSpecializationError)");

        let production_error = SpecializationCacheError::Production("production failed");
        assert_eq!(production_error.to_string(), "production failed");
        assert_eq!(format!("{production_error:?}"), "Production(\"production failed\")");
    }

    #[test]
    fn test_specialization_cache_debug_does_not_format_entries() {
        #[derive(Clone, PartialEq, Eq, Hash)]
        struct UnformattableKey(u32);

        impl Debug for UnformattableKey {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                let _formatter = formatter;
                panic!("cache debug must not format retained keys")
            }
        }

        #[derive(Clone)]
        struct UnformattableArtifact;

        impl Debug for UnformattableArtifact {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                let _formatter = formatter;
                panic!("cache debug must not format retained artifacts")
            }
        }

        let cache = SpecializationCache::new(2);
        expect_vacant_producer(cache.try_entry(UnformattableKey(1))).insert(UnformattableArtifact);

        assert_eq!(
            format!("{cache:?}"),
            "SpecializationCache { len: 1, capacity: 2, statistics: SpecializationCacheStatistics { hits: 0, \
             misses: 1, productions: 1, abandoned_productions: 0, evictions: 0 }, .. }",
        );
    }

    #[test]
    fn test_specialization_cache_new() {
        let cache = SpecializationCache::<u32, &'static str>::new(2);
        assert_eq!(cache.capacity(), 2);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.keys(), Vec::<u32>::new());

        // Zero capacity is clamped to one so that a producer's work is never discarded immediately.
        let clamped_cache = SpecializationCache::<u32, &'static str>::new(0);
        assert_eq!(clamped_cache.capacity(), 1);
    }

    #[test]
    fn test_specialization_cache_try_entry_refreshes_lru_recency() {
        let cache = SpecializationCache::<u32, &'static str>::new(2);
        expect_vacant_producer(cache.try_entry(1)).insert("one");
        expect_vacant_producer(cache.try_entry(2)).insert("two");
        assert_eq!(cache.keys(), vec![2, 1]);

        // A hit refreshes recency, so the other key becomes the eviction candidate.
        assert_eq!(expect_occupied_artifact(cache.try_entry(1)), "one");
        assert_eq!(cache.keys(), vec![1, 2]);
        expect_vacant_producer(cache.try_entry(3)).insert("three");
        assert_eq!(cache.keys(), vec![3, 1]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.statistics().evictions, 1);
    }

    #[test]
    fn test_specialization_cache_try_entry_uses_key_equality() {
        // Key whose `Hash` implementation collides for every value while `Eq` still distinguishes values, pinning
        // that reuse is decided by equality alone.
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct CollidingKey(u32);

        impl Hash for CollidingKey {
            fn hash<H: Hasher>(&self, state: &mut H) {
                state.write_u8(0);
            }
        }

        let cache = SpecializationCache::<CollidingKey, &'static str>::new(4);
        expect_vacant_producer(cache.try_entry(CollidingKey(1))).insert("one");
        expect_vacant_producer(cache.try_entry(CollidingKey(2))).insert("two");

        // Colliding hashes must not alias: each key keeps its own artifact.
        assert_eq!(expect_occupied_artifact(cache.try_entry(CollidingKey(1))), "one");
        assert_eq!(expect_occupied_artifact(cache.try_entry(CollidingKey(2))), "two");
        assert_eq!(cache.len(), 2);

        // An equal key reuses, and an unequal key with the same hash still misses.
        drop(expect_vacant_producer(cache.try_entry(CollidingKey(3))));
        assert_eq!(cache.statistics().misses, 3);
    }

    #[test]
    fn test_specialization_cache_producer_insert() {
        let cache = SpecializationCache::<u32, String>::new(4);
        let producer = expect_vacant_producer(cache.try_entry(7));
        assert_eq!(producer.key(), &7);

        let artifact = producer.insert("compiled".to_string());
        assert_eq!(artifact, "compiled");
        assert_eq!(expect_occupied_artifact(cache.try_entry(7)), "compiled");
    }

    #[test]
    fn test_specialization_cache_try_entry_rejects_same_thread_same_key_reentrancy() {
        let cache = SpecializationCache::<u32, &'static str>::new(4);
        let producer = expect_vacant_producer(cache.try_entry(1));

        // Recursively producing the in-flight key cannot terminate, so it is rejected rather than awaited.
        assert!(matches!(cache.try_entry(1), Err(ReentrantSpecializationError)));

        // Nested production of a different key is legitimate and proceeds.
        expect_vacant_producer(cache.try_entry(2)).insert("two");
        producer.insert("one");
        assert_eq!(expect_occupied_artifact(cache.try_entry(1)), "one");
        assert_eq!(expect_occupied_artifact(cache.try_entry(2)), "two");

        // The marker is released once production completes, so the same key can be produced again.
        cache.clear();
        expect_vacant_producer(cache.try_entry(1)).insert("one again");
        assert_eq!(expect_occupied_artifact(cache.try_entry(1)), "one again");
    }

    #[test]
    fn test_specialization_cache_producer_drop_releases_marker() {
        let cache = SpecializationCache::<u32, &'static str>::new(4);

        drop(expect_vacant_producer(cache.try_entry(1)));
        assert!(cache.is_empty());
        assert_eq!(cache.statistics().abandoned_productions, 1);

        // Releasing the marker makes the same key immediately producible again.
        expect_vacant_producer(cache.try_entry(1)).insert("one");
        assert_eq!(expect_occupied_artifact(cache.try_entry(1)), "one");
    }

    #[test]
    fn test_specialization_cache_producer_drop_releases_marker_after_panic() {
        let cache = SpecializationCache::<u32, &'static str>::new(4);
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            let producer = expect_vacant_producer(cache.try_entry(1));
            assert_eq!(producer.key(), &1);
            panic!("production panicked");
        }));
        assert!(panicked.is_err());

        // The unwound producer's `Drop` cleared the in-flight marker and cached nothing.
        assert!(cache.is_empty());
        assert_eq!(cache.statistics().abandoned_productions, 1);
        expect_vacant_producer(cache.try_entry(1)).insert("one");
        assert_eq!(expect_occupied_artifact(cache.try_entry(1)), "one");
    }

    #[test]
    fn test_specialization_cache_get_or_try_insert_with() {
        let cache = SpecializationCache::<u32, &'static str>::new(4);

        let producer = expect_vacant_producer(cache.try_entry(1));
        assert_eq!(
            cache.get_or_try_insert_with::<&'static str, _>(1, || panic!("must reject before producing")),
            Err(SpecializationCacheError::Reentrant(ReentrantSpecializationError)),
        );
        drop(producer);

        let failed = cache.get_or_try_insert_with(1, || Err::<&'static str, _>("production failed"));
        assert_eq!(failed, Err(SpecializationCacheError::Production("production failed")));
        assert!(cache.is_empty());

        assert_eq!(cache.get_or_try_insert_with::<&'static str, _>(1, || Ok("one")), Ok("one"));
        assert_eq!(cache.get_or_try_insert_with::<&'static str, _>(1, || panic!("must not reproduce")), Ok("one"));
        assert_eq!(
            cache.statistics(),
            SpecializationCacheStatistics {
                hits: 1,
                misses: 4,
                productions: 1,
                abandoned_productions: 2,
                evictions: 0,
            },
        );
    }

    #[test]
    fn test_specialization_cache_clear() {
        let cache = SpecializationCache::<u32, &'static str>::new(8);
        expect_vacant_producer(cache.try_entry(1)).insert("one");
        expect_vacant_producer(cache.try_entry(2)).insert("two");
        let producer = expect_vacant_producer(cache.try_entry(3));
        let statistics = cache.statistics();

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.statistics(), statistics);

        // Clearing does not cancel an active producer, which may still publish afterwards.
        producer.insert("three");
        assert_eq!(cache.keys(), vec![3]);
    }

    #[test]
    fn test_specialization_cache_invalidate_entries_if() {
        let cache = SpecializationCache::<u32, &'static str>::new(8);
        expect_vacant_producer(cache.try_entry(1)).insert("one");
        expect_vacant_producer(cache.try_entry(2)).insert("two");
        expect_vacant_producer(cache.try_entry(3)).insert("three");

        // Selective invalidation reports how many retained keys it removed.
        assert_eq!(cache.invalidate_entries_if(|key| key % 2 == 1), 2);
        assert_eq!(cache.keys(), vec![2]);
        assert_eq!(cache.invalidate_entries_if(|key| *key == 99), 0);

        // Removal is not eviction, so the eviction counter stays untouched.
        assert_eq!(cache.statistics().evictions, 0);

        // Selection uses the original key snapshot, but removal targets the current equality class. A post-snapshot
        // key is never offered to the predicate, while an equal replacement of a selected key is removed.
        cache.clear();
        expect_vacant_producer(cache.try_entry(1)).insert("original");
        expect_vacant_producer(cache.try_entry(2)).insert("two");
        let mut offered_keys = Vec::new();
        assert_eq!(
            cache.invalidate_entries_if(|key| {
                if offered_keys.is_empty() {
                    cache.clear();
                    expect_vacant_producer(cache.try_entry(1)).insert("replacement");
                    expect_vacant_producer(cache.try_entry(3)).insert("post-snapshot");
                }
                offered_keys.push(*key);
                *key == 1
            }),
            1,
        );
        assert_eq!(offered_keys, vec![2, 1]);
        assert_eq!(cache.keys(), vec![3]);
    }

    #[test]
    fn test_specialization_cache_invalidate_entries_if_runs_predicate_outside_locks() {
        let cache = SpecializationCache::<u32, &'static str>::new(8);
        expect_vacant_producer(cache.try_entry(1)).insert("one");
        expect_vacant_producer(cache.try_entry(2)).insert("two");
        expect_vacant_producer(cache.try_entry(3)).insert("three");

        // The predicate observes the full snapshot while it runs, which pins that removal happens afterwards,
        // and it reenters the cache without deadlocking because no lock is held while it runs.
        assert_eq!(cache.invalidate_entries_if(|key| cache.len() == 3 && key % 2 == 1), 2);
        assert_eq!(cache.keys(), vec![2]);

        // A panicking predicate poisons nothing, because the panic unwinds with no cache lock held.
        let panicked = catch_unwind(AssertUnwindSafe(|| cache.invalidate_entries_if(|_| panic!("predicate panicked"))));
        assert!(panicked.is_err());
        assert_eq!(cache.keys(), vec![2]);
        expect_vacant_producer(cache.try_entry(4)).insert("four");
        assert_eq!(expect_occupied_artifact(cache.try_entry(2)), "two");
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_specialization_cache_statistics_and_clear_statistics() {
        let cache = SpecializationCache::<u32, &'static str>::new(1);
        assert_eq!(cache.statistics(), SpecializationCacheStatistics::default());

        expect_vacant_producer(cache.try_entry(1)).insert("one");
        assert_eq!(expect_occupied_artifact(cache.try_entry(1)), "one");
        drop(expect_vacant_producer(cache.try_entry(2)));
        expect_vacant_producer(cache.try_entry(2)).insert("two");
        assert_eq!(
            cache.statistics(),
            SpecializationCacheStatistics {
                hits: 1,
                misses: 3,
                productions: 2,
                abandoned_productions: 1,
                evictions: 1
            },
        );

        // Clearing statistics leaves the retained artifacts alone.
        cache.clear_statistics();
        assert_eq!(cache.statistics(), SpecializationCacheStatistics::default());
        assert_eq!(expect_occupied_artifact(cache.try_entry(2)), "two");
        assert_eq!(cache.statistics(), SpecializationCacheStatistics { hits: 1, ..Default::default() });
    }

    #[test]
    fn test_specialization_cache_statistics_accumulator() {
        let statistics = SpecializationCacheStatisticsAccumulator::default();
        statistics.increment_hits();
        statistics.increment_misses();
        statistics.increment_productions();
        statistics.increment_abandoned_productions();
        statistics.increment_evictions();
        assert_eq!(
            statistics.snapshot(),
            SpecializationCacheStatistics {
                hits: 1,
                misses: 1,
                productions: 1,
                abandoned_productions: 1,
                evictions: 1,
            },
        );

        statistics.reset();
        assert_eq!(statistics.snapshot(), SpecializationCacheStatistics::default());

        // Every counter saturates instead of wrapping.
        statistics.hits.store(u64::MAX, Ordering::Relaxed);
        statistics.misses.store(u64::MAX, Ordering::Relaxed);
        statistics.productions.store(u64::MAX, Ordering::Relaxed);
        statistics.abandoned_productions.store(u64::MAX, Ordering::Relaxed);
        statistics.evictions.store(u64::MAX, Ordering::Relaxed);

        statistics.increment_hits();
        statistics.increment_misses();
        statistics.increment_productions();
        statistics.increment_abandoned_productions();
        statistics.increment_evictions();

        assert_eq!(
            statistics.snapshot(),
            SpecializationCacheStatistics {
                hits: u64::MAX,
                misses: u64::MAX,
                productions: u64::MAX,
                abandoned_productions: u64::MAX,
                evictions: u64::MAX,
            },
        );
    }

    #[test]
    fn test_specialization_cache_removes_artifacts_outside_locks() {
        // Artifact whose destructor reads the cache holding it, which deadlocks unless every removal drops its
        // artifact after releasing the retained map's lock.
        #[derive(Clone)]
        struct ReenteringArtifact(&'static SpecializationCache<u32, ReenteringArtifact>, u64);

        impl Drop for ReenteringArtifact {
            fn drop(&mut self) {
                let _cache_length = self.0.len();
                REMOVAL_PATHS.fetch_or(self.1, Ordering::Relaxed);
            }
        }

        const EVICTION: u64 = 1 << 0;
        const INVALIDATION: u64 = 1 << 1;
        const CLEARING: u64 = 1 << 2;
        static REMOVAL_PATHS: AtomicU64 = AtomicU64::new(0);

        // The cache is leaked so that artifacts can name it for their whole lifetime, which is what lets a destructor
        // reenter it at all.
        let cache: &'static SpecializationCache<u32, ReenteringArtifact> =
            Box::leak(Box::new(SpecializationCache::new(1)));
        REMOVAL_PATHS.store(0, Ordering::Relaxed);

        // Eviction replaces a retained artifact, invalidation removes selected ones, and clearing removes all of them.
        let _eviction_artifact = expect_vacant_producer(cache.try_entry(1)).insert(ReenteringArtifact(cache, EVICTION));
        let _invalidation_artifact =
            expect_vacant_producer(cache.try_entry(2)).insert(ReenteringArtifact(cache, INVALIDATION));
        assert_eq!(REMOVAL_PATHS.load(Ordering::Relaxed), EVICTION);
        assert_eq!(cache.statistics().evictions, 1);

        assert_eq!(cache.invalidate_entries_if(|key| *key == 2), 1);
        assert_eq!(REMOVAL_PATHS.load(Ordering::Relaxed), EVICTION | INVALIDATION);
        assert!(cache.is_empty());

        let _clearing_artifact = expect_vacant_producer(cache.try_entry(3)).insert(ReenteringArtifact(cache, CLEARING));
        cache.clear();
        assert_eq!(REMOVAL_PATHS.load(Ordering::Relaxed), EVICTION | INVALIDATION | CLEARING);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_specialization_cache_producer_insert_publishes_before_dropping_displaced_artifact() {
        // Artifact whose destructor panics exactly once, after it has been armed. Arming it only once the artifact
        // is retained keeps every other copy's destructor harmless, and disarming inside the destructor keeps the
        // panic from firing again while the stack unwinds (which would abort the process).
        #[derive(Clone)]
        struct PanickingArtifact;

        impl Drop for PanickingArtifact {
            fn drop(&mut self) {
                if ARMED.swap(false, Ordering::Relaxed) {
                    panic!("displaced artifact destructor panicked");
                }
            }
        }

        // Whether the next destructor run must panic.
        static ARMED: AtomicBool = AtomicBool::new(false);

        let cache = SpecializationCache::<u32, PanickingArtifact>::new(1);
        expect_vacant_producer(cache.try_entry(1)).insert(PanickingArtifact);
        ARMED.store(true, Ordering::Relaxed);

        // Inserting a second key displaces the armed artifact, whose destructor unwinds out of the insertion.
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            expect_vacant_producer(cache.try_entry(2)).insert(PanickingArtifact);
        }));
        assert!(panicked.is_err());

        // Publication, the counter updates, and the marker release all happened before that destructor ran, so the
        // artifact is retained, the same thread may look its key up again instead of being rejected as reentrant, and
        // both the production and the eviction were counted.
        assert_eq!(cache.keys(), vec![2]);
        assert!(matches!(cache.try_entry(2), Ok(SpecializationCacheEntry::Occupied(_))));
        assert_eq!(
            cache.statistics(),
            SpecializationCacheStatistics {
                hits: 1,
                misses: 2,
                productions: 2,
                abandoned_productions: 0,
                evictions: 1
            },
        );
    }

    #[test]
    fn test_specialization_cache_removes_keys_outside_locks() {
        // Key whose destructor reenters the cache holding it, which deadlocks unless every owned key is dropped
        // after the guard that reached it has been released.
        #[derive(Clone, Debug)]
        struct ReenteringKey {
            value: u32,
            removal_path: u64,
            armed: Arc<AtomicBool>,
        }

        impl ReenteringKey {
            // Creates a key whose next clone to be dropped records `removal_path` and reenters the cache.
            fn armed(value: u32, removal_path: u64) -> (Self, Arc<AtomicBool>) {
                let armed = Arc::new(AtomicBool::new(false));
                (Self { value, removal_path, armed: armed.clone() }, armed)
            }

            // Creates a key whose destructor does not reenter the cache.
            fn unarmed(value: u32) -> Self {
                Self { value, removal_path: 0, armed: Arc::new(AtomicBool::new(false)) }
            }
        }

        impl PartialEq for ReenteringKey {
            fn eq(&self, other: &Self) -> bool {
                self.value == other.value
            }
        }

        impl Eq for ReenteringKey {}

        impl Hash for ReenteringKey {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.value.hash(state);
            }
        }

        impl Drop for ReenteringKey {
            fn drop(&mut self) {
                if !self.armed.swap(false, Ordering::Relaxed) {
                    return;
                }

                // `len` reenters the retained map's lock and the nested lookup additionally reenters the in-flight
                // lock, so a key dropped under either guard deadlocks here.
                let cache = CACHE.get().unwrap();
                let _cache_length = cache.len();
                drop(expect_vacant_producer(cache.try_entry(ReenteringKey::unarmed(u32::MAX))));
                REMOVAL_PATHS.fetch_or(self.removal_path, Ordering::Relaxed);
            }
        }

        // Cache the destructors above reenter. It is leaked so that keys can name it for their whole lifetime.
        static CACHE: OnceLock<&'static SpecializationCache<ReenteringKey, &'static str>> = OnceLock::new();
        static REMOVAL_PATHS: AtomicU64 = AtomicU64::new(0);

        const EVICTION: u64 = 1 << 0;
        const INVALIDATION: u64 = 1 << 1;
        const CLEARING: u64 = 1 << 2;
        const REJECTED_REGISTRATION: u64 = 1 << 3;
        const INSERTION_MARKER: u64 = 1 << 4;
        const ABANDONED_MARKER: u64 = 1 << 5;

        let cache: &'static SpecializationCache<ReenteringKey, &'static str> =
            Box::leak(Box::new(SpecializationCache::new(1)));
        CACHE.set(cache).unwrap();
        REMOVAL_PATHS.store(0, Ordering::Relaxed);

        // Retained keys are dropped through eviction, invalidation, and clearing.
        let (eviction_key, eviction_armed) = ReenteringKey::armed(1, EVICTION);
        expect_vacant_producer(cache.try_entry(eviction_key)).insert("one");
        eviction_armed.store(true, Ordering::Relaxed);

        let (invalidation_key, invalidation_armed) = ReenteringKey::armed(2, INVALIDATION);
        expect_vacant_producer(cache.try_entry(invalidation_key)).insert("two");
        assert_eq!(REMOVAL_PATHS.load(Ordering::Relaxed), EVICTION);
        invalidation_armed.store(true, Ordering::Relaxed);
        assert_eq!(cache.invalidate_entries_if(|key| key.value == 2), 1);
        assert_eq!(REMOVAL_PATHS.load(Ordering::Relaxed), EVICTION | INVALIDATION);

        let (clearing_key, clearing_armed) = ReenteringKey::armed(3, CLEARING);
        expect_vacant_producer(cache.try_entry(clearing_key)).insert("three");
        clearing_armed.store(true, Ordering::Relaxed);
        cache.clear();
        assert_eq!(REMOVAL_PATHS.load(Ordering::Relaxed), EVICTION | INVALIDATION | CLEARING);

        // In-flight keys are dropped after rejected registration, successful insertion, and abandoned production.
        let (rejected_key, rejected_armed) = ReenteringKey::armed(4, REJECTED_REGISTRATION);
        let rejected_producer = expect_vacant_producer(cache.try_entry(rejected_key));
        rejected_armed.store(true, Ordering::Relaxed);
        assert!(matches!(cache.try_entry(ReenteringKey::unarmed(4)), Err(ReentrantSpecializationError)));
        assert_eq!(REMOVAL_PATHS.load(Ordering::Relaxed), EVICTION | INVALIDATION | CLEARING | REJECTED_REGISTRATION,);
        drop(rejected_producer);

        let (insertion_key, insertion_armed) = ReenteringKey::armed(5, INSERTION_MARKER);
        let insertion_producer = expect_vacant_producer(cache.try_entry(insertion_key));
        insertion_armed.store(true, Ordering::Relaxed);
        insertion_producer.insert("five");
        assert_eq!(
            REMOVAL_PATHS.load(Ordering::Relaxed),
            EVICTION | INVALIDATION | CLEARING | REJECTED_REGISTRATION | INSERTION_MARKER,
        );

        let (abandoned_key, abandoned_armed) = ReenteringKey::armed(6, ABANDONED_MARKER);
        let abandoned_producer = expect_vacant_producer(cache.try_entry(abandoned_key));
        abandoned_armed.store(true, Ordering::Relaxed);
        drop(abandoned_producer);
        assert_eq!(
            REMOVAL_PATHS.load(Ordering::Relaxed),
            EVICTION | INVALIDATION | CLEARING | REJECTED_REGISTRATION | INSERTION_MARKER | ABANDONED_MARKER,
        );
    }

    #[test]
    fn test_specialization_cache_try_entry_allows_concurrent_same_key_production() {
        let cache = SpecializationCache::<u32, String>::new(4);
        let barrier = Barrier::new(2);
        thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    // The barrier pins that both lookups miss before either producer publishes, so this exercises
                    // genuine cross-thread duplicate production rather than a hit after a race.
                    let producer = expect_vacant_producer(cache.try_entry(1));
                    barrier.wait();
                    producer.insert("compiled".to_string());
                });
            }
        });

        // Duplicate production is allowed across threads and inserts are idempotent, so one entry remains. The second
        // insert replaces the first rather than evicting anything, which the zero eviction count below pins.
        assert_eq!(cache.len(), 1);
        assert_eq!(expect_occupied_artifact(cache.try_entry(1)), "compiled".to_string());
        assert_eq!(
            cache.statistics(),
            SpecializationCacheStatistics {
                hits: 1,
                misses: 2,
                productions: 2,
                abandoned_productions: 0,
                evictions: 0
            },
        );
    }

    #[test]
    fn test_specialization_cache_try_entry_serves_concurrent_hits() {
        let cache = SpecializationCache::<u32, String>::new(4);
        expect_vacant_producer(cache.try_entry(1)).insert("compiled".to_string());
        cache.clear_statistics();
        let barrier = Barrier::new(4);
        thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    barrier.wait();
                    assert_eq!(expect_occupied_artifact(cache.try_entry(1)), "compiled".to_string());
                });
            }
        });
        assert_eq!(cache.statistics(), SpecializationCacheStatistics { hits: 4, ..Default::default() });
    }

    #[test]
    fn test_specialization_cache_is_send_and_sync() {
        fn assert_send_and_sync<T: Send + Sync>() {}

        // Thread safety derives from the `Mutex`/`AtomicU64` state; there are no `unsafe impl`s in this module.
        assert_send_and_sync::<SpecializationCache<u32, &'static str>>();
        assert_send_and_sync::<SpecializationCacheStatistics>();
        assert_send_and_sync::<ReentrantSpecializationError>();

        // `SpecializationCacheProducer` is deliberately not `Send` or `Sync`, because its in-flight marker is keyed
        // by the thread that looked the key up. That negative bound is enforced by the guard's `PhantomData<*mut ()>`
        // marker and can only be observed as a compilation failure, which needs a compile-fail harness this crate
        // intentionally does not depend on, and so it is pinned here as documentation rather than as an assertion.
    }
}
