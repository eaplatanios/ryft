use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};

use ryft_core::arrays::ArrayType;

use crate::Array;

/// Maximum number of ready bound-shaped materializations retained by one logical [`Array`].
///
/// Each retained entry pins device buffers of the *bound* shape, so the worst-case device memory retained per logical
/// array is this many bound-shaped copies on top of the array's own storage. The capacity trades that retention
/// against re-padding cost when one array flows into executables compiled for several different bounds.
const BOUNDED_MATERIALIZATION_CACHE_CAPACITY: usize = 4;

/// Maximum logical extent scalars retained by one logical [`Array`].
pub(crate) const LOGICAL_EXTENT_SCALAR_CACHE_CAPACITY: usize = 16;

/// Version of the physical padding convention encoded by [`BoundedMaterializationKey`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum BoundedMaterializationVersion {
    /// Dense row-major source data is copied at the origin and every remaining physical element is zero.
    ZeroOriginV1,
}

/// Structural identity of one reusable bound-shaped physical materialization.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BoundedMaterializationKey {
    /// Physical padding convention used to produce the cached array.
    version: BoundedMaterializationVersion,

    /// Complete physical array type, including bound shape, sharding, layout, and memory metadata.
    physical_type: ArrayType,
}

impl BoundedMaterializationKey {
    /// Creates a key for the current zero-origin padding convention and `physical_type`.
    pub(crate) fn new(physical_type: ArrayType) -> Self {
        Self { version: BoundedMaterializationVersion::ZeroOriginV1, physical_type }
    }
}

/// One ready cache entry ordered from least to most recently used.
struct BoundedMaterializationEntry<'o> {
    /// Structural materialization identity.
    key: BoundedMaterializationKey,

    /// Ready bound-shaped array owning its device buffers.
    array: Array<'o>,
}

/// One generation of single-flight production for a structural materialization key.
struct BoundedMaterializationFlight<'o> {
    /// Monotonic value distinguishing this flight from a failed predecessor for the same key.
    generation: u64,

    /// Number of registered waiters that have not consumed or abandoned this flight.
    waiter_count: usize,

    /// Completed result pinned until every registered waiter has consumed or abandoned it.
    result: Option<Array<'o>>,
}

/// Mutable state protected by [`BoundedMaterializationCache::state`].
#[derive(Default)]
struct BoundedMaterializationCacheState<'o> {
    /// Ready entries in least-to-most-recently-used order.
    ready: VecDeque<BoundedMaterializationEntry<'o>>,

    /// Current production flight for each key, retained after completion only while registered waiters remain.
    flights: HashMap<BoundedMaterializationKey, BoundedMaterializationFlight<'o>>,

    /// Generation assigned to the next newly reserved flight.
    next_generation: u64,
}

/// Value-local cache of lazily produced bound-shaped device arrays.
///
/// The cache is shared by every clone of a logical [`Array`]. Ready entries are capped at
/// [`BOUNDED_MATERIALIZATION_CACHE_CAPACITY`] and evicted in least-recently-used order. An in-flight key elects exactly
/// one producer; waiters sleep without spinning. Completed flight results remain pinned outside the capacity-bounded
/// ready LRU until every waiter registered for that generation consumes or abandons the result. Producer failure
/// removes the reservation and wakes waiters, allowing a later caller to retry rather than retaining a failed entry.
///
/// Each key's flight moves through these states under the [`Self::state`] mutex, with [`Self::ready`] signaled on
/// every transition that can unblock a waiter:
///
///   - *vacant*: no flight exists; the next probe reserves one and becomes the producer.
///   - *producing*: a flight exists with no result; probes register as waiters and sleep.
///   - *completed-pinned*: the flight holds its result until the last registered waiter consumes or abandons it,
///     independently of ready-LRU eviction; new probes hit without registering.
///   - *failed*: the producer dropped without completing, removing the flight; woken waiters race to re-reserve, one
///     becomes the retry producer, and the rest re-register against the new generation.
pub(crate) struct BoundedMaterializationCache<'o> {
    /// Ready entries and in-flight reservations.
    state: Mutex<BoundedMaterializationCacheState<'o>>,

    /// Notification for a key becoming ready or a failed reservation becoming retryable.
    ready: Condvar,
}

impl<'o> Default for BoundedMaterializationCache<'o> {
    fn default() -> Self {
        Self { state: Mutex::new(BoundedMaterializationCacheState::default()), ready: Condvar::new() }
    }
}

impl<'o> BoundedMaterializationCache<'o> {
    /// Looks up `key` without blocking, reserving a missing key for this caller when possible.
    pub(crate) fn probe(self: &Arc<Self>, key: BoundedMaterializationKey) -> BoundedMaterializationProbe<'o> {
        let mut state = self.state.lock().expect("bounded materialization cache mutex poisoned");
        if let Some(array) = Self::take_ready(&mut state, &key) {
            return BoundedMaterializationProbe::Hit(array);
        }
        if let Some(flight) = state.flights.get_mut(&key) {
            if let Some(array) = &flight.result {
                // Reaching a pinned flight result means the ready entry inserted at completion was since evicted:
                // `take_ready` above would have served it otherwise. The result is served without reviving the ready
                // entry so waiter pinning does not override the LRU eviction decision.
                return BoundedMaterializationProbe::Hit(array.clone());
            }
            flight.waiter_count += 1;
            return BoundedMaterializationProbe::Wait(BoundedMaterializationWaiter {
                cache: Arc::clone(self),
                key,
                generation: Some(flight.generation),
            });
        }
        let generation = Self::reserve_flight(&mut state, key.clone());
        BoundedMaterializationProbe::Produce(BoundedMaterializationProducer {
            cache: Arc::clone(self),
            key,
            generation,
            completed: false,
        })
    }

    /// Returns and refreshes a ready entry while holding `state`.
    fn take_ready(
        state: &mut BoundedMaterializationCacheState<'o>,
        key: &BoundedMaterializationKey,
    ) -> Option<Array<'o>> {
        let index = state.ready.iter().position(|entry| &entry.key == key)?;
        let entry = state.ready.remove(index).unwrap();
        let array = entry.array.clone();
        state.ready.push_back(entry);
        Some(array)
    }

    /// Reserves a new producer generation for `key` while holding `state`.
    fn reserve_flight(state: &mut BoundedMaterializationCacheState<'o>, key: BoundedMaterializationKey) -> u64 {
        let generation = state.next_generation;
        state.next_generation = state.next_generation.checked_add(1).unwrap();
        let previous = state
            .flights
            .insert(key, BoundedMaterializationFlight { generation, waiter_count: 0, result: None });
        assert!(previous.is_none(), "new bounded materialization flight key must be vacant");
        generation
    }

    /// Inserts or refreshes one ready entry and evicts excess least-recently-used entries.
    fn insert_ready(
        state: &mut BoundedMaterializationCacheState<'o>,
        key: BoundedMaterializationKey,
        array: Array<'o>,
    ) {
        if let Some(index) = state.ready.iter().position(|entry| entry.key == key) {
            state.ready.remove(index);
        }
        state.ready.push_back(BoundedMaterializationEntry { key, array });
        while state.ready.len() > BOUNDED_MATERIALIZATION_CACHE_CAPACITY {
            state.ready.pop_front();
        }
    }
}

/// Result of a non-blocking lookup in a [`BoundedMaterializationCache`].
pub(crate) enum BoundedMaterializationProbe<'o> {
    /// A ready bound-shaped array was found.
    Hit(Array<'o>),

    /// This caller owns the missing key's single-flight reservation.
    Produce(BoundedMaterializationProducer<'o>),

    /// Another caller currently owns the key's single-flight reservation.
    Wait(BoundedMaterializationWaiter<'o>),
}

/// Single-flight producer reservation for one bounded materialization.
pub(crate) struct BoundedMaterializationProducer<'o> {
    /// Cache that owns the reservation.
    cache: Arc<BoundedMaterializationCache<'o>>,

    /// Reserved structural key.
    key: BoundedMaterializationKey,

    /// Generation of the reserved flight.
    generation: u64,

    /// Whether [`Self::complete`] installed a ready value.
    completed: bool,
}

impl<'o> BoundedMaterializationProducer<'o> {
    /// Installs ready `array`, evicts the least-recently-used excess entry, and wakes all waiters.
    ///
    /// The caller must establish that every asynchronous operation producing `array` completed successfully before
    /// calling this method. Registered waiters pin this flight result independently of ordinary LRU eviction.
    pub(crate) fn complete(mut self, array: Array<'o>) -> Array<'o> {
        let returned = array.clone();
        let mut state = self.cache.state.lock().expect("bounded materialization cache mutex poisoned");
        let flight = state
            .flights
            .get_mut(&self.key)
            .filter(|flight| flight.generation == self.generation)
            .expect("bounded materialization producer must own its flight generation");
        if flight.waiter_count == 0 {
            state.flights.remove(&self.key);
        } else {
            flight.result = Some(array.clone());
        }
        BoundedMaterializationCache::insert_ready(&mut state, self.key.clone(), array);
        self.completed = true;
        drop(state);
        self.cache.ready.notify_all();
        returned
    }
}

impl Drop for BoundedMaterializationProducer<'_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        // This drop runs during unwinding when materialization fails, so recover a poisoned mutex instead of
        // panicking: a second panic here would abort the process before the failure can be reported.
        let mut state = self.cache.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let owns_flight = state
            .flights
            .get(&self.key)
            .is_some_and(|flight| flight.generation == self.generation && flight.result.is_none());
        if owns_flight {
            state.flights.remove(&self.key);
        }
        drop(state);
        self.cache.ready.notify_all();
    }
}

/// Wait handle for an identical materialization currently produced by another caller.
pub(crate) struct BoundedMaterializationWaiter<'o> {
    /// Cache containing the in-flight reservation.
    cache: Arc<BoundedMaterializationCache<'o>>,

    /// Structural key being awaited.
    key: BoundedMaterializationKey,

    /// Flight generation against which this waiter is currently registered.
    generation: Option<u64>,
}

impl<'o> BoundedMaterializationWaiter<'o> {
    /// Waits for a ready entry, or becomes the retry producer after the previous producer fails.
    pub(crate) fn resolve(mut self) -> Result<Array<'o>, BoundedMaterializationProducer<'o>> {
        let mut state = self.cache.state.lock().expect("bounded materialization cache mutex poisoned");
        loop {
            let registered_generation = self.generation.unwrap();
            if let Some(flight) = state.flights.get_mut(&self.key) {
                if flight.generation == registered_generation {
                    if let Some(array) = &flight.result {
                        let array = array.clone();
                        flight.waiter_count = flight.waiter_count.checked_sub(1).unwrap();
                        let remove_flight = flight.waiter_count == 0;
                        self.generation = None;
                        if remove_flight {
                            state.flights.remove(&self.key);
                        }
                        return Ok(array);
                    }
                    state = self.cache.ready.wait(state).expect("bounded materialization cache mutex poisoned");
                    continue;
                }
                if let Some(array) = &flight.result {
                    self.generation = None;
                    return Ok(array.clone());
                }
                flight.waiter_count += 1;
                self.generation = Some(flight.generation);
                continue;
            }
            if let Some(array) = BoundedMaterializationCache::take_ready(&mut state, &self.key) {
                self.generation = None;
                return Ok(array);
            }
            let generation = BoundedMaterializationCache::reserve_flight(&mut state, self.key.clone());
            self.generation = None;
            return Err(BoundedMaterializationProducer {
                cache: Arc::clone(&self.cache),
                key: self.key.clone(),
                generation,
                completed: false,
            });
        }
    }
}

impl Drop for BoundedMaterializationWaiter<'_> {
    fn drop(&mut self) {
        let Some(generation) = self.generation else {
            return;
        };
        // This drop can run during unwinding, so recover a poisoned mutex instead of panicking: a second panic here
        // would abort the process before the original failure can be reported.
        let mut state = self.cache.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut remove_flight = false;
        if let Some(flight) = state.flights.get_mut(&self.key)
            && flight.generation == generation
        {
            flight.waiter_count = flight.waiter_count.checked_sub(1).unwrap();
            remove_flight = flight.waiter_count == 0 && flight.result.is_some();
        }
        if remove_flight {
            state.flights.remove(&self.key);
        }
    }
}
