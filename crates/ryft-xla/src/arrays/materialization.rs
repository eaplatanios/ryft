//! Padded copies of an [`Array`] that bounded dynamic executables consume, with the cache that retains them.
//!
//! An executable compiled for bounded dynamic shapes takes every such input with its static bound shape, followed by
//! a scalar argument for each logical extent. [`materialize_bounded_dynamic_inputs`] produces those physical inputs.
//! It reuses every input whose shape equals its bound, and it pads every smaller input through the host and retains
//! the padded copy in the [`BoundedMaterializationCache`] of the source array, which every clone of that array shares.
//! Concurrent requests for the same [`BoundedMaterializationKey`] wait for a single producer.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};

use ryft_core::{ArrayType, DataType, ProgramError, StaticShape};

use crate::arrays::host::DenseArrayHostCopy;
use crate::arrays::{Array, ArrayTypeExtension};
use crate::lowering::signature::ExecutableSignature;
use crate::{Domain, Error, ToPjrt};

// TODO(eaplatanios): Review this module.

/// Maximum number of ready materializations with bound shapes retained by one logical [`Array`].
///
/// Each retained entry pins device buffers of the *bound* shape, so in the worst case, the device memory retained per
/// logical array is this many copies with bound shapes on top of the array's own storage. The capacity trades that
/// retention against the cost of padding again when one array flows into executables compiled for several different
/// bounds.
const BOUNDED_MATERIALIZATION_CACHE_CAPACITY: usize = 4;

/// Maximum number of logical extent scalars retained by one logical [`Array`] (refer to
/// [`Array::logical_extent_scalar`]).
pub(crate) const LOGICAL_EXTENT_SCALAR_CACHE_CAPACITY: usize = 16;

/// Structural identity of one reusable physical materialization with a bound shape.
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

/// Value-local cache of lazily produced device arrays with bound shapes.
///
/// The cache is shared by every clone of a logical [`Array`]. Ready entries are capped at
/// [`BOUNDED_MATERIALIZATION_CACHE_CAPACITY`] and evicted in least recently used order. An in-flight key elects exactly
/// one producer, and waiters sleep without spinning. Completed flight results remain pinned outside the ready LRU,
/// whose capacity is bounded, until every waiter registered for that generation consumes or abandons the result. A
/// producer failure removes the reservation and wakes the waiters, which lets a later caller retry rather than
/// retaining a failed entry.
///
/// The flight of each key moves through the following states under the [`Self::state`] mutex, and [`Self::ready`] is
/// signaled on every transition that can unblock a waiter.
///
///   - *Vacant:* No flight exists, and the next probe reserves one and becomes the producer.
///   - *Producing:* A flight exists with no result, and probes register as waiters and sleep.
///   - *Completed and pinned:* The flight holds its result until the last registered waiter consumes or abandons it,
///     independently of evictions from the ready LRU, and new probes hit without registering.
///   - *Failed:* The producer was dropped without completing, which removed the flight. The woken waiters race to
///     reserve it again, one of them becomes the retry producer, and the rest register again against the new
///     generation.
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
                // Reaching a pinned flight result means that the ready entry inserted at completion was since evicted,
                // because `take_ready` above would have served it otherwise. The result is served without reviving the
                // ready entry so that waiter pinning does not override the LRU eviction decision.
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

    /// Inserts or refreshes one ready entry and evicts the excess least recently used entries.
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
    /// A ready array with the bound shape was found.
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
    /// Installs the ready `array`, evicts the least recently used excess entry, and wakes all waiters.
    ///
    /// The caller must establish that every asynchronous operation producing `array` completed successfully before
    /// calling this function. Registered waiters pin this flight result independently of ordinary LRU eviction.
    pub(crate) fn complete(mut self, array: Array<'o>) -> Array<'o> {
        let returned = array.clone();
        let mut state = self.cache.state.lock().expect("bounded materialization cache mutex poisoned");
        let flight = state.flights.get_mut(&self.key).filter(|flight| flight.generation == self.generation).unwrap();
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
        // panicking, because a second panic here would abort the process before the failure can be reported.
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
        // This drop can run during unwinding, so recover a poisoned mutex instead of panicking, because a second panic
        // here would abort the process before the original failure can be reported.
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

/// Physical inputs and the deterministic report of the bounded materialization work that produced them.
pub(crate) struct MaterializedBoundedInputs<'c> {
    /// Physical executable inputs, including hidden extent scalars.
    inputs: Vec<Array<'c>>,

    /// Work and transport counts of the materialization that produced [`Self::inputs`].
    #[cfg_attr(not(test), allow(dead_code))]
    report: BoundedInputMaterializationReport,
}

impl<'c> MaterializedBoundedInputs<'c> {
    /// Returns the physical executable inputs, including the hidden extent scalars.
    pub(crate) fn into_inputs(self) -> Vec<Array<'c>> {
        self.inputs
    }
}

/// Pads bounded-dynamic inputs to static executable storage and appends their logical extents.
///
/// Every input takes one of two paths. An input whose actual shape equals its physical bound reuses the source array.
/// A smaller input uses a retained materialization that the clones of the source array share, with one single-flight
/// producer per structural [`BoundedMaterializationKey`] (i.e., the zero-origin padding convention and the physical
/// [`ArrayType`]). Zero-space inputs below their bound are deferred to the zero-space carrier materialization of the
/// execution boundary, which synthesizes their carriers with bound shapes directly. Other inputs below their bound with
/// non-addressable shards are rejected, because padding must read the bytes of every shard on this process.
///
/// Every device-to-host copy of a producer is issued during the first scan. The resulting host-to-device uploads are
/// also all issued before waiting for the readiness of any upload, and retained values are published only after the
/// complete batch succeeds. A cache miss costs O(bound) bytes plus overlap bookkeeping that is linear in the number of
/// shard segments, while a ready cache hit performs no allocation or transport for padding. A failed asynchronous
/// upload stores no cache entry, and a later call retries the ordinary path for cache misses.
pub(crate) fn materialize_bounded_dynamic_inputs<'c>(
    domain: &Domain<'c>,
    signature: &ExecutableSignature,
    declared_types: &[ArrayType],
    actual_types: &[ArrayType],
    inputs: Vec<Array<'c>>,
) -> Result<MaterializedBoundedInputs<'c>, Error> {
    materialize_bounded_dynamic_inputs_with_readiness(
        domain,
        signature,
        declared_types,
        actual_types,
        inputs,
        |publications| {
            for publication in publications {
                publication.physical.block_until_ready()?;
            }
            Ok(())
        },
    )
}

/// Version of the physical padding convention encoded by [`BoundedMaterializationKey`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum BoundedMaterializationVersion {
    /// Dense row-major source data is copied at the origin and every remaining physical element is zero.
    ZeroOriginV1,
}

/// One ready cache entry ordered from least to most recently used.
struct BoundedMaterializationEntry<'o> {
    /// Structural materialization identity.
    key: BoundedMaterializationKey,

    /// Ready array with the bound shape, which owns its device buffers.
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
    /// Ready entries, ordered from least to most recently used.
    ready: VecDeque<BoundedMaterializationEntry<'o>>,

    /// Current production flight for each key, retained after completion only while registered waiters remain.
    flights: HashMap<BoundedMaterializationKey, BoundedMaterializationFlight<'o>>,

    /// Generation assigned to the next newly reserved flight.
    next_generation: u64,
}

/// Host copy that has been issued for one bounded input requiring padding.
struct PendingBoundedInputHostCopy<'c> {
    /// Logical input index used to recover the runtime array metadata.
    logical_input_index: usize,

    /// Physical executable input slot to populate.
    physical_input_index: usize,

    /// Static executable type defining the padding target.
    physical_type: ArrayType,

    /// Concrete logical shape copied from the runtime input.
    actual_shape: StaticShape,

    /// Static bound shape of the executable input, which is used for physical storage.
    physical_shape: StaticShape,

    /// Host copy of the logical input, which has already been issued.
    host_copy: DenseArrayHostCopy,

    /// Single-flight reservation that retains success.
    producer: BoundedMaterializationProducer<'c>,
}

/// Upload with the bound shape that was issued for one padded input but not yet published to its cache or executable
/// slot.
struct PendingBoundedInputPublication<'c> {
    /// Physical executable input slot to populate after readiness succeeds.
    physical_input_index: usize,

    /// Uploaded physical array whose asynchronous transfer must become ready before publication.
    physical: Array<'c>,

    /// Single-flight reservation that retains ready success.
    producer: BoundedMaterializationProducer<'c>,
}

/// Cache waiter retained until all producer copies in this call have completed.
struct WaitingBoundedInputMaterialization<'c> {
    /// Logical input index used to retry production if the prior producer fails.
    logical_input_index: usize,

    /// Physical executable input slot to populate.
    physical_input_index: usize,

    /// Static executable type defining the cache key and padding target.
    physical_type: ArrayType,

    /// Concrete logical shape copied from the runtime input on a retry.
    actual_shape: StaticShape,

    /// Static bound shape of the executable input, which is used for physical storage on a retry.
    physical_shape: StaticShape,

    /// Non-spinning single-flight wait handle.
    waiter: BoundedMaterializationWaiter<'c>,
}

/// Deterministic work and transport report for one materialization of bounded inputs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BoundedInputMaterializationReport {
    /// Inputs that reused original device buffers because the runtime shape equaled the bound.
    at_bound_reuses: usize,

    /// Ready retained materializations reused without padding work or transport.
    cache_hits: usize,

    /// Missing cache entries whose successful production was retained.
    retained_misses: usize,

    /// Device-to-host shard copies issued by padding work.
    device_to_host_shard_copies: usize,

    /// Host merge buffers for whole global arrays allocated after shard copies.
    host_merge_buffer_allocations: usize,

    /// O(bound) host padding payloads allocated.
    host_padding_payload_allocations: usize,

    /// Host-to-device shard uploads with bound shapes.
    host_to_device_shard_uploads: usize,

    /// Host-to-device uploads of hidden logical extent scalars.
    extent_scalar_uploads: usize,

    /// Sum of logical payload bytes for bounded inputs.
    actual_bytes: usize,

    /// Sum of the payload bytes of the physical bounds of bounded inputs.
    bound_bytes: usize,
}

/// Selects whether a bounded input can reuse its device buffer or requires host padding.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BoundedInputPacking {
    /// The runtime shape already equals the executable's physical bound.
    Reuse,

    /// At least one runtime extent is smaller than its executable bound.
    Pad,
}

impl BoundedInputPacking {
    /// Determines the packing of a bounded input from its concrete runtime shape and its executable shape.
    fn from_shapes(actual_shape: &StaticShape, physical_shape: &StaticShape) -> Self {
        if actual_shape == physical_shape { Self::Reuse } else { Self::Pad }
    }
}

/// Completes one host copy that was already issued and issues its upload with the bound shape without publishing it.
fn upload_bounded_input_host_copy<'c>(
    domain: &Domain<'c>,
    input: &Array<'c>,
    pending: PendingBoundedInputHostCopy<'c>,
    report: &mut BoundedInputMaterializationReport,
) -> Result<PendingBoundedInputPublication<'c>, Error> {
    report.device_to_host_shard_copies += pending.host_copy.shard_copy_count();
    let (source_bytes, allocated_merge_buffer) = pending.host_copy.finish_with_measurements()?;
    report.host_merge_buffer_allocations += usize::from(allocated_merge_buffer);
    let (padded_bytes, padding_payload_allocations) = pad_dense_array_bytes_with_target_allocation_count(
        source_bytes.as_slice(),
        pending.actual_shape.as_slice(),
        pending.physical_shape.as_slice(),
        pending.physical_type.data_type(),
    )?;
    report.host_padding_payload_allocations += padding_payload_allocations;
    let physical = Array::from_host_buffer(domain, pending.physical_type, input.mesh().clone(), padded_bytes)?;
    report.host_to_device_shard_uploads += physical.shards().iter().filter(|shard| shard.buffer().is_some()).count();
    Ok(PendingBoundedInputPublication {
        physical_input_index: pending.physical_input_index,
        physical,
        producer: pending.producer,
    })
}

/// Publishes one batch only after every asynchronously uploaded physical array passes `check_readiness`.
fn publish_bounded_input_uploads<'c>(
    publications: Vec<PendingBoundedInputPublication<'c>>,
    physical_inputs: &mut [Array<'c>],
    check_readiness: &mut impl FnMut(&[PendingBoundedInputPublication<'c>]) -> Result<(), Error>,
) -> Result<(), Error> {
    // Readiness is checked for the complete batch before any producer publishes. An error therefore drops every
    // reservation in the batch, caches no failed upload, and leaves all concurrent host-to-device work issued.
    check_readiness(publications.as_slice())?;
    for publication in publications {
        physical_inputs[publication.physical_input_index] = publication.producer.complete(publication.physical);
    }
    Ok(())
}

/// Implements [`materialize_bounded_dynamic_inputs`] with an injectable batch readiness check for deterministic failure
/// tests.
fn materialize_bounded_dynamic_inputs_with_readiness<'c>(
    domain: &Domain<'c>,
    signature: &ExecutableSignature,
    declared_types: &[ArrayType],
    actual_types: &[ArrayType],
    inputs: Vec<Array<'c>>,
    mut check_readiness: impl FnMut(&[PendingBoundedInputPublication<'c>]) -> Result<(), Error>,
) -> Result<MaterializedBoundedInputs<'c>, Error> {
    assert_eq!(declared_types.len(), inputs.len());
    assert_eq!(actual_types.len(), inputs.len());
    let physical_types = signature.physical_input_types(declared_types);
    let mut physical_inputs = signature.project_inputs(inputs.as_slice());
    let mut pending_host_copies = Vec::new();
    let mut waiting_materializations = Vec::new();
    let mut report = BoundedInputMaterializationReport::default();
    for logical_input_index in 0..inputs.len() {
        if !signature
            .input_dimensions()
            .iter()
            .any(|input_dimension| input_dimension.logical_input_index() == logical_input_index)
        {
            continue;
        }
        let physical_input_index = signature.input_mapping()[logical_input_index].unwrap();
        let physical_type = physical_types[physical_input_index].clone();
        let physical_shape = physical_type.static_shape().ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("bounded-dynamic executable input {logical_input_index} has no static upper shape"),
        })?;
        let actual_shape =
            actual_types[logical_input_index].static_shape().ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("runtime executable input {logical_input_index} must have a static shape"),
            })?;
        report.actual_bytes += actual_types[logical_input_index].size_in_bytes()?;
        report.bound_bytes += physical_type.size_in_bytes()?;
        if BoundedInputPacking::from_shapes(&actual_shape, &physical_shape) == BoundedInputPacking::Reuse {
            report.at_bound_reuses += 1;
            continue;
        }

        // A zero-space input below its bound carries no payload bytes to pad or cache. Its dense host copy is empty,
        // so the retained cache and byte padding below must not run. The zero-space carrier materialization that
        // follows this boundary synthesizes its all-false predicate with the bound shape directly instead.
        if physical_type.data_type().is_zero() {
            continue;
        }

        let input = &inputs[logical_input_index];
        if input.supports_bounded_materialization_cache() {
            match input.probe_bounded_materialization(BoundedMaterializationKey::new(physical_type.clone())) {
                BoundedMaterializationProbe::Hit(physical) => {
                    report.cache_hits += 1;
                    physical_inputs[physical_input_index] = physical;
                }
                BoundedMaterializationProbe::Produce(producer) => {
                    input.validate_host_round_trip("pad")?;
                    report.retained_misses += 1;
                    pending_host_copies.push(PendingBoundedInputHostCopy {
                        logical_input_index,
                        physical_input_index,
                        physical_type,
                        actual_shape,
                        physical_shape,
                        host_copy: input.begin_copy_to_host()?,
                        producer,
                    });
                }
                BoundedMaterializationProbe::Wait(waiter) => {
                    waiting_materializations.push(WaitingBoundedInputMaterialization {
                        logical_input_index,
                        physical_input_index,
                        physical_type,
                        actual_shape,
                        physical_shape,
                        waiter,
                    });
                }
            }
        } else {
            // Padding reads the bytes of every shard on this process, so an input with a non-addressable shard cannot
            // be materialized here at all. Reject it explicitly instead of failing deep inside the host copy.
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "bounded-dynamic executable input {logical_input_index} is below its bound but has \
                     non-addressable shards, so it cannot be padded on this process",
                ),
            }
            .into());
        }
    }

    let mut pending_publications = Vec::with_capacity(pending_host_copies.len());
    for pending in pending_host_copies {
        let input = &inputs[pending.logical_input_index];
        pending_publications.push(upload_bounded_input_host_copy(domain, input, pending, &mut report)?);
    }
    publish_bounded_input_uploads(pending_publications, &mut physical_inputs, &mut check_readiness)?;

    for waiting in waiting_materializations {
        let WaitingBoundedInputMaterialization {
            logical_input_index,
            physical_input_index,
            physical_type,
            actual_shape,
            physical_shape,
            waiter,
        } = waiting;
        match waiter.resolve() {
            Ok(physical) => {
                report.cache_hits += 1;
                physical_inputs[physical_input_index] = physical;
            }
            Err(producer) => {
                // The prior producer failed and removed its reservation, so this waiter is now the retry producer. Its
                // copy necessarily starts after that failure, while the ordinary path above, in which every input is a
                // cache miss, stays fully concurrent.
                report.retained_misses += 1;
                let input = &inputs[logical_input_index];
                input.validate_host_round_trip("pad")?;
                let pending = PendingBoundedInputHostCopy {
                    logical_input_index,
                    physical_input_index,
                    physical_type,
                    actual_shape,
                    physical_shape,
                    host_copy: input.begin_copy_to_host()?,
                    producer,
                };
                let publication = upload_bounded_input_host_copy(domain, input, pending, &mut report)?;
                publish_bounded_input_uploads(vec![publication], &mut physical_inputs, &mut check_readiness)?;
            }
        }
    }

    for input_dimension in signature.input_dimensions() {
        let input = &inputs[input_dimension.logical_input_index()];
        let extent = actual_types[input_dimension.logical_input_index()].shape().dimensions()[input_dimension.axis()]
            .value()
            .unwrap();
        let extent = i32::try_from(extent).map_err(|_| ProgramError::InvalidArgument {
            message: format!("runtime dimension extent {extent} does not fit in a StableHLO `i32` scalar"),
        })?;
        let (extent_scalar, uploaded) = input.logical_extent_scalar(domain, extent)?;
        report.extent_scalar_uploads += usize::from(uploaded);
        physical_inputs.push(extent_scalar);
    }
    assert_eq!(physical_inputs.len(), signature.array_input_count());
    Ok(MaterializedBoundedInputs { inputs: physical_inputs, report })
}

/// Zero-pads the dense row-major `source` array of shape `source_shape` and element type `data_type` into
/// `target_shape`, placing the source at the origin, and reports the number of target payload allocations.
///
/// The count deliberately covers the O(bound) payload allocation, which dominates this path. Small O(rank) stride
/// vectors are implementation metadata and are not included.
///
/// # Errors
///
/// Returns [`ProgramError::InvalidArgument`] when the shapes have different ranks, when `source_shape` exceeds
/// `target_shape` along some axis, or when a stride, element count, or byte count overflows [`usize`].
fn pad_dense_array_bytes_with_target_allocation_count(
    source: &[u8],
    source_shape: &[usize],
    target_shape: &[usize],
    data_type: DataType,
) -> Result<(Vec<u8>, usize), Error> {
    if source_shape.len() != target_shape.len()
        || source_shape.iter().zip(target_shape).any(|(source, target)| source > target)
    {
        return Err(ProgramError::InvalidArgument {
            message: format!(
                "cannot pad runtime shape `{:?}` into bounded physical shape `{:?}`",
                source_shape, target_shape,
            ),
        }
        .into());
    }
    let element_size = data_type.to_pjrt().element_size_in_bytes()?;
    let mut source_strides = vec![1usize; source_shape.len()];
    let mut target_strides = vec![1usize; target_shape.len()];
    for axis in (0..source_shape.len().saturating_sub(1)).rev() {
        source_strides[axis] = source_strides[axis + 1].checked_mul(source_shape[axis + 1]).ok_or_else(|| {
            ProgramError::InvalidArgument { message: "runtime input shape strides exceed `usize`".to_string() }
        })?;
        target_strides[axis] = target_strides[axis + 1].checked_mul(target_shape[axis + 1]).ok_or_else(|| {
            ProgramError::InvalidArgument { message: "bounded physical shape strides exceed `usize`".to_string() }
        })?;
    }
    let target_elements = target_shape.iter().try_fold(1usize, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| ProgramError::InvalidArgument {
            message: "bounded physical input element count exceeds `usize`".to_string(),
        })
    })?;
    let mut target = vec![
        0u8;
        target_elements.checked_mul(element_size).ok_or_else(|| ProgramError::InvalidArgument {
            message: "bounded physical input byte count exceeds `usize`".to_string(),
        })?
    ];
    if source_shape.is_empty() {
        target.copy_from_slice(source);
        let allocation_count = usize::from(!target.is_empty());
        return Ok((target, allocation_count));
    }
    copy_dense_array_into_padding(
        source,
        &mut target,
        source_shape,
        source_strides.as_slice(),
        target_strides.as_slice(),
        element_size,
        0,
        0,
        0,
    );
    let allocation_count = usize::from(!target.is_empty());
    Ok((target, allocation_count))
}

/// Recursively copies one dense row-major source into the region of a padded target that starts at its origin, from
/// axis `axis` onward and starting at the element offsets `source_offset` and `target_offset`.
#[allow(clippy::too_many_arguments)]
fn copy_dense_array_into_padding(
    source: &[u8],
    target: &mut [u8],
    source_shape: &[usize],
    source_strides: &[usize],
    target_strides: &[usize],
    element_size: usize,
    axis: usize,
    source_offset: usize,
    target_offset: usize,
) {
    if axis + 1 == source_shape.len() {
        let byte_count = source_shape[axis] * element_size;
        let source_offset = source_offset * element_size;
        let target_offset = target_offset * element_size;
        target[target_offset..target_offset + byte_count]
            .copy_from_slice(&source[source_offset..source_offset + byte_count]);
        return;
    }
    for index in 0..source_shape[axis] {
        copy_dense_array_into_padding(
            source,
            target,
            source_shape,
            source_strides,
            target_strides,
            element_size,
            axis + 1,
            source_offset + index * source_strides[axis],
            target_offset + index * target_strides[axis],
        );
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Device, DeviceMesh, Dimension as CoreDimension, DimensionBounds, DimensionVariable, Error as CoreError,
        LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension,
    };
    use ryft_pjrt::BufferType;

    use crate::domains::tests::{domain_mesh, replicated_vector_type};
    use crate::execution::ExecutionArguments;
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{FromPjrt, Session};

    use super::*;

    #[test]
    fn test_materialize_bounded_dynamic_inputs() {
        let client = cpu_client(1);
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        // An input already at its bound is reused, so the physical payload preserves storage identity and the only new
        // value is its hidden logical extent scalar.
        let exact_type = replicated_vector_type(&mesh, 8);
        let exact = Array::from_host_buffer(
            &domain,
            exact_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[0.0; 8]).as_slice(),
        )
        .unwrap();
        assert_eq!(
            BoundedInputPacking::from_shapes(&exact_type.static_shape().unwrap(), &StaticShape::new(vec![8])),
            BoundedInputPacking::Reuse,
        );
        let exact_storage = exact.clone();
        let at_bound = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&exact_type),
            vec![exact],
        )
        .unwrap();
        assert_eq!(at_bound.inputs.len(), 2);
        assert_eq!(at_bound.inputs[0], exact_storage);
        assert_eq!(
            at_bound.report.clone(),
            BoundedInputMaterializationReport {
                at_bound_reuses: 1,
                extent_scalar_uploads: 1,
                actual_bytes: 32,
                bound_bytes: 32,
                ..Default::default()
            },
        );
        assert_eq!(at_bound.into_inputs()[0], exact_storage);

        // A smaller input with a single shard pays the padding cost of a cache miss exactly once and retains the
        // physical buffer. The second call reuses identical PJRT buffer `Arc`s and reports no allocation or transport
        // for padding. Explicit counts in the report of each call avoid the allocator noise of the whole process that
        // PJRT and Rust's parallel test runner would add.
        let smaller_type = replicated_vector_type(&mesh, 4);
        assert_eq!(
            BoundedInputPacking::from_shapes(&smaller_type.static_shape().unwrap(), &StaticShape::new(vec![8])),
            BoundedInputPacking::Pad,
        );
        let smaller = Array::from_host_buffer(
            &domain,
            smaller_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let cold = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&smaller_type),
            vec![smaller.clone()],
        )
        .unwrap();
        assert_eq!(
            cold.report,
            BoundedInputMaterializationReport {
                retained_misses: 1,
                device_to_host_shard_copies: 1,
                host_padding_payload_allocations: 1,
                host_to_device_shard_uploads: 1,
                extent_scalar_uploads: 1,
                actual_bytes: 16,
                bound_bytes: 32,
                ..Default::default()
            },
        );
        let retained_storage = cold.inputs[0].clone();
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let donation_arguments =
            ExecutionArguments::from_arrays_with_donation(vec![cold.inputs[0].clone()], [device_id].into(), &[true])
                .unwrap();
        assert!(!donation_arguments.inputs_by_device()[0][0].donatable);
        let warm =
            materialize_bounded_dynamic_inputs(&domain, &signature, &[declared_type], &[smaller_type], vec![smaller])
                .unwrap();
        assert_eq!(warm.inputs[0], retained_storage);
        assert_eq!(
            warm.report,
            BoundedInputMaterializationReport {
                cache_hits: 1,
                extent_scalar_uploads: 0,
                actual_bytes: 16,
                bound_bytes: 32,
                ..Default::default()
            },
        );
    }

    #[test]
    fn test_materialize_bounded_dynamic_inputs_independent_misses_share_one_issue_phase() {
        let client = cpu_client(1);
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let actual_type = replicated_vector_type(&mesh, 4);
        let left = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let right = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[5.0, 6.0, 7.0, 8.0]).as_slice(),
        )
        .unwrap();
        let declared_types = [declared_type.clone(), declared_type];
        let actual_types = [actual_type.clone(), actual_type];
        let signature = ExecutableSignature::new(&declared_types, &[]);

        // The first scan of the materialization reserves both cache entries and issues both device-to-host copies
        // before its completion loop awaits either one. The report of each call guards the two independent cache
        // misses and the subsequent cache hits, which do no work, without relying on scheduler timing.
        let cold = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            &declared_types,
            &actual_types,
            vec![left.clone(), right.clone()],
        )
        .unwrap();
        assert_eq!(
            cold.report,
            BoundedInputMaterializationReport {
                retained_misses: 2,
                device_to_host_shard_copies: 2,
                host_padding_payload_allocations: 2,
                host_to_device_shard_uploads: 2,
                extent_scalar_uploads: 2,
                actual_bytes: 32,
                bound_bytes: 64,
                ..Default::default()
            },
        );
        let retained = [cold.inputs[0].clone(), cold.inputs[1].clone()];

        let warm =
            materialize_bounded_dynamic_inputs(&domain, &signature, &declared_types, &actual_types, vec![left, right])
                .unwrap();
        assert_eq!(warm.inputs[0], retained[0]);
        assert_eq!(warm.inputs[1], retained[1]);
        assert_eq!(
            warm.report,
            BoundedInputMaterializationReport {
                cache_hits: 2,
                actual_bytes: 32,
                bound_bytes: 64,
                ..Default::default()
            },
        );
    }

    #[test]
    fn test_materialize_bounded_dynamic_inputs_sharded_input_reuses_every_device_buffer() {
        let client = cpu_client(4);
        // Executable signatures record placement over explicit mesh axes only, so the sharded inputs use an explicit
        // axis.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap(),
            client
                .addressable_devices()
                .unwrap()
                .iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect(),
        )
        .unwrap();
        let domain = Session::new(&client).domain();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let actual_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let source = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        let cold = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&actual_type),
            vec![source.clone()],
        )
        .unwrap();
        assert_eq!(cold.report.device_to_host_shard_copies, 4);
        // Each shard of the leading dimension is a contiguous range of the dense source, so the shards are copied
        // directly into one destination and no merge buffer is allocated.
        assert_eq!(cold.report.host_merge_buffer_allocations, 0);
        assert_eq!(cold.report.host_padding_payload_allocations, 1);
        assert_eq!(cold.report.host_to_device_shard_uploads, 4);
        let retained = cold.inputs[0].clone();

        let warm =
            materialize_bounded_dynamic_inputs(&domain, &signature, &[declared_type], &[actual_type], vec![source])
                .unwrap();
        assert_eq!(warm.inputs[0], retained);
        assert_eq!(warm.report.cache_hits, 1);
        assert_eq!(warm.report.device_to_host_shard_copies, 0);
        assert_eq!(warm.report.host_to_device_shard_uploads, 0);
    }

    #[test]
    fn test_materialize_bounded_dynamic_inputs_rejects_unreduced_inputs() {
        // Padding copies a bounded input through the host as one dense array, which would replace the separate
        // summands `[1, 2]` and `[3, 4]` that the two devices of this unreduced input hold, so it is rejected.
        let client = cpu_client(2);
        let devices = client.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let domain = Session::new(&client).domain();
        let sharding = Sharding::replicated(logical_mesh, 1).with_unreduced_axes(["x"]).unwrap();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let actual_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding).unwrap();
        let buffers = [[1f32, 2.], [3., 4.]]
            .iter()
            .zip(&devices)
            .map(|(values, device)| {
                client
                    .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let source = Array::from_addressable_buffers(&domain, actual_type.clone(), mesh, buffers).unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);
        let Err(error) =
            materialize_bounded_dynamic_inputs(&domain, &signature, &[declared_type], &[actual_type], vec![source])
        else {
            panic!("bounded materialization unexpectedly padded an unreduced input");
        };
        assert_eq!(
            error.to_string(),
            "cannot pad an array of type `f32[2][sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}}]` through \
             the host, because its devices hold separate summands along its unreduced axes, which a dense host copy \
             would not preserve",
        );
    }

    #[test]
    fn test_materialize_bounded_dynamic_inputs_concurrent_calls_are_single_flight() {
        let client = cpu_client(1);
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let actual_type = replicated_vector_type(&mesh, 4);
        let source = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        let (left, right) = std::thread::scope(|scope| {
            let left_source = source.clone();
            let left_declared_type = declared_type.clone();
            let left_actual_type = actual_type.clone();
            let left = scope.spawn(|| {
                materialize_bounded_dynamic_inputs(
                    &domain,
                    &signature,
                    &[left_declared_type],
                    &[left_actual_type],
                    vec![left_source],
                )
                .unwrap()
            });
            let right = scope.spawn(|| {
                materialize_bounded_dynamic_inputs(
                    &domain,
                    &signature,
                    std::slice::from_ref(&declared_type),
                    std::slice::from_ref(&actual_type),
                    vec![source],
                )
                .unwrap()
            });
            (left.join().unwrap(), right.join().unwrap())
        });

        assert_eq!(left.inputs[0], right.inputs[0]);
        let retained_misses = left.report.retained_misses + right.report.retained_misses;
        let cache_hits = left.report.cache_hits + right.report.cache_hits;
        assert_eq!(retained_misses, 1);
        assert_eq!(cache_hits, 1);
        assert_eq!(left.report.device_to_host_shard_copies + right.report.device_to_host_shard_copies, 1);
        assert_eq!(left.report.host_to_device_shard_uploads + right.report.host_to_device_shard_uploads, 1);
    }

    #[test]
    fn test_materialize_bounded_dynamic_inputs_skips_padding_for_zero_space_inputs() {
        let client = cpu_client(1);
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let declared_type = ArrayType::new(
            DataType::Zero,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::non_negative(Some(4)).unwrap()).into()]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);
        let actual_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(sharding)
            .unwrap();
        let zero = Array::from_host_buffer(&domain, actual_type.clone(), mesh, []).unwrap();

        // A zero-space input below its bound has an empty dense host copy, so it takes neither the retained cache
        // nor the padding tier, and only its hidden extent scalar is uploaded.
        let materialized = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&actual_type),
            vec![zero],
        )
        .unwrap();
        assert_eq!(materialized.inputs.len(), 2);
        assert_eq!(
            materialized.report,
            BoundedInputMaterializationReport { extent_scalar_uploads: 1, ..Default::default() },
        );
    }

    #[test]
    fn test_materialize_bounded_dynamic_inputs_with_readiness_failed_batch_is_not_published_and_retries() {
        let client = cpu_client(1);
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let actual_type = replicated_vector_type(&mesh, 4);
        let left = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let right = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[5.0, 6.0, 7.0, 8.0]).as_slice(),
        )
        .unwrap();
        let declared_types = [declared_type.clone(), declared_type];
        let actual_types = [actual_type.clone(), actual_type];
        let signature = ExecutableSignature::new(&declared_types, &[]);
        let mut observed_publications = 0usize;
        let mut observed_uploads = 0usize;

        // The injected failure runs only after both physical arrays and their host-to-device uploads exist. Returning
        // the error before publication drops both producer reservations, so neither failed result can become a cache
        // hit later.
        let error = match materialize_bounded_dynamic_inputs_with_readiness(
            &domain,
            &signature,
            &declared_types,
            &actual_types,
            vec![left.clone(), right.clone()],
            |publications| {
                observed_publications = publications.len();
                observed_uploads = publications
                    .iter()
                    .flat_map(|publication| publication.physical.shards())
                    .filter(|shard| shard.buffer().is_some())
                    .count();
                Err(ProgramError::InvalidArgument { message: "injected bounded upload readiness failure".to_string() }
                    .into())
            },
        ) {
            Ok(_) => panic!("injected upload readiness failure must reject materialization"),
            Err(error) => error,
        };
        assert!(matches!(
            &error,
            Error::Core(error) if matches!(error.as_ref(),
                CoreError::Program(ProgramError::InvalidArgument { message })
                    if message == "injected bounded upload readiness failure",
            ),
        ));
        assert_eq!(error.to_string(), "injected bounded upload readiness failure");
        assert_eq!(observed_publications, 2);
        assert_eq!(observed_uploads, 2);

        let retry = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            &declared_types,
            &actual_types,
            vec![left.clone(), right.clone()],
        )
        .unwrap();
        assert_eq!(retry.report.retained_misses, 2);
        assert_eq!(retry.report.cache_hits, 0);
        assert_eq!(retry.report.device_to_host_shard_copies, 2);
        assert_eq!(retry.report.host_to_device_shard_uploads, 2);
        let retained = [retry.inputs[0].clone(), retry.inputs[1].clone()];

        let warm =
            materialize_bounded_dynamic_inputs(&domain, &signature, &declared_types, &actual_types, vec![left, right])
                .unwrap();
        assert_eq!(warm.inputs[0], retained[0]);
        assert_eq!(warm.inputs[1], retained[1]);
        assert_eq!(warm.report.cache_hits, 2);
        assert_eq!(warm.report.device_to_host_shard_copies, 0);
        assert_eq!(warm.report.host_to_device_shard_uploads, 0);
    }

    #[test]
    fn test_pad_dense_array_bytes_with_target_allocation_count() {
        let source = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);

        let (padded, allocation_count) =
            pad_dense_array_bytes_with_target_allocation_count(source.as_slice(), &[2, 2], &[3, 4], DataType::F32)
                .unwrap();
        assert_eq!(
            values_from_bytes::<f32>(padded.as_slice()),
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(allocation_count, 1);

        // A source that does not fit into the target shape cannot be padded.
        assert!(matches!(
            pad_dense_array_bytes_with_target_allocation_count(source.as_slice(), &[2, 2], &[3, 1], DataType::F32),
            Err(Error::Core(error)) if matches!(error.as_ref(),
                CoreError::Program(ProgramError::InvalidArgument { message })
                    if message == "cannot pad runtime shape `[2, 2]` into bounded physical shape `[3, 1]`",
            ),
        ));
    }
}
