//! Process-local compile cache keyed by domain-computed structural keys.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use lru::LruCache;

use super::disk_cache::{CacheDigest, DiskCache};
use super::domain::CompilationDomain;

/// Default in-memory compile-cache capacity. Use [`CompilationContext::with_capacity`] when a
/// workload needs a different bound.
const DEFAULT_CACHE_CAPACITY: usize = 8192;

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
}

#[derive(Default)]
struct AtomicCompilationCacheStatistics {
    memory_hits: AtomicU64,
    persistent_hits: AtomicU64,
    misses: AtomicU64,
    compilations: AtomicU64,
    waits: AtomicU64,
    persistent_errors: AtomicU64,
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
        }
    }

    fn clear(&self) {
        self.memory_hits.store(0, Ordering::Relaxed);
        self.persistent_hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.compilations.store(0, Ordering::Relaxed);
        self.waits.store(0, Ordering::Relaxed);
        self.persistent_errors.store(0, Ordering::Relaxed);
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
/// calls to [`compile_with_options`](super::compile_with_options) and any backend-specific
/// helpers that look up entries in the cache.
///
/// The cache is keyed by the domain's structurally-typed [`CompilationDomain::CompilationKey`] — `Eq` on the key
/// guarantees no silent collisions, in contrast to a hash-only cache. On cache hit the cached program is returned
/// without invoking the producer closure. On miss the producer runs and the result is inserted.
///
/// The in-memory tier uses LRU eviction with a default capacity of `8192` entries.
/// Use [`CompilationContext::with_capacity`] to override.
///
/// An optional [`DiskCache`] second-tier is configured via [`CompilationContext::with_disk_cache`]. When present, the
/// disk tier is consulted between the in-memory tier and the producer closure. The cache uses
/// [`CompilationDomain::serialize_program`] and [`CompilationDomain::deserialize_program`] to round-trip programs; any
/// error from either method is treated as a cache miss for that entry.
pub struct CompilationContext<D: CompilationDomain> {
    /// In-memory LRU keyed by the domain's structural [`CompilationKey`].
    programs: Mutex<LruCache<D::CompilationKey, Arc<D::CompiledProgram>>>,

    /// Per-key producer coordination. Entries exist only while a cache miss is being restored or
    /// compiled, so unrelated keys never wait on one another's backend work.
    in_flight: Mutex<HashMap<D::CompilationKey, Arc<InFlightCompilation<D::CompiledProgram>>>>,

    /// Optional disk-backed second-tier cache.
    disk_cache: Option<DiskCache>,

    /// Lock-free counters used for verification and operational observability.
    statistics: AtomicCompilationCacheStatistics,
}

struct InFlightProducer<'a, D: CompilationDomain> {
    context: &'a CompilationContext<D>,
    cache_key: Option<D::CompilationKey>,
    in_flight: Option<Arc<InFlightCompilation<D::CompiledProgram>>>,
}

impl<'a, D: CompilationDomain> InFlightProducer<'a, D> {
    fn new(
        context: &'a CompilationContext<D>,
        cache_key: D::CompilationKey,
        in_flight: Arc<InFlightCompilation<D::CompiledProgram>>,
    ) -> Self {
        Self { context, cache_key: Some(cache_key), in_flight: Some(in_flight) }
    }

    fn cache_key(&self) -> &D::CompilationKey {
        self.cache_key.as_ref().expect("active producer owns its cache key")
    }

    fn finish(mut self, program: D::CompiledProgram) -> Arc<D::CompiledProgram> {
        let cache_key = self.cache_key.take().expect("active producer owns its cache key");
        let in_flight = self.in_flight.take().expect("active producer owns its in-flight state");
        self.context.finish_success(cache_key, in_flight, program)
    }
}

impl<D: CompilationDomain> Drop for InFlightProducer<'_, D> {
    fn drop(&mut self) {
        if let (Some(cache_key), Some(in_flight)) = (self.cache_key.take(), self.in_flight.take()) {
            self.context.finish_failure(&cache_key, in_flight);
        }
    }
}

impl<D: CompilationDomain> CompilationContext<D> {
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
            statistics: AtomicCompilationCacheStatistics::default(),
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

    /// Returns the attached [`DiskCache`], if any.
    #[inline]
    pub fn disk_cache(&self) -> Option<&DiskCache> {
        self.disk_cache.as_ref()
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
    /// [`CompilationDomain::persistent_cache_key`] returns stable key bytes. Persistent I/O or
    /// codec failures are counted and degraded to misses.
    pub fn get_or_compile<F: FnOnce() -> Result<D::CompiledProgram, D::Error>>(
        &self,
        domain: &D,
        cache_key: D::CompilationKey,
        produce: F,
    ) -> Result<Arc<D::CompiledProgram>, D::Error> {
        let mut produce = Some(produce);
        loop {
            if let Some(program) = self
                .programs
                .lock()
                .expect("compile cache mutex should not be poisoned")
                .get(&cache_key)
                .map(Arc::clone)
            {
                self.statistics.memory_hits.fetch_add(1, Ordering::Relaxed);
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
                if let Some(program) = in_flight.wait() {
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
                return Ok(program);
            }

            self.statistics.misses.fetch_add(1, Ordering::Relaxed);
            let producer = produce.take().expect("each cache lookup becomes a producer at most once");
            return self.restore_or_compile(domain, cache_key, in_flight, producer);
        }
    }

    fn restore_or_compile<F: FnOnce() -> Result<D::CompiledProgram, D::Error>>(
        &self,
        domain: &D,
        cache_key: D::CompilationKey,
        in_flight: Arc<InFlightCompilation<D::CompiledProgram>>,
        produce: F,
    ) -> Result<Arc<D::CompiledProgram>, D::Error> {
        let in_flight_producer = InFlightProducer::new(self, cache_key, in_flight);
        let persistent = self.disk_cache.as_ref().and_then(|cache| {
            domain
                .persistent_cache_key(in_flight_producer.cache_key())
                .map(|key| (cache, CacheDigest::from_bytes(key.as_slice())))
        });

        if let Some((disk_cache, digest)) = persistent.as_ref() {
            match disk_cache.get(digest) {
                Ok(Some(bytes)) => match domain.deserialize_program(bytes.as_slice()) {
                    Ok(Some(program)) => {
                        self.statistics.persistent_hits.fetch_add(1, Ordering::Relaxed);
                        return Ok(in_flight_producer.finish(program));
                    }
                    Ok(None) => {}
                    Err(_error) => {
                        self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
                    }
                },
                Ok(None) => {}
                Err(_error) => {
                    self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        self.statistics.compilations.fetch_add(1, Ordering::Relaxed);
        let compile_start = Instant::now();
        let program = match produce() {
            Ok(program) => program,
            Err(error) => return Err(error),
        };
        let compile_duration = compile_start.elapsed();

        if let Some((disk_cache, digest)) = persistent
            && disk_cache.should_serialize(compile_duration)
        {
            match domain.serialize_program(&program) {
                Ok(Some(bytes)) if disk_cache.should_persist(compile_duration, bytes.len()) => {
                    if let Err(_error) = disk_cache.put(&digest, bytes.as_slice()) {
                        self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Ok(Some(_)) | Ok(None) => {}
                Err(_error) => {
                    self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        Ok(in_flight_producer.finish(program))
    }

    fn finish_success(
        &self,
        cache_key: D::CompilationKey,
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

    fn finish_failure(&self, cache_key: &D::CompilationKey, in_flight: Arc<InFlightCompilation<D::CompiledProgram>>) {
        self.in_flight.lock().expect("in-flight cache mutex should not be poisoned").remove(cache_key);
        in_flight.finish(InFlightState::Failed);
    }
}

impl<D: CompilationDomain> Default for CompilationContext<D> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use crate::contexts::Domain;
    use crate::operations::scalars::ScalarOperation;
    use crate::programs::{Program, ProgramError};
    use crate::scalars::Scalar;
    use crate::types::DataType;

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
        type Type = DataType;
        type Value = Scalar;
        type Constant = Scalar;
        type Operation = ScalarOperation<Scalar>;
    }

    impl CompilationDomain for TestDomain {
        type LoweredProgram = Vec<DataType>;
        type CompiledProgram = TestCompiledProgram;
        type Options = ();
        type Error = ProgramError;
        type CompilationKey = u8;

        fn lower(
            &self,
            _program: &Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>>,
            _capture_count: usize,
            _options: &(),
        ) -> Result<Self::LoweredProgram, Self::Error> {
            Ok(Vec::new())
        }

        fn lowered_output_types<'a>(&self, program: &'a Self::LoweredProgram) -> &'a [DataType] {
            program
        }

        fn compilation_key(
            &self,
            _program: &Self::LoweredProgram,
            _options: &(),
        ) -> Result<Self::CompilationKey, Self::Error> {
            Ok(0)
        }

        fn compile(
            &self,
            _program: &Self::LoweredProgram,
            _options: &(),
        ) -> Result<Self::CompiledProgram, Self::Error> {
            Ok(TestCompiledProgram(0))
        }

        fn compiled_output_types<'a>(&self, _program: &'a Self::CompiledProgram) -> &'a [DataType] {
            &[]
        }

        fn execute(&self, _program: &Self::CompiledProgram, _inputs: Vec<Scalar>) -> Result<Vec<Scalar>, Self::Error> {
            Ok(Vec::new())
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
        assert_eq!(
            context.statistics(),
            CompilationCacheStatistics {
                memory_hits: 1,
                misses: 1,
                compilations: 1,
                ..CompilationCacheStatistics::default()
            }
        );
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
        while context.statistics().waits < (THREAD_COUNT - 1) as u64
            && producer_calls.load(Ordering::Relaxed) == 1
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
}
