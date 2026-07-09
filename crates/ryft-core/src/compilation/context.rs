//! Process-local compile cache keyed by domain-computed structural keys.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Mutex;

use lru::LruCache;

use crate::BatchableOperation;
use crate::batching::BatchingContext;
use crate::contexts::Context;
use crate::differentiation::{DifferentiableOperation, DifferentiationContext};
use crate::operations::constants::Zero;
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
use crate::programs::{ProgramError, Value};
use crate::tracing::NestedTracingContext;
use crate::types::ArrayType;

use super::disk_cache::{CacheDigest, DiskCache};
use super::domain::CompilationDomain;

/// Default in-memory compile-cache capacity. Matches the JAX `_cpp_pjit_cache_fun_only` default
/// (~8192 entries). Long-running training processes that exceed this are expected to be rare;
/// if they're a real concern, use [`CompilationContext::with_capacity`] for a higher bound.
const DEFAULT_CACHE_CAPACITY: usize = 8192;

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
    programs: Mutex<LruCache<D::CompilationKey, D::CompiledProgram>>,

    /// Optional disk-backed second-tier cache.
    disk_cache: Option<DiskCache>,
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
        let capacity = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self { programs: Mutex::new(LruCache::new(capacity)), disk_cache: None }
    }

    /// Attaches a [`DiskCache`] rooted at `directory` as a second-tier persistent compile cache
    /// behind this context's in-memory LRU. On cache miss, [`Self::get_or_compile`] consults
    /// disk before invoking the supplied producer; freshly-compiled programs are serialized
    /// back to disk so future processes restart warm. Mirrors JAX's `JAX_COMPILATION_CACHE_DIR`.
    ///
    /// Returns an [`std::io::Error`] only when the directory itself can't be opened or created;
    /// all subsequent disk read / write / serialization errors are non-fatal and degrade
    /// transparently to the "no disk cache" path.
    pub fn with_disk_cache(mut self, directory: impl Into<PathBuf>) -> std::io::Result<Self> {
        self.disk_cache = Some(DiskCache::open(directory)?);
        Ok(self)
    }

    /// Attaches a [`DiskCache`] populated from the [`DiskCache::ENV_VAR`] environment variable,
    /// if it's set. Returns a context without a disk cache when the variable is absent or
    /// unparseable. Never fails — this is the most ergonomic constructor for environments that
    /// opt into persistent caching via configuration.
    pub fn with_disk_cache_from_env(mut self) -> Self {
        self.disk_cache = DiskCache::from_env();
        self
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

    /// Removes every entry from the in-memory cache. Mirrors JAX's
    /// `clear_in_memory_compilation_cache()`.
    #[inline]
    pub fn clear_cache(&self) {
        self.programs.lock().expect("compile cache mutex should not be poisoned").clear();
    }

    /// Returns a cached program for `cache_key` if present, otherwise invokes `produce` to
    /// produce a fresh program, caches the result under `cache_key`, and returns it.
    ///
    /// On cache hit, the entry is moved to the most-recently-used position. On miss, the new
    /// entry is inserted; if the cache is at capacity, the least-recently-used entry is evicted.
    ///
    /// When a disk-cache tier is configured, an in-memory miss falls through to the disk tier before invoking
    /// `produce`. Disk-tier entries are deserialized via [`CompilationDomain::deserialize_program`]; any error from the
    /// deserialize step is treated as a miss and the producer runs as usual.
    pub fn get_or_compile<F: FnOnce() -> Result<D::CompiledProgram, D::Error>>(
        &self,
        domain: &D,
        cache_key: D::CompilationKey,
        produce: F,
    ) -> Result<D::CompiledProgram, D::Error> {
        // Tier 1: in-memory LRU.
        {
            let mut cache = self.programs.lock().expect("compile cache mutex should not be poisoned");
            if let Some(program) = cache.get(&cache_key) {
                return Ok(program.clone());
            }
        }

        // Tier 2: on-disk cache, when configured. Disk hits are deserialized via the domain and
        // promoted into the in-memory LRU. Any disk read or deserialization error falls through
        // to a fresh compile.
        let disk_digest = self.disk_cache.as_ref().map(|_| CacheDigest::from_key(&cache_key));
        if let (Some(disk_cache), Some(digest)) = (self.disk_cache.as_ref(), disk_digest.as_ref())
            && let Some(bytes) = disk_cache.get(digest)
            && let Ok(program) = domain.deserialize_program(&bytes)
        {
            let mut cache = self.programs.lock().expect("compile cache mutex should not be poisoned");
            cache.put(cache_key, program.clone());
            return Ok(program);
        }

        // Miss in both tiers: invoke the producer, populate both tiers.
        let program = produce()?;

        // Persist to disk before inserting into the in-memory cache so a crash mid-rename
        // doesn't leave an unfollowable phantom entry. Disk write failures are non-fatal, as
        // are domain serialization failures (backends that don't support serialization simply
        // return an error from `serialize_program`).
        if let (Some(disk_cache), Some(digest)) = (self.disk_cache.as_ref(), disk_digest.as_ref())
            && let Ok(serialized) = domain.serialize_program(&program)
        {
            // Disk writes are best-effort; a failed write should not make compilation fail.
            let _ = disk_cache.put(digest, &serialized);
        }

        let mut cache = self.programs.lock().expect("compile cache mutex should not be poisoned");
        if let Some(existing) = cache.get(&cache_key) {
            return Ok(existing.clone());
        }
        cache.put(cache_key, program.clone());
        Ok(program)
    }
}

impl<D: CompilationDomain> Default for CompilationContext<D> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Active tracing [`Context`] that can register runtime values as captures of the program being built. The returned
/// value is the context's staged constant payload. For captured-program backends this is usually a lifetime-free
/// reference into a side table owned by the surrounding compiled function
/// (see [`CaptureReference`](super::captures::CaptureReference)). Stackable transform contexts implement this by
/// delegating to their parent context so capture registration follows the same nesting path as ordinary operation
/// staging.
///
/// Capturing a closed-over value (rather than staging it as a literal constant) keeps the staged program independent
/// of that value's concrete data, so the compiled program depends only on its abstract type — which enables executable
/// reuse across captured values, keeps captured device buffers on-device, and avoids bloating the IR. See
/// [`CaptureReference`](super::captures::CaptureReference) for the full rationale.
pub trait CapturingContext<C: Value<Type = Self::Type>>: Context {
    /// Appends `value` to the active capture table and returns the constant payload that refers to it.
    fn capture(&self, value: C) -> Result<Self::Constant, ProgramError>;
}

impl<Capture: Value<Type = C::Type>, C: CapturingContext<Capture>> CapturingContext<Capture>
    for NestedTracingContext<C>
{
    #[inline]
    fn capture(&self, value: Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

impl<Capture, C> CapturingContext<Capture> for PartialEvaluationContext<C>
where
    Capture: Value<Type = C::Type>,
    C: CapturingContext<Capture>,
    C::Operation: PartiallyEvaluatableOperation<C>,
{
    #[inline]
    fn capture(&self, value: Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

impl<
    Capture: Value<Type = ArrayType>,
    C: CapturingContext<Capture, Type = ArrayType, Operation: BatchableOperation<C::Value, BatchingContext<C>>>,
> CapturingContext<Capture> for BatchingContext<C>
{
    #[inline]
    fn capture(&self, value: Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

impl<
    Capture: Value<Type = C::Type>,
    C: CapturingContext<Capture, Operation: Clone + DifferentiableOperation<C>> + Zero<C::Value>,
> CapturingContext<Capture> for DifferentiationContext<C>
{
    #[inline]
    fn capture(&self, value: Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::hash::{Hash, Hasher};
    use std::sync::Mutex;

    use lru::LruCache;

    /// A bucket-collision key: every instance hashes identically, so the in-memory `LruCache`
    /// puts them in the same hash bucket. A naive `HashMap<u64, _>` cache would silently
    /// alias them; an `Eq`-based cache MUST still disambiguate them structurally.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CollidingKey(u8);

    impl Hash for CollidingKey {
        fn hash<H: Hasher>(&self, _state: &mut H) {
            // Intentionally constant hash: every key hashes identically.
        }
    }

    /// Verifies that the `LruCache<D::CompilationKey, _>` shape used by [`CompilationContext`]
    /// disambiguates structurally-unequal keys even when they hash to the same bucket. This
    /// is the property the structural-key design provides over a hash-only `u64` cache.
    #[test]
    fn test_lru_cache_disambiguates_hash_colliding_but_unequal_keys() {
        // Mirror the in-memory tier shape that `CompilationContext` uses internally.
        let cache: Mutex<LruCache<CollidingKey, u32>> =
            Mutex::new(LruCache::new(std::num::NonZeroUsize::new(16).unwrap()));

        {
            let mut guard = cache.lock().unwrap();
            guard.put(CollidingKey(1), 100);
            guard.put(CollidingKey(2), 200);
        }

        // Both keys must coexist; lookup by equality returns the correct value.
        let guard = cache.lock().unwrap();
        assert_eq!(guard.peek(&CollidingKey(1)), Some(&100));
        assert_eq!(guard.peek(&CollidingKey(2)), Some(&200));
        assert_eq!(guard.peek(&CollidingKey(3)), None, "absent key returns None");
    }

    /// Confirms that the LRU cache hits without invoking the producer when the same key is
    /// queried twice. (Sanity-check covers the `get_or_compile` happy path through
    /// `LruCache::get`.)
    #[test]
    fn test_lru_cache_hits_avoid_recompute() {
        let cache: Mutex<LruCache<CollidingKey, u32>> =
            Mutex::new(LruCache::new(std::num::NonZeroUsize::new(16).unwrap()));
        let producer_calls = Cell::new(0);

        // First insertion: producer runs, value stored under key.
        let key = CollidingKey(42);
        {
            let mut guard = cache.lock().unwrap();
            if guard.get(&key).is_none() {
                producer_calls.set(producer_calls.get() + 1);
                guard.put(key.clone(), 7);
            }
        }
        // Second lookup: producer must NOT run.
        {
            let mut guard = cache.lock().unwrap();
            if guard.get(&key).is_none() {
                producer_calls.set(producer_calls.get() + 1);
            }
        }
        assert_eq!(producer_calls.get(), 1, "second lookup must hit the cache");
    }
}
