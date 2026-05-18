//! Process-local compile cache keyed by engine-computed [`u64`] keys.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Mutex;

use lru::LruCache;

use super::disk_cache::{CacheDigest, DiskCache};
use super::domain::CompilationDomain;

/// Default in-memory compile-cache capacity. Matches the JAX `_cpp_pjit_cache_fun_only` default
/// (~8192 entries). Long-running training processes that exceed this are expected to be rare;
/// if they're a real concern, use [`CompilationContext::with_capacity`] for a higher bound.
const DEFAULT_CACHE_CAPACITY: usize = 8192;

/// Process-local cache of compiled programs, generic over the
/// [`CompilationDomain`](super::CompilationDomain) backend.
///
/// Construct one [`CompilationContext`] per backend handle at program start and reuse it across
/// calls to [`compile_and_execute_with_options`](super::compile_and_execute_with_options) and
/// any backend-specific helpers that look up entries in the cache.
///
/// The cache is keyed by the engine's [`CompilationDomain::fingerprint`](
/// super::CompilationDomain::fingerprint) output. On cache hit the cached program is returned
/// without invoking the producer closure. On miss the producer runs and the result is inserted.
///
/// The in-memory tier uses LRU eviction with a default capacity of [`DEFAULT_CACHE_CAPACITY`].
/// Use [`CompilationContext::with_capacity`] to override.
///
/// An optional [`DiskCache`] second-tier is configured via
/// [`CompilationContext::with_disk_cache`]. When present, the disk tier is consulted between
/// the in-memory tier and the producer closure. The cache uses
/// [`CompilationDomain::serialize_program`](super::CompilationDomain::serialize_program) and
/// [`CompilationDomain::deserialize_program`](super::CompilationDomain::deserialize_program)
/// to round-trip programs; any error from either method is treated as a cache miss for that
/// entry.
pub struct CompilationContext<E: CompilationDomain> {
    /// In-memory LRU keyed by the engine's `u64` cache key.
    programs: Mutex<LruCache<u64, E::CompiledProgram>>,

    /// Optional disk-backed second-tier cache.
    disk_cache: Option<DiskCache>,
}

impl<E: CompilationDomain> CompilationContext<E> {
    /// Creates a [`CompilationContext`] with the default cache capacity and no disk-cache tier.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CACHE_CAPACITY)
    }

    /// Creates a [`CompilationContext`] with an explicit cache capacity. `capacity` must be
    /// greater than zero; values of zero are silently clamped to one entry.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = NonZeroUsize::new(capacity.max(1)).expect("clamped capacity is at least one");
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
    /// When a disk-cache tier is configured, an in-memory miss falls through to the disk tier
    /// before invoking `produce`. Disk-tier entries are deserialized via
    /// [`CompilationDomain::deserialize_program`](super::CompilationDomain::deserialize_program);
    /// any error from the deserialize step is treated as a miss and the producer runs as usual.
    pub fn get_or_compile<F>(&self, engine: &E, cache_key: u64, produce: F) -> Result<E::CompiledProgram, E::Error>
    where
        F: FnOnce() -> Result<E::CompiledProgram, E::Error>,
    {
        // Tier 1: in-memory LRU.
        {
            let mut cache = self.programs.lock().expect("compile cache mutex should not be poisoned");
            if let Some(program) = cache.get(&cache_key) {
                return Ok(program.clone());
            }
        }

        // Tier 2: on-disk cache, when configured. Disk hits are deserialized via the engine and
        // promoted into the in-memory LRU. Any disk read or deserialization error falls through
        // to a fresh compile.
        let disk_digest = self.disk_cache.as_ref().map(|_| CacheDigest::from_cache_key(cache_key));
        if let (Some(disk_cache), Some(digest)) = (self.disk_cache.as_ref(), disk_digest.as_ref()) {
            if let Some(bytes) = disk_cache.get(digest) {
                if let Ok(program) = engine.deserialize_program(&bytes) {
                    let mut cache = self.programs.lock().expect("compile cache mutex should not be poisoned");
                    cache.put(cache_key, program.clone());
                    return Ok(program);
                }
            }
        }

        // Miss in both tiers: invoke the producer, populate both tiers.
        let program = produce()?;

        // Persist to disk before inserting into the in-memory cache so a crash mid-rename
        // doesn't leave an unfollowable phantom entry. Disk write failures are non-fatal, as
        // are engine serialization failures (backends that don't support serialization simply
        // return an error from `serialize_program`).
        if let (Some(disk_cache), Some(digest)) = (self.disk_cache.as_ref(), disk_digest.as_ref()) {
            if let Ok(serialized) = engine.serialize_program(&program) {
                let _ = disk_cache.put(digest, &serialized);
            }
        }

        let mut cache = self.programs.lock().expect("compile cache mutex should not be poisoned");
        if let Some(existing) = cache.get(&cache_key) {
            return Ok(existing.clone());
        }
        cache.put(cache_key, program.clone());
        Ok(program)
    }
}

impl<E: CompilationDomain> Default for CompilationContext<E> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
