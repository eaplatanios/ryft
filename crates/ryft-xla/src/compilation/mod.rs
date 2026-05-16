use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::panic::Location;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use lru::LruCache;
use ryft_core::sharding::DeviceMesh;
use ryft_core::types::ArrayType;
use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Client, LoadedExecutable, Program};

pub mod disk_cache;

use disk_cache::CacheDigest;
pub use disk_cache::DiskCache;

/// Default in-memory compile-cache capacity. Matches the JAX `_cpp_pjit_cache_fun_only` default
/// (~8192 entries). Long-running training processes that exceed this are expected to be rare; if
/// they're a real concern, use [`CompilationContext::with_capacity`] for a higher bound.
const DEFAULT_CACHE_CAPACITY: usize = 8192;

/// Thin wrapper around a PJRT [`Client`] that adds a process-local cache of compiled
/// [`LoadedExecutable`]s plus a customizable base [`CompilationOptions`] template.
///
/// Construct one [`CompilationContext`] per `Client` at program start and reuse it across calls
/// to [`Array::to_placement`](crate::Array::to_placement),
/// [`Array::to_device`](crate::Array::to_device),
/// [`device_put`](crate::arrays_v0::device_put), and [`jit`](crate::jit).
///
/// The cache is keyed by a **structural signature** the caller provides up front — for example,
/// `(input ArrayType, src Sharding, dst Sharding, DeviceMesh)` for a reshard call — mixed with a
/// stable hash of the [`CompilationOptions`]. Repeat calls that hash to the same key short-circuit
/// without running the trace + lower work that would otherwise produce the MLIR text; only
/// cache-miss calls invoke the supplied closure to materialize MLIR and feed it to
/// [`Client::compile`].
///
/// This mirrors how JAX's compile cache is keyed on abstract value signatures: on a warm call,
/// neither tracing nor lowering runs.
///
/// The in-memory cache uses LRU eviction with a default capacity of [`DEFAULT_CACHE_CAPACITY`]
/// entries (matching JAX's `_cpp_pjit_cache_fun_only` default). Use
/// [`CompilationContext::with_capacity`] to override.
pub struct CompilationContext<'c> {
    /// PJRT client wrapped by this context.
    client: &'c Client<'c>,

    /// Base [`CompilationOptions`] template. Callers that want non-default options (e.g. a
    /// specific matrix-unit precision) construct the context via
    /// [`CompilationContext::with_options`]. Reshard callers overlay mesh-derived
    /// `partition_count` / SPMD flags on top of this template before compiling.
    base_options: CompilationOptions,

    /// Compile-cache keyed by `(structural-signature hash, options-debug hash)`, bounded by an
    /// LRU policy to prevent unbounded growth in long-running processes.
    executables: Mutex<LruCache<u64, Arc<LoadedExecutable<'c>>>>,

    /// Optional disk-backed second-tier cache. When present, [`Self::get_or_compile`] consults
    /// disk on in-memory cache miss before invoking the supplied MLIR-production closure, and
    /// writes freshly-compiled executables back to disk for future processes to reuse.
    disk_cache: Option<DiskCache>,
}

impl<'c> CompilationContext<'c> {
    /// Creates a [`CompilationContext`] wrapping the provided PJRT [`Client`] with the default
    /// [`CompilationOptions`] template and the default cache capacity.
    #[inline]
    pub fn new(client: &'c Client<'c>) -> Self {
        Self::with_options_and_capacity(client, CompilationOptions::default(), DEFAULT_CACHE_CAPACITY)
    }

    /// Creates a [`CompilationContext`] with an explicit [`CompilationOptions`] template.
    ///
    /// Reshard callers can override compilation-time knobs (e.g. matrix-unit operand precision,
    /// custom environment options) by constructing the context with the desired
    /// [`CompilationOptions`]; the reshard machinery then overlays the mesh-derived SPMD fields
    /// on top of this template per call.
    #[inline]
    pub fn with_options(client: &'c Client<'c>, options: CompilationOptions) -> Self {
        Self::with_options_and_capacity(client, options, DEFAULT_CACHE_CAPACITY)
    }

    /// Creates a [`CompilationContext`] with the default options but an explicit cache capacity.
    /// Useful for tests that want to exercise LRU eviction without piling up entries.
    #[inline]
    pub fn with_capacity(client: &'c Client<'c>, capacity: usize) -> Self {
        Self::with_options_and_capacity(client, CompilationOptions::default(), capacity)
    }

    /// Creates a [`CompilationContext`] with both an explicit [`CompilationOptions`] template
    /// and an explicit cache capacity.
    ///
    /// `capacity` must be greater than zero; values of zero are silently clamped to one entry.
    pub fn with_options_and_capacity(client: &'c Client<'c>, options: CompilationOptions, capacity: usize) -> Self {
        let capacity = NonZeroUsize::new(capacity.max(1)).expect("clamped capacity is at least one");
        Self { client, base_options: options, executables: Mutex::new(LruCache::new(capacity)), disk_cache: None }
    }

    /// Attaches a [`DiskCache`] rooted at `directory` as a second-tier persistent compile cache
    /// behind this context's in-memory LRU. On cache miss, [`Self::get_or_compile`] consults the
    /// disk before invoking the supplied MLIR closure; freshly-compiled executables are
    /// serialized back to disk so future processes restart warm. Mirrors JAX's
    /// `JAX_COMPILATION_CACHE_DIR`.
    ///
    /// Returns an `std::io::Error` only when the directory itself can't be opened or created;
    /// all subsequent disk read / write errors are non-fatal and degrade transparently to the
    /// "no disk cache" path.
    pub fn with_disk_cache(client: &'c Client<'c>, directory: impl Into<PathBuf>) -> std::io::Result<Self> {
        let mut context = Self::new(client);
        context.disk_cache = Some(DiskCache::open(directory)?);
        Ok(context)
    }

    /// Attaches a [`DiskCache`] populated from the
    /// [`DiskCache::ENV_VAR`](disk_cache::DiskCache::ENV_VAR) environment variable, if it's set.
    /// Returns a context without a disk cache when the variable is absent or unparseable. Never
    /// fails — this is the most ergonomic constructor for environments that opt into persistent
    /// caching via configuration.
    pub fn with_disk_cache_from_env(client: &'c Client<'c>) -> Self {
        let mut context = Self::new(client);
        context.disk_cache = DiskCache::from_env();
        context
    }

    /// Returns the attached [`DiskCache`], if any.
    #[inline]
    pub fn disk_cache(&self) -> Option<&DiskCache> {
        self.disk_cache.as_ref()
    }

    /// Returns the PJRT [`Client`] wrapped by this context.
    #[inline]
    pub fn client(&self) -> &'c Client<'c> {
        self.client
    }

    /// Returns the base [`CompilationOptions`] template this context was constructed with.
    #[inline]
    pub fn base_options(&self) -> &CompilationOptions {
        &self.base_options
    }

    /// Returns the number of compiled [`LoadedExecutable`]s currently cached.
    ///
    /// Mostly useful for telemetry and tests that need to confirm that repeated compilations of
    /// the same structural signature reuse the cached executable instead of recompiling.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.executables.lock().expect("compile cache mutex should not be poisoned").len()
    }

    /// Removes every entry from the in-memory cache. Mirrors JAX's
    /// `clear_in_memory_compilation_cache()`.
    #[inline]
    pub fn clear_cache(&self) {
        self.executables.lock().expect("compile cache mutex should not be poisoned").clear();
    }

    /// Returns a cached executable for `key` if present, otherwise invokes `produce_mlir` to
    /// materialize the MLIR text, compiles it via PJRT, caches the result under `(key, options)`,
    /// and returns it.
    ///
    /// `key` is the structural cache signature — typically a tuple of input/output types, sharding
    /// annotations, mesh device order, and any other data that uniquely identifies the MLIR text
    /// that `produce_mlir` would emit. The closure runs only on cache miss, so repeat calls with
    /// the same `(key, options)` pay no tracing / lowering cost.
    ///
    /// On cache hit, the entry is moved to the most-recently-used position. On miss, the new
    /// entry is inserted; if the cache is at capacity, the least-recently-used entry is evicted.
    ///
    /// # Parameters
    ///
    ///   - `key`: Structural signature. Must implement [`Hash`].
    ///   - `options`: Compile options. Mixed into the cache key via their `Debug` representation.
    ///   - `produce_mlir`: Closure that materializes the MLIR text on cache miss.
    pub(crate) fn get_or_compile<K, F, E>(
        &self,
        key: &K,
        options: &CompilationOptions,
        produce_mlir: F,
    ) -> Result<Arc<LoadedExecutable<'c>>, E>
    where
        K: Hash,
        F: FnOnce() -> Result<String, E>,
        E: From<ryft_pjrt::Error>,
    {
        let cache_key = hash_signature(key, options);

        // Tier 1: in-memory LRU.
        {
            let mut cache = self.executables.lock().expect("compile cache mutex should not be poisoned");
            if let Some(executable) = cache.get(&cache_key) {
                return Ok(executable.clone());
            }
        }

        // Tier 2: on-disk cache, when configured. Disk hits are deserialized into a fresh
        // `LoadedExecutable` and promoted into the in-memory LRU. Any disk read or PJRT
        // deserialization error falls through to a fresh compile.
        let disk_digest = self.disk_cache_digest(cache_key);
        if let (Some(disk_cache), Some(digest)) = (self.disk_cache.as_ref(), disk_digest.as_ref()) {
            if let Some(bytes) = disk_cache.get(digest) {
                if let Ok(loaded) = self.client.deserialize_and_load_executable(&bytes, Some(options)) {
                    let executable = Arc::new(loaded);
                    let mut cache = self.executables.lock().expect("compile cache mutex should not be poisoned");
                    cache.put(cache_key, executable.clone());
                    return Ok(executable);
                }
            }
        }

        // Miss in both tiers: produce MLIR, compile, populate both tiers.
        let mlir_text = produce_mlir()?;
        let program = Program::Mlir { bytecode: mlir_text.into_bytes() };
        let executable = Arc::new(self.client.compile(&program, options)?);

        // Persist to disk before inserting into the in-memory cache so a crash mid-rename
        // doesn't leave an unfollowable phantom entry. Disk write failures are non-fatal.
        if let (Some(disk_cache), Some(digest)) = (self.disk_cache.as_ref(), disk_digest.as_ref()) {
            if let Ok(serialized) =
                executable.executable().and_then(|exec| exec.serialize().map(|bytes| bytes.data().to_vec()))
            {
                let _ = disk_cache.put(digest, &serialized);
            }
        }

        let mut cache = self.executables.lock().expect("compile cache mutex should not be poisoned");
        if let Some(existing) = cache.get(&cache_key) {
            return Ok(existing.clone());
        }
        cache.put(cache_key, executable.clone());
        Ok(executable)
    }

    /// Computes the digest used to key disk-cache entries. Returns `None` when no disk cache is
    /// configured or the PJRT client can't report its platform identifiers (in which case we
    /// can't safely scope the digest and bypass the disk cache for this request).
    fn disk_cache_digest(&self, cache_key: u64) -> Option<CacheDigest> {
        if self.disk_cache.is_none() {
            return None;
        }
        let platform_name = self.client.platform_name().ok()?;
        let platform_version = self.client.platform_version().ok()?;
        Some(CacheDigest::from_parts(cache_key, &platform_name, &platform_version))
    }
}

fn hash_signature<K: Hash>(key: &K, options: &CompilationOptions) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    // `CompilationOptions` is a `prost::Message` and does not derive `Hash`. Its `Debug` impl is
    // stable enough for cache-key purposes and avoids pulling `prost` into ryft-xla.
    format!("{options:?}").hash(&mut hasher);
    hasher.finish()
}

/// Identifier for the function or primitive being compiled.
///
/// Two compilations with the same [`FunctionFingerprint`], the same [`CompilationKey::input_types`],
/// and the same destination mesh produce the same executable and share a cache entry. This
/// mirrors how JAX's compile cache combines a function fingerprint with abstract input value
/// signatures.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FunctionFingerprint {
    /// A `ryft` built-in primitive identified by a static name. Reserved for internal use cases
    /// like the compiled-reshard path.
    Primitive(&'static str),

    /// A user function identified by the source location of its outer entry point (for example
    /// the call site of a future `jit`). Construct via [`FunctionFingerprint::from_caller`].
    ///
    /// JAX uses the Python function's identity plus closure-captured cells as the fingerprint;
    /// Rust's closures don't expose an equivalent stable identity, so `ryft` uses the call-site
    /// location as a best-effort proxy. Callers that capture state in their closures should
    /// embed the captured values themselves into the fingerprint (see
    /// [`FunctionFingerprint::Composite`]).
    SourceLocation { file: &'static str, line: u32, column: u32 },

    /// A composite fingerprint: a base fingerprint mixed with an opaque 64-bit hash of any
    /// additional state (e.g. captured constants) that uniquely identifies a function instance.
    /// Use this when the call site alone is not enough to distinguish two logical functions.
    Composite { base: Box<FunctionFingerprint>, extra: u64 },
}

impl FunctionFingerprint {
    /// Constructs a [`FunctionFingerprint::SourceLocation`] from the call site that invokes this
    /// function. The call-site location is captured at compile time via `#[track_caller]` and is
    /// stable as long as the call site doesn't move.
    #[inline]
    #[track_caller]
    pub fn from_caller() -> Self {
        let location = Location::caller();
        Self::SourceLocation { file: location.file(), line: location.line(), column: location.column() }
    }
}

/// Generic cache key for the compiled-executable cache.
///
/// Captures the four pieces of information that, together with the [`CompilationOptions`], make
/// one compilation distinct from another:
///
///   1. [`Self::fingerprint`] — what function is being compiled.
///   2. [`Self::input_types`] — abstract input value signatures (shape, dtype, sharding).
///   3. [`Self::output_types`] — abstract output value signatures.
///   4. [`Self::mesh`] — concrete device topology the program will run on.
///
/// The [`compiled_reshard`](crate::arrays_v0::compiled_reshard) module uses this key with
/// [`FunctionFingerprint::Primitive("compiled_reshard.identity")`](FunctionFingerprint::Primitive).
/// A future user-facing `jit` would use [`FunctionFingerprint::SourceLocation`] (or
/// [`FunctionFingerprint::from_caller`]) and the same `CompilationKey` shape.
pub struct CompilationKey<'a> {
    /// Identifier for the function/primitive being compiled.
    pub fingerprint: FunctionFingerprint,

    /// Abstract input value types — shape, dtype, and sharding metadata for each program input.
    pub input_types: &'a [ArrayType],

    /// Abstract output value types.
    pub output_types: &'a [ArrayType],

    /// Concrete device topology. Different device orderings of the same logical mesh produce
    /// different executables, so the full mesh is part of the key.
    pub mesh: &'a DeviceMesh,
}

impl<'a> Hash for CompilationKey<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fingerprint.hash(state);
        self.input_types.hash(state);
        self.output_types.hash(state);
        // `DeviceMesh` does not derive `Hash`; hash its (logical mesh, device order) tuple
        // manually instead. `LogicalMesh` and `Device` both implement `Hash`.
        self.mesh.logical_mesh().hash(state);
        self.mesh.devices().hash(state);
    }
}
