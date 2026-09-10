//! Context-aware kernel launching, bounded module retention, and explicit resource cleanup.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::artifacts::CudaKernelContentAddress;
use crate::drivers::{CudaDriver, CudaDriverApi, CudaLoadedKernel};
use crate::launches::{CudaKernelArgumentStorage, validate_launch};
use crate::{CudaKernelArtifact, CudaKernelLaunch, CudaStream, CudaVersion, Error};

/// Default maximum cached entries in one context/device partition.
const DEFAULT_KERNEL_CACHE_CAPACITY_PER_CONTEXT_DEVICE: usize = 128;
/// Default source-byte budget in one context/device partition.
const DEFAULT_KERNEL_CACHE_ARTIFACT_BYTES_PER_CONTEXT_DEVICE: usize = 256 * 1024 * 1024;

/// Resource limits applied independently to each CUDA context/device kernel-cache partition.
///
/// Artifact bytes are a deterministic proxy for loaded module resources. The CUDA Driver API does not expose the
/// actual memory occupied by a loaded module. The default retains up to 128 entries representing up to 256 MiB of
/// source artifact bytes for each context/device pair.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CudaKernelCacheLimits {
    /// Maximum loaded modules retained for one CUDA context/device pair.
    max_entries_per_context_device: usize,

    /// Maximum source artifact bytes represented by loaded modules for one CUDA context/device pair.
    max_artifact_bytes_per_context_device: usize,
}

impl CudaKernelCacheLimits {
    /// Creates positive per-context/device entry and artifact-byte limits.
    ///
    /// # Parameters
    ///
    ///   - `max_entries_per_context_device`: Maximum loaded modules retained for one CUDA context/device pair.
    ///   - `max_artifact_bytes_per_context_device`: Maximum source artifact bytes represented by those modules.
    pub fn new(
        max_entries_per_context_device: usize,
        max_artifact_bytes_per_context_device: usize,
    ) -> Result<Self, Error> {
        if max_entries_per_context_device == 0 {
            return Err(Error::invalid_argument("cuda kernel cache entries per context/device must be positive"));
        }
        if max_artifact_bytes_per_context_device == 0 {
            return Err(Error::invalid_argument(
                "cuda kernel cache artifact bytes per context/device must be positive",
            ));
        }
        Ok(Self { max_entries_per_context_device, max_artifact_bytes_per_context_device })
    }

    /// Returns the maximum number of loaded modules retained for one CUDA context/device pair.
    pub fn max_entries_per_context_device(self) -> usize {
        self.max_entries_per_context_device
    }

    /// Returns the maximum source artifact bytes represented by loaded modules for one CUDA context/device pair.
    pub fn max_artifact_bytes_per_context_device(self) -> usize {
        self.max_artifact_bytes_per_context_device
    }
}

impl Default for CudaKernelCacheLimits {
    fn default() -> Self {
        Self {
            max_entries_per_context_device: DEFAULT_KERNEL_CACHE_CAPACITY_PER_CONTEXT_DEVICE,
            max_artifact_bytes_per_context_device: DEFAULT_KERNEL_CACHE_ARTIFACT_BYTES_PER_CONTEXT_DEVICE,
        }
    }
}

/// Cumulative kernel-cache activity observed by one [`CudaKernelLauncher`].
///
/// A snapshot obtained during concurrent launches may observe the individual lock-free counters at slightly different
/// instants.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaKernelCacheStatistics {
    /// Number of launches that reused a loaded module/function.
    hits: u64,

    /// Number of launches that did not find a loaded module/function.
    misses: u64,

    /// Number of loaded modules successfully removed by least-recently-used eviction.
    evictions: u64,
}

impl CudaKernelCacheStatistics {
    /// Returns the cumulative cache-hit count.
    pub fn hits(self) -> u64 {
        self.hits
    }

    /// Returns the cumulative cache-miss count.
    pub fn misses(self) -> u64 {
        self.misses
    }

    /// Returns the cumulative successful-eviction count.
    pub fn evictions(self) -> u64 {
        self.evictions
    }
}

/// CUDA Driver API launcher with context/device/artifact-keyed module ownership and bounded caching.
///
/// The required CUDA version is supplied by the artifact producer or framework integration, and the launcher rejects
/// an older NVIDIA driver before resolving or invoking any operational entry points.
///
/// Call [`Self::shutdown`] before destroying the external runtimes whose contexts were observed by [`Self::launch`] to
/// deterministically synchronize and unload cached modules. Dropping the launcher performs the same cleanup on a
/// best-effort basis without panicking; CUDA context destruction remains the fallback owner of modules whose context
/// is no longer available. Cleanup failures during drop are retained for [`Error::take_cleanup_errors`].
///
/// CUDA graphs must not retain references to modules owned by this launcher. Passing a capturing stream to
/// [`Self::launch`] is supported and returns an error before cache mutation. Other streams' capture state is not
/// checked. Callers must exclude capture in every retained context during cleanup, including cleanup performed by
/// dropping the launcher, as required by [`Self::launch`], [`Self::clear`], and [`Self::shutdown`].
///
/// Cache lookup, module loading, and synchronized eviction share one mutex across partitions. A slow load or eviction
/// can therefore delay other contexts. Kernel enqueue occurs after releasing that mutex; concurrent launches retain
/// independent module borrows so eviction cannot unload a module still being enqueued.
pub struct CudaKernelLauncher {
    /// CUDA version required by this launcher.
    cuda_version: CudaVersion,

    /// Loaded CUDA Driver API and its resolved functions.
    driver: Arc<dyn CudaDriverApi>,

    /// Loaded kernels keyed by unique CUDA context/device pairs and artifact content addresses.
    kernels: Mutex<CudaKernelCache>,

    /// Whether explicit shutdown has permanently disabled new launches.
    is_shutdown: AtomicBool,

    /// Number of launches that reused a cached module/function.
    cache_hits: AtomicU64,

    /// Number of launches that did not find a cached module/function.
    cache_misses: AtomicU64,

    /// Number of modules successfully removed by least-recently-used eviction.
    cache_evictions: AtomicU64,
}

impl CudaKernelLauncher {
    /// Loads a CUDA Driver API compatible with `cuda_version` and constructs a launcher with a bounded default cache.
    pub fn new(cuda_version: CudaVersion) -> Result<Self, Error> {
        Self::with_cache_limits(cuda_version, CudaKernelCacheLimits::default())
    }

    /// Constructs a launcher with `cache_capacity_per_context_device` entries for each CUDA context/device pair.
    ///
    /// The default artifact-byte budget remains in effect. Use [`Self::with_cache_limits`] to configure both limits.
    pub fn with_cache_capacity(
        cuda_version: CudaVersion,
        cache_capacity_per_context_device: usize,
    ) -> Result<Self, Error> {
        let limits = CudaKernelCacheLimits::new(
            cache_capacity_per_context_device,
            DEFAULT_KERNEL_CACHE_ARTIFACT_BYTES_PER_CONTEXT_DEVICE,
        )?;
        Self::with_cache_limits(cuda_version, limits)
    }

    /// Loads a compatible CUDA Driver API and constructs a launcher with per-context/device cache limits.
    pub fn with_cache_limits(cuda_version: CudaVersion, limits: CudaKernelCacheLimits) -> Result<Self, Error> {
        let driver = Arc::new(CudaDriver::load(cuda_version)?);
        Ok(Self::with_driver_for_version(driver, limits, cuda_version))
    }

    /// Initializes cache ownership and counters around an already loaded driver.
    fn with_driver_for_version(
        driver: Arc<dyn CudaDriverApi>,
        limits: CudaKernelCacheLimits,
        cuda_version: CudaVersion,
    ) -> Self {
        Self {
            cuda_version,
            driver,
            kernels: Mutex::new(CudaKernelCache::new(limits)),
            is_shutdown: AtomicBool::new(false),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            cache_evictions: AtomicU64::new(0),
        }
    }

    /// Returns the CUDA version required by this launcher.
    pub fn cuda_version(&self) -> CudaVersion {
        self.cuda_version
    }

    /// Returns a lock-free snapshot of cumulative kernel-cache activity.
    pub fn cache_statistics(&self) -> CudaKernelCacheStatistics {
        CudaKernelCacheStatistics {
            hits: self.cache_hits.load(Ordering::Relaxed),
            misses: self.cache_misses.load(Ordering::Relaxed),
            evictions: self.cache_evictions.load(Ordering::Relaxed),
        }
    }

    /// Enqueues `artifact` on the CUDA stream stored in `launch`.
    ///
    /// Success reports enqueue completion, not GPU execution completion. Module loading and symbol resolution are
    /// cached per unique CUDA context/device pair and artifact content address. The stream must belong to the current
    /// context. Passing a capturing stream returns an error before any cache mutation.
    /// Architecture compatibility is checked before loading; CUDA validates image-specific restrictions. Least-recently
    /// used idle modules in the same partition are synchronized and unloaded before either cache limit is exceeded.
    ///
    /// # Safety
    ///
    /// The executable image and declared ABI must agree exactly with the actual kernel signature, including parameter
    /// count, sizes, and representations. Artifact and launch validation cannot prove this correspondence or the
    /// memory safety of executable device code. Every argument must satisfy the kernel's requirements for allocation
    /// bounds, alignment, access permissions, aliasing, and cross-stream ordering. Device allocations and other
    /// asynchronously accessed resources must remain valid through GPU completion, even after this function returns.
    ///
    /// Every external runtime owning an observed context must remain alive until that context is successfully released
    /// by [`Self::clear_context`], until [`Self::clear`] or [`Self::shutdown`] succeeds, or until the launcher is
    /// dropped. Retained context handles cannot encode external ownership in Rust.
    ///
    /// Passing an already capturing stream is supported and is rejected. Otherwise, the caller must prevent capture
    /// from starting on that stream and exclude capture on other streams in the launch context throughout this call:
    /// cache admission or eviction may synchronize the whole context. CUDA graphs must not retain references to any
    /// module owned by this launcher. These obligations also apply to later cleanup: the caller must ensure no stream
    /// in any context still retained by the launcher is capturing when the launcher is dropped. Explicit cleanup has
    /// the corresponding requirements documented on [`Self::clear_context`], [`Self::clear`], and [`Self::shutdown`].
    /// This unsafe boundary supports integration with externally owned CUDA contexts, streams, and device allocations.
    pub unsafe fn launch(&self, artifact: &CudaKernelArtifact, launch: &CudaKernelLaunch<'_>) -> Result<(), Error> {
        if self.is_shutdown.load(Ordering::Acquire) {
            return Err(Error::unavailable("cuda kernel launcher has been shut down"));
        }
        validate_launch(artifact, launch)?;
        let mut storage = launch.arguments().iter().map(CudaKernelArgumentStorage::from_argument).collect::<Vec<_>>();
        let context = self.driver.context_for_stream(launch.stream())?;
        let scope = CudaKernelCacheScope { context_id: context.id, device: context.device };
        let key = CudaKernelCacheKey { scope, content_address: artifact.content_address() };
        let kernel = {
            let mut kernels = self.kernels.lock().expect("CUDA kernel cache mutex is poisoned");
            let use_index = kernels.next_use_index();
            if let Some(entry) = kernels.entries.get_mut(&key) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                entry.last_used = use_index;
                entry.kernel.clone()
            } else {
                self.cache_misses.fetch_add(1, Ordering::Relaxed);
                let capability = self.driver.device_compute_capability(context.device)?;
                artifact.validate_device_compatibility(context.device, capability)?;
                Self::cleanup_pending_unloads(self.driver.as_ref(), &mut kernels, Some(scope))?;
                self.make_cache_space(&mut kernels, scope, artifact.bytes().len())?;
                let kernel = match self.driver.load_kernel(context, artifact) {
                    Ok(kernel) => Arc::new(kernel),
                    Err(error) => {
                        if let Some(kernel) = error.pending_unload {
                            kernels.pending_unloads.push(*kernel);
                        }
                        return Err(error.error);
                    }
                };
                kernels.entries.insert(key, CudaKernelCacheEntry { kernel: kernel.clone(), last_used: use_index });
                kernel
            }
        };

        let mut parameter_pointers = storage.iter_mut().map(CudaKernelArgumentStorage::as_mut_ptr).collect::<Vec<_>>();
        let parameter_pointers =
            if parameter_pointers.is_empty() { std::ptr::null_mut() } else { parameter_pointers.as_mut_ptr() };
        let dimensions = artifact.launch_dimensions();
        self.driver.launch_kernel(&kernel, dimensions, launch.stream(), parameter_pointers)
    }

    /// Synchronizes and unloads every cached module while leaving the launcher available for future launches.
    ///
    /// Failed entries remain cached so cleanup can be retried. Call this function before destroying the external
    /// runtimes that own contexts observed by [`Self::launch`].
    ///
    /// # Safety
    ///
    /// Every external runtime owning a retained context must remain alive, and no stream in any retained context may
    /// be capturing a CUDA graph. Context-wide synchronization requires exclusive coordination with those runtimes;
    /// this function is exposed for integrations that control their externally owned contexts' lifecycles.
    pub unsafe fn clear(&mut self) -> Result<(), Error> {
        self.clear_scope(None)
    }

    /// Synchronizes and unloads modules belonging to the current context identified by `stream`.
    ///
    /// Other contexts and their pending cleanup failures are untouched. A successful call releases this launcher's
    /// ownership obligations for the selected context until a subsequent launch observes it again. Failed entries
    /// remain retained for retry. The stream must belong to the current context and must not be capturing a graph.
    ///
    /// # Safety
    ///
    /// The selected context and stream must remain alive throughout this call, and no stream in that context may be
    /// capturing a CUDA graph. This function supports releasing one externally owned runtime from a shared launcher
    /// before that runtime destroys its context; other retained contexts must still obey [`Self::launch`]'s contract.
    pub unsafe fn clear_context(&mut self, stream: CudaStream<'_>) -> Result<(), Error> {
        let context = self.driver.context_for_stream(stream.as_raw())?;
        self.clear_scope(Some(CudaKernelCacheScope { context_id: context.id, device: context.device }))
    }

    /// Permanently disables launches, then synchronizes and unloads every cached module.
    ///
    /// Failed entries remain cached so this function can be called again to retry cleanup. Call this function before
    /// destroying the external runtimes that own contexts observed by [`Self::launch`].
    ///
    /// # Safety
    ///
    /// Every external runtime owning a retained context must remain alive, and no stream in any retained context may
    /// be capturing a CUDA graph. This function exposes deterministic cleanup for integrations controlling external
    /// resource lifecycles; it has the same synchronization obligations as [`Self::clear`].
    pub unsafe fn shutdown(&mut self) -> Result<(), Error> {
        self.is_shutdown.store(true, Ordering::Release);
        unsafe { self.clear() }
    }

    /// Releases all entries in the selected partition, or every partition for an explicit full cleanup.
    fn clear_scope(&mut self, scope: Option<CudaKernelCacheScope>) -> Result<(), Error> {
        let driver = self.driver.clone();
        let kernels = match self.kernels.get_mut() {
            Ok(kernels) => kernels,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut first_error = Self::cleanup_pending_unloads(driver.as_ref(), kernels, scope).err();
        let keys = kernels
            .entries
            .keys()
            .copied()
            .filter(|key| scope.is_none_or(|scope| key.scope == scope))
            .collect::<Vec<_>>();
        for key in keys {
            let entry = kernels.entries.remove(&key).unwrap();
            let kernel = Arc::try_unwrap(entry.kernel).unwrap_or_else(|_| {
                unreachable!("mutable launcher access guarantees no active cached-kernel borrowers")
            });
            if let Err(unload_error) = driver.unload_kernel(&kernel) {
                if unload_error.module_is_loaded {
                    kernels
                        .entries
                        .insert(key, CudaKernelCacheEntry { kernel: Arc::new(kernel), last_used: entry.last_used });
                }
                if first_error.is_none() {
                    first_error = Some(unload_error.error);
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Retries retained partial-load cleanup only within the requested failure domain.
    fn cleanup_pending_unloads(
        driver: &dyn CudaDriverApi,
        kernels: &mut CudaKernelCache,
        scope: Option<CudaKernelCacheScope>,
    ) -> Result<(), Error> {
        let pending_unloads = std::mem::take(&mut kernels.pending_unloads);
        let mut first_error = None;
        for kernel in pending_unloads {
            if scope.is_some_and(|scope| kernel.context.id != scope.context_id || kernel.context.device != scope.device)
            {
                kernels.pending_unloads.push(kernel);
                continue;
            }
            if let Err(unload_error) = driver.unload_kernel(&kernel) {
                if unload_error.module_is_loaded {
                    kernels.pending_unloads.push(kernel);
                }
                if first_error.is_none() {
                    first_error = Some(unload_error.error);
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Evicts the least recently used unborrowed entries until the partition admits a new artifact.
    fn make_cache_space(
        &self,
        kernels: &mut CudaKernelCache,
        scope: CudaKernelCacheScope,
        artifact_bytes: usize,
    ) -> Result<(), Error> {
        if artifact_bytes > kernels.limits.max_artifact_bytes_per_context_device {
            return Err(Error::unavailable(format!(
                "cuda kernel artifact contains {artifact_bytes} bytes, exceeding the per-context/device cache budget \
                 of {} bytes",
                kernels.limits.max_artifact_bytes_per_context_device,
            )));
        }
        loop {
            let (entry_count, represented_artifact_bytes) = kernels.scope_usage(scope);
            if entry_count < kernels.limits.max_entries_per_context_device
                && represented_artifact_bytes.saturating_add(artifact_bytes)
                    <= kernels.limits.max_artifact_bytes_per_context_device
            {
                return Ok(());
            }
            let key = kernels
                .entries
                .iter()
                .filter(|(key, entry)| key.scope == scope && Arc::strong_count(&entry.kernel) == 1)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key)
                .ok_or_else(|| {
                    Error::unavailable(
                        "cuda kernel cache partition is full and all cached modules in that context/device are in use",
                    )
                })?;
            let entry = kernels.entries.remove(&key).unwrap();
            let kernel = Arc::try_unwrap(entry.kernel)
                .unwrap_or_else(|_| unreachable!("the selected cached module has no concurrent borrowers"));
            if let Err(unload_error) = self.driver.unload_kernel(&kernel) {
                if unload_error.module_is_loaded {
                    kernels
                        .entries
                        .insert(key, CudaKernelCacheEntry { kernel: Arc::new(kernel), last_used: entry.last_used });
                } else {
                    self.cache_evictions.fetch_add(1, Ordering::Relaxed);
                }
                return Err(unload_error.error);
            }
            self.cache_evictions.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Drop for CudaKernelLauncher {
    fn drop(&mut self) {
        // Drop cannot report cleanup failures. Context destruction owns any modules that could not be unloaded here.
        if let Err(error) = unsafe { self.clear() } {
            error.record_cleanup_error();
        }
    }
}

/// Identity of one independently budgeted CUDA context/device cache partition.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct CudaKernelCacheScope {
    /// Unique context identity unaffected by raw-handle reuse.
    context_id: u64,
    /// CUDA device ordinal associated with the context.
    device: i32,
}

/// Identity of one loaded symbol within a context/device partition.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct CudaKernelCacheKey {
    /// Independently budgeted context/device owner.
    scope: CudaKernelCacheScope,
    /// Digest identifying loaded image bytes and kernel symbol.
    content_address: CudaKernelContentAddress,
}

/// Loaded modules and failed partial loads governed by shared partition budgets.
struct CudaKernelCache {
    /// Successfully loaded symbols available for reuse.
    entries: HashMap<CudaKernelCacheKey, CudaKernelCacheEntry>,
    /// Partially loaded modules whose cleanup failed and must be retried.
    pending_unloads: Vec<CudaLoadedKernel>,
    /// Entry and source-byte budgets applied independently to each partition.
    limits: CudaKernelCacheLimits,
    /// Logical clock used to distinguish recent accesses.
    next_use_index: u64,
}

impl CudaKernelCache {
    /// Creates an empty cache with the provided partition budgets.
    fn new(limits: CudaKernelCacheLimits) -> Self {
        Self { entries: HashMap::new(), pending_unloads: Vec::new(), limits, next_use_index: 0 }
    }

    /// Advances the logical access clock used to order eviction candidates.
    fn next_use_index(&mut self) -> u64 {
        let use_index = self.next_use_index;
        self.next_use_index = self.next_use_index.wrapping_add(1);
        use_index
    }

    /// Counts retained resources, including failed partial loads, in one partition.
    fn scope_usage(&self, scope: CudaKernelCacheScope) -> (usize, usize) {
        self.entries
            .iter()
            .filter(|(key, _)| key.scope == scope)
            .map(|(_, entry)| entry.kernel.artifact_bytes)
            .chain(
                self.pending_unloads
                    .iter()
                    .filter(|kernel| kernel.context.id == scope.context_id && kernel.context.device == scope.device)
                    .map(|kernel| kernel.artifact_bytes),
            )
            .fold((0, 0usize), |(entry_count, artifact_bytes), entry_bytes| {
                (entry_count + 1, artifact_bytes.saturating_add(entry_bytes))
            })
    }
}

/// A shared module borrow and its last cache access.
struct CudaKernelCacheEntry {
    /// Retains module ownership while callers enqueue using the resolved function.
    kernel: Arc<CudaLoadedKernel>,
    /// Last logical clock value assigned by a cache lookup or admission.
    last_used: u64,
}

#[cfg(test)]
mod tests {
    use std::backtrace::Backtrace;
    use std::collections::HashSet;
    use std::ffi::{CStr, c_void};
    use std::process::Command;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::thread::JoinHandle;
    use std::time::Duration;

    use indoc::indoc;
    use libloading::Library;
    use pretty_assertions::assert_eq;

    use crate::artifacts::CudaComputeCapability;
    use crate::drivers::{CudaContext, CudaKernelLoadError, CudaUnloadError};
    use crate::tests::{CLEANUP_ERROR_TEST_LOCK, test_artifact, test_artifact_with_bytes, test_cubin};
    use crate::{
        CudaArtifactFormat, CudaDevicePointer, CudaKernelAbi, CudaKernelArgument, CudaKernelLaunchDimensions,
        CudaKernelParameterType, CudaScalarType, CudaScalarValue,
    };

    use super::*;

    impl CudaKernelLauncher {
        /// Constructs a launcher backed by the deterministic test driver.
        fn with_driver(driver: Arc<dyn CudaDriverApi>, cache_capacity_per_context_device: usize) -> Self {
            let limits = CudaKernelCacheLimits::new(
                cache_capacity_per_context_device,
                DEFAULT_KERNEL_CACHE_ARTIFACT_BYTES_PER_CONTEXT_DEVICE,
            )
            .unwrap();
            Self::with_driver_for_version(driver, limits, CudaVersion::from_encoded(12_090).unwrap())
        }

        /// Constructs a test launcher with explicit resource budgets.
        fn with_driver_and_limits(driver: Arc<dyn CudaDriverApi>, limits: CudaKernelCacheLimits) -> Self {
            Self::with_driver_for_version(driver, limits, CudaVersion::from_encoded(12_090).unwrap())
        }

        /// Counts both usable entries and retained partial loads for lifecycle assertions.
        fn cached_kernel_count(&self) -> usize {
            let kernels = self.kernels.lock().unwrap();
            kernels.entries.len() + kernels.pending_unloads.len()
        }
    }

    /// Values copied synchronously from the launch argument pointer array.
    #[derive(Clone, Debug, PartialEq)]
    enum RecordedArgument {
        DevicePointer(usize),
        I32(i32),
        F64(f64),
    }

    /// Bounded rendezvous that holds a module borrow while a competing launch exercises eviction.
    struct TestLaunchGate {
        /// Notifies the controlling test that the driver holds the active borrow.
        started: Sender<()>,
        /// Releases the driver after the competing cache operation finishes.
        resume: Receiver<()>,
    }

    /// Deterministic CUDA policy fixture with injected failures and recorded resource operations.
    struct TestCudaDriver {
        context_id: AtomicU64,
        device: AtomicI32,
        context_matches_stream: AtomicBool,
        stream_is_capturing: AtomicBool,
        context_from_stream: AtomicBool,
        load_count: AtomicUsize,
        launch_count: AtomicUsize,
        unload_count: AtomicUsize,
        fail_after_module_load: AtomicBool,
        fail_unload: AtomicBool,
        fail_after_unload: AtomicBool,
        compute_capability: Mutex<CudaComputeCapability>,
        supported_architecture: Mutex<String>,
        parameter_types: Mutex<Vec<CudaKernelParameterType>>,
        recorded_arguments: Mutex<Vec<RecordedArgument>>,
        recorded_launch: Mutex<Option<(CudaKernelLaunchDimensions, usize)>>,
        unloaded_modules: Mutex<Vec<usize>>,
        unload_contexts: Mutex<Vec<u64>>,
        fail_launch: AtomicBool,
        fail_unload_context: AtomicU64,
        launch_gate: Mutex<Option<TestLaunchGate>>,
    }

    impl Default for TestCudaDriver {
        fn default() -> Self {
            Self {
                context_id: AtomicU64::new(1),
                device: AtomicI32::new(0),
                context_matches_stream: AtomicBool::new(true),
                stream_is_capturing: AtomicBool::new(false),
                context_from_stream: AtomicBool::new(false),
                load_count: AtomicUsize::new(0),
                launch_count: AtomicUsize::new(0),
                unload_count: AtomicUsize::new(0),
                fail_after_module_load: AtomicBool::new(false),
                fail_unload: AtomicBool::new(false),
                fail_after_unload: AtomicBool::new(false),
                compute_capability: Mutex::new(CudaComputeCapability { major: 10, minor: 0 }),
                supported_architecture: Mutex::new("sm_100".to_string()),
                parameter_types: Mutex::new(Vec::new()),
                recorded_arguments: Mutex::new(Vec::new()),
                recorded_launch: Mutex::new(None),
                unloaded_modules: Mutex::new(Vec::new()),
                unload_contexts: Mutex::new(Vec::new()),
                fail_launch: AtomicBool::new(false),
                fail_unload_context: AtomicU64::new(0),
                launch_gate: Mutex::new(None),
            }
        }
    }

    impl CudaDriverApi for TestCudaDriver {
        fn context_for_stream(&self, stream: *mut c_void) -> Result<CudaContext, Error> {
            if self.stream_is_capturing.load(Ordering::SeqCst) {
                return Err(Error::unavailable("cuda kernel launches on capturing streams are unsupported"));
            }
            if !self.context_matches_stream.load(Ordering::SeqCst) {
                return Err(Error::invalid_argument("the cuda stream does not belong to the current cuda context"));
            }
            if self.context_from_stream.load(Ordering::SeqCst) {
                // Streams model distinct contexts: odd stream handles live on device 1 and even ones on device 0.
                let stream = stream as u64;
                return Ok(CudaContext {
                    handle: stream as usize as *mut c_void,
                    id: stream,
                    device: (stream % 2) as i32,
                });
            }
            Ok(CudaContext {
                handle: 16usize as *mut c_void,
                id: self.context_id.load(Ordering::SeqCst),
                device: self.device.load(Ordering::SeqCst),
            })
        }

        fn device_compute_capability(&self, _device: i32) -> Result<CudaComputeCapability, Error> {
            Ok(*self.compute_capability.lock().unwrap())
        }

        fn load_kernel(
            &self,
            context: CudaContext,
            artifact: &CudaKernelArtifact,
        ) -> Result<CudaLoadedKernel, CudaKernelLoadError> {
            if artifact.bytes().ends_with(b"malformed") {
                return Err(CudaKernelLoadError::new(Error::invalid_argument(
                    "cuda driver rejected a malformed cubin",
                )));
            }
            if artifact.target_architecture() != self.supported_architecture.lock().unwrap().as_str() {
                return Err(CudaKernelLoadError::new(Error::Driver {
                    operation: "cuModuleLoadDataEx".to_string(),
                    code: 209,
                    name: "CUDA_ERROR_NO_BINARY_FOR_GPU".to_string(),
                    message: "no kernel image is available for execution on the device".to_string(),
                    backtrace: Backtrace::capture().to_string(),
                }));
            }
            let module = self.load_count.fetch_add(1, Ordering::SeqCst) + 1;
            let kernel = CudaLoadedKernel {
                context,
                module: module as *mut c_void,
                function: module as *mut c_void,
                artifact_bytes: artifact.bytes().len(),
                max_dynamic_shared_memory_bytes: u32::MAX,
            };
            if self.fail_after_module_load.load(Ordering::SeqCst) {
                let load_error = Error::internal("injected cuda symbol lookup failure");
                return match self.unload_kernel(&kernel) {
                    Ok(()) => Err(CudaKernelLoadError::new(load_error)),
                    Err(unload_error) => Err(CudaKernelLoadError {
                        error: unload_error.error,
                        pending_unload: unload_error.module_is_loaded.then(|| Box::new(kernel)),
                    }),
                };
            }
            Ok(kernel)
        }

        fn launch_kernel(
            &self,
            _kernel: &CudaLoadedKernel,
            dimensions: CudaKernelLaunchDimensions,
            stream: *mut c_void,
            parameters: *mut *mut c_void,
        ) -> Result<(), Error> {
            let gate = self.launch_gate.lock().unwrap().take();
            if let Some(gate) = gate {
                gate.started.send(()).unwrap();
                gate.resume.recv_timeout(Duration::from_secs(5)).unwrap();
            }
            *self.recorded_launch.lock().unwrap() = Some((dimensions, stream as usize));
            if self.fail_launch.load(Ordering::SeqCst) {
                return Err(Error::internal("injected cuda kernel launch failure"));
            }
            let parameter_types = self.parameter_types.lock().unwrap();
            assert_eq!(parameters.is_null(), parameter_types.is_empty());
            let mut recorded_arguments = Vec::with_capacity(parameter_types.len());
            for (index, parameter_type) in parameter_types.iter().enumerate() {
                // The production driver receives the same pointer array and synchronously copies each launch value.
                let parameter = unsafe { *parameters.add(index) };
                recorded_arguments.push(match parameter_type {
                    CudaKernelParameterType::DevicePointer => {
                        RecordedArgument::DevicePointer(unsafe { *(parameter as *const *mut c_void) } as usize)
                    }
                    CudaKernelParameterType::Scalar(CudaScalarType::I32) => {
                        RecordedArgument::I32(unsafe { *(parameter as *const i32) })
                    }
                    CudaKernelParameterType::Scalar(CudaScalarType::F64) => {
                        RecordedArgument::F64(unsafe { *(parameter as *const f64) })
                    }
                    other => panic!("unsupported test parameter type: {other:?}"),
                });
            }
            *self.recorded_arguments.lock().unwrap() = recorded_arguments;
            self.launch_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn unload_kernel(&self, kernel: &CudaLoadedKernel) -> Result<(), CudaUnloadError> {
            self.unload_count.fetch_add(1, Ordering::SeqCst);
            self.unload_contexts.lock().unwrap().push(kernel.context.id);
            if self.fail_unload.load(Ordering::SeqCst)
                || self.fail_unload_context.load(Ordering::SeqCst) == kernel.context.id
            {
                Err(CudaUnloadError::retained(Error::internal("injected cuda module unload failure")))
            } else if self.fail_after_unload.load(Ordering::SeqCst) {
                self.unloaded_modules.lock().unwrap().push(kernel.module as usize);
                Err(CudaUnloadError::unloaded(Error::internal("injected cuda context restoration failure")))
            } else {
                self.unloaded_modules.lock().unwrap().push(kernel.module as usize);
                Ok(())
            }
        }
    }

    /// Creates an empty execution frame on the default synthetic stream.
    fn test_launch() -> CudaKernelLaunch<'static> {
        test_launch_on_stream(16)
    }

    /// Creates an empty execution frame on a selected synthetic stream.
    fn test_launch_on_stream(stream: usize) -> CudaKernelLaunch<'static> {
        let stream = unsafe { CudaStream::from_raw(stream as *mut c_void) }.unwrap();
        CudaKernelLaunch::new(stream, Vec::new())
    }

    /// Starts a launch and waits with a deadline until its module borrow is active inside the driver.
    /// The fixture owns synthetic contexts that remain valid until the launch thread and launcher are dropped.
    fn start_blocked_launch(
        launcher: &Arc<CudaKernelLauncher>,
        driver: &TestCudaDriver,
        artifact: &CudaKernelArtifact,
    ) -> (JoinHandle<Result<(), Error>>, Sender<()>) {
        let (started, ready) = mpsc::channel();
        let (resume, release) = mpsc::channel();
        *driver.launch_gate.lock().unwrap() = Some(TestLaunchGate { started, resume: release });
        let launcher = launcher.clone();
        let artifact = artifact.clone();
        let thread = std::thread::spawn(move || unsafe { launcher.launch(&artifact, &test_launch()) });
        ready.recv_timeout(Duration::from_secs(5)).unwrap();
        (thread, resume)
    }

    /// Unique suffix for concurrent compiler output directories.
    static NEXT_COMPILATION: AtomicU64 = AtomicU64::new(0);

    /// An owned CUDA context, stream, and one output allocation used only by the real-driver tests.
    ///
    /// Each fixture creates a separate context so tests can run concurrently without sharing capture state. Launchers
    /// are explicitly shut down before fixture destruction, and every output read first synchronizes the stream.
    struct TestContext {
        /// Keeps test-only entry points loaded until all resources have been destroyed.
        library: Library,
        /// CUDA context owned by this fixture.
        context: *mut c_void,
        /// Context current before this fixture was constructed.
        previous_context: *mut c_void,
        /// Nonblocking stream owned by this fixture.
        stream: *mut c_void,
        /// Device allocation containing one native-endian `u32`.
        output: u64,
        /// Compute capability of the selected device.
        capability: u32,
    }

    impl TestContext {
        /// Creates an isolated context on device zero when a supported CUDA device is available.
        fn new() -> Option<Self> {
            match std::env::consts::OS {
                "linux" => {}
                platform => {
                    eprintln!("skipping CUDA execution test: fixture does not support {platform}");
                    return None;
                }
            }
            // Each symbol's concrete type below follows the installed CUDA Driver API declaration.
            unsafe {
                let library = match Library::new("libcuda.so.1") {
                    Ok(library) => library,
                    Err(error) => {
                        eprintln!("skipping CUDA execution test: NVIDIA driver unavailable: {error}");
                        return None;
                    }
                };
                let initialize = load::<unsafe extern "C" fn(u32) -> i32>(&library, c"cuInit");
                match initialize(0) {
                    0 => {}
                    100 => {
                        eprintln!("skipping CUDA execution test: no CUDA device");
                        return None;
                    }
                    status => panic!("CUDA initialization failed with status {status}"),
                }
                let get_version = load::<unsafe extern "C" fn(*mut i32) -> i32>(&library, c"cuDriverGetVersion");
                let mut version = 0;
                assert_eq!(get_version(&mut version), 0);
                if version < 12_000 {
                    eprintln!("skipping CUDA execution test: CUDA 12 or newer is required");
                    return None;
                }
                let get_device = load::<unsafe extern "C" fn(*mut i32, i32) -> i32>(&library, c"cuDeviceGet");
                let mut device = -1;
                assert_eq!(get_device(&mut device, 0), 0);
                let get_attribute =
                    load::<unsafe extern "C" fn(*mut i32, i32, i32) -> i32>(&library, c"cuDeviceGetAttribute");
                let mut major = 0;
                let mut minor = 0;
                assert_eq!(get_attribute(&mut major, 75, device), 0);
                assert_eq!(get_attribute(&mut minor, 76, device), 0);
                if major < 8 {
                    eprintln!("skipping CUDA execution test: compute capability 8.0 or newer is required");
                    return None;
                }
                let get_current = load::<unsafe extern "C" fn(*mut *mut c_void) -> i32>(&library, c"cuCtxGetCurrent");
                let mut previous_context = std::ptr::null_mut();
                assert_eq!(get_current(&mut previous_context), 0);
                let create_context =
                    load::<unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32>(&library, c"cuCtxCreate_v2");
                let mut context = std::ptr::null_mut();
                assert_eq!(create_context(&mut context, 0, device), 0);
                let create_stream =
                    load::<unsafe extern "C" fn(*mut *mut c_void, u32) -> i32>(&library, c"cuStreamCreate");
                let mut stream = std::ptr::null_mut();
                assert_eq!(create_stream(&mut stream, 1), 0);
                let allocate = load::<unsafe extern "C" fn(*mut u64, usize) -> i32>(&library, c"cuMemAlloc_v2");
                let mut output = 0;
                assert_eq!(allocate(&mut output, size_of::<u32>()), 0);
                Some(Self {
                    library,
                    context,
                    previous_context,
                    stream,
                    output,
                    capability: (major * 10 + minor) as u32,
                })
            }
        }

        /// Borrows the fixture's explicitly owned non-default stream.
        fn stream(&self) -> CudaStream<'_> {
            unsafe { CudaStream::from_raw(self.stream) }.unwrap()
        }

        /// Borrows the output allocation and copies a scalar into a launch frame.
        fn launch(&self, value: u32) -> CudaKernelLaunch<'_> {
            CudaKernelLaunch::new(
                self.stream(),
                [
                    CudaKernelArgument::DevicePointer(
                        unsafe { CudaDevicePointer::from_raw(self.output as usize as *mut c_void) }.unwrap(),
                    ),
                    CudaKernelArgument::Scalar(CudaScalarValue::U32(value)),
                ],
            )
        }

        /// Synchronizes execution, checks completion state, and copies the output to host memory.
        fn read(&self) -> u32 {
            unsafe {
                let synchronize =
                    load::<unsafe extern "C" fn(*mut c_void) -> i32>(&self.library, c"cuStreamSynchronize");
                assert_eq!(synchronize(self.stream), 0);
                let query = load::<unsafe extern "C" fn(*mut c_void) -> i32>(&self.library, c"cuStreamQuery");
                assert_eq!(query(self.stream), 0);
                let copy =
                    load::<unsafe extern "C" fn(*mut c_void, u64, usize) -> i32>(&self.library, c"cuMemcpyDtoH_v2");
                let mut bytes = [0u8; 4];
                assert_eq!(copy(bytes.as_mut_ptr().cast(), self.output, bytes.len()), 0);
                u32::from_ne_bytes(bytes)
            }
        }
    }

    impl Drop for TestContext {
        fn drop(&mut self) {
            // Tests shut launchers down before this fixture. Cleanup failures must fail the GPU test.
            unsafe {
                let synchronize = load::<unsafe extern "C" fn() -> i32>(&self.library, c"cuCtxSynchronize");
                assert_eq!(synchronize(), 0);
                let free = load::<unsafe extern "C" fn(u64) -> i32>(&self.library, c"cuMemFree_v2");
                assert_eq!(free(self.output), 0);
                let destroy_stream =
                    load::<unsafe extern "C" fn(*mut c_void) -> i32>(&self.library, c"cuStreamDestroy_v2");
                assert_eq!(destroy_stream(self.stream), 0);
                let destroy_context =
                    load::<unsafe extern "C" fn(*mut c_void) -> i32>(&self.library, c"cuCtxDestroy_v2");
                assert_eq!(destroy_context(self.context), 0);
                let set_current = load::<unsafe extern "C" fn(*mut c_void) -> i32>(&self.library, c"cuCtxSetCurrent");
                assert_eq!(set_current(self.previous_context), 0);
            }
        }
    }

    /// Loads a test-only CUDA entry point.
    ///
    /// # Safety
    ///
    /// `T` must be the exact function-pointer type of `name`, and the library must outlive all uses of the pointer.
    unsafe fn load<T: Copy>(library: &Library, name: &CStr) -> T {
        *unsafe { library.get::<T>(name.to_bytes_with_nul()) }.unwrap()
    }

    /// Creates the two-parameter ABI shared by the small GPU kernels.
    fn test_abi() -> CudaKernelAbi {
        CudaKernelAbi::new(
            "ryft.gpu-test",
            1,
            [CudaKernelParameterType::DevicePointer, CudaKernelParameterType::Scalar(CudaScalarType::U32)],
        )
        .unwrap()
    }

    /// Creates a PTX kernel that adds one and optionally uses the last word of a 64-KiB shared-memory allocation.
    fn test_ptx(shared_memory: bool) -> CudaKernelArtifact {
        let body = if shared_memory {
            "st.shared.u32 [scratch+65532], value; ld.shared.u32 value, [scratch+65532];"
        } else {
            ""
        };
        let source = format!(
            indoc! {"
                .version 8.0
                .target sm_80
                .address_size 64
                .extern .shared .align 4 .b8 scratch[];
                .visible .entry add_one(.param .u64 output, .param .u32 input) {{
                    .reg .u64 address;
                    .reg .u32 value;
                    ld.param.u64 address, [output];
                    ld.param.u32 value, [input];
                    {body}
                    add.u32 value, value, 1;
                    st.global.u32 [address], value;
                    ret;
                }}
            "},
            body = body,
        );
        CudaKernelArtifact::new(
            CudaArtifactFormat::Ptx,
            source.into_bytes(),
            "add_one",
            "compute_80",
            CudaKernelLaunchDimensions::new([1; 3], [1; 3], if shared_memory { 65_536 } else { 0 }).unwrap(),
            test_abi(),
        )
        .unwrap()
    }

    #[test]
    fn test_cuda_kernel_cache_limits_new() {
        assert_eq!(
            CudaKernelCacheLimits::new(3, 4096),
            Ok(CudaKernelCacheLimits {
                max_entries_per_context_device: 3,
                max_artifact_bytes_per_context_device: 4096,
            }),
        );
        assert!(matches!(
            CudaKernelCacheLimits::new(0, 1),
            Err(Error::InvalidArgument { message, .. })
                if message == "cuda kernel cache entries per context/device must be positive",
        ));
        assert!(matches!(
            CudaKernelCacheLimits::new(1, 0),
            Err(Error::InvalidArgument { message, .. })
                if message == "cuda kernel cache artifact bytes per context/device must be positive",
        ));
    }

    #[test]
    fn test_cuda_kernel_cache_limits_max_entries_per_context_device() {
        assert_eq!(CudaKernelCacheLimits::new(3, 4096).unwrap().max_entries_per_context_device(), 3);
    }

    #[test]
    fn test_cuda_kernel_cache_limits_max_artifact_bytes_per_context_device() {
        assert_eq!(CudaKernelCacheLimits::new(3, 4096).unwrap().max_artifact_bytes_per_context_device(), 4096);
    }

    #[test]
    fn test_cuda_kernel_cache_limits_default() {
        assert_eq!(CudaKernelCacheLimits::default(), CudaKernelCacheLimits::new(128, 256 * 1024 * 1024).unwrap());
    }

    #[test]
    fn test_cuda_kernel_cache_statistics_hits() {
        assert_eq!(CudaKernelCacheStatistics { hits: 3, misses: 4, evictions: 2 }.hits(), 3);
    }

    #[test]
    fn test_cuda_kernel_cache_statistics_misses() {
        assert_eq!(CudaKernelCacheStatistics { hits: 3, misses: 4, evictions: 2 }.misses(), 4);
    }

    #[test]
    fn test_cuda_kernel_cache_statistics_evictions() {
        assert_eq!(CudaKernelCacheStatistics { hits: 3, misses: 4, evictions: 2 }.evictions(), 2);
    }

    #[test]
    fn test_cuda_kernel_launcher_new() {
        match CudaKernelLauncher::new(CudaVersion::from_encoded(12_000).unwrap()) {
            Ok(mut launcher) => {
                assert_eq!(launcher.cuda_version(), CudaVersion::from_encoded(12_000).unwrap());
                assert_eq!(launcher.kernels.lock().unwrap().limits, CudaKernelCacheLimits::default());
                unsafe { launcher.shutdown() }.unwrap();
            }
            Err(error) => assert!(matches!(error, Error::Unavailable { .. } | Error::Driver { .. })),
        }
    }

    #[test]
    fn test_cuda_kernel_launcher_new_gpu() {
        let Some(_context) = TestContext::new() else { return };
        let version = CudaVersion::from_encoded(12_000).unwrap();
        let mut launcher = CudaKernelLauncher::new(version).unwrap();
        assert_eq!(launcher.cuda_version(), version);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_with_cache_capacity() {
        match CudaKernelLauncher::with_cache_capacity(CudaVersion::from_encoded(12_000).unwrap(), 2) {
            Ok(launcher) => assert_eq!(
                launcher.kernels.lock().unwrap().limits,
                CudaKernelCacheLimits::new(2, DEFAULT_KERNEL_CACHE_ARTIFACT_BYTES_PER_CONTEXT_DEVICE).unwrap(),
            ),
            Err(error) => assert!(matches!(error, Error::Unavailable { .. } | Error::Driver { .. })),
        }
        assert!(matches!(
            CudaKernelLauncher::with_cache_capacity(CudaVersion::from_encoded(12_000).unwrap(), 0),
            Err(Error::InvalidArgument { message, .. })
                if message == "cuda kernel cache entries per context/device must be positive",
        ));
    }

    #[test]
    fn test_cuda_kernel_launcher_with_cache_limits() {
        // Driver availability is intentionally optional in portable tests; GPU integration tests require success.
        let limits = CudaKernelCacheLimits::new(3, 4096).unwrap();
        match CudaKernelLauncher::with_cache_limits(CudaVersion::from_encoded(12_000).unwrap(), limits) {
            Ok(launcher) => assert_eq!(launcher.kernels.lock().unwrap().limits, limits),
            Err(error) => assert!(matches!(error, Error::Unavailable { .. } | Error::Driver { .. })),
        }
    }

    #[test]
    fn test_cuda_kernel_launcher_cuda_version() {
        let version = CudaVersion::from_encoded(13_000).unwrap();
        let launcher = CudaKernelLauncher::with_driver_for_version(
            Arc::new(TestCudaDriver::default()),
            CudaKernelCacheLimits::default(),
            version,
        );
        assert_eq!(launcher.cuda_version(), version);
    }

    #[test]
    fn test_cuda_kernel_launcher_cache_statistics() {
        let driver = Arc::new(TestCudaDriver::default());
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 4);
        let artifact = test_artifact(Vec::new());
        let reconstructed_artifact = test_artifact(Vec::new());
        let launch = test_launch();

        assert_eq!(unsafe { launcher.launch(&artifact, &launch) }, Ok(()));
        assert_eq!(unsafe { launcher.launch(&reconstructed_artifact, &launch) }, Ok(()));
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 1);
        assert_eq!(launcher.cached_kernel_count(), 1);
        assert_eq!(launcher.cache_statistics().hits(), 1);
        assert_eq!(launcher.cache_statistics().misses(), 1);
        assert_eq!(launcher.cache_statistics().evictions(), 0);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_launch() {
        let driver = Arc::new(TestCudaDriver::default());
        *driver.parameter_types.lock().unwrap() = vec![
            CudaKernelParameterType::DevicePointer,
            CudaKernelParameterType::Scalar(CudaScalarType::I32),
            CudaKernelParameterType::Scalar(CudaScalarType::F64),
        ];
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 4);
        let artifact = test_artifact(driver.parameter_types.lock().unwrap().clone());
        let stream = unsafe { CudaStream::from_raw(16usize as *mut c_void) }.unwrap();
        let device_pointer = unsafe { CudaDevicePointer::from_raw(0x1234usize as *mut c_void) }.unwrap();
        let launch = CudaKernelLaunch::new(
            stream,
            vec![
                CudaKernelArgument::DevicePointer(device_pointer),
                CudaKernelArgument::Scalar(CudaScalarValue::I32(-7)),
                CudaKernelArgument::Scalar(CudaScalarValue::F64(3.5)),
            ],
        );

        assert_eq!(unsafe { launcher.launch(&artifact, &launch) }, Ok(()));
        assert_eq!(
            *driver.recorded_arguments.lock().unwrap(),
            vec![RecordedArgument::DevicePointer(0x1234), RecordedArgument::I32(-7), RecordedArgument::F64(3.5)],
        );
        assert_eq!(*driver.recorded_launch.lock().unwrap(), Some((artifact.launch_dimensions(), 16)));
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_partition_limits() {
        let driver = Arc::new(TestCudaDriver::default());
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 1);
        let first_artifact = test_artifact(Vec::new());
        let second_artifact = test_artifact_with_bytes(test_cubin(100, &[4, 5, 6]), Vec::new());
        let third_artifact = test_artifact_with_bytes(test_cubin(100, &[7, 8, 9]), Vec::new());
        let launch = test_launch();

        assert_eq!(unsafe { launcher.launch(&first_artifact, &launch) }, Ok(()));
        driver.context_id.store(2, Ordering::SeqCst);
        driver.device.store(1, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&second_artifact, &launch) }, Ok(()));
        assert_eq!(launcher.cached_kernel_count(), 2);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 0);

        driver.context_id.store(1, Ordering::SeqCst);
        driver.device.store(0, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&third_artifact, &launch) }, Ok(()));
        assert_eq!(launcher.cached_kernel_count(), 2);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 1);
        assert_eq!(launcher.cache_statistics().misses(), 3);
        assert_eq!(launcher.cache_statistics().evictions(), 1);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_artifact_byte_budget() {
        let driver = Arc::new(TestCudaDriver::default());
        // Synthetic cubins carry a 64-byte ELF header, so the budget admits one 67-byte or one 68-byte artifact.
        let limits = CudaKernelCacheLimits::new(4, 69).unwrap();
        let mut launcher = CudaKernelLauncher::with_driver_and_limits(driver.clone(), limits);
        let first_artifact = test_artifact(Vec::new());
        let second_artifact = test_artifact_with_bytes(test_cubin(100, &[4, 5, 6, 7]), Vec::new());
        let oversized_artifact = test_artifact_with_bytes(test_cubin(100, &[8, 9, 10, 11, 12, 13]), Vec::new());
        let launch = test_launch();

        assert_eq!(unsafe { launcher.launch(&first_artifact, &launch) }, Ok(()));
        assert_eq!(unsafe { launcher.launch(&second_artifact, &launch) }, Ok(()));
        assert_eq!(launcher.cached_kernel_count(), 1);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 1);
        assert!(matches!(
            unsafe { launcher.launch(&oversized_artifact, &launch) },
            Err(Error::Unavailable { message, .. })
                if message == "cuda kernel artifact contains 70 bytes, exceeding the per-context/device cache budget \
                               of 69 bytes",
        ));
        assert_eq!(launcher.cached_kernel_count(), 1);
        assert_eq!(launcher.cache_statistics().hits(), 0);
        assert_eq!(launcher.cache_statistics().misses(), 3);
        assert_eq!(launcher.cache_statistics().evictions(), 1);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_invalid_artifacts() {
        let driver = Arc::new(TestCudaDriver::default());
        let launcher = CudaKernelLauncher::with_driver(driver.clone(), 4);
        let dimensions = CudaKernelLaunchDimensions::new([1, 1, 1], [1, 1, 1], 0).unwrap();
        let launch = test_launch();

        // Driver-level rejection of an image that passed static inspection.
        let malformed = CudaKernelArtifact::new(
            CudaArtifactFormat::Cubin,
            test_cubin(100, b"malformed"),
            "kernel",
            "sm_100",
            dimensions,
            CudaKernelAbi::new("test", 1, Vec::new()).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            unsafe { launcher.launch(&malformed, &launch) },
            Err(Error::InvalidArgument { message, .. }) if message == "cuda driver rejected a malformed cubin",
        ));

        // A cubin for another SM is rejected before the driver sees it, naming the artifact target and the device.
        let wrong_architecture = CudaKernelArtifact::new(
            CudaArtifactFormat::Cubin,
            test_cubin(90, &[1]),
            "kernel",
            "sm_90",
            dimensions,
            CudaKernelAbi::new("test", 1, Vec::new()).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            unsafe { launcher.launch(&wrong_architecture, &launch) },
            Err(Error::InvalidArgument { message, .. })
                if message == "cuda cubin targets `sm_90`, but cuda device 0 has compute capability 10.0",
        ));

        // PTX for a newer virtual architecture than the device is rejected before the driver sees it, while PTX for an
        // older one passes the prelaunch check and still surfaces the driver diagnostic when the driver rejects it.
        let ptx = |target: &str| {
            CudaKernelArtifact::new(
                CudaArtifactFormat::Ptx,
                indoc!(".version 8.0").as_bytes().to_vec(),
                "kernel",
                target,
                dimensions,
                CudaKernelAbi::new("test", 1, Vec::new()).unwrap(),
            )
            .unwrap()
        };
        assert!(matches!(
            unsafe { launcher.launch(&ptx("compute_120"), &launch) },
            Err(Error::InvalidArgument { message, .. })
                if message == "cuda PTX targets `compute_120`, but cuda device 0 has compute capability 10.0",
        ));
        assert!(matches!(
            unsafe { launcher.launch(&ptx("compute_90"), &launch) },
            Err(Error::Driver { code: 209, name, .. }) if name == "CUDA_ERROR_NO_BINARY_FOR_GPU",
        ));
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 0);
        assert_eq!(launcher.cached_kernel_count(), 0);
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_context_identity() {
        let driver = Arc::new(TestCudaDriver::default());
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 4);
        let artifact = test_artifact(Vec::new());
        let launch = test_launch();
        driver.context_matches_stream.store(false, Ordering::SeqCst);
        assert!(matches!(
            unsafe { launcher.launch(&artifact, &launch) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the cuda stream does not belong to the current cuda context",
        ));
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 0);

        driver.context_matches_stream.store(true, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&artifact, &launch) }, Ok(()));
        driver.context_id.store(2, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&artifact, &launch) }, Ok(()));
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 2);
        assert_eq!(launcher.cached_kernel_count(), 2);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_eviction_restoration_failure() {
        let driver = Arc::new(TestCudaDriver::default());
        let launcher = CudaKernelLauncher::with_driver(driver.clone(), 1);
        let first_artifact = test_artifact(Vec::new());
        let second_artifact = test_artifact_with_bytes(test_cubin(100, &[4, 5, 6]), Vec::new());
        let launch = test_launch();

        assert_eq!(unsafe { launcher.launch(&first_artifact, &launch) }, Ok(()));
        driver.fail_after_unload.store(true, Ordering::SeqCst);
        assert!(matches!(
            unsafe { launcher.launch(&second_artifact, &launch) },
            Err(Error::Internal { message, .. }) if message == "injected cuda context restoration failure",
        ));
        assert_eq!(launcher.cached_kernel_count(), 0);
        assert_eq!(launcher.cache_statistics().misses(), 2);
        assert_eq!(launcher.cache_statistics().evictions(), 1);
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_pending_cleanup_retry() {
        let driver = Arc::new(TestCudaDriver::default());
        driver.fail_after_module_load.store(true, Ordering::SeqCst);
        driver.fail_unload.store(true, Ordering::SeqCst);
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 1);
        let artifact = test_artifact(Vec::new());
        let launch = test_launch();

        assert!(matches!(
            unsafe { launcher.launch(&artifact, &launch) },
            Err(Error::Internal { message, .. }) if message == "injected cuda module unload failure",
        ));
        assert_eq!(launcher.cached_kernel_count(), 1);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 1);

        assert!(matches!(
            unsafe { launcher.launch(&artifact, &launch) },
            Err(Error::Internal { message, .. }) if message == "injected cuda module unload failure",
        ));
        assert_eq!(launcher.cached_kernel_count(), 1);
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 1);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 2);

        driver.fail_after_module_load.store(false, Ordering::SeqCst);
        driver.fail_unload.store(false, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&artifact, &launch) }, Ok(()));
        assert_eq!(launcher.cached_kernel_count(), 1);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 3);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
        assert_eq!(launcher.cached_kernel_count(), 0);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_concurrent_single_load() {
        let driver = Arc::new(TestCudaDriver::default());
        let launcher = Arc::new(CudaKernelLauncher::with_driver(driver.clone(), 4));
        let artifact = Arc::new(test_artifact(Vec::new()));
        let threads = (0..8)
            .map(|_| {
                let launcher = launcher.clone();
                let artifact = artifact.clone();
                std::thread::spawn(move || {
                    let launch = test_launch();
                    assert_eq!(unsafe { launcher.launch(&artifact, &launch) }, Ok(()));
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 1);
        assert_eq!(driver.launch_count.load(Ordering::SeqCst), 8);
        drop(launcher);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_concurrent_partitions() {
        // Four artifacts launched from eight threads over three streams that resolve to distinct contexts/devices must
        // load each artifact exactly once per context/device partition and unload everything on drop.
        let driver = Arc::new(TestCudaDriver::default());
        driver.context_from_stream.store(true, Ordering::SeqCst);
        let launcher = Arc::new(CudaKernelLauncher::with_driver(driver.clone(), 8));
        let artifacts = Arc::new(
            (0u8..4)
                .map(|index| test_artifact_with_bytes(test_cubin(100, &[index]), Vec::new()))
                .collect::<Vec<_>>(),
        );
        let streams = [16usize, 17, 18];
        let threads = (0..8)
            .map(|thread| {
                let launcher = launcher.clone();
                let artifacts = artifacts.clone();
                std::thread::spawn(move || {
                    for round in 0..4 {
                        for (index, artifact) in artifacts.iter().enumerate() {
                            let launch = test_launch_on_stream(streams[(thread + round + index) % streams.len()]);
                            assert_eq!(unsafe { launcher.launch(artifact, &launch) }, Ok(()));
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(driver.load_count.load(Ordering::SeqCst), artifacts.len() * streams.len());
        assert_eq!(driver.launch_count.load(Ordering::SeqCst), 8 * 4 * artifacts.len());
        assert_eq!(launcher.cached_kernel_count(), artifacts.len() * streams.len());
        let statistics = launcher.cache_statistics();
        assert_eq!(statistics.misses(), (artifacts.len() * streams.len()) as u64);
        assert_eq!(statistics.hits() + statistics.misses(), (8 * 4 * artifacts.len()) as u64);
        assert_eq!(statistics.evictions(), 0);
        drop(launcher);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), artifacts.len() * streams.len());
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_capture_rejected_before_cache_mutation() {
        let driver = Arc::new(TestCudaDriver::default());
        let launcher = CudaKernelLauncher::with_driver(driver.clone(), 1);
        let first = test_artifact(Vec::new());
        let second = test_artifact_with_bytes(test_cubin(100, &[4]), Vec::new());
        assert_eq!(unsafe { launcher.launch(&first, &test_launch()) }, Ok(()));
        driver.stream_is_capturing.store(true, Ordering::SeqCst);
        assert!(matches!(
            unsafe { launcher.launch(&first, &test_launch()) },
            Err(Error::Unavailable { message, .. })
                if message == "cuda kernel launches on capturing streams are unsupported",
        ));
        assert!(matches!(
            unsafe { launcher.launch(&second, &test_launch()) },
            Err(Error::Unavailable { message, .. })
                if message == "cuda kernel launches on capturing streams are unsupported",
        ));
        assert_eq!(launcher.cache_statistics(), CudaKernelCacheStatistics { hits: 0, misses: 1, evictions: 0 });
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 1);
        assert_eq!(driver.launch_count.load(Ordering::SeqCst), 1);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 0);
        driver.stream_is_capturing.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_reuses_module_with_different_dimensions() {
        let driver = Arc::new(TestCudaDriver::default());
        let launcher = CudaKernelLauncher::with_driver(driver.clone(), 2);
        let artifact = test_artifact(Vec::new());
        assert_eq!(unsafe { launcher.launch(&artifact, &test_launch()) }, Ok(()));
        let dimensions = CudaKernelLaunchDimensions::new([7, 8, 9], [10, 2, 3], 65_536).unwrap();
        let variant = artifact.with_launch_dimensions(dimensions);
        assert_eq!(unsafe { launcher.launch(&variant, &test_launch_on_stream(17)) }, Ok(()));
        assert_eq!(*driver.recorded_launch.lock().unwrap(), Some((dimensions, 17)));
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 1);
        assert_eq!(launcher.cache_statistics().hits(), 1);
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_failure() {
        let driver = Arc::new(TestCudaDriver::default());
        driver.fail_launch.store(true, Ordering::SeqCst);
        let launcher = CudaKernelLauncher::with_driver(driver.clone(), 2);
        let artifact = test_artifact(Vec::new());
        assert!(matches!(
            unsafe { launcher.launch(&artifact, &test_launch()) },
            Err(Error::Internal { message, .. }) if message == "injected cuda kernel launch failure",
        ));
        assert_eq!(launcher.cached_kernel_count(), 1);
        assert_eq!(driver.launch_count.load(Ordering::SeqCst), 0);
        driver.fail_launch.store(false, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&artifact, &test_launch()) }, Ok(()));
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 1);
        assert_eq!(driver.launch_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_pending_cleanup_partition_isolation() {
        let driver = Arc::new(TestCudaDriver::default());
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 2);
        let artifact = test_artifact(Vec::new());
        driver.fail_after_module_load.store(true, Ordering::SeqCst);
        driver.fail_unload_context.store(1, Ordering::SeqCst);
        assert!(matches!(
            unsafe { launcher.launch(&artifact, &test_launch()) },
            Err(Error::Internal { message, .. }) if message == "injected cuda module unload failure",
        ));

        // A retained partial load in context 1 cannot interfere with admitting a healthy context 2.
        driver.fail_after_module_load.store(false, Ordering::SeqCst);
        driver.context_id.store(2, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&artifact, &test_launch()) }, Ok(()));
        assert_eq!(*driver.unload_contexts.lock().unwrap(), vec![1]);
        assert_eq!(launcher.cached_kernel_count(), 2);
        driver.fail_unload_context.store(0, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.clear() }, Ok(()));
        assert_eq!(launcher.cached_kernel_count(), 0);
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_all_entries_borrowed() {
        let driver = Arc::new(TestCudaDriver::default());
        let launcher = Arc::new(CudaKernelLauncher::with_driver(driver.clone(), 1));
        let first = test_artifact(Vec::new());
        let second = test_artifact_with_bytes(test_cubin(100, &[4]), Vec::new());
        let (thread, resume) = start_blocked_launch(&launcher, &driver, &first);
        let result = unsafe { launcher.launch(&second, &test_launch()) };
        resume.send(()).unwrap();
        assert_eq!(thread.join().unwrap(), Ok(()));
        assert!(matches!(
            result,
            Err(Error::Unavailable { message, .. }) if message ==
                "cuda kernel cache partition is full and all cached modules in that context/device are in use",
        ));
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 0);
        assert_eq!(unsafe { launcher.launch(&second, &test_launch()) }, Ok(()));
        assert_eq!(*driver.unloaded_modules.lock().unwrap(), vec![1]);
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_evicts_idle_entry_while_another_is_borrowed() {
        let driver = Arc::new(TestCudaDriver::default());
        let launcher = Arc::new(CudaKernelLauncher::with_driver(driver.clone(), 2));
        let first = test_artifact(Vec::new());
        let second = test_artifact_with_bytes(test_cubin(100, &[4]), Vec::new());
        let third = test_artifact_with_bytes(test_cubin(100, &[5]), Vec::new());
        assert_eq!(unsafe { launcher.launch(&first, &test_launch()) }, Ok(()));
        assert_eq!(unsafe { launcher.launch(&second, &test_launch()) }, Ok(()));
        let (thread, resume) = start_blocked_launch(&launcher, &driver, &first);
        // Touch the idle module last: the active module is older but cannot be selected for eviction.
        let second_result = unsafe { launcher.launch(&second, &test_launch()) };
        let third_result = unsafe { launcher.launch(&third, &test_launch()) };
        resume.send(()).unwrap();
        assert_eq!(thread.join().unwrap(), Ok(()));
        assert_eq!(second_result, Ok(()));
        assert_eq!(third_result, Ok(()));
        assert_eq!(*driver.unloaded_modules.lock().unwrap(), vec![2]);
        assert_eq!(launcher.cache_statistics().evictions(), 1);
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_least_recently_used_eviction() {
        let driver = Arc::new(TestCudaDriver::default());
        let launcher = CudaKernelLauncher::with_driver(driver.clone(), 2);
        let first = test_artifact(Vec::new());
        let second = test_artifact_with_bytes(test_cubin(100, &[4]), Vec::new());
        let third = test_artifact_with_bytes(test_cubin(100, &[5]), Vec::new());
        assert_eq!(unsafe { launcher.launch(&first, &test_launch()) }, Ok(()));
        assert_eq!(unsafe { launcher.launch(&second, &test_launch()) }, Ok(()));
        assert_eq!(unsafe { launcher.launch(&first, &test_launch()) }, Ok(()));
        assert_eq!(unsafe { launcher.launch(&third, &test_launch()) }, Ok(()));
        assert_eq!(*driver.unloaded_modules.lock().unwrap(), vec![2]);
        assert_eq!(unsafe { launcher.launch(&first, &test_launch()) }, Ok(()));
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 3);
        assert_eq!(launcher.cache_statistics().evictions(), 1);
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_gpu_ptx() {
        let Some(context) = TestContext::new() else { return };
        let mut launcher = CudaKernelLauncher::new(CudaVersion::from_encoded(12_000).unwrap()).unwrap();
        let artifact = test_ptx(false);
        assert_eq!(unsafe { launcher.launch(&artifact, &context.launch(41)) }, Ok(()));
        assert_eq!(context.read(), 42);
        assert_eq!(unsafe { launcher.launch(&artifact, &context.launch(99)) }, Ok(()));
        assert_eq!(context.read(), 100);
        assert_eq!(launcher.cache_statistics().misses(), 1);
        assert_eq!(launcher.cache_statistics().hits(), 1);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_gpu_shared_memory() {
        let Some(context) = TestContext::new() else { return };
        let mut launcher = CudaKernelLauncher::new(CudaVersion::from_encoded(12_000).unwrap()).unwrap();
        let artifact = test_ptx(true);
        assert_eq!(unsafe { launcher.launch(&artifact, &context.launch(41)) }, Ok(()));
        assert_eq!(context.read(), 42);
        // The same loaded function supports later resource changes without reloading or lowering its opt-in allowance.
        let larger = artifact.with_launch_dimensions(CudaKernelLaunchDimensions::new([1; 3], [1; 3], 66_560).unwrap());
        assert_eq!(unsafe { launcher.launch(&larger, &context.launch(51)) }, Ok(()));
        assert_eq!(context.read(), 52);
        assert_eq!(launcher.cache_statistics().hits(), 1);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_gpu_cubin() {
        let Some(context) = TestContext::new() else { return };
        let compiler = std::env::var_os("CUDA_NVCC").unwrap_or_else(|| "/usr/local/cuda/bin/nvcc".into());
        match Command::new(&compiler).arg("--version").output() {
            Ok(result) => assert!(result.status.success(), "nvcc version query failed"),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && std::env::var_os("CUDA_NVCC").is_none() => {
                eprintln!("skipping cubin execution test: nvcc unavailable; set CUDA_NVCC to its path");
                return;
            }
            Err(error) => panic!("failed to run nvcc: {error}"),
        }
        // The major architecture's base cubin also tests forward minor compatibility on the Spark's sm_121 GPU.
        let architecture = context.capability / 10 * 10;
        let directory = std::env::temp_dir().join(format!(
            "ryft-cuda-gpu-{}-{}",
            std::process::id(),
            NEXT_COMPILATION.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir(&directory).unwrap();
        let image = directory.join("add_one.cubin");
        let source = directory.join("add_one.cu");
        std::fs::write(
            &source,
            indoc! {r#"
                // Independently compiled kernel for validating cubin loading
                // and forward minor compatibility on real GPUs.
                extern "C" __global__ void add_one(unsigned int* output, unsigned int input) {
                    *output = input + 1;
                }
            "#},
        )
        .unwrap();
        let result = Command::new(compiler)
            .arg("--cubin")
            .arg(format!("-arch=sm_{architecture}"))
            .arg(source)
            .arg("-o")
            .arg(&image)
            .output()
            .expect("failed to execute nvcc");
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
        let bytes = std::fs::read(&image).unwrap();
        std::fs::remove_dir_all(&directory).unwrap();
        let artifact = CudaKernelArtifact::new(
            CudaArtifactFormat::Cubin,
            bytes,
            "add_one",
            format!("sm_{architecture}"),
            CudaKernelLaunchDimensions::new([1; 3], [1; 3], 0).unwrap(),
            test_abi(),
        )
        .unwrap();
        assert_eq!(artifact.cubin_architecture(), Some(architecture));
        let mut launcher = CudaKernelLauncher::new(CudaVersion::from_encoded(12_000).unwrap()).unwrap();
        assert_eq!(unsafe { launcher.launch(&artifact, &context.launch(41)) }, Ok(()));
        assert_eq!(context.read(), 42);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_launch_gpu_capture_rejection() {
        let Some(context) = TestContext::new() else { return };
        let mut launcher =
            CudaKernelLauncher::with_cache_capacity(CudaVersion::from_encoded(12_000).unwrap(), 1).unwrap();
        let artifact = test_ptx(false);
        assert_eq!(unsafe { launcher.launch(&artifact, &context.launch(41)) }, Ok(()));
        assert_eq!(context.read(), 42);
        unsafe {
            let begin =
                load::<unsafe extern "C" fn(*mut c_void, i32) -> i32>(&context.library, c"cuStreamBeginCapture_v2");
            assert_eq!(begin(context.stream, 0), 0);
            // This miss would evict the first module if capture were not checked before cache mutation.
            let result = launcher.launch(&test_ptx(true), &context.launch(41));
            let status =
                load::<unsafe extern "C" fn(*mut c_void, *mut i32) -> i32>(&context.library, c"cuStreamIsCapturing");
            let mut capture_status = 0;
            let status_result = status(context.stream, &mut capture_status);
            let end = load::<unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32>(
                &context.library,
                c"cuStreamEndCapture",
            );
            let mut graph = std::ptr::null_mut();
            let end_result = end(context.stream, &mut graph);
            // End capture before asserting so an assertion failure does not leave fixture cleanup inside capture.
            assert_eq!(end_result, 0);
            let destroy = load::<unsafe extern "C" fn(*mut c_void) -> i32>(&context.library, c"cuGraphDestroy");
            assert_eq!(destroy(graph), 0);
            assert!(matches!(
                result,
                Err(Error::Unavailable { message, .. })
                    if message == "cuda kernel launches on capturing streams are unsupported",
            ));
            assert_eq!(status_result, 0);
            assert_eq!(capture_status, 1);
        }
        assert_eq!(launcher.cache_statistics().evictions(), 0);
        assert_eq!(unsafe { launcher.launch(&artifact, &context.launch(1)) }, Ok(()));
        assert_eq!(context.read(), 2);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_clear() {
        let driver = Arc::new(TestCudaDriver::default());
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 1);
        let artifact = test_artifact(Vec::new());
        let launch = test_launch();
        assert_eq!(unsafe { launcher.launch(&artifact, &launch) }, Ok(()));
        assert_eq!(unsafe { launcher.clear() }, Ok(()));
        assert_eq!(launcher.cached_kernel_count(), 0);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 1);
        assert_eq!(unsafe { launcher.launch(&artifact, &launch) }, Ok(()));
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_cuda_kernel_launcher_clear_retains_failures_and_continues_other_contexts() {
        let driver = Arc::new(TestCudaDriver::default());
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 2);
        let artifact = test_artifact(Vec::new());
        assert_eq!(unsafe { launcher.launch(&artifact, &test_launch()) }, Ok(()));
        driver.context_id.store(2, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&artifact, &test_launch()) }, Ok(()));
        driver.fail_unload_context.store(1, Ordering::SeqCst);
        assert!(matches!(
            unsafe { launcher.clear() },
            Err(Error::Internal { message, .. }) if message == "injected cuda module unload failure",
        ));
        assert_eq!(launcher.cached_kernel_count(), 1);
        let mut attempted_contexts = driver.unload_contexts.lock().unwrap().clone();
        attempted_contexts.sort();
        assert_eq!(attempted_contexts, vec![1, 2]);
        driver.fail_unload_context.store(0, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.clear() }, Ok(()));
        assert_eq!(launcher.cached_kernel_count(), 0);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_cuda_kernel_launcher_clear_context() {
        let driver = Arc::new(TestCudaDriver::default());
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 2);
        let artifact = test_artifact(Vec::new());
        assert_eq!(unsafe { launcher.launch(&artifact, &test_launch()) }, Ok(()));
        driver.context_id.store(2, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&artifact, &test_launch()) }, Ok(()));
        let stream = unsafe { CudaStream::from_raw(16usize as *mut c_void) }.unwrap();
        assert_eq!(unsafe { launcher.clear_context(stream) }, Ok(()));
        assert_eq!(*driver.unload_contexts.lock().unwrap(), vec![2]);
        assert_eq!(launcher.cached_kernel_count(), 1);
        driver.context_id.store(1, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&artifact, &test_launch()) }, Ok(()));
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 2);
        driver.context_id.store(2, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&artifact, &test_launch()) }, Ok(()));
        assert_eq!(driver.load_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_cuda_kernel_launcher_clear_context_pending_cleanup_isolation_and_retry() {
        let driver = Arc::new(TestCudaDriver::default());
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 2);
        let artifact = test_artifact(Vec::new());
        driver.fail_after_module_load.store(true, Ordering::SeqCst);
        driver.fail_unload_context.store(1, Ordering::SeqCst);
        assert!(matches!(
            unsafe { launcher.launch(&artifact, &test_launch()) },
            Err(Error::Internal { message, .. }) if message == "injected cuda module unload failure",
        ));
        driver.fail_after_module_load.store(false, Ordering::SeqCst);
        driver.context_id.store(2, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.launch(&artifact, &test_launch()) }, Ok(()));
        let stream = unsafe { CudaStream::from_raw(16usize as *mut c_void) }.unwrap();
        assert_eq!(unsafe { launcher.clear_context(stream) }, Ok(()));
        assert_eq!(*driver.unload_contexts.lock().unwrap(), vec![1, 2]);
        assert_eq!(launcher.cached_kernel_count(), 1);
        driver.context_id.store(1, Ordering::SeqCst);
        assert!(matches!(
            unsafe { launcher.clear_context(stream) },
            Err(Error::Internal { message, .. }) if message == "injected cuda module unload failure",
        ));
        assert_eq!(launcher.cached_kernel_count(), 1);
        driver.fail_unload_context.store(0, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.clear_context(stream) }, Ok(()));
        assert_eq!(launcher.cached_kernel_count(), 0);
    }

    #[test]
    fn test_cuda_kernel_launcher_clear_context_invalid_stream() {
        let driver = Arc::new(TestCudaDriver::default());
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 2);
        assert_eq!(unsafe { launcher.launch(&test_artifact(Vec::new()), &test_launch()) }, Ok(()));
        driver.context_matches_stream.store(false, Ordering::SeqCst);
        let stream = unsafe { CudaStream::from_raw(16usize as *mut c_void) }.unwrap();
        assert!(matches!(
            unsafe { launcher.clear_context(stream) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the cuda stream does not belong to the current cuda context",
        ));
        assert_eq!(launcher.cached_kernel_count(), 1);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_cuda_kernel_launcher_clear_context_gpu() {
        let Some(context) = TestContext::new() else { return };
        let mut launcher = CudaKernelLauncher::new(CudaVersion::from_encoded(12_000).unwrap()).unwrap();
        let artifact = test_ptx(false);
        assert_eq!(unsafe { launcher.launch(&artifact, &context.launch(41)) }, Ok(()));
        assert_eq!(unsafe { launcher.clear_context(context.stream()) }, Ok(()));
        assert_eq!(context.read(), 42);
        assert_eq!(unsafe { launcher.launch(&artifact, &context.launch(21)) }, Ok(()));
        assert_eq!(context.read(), 22);
        assert_eq!(launcher.cache_statistics().misses(), 2);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_shutdown() {
        let driver = Arc::new(TestCudaDriver::default());
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 1);
        let artifact = test_artifact(Vec::new());
        assert_eq!(unsafe { launcher.launch(&artifact, &test_launch()) }, Ok(()));
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
        assert_eq!(launcher.cached_kernel_count(), 0);
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 1);
        assert!(matches!(
            unsafe { launcher.launch(&artifact, &test_launch()) },
            Err(Error::Unavailable { message, .. }) if message == "cuda kernel launcher has been shut down",
        ));
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_cuda_kernel_launcher_shutdown_retry() {
        let driver = Arc::new(TestCudaDriver::default());
        driver.fail_unload.store(true, Ordering::SeqCst);
        let mut launcher = CudaKernelLauncher::with_driver(driver.clone(), 1);
        let artifact = test_artifact(Vec::new());
        let launch = test_launch();
        assert_eq!(unsafe { launcher.launch(&artifact, &launch) }, Ok(()));
        assert!(matches!(
            unsafe { launcher.shutdown() },
            Err(Error::Internal { message, .. }) if message == "injected cuda module unload failure",
        ));
        assert_eq!(launcher.cached_kernel_count(), 1);
        driver.fail_unload.store(false, Ordering::SeqCst);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
    }

    #[test]
    fn test_cuda_kernel_launcher_drop_cleanup_failure() {
        let _cleanup_errors_guard = CLEANUP_ERROR_TEST_LOCK.lock().unwrap();
        drop(Error::take_cleanup_errors());
        let driver = Arc::new(TestCudaDriver::default());
        driver.fail_unload.store(true, Ordering::SeqCst);
        let launcher = CudaKernelLauncher::with_driver(driver.clone(), 1);
        assert_eq!(unsafe { launcher.launch(&test_artifact(Vec::new()), &test_launch()) }, Ok(()));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(launcher))).is_ok());
        assert_eq!(driver.unload_count.load(Ordering::SeqCst), 1);
        // Other tests may record errors concurrently; require this destructor's precise diagnostic to be present.
        assert!(Error::take_cleanup_errors().iter().any(|error| matches!(
            error,
            Error::Internal { message, .. } if message == "injected cuda module unload failure",
        )));
    }

    #[test]
    fn test_cuda_kernel_cache_key() {
        let artifact = test_artifact(Vec::new());
        let different_artifact = test_artifact_with_bytes(test_cubin(100, &[4, 5, 6]), Vec::new());

        let keys = HashSet::from([
            CudaKernelCacheKey {
                scope: CudaKernelCacheScope { context_id: 1, device: 2 },
                content_address: artifact.content_address(),
            },
            CudaKernelCacheKey {
                scope: CudaKernelCacheScope { context_id: 5, device: 2 },
                content_address: artifact.content_address(),
            },
            CudaKernelCacheKey {
                scope: CudaKernelCacheScope { context_id: 1, device: 6 },
                content_address: artifact.content_address(),
            },
            CudaKernelCacheKey {
                scope: CudaKernelCacheScope { context_id: 1, device: 2 },
                content_address: different_artifact.content_address(),
            },
        ]);
        assert_eq!(keys.len(), 4);
    }
}
