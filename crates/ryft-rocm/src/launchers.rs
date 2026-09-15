//! Bounded, context-partitioned module retention and asynchronous native submission.

use std::ffi::{CString, c_void};
use std::sync::{Arc, Mutex};

use crate::drivers::HipDriver;
use crate::ffi::HipDeviceProperties;
use crate::{Error, RocmKernelArtifact, RocmKernelLaunch};

/// One loaded module retaining its device's primary context independently of borrowed execution streams.
struct LoadedKernel {
    /// Owning device ordinal.
    device: i32,

    /// Owning primary-context identity.
    context: usize,

    /// Complete immutable artifact identity.
    content_hash: [u8; 32],

    /// Loaded native module.
    module: usize,

    /// Whether this entry still owns a primary-context retain.
    owns_context: bool,

    /// Entry point resolved from the module.
    function: usize,

    /// Artifact byte count charged against the cache budget.
    bytes: usize,
}

/// Mutable cache state serialized across callers. Entries are ordered from least to most recently used.
struct LauncherState {
    /// Native modules retained through asynchronous execution.
    kernels: Vec<LoadedKernel>,

    /// Whether explicit shutdown completed or native context restoration failed.
    closed: bool,
}

/// Session-owned HIP launcher with bounded module retention across all primary contexts.
///
/// Default limits retain at most 128 modules and 256 MiB of source images. The caller must keep borrowed streams and
/// buffers alive until its execution fence completes. Cache eviction and shutdown synchronize affected contexts
/// before unloading modules; source bytes provide a deterministic cache budget, not a native allocation estimate.
pub struct RocmKernelLauncher {
    /// Resolved driver retained through every native module.
    driver: Arc<HipDriver>,

    /// All native module ownership and least-recently-used ordering.
    state: Mutex<LauncherState>,

    /// Maximum number of modules retained across all contexts.
    maximum_entries: usize,

    /// Maximum sum of source image bytes retained across all contexts.
    maximum_artifact_bytes: usize,
}

impl RocmKernelLauncher {
    /// Creates the strict HIP 7.13 launcher with bounded default retention.
    pub fn new() -> Result<Self, Error> {
        Self::with_cache_limits(128, 256 * 1024 * 1024)
    }

    /// Creates a launcher with positive global entry and source-image budgets.
    pub fn with_cache_limits(maximum_entries: usize, maximum_artifact_bytes: usize) -> Result<Self, Error> {
        if maximum_entries == 0 || maximum_artifact_bytes == 0 {
            return Err(Error::invalid_argument("rocm cache limits must be positive"));
        }
        Self::from_driver(HipDriver::new()?, maximum_entries, maximum_artifact_bytes)
    }

    /// Creates an initialized launcher from a concrete native function table.
    fn from_driver(
        driver: Arc<HipDriver>,
        maximum_entries: usize,
        maximum_artifact_bytes: usize,
    ) -> Result<Self, Error> {
        driver.initialize()?;
        Ok(Self {
            driver,
            state: Mutex::new(LauncherState { kernels: Vec::new(), closed: false }),
            maximum_entries,
            maximum_artifact_bytes,
        })
    }

    /// Submits a verified pointer-only artifact on the borrowed primary-context stream.
    ///
    /// # Safety
    /// All launch pointers and the stream must belong to the stream device's primary context and remain valid through
    /// asynchronous completion. The caller must establish input readiness, bounds, permissions and alias legality.
    /// No stream in a context subject to eviction or shutdown may be capturing, and no graph may reference its cached
    /// modules. These requirements are explicit because the embedding runtime owns buffers, streams and completion.
    pub unsafe fn launch(&self, artifact: &RocmKernelArtifact, launch: &RocmKernelLaunch<'_>) -> Result<(), Error> {
        if launch.arguments().len() != artifact.parameter_count() {
            return Err(Error::invalid_argument("rocm launch argument count does not match the artifact"));
        }
        if artifact.image().len() > self.maximum_artifact_bytes {
            return Err(Error::invalid_argument("rocm artifact exceeds the launcher cache byte budget"));
        }
        let stream = launch.stream().as_raw();
        let mut capture = -1;
        HipDriver::check(unsafe { (self.driver.stream_capture)(stream, &mut capture) }, "hipStreamIsCapturing")?;
        if capture != 0 {
            return Err(Error::unavailable("rocm graph capture is unsupported"));
        }
        let device = unsafe { (self.driver.stream_device)(stream) };
        if device < 0 {
            return Err(Error::internal("hip returned an invalid stream device"));
        }
        let mut state = self.state.lock().expect("rocm launcher mutex poisoned");
        if state.closed {
            return Err(Error::unavailable("rocm launcher is shut down"));
        }
        let result = self.driver.with_context(device, |context| {
            let mut properties = std::mem::MaybeUninit::<HipDeviceProperties>::zeroed();
            HipDriver::check(
                unsafe { (self.driver.properties)(properties.as_mut_ptr(), device) },
                "hipGetDevicePropertiesR0600",
            )?;
            let properties = unsafe { properties.assume_init() };
            let architecture_bytes: Vec<u8> = properties.gcnArchName.iter().map(|byte| *byte as u8).collect();
            let end = architecture_bytes
                .iter()
                .position(|byte| *byte == 0)
                .ok_or_else(|| Error::internal("hip architecture name is not terminated"))?;
            let architecture = std::str::from_utf8(&architecture_bytes[..end])
                .map_err(|_| Error::internal("hip architecture name is not UTF-8"))?;
            artifact.validate_device(architecture)?;
            let dimensions = artifact.launch_dimensions();
            let block = dimensions.block();
            let grid = dimensions.grid();
            if properties.maxThreadsPerBlock <= 0
                || properties.warpSize != 64
                || block.iter().map(|value| u64::from(*value)).product::<u64>() > properties.maxThreadsPerBlock as u64
                || block
                    .iter()
                    .zip(properties.maxThreadsDim)
                    .any(|(requested, limit)| limit <= 0 || u64::from(*requested) > limit as u64)
                || grid
                    .iter()
                    .zip(properties.maxGridSize)
                    .any(|(requested, limit)| limit <= 0 || u64::from(*requested) > limit as u64)
            {
                return Err(Error::invalid_argument("rocm launch dimensions exceed device limits"));
            }
            if dimensions.dynamic_shared_memory_bytes() as usize > properties.sharedMemPerBlock {
                return Err(Error::invalid_argument("rocm shared memory exceeds device limits"));
            }
            let content_hash = artifact.content_hash();
            let index = state.kernels.iter().position(|kernel| {
                kernel.device == device && kernel.context == context as usize && kernel.content_hash == content_hash
            });
            let index = if let Some(index) = index {
                if state.kernels[index].function == 0 {
                    self.unload(&mut state.kernels[index])?;
                    state.kernels.remove(index);
                    None
                } else {
                    Some(index)
                }
            } else {
                None
            };
            let function = if let Some(index) = index {
                let kernel = state.kernels.remove(index);
                let function = kernel.function;
                state.kernels.push(kernel);
                function
            } else {
                while state.kernels.len() >= self.maximum_entries
                    || state.kernels.iter().map(|kernel| kernel.bytes).sum::<usize>()
                        > self.maximum_artifact_bytes - artifact.image().len()
                {
                    self.unload(&mut state.kernels[0])?;
                    state.kernels.remove(0);
                }
                let mut retained = std::ptr::null_mut();
                HipDriver::check(
                    unsafe { (self.driver.context_retain)(&mut retained, device) },
                    "hipDevicePrimaryCtxRetain",
                )?;
                if retained != context {
                    if let Err(cleanup) =
                        HipDriver::check(unsafe { (self.driver.context_release)(device) }, "hipDevicePrimaryCtxRelease")
                    {
                        cleanup.record_cleanup();
                    }
                    return Err(Error::internal("hip primary context identity changed"));
                }
                // Record each acquired resource before the next fallible call. A failed load or symbol lookup leaves
                // an incomplete entry that a retry or shutdown cleans up, without losing context/module ownership.
                state.kernels.push(LoadedKernel {
                    device,
                    context: context as usize,
                    content_hash,
                    module: 0,
                    owns_context: true,
                    function: 0,
                    bytes: artifact.image().len(),
                });
                let kernel = state.kernels.last_mut().unwrap();
                let mut module = std::ptr::null_mut();
                HipDriver::check(
                    unsafe { (self.driver.module_load)(&mut module, artifact.image().as_ptr().cast()) },
                    "hipModuleLoadData",
                )?;
                kernel.module = module as usize;
                if module.is_null() {
                    return Err(Error::internal("hip returned a null module"));
                }
                let entry = CString::new(artifact.entry_name()).unwrap();
                let mut function = std::ptr::null_mut();
                HipDriver::check(
                    unsafe { (self.driver.module_function)(&mut function, module, entry.as_ptr()) },
                    "hipModuleGetFunction",
                )?;
                if function.is_null() {
                    return Err(Error::internal("hip returned a null function"));
                }
                kernel.function = function as usize;
                function as usize
            };
            // `HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` is one in the pinned HIP enum. The launch's dynamic bytes are
            // additional to native static LDS allocation, so both must fit the device's per-block budget.
            let mut maximum_threads = 0;
            HipDriver::check(
                unsafe { (self.driver.function_attribute)(&mut maximum_threads, 0, function as *mut c_void) },
                "hipFuncGetAttribute",
            )?;
            if maximum_threads <= 0
                || block.iter().map(|value| u64::from(*value)).product::<u64>() > maximum_threads as u64
            {
                return Err(Error::invalid_argument("rocm thread block exceeds kernel limits"));
            }
            let mut static_shared = 0;
            HipDriver::check(
                unsafe { (self.driver.function_attribute)(&mut static_shared, 1, function as *mut c_void) },
                "hipFuncGetAttribute",
            )?;
            if static_shared < 0
                || static_shared as u64 + u64::from(dimensions.dynamic_shared_memory_bytes())
                    > properties.sharedMemPerBlock as u64
            {
                return Err(Error::invalid_argument("rocm static and dynamic shared memory exceed device limits"));
            }
            let mut values: Vec<*mut c_void> = launch.arguments().iter().map(|pointer| pointer.as_raw()).collect();
            let mut arguments: Vec<*mut c_void> =
                values.iter_mut().map(|value| (value as *mut *mut c_void).cast()).collect();
            HipDriver::check(
                unsafe {
                    (self.driver.launch)(
                        function as *mut c_void,
                        grid[0],
                        grid[1],
                        grid[2],
                        block[0],
                        block[1],
                        block[2],
                        dimensions.dynamic_shared_memory_bytes(),
                        stream,
                        arguments.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                },
                "hipModuleLaunchKernel",
            )
        });
        if matches!(&result, Err(Error::Driver { operation: "hipCtxPopCurrent", .. }))
            || matches!(&result, Err(Error::Internal { message }) if message == "hip restored an unexpected context")
        {
            state.closed = true;
        }
        result
    }

    /// Synchronizes affected contexts, unloads all modules, and closes the launcher.
    ///
    /// # Safety
    /// Contexts must remain valid, graph capture must be excluded on all their streams, and no graph may retain these
    /// modules. The function is exposed for explicit session teardown after the embedding runtime drains its fences.
    pub unsafe fn shutdown(&self) -> Result<(), Error> {
        let mut state = self.state.lock().expect("rocm launcher mutex poisoned");
        while let Some(kernel) = state.kernels.last_mut() {
            self.unload(kernel)?;
            state.kernels.pop();
        }
        state.closed = true;
        Ok(())
    }

    /// Unloads one module only after its complete primary context has finished using it.
    fn unload(&self, kernel: &mut LoadedKernel) -> Result<(), Error> {
        if kernel.module == 0 && !kernel.owns_context {
            return Ok(());
        }
        self.driver.with_context(kernel.device, |context| {
            if context as usize != kernel.context {
                return Err(Error::internal("hip primary context identity changed"));
            }
            if kernel.module != 0 {
                HipDriver::check(unsafe { (self.driver.synchronize)() }, "hipCtxSynchronize")?;
                HipDriver::check(
                    unsafe { (self.driver.module_unload)(kernel.module as *mut c_void) },
                    "hipModuleUnload",
                )?;
                kernel.module = 0;
                kernel.function = 0;
            }
            if kernel.owns_context {
                HipDriver::check(
                    unsafe { (self.driver.context_release)(kernel.device) },
                    "hipDevicePrimaryCtxRelease",
                )?;
                kernel.owns_context = false;
            }
            Ok(())
        })
    }
}

impl Drop for RocmKernelLauncher {
    fn drop(&mut self) {
        // Owners must exclude capture and retain contexts until launcher destruction, as required by launch/shutdown.
        // A failed drain must not unload a module that native work can still reference. Preserve the driver library
        // and native ownership in that exceptional case, and expose the cleanup error to the application.
        if let Err(error) = unsafe { self.shutdown() } {
            error.record_cleanup();
            std::mem::forget(Arc::clone(&self.driver));
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::drivers::tests::{STATE, driver};
    use crate::{RocmDevicePointer, RocmKernelLaunchDimensions, RocmStream};

    use super::*;

    /// Loads the compiler fixture with its established workgroup and pointer ABI.
    fn artifact() -> RocmKernelArtifact {
        RocmKernelArtifact::new(
            Arc::from(include_bytes!("fixtures/vector-gfx942.hsaco").as_slice()),
            "ryft_kernel",
            "gfx942",
            3,
            RocmKernelLaunchDimensions::new([1; 3], [256, 1, 1], 0).unwrap(),
        )
        .unwrap()
    }

    /// Supplies opaque fake handles that the driver only records, without accessing device memory.
    fn launch() -> RocmKernelLaunch<'static> {
        unsafe {
            RocmKernelLaunch::new(
                RocmStream::from_raw(std::ptr::without_provenance_mut(8)).unwrap(),
                [16, 32, 48]
                    .map(|address| RocmDevicePointer::from_raw(std::ptr::without_provenance_mut(address)).unwrap()),
            )
        }
    }

    #[test]
    fn test_rocm_kernel_launcher_new() {
        if !cfg!(all(target_os = "linux", target_pointer_width = "64")) {
            assert!(matches!(RocmKernelLauncher::new(), Err(Error::Unavailable { message })
                if message == "rocm kernel execution requires 64-bit Linux"));
        }
    }

    #[test]
    fn test_rocm_kernel_launcher_with_cache_limits() {
        for limits in [(0, 1), (1, 0)] {
            assert!(matches!(
                RocmKernelLauncher::with_cache_limits(limits.0, limits.1),
                Err(Error::InvalidArgument { message }) if message == "rocm cache limits must be positive"
            ));
        }
    }

    #[test]
    fn test_rocm_kernel_launcher_launch() {
        let (_guard, driver) = driver();
        let launcher = RocmKernelLauncher::from_driver(driver, 2, 1024 * 1024).unwrap();
        let artifact = artifact();
        unsafe { launcher.launch(&artifact, &launch()) }.unwrap();
        let changed = artifact
            .with_launch_dimensions(RocmKernelLaunchDimensions::new([2, 1, 1], [256, 1, 1], 0).unwrap())
            .unwrap();
        unsafe { launcher.launch(&changed, &launch()) }.unwrap();
        {
            let state = STATE.lock().unwrap();
            assert_eq!(state.argument, 16);
            assert_eq!(state.retained, 1);
            assert!(state.contexts.is_empty());
            assert_eq!(state.calls.iter().filter(|call| **call == "load").count(), 1);
            assert_eq!(state.calls.iter().filter(|call| **call == "launch").count(), 2);
            assert!(!state.calls.contains(&"synchronize"));
        }
        unsafe { launcher.shutdown() }.unwrap();
        assert_eq!(STATE.lock().unwrap().retained, 0);
        assert!(launcher.state.lock().unwrap().kernels.is_empty());
        assert_eq!(
            unsafe { launcher.launch(&artifact, &launch()) },
            Err(Error::unavailable("rocm launcher is shut down"))
        );
    }

    #[test]
    fn test_rocm_kernel_launcher_launch_context_partition() {
        let (_guard, driver) = driver();
        let launcher = RocmKernelLauncher::from_driver(driver, 2, 1024 * 1024).unwrap();
        unsafe { launcher.launch(&artifact(), &launch()) }.unwrap();
        STATE.lock().unwrap().device = 1;
        unsafe { launcher.launch(&artifact(), &launch()) }.unwrap();
        assert_eq!(launcher.state.lock().unwrap().kernels.len(), 2);
        assert_eq!(STATE.lock().unwrap().retained, 2);
        assert_eq!(STATE.lock().unwrap().calls.iter().filter(|call| **call == "load").count(), 2);
        unsafe { launcher.shutdown() }.unwrap();
        assert_eq!(STATE.lock().unwrap().retained, 0);
    }

    #[test]
    fn test_rocm_kernel_launcher_launch_eviction() {
        let (_guard, driver) = driver();
        let launcher = RocmKernelLauncher::from_driver(driver, 1, 1024 * 1024).unwrap();
        unsafe { launcher.launch(&artifact(), &launch()) }.unwrap();
        STATE.lock().unwrap().device = 1;
        unsafe { launcher.launch(&artifact(), &launch()) }.unwrap();
        assert_eq!(launcher.state.lock().unwrap().kernels.len(), 1);
        let state = STATE.lock().unwrap();
        assert_eq!(state.retained, 1);
        let synchronize = state.calls.iter().position(|call| *call == "synchronize").unwrap();
        let unload = state.calls.iter().position(|call| *call == "unload").unwrap();
        let load = state.calls.iter().rposition(|call| *call == "load").unwrap();
        assert!(synchronize < unload && unload < load);
    }

    #[test]
    fn test_rocm_kernel_launcher_launch_rejections() {
        let (_guard, driver) = driver();
        let launcher = RocmKernelLauncher::from_driver(Arc::clone(&driver), 1, 1024 * 1024).unwrap();
        let empty = RocmKernelLaunch::new(launch().stream(), []);
        assert_eq!(
            unsafe { launcher.launch(&artifact(), &empty) },
            Err(Error::invalid_argument("rocm launch argument count does not match the artifact"))
        );
        STATE.lock().unwrap().capture = 1;
        assert_eq!(
            unsafe { launcher.launch(&artifact(), &launch()) },
            Err(Error::unavailable("rocm graph capture is unsupported"))
        );
        STATE.lock().unwrap().capture = 0;
        let small = RocmKernelLauncher::from_driver(driver, 1, 1).unwrap();
        assert_eq!(
            unsafe { small.launch(&artifact(), &launch()) },
            Err(Error::invalid_argument("rocm artifact exceeds the launcher cache byte budget"))
        );
        assert!(!STATE.lock().unwrap().calls.contains(&"load"));
    }

    #[test]
    fn test_rocm_kernel_launcher_launch_incomplete_module() {
        let (_guard, driver) = driver();
        let launcher = RocmKernelLauncher::from_driver(driver, 1, 1024 * 1024).unwrap();
        STATE.lock().unwrap().failure = Some("function");
        assert_eq!(
            unsafe { launcher.launch(&artifact(), &launch()) },
            Err(Error::Driver { operation: "hipModuleGetFunction", code: 700 })
        );
        assert_eq!(STATE.lock().unwrap().retained, 1);
        STATE.lock().unwrap().failure = None;
        unsafe { launcher.launch(&artifact(), &launch()) }.unwrap();
        assert_eq!(STATE.lock().unwrap().calls.iter().filter(|call| **call == "unload").count(), 1);
        assert_eq!(STATE.lock().unwrap().retained, 1);
        unsafe { launcher.shutdown() }.unwrap();
        assert_eq!(STATE.lock().unwrap().retained, 0);
    }

    #[test]
    fn test_rocm_kernel_launcher_launch_native_failure() {
        let (_guard, driver) = driver();
        let launcher = RocmKernelLauncher::from_driver(driver, 1, 1024 * 1024).unwrap();
        STATE.lock().unwrap().failure = Some("launch");
        assert_eq!(
            unsafe { launcher.launch(&artifact(), &launch()) },
            Err(Error::Driver { operation: "hipModuleLaunchKernel", code: 700 })
        );
        assert_eq!(STATE.lock().unwrap().retained, 1);
        STATE.lock().unwrap().failure = None;
        unsafe { launcher.shutdown() }.unwrap();
        assert_eq!(STATE.lock().unwrap().retained, 0);
    }

    #[test]
    fn test_rocm_kernel_launcher_launch_concurrent() {
        let (_guard, driver) = driver();
        let launcher = Arc::new(RocmKernelLauncher::from_driver(driver, 1, 1024 * 1024).unwrap());
        let first = Arc::clone(&launcher);
        let second = Arc::clone(&launcher);
        let first = std::thread::spawn(move || unsafe { first.launch(&artifact(), &launch()) });
        let second = std::thread::spawn(move || unsafe { second.launch(&artifact(), &launch()) });
        assert_eq!(first.join().unwrap(), Ok(()));
        assert_eq!(second.join().unwrap(), Ok(()));
        assert_eq!(STATE.lock().unwrap().calls.iter().filter(|call| **call == "load").count(), 1);
        assert_eq!(STATE.lock().unwrap().calls.iter().filter(|call| **call == "launch").count(), 2);
        unsafe { launcher.shutdown() }.unwrap();
        assert_eq!(STATE.lock().unwrap().retained, 0);
    }

    #[test]
    fn test_rocm_kernel_launcher_shutdown() {
        let (_guard, driver) = driver();
        let launcher = RocmKernelLauncher::from_driver(driver, 1, 1024 * 1024).unwrap();
        unsafe { launcher.launch(&artifact(), &launch()) }.unwrap();
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
        assert_eq!(STATE.lock().unwrap().retained, 0);
        let calls = STATE.lock().unwrap().calls.clone();
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
        assert_eq!(STATE.lock().unwrap().calls, calls);
    }

    #[test]
    fn test_rocm_kernel_launcher_shutdown_failure() {
        let (_guard, driver) = driver();
        for (failure, operation) in [("synchronize", "hipCtxSynchronize"), ("unload", "hipModuleUnload")] {
            let launcher = RocmKernelLauncher::from_driver(Arc::clone(&driver), 1, 1024 * 1024).unwrap();
            unsafe { launcher.launch(&artifact(), &launch()) }.unwrap();
            STATE.lock().unwrap().failure = Some(failure);
            assert_eq!(unsafe { launcher.shutdown() }, Err(Error::Driver { operation, code: 700 }));
            assert_eq!(STATE.lock().unwrap().retained, 1);
            assert_ne!(launcher.state.lock().unwrap().kernels[0].module, 0);
            STATE.lock().unwrap().failure = None;
            unsafe { launcher.shutdown() }.unwrap();
            assert_eq!(STATE.lock().unwrap().retained, 0);
        }
    }

    #[test]
    fn test_rocm_kernel_launcher_shutdown_release_failure() {
        let (_guard, driver) = driver();
        let launcher = RocmKernelLauncher::from_driver(driver, 1, 1024 * 1024).unwrap();
        unsafe { launcher.launch(&artifact(), &launch()) }.unwrap();
        {
            let mut state = STATE.lock().unwrap();
            state.failure = Some("release");
            state.failure_once = true;
        }
        assert_eq!(
            unsafe { launcher.shutdown() },
            Err(Error::Driver { operation: "hipDevicePrimaryCtxRelease", code: 700 })
        );
        assert_eq!(STATE.lock().unwrap().retained, 1);
        assert_eq!(launcher.state.lock().unwrap().kernels[0].module, 0);
        assert!(launcher.state.lock().unwrap().kernels[0].owns_context);
        assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
        assert_eq!(STATE.lock().unwrap().retained, 0);
        assert_eq!(STATE.lock().unwrap().calls.iter().filter(|call| **call == "unload").count(), 1);
    }
}
