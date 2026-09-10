//! CUDA context validation, kernel configuration, and module lifetime operations.

use std::ffi::{CString, c_void};

use crate::artifacts::CudaComputeCapability;
use crate::{CudaArtifactFormat, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaVersion, Error, ffi};

/// Unload failure retaining whether the module still needs cleanup.
pub(super) struct CudaUnloadError {
    /// Driver or validation failure reported to the caller.
    pub(super) error: Error,
    /// Whether the module must remain owned until another cleanup attempt.
    pub(super) module_is_loaded: bool,
}

/// Load failure with any partially loaded module that still needs cleanup.
pub(super) struct CudaKernelLoadError {
    /// Driver or validation failure reported to the caller.
    pub(super) error: Error,
    /// Partially loaded module retained when immediate cleanup fails.
    pub(super) pending_unload: Option<Box<CudaLoadedKernel>>,
}

impl CudaKernelLoadError {
    /// Records a load failure with no remaining module ownership.
    pub(super) fn new(error: Error) -> Self {
        Self { error, pending_unload: None }
    }
}

impl CudaUnloadError {
    /// Records a failure that left the module loaded.
    pub(super) fn retained(error: Error) -> Self {
        Self { error, module_is_loaded: true }
    }

    /// Records restoration failure after the module was successfully unloaded.
    pub(super) fn unloaded(error: Error) -> Self {
        Self { error, module_is_loaded: false }
    }
}

/// Borrowed identity of a live external CUDA context and its device.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct CudaContext {
    /// CUDA context handle, kept alive by the launcher caller.
    pub(super) handle: *mut c_void,
    /// Process-unique identity independent of context handle reuse.
    pub(super) id: u64,
    /// Device ordinal associated with this context.
    pub(super) device: i32,
}

/// Loaded CUDA module and kernel function owned by the launcher cache.
///
/// The external context must outlive explicit unload or launcher destruction. The cache prevents unload while a
/// host launch borrows this value; unloading synchronizes the context to finish queued GPU work.
pub(super) struct CudaLoadedKernel {
    /// Context that owns the module.
    pub(super) context: CudaContext,
    /// Module handle released only by the owning cache.
    pub(super) module: *mut c_void,
    /// Kernel entry point within the module; null only for pending failed-load cleanup.
    pub(super) function: *mut c_void,
    /// Source image byte count represented in the cache.
    pub(super) artifact_bytes: usize,
    /// Maximum dynamic shared memory configured once when loading the function.
    pub(super) max_dynamic_shared_memory_bytes: u32,
}

// CUDA module and function handles may be used from multiple host threads, and every operation is routed through the
// thread-safe Driver API while retaining the owning context identity.
unsafe impl Send for CudaLoadedKernel {}
unsafe impl Sync for CudaLoadedKernel {}

/// CUDA operations used by the cache, separated for deterministic policy testing.
///
/// Handles and argument pointers are validated by unsafe public constructors and launch contracts. A caller must
/// keep every referenced context and stream alive, keep argument storage valid through launch submission, and make
/// the stream's owning context current for load and launch. Context-wide synchronization requires that no stream in
/// the affected context is being captured. Implementations must preserve the caller's current context on success.
pub(super) trait CudaDriverApi: Send + Sync {
    /// Returns the current context identity after validating stream ownership and rejecting stream capture.
    fn context_for_stream(&self, stream: *mut c_void) -> Result<CudaContext, Error>;

    /// Queries the compute capability used for artifact compatibility.
    fn device_compute_capability(&self, device: i32) -> Result<CudaComputeCapability, Error>;

    /// Loads an image and configures its function in the supplied, already-current context.
    fn load_kernel(
        &self,
        context: CudaContext,
        artifact: &CudaKernelArtifact,
    ) -> Result<CudaLoadedKernel, CudaKernelLoadError>;

    /// Enqueues a configured kernel, copying the pointed-to host arguments before returning.
    fn launch_kernel(
        &self,
        kernel: &CudaLoadedKernel,
        dimensions: CudaKernelLaunchDimensions,
        stream: *mut c_void,
        parameters: *mut *mut c_void,
    ) -> Result<(), Error>;

    /// Synchronizes the owning context and unloads the module, restoring the previous current context.
    fn unload_kernel(&self, kernel: &CudaLoadedKernel) -> Result<(), CudaUnloadError>;
}

/// Typed adapter for a retained CUDA Driver API table.
pub(super) struct CudaDriver {
    /// Function table whose library remains loaded for every driver operation.
    api: ffi::Api,
}

/// Restores the caller's current context after temporarily selecting a module's context.
struct CudaCurrentContextGuard<'o> {
    /// Driver owning the context entry points.
    driver: &'o CudaDriver,
    /// Context current before selection, including null when no context was bound.
    previous: *mut c_void,
    /// Whether the selected context differs from the previous one and still needs restoration.
    restore: bool,
}

impl CudaCurrentContextGuard<'_> {
    /// Restores the previous context once, retaining the retry obligation when restoration fails.
    fn restore(&mut self) -> Result<(), Error> {
        if self.restore {
            self.driver
                .api
                .check(unsafe { (self.driver.api.context_set_current)(self.previous) }, "cuCtxSetCurrent")?;
            self.restore = false;
        }
        Ok(())
    }
}

impl Drop for CudaCurrentContextGuard<'_> {
    fn drop(&mut self) {
        if self.restore {
            // Retry after an explicit restoration failure or during unwinding. A second failure must remain visible.
            if let Err(error) = self.restore() {
                error.record_cleanup_error();
            }
        }
    }
}

impl CudaDriver {
    /// Loads the driver at the minimum CUDA version requested by the integration.
    pub(super) fn load(cuda_version: CudaVersion) -> Result<Self, Error> {
        Ok(Self { api: ffi::Api::load(cuda_version.encoded() as i32)? })
    }

    /// Selects a context while retaining the previous context for restoration.
    fn make_context_current(&self, context: *mut c_void) -> Result<CudaCurrentContextGuard<'_>, Error> {
        let mut previous = std::ptr::null_mut();
        self.api.check(unsafe { (self.api.context_get_current)(&mut previous) }, "cuCtxGetCurrent")?;
        let restore = previous != context;
        if restore {
            self.api.check(unsafe { (self.api.context_set_current)(context) }, "cuCtxSetCurrent")?;
        }
        Ok(CudaCurrentContextGuard { driver: self, previous, restore })
    }
}

impl CudaDriverApi for CudaDriver {
    fn context_for_stream(&self, stream: *mut c_void) -> Result<CudaContext, Error> {
        let mut current_context = std::ptr::null_mut();
        self.api.check(unsafe { (self.api.context_get_current)(&mut current_context) }, "cuCtxGetCurrent")?;
        if current_context.is_null() {
            return Err(Error::invalid_argument("the current cuda context is a null pointer"));
        }
        let mut stream_context = std::ptr::null_mut();
        self.api
            .check(unsafe { (self.api.stream_get_context)(stream, &mut stream_context) }, "cuStreamGetCtx")?;
        if stream_context.is_null() {
            return Err(Error::internal("cuda driver returned a null context for the cuda stream"));
        }
        if current_context != stream_context {
            return Err(Error::invalid_argument("the cuda stream does not belong to the current cuda context"));
        }
        let mut capture_status = ffi::STREAM_CAPTURE_STATUS_NONE;
        self.api
            .check(unsafe { (self.api.stream_is_capturing)(stream, &mut capture_status) }, "cuStreamIsCapturing")?;
        if capture_status != ffi::STREAM_CAPTURE_STATUS_NONE {
            return Err(Error::unavailable("cuda kernel launches on capturing streams are unsupported"));
        }
        let mut id = 0;
        self.api.check(unsafe { (self.api.context_get_id)(current_context, &mut id) }, "cuCtxGetId")?;
        let mut device = -1;
        self.api.check(unsafe { (self.api.context_get_device)(&mut device) }, "cuCtxGetDevice")?;
        if device < 0 {
            return Err(Error::internal("cuda driver returned a negative current device ordinal"));
        }
        Ok(CudaContext { handle: current_context, id, device })
    }

    fn device_compute_capability(&self, device: i32) -> Result<CudaComputeCapability, Error> {
        let attribute = |name: i32| -> Result<u32, Error> {
            let mut value = 0;
            self.api
                .check(unsafe { (self.api.device_get_attribute)(&mut value, name, device) }, "cuDeviceGetAttribute")?;
            u32::try_from(value).map_err(|_| {
                Error::internal(format!("cuda driver returned negative compute capability component {value}"))
            })
        };
        Ok(CudaComputeCapability {
            major: attribute(ffi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            minor: attribute(ffi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        })
    }

    fn load_kernel(
        &self,
        context: CudaContext,
        artifact: &CudaKernelArtifact,
    ) -> Result<CudaLoadedKernel, CudaKernelLoadError> {
        let ptx;
        let image = match artifact.format() {
            CudaArtifactFormat::Cubin => artifact.bytes().as_ptr().cast(),
            CudaArtifactFormat::Ptx => {
                ptx = CString::new(artifact.bytes()).unwrap();
                ptx.as_ptr().cast()
            }
        };
        let mut module = std::ptr::null_mut();
        self.api
            .check(
                unsafe {
                    (self.api.module_load_data_ex)(&mut module, image, 0, std::ptr::null_mut(), std::ptr::null_mut())
                },
                "cuModuleLoadDataEx",
            )
            .map_err(CudaKernelLoadError::new)?;
        if module.is_null() {
            return Err(CudaKernelLoadError::new(Error::internal(
                "cuda driver returned a null module after loading an artifact",
            )));
        }

        let symbol = CString::new(artifact.symbol()).unwrap();
        let mut function = std::ptr::null_mut();
        let function_result = self
            .api
            .check(
                unsafe { (self.api.module_get_function)(&mut function, module, symbol.as_ptr()) },
                "cuModuleGetFunction",
            )
            .and_then(|()| {
                if function.is_null() {
                    Err(Error::internal("cuda driver returned a null function after resolving a kernel symbol"))
                } else {
                    Ok(())
                }
            });
        let function_result = function_result.and_then(|()| {
            let mut static_shared_memory_bytes = 0;
            self.api.check(
                unsafe {
                    (self.api.function_get_attribute)(
                        &mut static_shared_memory_bytes,
                        ffi::FUNCTION_ATTRIBUTE_SHARED_SIZE_BYTES,
                        function,
                    )
                },
                "cuFuncGetAttribute",
            )?;
            let mut shared_memory_limit = 0;
            self.api.check(
                unsafe {
                    (self.api.device_get_attribute)(
                        &mut shared_memory_limit,
                        ffi::DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                        context.device,
                    )
                },
                "cuDeviceGetAttribute",
            )?;
            if static_shared_memory_bytes < 0 || shared_memory_limit < static_shared_memory_bytes {
                return Err(Error::internal("cuda driver returned inconsistent kernel shared-memory limits"));
            }
            let max_dynamic_shared_memory_bytes = shared_memory_limit - static_shared_memory_bytes;
            // Configure the full legal budget once. Cache hits may use different launch dimensions, and mutating
            // the allowance per launch could race with another thread launching the same function.
            self.api.check(
                unsafe {
                    (self.api.function_set_attribute)(
                        function,
                        ffi::FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        max_dynamic_shared_memory_bytes,
                    )
                },
                "cuFuncSetAttribute",
            )?;
            Ok(max_dynamic_shared_memory_bytes as u32)
        });
        let max_dynamic_shared_memory_bytes = match function_result {
            Ok(bytes) => bytes,
            Err(error) => {
                let kernel = CudaLoadedKernel {
                    context,
                    module,
                    function: std::ptr::null_mut(),
                    artifact_bytes: artifact.bytes().len(),
                    max_dynamic_shared_memory_bytes: 0,
                };
                return match self.unload_kernel(&kernel) {
                    Ok(()) => Err(CudaKernelLoadError::new(error)),
                    Err(unload_error) => Err(CudaKernelLoadError {
                        error: unload_error.error,
                        pending_unload: unload_error.module_is_loaded.then(|| Box::new(kernel)),
                    }),
                };
            }
        };
        Ok(CudaLoadedKernel {
            context,
            module,
            function,
            artifact_bytes: artifact.bytes().len(),
            max_dynamic_shared_memory_bytes,
        })
    }

    fn launch_kernel(
        &self,
        kernel: &CudaLoadedKernel,
        dimensions: CudaKernelLaunchDimensions,
        stream: *mut c_void,
        parameters: *mut *mut c_void,
    ) -> Result<(), Error> {
        if dimensions.dynamic_shared_memory_bytes() > kernel.max_dynamic_shared_memory_bytes {
            return Err(Error::invalid_argument(format!(
                "cuda kernel requests {} bytes of dynamic shared memory, exceeding its configured limit of {} bytes",
                dimensions.dynamic_shared_memory_bytes(),
                kernel.max_dynamic_shared_memory_bytes,
            )));
        }
        self.api.check(
            unsafe {
                (self.api.launch_kernel)(
                    kernel.function,
                    dimensions.grid()[0],
                    dimensions.grid()[1],
                    dimensions.grid()[2],
                    dimensions.block()[0],
                    dimensions.block()[1],
                    dimensions.block()[2],
                    dimensions.dynamic_shared_memory_bytes(),
                    stream,
                    parameters,
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    fn unload_kernel(&self, kernel: &CudaLoadedKernel) -> Result<(), CudaUnloadError> {
        let mut context_guard = self.make_context_current(kernel.context.handle).map_err(CudaUnloadError::retained)?;
        let unload_result = match self.api.check(unsafe { (self.api.context_synchronize)() }, "cuCtxSynchronize") {
            Ok(()) => self.api.check(unsafe { (self.api.module_unload)(kernel.module) }, "cuModuleUnload"),
            Err(error) => Err(error),
        };
        let restore_result = context_guard.restore();
        match (unload_result, restore_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Ok(()), Err(error)) => Err(CudaUnloadError::unloaded(error)),
            (Err(error), Ok(())) => Err(CudaUnloadError::retained(error)),
            (Err(unload_error), Err(restore_error)) => Err(CudaUnloadError::retained(Error::internal(format!(
                "failed to unload cuda module: {unload_error}; failed to restore previous cuda context: \
                 {restore_error}",
            )))),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::{HashMap, VecDeque};
    use std::ffi::{CStr, c_char};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::CudaKernelAbi;
    use crate::tests::{CLEANUP_ERROR_TEST_LOCK, test_artifact};

    use super::*;

    /// Thread-local CUDA state used to exercise the real adapter through its function table.
    #[derive(Clone)]
    struct TestDriverState {
        /// Entry points called in order.
        calls: Vec<&'static str>,
        /// Injected return codes consumed in order for individual entry points.
        results: HashMap<&'static str, VecDeque<i32>>,
        /// Current context, stream context, module, and function handles.
        handles: [usize; 4],
        /// Context identifier and current device ordinal.
        identity: (u64, i32),
        /// Capture state returned for the supplied stream.
        capture_status: i32,
        /// Compute capability components.
        capability: [i32; 2],
        /// Device opt-in and kernel static shared-memory byte counts.
        shared_memory: [i32; 2],
        /// Context handles passed to the setter.
        selected_contexts: Vec<usize>,
        /// Loaded image length, where `None` denotes NUL-terminated PTX.
        image_length: Option<usize>,
        /// Bytes passed to the module loader.
        image: Vec<u8>,
        /// Module-load JIT options and option-pointer addresses.
        load_options: (u32, usize, usize),
        /// Symbol passed to the function resolver.
        symbol: Vec<u8>,
        /// Device attribute requests and their device ordinals.
        device_attributes: Vec<(i32, i32)>,
        /// Kernel identities and attributes queried during configuration.
        function_attribute_queries: Vec<(i32, usize)>,
        /// Stream handles inspected for ownership and capture.
        inspected_streams: Vec<usize>,
        /// Context handles queried for unique identity.
        inspected_contexts: Vec<usize>,
        /// Module handles queried for kernel symbols.
        inspected_modules: Vec<usize>,
        /// Function attribute writes, including function identity.
        function_attributes: Vec<(usize, i32, i32)>,
        /// Function, grid, block, shared-memory, stream, parameter storage, and extra launch fields.
        launch: Option<(usize, [u32; 3], [u32; 3], u32, usize, usize, usize)>,
        /// Module handles passed to unload.
        unloaded_modules: Vec<usize>,
    }

    impl Default for TestDriverState {
        fn default() -> Self {
            Self {
                calls: Vec::new(),
                results: HashMap::new(),
                handles: [11, 11, 22, 33],
                identity: (123, 2),
                capture_status: 0,
                capability: [9, 0],
                shared_memory: [98_304, 1_024],
                selected_contexts: Vec::new(),
                image_length: None,
                image: Vec::new(),
                load_options: (0, 0, 0),
                symbol: Vec::new(),
                device_attributes: Vec::new(),
                function_attribute_queries: Vec::new(),
                inspected_streams: Vec::new(),
                inspected_contexts: Vec::new(),
                inspected_modules: Vec::new(),
                function_attributes: Vec::new(),
                launch: None,
                unloaded_modules: Vec::new(),
            }
        }
    }

    thread_local! {
        /// Each test and its synchronous C calls use isolated state, allowing the suite to run in parallel.
        static STATE: RefCell<TestDriverState> = RefCell::new(TestDriverState::default());
    }

    /// Records a C entry point and consumes one injected failure, if configured.
    fn record(operation: &'static str) -> i32 {
        STATE.with_borrow_mut(|state| {
            state.calls.push(operation);
            state.results.get_mut(operation).and_then(VecDeque::pop_front).unwrap_or(ffi::SUCCESS)
        })
    }

    /// Supplies an exact failure schedule for a test entry point.
    fn fail(operation: &'static str, results: impl IntoIterator<Item = i32>) {
        STATE.with_borrow_mut(|state| {
            state.results.insert(operation, results.into_iter().collect());
        });
    }

    /// Creates an adapter backed by local C functions while retaining the process library.
    fn test_driver() -> CudaDriver {
        STATE.with_borrow_mut(|state| *state = TestDriverState::default());
        #[cfg(unix)]
        let library = libloading::os::unix::Library::this().into();
        #[cfg(windows)]
        let library = libloading::os::windows::Library::this().unwrap().into();
        CudaDriver {
            api: ffi::Api {
                _library: library,
                device_get_attribute,
                context_get_current,
                context_get_id,
                context_get_device,
                context_set_current,
                context_synchronize,
                stream_get_context,
                stream_is_capturing,
                function_get_attribute,
                function_set_attribute,
                module_load_data_ex,
                module_unload,
                module_get_function,
                launch_kernel,
                get_error_name,
                get_error_string,
            },
        }
    }

    /// Returns a configured borrowed kernel without invoking a module load.
    fn test_kernel() -> CudaLoadedKernel {
        CudaLoadedKernel {
            context: CudaContext { handle: 11usize as *mut c_void, id: 123, device: 2 },
            module: 22usize as *mut c_void,
            function: 33usize as *mut c_void,
            artifact_bytes: 64,
            max_dynamic_shared_memory_bytes: 97_280,
        }
    }

    /// Returns recorded device attributes through the actual C signature.
    unsafe extern "C" fn device_get_attribute(value: *mut i32, attribute: i32, device: i32) -> i32 {
        let result = record("cuDeviceGetAttribute");
        STATE.with_borrow_mut(|state| {
            state.device_attributes.push((attribute, device));
            unsafe {
                *value = match attribute {
                    ffi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => state.capability[0],
                    ffi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => state.capability[1],
                    ffi::DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN => state.shared_memory[0],
                    _ => -1,
                };
            }
        });
        result
    }

    /// Reads the current context from thread-local state.
    unsafe extern "C" fn context_get_current(context: *mut *mut c_void) -> i32 {
        let result = record("cuCtxGetCurrent");
        STATE.with_borrow(|state| unsafe { *context = state.handles[0] as *mut c_void });
        result
    }

    /// Returns the process-unique context identifier.
    unsafe extern "C" fn context_get_id(context: *mut c_void, id: *mut u64) -> i32 {
        let result = record("cuCtxGetId");
        STATE.with_borrow_mut(|state| {
            state.inspected_contexts.push(context as usize);
            unsafe { *id = state.identity.0 };
        });
        result
    }

    /// Returns the device ordinal of the current context.
    unsafe extern "C" fn context_get_device(device: *mut i32) -> i32 {
        let result = record("cuCtxGetDevice");
        STATE.with_borrow(|state| unsafe { *device = state.identity.1 });
        result
    }

    /// Changes the current context only when the injected call succeeds.
    unsafe extern "C" fn context_set_current(context: *mut c_void) -> i32 {
        let result = record("cuCtxSetCurrent");
        STATE.with_borrow_mut(|state| {
            state.selected_contexts.push(context as usize);
            if result == ffi::SUCCESS {
                state.handles[0] = context as usize;
            }
        });
        result
    }

    /// Records context-wide synchronization and its injected result.
    unsafe extern "C" fn context_synchronize() -> i32 {
        record("cuCtxSynchronize")
    }

    /// Returns the owning context of the test stream.
    unsafe extern "C" fn stream_get_context(stream: *mut c_void, context: *mut *mut c_void) -> i32 {
        let result = record("cuStreamGetCtx");
        STATE.with_borrow_mut(|state| {
            state.inspected_streams.push(stream as usize);
            unsafe { *context = state.handles[1] as *mut c_void };
        });
        result
    }

    /// Returns the stream's capture status without changing it.
    unsafe extern "C" fn stream_is_capturing(stream: *mut c_void, status: *mut i32) -> i32 {
        let result = record("cuStreamIsCapturing");
        STATE.with_borrow_mut(|state| {
            state.inspected_streams.push(stream as usize);
            unsafe { *status = state.capture_status };
        });
        result
    }

    /// Returns a kernel's static shared-memory requirement.
    unsafe extern "C" fn function_get_attribute(value: *mut i32, attribute: i32, function: *mut c_void) -> i32 {
        let result = record("cuFuncGetAttribute");
        STATE.with_borrow_mut(|state| {
            state.function_attribute_queries.push((attribute, function as usize));
            unsafe { *value = state.shared_memory[1] };
        });
        result
    }

    /// Records the configured function attribute and allowance.
    unsafe extern "C" fn function_set_attribute(function: *mut c_void, attribute: i32, value: i32) -> i32 {
        let result = record("cuFuncSetAttribute");
        STATE.with_borrow_mut(|state| state.function_attributes.push((function as usize, attribute, value)));
        result
    }

    /// Copies supplied PTX or cubin bytes before returning a configured module handle.
    unsafe extern "C" fn module_load_data_ex(
        module: *mut *mut c_void,
        image: *const c_void,
        option_count: u32,
        options: *mut u32,
        option_values: *mut *mut c_void,
    ) -> i32 {
        let result = record("cuModuleLoadDataEx");
        STATE.with_borrow_mut(|state| unsafe {
            state.image = match state.image_length {
                Some(length) => std::slice::from_raw_parts(image.cast::<u8>(), length).to_vec(),
                None => CStr::from_ptr(image.cast()).to_bytes().to_vec(),
            };
            state.load_options = (option_count, options as usize, option_values as usize);
            *module = state.handles[2] as *mut c_void;
        });
        result
    }

    /// Records module ownership released by the driver adapter.
    unsafe extern "C" fn module_unload(module: *mut c_void) -> i32 {
        let result = record("cuModuleUnload");
        STATE.with_borrow_mut(|state| state.unloaded_modules.push(module as usize));
        result
    }

    /// Copies the NUL-terminated kernel symbol and returns the configured function handle.
    unsafe extern "C" fn module_get_function(
        function: *mut *mut c_void,
        module: *mut c_void,
        name: *const c_char,
    ) -> i32 {
        let result = record("cuModuleGetFunction");
        STATE.with_borrow_mut(|state| unsafe {
            state.inspected_modules.push(module as usize);
            state.symbol = CStr::from_ptr(name).to_bytes().to_vec();
            *function = state.handles[3] as *mut c_void;
        });
        result
    }

    /// Records every launch field without submitting GPU work.
    unsafe extern "C" fn launch_kernel(
        function: *mut c_void,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_memory_bytes: u32,
        stream: *mut c_void,
        parameters: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32 {
        let result = record("cuLaunchKernel");
        STATE.with_borrow_mut(|state| {
            state.launch = Some((
                function as usize,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                shared_memory_bytes,
                stream as usize,
                parameters as usize,
                extra as usize,
            ));
        });
        result
    }

    /// Supplies deterministic CUDA error names.
    unsafe extern "C" fn get_error_name(_error: i32, name: *mut *const c_char) -> i32 {
        unsafe { *name = c"CUDA_ERROR_TEST".as_ptr() };
        ffi::SUCCESS
    }

    /// Supplies deterministic CUDA error descriptions.
    unsafe extern "C" fn get_error_string(_error: i32, message: *mut *const c_char) -> i32 {
        unsafe { *message = c"injected test error".as_ptr() };
        ffi::SUCCESS
    }

    #[test]
    fn test_cuda_current_context_guard_restore() {
        let driver = test_driver();
        let mut guard = driver.make_context_current(77usize as *mut c_void).unwrap();
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 77);
        assert_eq!(guard.restore(), Ok(()));
        assert_eq!(guard.restore(), Ok(()));
        drop(guard);
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 11);
        assert_eq!(STATE.with_borrow(|state| state.selected_contexts.clone()), [77, 11]);

        // Selecting the already-current context requires no mutation or restoration.
        drop(driver.make_context_current(11usize as *mut c_void).unwrap());
        assert_eq!(STATE.with_borrow(|state| state.selected_contexts.clone()), [77, 11]);

        // A thread with no current context returns to that same unbound state.
        STATE.with_borrow_mut(|state| state.handles[0] = 0);
        drop(driver.make_context_current(77usize as *mut c_void).unwrap());
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 0);
    }

    #[test]
    fn test_cuda_current_context_guard_restore_retry() {
        let driver = test_driver();
        let mut guard = driver.make_context_current(77usize as *mut c_void).unwrap();
        fail("cuCtxSetCurrent", [701, 0]);
        assert!(matches!(
            guard.restore(),
            Err(Error::Driver { operation, code: 701, .. })
                if operation == "cuCtxSetCurrent",
        ));
        drop(guard);
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 11);
        assert_eq!(STATE.with_borrow(|state| state.selected_contexts.clone()), [77, 11, 11]);
    }

    #[test]
    fn test_cuda_current_context_guard_drop_failure() {
        let _cleanup_error_guard = CLEANUP_ERROR_TEST_LOCK.lock().unwrap();
        let driver = test_driver();
        let mut guard = driver.make_context_current(77usize as *mut c_void).unwrap();
        fail("cuCtxSetCurrent", [9_701, 9_702]);
        assert!(matches!(
            guard.restore(),
            Err(Error::Driver { operation, code: 9_701, .. }) if operation == "cuCtxSetCurrent",
        ));
        // A second restoration failure is recorded instead of panicking or retrying indefinitely.
        drop(guard);
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 77);
        assert_eq!(STATE.with_borrow(|state| state.selected_contexts.clone()), [77, 11, 11]);
        let mut errors = Error::take_cleanup_errors()
            .into_iter()
            .filter(|error| matches!(error, Error::Driver { code: 9_702, .. }))
            .collect::<Vec<_>>();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors.pop().unwrap(),
            Error::Driver { operation, code: 9_702, name, message, .. }
                if operation == "cuCtxSetCurrent"
                    && name == "CUDA_ERROR_TEST"
                    && message == "injected test error",
        ));
    }

    #[test]
    fn test_cuda_driver_make_context_current_failures() {
        let driver = test_driver();
        fail("cuCtxGetCurrent", [701]);
        assert!(matches!(
            driver.make_context_current(77usize as *mut c_void),
            Err(Error::Driver { operation, .. })
                if operation == "cuCtxGetCurrent",
        ));
        fail("cuCtxSetCurrent", [702]);
        assert!(matches!(
            driver.make_context_current(77usize as *mut c_void),
            Err(Error::Driver { operation, .. })
                if operation == "cuCtxSetCurrent",
        ));
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 11);
    }

    #[test]
    fn test_cuda_driver_context_for_stream() {
        let driver = test_driver();
        assert_eq!(driver.context_for_stream(44usize as *mut c_void), Ok(test_kernel().context));
        assert_eq!(STATE.with_borrow(|state| state.inspected_streams.clone()), [44, 44]);
        assert_eq!(STATE.with_borrow(|state| state.inspected_contexts.clone()), [11]);
        assert_eq!(
            STATE.with_borrow(|state| state.calls.clone()),
            ["cuCtxGetCurrent", "cuStreamGetCtx", "cuStreamIsCapturing", "cuCtxGetId", "cuCtxGetDevice"],
        );
    }

    #[test]
    fn test_cuda_driver_context_for_stream_invalid_outputs() {
        let driver = test_driver();
        STATE.with_borrow_mut(|state| state.handles[0] = 0);
        assert!(matches!(
            driver.context_for_stream(44usize as *mut c_void),
            Err(Error::InvalidArgument { message, .. })
                if message == "the current cuda context is a null pointer",
        ));
        STATE.with_borrow_mut(|state| {
            state.handles[0] = 11;
            state.handles[1] = 0;
        });
        assert!(matches!(
            driver.context_for_stream(44usize as *mut c_void),
            Err(Error::Internal { message, .. })
                if message == "cuda driver returned a null context for the cuda stream",
        ));
        STATE.with_borrow_mut(|state| state.handles[1] = 12);
        assert!(matches!(
            driver.context_for_stream(44usize as *mut c_void),
            Err(Error::InvalidArgument { message, .. })
                if message == "the cuda stream does not belong to the current cuda context",
        ));
        STATE.with_borrow_mut(|state| {
            state.handles[1] = 11;
            state.identity.1 = -1;
        });
        assert!(matches!(
            driver.context_for_stream(44usize as *mut c_void),
            Err(Error::Internal { message, .. })
                if message == "cuda driver returned a negative current device ordinal",
        ));
    }

    #[test]
    fn test_cuda_driver_context_for_stream_capture() {
        let driver = test_driver();
        STATE.with_borrow_mut(|state| state.capture_status = 1);
        assert!(matches!(
            driver.context_for_stream(44usize as *mut c_void),
            Err(Error::Unavailable { message, .. })
                if message == "cuda kernel launches on capturing streams are unsupported",
        ));
        assert_eq!(
            STATE.with_borrow(|state| state.calls.clone()),
            ["cuCtxGetCurrent", "cuStreamGetCtx", "cuStreamIsCapturing"],
        );
        STATE.with_borrow_mut(|state| state.capture_status = 2);
        assert!(matches!(
            driver.context_for_stream(44usize as *mut c_void),
            Err(Error::Unavailable { message, .. })
                if message == "cuda kernel launches on capturing streams are unsupported",
        ));
    }

    #[test]
    fn test_cuda_driver_context_for_stream_failures() {
        let driver = test_driver();
        // Every driver error is propagated without proceeding to later context or cache operations.
        fail("cuCtxGetCurrent", [701]);
        assert!(matches!(
            driver.context_for_stream(44usize as *mut c_void),
            Err(Error::Driver { operation, code: 701, .. })
                if operation == "cuCtxGetCurrent",
        ));
        assert_eq!(STATE.with_borrow(|state| state.calls.last().copied()), Some("cuCtxGetCurrent"));
        fail("cuStreamGetCtx", [701]);
        assert!(matches!(
            driver.context_for_stream(44usize as *mut c_void),
            Err(Error::Driver { operation, code: 701, .. })
                if operation == "cuStreamGetCtx",
        ));
        assert_eq!(STATE.with_borrow(|state| state.calls.last().copied()), Some("cuStreamGetCtx"));
        fail("cuStreamIsCapturing", [701]);
        assert!(matches!(
            driver.context_for_stream(44usize as *mut c_void),
            Err(Error::Driver { operation, code: 701, .. })
                if operation == "cuStreamIsCapturing",
        ));
        assert_eq!(STATE.with_borrow(|state| state.calls.last().copied()), Some("cuStreamIsCapturing"));
        fail("cuCtxGetId", [701]);
        assert!(matches!(
            driver.context_for_stream(44usize as *mut c_void),
            Err(Error::Driver { operation, code: 701, .. })
                if operation == "cuCtxGetId",
        ));
        assert_eq!(STATE.with_borrow(|state| state.calls.last().copied()), Some("cuCtxGetId"));
        fail("cuCtxGetDevice", [701]);
        assert!(matches!(
            driver.context_for_stream(44usize as *mut c_void),
            Err(Error::Driver { operation, code: 701, .. })
                if operation == "cuCtxGetDevice",
        ));
        assert_eq!(STATE.with_borrow(|state| state.calls.last().copied()), Some("cuCtxGetDevice"));
    }

    #[test]
    fn test_cuda_driver_device_compute_capability() {
        let driver = test_driver();
        assert_eq!(driver.device_compute_capability(2), Ok(CudaComputeCapability { major: 9, minor: 0 }));
        assert_eq!(STATE.with_borrow(|state| state.device_attributes.clone()), [(75, 2), (76, 2)]);
        STATE.with_borrow_mut(|state| state.capability[0] = -1);
        assert!(matches!(
            driver.device_compute_capability(2),
            Err(Error::Internal { message, .. })
                if message == "cuda driver returned negative compute capability component -1",
        ));
        fail("cuDeviceGetAttribute", [701]);
        assert!(matches!(
            driver.device_compute_capability(2),
            Err(Error::Driver { operation, code: 701, .. })
                if operation == "cuDeviceGetAttribute",
        ));
    }

    #[test]
    fn test_cuda_driver_load_kernel() {
        let driver = test_driver();
        let artifact = test_artifact(Vec::new());
        STATE.with_borrow_mut(|state| state.image_length = Some(artifact.bytes().len()));
        let kernel = driver
            .load_kernel(test_kernel().context, &artifact)
            .unwrap_or_else(|error| panic!("{}", error.error));
        assert_eq!(kernel.context, test_kernel().context);
        assert_eq!(kernel.module, test_kernel().module);
        assert_eq!(kernel.function, test_kernel().function);
        assert_eq!(kernel.artifact_bytes, artifact.bytes().len());
        assert_eq!(kernel.max_dynamic_shared_memory_bytes, 97_280);
        assert_eq!(STATE.with_borrow(|state| state.image.clone()), artifact.bytes());
        assert_eq!(STATE.with_borrow(|state| state.symbol.clone()), artifact.symbol().as_bytes());
        assert_eq!(STATE.with_borrow(|state| state.load_options), (0, 0, 0));
        assert_eq!(STATE.with_borrow(|state| state.function_attributes.clone()), [(33, 8, 97_280)]);
        assert_eq!(STATE.with_borrow(|state| state.device_attributes.clone()), [(97, 2)]);
        assert_eq!(STATE.with_borrow(|state| state.function_attribute_queries.clone()), [(1, 33)]);
        assert_eq!(STATE.with_borrow(|state| state.inspected_modules.clone()), [22]);
        assert_eq!(
            STATE.with_borrow(|state| state.calls.clone()),
            [
                "cuModuleLoadDataEx",
                "cuModuleGetFunction",
                "cuFuncGetAttribute",
                "cuDeviceGetAttribute",
                "cuFuncSetAttribute"
            ],
        );
    }

    #[test]
    fn test_cuda_driver_load_kernel_ptx() {
        let driver = test_driver();
        let artifact = CudaKernelArtifact::new(
            CudaArtifactFormat::Ptx,
            indoc! {b"
                .version 8.0
                .target sm_90
                .address_size 64
                .visible .entry kernel() { ret; }
            "}
            .to_vec(),
            "kernel",
            "compute_90",
            CudaKernelLaunchDimensions::new([1; 3], [1; 3], 0).unwrap(),
            CudaKernelAbi::new("test", 1, Vec::new()).unwrap(),
        )
        .unwrap();
        let kernel = driver
            .load_kernel(test_kernel().context, &artifact)
            .unwrap_or_else(|error| panic!("{}", error.error));
        assert_eq!(kernel.artifact_bytes, artifact.bytes().len());
        assert_eq!(STATE.with_borrow(|state| state.image.clone()), artifact.bytes());
        assert_eq!(STATE.with_borrow(|state| state.symbol.clone()), b"kernel");
    }

    #[test]
    fn test_cuda_driver_load_kernel_null_handles() {
        let driver = test_driver();
        let artifact = test_artifact(Vec::new());
        STATE.with_borrow_mut(|state| {
            state.image_length = Some(artifact.bytes().len());
            state.handles[2] = 0;
        });
        let error = driver.load_kernel(test_kernel().context, &artifact).err().unwrap();
        assert!(matches!(
            error.error,
            Error::Internal { message, .. }
                if message == "cuda driver returned a null module after loading an artifact",
        ));
        assert!(error.pending_unload.is_none());
        assert_eq!(STATE.with_borrow(|state| state.calls.clone()), ["cuModuleLoadDataEx"]);

        STATE.with_borrow_mut(|state| {
            state.handles[2] = 22;
            state.handles[3] = 0;
        });
        let error = driver.load_kernel(test_kernel().context, &artifact).err().unwrap();
        assert!(matches!(
            error.error,
            Error::Internal { message, .. }
                if message == "cuda driver returned a null function after resolving a kernel symbol",
        ));
        assert!(error.pending_unload.is_none());
        assert_eq!(STATE.with_borrow(|state| state.unloaded_modules.clone()), [22]);
    }

    #[test]
    fn test_cuda_driver_load_kernel_failures() {
        let driver = test_driver();
        let artifact = test_artifact(Vec::new());
        STATE.with_borrow_mut(|state| state.image_length = Some(artifact.bytes().len()));
        fail("cuModuleLoadDataEx", [701]);
        let error = driver.load_kernel(test_kernel().context, &artifact).err().unwrap();
        assert!(matches!(error.error, Error::Driver { operation, code: 701, .. } if operation == "cuModuleLoadDataEx"));
        assert!(error.pending_unload.is_none());
        assert_eq!(STATE.with_borrow(|state| state.unloaded_modules.clone()), Vec::<usize>::new());

        // Once a module exists, every subsequent configuration failure must release it.
        STATE.with_borrow_mut(|state| state.unloaded_modules.clear());
        fail("cuModuleGetFunction", [701]);
        let error = driver.load_kernel(test_kernel().context, &artifact).err().unwrap();
        assert!(matches!(
            error.error,
            Error::Driver { operation, code: 701, .. }
                if operation == "cuModuleGetFunction",
        ));
        assert!(error.pending_unload.is_none());
        assert_eq!(STATE.with_borrow(|state| state.unloaded_modules.clone()), [22]);
        STATE.with_borrow_mut(|state| state.unloaded_modules.clear());
        fail("cuFuncGetAttribute", [701]);
        let error = driver.load_kernel(test_kernel().context, &artifact).err().unwrap();
        assert!(matches!(error.error, Error::Driver { operation, code: 701, .. } if operation == "cuFuncGetAttribute"));
        assert!(error.pending_unload.is_none());
        assert_eq!(STATE.with_borrow(|state| state.unloaded_modules.clone()), [22]);
        STATE.with_borrow_mut(|state| state.unloaded_modules.clear());
        fail("cuDeviceGetAttribute", [701]);
        let error = driver.load_kernel(test_kernel().context, &artifact).err().unwrap();
        assert!(matches!(
            error.error,
            Error::Driver { operation, code: 701, .. }
                if operation == "cuDeviceGetAttribute",
        ));
        assert!(error.pending_unload.is_none());
        assert_eq!(STATE.with_borrow(|state| state.unloaded_modules.clone()), [22]);
        STATE.with_borrow_mut(|state| state.unloaded_modules.clear());
        fail("cuFuncSetAttribute", [701]);
        let error = driver.load_kernel(test_kernel().context, &artifact).err().unwrap();
        assert!(matches!(error.error, Error::Driver { operation, code: 701, .. } if operation == "cuFuncSetAttribute"));
        assert!(error.pending_unload.is_none());
        assert_eq!(STATE.with_borrow(|state| state.unloaded_modules.clone()), [22]);
    }

    #[test]
    fn test_cuda_driver_load_kernel_pending_unload() {
        let driver = test_driver();
        let artifact = test_artifact(Vec::new());
        STATE.with_borrow_mut(|state| state.image_length = Some(artifact.bytes().len()));
        fail("cuModuleGetFunction", [701]);
        fail("cuModuleUnload", [702]);
        let error = driver.load_kernel(test_kernel().context, &artifact).err().unwrap();
        assert!(matches!(error.error, Error::Driver { operation, code: 702, .. } if operation == "cuModuleUnload"));
        let pending = error.pending_unload.unwrap();
        assert_eq!(pending.module, test_kernel().module);
        assert!(pending.function.is_null());
    }

    #[test]
    fn test_cuda_driver_load_kernel_invalid_shared_memory_limits() {
        let driver = test_driver();
        let artifact = test_artifact(Vec::new());
        STATE.with_borrow_mut(|state| {
            state.image_length = Some(artifact.bytes().len());
            state.shared_memory = [512, 1024];
        });
        let error = driver.load_kernel(test_kernel().context, &artifact).err().unwrap();
        assert!(matches!(
            error.error,
            Error::Internal { message, .. }
                if message == "cuda driver returned inconsistent kernel shared-memory limits",
        ));
        assert!(error.pending_unload.is_none());
        assert_eq!(STATE.with_borrow(|state| state.unloaded_modules.clone()), [22]);
        assert_eq!(STATE.with_borrow(|state| state.function_attributes.clone()), Vec::<(usize, i32, i32)>::new());

        STATE.with_borrow_mut(|state| state.shared_memory = [98_304, -1]);
        let error = driver.load_kernel(test_kernel().context, &artifact).err().unwrap();
        assert!(matches!(
            error.error,
            Error::Internal { message, .. }
                if message == "cuda driver returned inconsistent kernel shared-memory limits",
        ));
        assert!(error.pending_unload.is_none());
        assert_eq!(STATE.with_borrow(|state| state.unloaded_modules.clone()), [22, 22]);
    }

    #[test]
    fn test_cuda_driver_launch_kernel() {
        let driver = test_driver();
        let kernel = test_kernel();
        let dimensions = CudaKernelLaunchDimensions::new([2, 3, 4], [8, 4, 2], 65_536).unwrap();
        let mut value = 7u32;
        let mut parameters = [(&mut value as *mut u32).cast::<c_void>()];
        assert_eq!(driver.launch_kernel(&kernel, dimensions, 44usize as *mut c_void, parameters.as_mut_ptr()), Ok(()));
        assert_eq!(
            STATE.with_borrow(|state| state.launch),
            Some((33, [2, 3, 4], [8, 4, 2], 65_536, 44, parameters.as_ptr() as usize, 0)),
        );
        assert_eq!(STATE.with_borrow(|state| state.calls.clone()), ["cuLaunchKernel"]);

        // A second launch can request less memory without lowering the shared function's configured allowance.
        let dimensions = CudaKernelLaunchDimensions::new([1; 3], [1; 3], 0).unwrap();
        assert_eq!(driver.launch_kernel(&kernel, dimensions, 44usize as *mut c_void, std::ptr::null_mut()), Ok(()));
        assert_eq!(STATE.with_borrow(|state| state.function_attributes.clone()), Vec::<(usize, i32, i32)>::new());
    }

    #[test]
    fn test_cuda_driver_launch_kernel_failures() {
        let driver = test_driver();
        let kernel = test_kernel();
        let dimensions = CudaKernelLaunchDimensions::new([1; 3], [1; 3], 97_281).unwrap();
        assert!(matches!(
            driver.launch_kernel(&kernel, dimensions, 44usize as *mut c_void, std::ptr::null_mut()),
            Err(Error::InvalidArgument { message, .. })
                if message ==
                    "cuda kernel requests 97281 bytes of dynamic shared memory, exceeding its configured limit of \
                     97280 bytes",
        ));
        assert_eq!(STATE.with_borrow(|state| state.calls.clone()), Vec::<&str>::new());
        fail("cuLaunchKernel", [701]);
        let dimensions = CudaKernelLaunchDimensions::new([1; 3], [1; 3], 0).unwrap();
        assert!(matches!(
            driver.launch_kernel(&kernel, dimensions, 44usize as *mut c_void, std::ptr::null_mut()),
            Err(Error::Driver { operation, code: 701, .. })
                if operation == "cuLaunchKernel",
        ));
    }

    #[test]
    fn test_cuda_driver_unload_kernel() {
        let driver = test_driver();
        STATE.with_borrow_mut(|state| state.handles[0] = 77);
        assert_eq!(driver.unload_kernel(&test_kernel()).map_err(|error| error.error), Ok(()));
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 77);
        assert_eq!(
            STATE.with_borrow(|state| state.calls.clone()),
            ["cuCtxGetCurrent", "cuCtxSetCurrent", "cuCtxSynchronize", "cuModuleUnload", "cuCtxSetCurrent"],
        );
        assert_eq!(STATE.with_borrow(|state| state.selected_contexts.clone()), [11, 77]);
        assert_eq!(STATE.with_borrow(|state| state.unloaded_modules.clone()), [22]);
    }

    #[test]
    fn test_cuda_driver_unload_kernel_failures() {
        let driver = test_driver();
        STATE.with_borrow_mut(|state| state.handles[0] = 77);
        fail("cuCtxSetCurrent", [700]);
        let error = driver.unload_kernel(&test_kernel()).err().unwrap();
        assert!(error.module_is_loaded);
        assert!(matches!(error.error, Error::Driver { operation, code: 700, .. } if operation == "cuCtxSetCurrent"));
        assert_eq!(STATE.with_borrow(|state| state.calls.clone()), ["cuCtxGetCurrent", "cuCtxSetCurrent"]);
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 77);

        fail("cuCtxSynchronize", [701]);
        let error = driver.unload_kernel(&test_kernel()).err().unwrap();
        assert!(error.module_is_loaded);
        assert!(matches!(error.error, Error::Driver { operation, code: 701, .. } if operation == "cuCtxSynchronize"));
        assert_eq!(STATE.with_borrow(|state| state.unloaded_modules.clone()), Vec::<usize>::new());
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 77);

        fail("cuModuleUnload", [702]);
        let error = driver.unload_kernel(&test_kernel()).err().unwrap();
        assert!(error.module_is_loaded);
        assert!(matches!(error.error, Error::Driver { operation, code: 702, .. } if operation == "cuModuleUnload"));
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 77);

        // Successful unload followed by failed restoration must not retain the now-invalid module handle.
        fail("cuCtxSetCurrent", [0, 703, 0]);
        let error = driver.unload_kernel(&test_kernel()).err().unwrap();
        assert!(!error.module_is_loaded);
        assert!(matches!(error.error, Error::Driver { operation, code: 703, .. } if operation == "cuCtxSetCurrent"));
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 77);
    }

    #[test]
    fn test_cuda_driver_unload_kernel_combined_failure() {
        let driver = test_driver();
        STATE.with_borrow_mut(|state| state.handles[0] = 77);
        fail("cuModuleUnload", [701]);
        fail("cuCtxSetCurrent", [0, 702, 0]);
        let error = driver.unload_kernel(&test_kernel()).err().unwrap();
        assert!(error.module_is_loaded);
        assert!(matches!(
            error.error,
            Error::Internal { message, .. }
                if message ==
                    "failed to unload cuda module: cuda driver function `cuModuleUnload` failed with `CUDA_ERROR_TEST` \
                     (701): injected test error; failed to restore previous cuda context: cuda driver function \
                     `cuCtxSetCurrent` failed with `CUDA_ERROR_TEST` (702): injected test error",
        ));
        assert_eq!(STATE.with_borrow(|state| state.handles[0]), 77);
    }
}
