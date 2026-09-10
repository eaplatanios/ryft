//! Producer-neutral CUDA cubin/PTX artifact loading and kernel launch support.
//!
//! This crate launches producer-independent CUDA artifacts on streams borrowed from external runtimes. It does not
//! depend on PJRT, XLA, a CUDA toolkit, or compiler-specific schemas for Mosaic, cuTile, and other kernel producers.
//!
//! # Type and Ownership Model
//!
//! - [`CudaKernelArtifact`] represents immutable, shareable producer output. It owns cubin/PTX bytes, a kernel symbol,
//!   a target architecture, default [`CudaKernelLaunchDimensions`], and a versioned [`CudaKernelAbi`]. The ABI is a
//!   flattened sequence of [`CudaKernelParameterType`] values, whose scalar cases use [`CudaScalarType`]. Cubin bytes
//!   are inspected statically: their ELF64 header must describe an NVIDIA CUDA object whose SM number agrees with the
//!   recorded `sm_<N>` target (see [`CudaKernelArtifact::cubin_architecture`]).
//! - [`CudaKernelLaunch`] is one execution frame. It borrows an external [`CudaStream`] and contains ordered
//!   [`CudaKernelArgument`] values. Device-pointer arguments borrow [`CudaDevicePointer`] values, while scalar
//!   arguments own [`CudaScalarValue`] storage bits.
//! - [`CudaKernelLauncher`] is configured with a required [`CudaVersion`], loads a compatible NVIDIA driver API,
//!   validates artifacts and launch frames, checks the artifact target against the compute capability of the current
//!   device before loading a module, and owns a bounded context/device/artifact-keyed module cache. It never owns the
//!   external CUDA contexts, streams, or device allocations.
//! - Internally, `CudaDriverApi` separates cache/lifecycle policy from CUDA calls so tests can use a deterministic
//!   mock. `CudaDriver` adapts the raw, version-aware Foreign Function Interface (FFI) table.
//!
//! CUDA 12 and CUDA 13 use the same canonical system driver (i.e., `libcuda.so.1` or `nvcuda.dll`), not
//! toolkit-versioned driver libraries. Each launcher retains its required CUDA version and rejects an older driver.
//! Operational entry points are resolved through
//! `cuGetProcAddress_v2` at the exact CUDA 12.0 ABI represented by this module's function-pointer definitions.
//!
//! The principal data flow is therefore:
//!
//! `CudaKernelArtifact + CudaKernelLaunch -> CudaKernelLauncher -> Cached CUDA Module/Function -> Borrowed Stream`.
//!
//! # Supported Launches
//!
//! Ordinary cubins support the same major and a greater or equal minor compute capability. Restricted architecture
//! targets are validated by CUDA itself. PTX for ordinary targets can be JIT-compiled for newer devices. Header and
//! parameter metadata checks do not establish the safety of executable code or verify its actual kernel signature.
//! [`CudaKernelArtifact::with_launch_dimensions`] changes launch resources while sharing the image and cached hash.
//!
//! Launches accept explicitly owned non-default streams, non-null device pointers, and the listed scalar types.
//! Null/default stream handles, optional null pointer arguments, by-value aggregates, cooperative launches, and runtime
//! cluster configuration are outside this API. Graph capture is unsupported: a capturing launch stream is rejected
//! before cache mutation. Callers must also exclude capture on other streams in any context being synchronized for
//! eviction or cleanup, including destructor cleanup, and exclude graph references to modules managed by this launcher.
//!
//! # Resource Lifetime and Completion
//!
//! The following integration function assumes the artifact has exactly an output-pointer parameter and a `u32`
//! parameter, and that its kernel writes within the supplied allocation. The framework owns the context, stream,
//! and allocation throughout this function. Its synchronization callback must wait for the submitted work and report
//! asynchronous errors. Calling code must satisfy every requirement of [`CudaKernelLauncher::launch`].
//!
//! ```no_run
//! use ryft_cuda::{
//!     CudaDevicePointer, CudaKernelArgument, CudaKernelArtifact, CudaKernelLaunch,
//!     CudaKernelLauncher, CudaScalarValue, CudaStream, CudaVersion, Error,
//! };
//!
//! unsafe fn run<'o>(
//!     artifact: &CudaKernelArtifact,
//!     stream: CudaStream<'o>,
//!     output: CudaDevicePointer<'o>,
//!     synchronize: impl FnOnce() -> Result<(), Error>,
//! ) -> Result<(), Error> {
//!     let mut launcher = CudaKernelLauncher::new(CudaVersion::from_encoded(12_000)?)?;
//!     let launch = CudaKernelLaunch::new(stream, [
//!         CudaKernelArgument::DevicePointer(output),
//!         CudaKernelArgument::Scalar(CudaScalarValue::U32(41)),
//!     ]);
//!     unsafe { launcher.launch(artifact, &launch) }?;
//!     // Enqueue success does not make device output ready or release the external allocation borrow.
//!     synchronize()?;
//!     unsafe { launcher.shutdown() }?;
//!     Ok(())
//! }
//! ```
//!
//! Explicit cleanup returns errors for retry while owners are still live. Destructor failures are retained for
//! observation through [`Error::take_cleanup_errors`]; context destruction owns resources that remain unreleased.

mod artifacts;
mod drivers;
mod errors;
mod launchers;
mod launches;
mod versions;

pub use artifacts::{
    CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaKernelParameterType,
    CudaScalarType,
};
pub use errors::Error;
pub use launchers::{CudaKernelCacheLimits, CudaKernelCacheStatistics, CudaKernelLauncher};
pub use launches::{CudaDevicePointer, CudaKernelArgument, CudaKernelLaunch, CudaScalarValue, CudaStream};
pub use versions::CudaVersion;

pub(crate) mod ffi {
    //! Raw CUDA Driver API bindings and version-aware dynamic loading.
    //!
    //! Signatures follow the [CUDA 12.0 Driver API](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-driver-api/).
    //! Operational entry points retain that ABI even when the installed driver supports newer CUDA versions.

    use std::ffi::{CStr, CString, c_char, c_void};

    use libloading::Library;

    use crate::Error;

    /// `CUDA_SUCCESS` from the CUDA Driver API result enumeration.
    pub const SUCCESS: i32 = 0;

    /// CUDA API version matching every function-pointer typedef in this module.
    ///
    /// This must remain tied to the typedefs rather than to the version reported by the loaded driver.
    pub const ENTRY_POINT_ABI_VERSION: i32 = 12_000;

    /// Resolves the legacy default-stream entry points; explicit special stream handles retain their semantics.
    const GET_PROC_ADDRESS_DEFAULT: u64 = 0;

    /// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` from the CUDA Driver API `CUdevice_attribute` enumeration.
    pub const DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;

    /// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR` from the CUDA Driver API `CUdevice_attribute` enumeration.
    pub const DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` from `CUdevice_attribute`.
    pub const DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: i32 = 97;

    /// `CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` from `CUfunction_attribute`.
    pub const FUNCTION_ATTRIBUTE_SHARED_SIZE_BYTES: i32 = 1;

    /// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` from `CUfunction_attribute`.
    pub const FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;

    /// `CU_STREAM_CAPTURE_STATUS_NONE` from `CUstreamCaptureStatus`.
    pub const STREAM_CAPTURE_STATUS_NONE: i32 = 0;

    /// Signature of `cuInit` at the selected CUDA ABI version.
    pub type CuInit = unsafe extern "C" fn(flags: u32) -> i32;

    /// Signature of `cuDriverGetVersion` at the selected CUDA ABI version.
    pub type CuDriverGetVersion = unsafe extern "C" fn(driver_version: *mut i32) -> i32;

    /// Signature of `cuGetProcAddress_v2` at the selected CUDA ABI version.
    pub type CuGetProcAddressV2 = unsafe extern "C" fn(
        symbol: *const c_char,
        function: *mut *mut c_void,
        cuda_version: i32,
        flags: u64,
        symbol_status: *mut i32,
    ) -> i32;

    /// Signature of `cuDeviceGetAttribute` at the selected CUDA ABI version.
    pub type CuDeviceGetAttribute = unsafe extern "C" fn(value: *mut i32, attribute: i32, device: i32) -> i32;

    /// Signature of `cuCtxGetCurrent` at the selected CUDA ABI version.
    pub type CuCtxGetCurrent = unsafe extern "C" fn(context: *mut *mut c_void) -> i32;

    /// Signature of `cuCtxGetId` at the selected CUDA ABI version.
    pub type CuCtxGetId = unsafe extern "C" fn(context: *mut c_void, id: *mut u64) -> i32;

    /// Signature of `cuCtxGetDevice` at the selected CUDA ABI version.
    pub type CuCtxGetDevice = unsafe extern "C" fn(device: *mut i32) -> i32;

    /// Signature of `cuCtxSetCurrent` at the selected CUDA ABI version.
    pub type CuCtxSetCurrent = unsafe extern "C" fn(context: *mut c_void) -> i32;

    /// Signature of `cuCtxSynchronize` at the selected CUDA ABI version.
    pub type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    /// Signature of `cuStreamGetCtx` at the selected CUDA ABI version.
    pub type CuStreamGetCtx = unsafe extern "C" fn(stream: *mut c_void, context: *mut *mut c_void) -> i32;

    /// Signature of `cuStreamIsCapturing` at the selected CUDA ABI version.
    pub type CuStreamIsCapturing = unsafe extern "C" fn(stream: *mut c_void, status: *mut i32) -> i32;

    /// Signature of `cuFuncGetAttribute` at the selected CUDA ABI version.
    pub type CuFuncGetAttribute = unsafe extern "C" fn(value: *mut i32, attribute: i32, function: *mut c_void) -> i32;

    /// Signature of `cuFuncSetAttribute` at the selected CUDA ABI version.
    pub type CuFuncSetAttribute = unsafe extern "C" fn(function: *mut c_void, attribute: i32, value: i32) -> i32;

    /// Signature of `cuModuleLoadDataEx` at the selected CUDA ABI version.
    pub type CuModuleLoadDataEx = unsafe extern "C" fn(
        module: *mut *mut c_void,
        image: *const c_void,
        option_count: u32,
        options: *mut u32,
        option_values: *mut *mut c_void,
    ) -> i32;

    /// Signature of `cuModuleUnload` at the selected CUDA ABI version.
    pub type CuModuleUnload = unsafe extern "C" fn(module: *mut c_void) -> i32;

    /// Signature of `cuModuleGetFunction` at the selected CUDA ABI version.
    pub type CuModuleGetFunction =
        unsafe extern "C" fn(function: *mut *mut c_void, module: *mut c_void, name: *const c_char) -> i32;

    /// Signature of `cuLaunchKernel` at the selected CUDA ABI version.
    pub type CuLaunchKernel = unsafe extern "C" fn(
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
    ) -> i32;

    /// Signature of `cuGetErrorName` at the selected CUDA ABI version.
    pub type CuGetErrorName = unsafe extern "C" fn(error: i32, name: *mut *const c_char) -> i32;

    /// Signature of `cuGetErrorString` at the selected CUDA ABI version.
    pub type CuGetErrorString = unsafe extern "C" fn(error: i32, message: *mut *const c_char) -> i32;

    /// Loaded CUDA Driver API table.
    pub struct Api {
        /// Keeps the CUDA driver library loaded for the lifetime of every resolved function pointer.
        pub(super) _library: Library,
        /// Queries a numeric device attribute.
        pub device_get_attribute: CuDeviceGetAttribute,
        /// Reads the context bound to the calling host thread.
        pub context_get_current: CuCtxGetCurrent,
        /// Queries the process-unique identity of a context.
        pub context_get_id: CuCtxGetId,
        /// Queries the current context's device ordinal.
        pub context_get_device: CuCtxGetDevice,
        /// Changes the context bound to the calling host thread.
        pub context_set_current: CuCtxSetCurrent,
        /// Waits for all preceding work in the current context.
        pub context_synchronize: CuCtxSynchronize,
        /// Queries the context owning a stream.
        pub stream_get_context: CuStreamGetCtx,
        /// Inspects stream capture without changing its state.
        pub stream_is_capturing: CuStreamIsCapturing,
        /// Queries a loaded function's resource requirements.
        pub function_get_attribute: CuFuncGetAttribute,
        /// Configures a loaded function's resource allowance.
        pub function_set_attribute: CuFuncSetAttribute,
        /// Loads a CUDA image with optional JIT settings.
        pub module_load_data_ex: CuModuleLoadDataEx,
        /// Releases a loaded module.
        pub module_unload: CuModuleUnload,
        /// Resolves a kernel symbol in a loaded module.
        pub module_get_function: CuModuleGetFunction,
        /// Enqueues a function with explicit dimensions and host argument storage.
        pub launch_kernel: CuLaunchKernel,
        /// Looks up the symbolic CUDA error name.
        pub get_error_name: CuGetErrorName,
        /// Looks up the CUDA diagnostic description.
        pub get_error_string: CuGetErrorString,
    }

    /// Bootstrap entry points resolved directly from the canonical CUDA driver library.
    struct BootstrapApi {
        /// Version-aware driver symbol resolver.
        get_proc_address: CuGetProcAddressV2,
        /// Driver error-name lookup.
        get_error_name: CuGetErrorName,
        /// Driver error-description lookup.
        get_error_string: CuGetErrorString,
    }

    impl Api {
        /// Loads the canonical CUDA driver library and resolves the exact API table required by this module.
        pub fn load(required_driver_version: i32) -> Result<Self, Error> {
            let library = load_library()?;
            let initialize = unsafe { load_bootstrap_symbol::<CuInit>(&library, b"cuInit\0")? };
            let driver_get_version =
                unsafe { load_bootstrap_symbol::<CuDriverGetVersion>(&library, b"cuDriverGetVersion\0")? };
            let bootstrap = BootstrapApi {
                get_proc_address: unsafe { load_bootstrap_symbol(&library, b"cuGetProcAddress_v2\0")? },
                get_error_name: unsafe { load_bootstrap_symbol(&library, b"cuGetErrorName\0")? },
                get_error_string: unsafe { load_bootstrap_symbol(&library, b"cuGetErrorString\0")? },
            };

            bootstrap.check(unsafe { initialize(0) }, "cuInit")?;
            let mut driver_version = 0;
            bootstrap.check(unsafe { driver_get_version(&mut driver_version) }, "cuDriverGetVersion")?;
            validate_driver_version(required_driver_version, driver_version)?;

            Ok(Self {
                device_get_attribute: unsafe { bootstrap.resolve("cuDeviceGetAttribute")? },
                context_get_current: unsafe { bootstrap.resolve("cuCtxGetCurrent")? },
                context_get_id: unsafe { bootstrap.resolve("cuCtxGetId")? },
                context_get_device: unsafe { bootstrap.resolve("cuCtxGetDevice")? },
                context_set_current: unsafe { bootstrap.resolve("cuCtxSetCurrent")? },
                context_synchronize: unsafe { bootstrap.resolve("cuCtxSynchronize")? },
                stream_get_context: unsafe { bootstrap.resolve("cuStreamGetCtx")? },
                stream_is_capturing: unsafe { bootstrap.resolve("cuStreamIsCapturing")? },
                function_get_attribute: unsafe { bootstrap.resolve("cuFuncGetAttribute")? },
                function_set_attribute: unsafe { bootstrap.resolve("cuFuncSetAttribute")? },
                module_load_data_ex: unsafe { bootstrap.resolve("cuModuleLoadDataEx")? },
                module_unload: unsafe { bootstrap.resolve("cuModuleUnload")? },
                module_get_function: unsafe { bootstrap.resolve("cuModuleGetFunction")? },
                launch_kernel: unsafe { bootstrap.resolve("cuLaunchKernel")? },
                get_error_name: bootstrap.get_error_name,
                get_error_string: bootstrap.get_error_string,
                _library: library,
            })
        }

        /// Converts a driver result code using the diagnostic functions retained by this table.
        pub fn check(&self, code: i32, operation: &str) -> Result<(), Error> {
            if code == SUCCESS {
                return Ok(());
            }
            Err(driver_error(code, operation, self.get_error_name, self.get_error_string))
        }
    }

    impl BootstrapApi {
        /// Resolves `name` for the exact CUDA version represented by the requested function-pointer type.
        ///
        /// # Safety
        ///
        /// If `name` resolves, `T` must be its exact function-pointer type at [`ENTRY_POINT_ABI_VERSION`], and the
        /// driver library must remain loaded while the returned pointer is used. CUDA exposes untyped symbols, so this
        /// unchecked correspondence is necessary to construct the typed table.
        unsafe fn resolve<T: Copy>(&self, name: &str) -> Result<T, Error> {
            if size_of::<T>() != size_of::<*mut c_void>() {
                return Err(Error::internal(format!(
                    "cuda entry point `{name}` has an unsupported function-pointer representation",
                )));
            }
            let name =
                CString::new(name).map_err(|_| Error::internal("cuda entry point name must not contain a NUL byte"))?;
            let mut function = std::ptr::null_mut();
            let mut symbol_status = 0;
            self.check(
                unsafe {
                    (self.get_proc_address)(
                        name.as_ptr(),
                        &mut function,
                        ENTRY_POINT_ABI_VERSION,
                        GET_PROC_ADDRESS_DEFAULT,
                        &mut symbol_status,
                    )
                },
                "cuGetProcAddress_v2",
            )?;
            if symbol_status != 0 || function.is_null() {
                return Err(Error::unavailable(format!(
                    "cuda driver could not resolve required entry point `{}` at API version {}: {}",
                    name.to_string_lossy(),
                    format_version(ENTRY_POINT_ABI_VERSION),
                    proc_address_status_description(symbol_status),
                )));
            }
            Ok(unsafe { std::mem::transmute_copy(&function) })
        }

        /// Converts a CUDA result code into the module's structured error representation.
        fn check(&self, code: i32, operation: &str) -> Result<(), Error> {
            if code == SUCCESS {
                return Ok(());
            }
            Err(driver_error(code, operation, self.get_error_name, self.get_error_string))
        }
    }

    /// Describes a `CUdriverProcAddressQueryResult` value without exposing the raw enum.
    fn proc_address_status_description(status: i32) -> String {
        match status {
            0 => "the driver returned a null pointer despite reporting success".to_string(),
            1 => "the symbol was not found".to_string(),
            2 => "the requested API version is insufficient for the symbol".to_string(),
            _ => format!("the driver reported unknown query status {status}"),
        }
    }

    /// Loads one bootstrap entry point directly from `library`.
    ///
    /// # Safety
    ///
    /// `T` must match the exported function signature named by `name`. The caller must retain `library` for every
    /// use of the returned pointer. These bootstrap functions are needed before CUDA's typed version lookup exists.
    unsafe fn load_bootstrap_symbol<T: Copy>(library: &Library, name: &'static [u8]) -> Result<T, Error> {
        unsafe { library.get::<T>(name) }.map(|symbol| *symbol).map_err(|error| {
            let name = String::from_utf8_lossy(name).trim_end_matches('\0').to_string();
            Error::unavailable(format!("failed to resolve CUDA driver bootstrap entry point `{name}`: {error}"))
        })
    }

    /// Constructs a structured CUDA driver error using the driver's diagnostic entry points.
    fn driver_error(
        code: i32,
        operation: &str,
        get_error_name: CuGetErrorName,
        get_error_string: CuGetErrorString,
    ) -> Error {
        let mut name = std::ptr::null();
        let name_status = unsafe { get_error_name(code, &mut name) };
        let name = if name_status == SUCCESS && !name.is_null() {
            unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
        } else {
            "CUDA_ERROR_UNKNOWN".to_string()
        };
        let mut message = std::ptr::null();
        let message_status = unsafe { get_error_string(code, &mut message) };
        let message = if message_status == SUCCESS && !message.is_null() {
            unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned()
        } else {
            "unknown CUDA driver error".to_string()
        };
        Error::driver(operation, code, name, message)
    }

    /// Requires the loaded driver to support the CUDA version requested by the framework integration.
    fn validate_driver_version(required: i32, loaded: i32) -> Result<(), Error> {
        if loaded < required {
            return Err(Error::unavailable(format!(
                "the integration requires CUDA {}, but the loaded CUDA driver supports CUDA {}",
                format_version(required),
                format_version(loaded),
            )));
        }
        Ok(())
    }

    /// Formats CUDA's integer version encoding as `major.minor`.
    pub fn format_version(version: i32) -> String {
        format!("{}.{}", version / 1000, version.rem_euclid(1000) / 10)
    }

    #[cfg(target_os = "linux")]
    /// Returns canonical CUDA driver library names for the current platform.
    pub fn library_candidates() -> &'static [&'static str] {
        &["libcuda.so.1"]
    }

    #[cfg(target_os = "windows")]
    /// Returns canonical CUDA driver library names for the current platform.
    pub fn library_candidates() -> &'static [&'static str] {
        &["nvcuda.dll"]
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    /// Returns canonical CUDA driver library names for the current platform.
    pub fn library_candidates() -> &'static [&'static str] {
        &[]
    }

    /// Loads the canonical CUDA driver library for the current platform.
    fn load_library() -> Result<Library, Error> {
        let candidates = library_candidates();
        if candidates.is_empty() {
            return Err(Error::unavailable(format!(
                "the CUDA driver API is unavailable on `{}`",
                std::env::consts::OS,
            )));
        }
        let mut errors = Vec::new();
        for candidate in candidates {
            match unsafe { Library::new(*candidate) } {
                Ok(library) => return Ok(library),
                Err(error) => errors.push(format!("`{candidate}`: {error}")),
            }
        }
        Err(Error::unavailable(format!("failed to load the canonical CUDA driver library; {}", errors.join("; "),)))
    }

    #[cfg(test)]
    mod tests {
        use std::cell::RefCell;

        use pretty_assertions::assert_eq;

        use super::*;

        /// Resolver outputs and recorded calls isolated to the current test thread.
        #[derive(Default)]
        struct TestBootstrapState {
            /// CUDA return code from symbol resolution.
            result: i32,
            /// Symbol query status returned independently from the CUDA result.
            symbol_status: i32,
            /// Whether a successful resolver lookup returns a null address.
            null_function: bool,
            /// Names, ABI versions, and flags supplied to the resolver.
            queries: Vec<(String, i32, u64)>,
            /// Error diagnostic result code and whether returned strings are null.
            diagnostic_result: i32,
            /// Whether error lookups return null despite reporting success.
            null_diagnostics: bool,
        }

        thread_local! {
            /// Bootstrap functions execute synchronously on the same thread as their tests.
            static STATE: RefCell<TestBootstrapState> = RefCell::new(TestBootstrapState::default());
        }

        /// Constructs the real bootstrap adapter with deterministic C functions.
        fn test_bootstrap() -> BootstrapApi {
            STATE.with_borrow_mut(|state| *state = TestBootstrapState::default());
            BootstrapApi { get_proc_address, get_error_name, get_error_string }
        }

        /// Resolved function with a distinctive return value to verify the pointer is callable.
        unsafe extern "C" fn initialize(flags: u32) -> i32 {
            100 + flags as i32
        }

        /// Records the requested exact symbol spelling, ABI version, and stream semantics.
        unsafe extern "C" fn get_proc_address(
            symbol: *const c_char,
            function: *mut *mut c_void,
            cuda_version: i32,
            flags: u64,
            symbol_status: *mut i32,
        ) -> i32 {
            STATE.with_borrow_mut(|state| unsafe {
                state.queries.push((CStr::from_ptr(symbol).to_string_lossy().into_owned(), cuda_version, flags));
                *symbol_status = state.symbol_status;
                *function = if state.null_function { std::ptr::null_mut() } else { initialize as *mut c_void };
                state.result
            })
        }

        /// Returns a stable CUDA error name or an injected failure/null output.
        unsafe extern "C" fn get_error_name(_error: i32, name: *mut *const c_char) -> i32 {
            STATE.with_borrow(|state| unsafe {
                *name = if state.null_diagnostics { std::ptr::null() } else { c"CUDA_ERROR_TEST".as_ptr() };
                state.diagnostic_result
            })
        }

        /// Returns a stable CUDA diagnostic or an injected failure/null output.
        unsafe extern "C" fn get_error_string(_error: i32, message: *mut *const c_char) -> i32 {
            STATE.with_borrow(|state| unsafe {
                *message = if state.null_diagnostics { std::ptr::null() } else { c"injected test failure".as_ptr() };
                state.diagnostic_result
            })
        }

        #[test]
        fn test_bootstrap_api_resolve() {
            let bootstrap = test_bootstrap();
            let function = unsafe { bootstrap.resolve::<CuInit>("cuInit") }.unwrap();
            assert_eq!(unsafe { function(7) }, 107);
            assert_eq!(STATE.with_borrow(|state| state.queries.clone()), [("cuInit".to_string(), 12_000, 0)]);
        }

        #[test]
        fn test_bootstrap_api_resolve_invalid_name() {
            let bootstrap = test_bootstrap();
            assert!(matches!(
                unsafe { bootstrap.resolve::<CuInit>("cuInit\0suffix") },
                Err(Error::Internal { message, .. })
                    if message == "cuda entry point name must not contain a NUL byte",
            ));
            assert_eq!(STATE.with_borrow(|state| state.queries.len()), 0);
        }

        #[test]
        fn test_bootstrap_api_resolve_query_status() {
            let bootstrap = test_bootstrap();
            STATE.with_borrow_mut(|state| state.symbol_status = 1);
            assert!(matches!(
                unsafe { bootstrap.resolve::<CuInit>("cuInit") },
                Err(Error::Unavailable { message, .. })
                    if message ==
                        "cuda driver could not resolve required entry point `cuInit` at API version 12.0: the symbol \
                         was not found",
            ));
            STATE.with_borrow_mut(|state| state.symbol_status = 2);
            assert!(matches!(
                unsafe { bootstrap.resolve::<CuInit>("cuInit") },
                Err(Error::Unavailable { message, .. })
                    if message ==
                        "cuda driver could not resolve required entry point `cuInit` at API version 12.0: the \
                         requested API version is insufficient for the symbol",
            ));
            STATE.with_borrow_mut(|state| state.symbol_status = 42);
            assert!(matches!(
                unsafe { bootstrap.resolve::<CuInit>("cuInit") },
                Err(Error::Unavailable { message, .. })
                    if message ==
                        "cuda driver could not resolve required entry point `cuInit` at API version 12.0: the driver \
                         reported unknown query status 42",
            ));
        }

        #[test]
        fn test_bootstrap_api_resolve_null_function() {
            let bootstrap = test_bootstrap();
            STATE.with_borrow_mut(|state| state.null_function = true);
            assert!(matches!(
                unsafe { bootstrap.resolve::<CuInit>("cuInit") },
                Err(Error::Unavailable { message, .. })
                    if message ==
                        "cuda driver could not resolve required entry point `cuInit` at API version 12.0: the driver \
                         returned a null pointer despite reporting success",
            ));
        }

        #[test]
        fn test_bootstrap_api_resolve_driver_failure() {
            let bootstrap = test_bootstrap();
            STATE.with_borrow_mut(|state| state.result = 701);
            assert!(matches!(
                unsafe { bootstrap.resolve::<CuInit>("cuInit") },
                Err(Error::Driver { operation, code: 701, name, message, .. })
                    if operation == "cuGetProcAddress_v2"
                        && name == "CUDA_ERROR_TEST"
                        && message == "injected test failure",
            ));
        }

        #[test]
        fn test_bootstrap_api_check() {
            let bootstrap = test_bootstrap();
            assert_eq!(bootstrap.check(SUCCESS, "cuInit"), Ok(()));
            assert!(matches!(
                bootstrap.check(701, "cuInit"),
                Err(Error::Driver { operation, code: 701, name, message, .. })
                    if operation == "cuInit"
                        && name == "CUDA_ERROR_TEST"
                        && message == "injected test failure",
            ));
        }

        #[test]
        fn test_driver_error() {
            let bootstrap = test_bootstrap();
            let error = driver_error(701, "cuInit", bootstrap.get_error_name, bootstrap.get_error_string);
            assert!(matches!(
                error,
                Error::Driver { operation, code: 701, name, message, .. }
                    if operation == "cuInit"
                        && name == "CUDA_ERROR_TEST"
                        && message == "injected test failure",
            ));
        }

        #[test]
        fn test_driver_error_diagnostic_failures() {
            let bootstrap = test_bootstrap();
            STATE.with_borrow_mut(|state| state.diagnostic_result = 1);
            let error = driver_error(701, "cuInit", bootstrap.get_error_name, bootstrap.get_error_string);
            assert!(matches!(
                error,
                Error::Driver { operation, code: 701, name, message, .. }
                    if operation == "cuInit"
                        && name == "CUDA_ERROR_UNKNOWN"
                        && message == "unknown CUDA driver error",
            ));
            STATE.with_borrow_mut(|state| {
                state.diagnostic_result = 0;
                state.null_diagnostics = true;
            });
            let error = driver_error(702, "cuInit", bootstrap.get_error_name, bootstrap.get_error_string);
            assert!(matches!(
                error,
                Error::Driver { operation, code: 702, name, message, .. }
                    if operation == "cuInit"
                        && name == "CUDA_ERROR_UNKNOWN"
                        && message == "unknown CUDA driver error",
            ));
        }

        #[test]
        fn test_validate_driver_version() {
            assert_eq!(validate_driver_version(12_090, 12_090), Ok(()));
            assert_eq!(validate_driver_version(12_090, 13_000), Ok(()));
            assert!(matches!(
                validate_driver_version(13_000, 12_090),
                Err(Error::Unavailable { message, .. })
                    if message == "the integration requires CUDA 13.0, but the loaded CUDA driver supports CUDA 12.9",
            ));
        }

        #[test]
        fn test_format_version() {
            assert_eq!(format_version(12_090), "12.9");
            assert_eq!(format_version(13_000), "13.0");
            assert_eq!(format_version(13_030), "13.3");
        }

        #[test]
        fn test_library_candidates() {
            #[cfg(target_os = "linux")]
            assert_eq!(library_candidates(), ["libcuda.so.1"]);
            #[cfg(target_os = "windows")]
            assert_eq!(library_candidates(), ["nvcuda.dll"]);
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            assert!(library_candidates().is_empty());
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! Shared immutable fixtures for owner-module tests.

    use std::sync::Mutex;

    use crate::{
        CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaKernelParameterType,
    };

    /// Serializes tests that drain process-wide cleanup errors while allowing other tests to record failures.
    pub(crate) static CLEANUP_ERROR_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Creates an artifact with the standard test symbol, resources, and supplied cubin image and ABI.
    pub(crate) fn test_artifact_with_bytes(
        bytes: Vec<u8>,
        parameters: Vec<CudaKernelParameterType>,
    ) -> CudaKernelArtifact {
        CudaKernelArtifact::new(
            CudaArtifactFormat::Cubin,
            bytes,
            "test_kernel",
            "sm_100",
            CudaKernelLaunchDimensions::new([1, 2, 3], [4, 5, 6], 128).unwrap(),
            CudaKernelAbi::new("ryft.test", 1, parameters).unwrap(),
        )
        .unwrap()
    }

    /// Creates the standard synthetic cubin artifact with the supplied flattened parameters.
    pub(crate) fn test_artifact(parameters: Vec<CudaKernelParameterType>) -> CudaKernelArtifact {
        test_artifact_with_bytes(test_cubin(100, &[1, 2, 3]), parameters)
    }

    /// Returns a synthetic cubin: a little-endian ELF64 header for NVIDIA CUDA (`e_machine` 190) using the
    /// `ELFABIVERSION_CUDA_V1` encoding, whose `e_flags` low byte records `architecture`, followed by `payload`.
    pub(crate) fn test_cubin(architecture: u32, payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        bytes[7] = 51;
        bytes[8] = 7;
        bytes[16..18].copy_from_slice(&2u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&190u16.to_le_bytes());
        bytes[20..24].copy_from_slice(&1u32.to_le_bytes());
        bytes[48..52].copy_from_slice(&(architecture & 0xff).to_le_bytes());
        bytes[52..54].copy_from_slice(&64u16.to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }
}
