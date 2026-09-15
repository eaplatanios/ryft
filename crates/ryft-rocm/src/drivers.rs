//! Dynamically resolved HIP 7.13 operations and explicit primary-context activation.

use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;

use libloading::Library;

use crate::Error;
use crate::ffi::{HipDeviceProperties, HipModuleLaunchKernel};

/// Resolved entry points whose library remains alive through all module and context ownership.
pub(crate) struct HipDriver {
    /// Native library, absent only for deterministic injected-driver tests.
    pub(crate) _library: Option<Library>,

    /// Queries runtime version before any versioned structure crosses the boundary.
    pub(crate) runtime_version: unsafe extern "C" fn(*mut i32) -> i32,

    /// Initializes HIP without creating a stream or allocation.
    pub(crate) initialize: unsafe extern "C" fn(u32) -> i32,

    /// Returns the physical device ordinal associated with a stream.
    pub(crate) stream_device: unsafe extern "C" fn(*mut c_void) -> i32,

    /// Queries stream graph-capture state.
    pub(crate) stream_capture: unsafe extern "C" fn(*mut c_void, *mut i32) -> i32,

    /// Retains the primary context of one device.
    pub(crate) context_retain: unsafe extern "C" fn(*mut *mut c_void, i32) -> i32,

    /// Releases one primary-context retain.
    pub(crate) context_release: unsafe extern "C" fn(i32) -> i32,

    /// Pushes a borrowed context onto the current thread's stack.
    pub(crate) context_push: unsafe extern "C" fn(*mut c_void) -> i32,

    /// Restores the previous current context.
    pub(crate) context_pop: unsafe extern "C" fn(*mut *mut c_void) -> i32,

    /// Waits for every stream in the current context before module destruction.
    pub(crate) synchronize: unsafe extern "C" fn() -> i32,

    /// Fills the exact `hipDeviceProp_tR0600` layout.
    pub(crate) properties: unsafe extern "C" fn(*mut HipDeviceProperties, i32) -> i32,

    /// Loads a complete HSACO image into the current context.
    pub(crate) module_load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32,

    /// Resolves a kernel symbol in a loaded module.
    pub(crate) module_function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> i32,

    /// Releases a module after its context has finished using it.
    pub(crate) module_unload: unsafe extern "C" fn(*mut c_void) -> i32,

    /// Queries the native kernel's static resource requirements.
    pub(crate) function_attribute: unsafe extern "C" fn(*mut i32, i32, *mut c_void) -> i32,

    /// Submits one explicit pointer-array launch.
    pub(crate) launch: HipModuleLaunchKernel,
}

impl HipDriver {
    /// Loads the selected HIP major/minor ABI and all required entry points.
    pub(crate) fn new() -> Result<Arc<Self>, Error> {
        if !cfg!(all(target_os = "linux", target_pointer_width = "64")) {
            return Err(Error::unavailable("rocm kernel execution requires 64-bit Linux"));
        }
        let library = unsafe { Library::new("libamdhip64.so") }
            .map_err(|error| Error::unavailable(format!("failed to load `libamdhip64.so`: {error}")))?;
        // Every signature comes from the pinned HIP headers. The library outlives all copied function pointers.
        unsafe {
            Ok(Arc::new(Self {
                runtime_version: Self::symbol(&library, b"hipRuntimeGetVersion\0")?,
                initialize: Self::symbol(&library, b"hipInit\0")?,
                stream_device: Self::symbol(&library, b"hipGetStreamDeviceId\0")?,
                stream_capture: Self::symbol(&library, b"hipStreamIsCapturing\0")?,
                context_retain: Self::symbol(&library, b"hipDevicePrimaryCtxRetain\0")?,
                context_release: Self::symbol(&library, b"hipDevicePrimaryCtxRelease\0")?,
                context_push: Self::symbol(&library, b"hipCtxPushCurrent\0")?,
                context_pop: Self::symbol(&library, b"hipCtxPopCurrent\0")?,
                synchronize: Self::symbol(&library, b"hipCtxSynchronize\0")?,
                properties: Self::symbol(&library, b"hipGetDevicePropertiesR0600\0")?,
                module_load: Self::symbol(&library, b"hipModuleLoadData\0")?,
                module_function: Self::symbol(&library, b"hipModuleGetFunction\0")?,
                module_unload: Self::symbol(&library, b"hipModuleUnload\0")?,
                function_attribute: Self::symbol(&library, b"hipFuncGetAttribute\0")?,
                launch: Self::symbol(&library, b"hipModuleLaunchKernel\0")?,
                _library: Some(library),
            }))
        }
    }

    /// Checks the runtime encoding before any device properties or module operations are used.
    pub(crate) fn initialize(&self) -> Result<(), Error> {
        let mut version = 0;
        Self::check(unsafe { (self.runtime_version)(&mut version) }, "hipRuntimeGetVersion")?;
        if version / 10_000_000 != 7 || version % 10_000_000 / 100_000 != 13 {
            return Err(Error::unavailable(format!(
                "hip runtime version `{version}` does not implement the pinned `7.13` ABI"
            )));
        }
        Self::check(unsafe { (self.initialize)(0) }, "hipInit")
    }

    /// Runs an operation inside a temporarily retained primary context and restores the calling thread.
    pub(crate) fn with_context<T>(
        self: &Arc<Self>,
        device: i32,
        operation: impl FnOnce(*mut c_void) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut context = std::ptr::null_mut();
        Self::check(unsafe { (self.context_retain)(&mut context, device) }, "hipDevicePrimaryCtxRetain")?;
        if context.is_null() {
            if let Err(error) = Self::check(unsafe { (self.context_release)(device) }, "hipDevicePrimaryCtxRelease") {
                std::mem::forget(Arc::clone(self));
                return Err(error);
            }
            return Err(Error::internal("hip returned a null primary context"));
        }
        if let Err(error) = Self::check(unsafe { (self.context_push)(context) }, "hipCtxPushCurrent") {
            if let Err(cleanup) = Self::check(unsafe { (self.context_release)(device) }, "hipDevicePrimaryCtxRelease") {
                std::mem::forget(Arc::clone(self));
                cleanup.record_cleanup();
            }
            return Err(error);
        }
        // Internal operation panics still restore the native calling-thread context before resuming unwinding.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(context)));
        let mut popped = std::ptr::null_mut();
        let restoration = Self::check(unsafe { (self.context_pop)(&mut popped) }, "hipCtxPopCurrent").and_then(|()| {
            if popped == context { Ok(()) } else { Err(Error::internal("hip restored an unexpected context")) }
        });
        if let Err(error) = restoration {
            // The context may remain current after a failed pop. Retain its library and primary-context reference;
            // releasing them could invalidate native work or a thread-local context that HIP still owns.
            std::mem::forget(Arc::clone(self));
            match result {
                Ok(Err(operation_error)) => operation_error.record_cleanup(),
                Err(payload) => {
                    error.record_cleanup();
                    std::panic::resume_unwind(payload);
                }
                Ok(Ok(_)) => {}
            }
            return Err(error);
        }
        let release = Self::check(unsafe { (self.context_release)(device) }, "hipDevicePrimaryCtxRelease");
        if release.is_err() {
            // HIP still owns the failed release's reference even though it is no longer current on this thread.
            std::mem::forget(Arc::clone(self));
        }
        let result = match result {
            Ok(result) => result,
            Err(payload) => {
                if let Err(cleanup) = release {
                    cleanup.record_cleanup();
                }
                std::panic::resume_unwind(payload);
            }
        };
        match (result, release) {
            (Err(error), Err(cleanup)) => {
                cleanup.record_cleanup();
                Err(error)
            }
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Ok(value), Ok(())) => Ok(value),
        }
    }

    /// Resolves one exact function-pointer ABI from the retained native library.
    unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, Error> {
        unsafe { library.get::<T>(name) }.map(|symbol| *symbol).map_err(|error| {
            Error::unavailable(format!(
                "missing HIP entry point `{}`: {error}",
                CStr::from_bytes_with_nul(name).unwrap().to_string_lossy()
            ))
        })
    }

    /// Converts a native return code without discarding its owning operation.
    pub(crate) fn check(code: i32, operation: &'static str) -> Result<(), Error> {
        if code == 0 { Ok(()) } else { Err(Error::Driver { operation, code }) }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{Mutex, MutexGuard};

    use super::*;

    /// Serializes tests that intentionally inject native context/module state.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Deterministic native state; no fake function dereferences a device address.
    pub(crate) static STATE: Mutex<FakeState> = Mutex::new(FakeState {
        calls: Vec::new(),
        contexts: Vec::new(),
        retained: 0,
        next_module: 16,
        failure: None,
        failure_once: false,
        version: 71_399_004,
        capture: 0,
        device: 0,
        argument: 0,
    });

    /// Observable native ownership and explicitly injected failures.
    pub(crate) struct FakeState {
        /// Native function call order.
        pub(crate) calls: Vec<&'static str>,

        /// Current thread's context stack.
        pub(crate) contexts: Vec<usize>,

        /// Outstanding primary-context retains.
        pub(crate) retained: i32,

        /// Next distinct module identity.
        pub(crate) next_module: usize,

        /// Native function that must fail until cleared.
        pub(crate) failure: Option<&'static str>,

        /// Whether the selected failure clears itself after its first matching call.
        pub(crate) failure_once: bool,

        /// Encoded runtime ABI version.
        pub(crate) version: i32,

        /// Current stream capture status.
        pub(crate) capture: i32,

        /// Current stream device ordinal.
        pub(crate) device: i32,

        /// First pointer value copied by the last native launch.
        pub(crate) argument: usize,
    }

    /// Records a native operation and applies a deterministic failure before mutation.
    fn call(name: &'static str) -> i32 {
        let mut state = STATE.lock().unwrap();
        state.calls.push(name);
        if state.failure == Some(name) {
            if state.failure_once {
                state.failure = None;
            }
            700
        } else {
            0
        }
    }

    /// Creates one isolated fake driver and resets its native ownership state.
    pub(crate) fn driver() -> (MutexGuard<'static, ()>, Arc<HipDriver>) {
        let guard = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        *STATE.lock().unwrap() = FakeState {
            calls: Vec::new(),
            contexts: Vec::new(),
            retained: 0,
            next_module: 16,
            failure: None,
            failure_once: false,
            version: 71_399_004,
            capture: 0,
            device: 0,
            argument: 0,
        };
        (
            guard,
            Arc::new(HipDriver {
                _library: None,
                runtime_version,
                initialize,
                stream_device,
                stream_capture,
                context_retain,
                context_release,
                context_push,
                context_pop,
                synchronize,
                properties,
                module_load,
                module_function,
                module_unload,
                function_attribute,
                launch,
            }),
        )
    }

    /// Returns the selected encoded runtime version.
    unsafe extern "C" fn runtime_version(output: *mut i32) -> i32 {
        let status = call("version");
        if status == 0 {
            unsafe {
                *output = STATE.lock().unwrap().version;
            }
        }
        status
    }

    /// Records native initialization without creating a context.
    unsafe extern "C" fn initialize(_: u32) -> i32 {
        call("initialize")
    }

    /// Returns the stream's physical device ordinal.
    unsafe extern "C" fn stream_device(_: *mut c_void) -> i32 {
        STATE.lock().unwrap().device
    }

    /// Returns the selected capture status.
    unsafe extern "C" fn stream_capture(_: *mut c_void, output: *mut i32) -> i32 {
        let status = call("capture");
        if status == 0 {
            unsafe {
                *output = STATE.lock().unwrap().capture;
            }
        }
        status
    }

    /// Retains a stable fake primary context.
    unsafe extern "C" fn context_retain(output: *mut *mut c_void, device: i32) -> i32 {
        let status = call("retain");
        if status == 0 {
            STATE.lock().unwrap().retained += 1;
            unsafe {
                *output = std::ptr::without_provenance_mut(100 + device as usize);
            }
        }
        status
    }

    /// Releases exactly one fake primary-context retain.
    unsafe extern "C" fn context_release(_: i32) -> i32 {
        let status = call("release");
        if status == 0 {
            STATE.lock().unwrap().retained -= 1;
        }
        status
    }

    /// Activates a fake context on the test thread.
    unsafe extern "C" fn context_push(context: *mut c_void) -> i32 {
        let status = call("push");
        if status == 0 {
            STATE.lock().unwrap().contexts.push(context as usize);
        }
        status
    }

    /// Restores the previous fake context.
    unsafe extern "C" fn context_pop(output: *mut *mut c_void) -> i32 {
        let status = call("pop");
        if status == 0 {
            let context = STATE.lock().unwrap().contexts.pop().unwrap();
            unsafe {
                *output = std::ptr::without_provenance_mut(context);
            }
        }
        status
    }

    /// Records completion before a module may be released.
    unsafe extern "C" fn synchronize() -> i32 {
        call("synchronize")
    }

    /// Supplies bounded properties for the compiler fixture's AMD architecture.
    unsafe extern "C" fn properties(output: *mut HipDeviceProperties, _: i32) -> i32 {
        let status = call("properties");
        if status == 0 {
            unsafe {
                output.write(std::mem::zeroed());
                (*output).maxThreadsPerBlock = 1024;
                (*output).maxThreadsDim = [1024; 3];
                (*output).maxGridSize = [i32::MAX; 3];
                (*output).sharedMemPerBlock = 64 * 1024;
                (*output).warpSize = 64;
                for (index, byte) in b"gfx942:sramecc+:xnack-".iter().enumerate() {
                    (*output).gcnArchName[index] = *byte as c_char;
                }
            }
        }
        status
    }

    /// Assigns a distinct loaded module identity.
    unsafe extern "C" fn module_load(output: *mut *mut c_void, _: *const c_void) -> i32 {
        let status = call("load");
        if status == 0 {
            let mut state = STATE.lock().unwrap();
            unsafe {
                *output = std::ptr::without_provenance_mut(state.next_module);
            }
            state.next_module += 16;
        }
        status
    }

    /// Resolves a stable fake function within its module.
    unsafe extern "C" fn module_function(output: *mut *mut c_void, module: *mut c_void, _: *const c_char) -> i32 {
        let status = call("function");
        if status == 0 {
            unsafe {
                *output = module;
            }
        }
        status
    }

    /// Releases one loaded module after synchronization.
    unsafe extern "C" fn module_unload(_: *mut c_void) -> i32 {
        call("unload")
    }

    /// Reports no additional static LDS allocation for the fake vector kernel.
    unsafe extern "C" fn function_attribute(output: *mut i32, attribute: i32, _: *mut c_void) -> i32 {
        let status = call("attribute");
        if status == 0 {
            unsafe {
                *output = if attribute == 0 { 1024 } else { 0 };
            }
        }
        status
    }

    /// Copies the first pointer argument from host argument storage without accessing device memory.
    unsafe extern "C" fn launch(
        _: *mut c_void,
        _: u32,
        _: u32,
        _: u32,
        _: u32,
        _: u32,
        _: u32,
        _: u32,
        _: *mut c_void,
        arguments: *mut *mut c_void,
        _: *mut *mut c_void,
    ) -> i32 {
        let status = call("launch");
        if status == 0 {
            STATE.lock().unwrap().argument = unsafe { *(*arguments).cast::<*mut c_void>() } as usize;
        }
        status
    }

    #[test]
    fn test_hip_driver_initialize() {
        let (_guard, driver) = driver();
        assert_eq!(driver.initialize(), Ok(()));
        assert_eq!(STATE.lock().unwrap().calls, ["version", "initialize"]);
        STATE.lock().unwrap().version = 71_125_424;
        assert_eq!(
            driver.initialize(),
            Err(Error::unavailable("hip runtime version `71125424` does not implement the pinned `7.13` ABI"))
        );
    }

    #[test]
    fn test_hip_driver_with_context() {
        let (_guard, driver) = driver();
        assert_eq!(driver.with_context(0, |context| Ok(context as usize)), Ok(100));
        assert_eq!(STATE.lock().unwrap().calls, ["retain", "push", "pop", "release"]);
        assert_eq!(STATE.lock().unwrap().retained, 0);
        let error = Error::invalid_argument("test operation failed");
        assert_eq!(driver.with_context::<()>(0, |_| Err(error.clone())), Err(error));
        assert!(STATE.lock().unwrap().contexts.is_empty());
        assert_eq!(STATE.lock().unwrap().retained, 0);
    }

    #[test]
    fn test_hip_driver_with_context_push_failure() {
        let (_guard, driver) = driver();
        STATE.lock().unwrap().failure = Some("push");
        assert_eq!(
            driver.with_context::<()>(0, |_| panic!("operation must not run")),
            Err(Error::Driver { operation: "hipCtxPushCurrent", code: 700 })
        );
        assert_eq!(STATE.lock().unwrap().calls, ["retain", "push", "release"]);
        assert_eq!(STATE.lock().unwrap().retained, 0);
    }

    #[test]
    fn test_hip_driver_with_context_unwind() {
        let (_guard, driver) = driver();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            driver.with_context::<()>(0, |_| panic!("injected operation panic"))
        }));
        assert!(panic.is_err());
        assert_eq!(STATE.lock().unwrap().calls, ["retain", "push", "pop", "release"]);
        assert_eq!(STATE.lock().unwrap().retained, 0);
        assert!(STATE.lock().unwrap().contexts.is_empty());
    }
}
