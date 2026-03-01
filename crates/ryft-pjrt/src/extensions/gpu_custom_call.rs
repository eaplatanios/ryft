use ryft_xla_sys::bindings::{XlaCustomCallStatus, XlaCustomCallStatusSetFailure, XlaCustomCallStatusSetSuccess};

use crate::extensions::ffi::FfiHandler;
use crate::{Api, Client, Error, Plugin, invoke_pjrt_api_error_fn};

/// The PJRT GPU custom call extension provides capabilities for registering custom call targets with GPU backends.
/// The extension is both optional for PJRT [`Plugin`]s and _experimental_, meaning that incompatible changes may be
/// introduced at any time, including changes that break _Application Binary Interface (ABI)_ compatibility. If present,
/// it enables runtime registration of custom call handlers that can be invoked from compiled PJRT programs.
#[derive(Copy, Clone)]
pub struct GpuCustomCallExtension {
    /// Handle that represents this [`GpuCustomCallExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_Gpu_Custom_Call_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl GpuCustomCallExtension {
    /// Constructs a new [`GpuCustomCallExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle if the type of that PJRT
    /// extension matches the PJRT GPU custom call extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_Gpu_Custom_Call {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_Gpu_Custom_Call`](ffi::PJRT_Gpu_Custom_Call_Extension) that corresponds to this
    /// [`GpuCustomCallExtension`] and which can be passed to functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_Gpu_Custom_Call_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for GpuCustomCallExtension {}
unsafe impl Sync for GpuCustomCallExtension {}

impl Client<'_> {
    /// Attempts to load the [`GpuCustomCallExtension`] from this [`Client`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub fn gpu_custom_call_extension(&self) -> Result<GpuCustomCallExtension, Error> {
        self.api().gpu_custom_call_extension()
    }

    /// Registers a GPU custom call handler under the provided `name`, making it available to any compiled PJRT
    /// [`Executable`](crate::Executable) that contains a `stablehlo.custom_call` operation with a matching
    /// `call_target_name`. Note that [`Client::register_ffi_handler`] and [`Plugin::register_ffi_handler`] are
    /// preferable over this function, and should be used instead, whenever possible.
    ///
    /// XLA [custom calls](https://openxla.org/xla/custom_call) are the primary escape hatch for
    /// running operations that the XLA compiler does not natively support. At compile time, a `stablehlo.custom_call`
    /// (or its HLO equivalent) records a target name and an optional opaque backend configuration. At runtime, XLA
    /// looks up the target name in a handler registry and invokes the corresponding function. This method populates
    /// that registry for GPU backends (e.g., CUDA and ROCm).
    ///
    /// The registered handler runs **on the host CPU** but receives the GPU stream handle and device-memory buffer
    /// pointers, so its typical job is to *enqueue* GPU work (e.g., launch a kernel, call into cuBLAS, cuDNN, hipBLAS,
    /// etc.) onto the provided stream. XLA pre-allocates all output buffers and uses destination-passing style, meaning
    /// that the handler writes results into XLA-owned device memory rather than allocating its own.
    ///
    /// Typical use cases include:
    ///
    ///   - **Custom Kernels:** Hand-optimized GPU kernels for operations that XLA's fusion strategies
    ///     cannot express efficiently (e.g., custom attention variants, specialized reductions).
    ///   - **Triton Kernels:** Registering handlers that load and dispatch Triton-compiled GPU binaries.
    ///   - **Vendor Library Calls:** Wrapping cuBLAS, cuDNN, cuFFT, NCCL, their ROCm equivalents, etc.,
    ///     with configurations that XLA does not emit on its own.
    ///   - **Distributed Communication Primitives:** Implementing MPI or custom collective operations
    ///     that execute on the GPU stream within an XLA program.
    ///
    /// Note that this function is only available on GPU PJRT plugins (e.g., CUDA and ROCm). It returns
    /// [`Error::Unimplemented`] when called on non-GPU backends (e.g., CPU, TPU, etc.).
    ///
    /// # Note Regarding [`GpuCustomCallHandler::Typed`]
    ///
    /// Note that if you are using a [`GpuCustomCallHandler::Typed`] implementation, XLA will invoke the `execute`
    /// handler during the registration process with a special XLA FFI metadata call frame in order to extract API
    /// version information. This call frame has no input/output buffers but it includes an
    /// [`XLA_FFI_Metadata_Extension`](crate::extensions::ffi::handlers::ffi::XLA_FFI_Metadata_Extension) in the
    /// extension chain. You must extract that extension in your handler implementation and set the API version using
    /// [`XLA_FFI_API_VERSION_MAJOR`](crate::extensions::ffi::versions::ffi::XLA_FFI_API_VERSION_MAJOR) and
    /// [`XLA_FFI_API_VERSION_MINOR`](crate::extensions::ffi::versions::ffi::XLA_FFI_API_VERSION_MINOR), and
    /// immediately return a null pointer, for this handler to work as expected.
    pub fn register_gpu_custom_call<N: AsRef<str>, H: Into<GpuCustomCallHandler>>(
        &self,
        name: N,
        handler: H,
    ) -> Result<(), Error> {
        self.api().register_gpu_custom_call(name, handler)
    }
}

impl Plugin {
    /// Attempts to load the [`GpuCustomCallExtension`] from this [`Plugin`] and returns [`Error::Unimplemented`]
    /// if it is not provided by this [`Plugin`].
    pub fn gpu_custom_call_extension(&self) -> Result<GpuCustomCallExtension, Error> {
        self.api().gpu_custom_call_extension()
    }

    /// Registers a GPU custom call handler. Refer to the documentation of
    /// [`Client::register_gpu_custom_call`] for more information.
    pub fn register_gpu_custom_call<N: AsRef<str>, H: Into<GpuCustomCallHandler>>(
        &self,
        name: N,
        handler: H,
    ) -> Result<(), Error> {
        self.api().register_gpu_custom_call(name, handler)
    }
}

impl Api {
    /// Attempts to load the [`GpuCustomCallExtension`] from this [`Api`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub(crate) fn gpu_custom_call_extension(&self) -> Result<GpuCustomCallExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let gpu_custom_call_extension = GpuCustomCallExtension::from_c_api(extension, *self);
                if let Some(gpu_custom_call_extension) = gpu_custom_call_extension {
                    return Ok(gpu_custom_call_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the GPU custom call extension is not provided by the PJRT plugin"))
        }
    }

    /// Registers a GPU custom call handler. Refer to the documentation of
    /// [`Client::register_gpu_custom_call`] for more information.
    pub(crate) fn register_gpu_custom_call<N: AsRef<str>, H: Into<GpuCustomCallHandler>>(
        &self,
        name: N,
        handler: H,
    ) -> Result<(), Error> {
        use ffi::PJRT_Gpu_Register_Custom_Call_Args;
        let extension = self.gpu_custom_call_extension()?;
        let name = name.as_ref();
        let handler = handler.into();
        match handler {
            GpuCustomCallHandler::Untyped(handler) => {
                invoke_pjrt_api_error_fn!(
                    @extension ffi::PJRT_Gpu_Custom_Call_Extension => extension,
                    PJRT_Gpu_Register_Custom_Call,
                    {
                        function_name = name.as_ptr() as *const _,
                        function_name_size = name.len(),
                        api_version = 0,
                        handler_instantiate = std::ptr::null_mut(),
                        handler_prepare = std::ptr::null_mut(),
                        handler_initialize = std::ptr::null_mut(),
                        handler_execute = handler.execute,
                    },
                )
            }
            GpuCustomCallHandler::Typed(handler) => {
                invoke_pjrt_api_error_fn!(
                    @extension ffi::PJRT_Gpu_Custom_Call_Extension => extension,
                    PJRT_Gpu_Register_Custom_Call,
                    {
                        function_name = name.as_ptr() as *const _,
                        function_name_size = name.len(),
                        api_version = 1,
                        handler_instantiate = handler.instantiate
                            .map(|handler| handler.to_c_api() as *mut std::ffi::c_void)
                            .unwrap_or(std::ptr::null_mut()),
                        handler_prepare = handler.prepare
                            .map(|handler| handler.to_c_api() as *mut std::ffi::c_void)
                            .unwrap_or(std::ptr::null_mut()),
                        handler_initialize = handler.initialize
                            .map(|handler| handler.to_c_api() as *mut std::ffi::c_void)
                            .unwrap_or(std::ptr::null_mut()),
                        handler_execute = handler.execute.to_c_api() as *mut std::ffi::c_void,
                    },
                )
            }
        }
    }
}

/// Handler for a GPU custom call registration.
#[derive(Copy, Clone)]
pub enum GpuCustomCallHandler {
    /// Legacy untyped XLA handler. All arguments to this handler are passed as a flat `*mut *mut c_void` buffer
    /// array (inputs followed by outputs) plus an opaque byte string representing any backend-specific configuration.
    /// This is generally dispreferred to the newer [`GpuCustomCallHandler::Typed`] as it provides no compile-time type
    /// safety for argument counts, buffer shapes, element types, or attributes. Furthermore, with the newer
    /// [`GpuCustomCallHandler::Typed`] handlers you can more easily wrap normal Rust functions to provide as GPU
    /// custom call handlers.
    Untyped(GpuCustomCallUntypedHandler),

    /// Typed [XLA FFI](https://openxla.org/xla/custom_call#xla-ffi) handler that provides type-safe buffer bindings
    /// structured attribute decoding from any backend-specific configuration, and access to the current execution
    /// context (e.g., stream, scratch allocator, etc.) through an XLA FFI call frame. Furthermore, it supports up to
    /// four separate handler stages (i.e., _instantiation_, _preparation_, _initialization_, and _execution_), of which
    /// only the _execution_ stage is required.
    Typed(GpuCustomCallTypedHandler),
}

/// Legacy untyped [`GpuCustomCallHandler`].
#[derive(Copy, Clone)]
pub struct GpuCustomCallUntypedHandler {
    /// Handle that represents this [`GpuCustomCallUntypedHandler`] in the PJRT C API.
    execute: *mut std::ffi::c_void,
}

impl GpuCustomCallUntypedHandler {
    /// Creates a new [`GpuCustomCallUntypedHandler`] using XLA's _original_ API which cannot return an
    /// [`XlaCustomCallStatus`] (i.e., in contrast to [`GpuCustomCallUntypedHandler::status_returning`]).
    /// The `callback` function's signature matches the signature of XLA's [`CustomCallWithOpaqueStreamHandle`](
    /// https://github.com/openxla/xla/blob/main/xla/backends/gpu/runtime/custom_call_target.h).
    pub const fn original(
        callback: unsafe extern "C" fn(
            stream: *mut std::ffi::c_void,
            buffers: *mut *mut std::ffi::c_void,
            backend_configuration: *const std::ffi::c_char,
            backend_configuration_len: usize,
        ),
    ) -> Self {
        Self { execute: callback as *mut std::ffi::c_void }
    }

    /// Creates a new [`GpuCustomCallUntypedHandler`] using XLA's _status-returning_ API which can return an
    /// [`XlaCustomCallStatus`] (i.e., in contrast to [`GpuCustomCallUntypedHandler::original`]). The `callback`
    /// function's signature matches the signature of XLA's [`CustomCallWithStatusAndOpaqueStreamHandle`](
    /// https://github.com/openxla/xla/blob/main/xla/backends/gpu/runtime/custom_call_target.h).
    pub const fn status_returning(
        callback: unsafe extern "C" fn(
            stream: *mut std::ffi::c_void,
            buffers: *mut *mut std::ffi::c_void,
            backend_configuration: *const std::ffi::c_char,
            backend_configuration_len: usize,
            status: *mut XlaCustomCallStatus,
        ),
    ) -> Self {
        Self { execute: callback as *mut std::ffi::c_void }
    }
}

unsafe impl Send for GpuCustomCallUntypedHandler {}
unsafe impl Sync for GpuCustomCallUntypedHandler {}

impl From<GpuCustomCallUntypedHandler> for GpuCustomCallHandler {
    fn from(value: GpuCustomCallUntypedHandler) -> Self {
        Self::Untyped(value)
    }
}

/// Represents the status of a GPU custom call using the legacy untyped API (i.e., [`GpuCustomCallHandler::Untyped`]).
pub struct GpuCustomCallStatus {
    /// Handle that represents this [`GpuCustomCallStatus`] in the PJRT C API.
    handle: *mut XlaCustomCallStatus,
}

impl GpuCustomCallStatus {
    /// Constructs a new [`GpuCustomCallStatus`] from the provided [`XlaCustomCallStatus`] handle that came from a
    /// function in the PJRT C API. This is typically used to interact with [`XlaCustomCallStatus`] handles from within
    /// [`GpuCustomCallHandler::Untyped`] handlers.
    pub unsafe fn from_c_api(handle: *mut XlaCustomCallStatus) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT GPU custom call status handle is a null pointer"))
        } else {
            Ok(Self { handle })
        }
    }

    /// Marks the GPU custom call invocation that this [`GpuCustomCallStatus`] is part of as successful.
    pub fn set_success(&mut self) {
        unsafe {
            XlaCustomCallStatusSetSuccess(self.handle);
        }
    }

    /// Marks the GPU custom call invocation that this [`GpuCustomCallStatus`] is part of as failed
    /// with the provided message offering information as to why the invocation failed.
    pub fn set_failure<M: AsRef<str>>(&mut self, message: M) {
        let message = message.as_ref();
        unsafe {
            XlaCustomCallStatusSetFailure(self.handle, message.as_ptr() as *const std::ffi::c_char, message.len());
        }
    }
}

/// Typed XLA FFI [`GpuCustomCallHandler`].
#[derive(Copy, Clone)]
pub struct GpuCustomCallTypedHandler {
    /// Refer to the documentation of [`GpuCustomCallFfiHandler::new`] for information on this field.
    instantiate: Option<FfiHandler>,

    /// Refer to the documentation of [`GpuCustomCallFfiHandler::new`] for information on this field.
    prepare: Option<FfiHandler>,

    /// Refer to the documentation of [`GpuCustomCallFfiHandler::new`] for information on this field.
    initialize: Option<FfiHandler>,

    /// Refer to the documentation of [`GpuCustomCallFfiHandler::new`] for information on this field.
    execute: FfiHandler,
}

impl GpuCustomCallTypedHandler {
    /// Constructs a new [`GpuCustomCallTypedHandler`].
    ///
    /// # Parameters
    ///
    ///   - `instantiate`: Optional [`FfiHandler`] for the _instantiation_ stage. When present, this callback is invoked
    ///     as part of [`Executable`](crate::Executable) instantiation (i.e., before program execution). Each GPU custom
    ///     call site gets its own handler instance at this stage, and instantiation is process-scoped rather than
    ///     device-scoped. At this stage, handlers may inspect attributes and execution context metadata, but do not
    ///     have access to device-specific resources (e.g., stream) or initialized input data.
    ///   - `prepare`: Optional [`FfiHandler`] for the _preparation_ stage. When present, this callback is invoked
    ///     before each execution of the compiled [`Executable`](crate::Executable). At this stage, handlers may request
    ///     runtime resources (e.g., collective cliques) needed by the upcoming execution. This handler should not
    ///     attempt to dereference input or output buffers because they may be uninitialized at this stage.
    ///   - `initialize`: Optional [`FfiHandler`] for the _initialization_ stage. When present, this callback is
    ///     invoked before execution and after all resources requested during `prepare` have been acquired. Similar
    ///     to `prepare`, this handler should not attempt to dereference input or output buffers because they may be
    ///     uninitialized at this stage.
    ///   - `execute`: [`FfiHandler`] for the _execution_ stage. This callback is invoked when the GPU custom call runs
    ///     as part of a PJRT program execution. For GPU backends, handlers typically run on the host CPU and enqueue
    ///     device work using the stream that they can obtain from the execution context. Note that this handler may run
    ///     during command-buffer capture (or CUDA graph capture), in which case the input buffers may contain
    ///     uninitialized values. This means that the handler should use the inputs as _device addresses_ to wire
    ///     into enqueue GPU operations, and not as host-readable values. Specifically, the handler can obtain input
    ///     shapes, data types, and other attributes from the XLA FFI call frame, and then enqueue operations to the
    ///     stream it obtains from the execution context, but it *must not* attempt to dereference any input device
    ///     buffers. This also means that it cannot have host-side control flow depend on the runtime values of those
    ///     buffers.
    pub fn new(
        instantiate: Option<FfiHandler>,
        prepare: Option<FfiHandler>,
        initialize: Option<FfiHandler>,
        execute: FfiHandler,
    ) -> Self {
        Self { instantiate, prepare, initialize, execute }
    }
}

unsafe impl Send for GpuCustomCallTypedHandler {}
unsafe impl Sync for GpuCustomCallTypedHandler {}

impl From<FfiHandler> for GpuCustomCallTypedHandler {
    fn from(value: FfiHandler) -> Self {
        Self::new(None, None, None, value)
    }
}

impl From<FfiHandler> for GpuCustomCallHandler {
    fn from(value: FfiHandler) -> Self {
        Self::Typed(GpuCustomCallTypedHandler::new(None, None, None, value))
    }
}

impl From<GpuCustomCallTypedHandler> for GpuCustomCallHandler {
    fn from(value: GpuCustomCallTypedHandler) -> Self {
        Self::Typed(value)
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;

    pub const PJRT_API_GPU_EXTENSION_VERSION: usize = 2;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct PJRT_Gpu_Register_Custom_Call_Args {
        pub struct_size: usize,
        pub function_name: *const std::ffi::c_char,
        pub function_name_size: usize,
        pub api_version: std::ffi::c_int,
        pub handler_instantiate: *mut std::ffi::c_void,
        pub handler_prepare: *mut std::ffi::c_void,
        pub handler_initialize: *mut std::ffi::c_void,
        pub handler_execute: *mut std::ffi::c_void,
    }

    impl PJRT_Gpu_Register_Custom_Call_Args {
        pub fn new(
            function_name: *const std::ffi::c_char,
            function_name_size: usize,
            api_version: std::ffi::c_int,
            handler_instantiate: *mut std::ffi::c_void,
            handler_prepare: *mut std::ffi::c_void,
            handler_initialize: *mut std::ffi::c_void,
            handler_execute: *mut std::ffi::c_void,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                function_name,
                function_name_size,
                api_version,
                handler_instantiate,
                handler_prepare,
                handler_initialize,
                handler_execute,
            }
        }
    }

    pub type PJRT_Gpu_Register_Custom_Call =
        unsafe extern "C" fn(args: *mut PJRT_Gpu_Register_Custom_Call_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Gpu_Custom_Call_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_Gpu_Register_Custom_Call: Option<PJRT_Gpu_Register_Custom_Call>,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;

    use crate::extensions::ffi::{
        FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiInput, FfiOutput, FfiTypeId,
        ffi::XLA_FFI_CallFrame, ffi::XLA_FFI_Error,
    };
    use crate::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use crate::tests::{TestPlatform, test_for_each_platform};
    use crate::{BufferType, Client, Error, ExecutionDeviceInputs, ExecutionInput, Program};

    use super::{GpuCustomCallStatus, GpuCustomCallUntypedHandler, XlaCustomCallStatus};

    /// Internal helper for our tests that compiles and executes a program that includes a custom call.
    fn execute_custom_call_program(client: &Client<'_>, api_version: i32) {
        let program = Program::Mlir {
            bytecode: indoc! {"
                module {
                func.func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
                    %0 = stablehlo.custom_call @\"ryft.test.gpu_custom_call\"(%arg0, %arg1) \
                        {api_version = __API_VERSION__ : i32} \
                    : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
                    return %0 : tensor<1xi32>
                }
                }
            "}
            .replace("__API_VERSION__", &api_version.to_string())
            .as_bytes()
            .to_vec(),
        };
        let options = CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 1,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        };
        let executable = client.compile(&program, &options).unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        let inputs = ExecutionDeviceInputs {
            inputs: &[
                ExecutionInput {
                    buffer: client
                        .buffer(7i32.to_ne_bytes().as_slice(), BufferType::I32, &[1], None, device.clone(), None)
                        .unwrap(),
                    donatable: false,
                },
                ExecutionInput {
                    buffer: client
                        .buffer(35i32.to_ne_bytes().as_slice(), BufferType::I32, &[1], None, device.clone(), None)
                        .unwrap(),
                    donatable: false,
                },
            ],
            ..Default::default()
        };
        let outputs = executable.execute(vec![inputs], 0, None, None, None, None).unwrap().remove(0);
        assert!(outputs.done.r#await().is_ok());
        assert_eq!(outputs.outputs.len(), 1);
    }

    #[test]
    fn test_gpu_custom_call_extension() {
        test_for_each_platform!(|plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    assert!(plugin.gpu_custom_call_extension().is_ok());
                    assert!(client.gpu_custom_call_extension().is_ok());
                }
                _ => {
                    assert!(matches!(plugin.gpu_custom_call_extension(), Err(Error::Unimplemented { .. })));
                    assert!(matches!(client.gpu_custom_call_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }

    #[test]
    fn test_client_register_untyped_gpu_custom_call() {
        test_for_each_platform!(|_plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    // We use an atomic counter to track the number of times our custom call function is invoked.
                    static CALLBACK_INVOCATIONS_COUNTER: AtomicUsize = AtomicUsize::new(0);

                    unsafe extern "C" fn custom_call(
                        _stream: *mut std::ffi::c_void,
                        buffers: *mut *mut std::ffi::c_void,
                        _opaque: *const std::ffi::c_char,
                        _opaque_len: usize,
                        status: *mut XlaCustomCallStatus,
                    ) {
                        unsafe {
                            let Ok(mut status) = GpuCustomCallStatus::from_c_api(status) else { return };
                            if buffers.is_null()
                                || (*buffers.add(0)).is_null()
                                || (*buffers.add(1)).is_null()
                                || (*buffers.add(2)).is_null()
                            {
                                status.set_failure("received null input/output buffers");
                                return;
                            }
                            CALLBACK_INVOCATIONS_COUNTER.fetch_add(1, Ordering::SeqCst);
                            status.set_success();
                        }
                    }

                    assert_eq!(
                        client.register_gpu_custom_call(
                            "ryft.test.gpu_custom_call",
                            GpuCustomCallUntypedHandler::status_returning(custom_call),
                        ),
                        Ok(()),
                    );

                    execute_custom_call_program(&client, 2);
                    assert!(CALLBACK_INVOCATIONS_COUNTER.load(Ordering::SeqCst) > 0);
                }
                _ => {
                    assert!(matches!(client.gpu_custom_call_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }

    #[test]
    fn test_client_register_typed_gpu_custom_call() {
        test_for_each_platform!(|_plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    // We use an atomic counter to track the number of times our custom call function is invoked.
                    static CALLBACK_INVOCATIONS_COUNTER: AtomicUsize = AtomicUsize::new(0);

                    unsafe extern "C" fn custom_call(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
                        unsafe {
                            match FfiCallFrame::from_c_api(call_frame) {
                                Err(_) => std::ptr::null_mut(),
                                Ok(call_frame) if call_frame.register_metadata(FfiTypeId::default()) => {
                                    std::ptr::null_mut()
                                }
                                Ok(call_frame) => {
                                    let api = call_frame.api().expect("encountered null XLA FFI API pointer");

                                    if call_frame.stage() != FfiExecutionStage::Execution {
                                        return FfiError::internal(
                                            "received non-execute stage in test FFI custom call",
                                        )
                                        .to_c_api(api);
                                    }

                                    let input_count = call_frame.input_count();
                                    let output_count = call_frame.output_count();

                                    if input_count != 2 {
                                        return FfiError::internal(format!("expected 2 inputs but got {input_count}"))
                                            .to_c_api(api);
                                    }

                                    if output_count != 1 {
                                        return FfiError::internal(format!("expected 1 output but got {output_count}"))
                                            .to_c_api(api);
                                    }

                                    if !matches!(
                                        call_frame.input(0),
                                        Ok(FfiInput::Buffer { buffer })
                                            if buffer.element_type() == FfiBufferType::I32
                                                && buffer.rank() == 1
                                                && buffer.dimensions() == &[1]
                                                && !buffer.data().is_null(),
                                    ) {
                                        return FfiError::internal("failed to decode inputs as scalar i32 buffers")
                                            .to_c_api(api);
                                    }

                                    if !matches!(
                                        call_frame.input(1),
                                        Ok(FfiInput::Buffer { buffer })
                                            if buffer.element_type() == FfiBufferType::I32
                                                && buffer.rank() == 1
                                                && buffer.dimensions() == &[1]
                                                && !buffer.data().is_null(),
                                    ) {
                                        return FfiError::internal("failed to decode inputs as scalar i32 buffers")
                                            .to_c_api(api);
                                    }

                                    if !matches!(
                                        call_frame.output(0),
                                        Ok(FfiOutput::Buffer { buffer })
                                            if buffer.element_type() == FfiBufferType::I32
                                                && buffer.rank() == 1
                                                && buffer.dimensions() == &[1]
                                                && !buffer.data().is_null(),
                                    ) {
                                        return FfiError::internal(
                                            "failed to decode outputs as a single scalar i32 buffer",
                                        )
                                        .to_c_api(api);
                                    }

                                    CALLBACK_INVOCATIONS_COUNTER.fetch_add(1, Ordering::SeqCst);
                                    std::ptr::null_mut()
                                }
                            }
                        }
                    }

                    assert_eq!(
                        client.register_gpu_custom_call("ryft.test.gpu_custom_call", FfiHandler::from_c_api(custom_call),),
                        Ok(())
                    );

                    execute_custom_call_program(&client, 4);
                    assert!(CALLBACK_INVOCATIONS_COUNTER.load(Ordering::SeqCst) > 0);
                }
                _ => {
                    assert!(matches!(client.gpu_custom_call_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }
}
