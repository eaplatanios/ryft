pub mod attributes;
pub mod buffers;
pub mod context;
pub mod errors;
pub mod futures;
pub mod handlers;
pub mod types;
pub mod versions;

pub use attributes::{FfiArray, FfiAttribute, FfiAttributes, FfiScalar};
pub use buffers::{FfiBuffer, FfiBufferType};
pub use context::{FfiDeviceId, FfiExecutionContext, FfiExecutionState, FfiRunId, FfiStream, FfiTask, FfiUserData};
pub use errors::FfiError;
pub use ffi::*;
pub use futures::FfiFuture;
pub use handlers::{
    FfiApi, FfiCallFrame, FfiExecutionStage, FfiHandler, FfiHandlerBundle, FfiHandlerTraits, FfiInput, FfiOutput,
};
pub use types::{FfiTypeId, FfiTypeInformation};
pub use versions::{VERSION, Version};

use crate::{Api, Client, Error, ExecutionContext, Plugin, invoke_pjrt_api_error_fn};

/// The PJRT FFI extension provides capabilities for integrating with backend-specific
/// [_Foreign Function Interfaces (FFI)_](https://en.wikipedia.org/wiki/Foreign_function_interface).
/// For example, XLA CPU, GPU, and TPU backends, it provides access to the XLA FFI internals.
///
/// Refer to the [official XLA documentation](https://openxla.org/xla/custom_call) for more information.
#[derive(Copy, Clone)]
pub struct FfiExtension {
    /// Handle that represents this [`FfiExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_FFI_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl FfiExtension {
    /// Constructs a new [`FfiExtension`] from the provided [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base)
    /// handle if the type of that PJRT extension matches the PJRT FFI extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_FFI {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_FFI_Extension`] that corresponds to this [`FfiExtension`] and which can be passed
    /// to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *const PJRT_FFI_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for FfiExtension {}
unsafe impl Sync for FfiExtension {}

impl Client<'_> {
    /// Attempts to load the [`FfiExtension`] from this [`Client`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub fn ffi_extension(&self) -> Result<FfiExtension, Error> {
        self.api().ffi_extension()
    }

    /// Registers a user-defined FFI type in the XLA runtime type registry. If the provided [`FfiTypeId`] is set to
    /// [`FfiTypeId::UNKNOWN`] then XLA will assign a unit type ID to the new type that will be returned by this
    /// function. Otherwise, it will verify that the provided type ID matches any previously registered type ID
    /// associated with the same type name.
    pub fn register_ffi_type<N: AsRef<str>>(
        &self,
        name: N,
        id: FfiTypeId,
        information: FfiTypeInformation,
    ) -> Result<FfiTypeId, Error> {
        self.api().register_ffi_type(name, id, information)
    }

    /// Registers an XLA custom call [`FfiHandler`] under the provided `name`, making it available to any compiled
    /// PJRT [`Executable`](crate::Executable) that contains a `stablehlo.custom_call` operation with a matching
    /// `call_target_name`.
    ///
    /// XLA [custom calls](https://openxla.org/xla/custom_call) are the primary escape hatch for
    /// running operations that the XLA compiler does not natively support. At compile time, a `stablehlo.custom_call`
    /// (or its HLO equivalent) records a target name and an optional opaque backend configuration. At runtime, XLA
    /// looks up the target name in a handler registry and invokes the corresponding function. This method populates
    /// that registry.
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
    /// Note that if the provided [`FfiHandler`] is stateful or requires access to some user-provided data
    /// (e.g., a boxed Rust closure), you must also use the [`Client::register_ffi_type`] and
    /// [`ExecutionContext::add_ffi_user_data`] functions, respectively.
    pub fn register_ffi_handler<N: AsRef<str>, P: AsRef<str>>(
        &self,
        name: N,
        platform: P,
        handler: FfiHandler,
        traits: FfiHandlerTraits,
    ) -> Result<(), Error> {
        self.api().register_ffi_handler(name, platform, handler, traits)
    }
}

impl Plugin {
    /// Attempts to load the [`FfiExtension`] from this [`Plugin`] and returns [`Error::Unimplemented`]
    /// if it is not provided by this [`Plugin`].
    pub fn ffi_extension(&self) -> Result<FfiExtension, Error> {
        self.api().ffi_extension()
    }

    /// Registers a user-defined FFI type in the XLA runtime type registry. Refer to the documentation
    /// of [`Client::register_ffi_type`] for more information.
    pub fn register_ffi_type<N: AsRef<str>>(
        &self,
        name: N,
        id: FfiTypeId,
        information: FfiTypeInformation,
    ) -> Result<FfiTypeId, Error> {
        self.api().register_ffi_type(name, id, information)
    }

    /// Registers an XLA custom call [`FfiHandler`] under the provided `name` and for the specified platform.
    /// Refer to the documentation of [`Client::register_ffi_handler`] for more information.
    pub fn register_ffi_handler<N: AsRef<str>, P: AsRef<str>>(
        &self,
        name: N,
        platform: P,
        handler: FfiHandler,
        traits: FfiHandlerTraits,
    ) -> Result<(), Error> {
        self.api().register_ffi_handler(name, platform, handler, traits)
    }
}

impl Api {
    /// Attempts to load the [`FfiExtension`] from this [`Api`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub(crate) fn ffi_extension(&self) -> Result<FfiExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let ffi_extension = FfiExtension::from_c_api(extension, *self);
                if let Some(ffi_extension) = ffi_extension {
                    return Ok(ffi_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the FFI extension is not provided by the PJRT plugin"))
        }
    }

    /// Registers a user-defined FFI type in the XLA runtime type registry. Refer to the documentation
    /// of [`Client::register_ffi_type`] for more information.
    pub(crate) fn register_ffi_type<N: AsRef<str>>(
        &self,
        name: N,
        id: FfiTypeId,
        information: FfiTypeInformation,
    ) -> Result<FfiTypeId, Error> {
        let extension = self.ffi_extension()?;
        let name = name.as_ref();
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_FFI_Type_Register,
            {
                type_name = name.as_ptr() as *const _,
                type_name_size = name.len(),
                type_id = id.into(),
                type_info = &mut PJRT_FFI_Type_Info {
                    deleter: information.deleter,
                    serialize: None,
                    deserialize: None,
                } as *mut _,
            },
            { type_id },
        )
        .map(FfiTypeId::new)
    }

    /// Registers an XLA custom call [`FfiHandler`] under the provided `name` and for the specified platform.
    /// Refer to the documentation of [`Client::register_ffi_handler`] for more information.
    pub(crate) fn register_ffi_handler<N: AsRef<str>, P: AsRef<str>>(
        &self,
        name: N,
        platform: P,
        handler: FfiHandler,
        traits: FfiHandlerTraits,
    ) -> Result<(), Error> {
        use ffi::PJRT_FFI_Register_Handler_Args;
        let extension = self.ffi_extension()?;
        let name = name.as_ref();
        let platform = platform.as_ref();
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_FFI_Register_Handler,
            {
                target_name = name.as_ptr() as *const _,
                target_name_size = name.len(),
                handler = handler.to_c_api(),
                platform_name = platform.as_ptr() as *const _,
                platform_name_size = platform.len(),
                traits = traits.to_c_api(),
            },
        )
    }
}

impl ExecutionContext {
    /// Adds the provided [`FfiUserData`] to this [`ExecutionContext`].
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data` remains valid/alive for the duration of the lifetime of the underlying
    /// XLA runtime execution context (since it may be accessed by custom [`FfiHandler`]s during that time).
    pub unsafe fn add_ffi_user_data(&self, data: FfiUserData) -> Result<(), Error> {
        use ffi::PJRT_FFI_UserData_Add_Args;
        let extension = self.api().ffi_extension()?;
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_FFI_UserData_Add,
            {
                context = self.to_c_api(),
                user_data = data.to_c_api(),
            },
        )
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub mod ffi {
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;
    use crate::programs::ffi::PJRT_ExecuteContext;

    pub use super::attributes::ffi::*;
    pub use super::buffers::ffi::*;
    pub use super::context::ffi::*;
    pub use super::errors::ffi::*;
    pub use super::futures::ffi::*;
    pub use super::handlers::ffi::*;
    pub use super::types::ffi::*;
    pub use super::versions::ffi::*;

    pub const PJRT_API_FFI_EXTENSION_VERSION: usize = 3;

    #[repr(C)]
    pub struct PJRT_FFI_Type_Info {
        pub deleter: Option<unsafe extern "C" fn(object: *mut std::ffi::c_void)>,
        pub serialize: Option<unsafe extern "C" fn()>,
        pub deserialize: Option<unsafe extern "C" fn()>,
    }

    #[repr(C)]
    pub struct PJRT_FFI_Type_Register_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub type_name: *const std::ffi::c_char,
        pub type_name_size: usize,
        pub type_id: i64,
        pub type_info: *mut PJRT_FFI_Type_Info,
    }

    impl PJRT_FFI_Type_Register_Args {
        pub fn new(
            type_name: *const std::ffi::c_char,
            type_name_size: usize,
            type_id: i64,
            type_info: *mut PJRT_FFI_Type_Info,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                type_name,
                type_name_size,
                type_id,
                type_info,
            }
        }
    }

    pub type PJRT_FFI_Type_Register = unsafe extern "C" fn(args: *mut PJRT_FFI_Type_Register_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_FFI_UserData {
        pub type_id: i64,
        pub data: *mut std::ffi::c_void,
    }

    #[repr(C)]
    pub struct PJRT_FFI_UserData_Add_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub context: *mut PJRT_ExecuteContext,
        pub user_data: PJRT_FFI_UserData,
    }

    impl PJRT_FFI_UserData_Add_Args {
        pub fn new(context: *mut PJRT_ExecuteContext, user_data: PJRT_FFI_UserData) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), context, user_data }
        }
    }

    pub type PJRT_FFI_UserData_Add = unsafe extern "C" fn(args: *mut PJRT_FFI_UserData_Add_Args) -> *mut PJRT_Error;

    pub type PJRT_FFI_Handler_TraitsBits = std::ffi::c_uint;
    pub const PJRT_FFI_Handler_TraitsBits_COMMAND_BUFFER_COMPATIBLE: PJRT_FFI_Handler_TraitsBits = 1;

    #[repr(C)]
    pub struct PJRT_FFI_Register_Handler_Args {
        pub struct_size: usize,
        pub target_name: *const std::ffi::c_char,
        pub target_name_size: usize,
        pub handler: XLA_FFI_Handler,
        pub platform_name: *const std::ffi::c_char,
        pub platform_name_size: usize,
        pub traits: PJRT_FFI_Handler_TraitsBits,
    }

    impl PJRT_FFI_Register_Handler_Args {
        pub fn new(
            target_name: *const std::ffi::c_char,
            target_name_size: usize,
            handler: XLA_FFI_Handler,
            platform_name: *const std::ffi::c_char,
            platform_name_size: usize,
            traits: PJRT_FFI_Handler_TraitsBits,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                target_name,
                target_name_size,
                handler,
                platform_name,
                platform_name_size,
                traits,
            }
        }
    }

    pub type PJRT_FFI_Register_Handler =
        unsafe extern "C" fn(args: *mut PJRT_FFI_Register_Handler_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_FFI_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_FFI_Type_Register: Option<PJRT_FFI_Type_Register>,
        pub PJRT_FFI_UserData_Add: Option<PJRT_FFI_UserData_Add>,
        pub PJRT_FFI_Register_Handler: Option<PJRT_FFI_Register_Handler>,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::any::Any;
    use std::collections::HashMap;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicI32, Ordering};

    use indoc::indoc;

    use crate::extensions::ffi::errors::ffi::XLA_FFI_Error;
    use crate::extensions::ffi::ffi::XLA_FFI_Api;
    use crate::extensions::ffi::handlers::ffi::XLA_FFI_CallFrame;
    use crate::extensions::ffi::{
        FfiApi, FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits, FfiInput,
        FfiOutput, FfiTypeId, FfiTypeInformation, FfiUserData,
    };
    use crate::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use crate::tests::{test_cpu_client, test_cpu_plugin};
    use crate::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program};

    const TEST_CALL_FRAME_USER_DATA_TYPE_NAME: &str = "ryft.test.ffi.call_frame.user_data";
    const TEST_CALL_FRAME_USER_DATA_TYPE_ID: FfiTypeId = FfiTypeId::new(9_000_001);

    /// Executes the provided closure with an [`FfiCallFrame`] captured from the current XLA runtime.
    pub(crate) fn with_test_ffi_call_frame<F>(handler: F)
    where
        F: FnOnce(FfiCallFrame<'_>),
    {
        /// Helper struct that is used to propagate panics from within the provided `handler` to the invocation
        /// context of `with_test_ffi_call_frame`.
        #[derive(Copy, Clone)]
        struct TestFfiCallFrameInvocation {
            handler_ptr: usize,
            panic_payload_ptr: usize,
        }

        // Use an [`AtomicUsize`] to ensure that each invocation of this function uses a different FFI handler name.
        static HANDLER_ID: AtomicI32 = AtomicI32::new(0);
        let handler_id = HANDLER_ID.fetch_add(1, Ordering::Relaxed) + 1;

        unsafe extern "C" fn custom_handler<F>(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error
        where
            F: FnOnce(FfiCallFrame<'_>),
        {
            unsafe {
                match FfiCallFrame::from_c_api(call_frame) {
                    Err(_) => std::ptr::null_mut(),
                    Ok(call_frame) if call_frame.register_metadata(TEST_CALL_FRAME_USER_DATA_TYPE_ID) => {
                        std::ptr::null_mut()
                    }
                    Ok(call_frame) if call_frame.stage() != FfiExecutionStage::Execution => std::ptr::null_mut(),
                    Ok(call_frame) => {
                        let Ok(context) = call_frame.context() else {
                            return std::ptr::null_mut();
                        };
                        let Ok(user_data) = context.user_data(TEST_CALL_FRAME_USER_DATA_TYPE_ID) else {
                            return std::ptr::null_mut();
                        };
                        if user_data.data.is_null() {
                            return std::ptr::null_mut();
                        }
                        let invocation = &mut *(user_data.data as *mut TestFfiCallFrameInvocation);
                        if invocation.handler_ptr == 0 {
                            return std::ptr::null_mut();
                        };
                        let handler = &mut *(invocation.handler_ptr as *mut Option<F>);
                        if let Some(handler) = handler.take() {
                            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                handler(call_frame);
                            }))
                            .err();
                            if let Some(panic) = panic {
                                let panic_payload =
                                    &mut *(invocation.panic_payload_ptr as *mut Option<Box<dyn Any + Send>>);
                                *panic_payload = Some(panic);
                            }
                        }
                        std::ptr::null_mut()
                    }
                }
            }
        }

        let mut handler = Some(handler);
        let mut panic_payload: Option<Box<dyn Any + Send>> = None;
        let handler_ptr = (&mut handler as *mut Option<F>) as usize;
        let panic_payload_ptr = (&mut panic_payload as *mut Option<Box<dyn Any + Send>>) as usize;
        let mut invocation = TestFfiCallFrameInvocation { handler_ptr, panic_payload_ptr };
        let invocation_ptr = (&mut invocation as *mut TestFfiCallFrameInvocation).cast::<std::ffi::c_void>();
        assert_ne!(handler_ptr, 0, "invalid test call-frame handler pointer");
        assert_ne!(panic_payload_ptr, 0, "invalid panic payload pointer");
        assert!(!invocation_ptr.is_null(), "invalid test invocation pointer");

        // Compile and execute a program that invokes our custom FFI handler.
        let ffi_handler = FfiHandler::from_c_api(custom_handler::<F>);
        let target_name = format!("ryft.test.ffi.handler.{handler_id}");
        let client = test_cpu_client();
        let platform_name = client.platform_name().unwrap().into_owned();
        assert_eq!(
            client.register_ffi_type(
                TEST_CALL_FRAME_USER_DATA_TYPE_NAME,
                TEST_CALL_FRAME_USER_DATA_TYPE_ID,
                FfiTypeInformation::new(None),
            ),
            Ok(TEST_CALL_FRAME_USER_DATA_TYPE_ID),
        );
        assert!(
            client
                .register_ffi_handler(target_name.as_str(), platform_name.as_str(), ffi_handler, FfiHandlerTraits::NONE)
                .is_ok()
        );
        let program = Program::Mlir {
            bytecode: indoc! {"
                module {
                func.func @main(%arg0: tensor<1xi32>) -> tensor<1xi32> {
                    %0 = stablehlo.custom_call @\"__TARGET_NAME__\"(%arg0) \
                        {
                          api_version = 4 : i32,
                          backend_config = {
                            array_attr = array<i32: 1, 2, 3>,
                            dictionary_attr = {
                              nested_array_attr = array<i32: 4, 5>,
                              nested_dictionary_attr = {
                                nested_inner_scalar_attr = 11 : i32,
                                nested_inner_string_attr = \"inner\"
                              },
                              nested_scalar_attr = 9 : i32,
                              nested_string_attr = \"nested_value\"
                            },
                            scalar_attr = 7 : i32,
                            string_attr = \"value\"
                          }
                        } \
                    : (tensor<1xi32>) -> tensor<1xi32>
                    return %0 : tensor<1xi32>
                }
                }
            "}
            .replace("__TARGET_NAME__", target_name.as_str())
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
        let execution_context = client.execution_context().unwrap();
        assert!(unsafe {
            execution_context
                .add_ffi_user_data(FfiUserData::new(TEST_CALL_FRAME_USER_DATA_TYPE_ID, invocation_ptr))
                .is_ok()
        });
        let device = executable.addressable_devices().unwrap()[0].clone();
        let inputs = ExecutionDeviceInputs {
            inputs: &[ExecutionInput {
                buffer: client
                    .buffer(7i32.to_ne_bytes().as_slice(), BufferType::I32, &[1], None, device, None)
                    .unwrap(),
                donatable: false,
            }],
            ..Default::default()
        };
        let outputs = executable.execute(vec![inputs], 0, Some(execution_context), None, None, None).unwrap().remove(0);
        assert!(outputs.done.r#await().is_ok());
        assert!(handler.is_none(), "test call-frame handler was never invoked");
        if let Some(panic_payload) = panic_payload {
            std::panic::resume_unwind(panic_payload);
        }
    }

    /// Returns an [`FfiApi`] captured from the current XLA runtime.
    pub(crate) fn test_ffi_api() -> FfiApi {
        static TEST_FFI_API_PTR: OnceLock<usize> = OnceLock::new();
        if TEST_FFI_API_PTR.get().is_none() {
            with_test_ffi_call_frame(|call_frame| {
                let _ = TEST_FFI_API_PTR.set(unsafe { (*call_frame.to_c_api()).api as usize });
            });
        }
        let ptr = TEST_FFI_API_PTR.get().copied().expect("failed to capture XLA FFI API pointer");
        unsafe { FfiApi::from_c_api(ptr as *const XLA_FFI_Api).unwrap() }
    }

    #[test]
    fn test_ffi_extension() {
        assert!(test_cpu_plugin().ffi_extension().is_ok());
        assert!(test_cpu_client().ffi_extension().is_ok());
    }

    #[test]
    fn test_ffi_handler() {
        const TEST_HANDLER_USER_DATA_TYPE_NAME: &str = "ryft.test.ffi.handler.user_data";
        const TEST_HANDLER_USER_DATA_TYPE_ID: FfiTypeId = FfiTypeId::new(9_000_002);

        struct TestFfiHandlerUserData {
            bias: i32,
            invocation_count: AtomicI32,
        }

        unsafe extern "C" fn custom_handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
            unsafe {
                match FfiCallFrame::from_c_api(call_frame) {
                    Err(_) => std::ptr::null_mut(),
                    Ok(call_frame) if call_frame.register_metadata(TEST_HANDLER_USER_DATA_TYPE_ID) => {
                        std::ptr::null_mut()
                    }
                    Ok(call_frame) => {
                        let api = match call_frame.api() {
                            Ok(api) => api,
                            Err(_) => return std::ptr::null_mut(),
                        };

                        if call_frame.stage() != FfiExecutionStage::Execution {
                            return FfiError::internal("received non-execute stage in test FFI handler").to_c_api(api);
                        }

                        let Ok(FfiInput::Buffer { buffer: lhs }) = call_frame.input(0) else {
                            return FfiError::internal("failed to decode first input buffer").to_c_api(api);
                        };

                        let Ok(FfiInput::Buffer { buffer: rhs }) = call_frame.input(1) else {
                            return FfiError::internal("failed to decode second input buffer").to_c_api(api);
                        };

                        let Ok(FfiOutput::Buffer { buffer: output }) = call_frame.output(0) else {
                            return FfiError::internal("failed to decode output buffer").to_c_api(api);
                        };

                        if lhs.element_type() != FfiBufferType::I32
                            || rhs.element_type() != FfiBufferType::I32
                            || output.element_type() != FfiBufferType::I32
                        {
                            return FfiError::internal("expected i32 input/output buffers").to_c_api(api);
                        }

                        if lhs.rank() != 1
                            || rhs.rank() != 1
                            || output.rank() != 1
                            || lhs.dimensions() != [1]
                            || rhs.dimensions() != [1]
                            || output.dimensions() != [1]
                        {
                            return FfiError::internal("expected rank-1 input/output buffers with one element")
                                .to_c_api(api);
                        }

                        if lhs.data().is_null() || rhs.data().is_null() || output.data().is_null() {
                            return FfiError::internal("encountered null input/output data pointers").to_c_api(api);
                        }

                        let context = match call_frame.context() {
                            Ok(context) => context,
                            Err(_) => {
                                return FfiError::internal("failed to decode FFI execution context").to_c_api(api);
                            }
                        };

                        let user_data = match context.user_data(TEST_HANDLER_USER_DATA_TYPE_ID) {
                            Ok(user_data) if !user_data.data.is_null() => user_data,
                            _ => return FfiError::internal("failed to get test handler user data").to_c_api(api),
                        };

                        let user_data = &*(user_data.data as *const TestFfiHandlerUserData);
                        let lhs_value = *lhs.data().cast::<i32>();
                        let rhs_value = *rhs.data().cast::<i32>();
                        *output.data().cast::<i32>() = lhs_value + rhs_value + user_data.bias;
                        user_data.invocation_count.fetch_add(1, Ordering::SeqCst);
                        std::ptr::null_mut()
                    }
                }
            }
        }

        static HANDLER_ID: AtomicI32 = AtomicI32::new(0);
        let handler_id = HANDLER_ID.fetch_add(1, Ordering::Relaxed) + 1;
        let target_name = format!("ryft.test.ffi.stateful.handler.{handler_id}");
        let client = test_cpu_client();
        let platform_name = client.platform_name().unwrap().into_owned();

        assert_eq!(
            client.register_ffi_type(
                TEST_HANDLER_USER_DATA_TYPE_NAME,
                TEST_HANDLER_USER_DATA_TYPE_ID,
                FfiTypeInformation::new(None),
            ),
            Ok(TEST_HANDLER_USER_DATA_TYPE_ID),
        );

        assert_eq!(
            client.register_ffi_handler(
                target_name.as_str(),
                platform_name.as_str(),
                FfiHandler::from_c_api(custom_handler),
                FfiHandlerTraits::NONE,
            ),
            Ok(()),
        );

        let program = Program::Mlir {
            bytecode: indoc! {"
                module {
                func.func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
                    %0 = stablehlo.custom_call @\"__TARGET_NAME__\"(%arg0, %arg1) \
                        {api_version = 4 : i32} \
                    : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
                    return %0 : tensor<1xi32>
                }
                }
            "}
            .replace("__TARGET_NAME__", target_name.as_str())
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
        let execution_context = client.execution_context().unwrap();

        let user_data = TestFfiHandlerUserData { bias: 11, invocation_count: AtomicI32::new(0) };
        assert!(unsafe {
            execution_context
                .add_ffi_user_data(FfiUserData::new(
                    TEST_HANDLER_USER_DATA_TYPE_ID,
                    (&user_data as *const TestFfiHandlerUserData).cast_mut().cast::<std::ffi::c_void>(),
                ))
                .is_ok()
        });

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

        let mut outputs = executable.execute(vec![inputs], 0, Some(execution_context), None, None, None).unwrap();
        assert_eq!(outputs.len(), 1);
        let mut outputs = outputs.remove(0);
        assert!(outputs.done.r#await().is_ok());
        assert_eq!(outputs.outputs.len(), 1);
        let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(output_bytes, 53i32.to_ne_bytes().to_vec());

        assert_eq!(user_data.invocation_count.load(Ordering::SeqCst), 1);
    }
}
