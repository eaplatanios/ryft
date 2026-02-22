use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::extensions::ffi::{FfiHandler, FfiHandlerTraits, FfiTypeId, FfiTypeInfo};
use crate::extensions::gpu_custom_call::{GpuCustomCallHandler, GpuCustomCallUntypedHandler};
use crate::{Client, Error};

use super::JaxPlatform;

/// JAX-style custom-call target wrapper.
#[derive(Copy, Clone)]
pub enum JaxCustomCallTarget {
    Typed(FfiHandler),
    Untyped(GpuCustomCallUntypedHandler),
}

/// JAX-style handler signature used for registering custom call targets.
pub type JaxCustomCallHandler = Arc<
    dyn Fn(&str, JaxCustomCallTarget, JaxPlatform, i32, FfiHandlerTraits) -> Result<(), Error> + Send + Sync + 'static,
>;

/// JAX-style handler signature used for registering custom type IDs.
pub type JaxCustomTypeHandler =
    Arc<dyn Fn(&str, FfiTypeId, FfiTypeInfo, JaxPlatform) -> Result<FfiTypeId, Error> + Send + Sync + 'static>;

#[derive(Copy, Clone)]
struct RegisteredCustomCall {
    target: JaxCustomCallTarget,
    api_version: i32,
    traits: FfiHandlerTraits,
}

#[derive(Clone)]
struct PendingCustomCall {
    target_name: String,
    target: JaxCustomCallTarget,
    api_version: i32,
    traits: FfiHandlerTraits,
}

#[derive(Copy, Clone)]
struct RegisteredCustomType {
    type_id: FfiTypeId,
    type_info: FfiTypeInfo,
}

#[derive(Clone)]
struct PendingCustomType {
    type_name: String,
    type_id: FfiTypeId,
    type_info: FfiTypeInfo,
}

#[derive(Default)]
struct JaxFfiRegistryState {
    custom_call_handlers: HashMap<String, JaxCustomCallHandler>,
    custom_type_handlers: HashMap<String, JaxCustomTypeHandler>,
    pending_custom_calls: HashMap<String, Vec<PendingCustomCall>>,
    pending_custom_types: HashMap<String, Vec<PendingCustomType>>,
    registered_custom_calls: HashMap<(String, String), RegisteredCustomCall>,
    registered_custom_types: HashMap<(String, String), RegisteredCustomType>,
}

/// Registry that mirrors JAX registration flow for custom-call and type handlers.
#[derive(Default)]
pub struct JaxFfiRegistry {
    state: Mutex<JaxFfiRegistryState>,
}

impl JaxFfiRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a custom-call handler for `platform`. The first handler wins.
    pub fn register_custom_call_handler<P: Into<JaxPlatform>>(
        &self,
        platform: P,
        handler: JaxCustomCallHandler,
    ) -> Result<(), Error> {
        let platform = platform.into();
        let platform_key = platform.canonical_handler_name().into_owned();

        let pending_custom_calls = {
            let mut state = self.state.lock().unwrap();
            if state.custom_call_handlers.contains_key(platform_key.as_str()) {
                return Ok(());
            }
            state.custom_call_handlers.insert(platform_key.clone(), handler.clone());
            state.pending_custom_calls.remove(platform_key.as_str()).unwrap_or_default()
        };

        for (index, pending_custom_call) in pending_custom_calls.iter().enumerate() {
            let result = handler(
                pending_custom_call.target_name.as_str(),
                pending_custom_call.target,
                platform.clone(),
                pending_custom_call.api_version,
                pending_custom_call.traits,
            );
            if let Err(error) = result {
                let mut state = self.state.lock().unwrap();
                state
                    .pending_custom_calls
                    .entry(platform_key.clone())
                    .or_default()
                    .extend_from_slice(&pending_custom_calls[index..]);
                for pending_custom_call in pending_custom_calls.iter().skip(index) {
                    state
                        .registered_custom_calls
                        .remove(&(platform_key.clone(), pending_custom_call.target_name.clone()));
                }
                return Err(error);
            }
        }

        Ok(())
    }

    /// Registers a custom call target. Typed registrations require `api_version = 1`,
    /// untyped registrations require `api_version = 0`.
    pub fn register_custom_call_target<P: Into<JaxPlatform>>(
        &self,
        target_name: &str,
        target: JaxCustomCallTarget,
        platform: P,
        api_version: i32,
        traits: FfiHandlerTraits,
    ) -> Result<(), Error> {
        if target_name.is_empty() {
            return Err(Error::invalid_argument("cannot register a custom call target with an empty name"));
        }
        match (target, api_version) {
            (JaxCustomCallTarget::Typed(_), 1) | (JaxCustomCallTarget::Untyped(_), 0) => {}
            (JaxCustomCallTarget::Typed(_), _) => {
                return Err(Error::invalid_argument("typed custom call targets require api_version = 1"));
            }
            (JaxCustomCallTarget::Untyped(_), _) => {
                return Err(Error::invalid_argument("untyped custom call targets require api_version = 0"));
            }
        }

        let platform = platform.into();
        let platform_key = platform.canonical_handler_name().into_owned();

        let handler = {
            let mut state = self.state.lock().unwrap();
            let registration_key = (platform_key.clone(), target_name.to_string());
            if let Some(existing_registration) = state.registered_custom_calls.get(&registration_key).copied() {
                let existing_target_key = custom_call_target_key(existing_registration.target);
                let target_key = custom_call_target_key(target);
                if existing_target_key == target_key
                    && existing_registration.api_version == api_version
                    && existing_registration.traits == traits
                {
                    return Ok(());
                }
                return Err(Error::already_exists(format!(
                    "a different custom call target is already registered for '{target_name}' on platform '{platform_key}'"
                )));
            }

            state
                .registered_custom_calls
                .insert(registration_key, RegisteredCustomCall { target, api_version, traits });
            let handler = state.custom_call_handlers.get(platform_key.as_str()).cloned();
            if handler.is_none() {
                state.pending_custom_calls.entry(platform_key.clone()).or_default().push(PendingCustomCall {
                    target_name: target_name.to_string(),
                    target,
                    api_version,
                    traits,
                });
            }
            handler
        };

        if let Some(handler) = handler {
            let result = handler(target_name, target, platform, api_version, traits);
            if let Err(error) = result {
                let mut state = self.state.lock().unwrap();
                state.registered_custom_calls.remove(&(platform_key, target_name.to_string()));
                return Err(error);
            }
        }

        Ok(())
    }

    /// Registers a custom-type handler for `platform`. The first handler wins.
    pub fn register_custom_type_handler<P: Into<JaxPlatform>>(
        &self,
        platform: P,
        handler: JaxCustomTypeHandler,
    ) -> Result<(), Error> {
        let platform = platform.into();
        let platform_key = platform.canonical_handler_name().into_owned();

        let pending_custom_types = {
            let mut state = self.state.lock().unwrap();
            if state.custom_type_handlers.contains_key(platform_key.as_str()) {
                return Ok(());
            }
            state.custom_type_handlers.insert(platform_key.clone(), handler.clone());
            state.pending_custom_types.remove(platform_key.as_str()).unwrap_or_default()
        };

        for (index, pending_custom_type) in pending_custom_types.iter().enumerate() {
            let result = handler(
                pending_custom_type.type_name.as_str(),
                pending_custom_type.type_id,
                pending_custom_type.type_info,
                platform.clone(),
            );
            if let Err(error) = result {
                let mut state = self.state.lock().unwrap();
                state
                    .pending_custom_types
                    .entry(platform_key.clone())
                    .or_default()
                    .extend_from_slice(&pending_custom_types[index..]);
                for pending_custom_type in pending_custom_types.iter().skip(index) {
                    state
                        .registered_custom_types
                        .remove(&(platform_key.clone(), pending_custom_type.type_name.clone()));
                }
                return Err(error);
            }
        }

        Ok(())
    }

    /// Registers a custom type ID for `platform`.
    pub fn register_custom_type<P: Into<JaxPlatform>>(
        &self,
        type_name: &str,
        type_id: FfiTypeId,
        type_info: FfiTypeInfo,
        platform: P,
    ) -> Result<(), Error> {
        if type_name.is_empty() {
            return Err(Error::invalid_argument("cannot register a custom type with an empty name"));
        }

        let platform = platform.into();
        let platform_key = platform.canonical_handler_name().into_owned();

        let handler = {
            let mut state = self.state.lock().unwrap();
            let registration_key = (platform_key.clone(), type_name.to_string());
            if let Some(existing_registration) = state.registered_custom_types.get(&registration_key).copied() {
                let existing_type_key = custom_type_key(existing_registration.type_id, existing_registration.type_info);
                let type_key = custom_type_key(type_id, type_info);
                if existing_type_key == type_key {
                    return Ok(());
                }
                return Err(Error::already_exists(format!(
                    "a different custom type is already registered for '{type_name}' on platform '{platform_key}'"
                )));
            }

            state.registered_custom_types.insert(registration_key, RegisteredCustomType { type_id, type_info });
            let handler = state.custom_type_handlers.get(platform_key.as_str()).cloned();
            if handler.is_none() {
                state.pending_custom_types.entry(platform_key.clone()).or_default().push(PendingCustomType {
                    type_name: type_name.to_string(),
                    type_id,
                    type_info,
                });
            }
            handler
        };

        if let Some(handler) = handler {
            let result = handler(type_name, type_id, type_info, platform);
            if let Err(error) = result {
                let mut state = self.state.lock().unwrap();
                state.registered_custom_types.remove(&(platform_key, type_name.to_string()));
                return Err(error);
            }
        }

        Ok(())
    }

    /// Installs default PJRT-backed handlers for the platform associated with `client`.
    pub fn install_default_pjrt_handlers(&self, client: &Client<'_>) -> Result<(), Error> {
        let platform = JaxPlatform::from_client(client)?;
        let ffi_extension = client.ffi_extension();
        let api = client.api();

        self.register_custom_call_handler(
            platform.clone(),
            Arc::new(move |target_name, target, platform, api_version, traits| match target {
                JaxCustomCallTarget::Typed(handler) => {
                    if api_version != 1 {
                        return Err(Error::invalid_argument(
                            "typed custom call target registrations require api_version = 1",
                        ));
                    }
                    let ffi_extension = ffi_extension.as_ref().map_err(|_| {
                        Error::unimplemented("the FFI extension is required to register typed custom call targets")
                    })?;

                    let mut last_error = None;
                    for platform_name in platform.registration_name_candidates() {
                        match ffi_extension.register_handler(target_name, platform_name.as_str(), handler, traits) {
                            Ok(()) => return Ok(()),
                            Err(error) => last_error = Some(error),
                        }
                    }
                    Err(last_error.unwrap_or_else(|| {
                        Error::internal("failed to register typed custom call target for all platform candidates")
                    }))
                }
                JaxCustomCallTarget::Untyped(handler) => {
                    if api_version != 0 {
                        return Err(Error::invalid_argument(
                            "untyped custom call target registrations require api_version = 0",
                        ));
                    }
                    if !platform.is_gpu() {
                        return Err(Error::unimplemented(
                            "untyped custom call targets are only supported on GPU backends",
                        ));
                    }
                    api.register_gpu_custom_call(target_name, GpuCustomCallHandler::Untyped(handler))
                }
            }),
        )?;

        let ffi_extension = client.ffi_extension();
        self.register_custom_type_handler(
            platform,
            Arc::new(move |type_name, type_id, type_info, _platform| {
                let ffi_extension = ffi_extension
                    .as_ref()
                    .map_err(|_| Error::unimplemented("the FFI extension is required to register custom types"))?;
                ffi_extension.register_type(type_name, type_id, type_info)
            }),
        )
    }
}

fn custom_call_target_key(target: JaxCustomCallTarget) -> (u8, usize) {
    match target {
        JaxCustomCallTarget::Typed(handler) => {
            let pointer = handler.to_handler().map(|callback| callback as usize).unwrap_or_default();
            (1, pointer)
        }
        JaxCustomCallTarget::Untyped(handler) => (2, handler.to_handler_ptr() as usize),
    }
}

fn custom_type_key(type_id: FfiTypeId, type_info: FfiTypeInfo) -> (i64, usize) {
    let deleter_pointer = type_info.deleter().map(|deleter| deleter as usize).unwrap_or_default();
    (type_id.to_i64(), deleter_pointer)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::env;
    use std::ffi::c_void;
    use std::sync::{Arc, LazyLock, Mutex};

    use ryft_mlir::dialects::func;
    use ryft_mlir::dialects::stable_hlo::{CustomCallApiVersion, custom_call};
    use ryft_mlir::{Block, Context, Operation, Size};
    use ryft_xla_sys::bindings as xla_sys;

    use crate::extensions::ffi::{FfiError, ffi as xla_ffi};
    use crate::jax::gpu_runtime;
    use crate::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use crate::tests::{TestPlatform, test_for_each_platform};
    use crate::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program};

    use super::*;

    const TEST_VALUE_LHS: i32 = 7;
    const TEST_VALUE_RHS: i32 = 35;
    const EXPECTED_SUM: i32 = 42;

    fn make_xla_error(call_frame: *mut xla_ffi::XLA_FFI_CallFrame, message: &str) -> *mut xla_ffi::XLA_FFI_Error {
        if call_frame.is_null() {
            return std::ptr::null_mut();
        }
        let api = unsafe { (*call_frame).api };
        if api.is_null() {
            return std::ptr::null_mut();
        }
        FfiError::internal(message).to_c_api(api)
    }

    fn is_real_gpu_test_enabled() -> bool {
        env::var("RYFT_PJRT_RUN_GPU_KERNEL_TESTS").ok().as_deref() == Some("1")
    }

    fn is_gpu_platform(platform: TestPlatform) -> bool {
        matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7)
    }

    fn target_name(prefix: &str, platform: TestPlatform) -> String {
        let platform_suffix = match platform {
            TestPlatform::Cpu => "cpu",
            TestPlatform::Cuda12 => "cuda12",
            TestPlatform::Cuda13 => "cuda13",
            TestPlatform::Rocm7 => "rocm7",
            TestPlatform::Tpu => "tpu",
            TestPlatform::Neuron => "neuron",
            TestPlatform::Metal => "metal",
        };
        format!("ryft.test.jax.ffi.{prefix}.{platform_suffix}")
    }

    fn test_compilation_options() -> CompilationOptions {
        CompilationOptions {
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
        }
    }

    fn test_custom_call_program(target_name: &str) -> Program {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type =
            context.tensor_type(context.signless_integer_type(32), &[Size::Static(1)], None, location).unwrap();

        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let custom_call = custom_call(
                &[lhs, rhs],
                target_name,
                false,
                None,
                CustomCallApiVersion::TypedFfi,
                &[],
                None,
                &[],
                &[tensor_type],
                location,
            );
            let custom_call = block.append_operation(custom_call);
            block.append_operation(func::r#return(&[custom_call.result(0).unwrap()], location));
            func::func(
                "main",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), tensor_type.into()],
                    results: vec![tensor_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        Program::Mlir { bytecode: module.as_operation().bytecode() }
    }

    fn test_run_custom_call_program(client: &Client<'_>, target_name: &str) {
        let program = test_custom_call_program(target_name);
        let executable = client.compile(&program, &test_compilation_options()).unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();

        let lhs_bytes = TEST_VALUE_LHS.to_ne_bytes();
        let lhs = client.buffer(lhs_bytes.as_slice(), BufferType::I32, &[1], None, device.clone(), None).unwrap();
        let rhs_bytes = TEST_VALUE_RHS.to_ne_bytes();
        let rhs = client.buffer(rhs_bytes.as_slice(), BufferType::I32, &[1], None, device, None).unwrap();

        let inputs = ExecutionDeviceInputs {
            inputs: &[
                ExecutionInput { buffer: lhs, donatable: false },
                ExecutionInput { buffer: rhs, donatable: false },
            ],
            ..Default::default()
        };

        let mut outputs = executable.execute(vec![inputs], 0, None, None, None, None).unwrap();
        let mut outputs = outputs.remove(0);
        outputs.done.r#await().unwrap();
        let output = outputs.outputs.remove(0);
        let output_bytes = output.copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(output_bytes, EXPECTED_SUM.to_ne_bytes().to_vec());
    }

    unsafe fn ffi_buffer_data(
        types: *mut xla_sys::XLA_FFI_ArgType,
        buffers: *mut *mut c_void,
        index: usize,
    ) -> Option<*mut c_void> {
        if types.is_null() || buffers.is_null() {
            return None;
        }
        if unsafe { *types.add(index) } != xla_sys::XLA_FFI_ArgType_XLA_FFI_ArgType_BUFFER {
            return None;
        }
        let buffer = unsafe { *buffers.add(index) };
        if buffer.is_null() {
            return None;
        }
        let buffer = unsafe { &*(buffer as *const xla_sys::XLA_FFI_Buffer) };
        if buffer.dtype != xla_sys::XLA_FFI_DataType_XLA_FFI_DataType_S32 {
            return None;
        }
        if buffer.rank != 1 || buffer.dims.is_null() || unsafe { *buffer.dims } != 1 {
            return None;
        }
        Some(buffer.data)
    }

    unsafe fn ffi_result_buffer_data(
        types: *mut xla_sys::XLA_FFI_RetType,
        buffers: *mut *mut c_void,
        index: usize,
    ) -> Option<*mut c_void> {
        if types.is_null() || buffers.is_null() {
            return None;
        }
        if unsafe { *types.add(index) } != xla_sys::XLA_FFI_RetType_XLA_FFI_RetType_BUFFER {
            return None;
        }
        let buffer = unsafe { *buffers.add(index) };
        if buffer.is_null() {
            return None;
        }
        let buffer = unsafe { &*(buffer as *const xla_sys::XLA_FFI_Buffer) };
        if buffer.dtype != xla_sys::XLA_FFI_DataType_XLA_FFI_DataType_S32 {
            return None;
        }
        if buffer.rank != 1 || buffer.dims.is_null() || unsafe { *buffer.dims } != 1 {
            return None;
        }
        Some(buffer.data)
    }

    unsafe fn ffi_stream(call_frame: *const xla_sys::XLA_FFI_CallFrame) -> Option<*mut c_void> {
        if call_frame.is_null() {
            return None;
        }
        let call_frame = unsafe { &*call_frame };
        if call_frame.ctx.is_null() || call_frame.api.is_null() {
            return None;
        }
        let stream_get = unsafe { (*call_frame.api).XLA_FFI_Stream_Get }?;
        let mut args = xla_sys::XLA_FFI_Stream_Get_Args {
            struct_size: std::mem::size_of::<xla_sys::XLA_FFI_Stream_Get_Args>(),
            extension_start: std::ptr::null_mut(),
            ctx: call_frame.ctx,
            stream: std::ptr::null_mut(),
        };
        let error = unsafe { stream_get(&mut args as *mut _) };
        if !error.is_null() {
            if let Some(error_destroy) = unsafe { (*call_frame.api).XLA_FFI_Error_Destroy } {
                let mut destroy_args = xla_sys::XLA_FFI_Error_Destroy_Args {
                    struct_size: std::mem::size_of::<xla_sys::XLA_FFI_Error_Destroy_Args>(),
                    extension_start: std::ptr::null_mut(),
                    error,
                };
                unsafe { error_destroy(&mut destroy_args as *mut _) };
            }
            return None;
        }
        Some(args.stream)
    }

    unsafe extern "C" fn test_gpu_add_kernel_ffi_handler(
        call_frame: *mut xla_ffi::XLA_FFI_CallFrame,
    ) -> *mut xla_ffi::XLA_FFI_Error {
        if call_frame.is_null() {
            return std::ptr::null_mut();
        }

        let raw_call_frame = call_frame as *const xla_sys::XLA_FFI_CallFrame;
        let raw_call_frame_ref = unsafe { &*raw_call_frame };
        let argument_count = usize::try_from(raw_call_frame_ref.args.size).unwrap_or(0);
        let result_count = usize::try_from(raw_call_frame_ref.rets.size).unwrap_or(0);
        if argument_count != 2 || result_count != 1 {
            return make_xla_error(call_frame, "unexpected argument/result arity for add-kernel test custom call");
        }

        let stream = unsafe { ffi_stream(raw_call_frame) };
        let Some(stream) = stream else {
            return make_xla_error(call_frame, "failed to get stream from XLA FFI call frame");
        };

        let lhs_buffer = unsafe { ffi_buffer_data(raw_call_frame_ref.args.types, raw_call_frame_ref.args.args, 0) };
        let rhs_buffer = unsafe { ffi_buffer_data(raw_call_frame_ref.args.types, raw_call_frame_ref.args.args, 1) };
        let out_buffer =
            unsafe { ffi_result_buffer_data(raw_call_frame_ref.rets.types, raw_call_frame_ref.rets.rets, 0) };
        let (Some(lhs_buffer), Some(rhs_buffer), Some(out_buffer)) = (lhs_buffer, rhs_buffer, out_buffer) else {
            return make_xla_error(call_frame, "failed to decode scalar i32 buffers for add-kernel test");
        };

        match gpu_runtime::launch_add_i32(stream, lhs_buffer, rhs_buffer, out_buffer) {
            Ok(()) => std::ptr::null_mut(),
            Err(error) => {
                make_xla_error(call_frame, format!("failed to launch add-kernel test callback: {error}").as_str())
            }
        }
    }

    static CUSTOM_CALL_INVOCATIONS: LazyLock<Mutex<Vec<(String, i32, u32, String)>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));
    static CUSTOM_TYPE_INVOCATIONS: LazyLock<Mutex<Vec<(String, i64, String)>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));

    fn test_platform(platform: &str) -> JaxPlatform {
        JaxPlatform::from(platform)
    }

    #[test]
    fn test_register_custom_call_handler_flushes_pending_registrations() {
        let registry = JaxFfiRegistry::new();
        let platform = test_platform("cuda");

        assert_eq!(
            registry.register_custom_call_target(
                "ryft.test.pending.typed",
                JaxCustomCallTarget::Typed(FfiHandler::new(test_gpu_add_kernel_ffi_handler)),
                platform.clone(),
                1,
                FfiHandlerTraits::COMMAND_BUFFER_COMPATIBLE,
            ),
            Ok(()),
        );

        assert_eq!(CUSTOM_CALL_INVOCATIONS.lock().unwrap().len(), 0);
        let handler: JaxCustomCallHandler = Arc::new(|name, _target, platform, api_version, traits| {
            CUSTOM_CALL_INVOCATIONS.lock().unwrap().push((
                name.to_string(),
                api_version,
                traits.bits(),
                platform.canonical_handler_name().into_owned(),
            ));
            Ok(())
        });
        assert_eq!(registry.register_custom_call_handler(platform.clone(), handler), Ok(()));

        assert_eq!(CUSTOM_CALL_INVOCATIONS.lock().unwrap().len(), 1);
        assert_eq!(
            CUSTOM_CALL_INVOCATIONS.lock().unwrap()[0],
            (
                "ryft.test.pending.typed".to_string(),
                1,
                FfiHandlerTraits::COMMAND_BUFFER_COMPATIBLE.bits(),
                platform.canonical_handler_name().into_owned(),
            ),
        );
    }

    #[test]
    fn test_register_custom_type_handler_flushes_pending_registrations() {
        let registry = JaxFfiRegistry::new();
        let platform = test_platform("cpu");
        assert_eq!(
            registry.register_custom_type(
                "ryft.test.pending.type",
                FfiTypeId::UNKNOWN,
                FfiTypeInfo::new(),
                platform.clone()
            ),
            Ok(()),
        );

        assert_eq!(CUSTOM_TYPE_INVOCATIONS.lock().unwrap().len(), 0);
        let handler: JaxCustomTypeHandler = Arc::new(|type_name, type_id, _type_info, platform| {
            CUSTOM_TYPE_INVOCATIONS.lock().unwrap().push((
                type_name.to_string(),
                type_id.to_i64(),
                platform.canonical_handler_name().into_owned(),
            ));
            Ok(type_id)
        });
        assert_eq!(registry.register_custom_type_handler(platform.clone(), handler), Ok(()));

        assert_eq!(CUSTOM_TYPE_INVOCATIONS.lock().unwrap().len(), 1);
        assert_eq!(
            CUSTOM_TYPE_INVOCATIONS.lock().unwrap()[0],
            (
                "ryft.test.pending.type".to_string(),
                FfiTypeId::UNKNOWN.to_i64(),
                platform.canonical_handler_name().into_owned(),
            ),
        );
    }

    #[test]
    fn test_register_custom_call_target_duplicate_behavior() {
        let registry = JaxFfiRegistry::new();
        let platform = test_platform("cuda");

        let typed_a = JaxCustomCallTarget::Typed(FfiHandler::new(test_gpu_add_kernel_ffi_handler));
        let typed_b = JaxCustomCallTarget::Typed(FfiHandler::new(test_gpu_add_kernel_ffi_handler));

        assert_eq!(
            registry.register_custom_call_target(
                "ryft.test.duplicate.typed",
                typed_a,
                platform.clone(),
                1,
                FfiHandlerTraits::NONE,
            ),
            Ok(()),
        );
        assert_eq!(
            registry.register_custom_call_target(
                "ryft.test.duplicate.typed",
                typed_b,
                platform.clone(),
                1,
                FfiHandlerTraits::NONE,
            ),
            Ok(()),
        );
        assert!(matches!(
            registry.register_custom_call_target(
                "ryft.test.duplicate.typed",
                typed_b,
                platform,
                1,
                FfiHandlerTraits::COMMAND_BUFFER_COMPATIBLE,
            ),
            Err(Error::AlreadyExists { .. })
        ));
    }

    #[test]
    fn test_install_default_pjrt_handlers() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let registry = JaxFfiRegistry::new();
            assert_eq!(registry.install_default_pjrt_handlers(&client), Ok(()));

            let platform = JaxPlatform::from_client(&client).unwrap();
            let registration_result = registry.register_custom_call_target(
                "ryft.test.default.install.typed",
                JaxCustomCallTarget::Typed(FfiHandler::new(test_gpu_add_kernel_ffi_handler)),
                platform,
                1,
                FfiHandlerTraits::NONE,
            );

            if client.ffi_extension().is_ok() {
                assert!(registration_result.is_ok());
            } else {
                assert!(matches!(registration_result, Err(Error::Unimplemented { .. })));
            }
        });
    }

    #[test]
    fn test_typed_ffi_gpu_custom_call_add_kernel() {
        if !is_real_gpu_test_enabled() {
            return;
        }

        test_for_each_platform!(|_plugin, client, platform| {
            if !is_gpu_platform(platform) {
                return;
            }

            let registry = JaxFfiRegistry::new();
            assert_eq!(registry.install_default_pjrt_handlers(&client), Ok(()));

            let target_name = target_name("typed_add", platform);
            let platform = JaxPlatform::from_client(&client).unwrap();
            assert_eq!(
                registry.register_custom_call_target(
                    target_name.as_str(),
                    JaxCustomCallTarget::Typed(FfiHandler::new(test_gpu_add_kernel_ffi_handler)),
                    platform,
                    1,
                    FfiHandlerTraits::NONE,
                ),
                Ok(()),
            );

            test_run_custom_call_program(&client, target_name.as_str());
        });
    }
}
