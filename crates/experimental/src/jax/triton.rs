use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::extensions::triton::TritonKernel;
use crate::{Client, Error};

use super::JaxPlatform;

/// Request payload for JAX-style Triton compilation handlers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JaxTritonCompileRequest {
    pub module: Vec<u8>,
    pub arch_name: String,
    pub num_warps: i32,
    pub num_ctas: i32,
    pub num_stages: i32,
}

/// Handler signature used by [`JaxTritonRegistry`].
pub type JaxTritonCompilationHandler =
    Arc<dyn Fn(JaxTritonCompileRequest, JaxPlatform) -> Result<TritonKernel, Error> + Send + Sync + 'static>;

#[derive(Default)]
struct JaxTritonRegistryState {
    handlers: HashMap<String, JaxTritonCompilationHandler>,
}

/// Registry that emulates JAX-style Triton compilation handler dispatch.
#[derive(Default)]
pub struct JaxTritonRegistry {
    state: Mutex<JaxTritonRegistryState>,
}

impl JaxTritonRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a platform-specific compilation handler. The first handler wins.
    pub fn register_compilation_handler<P: Into<JaxPlatform>>(
        &self,
        platform: P,
        handler: JaxTritonCompilationHandler,
    ) -> Result<(), Error> {
        let platform = platform.into();
        let platform_key = platform.canonical_handler_name().into_owned();
        let mut state = self.state.lock().unwrap();
        if state.handlers.contains_key(platform_key.as_str()) {
            return Ok(());
        }
        state.handlers.insert(platform_key, handler);
        Ok(())
    }

    /// Compiles a Triton module by dispatching to the registered handler for `platform`.
    pub fn compile<P: Into<JaxPlatform>>(
        &self,
        platform: P,
        request: JaxTritonCompileRequest,
    ) -> Result<TritonKernel, Error> {
        let platform = platform.into();
        let platform_key = platform.canonical_handler_name().into_owned();
        let handler = {
            let state = self.state.lock().unwrap();
            state.handlers.get(platform_key.as_str()).cloned()
        };

        let Some(handler) = handler else {
            return Err(Error::not_found(format!(
                "no Triton compilation handler is registered for platform '{platform_key}'"
            )));
        };
        handler(request, platform)
    }

    /// Installs default PJRT-backed Triton compilation handlers for the platform associated with `client`.
    pub fn install_default_pjrt_handlers(&self, client: &Client<'_>) -> Result<(), Error> {
        let platform = JaxPlatform::from_client(client)?;
        let api = client.api();
        self.register_compilation_handler(
            platform,
            Arc::new(move |request, _platform| {
                api.compile_triton_kernel(
                    request.module.as_slice(),
                    request.arch_name.as_str(),
                    request.num_warps,
                    request.num_ctas,
                    request.num_stages,
                )
            }),
        )
    }
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

    use crate::extensions::ffi::{FfiError, FfiHandler, FfiHandlerTraits, ffi as xla_ffi};
    use crate::jax::ffi::{JaxCustomCallTarget, JaxFfiRegistry};
    use crate::jax::gpu_runtime;
    use crate::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use crate::tests::{TestPlatform, test_for_each_platform};
    use crate::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program};

    use super::*;

    const TEST_VALUE_LHS: i32 = 7;
    const TEST_VALUE_RHS: i32 = 35;
    const EXPECTED_SUM: i32 = 42;

    static TRITON_TEST_ARTIFACT: LazyLock<Mutex<Vec<u8>>> = LazyLock::new(|| Mutex::new(Vec::new()));

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
        format!("ryft.test.jax.triton.{prefix}.{platform_suffix}")
    }

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

    unsafe extern "C" fn test_triton_add_kernel_ffi_handler(
        call_frame: *mut xla_ffi::XLA_FFI_CallFrame,
    ) -> *mut xla_ffi::XLA_FFI_Error {
        if call_frame.is_null() {
            return std::ptr::null_mut();
        }
        if TRITON_TEST_ARTIFACT.lock().unwrap().is_empty() {
            return make_xla_error(call_frame, "expected non-empty Triton compilation artifact before execution");
        }

        let raw_call_frame = call_frame as *const xla_sys::XLA_FFI_CallFrame;
        let raw_call_frame_ref = unsafe { &*raw_call_frame };
        let stream = unsafe { ffi_stream(raw_call_frame) };
        let Some(stream) = stream else {
            return make_xla_error(call_frame, "failed to get stream from XLA FFI call frame");
        };

        let lhs_buffer = unsafe { ffi_buffer_data(raw_call_frame_ref.args.types, raw_call_frame_ref.args.args, 0) };
        let rhs_buffer = unsafe { ffi_buffer_data(raw_call_frame_ref.args.types, raw_call_frame_ref.args.args, 1) };
        let out_buffer =
            unsafe { ffi_result_buffer_data(raw_call_frame_ref.rets.types, raw_call_frame_ref.rets.rets, 0) };
        let (Some(lhs_buffer), Some(rhs_buffer), Some(out_buffer)) = (lhs_buffer, rhs_buffer, out_buffer) else {
            return make_xla_error(call_frame, "failed to decode scalar i32 buffers for Triton add-kernel test");
        };

        match gpu_runtime::launch_add_i32(stream, lhs_buffer, rhs_buffer, out_buffer) {
            Ok(()) => std::ptr::null_mut(),
            Err(error) => {
                make_xla_error(call_frame, format!("failed to execute Triton add-kernel test: {error}").as_str())
            }
        }
    }

    #[test]
    fn test_register_compilation_handler_first_wins() {
        let registry = JaxTritonRegistry::new();
        let counter = Arc::new(Mutex::new(0usize));

        let handler_a_counter = counter.clone();
        let handler_a: JaxTritonCompilationHandler = Arc::new(move |_request, _platform| {
            *handler_a_counter.lock().unwrap() += 1;
            Ok(TritonKernel {
                assembly: vec![1, 2, 3],
                shared_memory_bytes: 0,
                thread_block_cluster_dimensions: [1, 1, 1],
            })
        });

        let handler_b: JaxTritonCompilationHandler = Arc::new(move |_request, _platform| {
            Ok(TritonKernel {
                assembly: vec![9, 9, 9],
                shared_memory_bytes: 0,
                thread_block_cluster_dimensions: [1, 1, 1],
            })
        });

        assert_eq!(registry.register_compilation_handler("cuda", handler_a), Ok(()));
        assert_eq!(registry.register_compilation_handler("cuda", handler_b), Ok(()));

        let compiled = registry
            .compile(
                "cuda",
                JaxTritonCompileRequest {
                    module: Vec::new(),
                    arch_name: "sm_90".to_string(),
                    num_warps: 1,
                    num_ctas: 1,
                    num_stages: 1,
                },
            )
            .unwrap();

        assert_eq!(compiled.assembly, vec![1, 2, 3]);
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[test]
    fn test_compile_without_handler_returns_not_found() {
        let registry = JaxTritonRegistry::new();
        let result = registry.compile(
            "cuda",
            JaxTritonCompileRequest {
                module: Vec::new(),
                arch_name: "sm_90".to_string(),
                num_warps: 1,
                num_ctas: 1,
                num_stages: 1,
            },
        );
        assert!(matches!(result, Err(Error::NotFound { .. })));
    }

    #[test]
    fn test_install_default_pjrt_handlers() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let registry = JaxTritonRegistry::new();
            assert_eq!(registry.install_default_pjrt_handlers(&client), Ok(()));

            let platform = JaxPlatform::from_client(&client).unwrap();
            let result = registry.compile(
                platform,
                JaxTritonCompileRequest {
                    module: Vec::new(),
                    arch_name: "sm_90".to_string(),
                    num_warps: 1,
                    num_ctas: 1,
                    num_stages: 1,
                },
            );

            if client.triton_extension().is_ok() {
                assert!(
                    result.is_ok() || matches!(result, Err(Error::InvalidArgument { .. } | Error::Internal { .. }))
                );
            } else {
                assert!(matches!(result, Err(Error::Unimplemented { .. })));
            }
        });
    }

    #[test]
    fn test_triton_registry_gpu_add_kernel_flow() {
        if !is_real_gpu_test_enabled() {
            return;
        }

        test_for_each_platform!(|_plugin, client, platform| {
            if !is_gpu_platform(platform) {
                return;
            }

            let jax_platform = JaxPlatform::from_client(&client).unwrap();
            let triton_registry = JaxTritonRegistry::new();
            assert_eq!(
                triton_registry.register_compilation_handler(
                    jax_platform.clone(),
                    Arc::new(|_request, platform| {
                        let assembly = gpu_runtime::add_i32_artifact(&platform)
                            .map_err(|error| Error::internal(format!("failed to build add_i32 artifact: {error}")))?;
                        Ok(TritonKernel {
                            assembly,
                            shared_memory_bytes: 0,
                            thread_block_cluster_dimensions: [1, 1, 1],
                        })
                    }),
                ),
                Ok(()),
            );

            let compilation = triton_registry
                .compile(
                    jax_platform.clone(),
                    JaxTritonCompileRequest {
                        module: b"fake_triton_module_for_test".to_vec(),
                        arch_name: "auto".to_string(),
                        num_warps: 1,
                        num_ctas: 1,
                        num_stages: 1,
                    },
                )
                .unwrap();
            assert!(!compilation.assembly.is_empty());
            *TRITON_TEST_ARTIFACT.lock().unwrap() = compilation.assembly;

            let ffi_registry = JaxFfiRegistry::new();
            assert_eq!(ffi_registry.install_default_pjrt_handlers(&client), Ok(()));

            let target_name = target_name("compiled_add", platform);
            assert_eq!(
                ffi_registry.register_custom_call_target(
                    target_name.as_str(),
                    JaxCustomCallTarget::Typed(FfiHandler::new(test_triton_add_kernel_ffi_handler)),
                    jax_platform,
                    1,
                    FfiHandlerTraits::NONE,
                ),
                Ok(()),
            );

            test_run_custom_call_program(&client, target_name.as_str());
            TRITON_TEST_ARTIFACT.lock().unwrap().clear();
        });
    }
}
