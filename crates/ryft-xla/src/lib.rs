//! XLA backend of `ryft`, which stages `ryft-core` programs, lowers them to StableHLO and Shardy MLIR, and compiles
//! and executes them through PJRT.
//!
//! A [`Session`] executes on one PJRT client, which a [`SessionBuilder`] creates for a [`Platform`] or borrows from the
//! caller. It is a cheap handle to the compilation cache and runtime extensions that it creates for that client, and
//! it implements [`ryft_core::Domain`] for the XLA backend. Device data lives in [`Array`]s,
//! and the [`Value`]s of a session (i.e., arrays, first-class [`Dimension`]s, and array [`Reference`]s) carry the
//! session that they belong to, so eager operations and transforms recover their execution context from their inputs.
//!
//! The entry points of the compilation lifecycle are [`stage`], which stages a function into a program without
//! compiling it, [`compile`], which compiles one explicit abstract signature of a function, and [`jit`], which returns
//! a dispatcher that selects and retains specializations of a function for its runtime inputs. [`Options`] configures
//! them, and the [`compilation`] module documents the lifecycle in detail. [`Operation`] is the backend's operation
//! family, the [`lowering`] module lowers programs of that family to MLIR, and the [`execution`] module executes
//! compiled programs.
//!
//! The [`kernels`] module embeds natively compiled kernels into this lifecycle, and [`DistributedRuntime`] sets up
//! multi-process jobs. Errors of every public entry point are reported as [`Error`].

pub mod arrays;
pub mod compilation;
pub mod devices;
pub mod distributed;
pub mod errors;
pub mod execution;
pub mod kernels;
pub mod lowering;
pub mod mlir;
pub mod operations;
pub mod pjrt;
pub mod sessions;
pub mod sharding;
pub mod tracing;
pub mod types;
pub mod values;

pub use arrays::{
    AliasingParameter, Array, ArrayError, ArrayPlacement, ArrayPlacementTarget, ArrayShard, ArrayShardStorage,
    ArrayStatistics, HostOrArray, LiveArray, Placeable, PlaceableParameter, PlacementOptions, RetainedArray,
    ShardDescriptor, ShardIndex, ShardLayout, block_until_ready, ready,
};
pub use compilation::{
    AdaptiveProfileGuidedFunction, AdaptiveProfileGuidedOptions, AdaptiveProfileGuidedState,
    AdaptiveProfileGuidedStatistics, AnalysisValue, CompilationAnalysis, CompilationError, CompilationKey,
    CompileTracer, CompiledFunction, CompiledFunctionDispatcher, CompiledProgram, DispatchKey, ExecutableFunction,
    FeedbackDirectedProfile, InputBoundBucketing, LoweredProgram, MemoryAnalysis, OptimizedProgram, Options,
    StagedFunction, StatefulCompileTracer, StatefulCompiledFunction, StatefulCompiledFunctionDispatcher, StatefulValue,
    compile, compile_statefully, compile_statefully_with_captures, compile_with_captures, compile_with_options,
    infer_output_types, jit, jit_statefully, jit_statefully_with_options, jit_with_options, stage, stage_with_captures,
    with_input_shardings,
};
pub use distributed::DistributedRuntime;
pub use errors::Error;
pub use execution::{ExecutionArguments, ExecutionError};
#[cfg(feature = "cutile")]
pub use kernels::CuTileEmbedding;
#[cfg(feature = "mosaic-gpu")]
pub use kernels::MosaicGpuEmbedding;
#[cfg(feature = "rocm")]
pub use kernels::RocmKernelEmbedding;
#[cfg(feature = "triton")]
pub use kernels::TritonEmbedding;
pub use kernels::{
    CUDA_KERNEL_CUSTOM_CALL_TARGET, CompiledKernel, CudaKernelBufferBinding, CudaKernelEmbedding,
    CudaKernelParameterBinding, DistributedKernel, DistributedKernelError, DistributedKernelOptions, KernelAotBundle,
    KernelAotError, KernelCompilationReport, KernelCompilerBinding, KernelDeviceFacts, KernelEmbeddingError,
    KernelExecutionFacts, KernelExtension, KernelOperation, KernelOutputEmbedding, KernelTarget, KernelTuner,
    KernelTuningBudget, KernelTuningError, KernelTuningRequest, KernelTuningResult, KernelTuningRunner, LoadedKernel,
    ROCM_KERNEL_CUSTOM_CALL_TARGET, stage_kernel, stage_kernel_with_fallback, stage_kernel_with_jvp,
    stage_kernel_with_rules, stage_kernel_with_vjp,
};
pub use lowering::{
    LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError, MemberLowerableOperation,
    lower_projected_operation, to_mlir_module_for_program,
};
pub use mlir::ToMlir;
pub use operations::{
    FlatProgram, JIT_CALL_OPERATION_NAME, JitCallOperation, Operation, PRINT_CUSTOM_CALL_TARGET, PRINT_LABEL_ATTRIBUTE,
    Program, ProgramBuilder, RaggedDotLoweringStrategy, ensure_print_handler_registered, transpose_primal_jit_call,
    with_captured_prints,
};
pub use pjrt::{FromPjrt, ToPjrt};
pub use sessions::{
    CustomCallOperationExtension, ExecutionResources, MAX_RUNTIME_REQUIREMENTS_BYTES, Platform,
    RUNTIME_REQUIREMENTS_ATTRIBUTE, RuntimeError, RuntimeExtension, RuntimeExtensions, RuntimeProvider,
    RuntimeRequirement, Session, SessionBuilder, SessionError, Target, TargetDevice, ThroughSession, Tracer,
    normalize_runtime_requirements,
};
pub use sharding::{OpShardingExtension, SHARDY_MESH_SYMBOL_NAME};
pub use tracing::{ArrayTracer, TracedProgram, trace, trace_with_context};
pub use values::{ArrayConstant, Constant, Dimension, Reference, Type, Value};

// TODO(eaplatanios): Review this module.

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::RefCell;
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::mem::MaybeUninit;
    use std::sync::{Mutex, Once, OnceLock, PoisonError};

    use ryft_core::{
        ArrayType, DataType, Device, DeviceMesh, Dimension, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding,
        StaticShape,
    };
    use ryft_pjrt::extensions::ffi::{
        FfiAttribute, FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits, FfiInput,
        FfiOutput, FfiScalar, FfiTypeId, XLA_FFI_Handler,
    };
    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions};
    use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};

    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform};

    use crate::compilation::options::resolve_shardy_debug_options;
    use crate::{Array, FromPjrt, Session};

    #[cfg(any(feature = "cuda-13", feature = "mps"))]
    use crate::Platform;

    /// Returns a CPU client with `device_count` addressable devices, for tests that need a fixed device count or that
    /// exercise CPU-only behavior. Tests that only need some single device use [`execution_client`] instead.
    ///
    /// Clients live for the rest of the process and are pooled. A thread keeps the client of each device count that
    /// it requests until it exits, and so repeated calls in one test return the same client while concurrently running
    /// tests never share one. That matters because the CPU plugin executes a small program inline only when its
    /// inputs are already available, and transfers that other tests queue on a shared client would delay them.
    pub(crate) fn cpu_client(device_count: usize) -> &'static Client<'static> {
        /// Clients that no thread currently holds, with their device counts.
        static IDLE_CLIENTS: Mutex<Vec<(usize, &'static Client<'static>)>> = Mutex::new(Vec::new());

        /// Clients that the current thread holds, which return to [`IDLE_CLIENTS`] when the thread exits.
        struct HeldClients(Vec<(usize, &'static Client<'static>)>);

        impl Drop for HeldClients {
            fn drop(&mut self) {
                IDLE_CLIENTS.lock().unwrap_or_else(PoisonError::into_inner).append(&mut self.0);
            }
        }

        thread_local! {
            static HELD_CLIENTS: RefCell<HeldClients> = const { RefCell::new(HeldClients(Vec::new())) };
        }

        HELD_CLIENTS.with(|held| {
            let held = &mut held.borrow_mut().0;
            if let Some((_, client)) = held.iter().find(|(count, _)| *count == device_count) {
                return *client;
            }
            let idle = {
                let mut idle = IDLE_CLIENTS.lock().unwrap_or_else(PoisonError::into_inner);
                idle.iter().position(|(count, _)| *count == device_count).map(|index| idle.swap_remove(index).1)
            };
            let client = idle.unwrap_or_else(|| {
                let options =
                    ClientOptions::CPU(CpuClientOptions { device_count: Some(device_count), ..Default::default() });
                Box::leak(Box::new(load_cpu_plugin().unwrap().client(options).unwrap()))
            });
            held.push((device_count, client));
            client
        })
    }

    /// Returns a process-wide CPU client with `device_count` addressable devices that is distinct from every client
    /// that [`cpu_client`] and [`execution_client`] return, for tests of behavior across two clients. All tests that
    /// request the same device count share one such client.
    pub(crate) fn other_cpu_client(device_count: usize) -> &'static Client<'static> {
        static CLIENTS: Mutex<Vec<(usize, &'static Client<'static>)>> = Mutex::new(Vec::new());
        let mut clients = CLIENTS.lock().unwrap();
        if let Some((_, client)) = clients.iter().find(|(count, _)| *count == device_count) {
            return client;
        }
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(device_count), ..Default::default() });
        let client: &'static Client<'static> = Box::leak(Box::new(load_cpu_plugin().unwrap().client(options).unwrap()));
        clients.push((device_count, client));
        client
    }

    /// Returns the process-wide CUDA client for tests that explicitly exercise the CUDA plugin.
    #[cfg(feature = "cuda-13")]
    pub(crate) fn cuda_client() -> &'static Client<'static> {
        Platform::Cuda
            .client(Some(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            })))
            .unwrap()
    }

    /// Returns the process-wide MPS client for tests that explicitly exercise the MPS plugin.
    #[cfg(feature = "mps")]
    pub(crate) fn mps_client() -> &'static Client<'static> {
        Platform::Mps.client(None).unwrap()
    }

    /// Returns the execution client selected by `RYFT_TEST_EXECUTION_PLATFORM`, which is a [`cpu_client`] with one
    /// device or the process-wide CUDA client.
    ///
    /// The default is `cpu`; `cuda-13` explicitly requires the corresponding feature, plugin, and a CUDA device.
    /// Selection errors fail the test instead of silently falling back to CPU or skipping GPU coverage.
    pub(crate) fn execution_client() -> &'static Client<'static> {
        let platform = std::env::var("RYFT_TEST_EXECUTION_PLATFORM").unwrap_or_else(|_| "cpu".to_string());
        let client = match platform.as_str() {
            "cpu" => cpu_client(1),
            #[cfg(feature = "cuda-13")]
            "cuda-13" => cuda_client(),
            _ => panic!("unsupported execution platform `{platform}`; CUDA testing requires `--features cuda-13`"),
        };
        let expected = if platform == "cpu" { "cpu" } else { "cuda" };
        assert!(client.platform_name().unwrap().eq_ignore_ascii_case(expected));
        let devices = client.addressable_devices().unwrap();
        assert!(!devices.is_empty(), "execution platform `{platform}` has no addressable devices");
        static REPORTED_PLATFORM: Once = Once::new();
        REPORTED_PLATFORM.call_once(|| {
            eprintln!("execution test platform: {platform}; addressable devices: {devices:?}");
        });
        client
    }

    /// Builds options for tests that call [`Client::compile`] directly, preserving the client's Shardy defaults.
    /// `partition_count` selects Shardy SPMD partitioning when supplied. `None` selects single-device compilation.
    pub(crate) fn test_compilation_options(
        client: &'static Client<'static>,
        partition_count: Option<usize>,
    ) -> CompilationOptions {
        let mut options = CompilationOptions {
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: partition_count.unwrap_or(1) as i64,
                use_spmd_partitioning: partition_count.is_some(),
                use_shardy_partitioner: partition_count.is_some(),
                ..Default::default()
            }),
            ..Default::default()
        };
        resolve_shardy_debug_options(client, &mut options);
        options
    }

    /// Returns the `f64` vector type with four elements that staged program fixtures use for their values.
    pub(crate) fn vector_type() -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)]))
    }

    /// Returns a mesh with one auto axis `x` over the four addressable CPU devices of `client`.
    pub(crate) fn four_device_mesh_x(client: &ryft_pjrt::Client<'_>) -> DeviceMesh {
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
            .unwrap()
    }

    /// Returns a logical mesh with the auto axes `x` and `y`, each of size two.
    pub(crate) fn logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    /// Returns a logical mesh with the auto axes `x`, `y`, and `z` of sizes three, two, and one.
    pub(crate) fn logical_mesh_3x2x1() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 3, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 1, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    /// Returns [`logical_mesh_2x2`] over devices 0 and 1 of process 0 and devices 2 and 3 of process 1.
    pub(crate) fn device_mesh_2x2() -> DeviceMesh {
        DeviceMesh::new(
            logical_mesh_2x2(),
            vec![Device::new(0, 0), Device::new(1, 0), Device::new(2, 1), Device::new(3, 1)],
        )
        .unwrap()
    }

    /// Uploads `bytes` as an [`Array`] of element type `data_type` and shape `dimensions` that is replicated over
    /// `mesh`. Read it back with [`Array::copy_to_host`] and [`values_from_bytes`].
    pub(crate) fn replicated_array(
        session: &Session,
        mesh: &DeviceMesh,
        data_type: DataType,
        dimensions: &[usize],
        bytes: &[u8],
    ) -> Array {
        let r#type = ArrayType::new(data_type, StaticShape::new(dimensions.to_vec()).into())
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), dimensions.len()))
            .unwrap();
        Array::from_host_buffer(session, r#type, mesh.clone(), None, bytes).unwrap()
    }

    /// Returns the native-endian bytes of `values`, in order.
    pub(crate) fn values_to_bytes<V: Copy>(values: &[V]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(size_of_val(values));
        for value in values {
            let value_bytes = unsafe { std::slice::from_raw_parts(value as *const V as *const u8, size_of::<V>()) };
            bytes.extend_from_slice(value_bytes);
        }
        bytes
    }

    /// Decodes `bytes` as consecutive native-endian values of type `V`. Panics if the length of `bytes` is not a
    /// multiple of the size of `V`.
    pub(crate) fn values_from_bytes<V: Copy>(bytes: &[u8]) -> Vec<V> {
        assert_eq!(bytes.len() % size_of::<V>(), 0);
        bytes
            .chunks_exact(size_of::<V>())
            .map(|chunk| {
                let mut value = MaybeUninit::<V>::uninit();
                unsafe {
                    std::ptr::copy_nonoverlapping(chunk.as_ptr(), value.as_mut_ptr() as *mut u8, size_of::<V>());
                    value.assume_init()
                }
            })
            .collect()
    }

    /// Compiles only when `T` is both [`Send`] and [`Sync`], so that tests can pin these auto traits of a type.
    pub(crate) fn assert_send_sync<T: Send + Sync>() {}

    /// Returns the hash of `value` under the standard library's default hasher, so that tests can check that equal
    /// values hash equally. `value` may be unsized (e.g., a trait object whose [`Hash`] implementation must agree with
    /// that of an owned type that borrows as it).
    pub(crate) fn hash_of<T: ?Sized + Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Name of the XLA custom call target registered by [`ensure_add_one_handler_registered`]. It names an elementwise
    /// `f32` kernel computing `x + increment`, where `increment` is an optional `f64` backend config attribute
    /// defaulting to `1`, and is used to exercise the traced custom call path, including typed attribute decoding, end
    /// to end.
    pub(crate) const ADD_ONE_CUSTOM_CALL_TARGET: &str = "ryft.test.add_one";

    /// Registers the [`ADD_ONE_CUSTOM_CALL_TARGET`] XLA FFI handler with the plugin backing the provided client.
    /// Registration is idempotent and process-global, because the XLA runtime rejects registering the same target name
    /// again, so the outcome is cached in a [`OnceLock`](std::sync::OnceLock).
    pub(crate) fn ensure_add_one_handler_registered(client: &ryft_pjrt::Client<'_>) -> Result<(), ryft_pjrt::Error> {
        static ADD_ONE_HANDLER_REGISTRATION: OnceLock<Result<(), ryft_pjrt::Error>> = OnceLock::new();
        ADD_ONE_HANDLER_REGISTRATION
            .get_or_init(|| {
                let platform_name = client.platform_name()?.into_owned();
                client.register_ffi_handler(
                    ADD_ONE_CUSTOM_CALL_TARGET,
                    platform_name,
                    FfiHandler::from(add_one_handler as XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                )
            })
            .clone()
    }

    /// XLA FFI handler for [`ADD_ONE_CUSTOM_CALL_TARGET`] custom calls, which reads one `f32` input buffer and writes
    /// `input + increment` elementwise into the single `f32` output buffer. Tests may pass a second `i32` extent
    /// buffer, which the handler validates against the leading batch and trailing packed axes.
    unsafe extern "C" fn add_one_handler(
        call_frame: *mut ryft_pjrt::extensions::ffi::XLA_FFI_CallFrame,
    ) -> *mut ryft_pjrt::extensions::ffi::XLA_FFI_Error {
        // SAFETY: The XLA runtime passes a call frame that is valid for the duration of this invocation, and all
        // further unsafe access to it is localized in the safe `FfiCallFrame` wrapper and
        // `handle_add_one_call_frame`.
        unsafe {
            match FfiCallFrame::from_c_api(call_frame) {
                Err(_) => std::ptr::null_mut(),
                Ok(call_frame) if call_frame.register_metadata(FfiTypeId::default()) => std::ptr::null_mut(),
                Ok(call_frame) if call_frame.stage() != FfiExecutionStage::Execution => std::ptr::null_mut(),
                Ok(call_frame) => match call_frame.api() {
                    Err(_) => std::ptr::null_mut(),
                    Ok(api) => match handle_add_one_call_frame(&call_frame) {
                        Ok(()) => std::ptr::null_mut(),
                        Err(error) => error.to_c_api(api),
                    },
                },
            }
        }
    }

    /// Decodes an [`ADD_ONE_CUSTOM_CALL_TARGET`] call frame and fills its output buffer with `input + increment`,
    /// where `increment` is decoded from the optional `f64` backend-config attribute of the same name and
    /// defaults to `1`.
    fn handle_add_one_call_frame(
        call_frame: &ryft_pjrt::extensions::ffi::FfiCallFrame<'_>,
    ) -> Result<(), ryft_pjrt::extensions::ffi::FfiError> {
        let mut increment = 1.0f32;
        for attribute in call_frame.attributes() {
            let (name, attribute) = attribute?;
            if name == "increment" {
                let FfiAttribute::Scalar { scalar: FfiScalar::F64(value) } = attribute else {
                    return Err(FfiError::invalid_argument(format!(
                        "expected the `increment` attribute of the `{ADD_ONE_CUSTOM_CALL_TARGET}` custom call to \
                         be an `f64` scalar"
                    )));
                };
                increment = value as f32;
            }
        }
        let mut inputs = call_frame.inputs();
        let Some(Ok(FfiInput::Buffer { buffer: input })) = inputs.next() else {
            return Err(FfiError::invalid_argument(format!(
                "expected the `{ADD_ONE_CUSTOM_CALL_TARGET}` custom call to have an input buffer"
            )));
        };
        let mut extents = None;
        for input in inputs {
            let FfiInput::Buffer { buffer } = input?;
            // Side-effecting calls carry a hidden ordered-I/O token after their explicit operands. It is part of the
            // FFI frame but not part of this handler's public array contract.
            if buffer.element_type() == FfiBufferType::Token {
                continue;
            }
            if extents.replace(buffer).is_some() {
                return Err(FfiError::invalid_argument(format!(
                    "expected the `{ADD_ONE_CUSTOM_CALL_TARGET}` custom call to have at most one extent input"
                )));
            }
        }
        let mut outputs = call_frame.outputs();
        let Some(Ok(FfiOutput::Buffer { buffer: output })) = outputs.next() else {
            return Err(FfiError::invalid_argument(format!(
                "expected the `{ADD_ONE_CUSTOM_CALL_TARGET}` custom call to have one output buffer"
            )));
        };
        if input.element_type() != FfiBufferType::F32
            || output.element_type() != FfiBufferType::F32
            || input.dimensions() != output.dimensions()
        {
            return Err(FfiError::invalid_argument(format!(
                "expected the `{ADD_ONE_CUSTOM_CALL_TARGET}` custom call input and output to be `f32` buffers with \
                 matching shapes"
            )));
        }
        let count = input.dimensions().iter().map(|&dimension| dimension.max(0) as usize).product::<usize>();
        if let Some(extents) = extents {
            let dimensions = input.dimensions();
            if extents.element_type() != FfiBufferType::I32
                || dimensions.len() != 2
                || extents.dimensions() != [dimensions[0]]
            {
                return Err(FfiError::invalid_argument(format!(
                    "expected the optional extent input of the `{ADD_ONE_CUSTOM_CALL_TARGET}` custom call to be an \
                     `i32` vector matching the data buffer's leading axis"
                )));
            }
            let extent_count = dimensions[0].max(0) as usize;
            let packed_bound = dimensions[1].max(0) as i32;
            // SAFETY: The runtime-owned extent buffer is valid for the duration of this FFI handler.
            let extent_data = unsafe { extents.data() } as *const i32;
            if extent_count > 0 && extent_data.is_null() {
                return Err(FfiError::internal(format!(
                    "encountered null extent pointer in the `{ADD_ONE_CUSTOM_CALL_TARGET}` custom call"
                )));
            }
            // SAFETY: The XLA runtime owns the optional extent buffer for the duration of the call and the checks
            // above establish an `i32[extent_count]` allocation.
            for index in 0..extent_count {
                let extent = unsafe { *extent_data.add(index) };
                if !(0..=packed_bound).contains(&extent) {
                    return Err(FfiError::invalid_argument(format!(
                        "extent {extent} at batch item {index} lies outside the packed bound {packed_bound}"
                    )));
                }
            }
        }
        // SAFETY: Both data pointers are provided by the XLA runtime, are valid for the duration of the handler
        // invocation, and, per the element type and shape equality checks above, are backed by allocations of at
        // least `count` `f32` elements. The elementwise loop is also valid when the output aliases the input because
        // it reads each element before overwriting that same element.
        unsafe {
            let source = input.data() as *const f32;
            let destination = output.data() as *mut f32;
            if count > 0 && (source.is_null() || destination.is_null()) {
                return Err(FfiError::internal(format!(
                    "encountered null data pointer in the `{ADD_ONE_CUSTOM_CALL_TARGET}` custom call"
                )));
            }
            for index in 0..count {
                *destination.add(index) = *source.add(index) + increment;
            }
        }
        Ok(())
    }
}
