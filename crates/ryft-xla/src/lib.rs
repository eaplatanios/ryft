pub mod arrays;
pub mod arrays_v0;
pub mod distributed;
pub mod eager;
pub mod errors;
pub mod experimental;
pub mod jit;
pub mod mlir;
pub mod pjrt;
pub mod profile_guided;
pub mod sharding;
pub mod telemetry;
pub mod types;

pub use arrays::{Array, ArrayShard, ShardDescriptor, ShardIndex, ShardLayout, block_until_ready, ready};
pub use arrays_v0::ArrayError;
pub use distributed::DistributedRuntime;
pub use errors::Error;
pub use experimental::domains::{
    XlaAnalysisValue, XlaCompilationAnalysis, XlaDomain, XlaFeedbackDirectedProfile, XlaInputBoundBucketing,
    XlaMemoryAnalysis, XlaOptimizedProgram, XlaOptions,
};
pub use experimental::shard_map::{reshard, sharding_constraint};
pub use jit::{
    CompiledXlaFunction, ExecutableXlaProgram, JittedXlaFunction, StagedXlaFunction, XlaCompileTracer, compile,
    compile_with_captures, compile_with_options, infer_output_types, jitted, jitted_with_options, stage,
    stage_with_captures, try_jitted_with_options,
};
pub use mlir::ToMlir;
pub use pjrt::{FromPjrt, ToPjrt};
pub use profile_guided::{
    AdaptiveProfileGuidedOptions, AdaptiveProfileGuidedState, AdaptiveProfileGuidedStatistics,
    AdaptiveProfileGuidedXlaFunction,
};
pub use telemetry::live_array_count;

#[cfg(test)]
pub(crate) mod tests {
    use std::mem::MaybeUninit;

    use ryft_core::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType};

    pub(crate) fn logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    pub(crate) fn logical_mesh_3x2x1() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 3, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 1, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    pub(crate) fn device_mesh_2x2() -> DeviceMesh {
        DeviceMesh::new(
            logical_mesh_2x2(),
            vec![Device::new(0, 0), Device::new(1, 0), Device::new(2, 1), Device::new(3, 1)],
        )
        .unwrap()
    }

    pub(crate) fn values_to_bytes<V: Copy>(values: &[V]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(size_of_val(values));
        for value in values {
            let value_bytes = unsafe { std::slice::from_raw_parts(value as *const V as *const u8, size_of::<V>()) };
            bytes.extend_from_slice(value_bytes);
        }
        bytes
    }

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

    /// Name of the XLA custom call target registered by [`ensure_add_one_handler_registered`]: an elementwise
    /// `f32` kernel computing `x + increment`, where `increment` is an optional `f64` backend-config attribute
    /// defaulting to `1`, used to exercise the traced custom-call path (including typed attribute decoding) end
    /// to end.
    pub(crate) const ADD_ONE_CUSTOM_CALL_TARGET: &str = "ryft.test.add_one";

    /// Registers the [`ADD_ONE_CUSTOM_CALL_TARGET`] XLA FFI handler with the plugin backing the provided client.
    /// Registration is idempotent and process-global (re-registering the same target name is rejected by the XLA
    /// runtime), so the outcome is cached in a [`OnceLock`](std::sync::OnceLock).
    pub(crate) fn ensure_add_one_handler_registered(client: &ryft_pjrt::Client<'_>) -> Result<(), ryft_pjrt::Error> {
        use std::sync::OnceLock;

        use ryft_pjrt::extensions::ffi::{FfiHandler, FfiHandlerTraits, XLA_FFI_Handler};

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

    /// XLA FFI handler for [`ADD_ONE_CUSTOM_CALL_TARGET`] custom calls: reads one `f32` input buffer and writes
    /// `input + increment` elementwise into the single `f32` output buffer.
    unsafe extern "C" fn add_one_handler(
        call_frame: *mut ryft_pjrt::extensions::ffi::XLA_FFI_CallFrame,
    ) -> *mut ryft_pjrt::extensions::ffi::XLA_FFI_Error {
        use ryft_pjrt::extensions::ffi::{FfiCallFrame, FfiExecutionStage, FfiTypeId};

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
        use ryft_pjrt::extensions::ffi::{FfiAttribute, FfiBufferType, FfiError, FfiInput, FfiOutput, FfiScalar};

        let mut increment = 1.0f32;
        for attribute in call_frame.attributes() {
            let (name, attribute) = attribute?;
            if name == "increment" {
                let FfiAttribute::Scalar { scalar: FfiScalar::F64(value) } = attribute else {
                    return Err(FfiError::invalid_argument(format!(
                        "expected the 'increment' attribute of the '{ADD_ONE_CUSTOM_CALL_TARGET}' custom call to \
                         be an f64 scalar"
                    )));
                };
                increment = value as f32;
            }
        }
        let mut inputs = call_frame.inputs();
        let Some(Ok(FfiInput::Buffer { buffer: input })) = inputs.next() else {
            return Err(FfiError::invalid_argument(format!(
                "expected the '{ADD_ONE_CUSTOM_CALL_TARGET}' custom call to have one input buffer"
            )));
        };
        let mut outputs = call_frame.outputs();
        let Some(Ok(FfiOutput::Buffer { buffer: output })) = outputs.next() else {
            return Err(FfiError::invalid_argument(format!(
                "expected the '{ADD_ONE_CUSTOM_CALL_TARGET}' custom call to have one output buffer"
            )));
        };
        if input.element_type() != FfiBufferType::F32
            || output.element_type() != FfiBufferType::F32
            || input.dimensions() != output.dimensions()
        {
            return Err(FfiError::invalid_argument(format!(
                "expected the '{ADD_ONE_CUSTOM_CALL_TARGET}' custom call input and output to be f32 buffers with \
                 matching shapes"
            )));
        }
        let count = input.dimensions().iter().map(|&dimension| dimension.max(0) as usize).product::<usize>();
        // SAFETY: Both data pointers are provided by the XLA runtime, are valid for the duration of the handler
        // invocation, and (per the element type and shape equality checks above) are backed by allocations of at
        // least `count` `f32` elements. The elementwise loop is also valid when the output aliases the input because
        // it reads each element before overwriting that same element.
        unsafe {
            let source = input.data() as *const f32;
            let destination = output.data() as *mut f32;
            if count > 0 && (source.is_null() || destination.is_null()) {
                return Err(FfiError::internal(format!(
                    "encountered null data pointer in the '{ADD_ONE_CUSTOM_CALL_TARGET}' custom call"
                )));
            }
            for index in 0..count {
                *destination.add(index) = *source.add(index) + increment;
            }
        }
        Ok(())
    }
}
