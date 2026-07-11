use std::sync::{Mutex, OnceLock, PoisonError};

use ryft_pjrt::extensions::ffi::{
    FfiAttribute, FfiBuffer, FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits,
    FfiInput, FfiOutput, FfiTypeId, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Handler,
};
use ryft_pjrt::{Client, Error};

/// Name of the XLA custom call target that implements the host-callback side of `print` lowering.
///
/// A lowered [`PrintOperation`](ryft_core::operations::debugging::PrintOperation) becomes a
/// `stablehlo.custom_call` with the following calling convention, which the FFI handler registered by
/// [`ensure_print_handler_registered`] decodes:
///
///   - `call_target_name` is [`PRINT_CUSTOM_CALL_TARGET`] (i.e., `"ryft.print"`).
///   - `api_version` is `4` (the typed FFI API version), so that the `backend_config` dictionary entries arrive
///     in the XLA FFI call frame as decoded [`FfiAttribute`]s.
///   - `has_side_effect` is `true`, so that the XLA compiler never elides or reorders the call.
///   - `backend_config` is a dictionary attribute containing a single [`PRINT_LABEL_ATTRIBUTE`] (i.e., `"label"`)
///     entry whose value is the string label to print.
///   - The operands are `[value, token]`: the dense array value to print followed by a `!stablehlo.token` that
///     orders this print relative to other side-effecting operations.
///   - The single result is a `!stablehlo.token` that continues the effect-ordering token chain.
///
/// The handler prints the first non-token operand buffer and ignores token operands and results, so it also
/// tolerates the degenerate `[value] -> [token]` shape. If a future variant adds a non-token result (e.g., a
/// value passthrough), the handler copies the printed operand into any result buffer with a matching element
/// type and shape.
pub const PRINT_CUSTOM_CALL_TARGET: &str = "ryft.print";

/// Name of the `backend_config` dictionary entry that carries the label of a [`PRINT_CUSTOM_CALL_TARGET`]
/// custom call. Refer to the documentation of [`PRINT_CUSTOM_CALL_TARGET`] for the full calling convention.
pub const PRINT_LABEL_ATTRIBUTE: &str = "label";

/// Maximum number of raw bytes included in the fallback hexadecimal rendering of printed values whose element
/// type has no dedicated numeric rendering. Longer buffers are truncated with a trailing ellipsis.
const MAX_RENDERED_FALLBACK_BYTES: usize = 128;

/// Serializes [`with_captured_prints`] sessions across threads. Because the print sink is process-global,
/// two concurrently capturing tests would otherwise interleave their captured lines. Holding this lock for
/// the whole duration of a capture session keeps each session's captured output self-contained.
static PRINT_CAPTURE_SESSION_LOCK: Mutex<()> = Mutex::new(());

/// Capture buffer of the process-global print sink. When `None` (the default), printed lines are written to
/// standard error. While a [`with_captured_prints`] session is active, this holds `Some` buffer that collects
/// the printed lines instead. This lock is only held for the duration of a single line append, never while
/// user code runs, so print-emitting executions never deadlock against an active capture session.
static CAPTURED_PRINT_LINES: Mutex<Option<Vec<String>>> = Mutex::new(None);

/// Writes one line to the process-global print sink: standard error by default, or the active
/// [`with_captured_prints`] capture buffer if one is installed. Lock poisoning is ignored (via
/// [`PoisonError::into_inner`]) because the sink state is a plain buffer that remains valid even if a
/// capturing test panicked while holding the lock.
fn emit_print_line(line: String) {
    let mut captured = CAPTURED_PRINT_LINES.lock().unwrap_or_else(PoisonError::into_inner);
    match captured.as_mut() {
        Some(lines) => lines.push(line),
        None => eprintln!("{line}"),
    }
}

/// Executes the provided closure while capturing all lines written to the process-global print sink, and returns
/// the closure result together with the captured lines in emission order.
///
/// Capture sessions are serialized through a process-global lock so that concurrently running tests cannot
/// interleave their captured output (parallel test threads block until the active session finishes). If the
/// closure panics, the capture buffer is uninstalled before the panic propagates, so subsequent prints fall
/// back to standard error instead of accumulating into a dead buffer.
pub fn with_captured_prints<R, F: FnOnce() -> R>(body: F) -> (R, Vec<String>) {
    /// Guard that uninstalls the capture buffer when dropped, including during panic unwinding.
    struct CaptureGuard;

    impl Drop for CaptureGuard {
        fn drop(&mut self) {
            *CAPTURED_PRINT_LINES.lock().unwrap_or_else(PoisonError::into_inner) = None;
        }
    }

    let _session = PRINT_CAPTURE_SESSION_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let guard = CaptureGuard;
    *CAPTURED_PRINT_LINES.lock().unwrap_or_else(PoisonError::into_inner) = Some(Vec::new());
    let result = body();
    let lines = CAPTURED_PRINT_LINES.lock().unwrap_or_else(PoisonError::into_inner).take().unwrap_or_default();
    drop(guard);
    (result, lines)
}

/// Registers the [`PRINT_CUSTOM_CALL_TARGET`] XLA FFI handler with the plugin backing the provided
/// [`Client`], making `stablehlo.custom_call @ryft.print` operations executable by that plugin.
///
/// Registration is lazy (i.e., performed at the first compile/execute touchpoint instead of at library load)
/// because it requires a live PJRT plugin that exposes the FFI extension, and no plugin is loaded when this
/// crate is initialized. It is also idempotent and thread-safe: the handler is registered into the XLA
/// runtime's process-global registry at most once, guarded by a [`OnceLock`], because re-registering the same
/// target name is rejected by the runtime. The registration outcome (including a failure, e.g., when the
/// plugin does not provide the FFI extension) is cached for the lifetime of the process.
///
/// The handler dereferences its operand buffers on the host, so in its current form it must only be registered
/// with CPU clients. The handler is registered for the platform reported by the provided client (e.g., `"cpu"`
/// for the built-in CPU plugin, which the XLA runtime canonicalizes to its `"Host"` platform).
pub fn ensure_print_handler_registered(client: &Client<'_>) -> Result<(), Error> {
    static PRINT_HANDLER_REGISTRATION: OnceLock<Result<(), Error>> = OnceLock::new();
    PRINT_HANDLER_REGISTRATION
        .get_or_init(|| {
            let platform_name = client.platform_name()?.into_owned();
            client.register_ffi_handler(
                PRINT_CUSTOM_CALL_TARGET,
                platform_name,
                FfiHandler::from(print_handler as XLA_FFI_Handler),
                FfiHandlerTraits::NONE,
            )
        })
        .clone()
}

/// XLA FFI handler for [`PRINT_CUSTOM_CALL_TARGET`] custom calls. Refer to the documentation of
/// [`PRINT_CUSTOM_CALL_TARGET`] for the calling convention that this handler decodes.
unsafe extern "C" fn print_handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    // SAFETY: The XLA runtime passes a call frame that is valid for the duration of this invocation, and all
    // further unsafe access to it is localized in the safe `FfiCallFrame` wrapper and `handle_print_call_frame`.
    unsafe {
        match FfiCallFrame::from_c_api(call_frame) {
            Err(_) => std::ptr::null_mut(),
            Ok(call_frame) if call_frame.register_metadata(FfiTypeId::default()) => std::ptr::null_mut(),
            Ok(call_frame) if call_frame.stage() != FfiExecutionStage::Execution => std::ptr::null_mut(),
            Ok(call_frame) => match call_frame.api() {
                Err(_) => std::ptr::null_mut(),
                Ok(api) => match handle_print_call_frame(&call_frame) {
                    Ok(()) => std::ptr::null_mut(),
                    Err(error) => error.to_c_api(api),
                },
            },
        }
    }
}

/// Decodes a [`PRINT_CUSTOM_CALL_TARGET`] call frame, writes one `"<label>: <rendered value>"` line to the
/// process-global print sink, and fills any non-token result buffers.
fn handle_print_call_frame(call_frame: &FfiCallFrame<'_>) -> Result<(), FfiError> {
    let label = decode_label(call_frame)?;
    let mut value = None;
    for input in call_frame.inputs() {
        let FfiInput::Buffer { buffer } = input?;
        if buffer.element_type() != FfiBufferType::Token {
            value = Some(buffer);
            break;
        }
    }
    let Some(value) = value else {
        return Err(FfiError::invalid_argument(format!(
            "expected the '{PRINT_CUSTOM_CALL_TARGET}' custom call to have one non-token input buffer"
        )));
    };
    emit_print_line(format!("{label}: {}", render_buffer(&value)?));
    copy_value_to_outputs(call_frame, &value)
}

/// Decodes the [`PRINT_LABEL_ATTRIBUTE`] string attribute of a [`PRINT_CUSTOM_CALL_TARGET`] call frame.
fn decode_label<'o>(call_frame: &FfiCallFrame<'o>) -> Result<&'o str, FfiError> {
    for attribute in call_frame.attributes() {
        let (name, attribute) = attribute?;
        if name == PRINT_LABEL_ATTRIBUTE {
            return match attribute {
                FfiAttribute::String { string } => Ok(string),
                _ => Err(FfiError::invalid_argument(format!(
                    "expected the '{PRINT_LABEL_ATTRIBUTE}' attribute of the '{PRINT_CUSTOM_CALL_TARGET}' custom \
                     call to be a string"
                ))),
            };
        }
    }
    Err(FfiError::invalid_argument(format!(
        "missing required '{PRINT_LABEL_ATTRIBUTE}' string attribute in the '{PRINT_CUSTOM_CALL_TARGET}' custom call"
    )))
}

/// Returns the number of elements stored in the provided [`FfiBuffer`] (i.e., the product of its dimensions).
fn element_count(buffer: &FfiBuffer<'_>) -> usize {
    buffer.dimensions().iter().map(|&dimension| dimension.max(0) as usize).product()
}

/// Returns the number of bytes used to store each element of the provided [`FfiBufferType`], and `None` for
/// types without a defined host byte width. Sub-byte integer types are byte-backed (one logical element per
/// byte) in XLA FFI buffers.
fn element_size_in_bytes(element_type: FfiBufferType) -> Option<usize> {
    match element_type {
        FfiBufferType::Invalid | FfiBufferType::Token => None,
        FfiBufferType::Predicate
        | FfiBufferType::I1
        | FfiBufferType::I2
        | FfiBufferType::I4
        | FfiBufferType::I8
        | FfiBufferType::U1
        | FfiBufferType::U2
        | FfiBufferType::U4
        | FfiBufferType::U8
        | FfiBufferType::F4E2M1FN
        | FfiBufferType::F8E3M4
        | FfiBufferType::F8E4M3
        | FfiBufferType::F8E4M3FN
        | FfiBufferType::F8E4M3FNUZ
        | FfiBufferType::F8E4M3B11FNUZ
        | FfiBufferType::F8E5M2
        | FfiBufferType::F8E5M2FNUZ
        | FfiBufferType::F8E8M0FNU => Some(1),
        FfiBufferType::I16 | FfiBufferType::U16 | FfiBufferType::BF16 | FfiBufferType::F16 => Some(2),
        FfiBufferType::I32 | FfiBufferType::U32 | FfiBufferType::F32 => Some(4),
        FfiBufferType::I64 | FfiBufferType::U64 | FfiBufferType::F64 | FfiBufferType::C64 => Some(8),
        FfiBufferType::C128 => Some(16),
    }
}

/// Renders the provided [`FfiBuffer`] as a human-readable string. `f64` buffers are rendered numerically:
/// rank-0 buffers as a bare scalar (e.g., `3.5`) and higher-rank buffers as a flat row-major list (e.g.,
/// `[1.5, 2.5]`). All other element types fall back to a `<type>[<dimensions>] 0x<bytes>` hexadecimal
/// rendering, truncated to [`MAX_RENDERED_FALLBACK_BYTES`] bytes.
fn render_buffer(buffer: &FfiBuffer<'_>) -> Result<String, FfiError> {
    let element_type = buffer.element_type();
    let count = element_count(buffer);
    // SAFETY: The data pointer is provided by the XLA runtime and is valid for the duration of the handler
    // invocation. It is only dereferenced after checking that it is non-null and only for `count` elements,
    // which the runtime guarantees are backed by the buffer allocation.
    let data = unsafe { buffer.data() };
    if count > 0 && data.is_null() {
        return Err(FfiError::internal(format!(
            "encountered null data pointer for a non-empty '{PRINT_CUSTOM_CALL_TARGET}' input buffer"
        )));
    }
    if element_type == FfiBufferType::F64 {
        // SAFETY: The buffer element type is F64 and so its allocation holds `count` contiguous `f64` values.
        let values = unsafe { std::slice::from_raw_parts(data as *const f64, count) };
        return if buffer.rank() == 0 {
            Ok(format!("{:?}", values[0]))
        } else {
            let values = values.iter().map(|value| format!("{value:?}")).collect::<Vec<_>>();
            Ok(format!("[{}]", values.join(", ")))
        };
    }
    let dimensions = buffer.dimensions().iter().map(|dimension| dimension.to_string()).collect::<Vec<_>>();
    let dimensions = dimensions.join(",");
    match element_size_in_bytes(element_type) {
        Some(element_size) if count > 0 => {
            let byte_count = count * element_size;
            let rendered_byte_count = byte_count.min(MAX_RENDERED_FALLBACK_BYTES);
            // SAFETY: The buffer allocation holds `byte_count` bytes, of which we read the leading
            // `rendered_byte_count`.
            let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, rendered_byte_count) };
            let rendered_bytes = bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
            let ellipsis = if byte_count > rendered_byte_count { "…" } else { "" };
            Ok(format!("{element_type}[{dimensions}] 0x{rendered_bytes}{ellipsis}"))
        }
        _ => Ok(format!("{element_type}[{dimensions}]")),
    }
}

/// Copies the printed `value` buffer into every non-token output buffer of the provided call frame. In the v1
/// calling convention the only result is a token (which carries no data and is skipped), so this is a no-op,
/// but it keeps value-passthrough result shapes working if a future lowering adds them.
fn copy_value_to_outputs(call_frame: &FfiCallFrame<'_>, value: &FfiBuffer<'_>) -> Result<(), FfiError> {
    for output in call_frame.outputs() {
        let FfiOutput::Buffer { buffer: output } = output?;
        if output.element_type() == FfiBufferType::Token {
            continue;
        }
        if output.element_type() != value.element_type() || output.dimensions() != value.dimensions() {
            return Err(FfiError::invalid_argument(format!(
                "expected the '{PRINT_CUSTOM_CALL_TARGET}' custom call outputs to be tokens or to match the \
                 printed value buffer"
            )));
        }
        let Some(element_size) = element_size_in_bytes(value.element_type()) else {
            return Err(FfiError::invalid_argument(format!(
                "cannot copy a '{PRINT_CUSTOM_CALL_TARGET}' value buffer with element type '{}' to an output",
                value.element_type(),
            )));
        };
        let byte_count = element_count(value) * element_size;
        if byte_count == 0 {
            continue;
        }
        // SAFETY: Both data pointers are provided by the XLA runtime, are valid for the duration of the handler
        // invocation, and (per the shape and element type equality checks above) are backed by allocations of at
        // least `byte_count` bytes. The runtime allocates inputs and outputs separately so they do not overlap.
        unsafe {
            let source = value.data() as *const u8;
            let destination = output.data() as *mut u8;
            if source.is_null() || destination.is_null() {
                return Err(FfiError::internal(format!(
                    "encountered null data pointer while copying a '{PRINT_CUSTOM_CALL_TARGET}' value to an output"
                )));
            }
            std::ptr::copy_nonoverlapping(source, destination, byte_count);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use ryft_pjrt::{
        BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program, load_cpu_plugin,
    };

    use crate::tests::{values_from_bytes, values_to_bytes};

    use super::{PRINT_CUSTOM_CALL_TARGET, emit_print_line, ensure_print_handler_registered, with_captured_prints};

    #[test]
    fn test_with_captured_prints() {
        // Captured lines are returned in emission order together with the closure result.
        let (result, lines) = with_captured_prints(|| {
            emit_print_line("x: 3.5".to_string());
            emit_print_line("y: [1.5, 2.5]".to_string());
            42
        });
        assert_eq!(result, 42);
        assert_eq!(lines, vec!["x: 3.5".to_string(), "y: [1.5, 2.5]".to_string()]);

        // Each capture session starts with an empty buffer.
        let ((), lines) = with_captured_prints(|| ());
        assert_eq!(lines, Vec::<String>::new());

        // A panicking session uninstalls its capture buffer, and later sessions keep working even though the
        // panic poisoned the session lock while it was held.
        let panic = std::panic::catch_unwind(|| with_captured_prints(|| panic!("boom")));
        assert!(panic.is_err());
        let ((), lines) = with_captured_prints(|| emit_print_line("z: 4.5".to_string()));
        assert_eq!(lines, vec!["z: 4.5".to_string()]);
    }

    #[test]
    fn test_print_custom_call_executes_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        assert_eq!(ensure_print_handler_registered(&client), Ok(()));
        // Registration is idempotent.
        assert_eq!(ensure_print_handler_registered(&client), Ok(()));

        // The program threads one token chain through three prints: an `f64` vector argument, an `f64` scalar
        // constant, and an `f32` constant that exercises the hexadecimal fallback rendering.
        let program = Program::Mlir {
            bytecode: indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf64>) -> tensor<2xf64> {
                    %scalar = stablehlo.constant dense<3.5> : tensor<f64>
                    %floats = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
                    %token0 = stablehlo.create_token : !stablehlo.token
                    %token1 = stablehlo.custom_call @"__TARGET__"(%arg0, %token0)
                      {api_version = 4 : i32, backend_config = {label = "x"}, has_side_effect = true}
                      : (tensor<2xf64>, !stablehlo.token) -> !stablehlo.token
                    %token2 = stablehlo.custom_call @"__TARGET__"(%scalar, %token1)
                      {api_version = 4 : i32, backend_config = {label = "scalar"}, has_side_effect = true}
                      : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                    %token3 = stablehlo.custom_call @"__TARGET__"(%floats, %token2)
                      {api_version = 4 : i32, backend_config = {label = "floats"}, has_side_effect = true}
                      : (tensor<2xf32>, !stablehlo.token) -> !stablehlo.token
                    return %arg0 : tensor<2xf64>
                  }
                }
            "#}
            .replace("__TARGET__", PRINT_CUSTOM_CALL_TARGET)
            .into_bytes(),
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
        let input_values = [1.5f64, 2.5f64];
        let input_bytes = values_to_bytes::<f64>(&input_values);

        let ((), lines) = with_captured_prints(|| {
            let inputs = ExecutionDeviceInputs {
                inputs: &[ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(input_bytes.as_slice(), BufferType::F64, &[2], None, device.clone(), None)
                            .unwrap(),
                    ),
                    donatable: false,
                }],
                ..Default::default()
            };
            let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
            let mut outputs = execution.block_until_ready().unwrap().remove(0);
            assert_eq!(outputs.outputs.len(), 1);
            let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f64>(output_bytes.as_slice()), input_values.to_vec());
        });

        let expected_floats_bytes = values_to_bytes::<f32>(&[1.0f32, 2.0f32])
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            lines,
            vec![
                "x: [1.5, 2.5]".to_string(),
                "scalar: 3.5".to_string(),
                format!("floats: f32[2] 0x{expected_floats_bytes}"),
            ],
        );
    }
}
