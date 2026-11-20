use std::sync::{Arc, Mutex, PoisonError};

use ryft_core::macros::check_count;
use ryft_core::{Array, ArrayAddressing, ArrayType, DataType, EffectClass, PrintOperation, Typed};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::{CustomCallApiVersion, CustomCallMemoryLayouts};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue,
    ValueRef,
};
use ryft_pjrt::extensions::ffi::{
    FfiAttribute, FfiBuffer, FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits,
    FfiInput, FfiOutput, FfiTypeId, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Handler,
};
use ryft_pjrt::{BufferType, Client, Error as PjrtError};

use crate::lowering::types::{lower_signed_one_bit_to_mlir, lower_tensor_type};
use crate::lowering::{EffectTokens, current_or_new_token};
use crate::sessions::effects::EffectCallbackGuard;
use crate::{
    Error, ExecutionError, FromPjrt, LowerableOperation, LowerableOperationFamily, Lowerer, RuntimeExtension,
    RuntimeProvider, RuntimeRequirement,
};

#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
use ryft_pjrt::extensions::ffi::FfiStream;

#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
use crate::operations::assertions::copy_cuda_bytes;

/// Name of the XLA custom call target that implements the host callback side of `print` lowering.
///
/// A lowered [`PrintOperation`] becomes a `stablehlo.custom_call` with the following calling convention, which the FFI
/// handler registered by [`ensure_print_handler_registered`] decodes:
///
///   - `call_target_name` is [`PRINT_CUSTOM_CALL_TARGET`] (i.e., `"ryft.print"`).
///   - `api_version` is `4` (i.e., the typed FFI API version), so that the `backend_config` dictionary entries arrive
///     in the XLA FFI call frame as decoded [`FfiAttribute`]s.
///   - `has_side_effect` is `true`, so that the XLA compiler never elides or reorders the call.
///   - `backend_config` contains the string label to print and may supply the logical type of the printed buffer.
///   - The operands are `[value, token]`, which consist of the dense row-major array value to print followed by a
///     `!stablehlo.token` that orders this print relative to other operations with side effects.
///   - The single result is a `!stablehlo.token` that continues the effect ordering token chain.
///
/// The handler prints the first non-token operand buffer and ignores token operands and results, so it also
/// tolerates the degenerate `[value] -> [token]` shape. If a future variant adds a non-token result (e.g., a
/// value passthrough), the handler copies the printed operand into any result buffer with a matching element
/// type and shape.
pub const PRINT_CUSTOM_CALL_TARGET: &str = "ryft.print";

/// Name of the `backend_config` dictionary entry that carries the label of a [`PRINT_CUSTOM_CALL_TARGET`]
/// custom call. Refer to the documentation of [`PRINT_CUSTOM_CALL_TARGET`] for the full calling convention.
pub const PRINT_LABEL_ATTRIBUTE: &str = "label";

/// Name of the optional attribute that identifies a logical type whose formatting differs from its FFI carrier.
const PRINT_LOGICAL_TYPE_ATTRIBUTE: &str = "logical_type";

/// Serializes [`with_captured_prints`] sessions across threads. Because the print sink is process-global,
/// two concurrently capturing tests would otherwise interleave their captured lines. Holding this lock for
/// the whole duration of a capture session keeps each session's captured output self-contained.
static PRINT_CAPTURE_SESSION_LOCK: Mutex<()> = Mutex::new(());

/// Capture buffer of the process-global print sink. When it is `None`, which is the default, printed lines are
/// written to standard error. While a [`with_captured_prints`] session is active, this holds `Some` buffer that
/// collects the printed lines instead. This lock is only held for the duration of a single line append, never while
/// user code runs, so executions that print never deadlock against an active capture session.
static CAPTURED_PRINT_LINES: Mutex<Option<Vec<String>>> = Mutex::new(None);

/// Registers the [`PRINT_CUSTOM_CALL_TARGET`] XLA FFI handler with the plugin backing the provided
/// [`Client`], making `stablehlo.custom_call @ryft.print` operations executable by that plugin.
///
/// Registration is lazy because it requires a live plugin exposing the FFI extension. It may be repeated, e.g., once
/// per session, because XLA accepts re-registration of an identical handler for the same target and platform.
///
/// The handler is selected by the platform reported by the provided client. The CPU handler renders its operand
/// buffer directly from host memory. With a `cuda-*` feature enabled, the CUDA handler copies the operand through
/// the invocation's CUDA stream before rendering it. Other platforms are rejected because these handlers cannot
/// read their buffers. An integration for another platform can provide its own runtime provider and handler.
pub fn ensure_print_handler_registered(client: &Client<'_>) -> Result<(), Error> {
    let platform_name = client.platform_name()?.into_owned();
    if platform_name.eq_ignore_ascii_case("cpu") {
        return Ok(client.register_ffi_handler(
            PRINT_CUSTOM_CALL_TARGET,
            platform_name,
            FfiHandler::from(print_handler as XLA_FFI_Handler),
            FfiHandlerTraits::NONE,
        )?);
    }
    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    if platform_name.eq_ignore_ascii_case("cuda") {
        return Ok(client.register_ffi_handler(
            PRINT_CUSTOM_CALL_TARGET,
            platform_name,
            FfiHandler::from(cuda_print_handler as XLA_FFI_Handler),
            FfiHandlerTraits::NONE,
        )?);
    }
    Err(PjrtError::unimplemented(format!(
        "the `{PRINT_CUSTOM_CALL_TARGET}` handler cannot read operand buffers of the `{platform_name}` platform",
    ))
    .into())
}

/// Executes the provided closure while capturing all lines written to the process-global print sink, and returns
/// the closure result together with the captured lines in emission order.
///
/// Capture sessions are serialized through a process-global lock so that concurrently running tests cannot
/// interleave their captured output, and parallel test threads block until the active session finishes. If the
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

/// Returns the built-in print provider without loading a PJRT plugin.
pub(crate) fn print_runtime_provider() -> Arc<dyn RuntimeProvider> {
    Arc::new(PrintRuntimeProvider)
}

/// Returns the built-in ordered I/O transport provider without performing native initialization.
pub(crate) fn ordered_io_runtime_provider() -> Arc<dyn RuntimeProvider> {
    Arc::new(OrderedIoRuntimeProvider)
}

impl LowerableOperation for PrintOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        // `print` is the identity on its data flow output, and its observable effect lowers to a host callback custom
        // call that consumes and produces a StableHLO token, so the effect ordering rides the scope's token chain
        // instead of the value data flow.
        check_count!("input", input_values, 1, ProgramError);
        check_count!("input type", input_types, 1, TypeError);
        lowerer.require_runtime(RuntimeRequirement::builtin(PRINT_CUSTOM_CALL_TARGET))?;
        let (mut block, context, location) = (lowerer.block(), lowerer.context(), lowerer.location());
        lower_print_to_custom_call(
            self.label(),
            self.effect_class().is_ordered(),
            input_values[0],
            &input_types[0],
            lowerer.effect_tokens_mut(),
            &mut block,
            context,
            location,
        )?;
        Ok(vec![input_values[0]])
    }
}

/// Built-in platform admission and registration for printing.
struct PrintRuntimeProvider;

impl RuntimeProvider for PrintRuntimeProvider {
    fn identifier(&self) -> &str {
        PRINT_CUSTOM_CALL_TARGET
    }

    fn initialize(
        &self,
        client: &'static Client<'static>,
        requirement: &RuntimeRequirement,
    ) -> Result<Arc<dyn RuntimeExtension>, Error> {
        requirement.validate_builtin(self.identifier())?;
        ensure_print_handler_registered(client)?;
        Ok(Arc::new(PrintRuntime))
    }
}

/// Printing needs native handler registration but no invocation-owned user data.
struct PrintRuntime;

impl RuntimeExtension for PrintRuntime {}

/// Verified native ordered I/O token transport admission, independent of print registration.
struct OrderedIoRuntimeProvider;

impl RuntimeProvider for OrderedIoRuntimeProvider {
    fn identifier(&self) -> &str {
        "ryft.ordered-io"
    }

    fn initialize(
        &self,
        client: &'static Client<'static>,
        requirement: &RuntimeRequirement,
    ) -> Result<Arc<dyn RuntimeExtension>, Error> {
        requirement.validate_builtin(self.identifier())?;
        let platform_name = client.platform_name()?;
        if !["cpu", "cuda"].iter().any(|platform| platform_name.eq_ignore_ascii_case(platform)) {
            return Err(ExecutionError::UnsupportedEffectPlatform { platform: platform_name.into_owned() }.into());
        }
        Ok(Arc::new(OrderedIoRuntime))
    }
}

/// Ordered I/O admission introduces no per invocation user data of its own.
struct OrderedIoRuntime;

impl RuntimeExtension for OrderedIoRuntime {}

/// Memory location of the value buffer passed to one print callback.
#[derive(Copy, Clone)]
enum PrintBufferMemory {
    /// CPU buffer that the callback can read directly.
    Host,

    /// CUDA device buffer read back through the invocation's stream.
    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    Cuda(FfiStream),
}

/// Lowers one `print` operation to the [`PRINT_CUSTOM_CALL_TARGET`] host callback custom call and advances the
/// scope's effect token chain past it when ordered. Unordered printing retains the same handler ABI but uses an
/// independent local token.
///
/// The emitted operation follows the calling convention decoded by the FFI handler registered by
/// [`ensure_print_handler_registered`]:
///
/// ```mlir
/// %token_out = stablehlo.custom_call @"ryft.print"(%value, %token_in)
///   {api_version = 4 : i32, backend_config = {label = "<label>"}, has_side_effect = true, ...}
///   : (tensor<...>, !stablehlo.token) -> !stablehlo.token
/// ```
///
/// The custom call's only result is the continuation token, while the `print` operation's data flow output is the
/// forwarded input value, which the caller returns directly.
#[allow(clippy::too_many_arguments)]
fn lower_print_to_custom_call<'b, 'c: 'b, 't: 'c>(
    label: &str,
    ordered: bool,
    value: ValueRef<'b, 'c, 't>,
    value_type: &ArrayType,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), Error> {
    // Unordered calls retain the registered handler ABI, using a fresh local token with no boundary dependency.
    let mut local_tokens = EffectTokens::default();
    let effect_tokens = if ordered { effect_tokens } else { &mut local_tokens };
    let input_token = current_or_new_token(EffectClass::OrderedIo, effect_tokens, block, location)?;
    let token_type = context.stable_hlo_token_type()?;
    let mut attributes =
        vec![context.named_attribute(context.identifier(PRINT_LABEL_ATTRIBUTE), context.string_attribute(label))];
    let data_type = value_type.data_type();
    if matches!(data_type, DataType::Zero | DataType::F4E2M1FN | DataType::F6E2M3FN | DataType::F6E3M2FN) {
        attributes.push(context.named_attribute(
            context.identifier(PRINT_LOGICAL_TYPE_ATTRIBUTE),
            context.string_attribute(data_type.to_string().as_str()),
        ));
    }
    let backend_config = context.dictionary_attribute(&attributes);
    let value = if data_type == DataType::I1 {
        lower_signed_one_bit_to_mlir(value, block, context, location)?
    } else {
        // XLA can pack sub-byte elements, and the FFI buffer view does not describe that packing.
        let carrier = match data_type {
            DataType::I2 | DataType::I4 => Some(DataType::I8),
            DataType::U1 | DataType::U2 | DataType::U4 => Some(DataType::U8),
            DataType::F4E2M1FN | DataType::F6E2M3FN | DataType::F6E3M2FN => Some(DataType::F32),
            _ => None,
        };
        if let Some(carrier) = carrier {
            let carrier_type = lower_tensor_type(&value_type.clone().with_data_type(carrier), context, location)?;
            block
                .append_operation(stable_hlo::convert(value, carrier_type, location)?)?
                .result(0)
                .unwrap()
                .as_ref()
        } else {
            value
        }
    };
    // An FFI view carries dimensions and an element type. Pin the layout so its bytes are in logical order.
    let memory_layouts = CustomCallMemoryLayouts {
        operands: vec![(0..value_type.rank()).rev().collect(), Vec::new()],
        results: vec![Vec::new()],
    };
    let operation = block.append_operation(stable_hlo::custom_call(
        &[value, input_token],
        PRINT_CUSTOM_CALL_TARGET,
        true,
        Some(backend_config.as_ref()),
        CustomCallApiVersion::TypedFfi,
        &[],
        Some(memory_layouts),
        &[],
        None,
        &[token_type],
        location,
    )?)?;
    effect_tokens.set(EffectClass::OrderedIo, operation.result(0).unwrap().as_ref());
    Ok(())
}

/// XLA FFI handler for [`PRINT_CUSTOM_CALL_TARGET`] custom calls whose operands live in host memory. Refer to the
/// documentation of [`PRINT_CUSTOM_CALL_TARGET`] for the calling convention that this handler decodes.
unsafe extern "C" fn print_handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    // SAFETY: XLA passes a call frame that it owns and keeps valid for the duration of this invocation.
    unsafe { print_handler_for_memory(call_frame, |_| Ok(PrintBufferMemory::Host)) }
}

/// XLA FFI handler for [`PRINT_CUSTOM_CALL_TARGET`] custom calls whose operands live in CUDA device memory.
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
unsafe extern "C" fn cuda_print_handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    // SAFETY: XLA passes a call frame that it owns and keeps valid for the duration of this invocation, and the
    // stream that the call frame context returns stays valid for the same duration.
    unsafe {
        print_handler_for_memory(call_frame, |call_frame| Ok(PrintBufferMemory::Cuda(call_frame.context()?.stream()?)))
    }
}

/// Decodes one XLA FFI invocation and prints it using the memory location returned by `memory`. Failures are
/// returned to XLA as FFI errors, and the effect callback guard is held for the whole invocation.
///
/// # Safety
///
/// `call_frame` must be the call frame that XLA passed to the invoking handler, and it must remain valid until this
/// function returns. Its buffers must reside in the memory location that `memory` reports.
unsafe fn print_handler_for_memory(
    call_frame: *mut XLA_FFI_CallFrame,
    memory: impl FnOnce(&FfiCallFrame<'_>) -> Result<PrintBufferMemory, FfiError>,
) -> *mut XLA_FFI_Error {
    let _callback_guard = EffectCallbackGuard::enter();
    // SAFETY: The XLA runtime passes a call frame that is valid for the duration of this invocation, and all
    // further unsafe access to it is localized in the safe `FfiCallFrame` wrapper and `handle_print_call_frame`.
    unsafe {
        match FfiCallFrame::from_c_api(call_frame) {
            Err(_) => std::ptr::null_mut(),
            Ok(call_frame) if call_frame.register_metadata(FfiTypeId::default()) => std::ptr::null_mut(),
            Ok(call_frame) if call_frame.stage() != FfiExecutionStage::Execution => std::ptr::null_mut(),
            Ok(call_frame) => match call_frame.api() {
                Err(_) => std::ptr::null_mut(),
                Ok(api) => match memory(&call_frame).and_then(|memory| handle_print_call_frame(&call_frame, memory)) {
                    Ok(()) => std::ptr::null_mut(),
                    Err(error) => error.to_c_api(api),
                },
            },
        }
    }
}

/// Decodes a [`PRINT_CUSTOM_CALL_TARGET`] call frame, writes one `"<label>: <rendered value>"` line to the
/// process-global print sink, and fills any non-token result buffers.
fn handle_print_call_frame(call_frame: &FfiCallFrame<'_>, memory: PrintBufferMemory) -> Result<(), FfiError> {
    let (label, logical_type) = decode_print_attributes(call_frame)?;
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
            "expected the `{PRINT_CUSTOM_CALL_TARGET}` custom call to have one non-token input buffer",
        )));
    };
    let array = read_print_buffer(&value, memory)?;
    let byte_count = array.storage_bytes().len();
    let array = match (logical_type, value.element_type()) {
        (None, _) => Ok(array),
        (Some(DataType::Zero), FfiBufferType::Predicate) => {
            Array::new(array.r#type().into_owned().with_data_type(DataType::Zero), Vec::new())
        }
        (Some(data_type), FfiBufferType::F32) if !data_type.is_zero() => array.converted_to(data_type),
        (Some(data_type), carrier) => {
            return Err(FfiError::invalid_argument(format!(
                "invalid `{PRINT_CUSTOM_CALL_TARGET}` buffer type `{carrier}` for logical type `{data_type}`",
            )));
        }
    }
    .map_err(|error| FfiError::invalid_argument(error.to_string()))?;
    emit_print_line(format!("{label}: {array}"));
    copy_value_to_outputs(call_frame, &value, memory, byte_count)
}

/// Writes one line to the process-global print sink, which is standard error by default, or the active
/// [`with_captured_prints`] capture buffer if one is installed. Lock poisoning is ignored using
/// [`PoisonError::into_inner`] because the sink state is a plain buffer that remains valid even if a capturing test
/// panicked while holding the lock.
fn emit_print_line(line: String) {
    let mut captured = CAPTURED_PRINT_LINES.lock().unwrap_or_else(PoisonError::into_inner);
    match captured.as_mut() {
        Some(lines) => lines.push(line),
        None => eprintln!("{line}"),
    }
}

/// Decodes the print label and optional logical element type from a print call frame.
fn decode_print_attributes<'o>(call_frame: &FfiCallFrame<'o>) -> Result<(&'o str, Option<DataType>), FfiError> {
    let mut label = None;
    let mut logical_type = None;
    for attribute in call_frame.attributes() {
        let (name, attribute) = attribute?;
        if !matches!(name, PRINT_LABEL_ATTRIBUTE | PRINT_LOGICAL_TYPE_ATTRIBUTE) {
            continue;
        }
        let FfiAttribute::String { string } = attribute else {
            return Err(FfiError::invalid_argument(format!(
                "expected the `{name}` attribute of the `{PRINT_CUSTOM_CALL_TARGET}` custom call to be a string",
            )));
        };
        if name == PRINT_LABEL_ATTRIBUTE {
            label = Some(string);
        } else {
            logical_type = Some(match string {
                "zero" => DataType::Zero,
                "f4e2m1fn" => DataType::F4E2M1FN,
                "f6e2m3fn" => DataType::F6E2M3FN,
                "f6e3m2fn" => DataType::F6E3M2FN,
                _ => {
                    return Err(FfiError::invalid_argument(format!(
                        "unsupported logical type `{string}` for the `{PRINT_CUSTOM_CALL_TARGET}` custom call",
                    )));
                }
            });
        }
    }
    let label = label.ok_or_else(|| {
        FfiError::invalid_argument(format!(
            "missing required `{PRINT_LABEL_ATTRIBUTE}` string attribute in the `{PRINT_CUSTOM_CALL_TARGET}` custom call",
        ))
    })?;
    Ok((label, logical_type))
}

/// Materializes a checked host array from a dense row-major print buffer.
fn read_print_buffer(buffer: &FfiBuffer<'_>, memory: PrintBufferMemory) -> Result<Array, FfiError> {
    let data_type = DataType::from_pjrt(BufferType::from_proto(buffer.element_type().proto()))
        .map_err(|error| FfiError::invalid_argument(error.to_string()))?;
    if data_type.is_token() || !data_type.bit_width().is_multiple_of(8) {
        return Err(FfiError::invalid_argument(format!(
            "the `{PRINT_CUSTOM_CALL_TARGET}` custom call requires a byte-aligned value buffer, got `{data_type}`",
        )));
    }
    let dimensions = buffer
        .dimensions()
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension).map_err(|_| {
                FfiError::invalid_argument(format!("invalid `{PRINT_CUSTOM_CALL_TARGET}` buffer dimension {dimension}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let array_type = ArrayType::new_static(data_type, dimensions);
    let byte_count = ArrayAddressing::new(array_type.clone())
        .map_err(|error| FfiError::invalid_argument(error.to_string()))?
        .storage_byte_len();
    if byte_count > isize::MAX as usize {
        return Err(FfiError::invalid_argument(format!(
            "`{PRINT_CUSTOM_CALL_TARGET}` buffer exceeds the maximum slice length",
        )));
    }
    let bytes = read_print_buffer_bytes(buffer, memory, byte_count)?;
    Array::new(array_type, bytes).map_err(|error| FfiError::invalid_argument(error.to_string()))
}

/// Reads the checked byte length of a print buffer from host memory or through the current CUDA stream.
fn read_print_buffer_bytes(
    buffer: &FfiBuffer<'_>,
    memory: PrintBufferMemory,
    byte_count: usize,
) -> Result<Vec<u8>, FfiError> {
    // An empty buffer may have a null data pointer.
    if byte_count == 0 {
        return Ok(Vec::new());
    }
    // SAFETY: XLA keeps the buffer allocation alive for the entire callback invocation.
    let data = unsafe { buffer.data() };
    if data.is_null() {
        return Err(FfiError::internal(format!(
            "encountered null data pointer for a non-empty `{PRINT_CUSTOM_CALL_TARGET}` input buffer",
        )));
    }
    match memory {
        PrintBufferMemory::Host => {
            // SAFETY: The handler receives host memory in this path. The dense byte-aligned layout and the checked
            // buffer shape bound the byte count to the allocation and to the maximum slice size.
            Ok(unsafe { std::slice::from_raw_parts(data.cast::<u8>(), byte_count) }.to_vec())
        }
        #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
        PrintBufferMemory::Cuda(stream) => {
            let mut bytes = vec![0u8; byte_count];
            // SAFETY: The device allocation and the host destination each hold at least `byte_count` bytes, and
            // the stream is the stream of this invocation.
            unsafe { copy_cuda_bytes(data, bytes.as_mut_ptr().cast(), byte_count, stream) }?;
            Ok(bytes)
        }
    }
}

/// Copies the printed `value` buffer into every non-token output buffer of the provided call frame. In the calling
/// convention that [`PRINT_CUSTOM_CALL_TARGET`] documents, the only result is a token, which carries no data and is
/// skipped, so this is a no-op, but it keeps value passthrough result shapes working if a future lowering adds them.
/// Only host buffers can be copied, because a CUDA device passthrough would need a device-to-device copy that no
/// current lowering requires.
fn copy_value_to_outputs(
    call_frame: &FfiCallFrame<'_>,
    value: &FfiBuffer<'_>,
    memory: PrintBufferMemory,
    byte_count: usize,
) -> Result<(), FfiError> {
    for output in call_frame.outputs() {
        let FfiOutput::Buffer { buffer: output } = output?;
        if output.element_type() == FfiBufferType::Token {
            continue;
        }
        match memory {
            PrintBufferMemory::Host => {}
            #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
            PrintBufferMemory::Cuda(_) => {
                return Err(FfiError::unimplemented(format!(
                    "the `{PRINT_CUSTOM_CALL_TARGET}` custom call cannot copy its value to a CUDA device output buffer",
                )));
            }
        }
        if output.element_type() != value.element_type() || output.dimensions() != value.dimensions() {
            return Err(FfiError::invalid_argument(format!(
                "expected the `{PRINT_CUSTOM_CALL_TARGET}` custom call outputs to be tokens or to match the \
                 printed value buffer",
            )));
        }
        if byte_count == 0 {
            continue;
        }
        // SAFETY: Both data pointers are provided by the XLA runtime, are valid for the duration of the handler
        // invocation, and are backed by allocations of at least `byte_count` bytes, as the shape and element type
        // equality checks above ensure. A custom call may declare an input-output alias.
        unsafe {
            let source = value.data() as *const u8;
            let destination = output.data() as *mut u8;
            if source.is_null() || destination.is_null() {
                return Err(FfiError::internal(format!(
                    "encountered null data pointer while copying a `{PRINT_CUSTOM_CALL_TARGET}` value to an output",
                )));
            }
            std::ptr::copy(source, destination, byte_count);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayType, BatchAxis, DataType, Device, DeviceMesh, Dimension, Layout, LogicalMesh, MeshAxis,
        MeshAxisType, Placeholder, Print, PrintOperation, Shape, Sharding, TiledLayout, Typed, batch,
    };
    use ryft_pjrt::extensions::ffi::XLA_FFI_Buffer;
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program};
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{ClientOptions, GpuClientOptions, GpuMemoryAllocator, GpuPlatform};

    #[cfg(feature = "cuda-13")]
    use crate::Platform;
    use crate::lowering::lower_mlir_module_for_program_with_reference_state;
    use crate::lowering::tests::{ProgramBuilder, unproject_plain_program};
    use crate::operations::dot::RaggedDotLoweringStrategy;
    use crate::sessions::tests::{domain_mesh, f64_vector, read_first_shard};
    use crate::tests::{
        cpu_client, execution_client, replicated_array, test_compilation_options, values_from_bytes, values_to_bytes,
    };
    use crate::{
        Array, ArrayConstant, CompiledFunction, FromPjrt, Options, RuntimeError, Session, compile,
        to_mlir_module_for_program,
    };

    use super::*;

    /// Eagerly binds one `print` on `client` and verifies that the handler fires once with the device value and that
    /// the payload passes through unchanged.
    fn assert_eager_bind_executes_print_effect(client: &'static Client<'static>) {
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let values = [1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5];
        let r#type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new())));
        let input = Array::from_host_buffer(&session, r#type, mesh, None, values_to_bytes(&values)).unwrap();
        let input_type = input.r#type().into_owned();

        // Eager dispatch registers the callback and prints the matrix in logical order despite its storage layout.
        let (output, lines) = with_captured_prints(|| {
            let output = input.print("x").unwrap();
            session.synchronize().unwrap();
            output
        });

        assert_eq!(lines, vec!["x: [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]"]);
        assert_eq!(output.r#type().as_ref(), &input_type);
        assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
    }

    #[test]
    fn test_ensure_print_handler_registered() {
        let client = cpu_client(1);
        ensure_print_handler_registered(client).unwrap();
        // Registration is idempotent.
        ensure_print_handler_registered(client).unwrap();

        // The program threads one token chain through three prints of an `f64` vector argument, an `f64` scalar
        // constant, and an `f32` vector constant.
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

        let executable = client.compile(&program, &test_compilation_options(client, None)).unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        let input_values = [1.5f64, 2.5f64];
        let input_bytes = values_to_bytes::<f64>(&input_values);

        let ((), lines) = with_captured_prints(|| {
            let inputs = ExecutionDeviceInputs {
                inputs: &[ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(input_bytes.as_slice(), BufferType::F64, [2], None, device.clone(), None)
                            .unwrap(),
                    ),
                    donatable: false,
                }],
                ..Default::default()
            };
            let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None, &[]).unwrap();
            let mut outputs = execution.block_until_ready().unwrap().remove(0);
            assert_eq!(outputs.outputs.len(), 1);
            let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f64>(output_bytes.as_slice()), input_values.to_vec());
        });

        assert_eq!(lines, vec!["x: [1.5, 2.5]", "scalar: 3.5", "floats: [1.0, 2.0]"]);
    }

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
    fn test_print_runtime_provider_initialize() {
        let client = cpu_client(1);
        let provider = print_runtime_provider();
        let requirement = RuntimeRequirement::builtin(PRINT_CUSTOM_CALL_TARGET);

        // Repeated initialization registers the same handler again, which XLA accepts.
        provider.initialize(client, &requirement).unwrap();
        provider.initialize(client, &requirement).unwrap();

        let requirement = RuntimeRequirement::new(PRINT_CUSTOM_CALL_TARGET, 2).unwrap();
        assert!(matches!(
            provider.initialize(client, &requirement),
            Err(Error::Runtime(RuntimeError::UnsupportedVersion { identifier, version: 2 }))
                if identifier == PRINT_CUSTOM_CALL_TARGET,
        ));
    }

    #[test]
    fn test_ordered_io_runtime_provider_initialize() {
        let client = cpu_client(1);
        let provider = ordered_io_runtime_provider();
        provider.initialize(client, &RuntimeRequirement::builtin("ryft.ordered-io")).unwrap();
    }

    #[test]
    fn test_print_operation_lower() {
        // In a flat program with two prints, the token chain is created lazily by one zero-operand
        // `stablehlo.after_all` at the first print, the second print consumes the first print's token result, and
        // each print's data flow output is its forwarded operand. The final add therefore reads `%arg0` and `%0`
        // rather than custom call results.
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let first = builder
            .add_instruction(PrintOperation::<ArrayType>::new("first"), Vec::new(), vec![input], None)
            .unwrap()[0];
        let second = builder
            .add_instruction(PrintOperation::<ArrayType>::new("second"), Vec::new(), vec![doubled], None)
            .unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf64>) -> tensor<2xf64> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<2xf64>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.print(%arg0, %1) {api_version = 4 : i32, backend_config = {label = "first"}, has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<2xf64>, !stablehlo.token) -> !stablehlo.token
                    %3 = stablehlo.custom_call @ryft.print(%0, %2) {api_version = 4 : i32, backend_config = {label = "second"}, has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<2xf64>, !stablehlo.token) -> !stablehlo.token
                    %4 = stablehlo.add %arg0, %0 : tensor<2xf64>
                    return %4 : tensor<2xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_print_operation_lower_executable_effect_boundary() {
        for effect_class in [EffectClass::UnorderedIo, EffectClass::OrderedIo, EffectClass::DeviceOrderedIo] {
            let ordered = effect_class.is_ordered();
            let array_type = ArrayType::scalar(DataType::F32);
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(array_type.clone());
            builder
                .add_instruction(
                    PrintOperation::<ArrayType>::new("value").with_effect_class(effect_class),
                    Vec::new(),
                    vec![input],
                    None,
                )
                .unwrap();
            let program =
                builder.build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![]).unwrap();
            let lowered = lower_mlir_module_for_program_with_reference_state(
                &unproject_plain_program(program),
                &[],
                &vec![array_type],
                &Vec::<ArrayType>::new(),
                "main",
                None,
                None,
                Some("cpu"),
                &[],
                RaggedDotLoweringStrategy::default(),
                &[],
            )
            .unwrap();
            let (stable_hlo, signature, _, runtime_requirements) = lowered.into_parts();
            assert_eq!(signature.has_ordered_io(), ordered);
            let requirements = if ordered {
                vec![
                    RuntimeRequirement::builtin("ryft.ordered-io"),
                    RuntimeRequirement::builtin(PRINT_CUSTOM_CALL_TARGET),
                ]
            } else {
                vec![RuntimeRequirement::builtin(PRINT_CUSTOM_CALL_TARGET)]
            };
            assert_eq!(runtime_requirements.as_ref(), requirements);
            assert_eq!(signature.has_device_ordered_io(), effect_class == EffectClass::DeviceOrderedIo);
            assert_eq!(signature.has_unordered_io(), !ordered);
            assert!(signature.has_effects());
            assert!(!signature.requires_assertion_handler());
            assert_eq!(signature.ordered_io_input_index(), ordered.then_some(1));
            assert_eq!(signature.ordered_io_output_index(), ordered.then_some(0));
            assert_eq!(signature.physical_input_count(), 1 + usize::from(ordered));
            let expected = if ordered {
                indoc! {"
                    module {
                      func.func @main(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
                        %0 = stablehlo.custom_call @ryft.print(%arg0, %arg1) {api_version = 4 : i32, backend_config = \
                            {label = \"value\"}, has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, \
                            dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : \
                            (tensor<f32>, !stablehlo.token) -> !stablehlo.token
                        return %0 : !stablehlo.token
                      }
                    }
                "}
            } else {
                indoc! {"
                    module {
                      func.func @main(%arg0: tensor<f32>) {
                        %0 = stablehlo.after_all  : !stablehlo.token
                        %1 = stablehlo.custom_call @ryft.print(%arg0, %0) {api_version = 4 : i32, backend_config = \
                            {label = \"value\"}, has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, \
                            dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : \
                            (tensor<f32>, !stablehlo.token) -> !stablehlo.token
                        return
                      }
                    }
                "}
            };
            assert_eq!(stable_hlo, expected);
        }
    }

    #[test]
    fn test_print_operation_lower_executes_in_order_on_cpu() {
        let client = cpu_client(1);
        ensure_print_handler_registered(client).unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let engine = Session::builder().client(client).build().unwrap();

        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |x| {
                let x = x.print("x").unwrap();
                (x.clone() + x).print("doubled").unwrap()
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        let values = [1.5f64, 2.5];
        let source = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f64>(&values).as_slice(),
        )
        .unwrap();
        let (output, lines) = with_captured_prints(|| compiled.executable().execute(&engine, source).unwrap());

        // Both prints fire in program order, and the printed values are the forwarded operands.
        assert_eq!(lines, vec!["x: [1.5, 2.5]".to_string(), "doubled: [3.0, 5.0]".to_string()]);

        // The compiled function still computes its data flow output, because each print is the identity on its
        // operand.
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let output_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f64>(output_bytes.as_slice()), vec![3.0, 5.0]);
    }

    #[test]
    fn test_print_operation_lower_executes_eagerly() {
        let client = execution_client();
        assert_eager_bind_executes_print_effect(client);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_print_operation_lower_executes_eagerly_on_cuda() {
        let client = Platform::Cuda
            .client(Some(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            })))
            .unwrap();
        assert_eager_bind_executes_print_effect(&client);
    }

    #[test]
    fn test_print_operation_lower_preserves_sub_byte_and_zero_values() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        for (data_type, bytes, expected) in [
            (DataType::I1, &[0, 1][..], "x: [0, -1]"),
            (DataType::I4, &[14, 3], "x: [-2, 3]"),
            (DataType::U2, &[2, 3], "x: [2, 3]"),
            (DataType::F4E2M1FN, &[0x02, 0x0a], "x: [1, -1]"),
            (DataType::F6E2M3FN, &[0x08, 0x28], "x: [1, -1]"),
            (DataType::F6E3M2FN, &[0x0c, 0x2c], "x: [1, -1]"),
            (DataType::Zero, &[], "x: [zero, zero]"),
        ] {
            let input = replicated_array(&session, &mesh, data_type, &[2], bytes);
            let (output, lines) = with_captured_prints(|| {
                let output = input.print("x").unwrap();
                session.synchronize().unwrap();
                output
            });
            assert_eq!(lines, vec![expected], "{data_type}");
            assert_eq!(output.data_type(), data_type);
            assert_eq!(output.copy_to_host_bytes().unwrap().as_slice(), bytes);
        }
    }

    #[test]
    fn test_print_operation_lower_executes_batched_prints_on_cpu() {
        let client = cpu_client(1);
        ensure_print_handler_registered(client).unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let input = f64_vector(&session, &mesh, &[1.5, 2.5]);

        // Eager batching binds one print of the whole batch through the `@ryft.print` host callback.
        let (output, lines) = with_captured_prints(|| {
            batch(|item| item.print("x"), input.clone(), BatchAxis::new(0), BatchAxis::new(0), None).unwrap()
        });
        assert_eq!(lines, vec!["x: [1.5, 2.5]".to_string()]);
        assert_eq!(read_first_shard::<f64>(client, &output), vec![1.5, 2.5]);

        // Batching under a staging parent keeps the print in the staged program, which prints the whole batch once
        // per execution.
        let staged = crate::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                let output =
                    batch(|item| item.print("x"), inputs[0].clone(), BatchAxis::new(0), BatchAxis::new(0), None);
                vec![output.unwrap()]
            },
            vec![input.r#type().into_owned()],
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let lowered = session.lower_program(staged.source_program().program(), 0, &Options::new(mesh)).unwrap();
        assert!(lowered.stable_hlo().contains("@ryft.print"), "{}", lowered.stable_hlo());
        let compiled = session.compile_program(&lowered).unwrap();
        let (outputs, lines) = with_captured_prints(|| session.execute_program(&compiled, vec![input]).unwrap());
        assert_eq!(lines, vec!["x: [1.5, 2.5]".to_string()]);
        assert_eq!(outputs.len(), 1);
        assert_eq!(read_first_shard::<f64>(client, &outputs[0]), vec![1.5, 2.5]);
    }

    #[test]
    fn test_read_print_buffer() {
        let ffi_buffer = |r#type: FfiBufferType, bytes: &[u8], dimensions: &[i64]| XLA_FFI_Buffer {
            struct_size: size_of::<XLA_FFI_Buffer>(),
            extension_start: std::ptr::null_mut(),
            data_type: unsafe { r#type.to_c_api() },
            data: bytes.as_ptr().cast_mut().cast(),
            rank: dimensions.len() as i64,
            dimensions: dimensions.as_ptr(),
        };

        let dimensions = [2, 2];
        let integers = ffi_buffer(FfiBufferType::I8, &[255, 2, 3, 4], &dimensions);
        let integers = unsafe { FfiBuffer::from_c_api(&integers) }.unwrap();
        assert_eq!(read_print_buffer(&integers, PrintBufferMemory::Host).unwrap().to_string(), "[[-1, 2], [3, 4]]");

        let dimensions = [2];
        let predicates = ffi_buffer(FfiBufferType::Predicate, &[0, 1], &dimensions);
        let predicates = unsafe { FfiBuffer::from_c_api(&predicates) }.unwrap();
        assert_eq!(read_print_buffer(&predicates, PrintBufferMemory::Host).unwrap().to_string(), "[false, true]");

        // Empty buffers may have a null data pointer and still render their outer dimensions.
        let dimensions = [2, 0];
        let empty = XLA_FFI_Buffer { data: std::ptr::null_mut(), ..ffi_buffer(FfiBufferType::F32, &[], &dimensions) };
        let empty = unsafe { FfiBuffer::from_c_api(&empty) }.unwrap();
        assert_eq!(read_print_buffer(&empty, PrintBufferMemory::Host).unwrap().to_string(), "[[], []]");

        // Raw sub-byte buffers lack the packing information needed to read their storage.
        let dimensions = [2];
        let packed = XLA_FFI_Buffer { data: std::ptr::null_mut(), ..ffi_buffer(FfiBufferType::I4, &[], &dimensions) };
        let packed = unsafe { FfiBuffer::from_c_api(&packed) }.unwrap();
        assert!(matches!(
            read_print_buffer(&packed, PrintBufferMemory::Host),
            Err(FfiError::InvalidArgument { message, .. })
                if message == "the `ryft.print` custom call requires a byte-aligned value buffer, got `i4`",
        ));
    }
}
