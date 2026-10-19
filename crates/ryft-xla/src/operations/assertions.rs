use std::num::NonZeroUsize;
use std::sync::OnceLock;

use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, AssertOperation, ComparisonDirection, DYNAMIC_SLICE_OPERATION_NAME, DataType, Dimension,
    DimensionType, DimensionVariable, EffectClass, GUARD_OPERATION_NAME, GuardOperation, Operation,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::{CustomCallApiVersion, CustomCallMemoryLayouts};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, DictionaryAttributeRef, LocationRef,
    Operation as MlirOperation, TensorTypeRef, Type, Value as MlirValue, ValueRef,
};
use ryft_pjrt::extensions::ffi::{
    FfiAttribute, FfiBuffer, FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits,
    FfiInput, FfiTypeId, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Handler,
};
use ryft_pjrt::{Client, Error};

use crate::experimental::lowering::types::{
    lower_physical_bound_value, lower_runtime_dimension_size_i64, lower_static_index_constants, lower_tensor_type,
};
use crate::experimental::lowering::{EffectTokens, current_or_new_token};
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::operations::manipulation::conversions::lower_cast_shape;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
use ryft_pjrt::extensions::ffi::FfiStream;

/// Emits one typed assertion callback and advances only the ordered-assertion chain.
pub(crate) fn lower_assertion_custom_call<'b, 'c: 'b, 't: 'c>(
    predicate: ValueRef<'b, 'c, 't>,
    observed_values: &[ValueRef<'b, 'c, 't>],
    backend_config: DictionaryAttributeRef<'c, 't>,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let input_token = current_or_new_token(EffectClass::OrderedAssertion, effect_tokens, block, location)?;
    let mut inputs = Vec::with_capacity(2 + observed_values.len());
    inputs.push(predicate);
    inputs.extend_from_slice(observed_values);
    inputs.push(input_token);
    let layouts = inputs
        .iter()
        .map(|input| {
            Ok(input
                .r#type()?
                .cast::<TensorTypeRef>()
                .map(|tensor| (0..tensor.dimensions().count()).rev().collect::<Vec<_>>())
                .unwrap_or_default())
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let layouts = layouts
        .iter()
        .any(|layout| !layout.is_empty())
        .then_some(CustomCallMemoryLayouts { operands: layouts, results: vec![Vec::new()] });
    let operation = block.append_operation(stable_hlo::custom_call(
        inputs.as_slice(),
        ASSERT_CUSTOM_CALL_TARGET,
        true,
        Some(backend_config.as_ref()),
        CustomCallApiVersion::TypedFfi,
        &[],
        layouts,
        &[],
        None,
        &[context.stable_hlo_token_type()?],
        location,
    )?)?;
    effect_tokens.set(
        EffectClass::OrderedAssertion,
        operation.result(0).expect("the assertion custom call should return one token result").as_ref(),
    );
    Ok(())
}

/// Emits an assertion with a literal message and named observations. Bounded element reports carry logical
/// extents. Diagnostic buffers retain their element types so unsigned values and floating-point values reach the
/// callback without lossy conversions.
pub(crate) fn lower_assert_to_custom_call<'b, 'c: 'b, 't: 'c>(
    actor: &str,
    message: &str,
    labels: &[String],
    failure_limit: Option<NonZeroUsize>,
    input_types: &[ArrayType],
    input_values: &[ValueRef<'b, 'c, 't>],
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    check_count!("input", input_values, labels.len() + 1, ProgramError);
    check_count!("input", input_types, input_values.len(), ProgramError);
    let mut attributes = vec![
        context.named_attribute(context.identifier(ASSERT_ACTOR_ATTRIBUTE), context.string_attribute(actor)),
        context.named_attribute(
            context.identifier(ASSERT_KIND_ATTRIBUTE),
            context.string_attribute(if failure_limit.is_some() { ASSERT_ELEMENTS_KIND } else { ASSERT_GENERIC_KIND }),
        ),
        context.named_attribute(context.identifier(ASSERT_MESSAGE_ATTRIBUTE), context.string_attribute(message)),
        context.named_attribute(
            context.identifier(ASSERT_LABEL_COUNT_ATTRIBUTE),
            context.string_attribute(labels.len().to_string().as_str()),
        ),
    ];
    for (index, label) in labels.iter().enumerate() {
        attributes.push(context.named_attribute(
            context.identifier(format!("label_{index}").as_str()),
            context.string_attribute(label.as_str()),
        ));
    }
    let mut arguments = input_values.to_vec();
    if let Some(limit) = failure_limit {
        // Use byte-wide diagnostic carriers rather than relying on backend sub-byte buffer packing.
        for (value, input_type) in arguments[1..].iter_mut().zip(&input_types[1..]) {
            let data_type = input_type.data_type();
            let carrier = match data_type {
                DataType::I1 | DataType::I2 | DataType::I4 => DataType::I8,
                DataType::U1 | DataType::U2 | DataType::U4 => DataType::U8,
                _ => continue,
            };
            let tensor_type = lower_tensor_type(&input_type.clone().with_data_type(carrier), context, location)?;
            let converted = block.append_operation(stable_hlo::convert(*value, tensor_type, location)?)?;
            *value = converted.result(0).unwrap().as_ref();
            if data_type == DataType::I1 {
                let signed = block.append_operation(stable_hlo::negate(*value, location)?)?;
                *value = signed.result(0).unwrap().as_ref();
            }
        }
        let rank = input_values[0].r#type()?.cast::<TensorTypeRef>().unwrap().dimensions().count();
        attributes.push(context.named_attribute(
            context.identifier(ASSERT_FAILURE_LIMIT_ATTRIBUTE),
            context.string_attribute(limit.get().to_string().as_str()),
        ));
        attributes.push(context.named_attribute(
            context.identifier(ASSERT_PREDICATE_RANK_ATTRIBUTE),
            context.string_attribute(rank.to_string().as_str()),
        ));
        for axis in 0..rank {
            arguments.push(lower_runtime_dimension_size_i64(input_values[0], axis, block, context, location)?);
        }
        // XLA custom calls require static physical carriers; the separately captured logical extents exclude padding.
        for (value, input_type) in arguments.iter_mut().zip(input_types) {
            let data_type = match input_type.data_type() {
                DataType::I1 | DataType::I2 | DataType::I4 => DataType::I8,
                DataType::U1 | DataType::U2 | DataType::U4 => DataType::U8,
                data_type => data_type,
            };
            *value = lower_physical_bound_value(
                *value,
                &input_type.clone().with_data_type(data_type),
                0.0,
                block,
                context,
                location,
            )?;
        }
    }
    lower_assertion_custom_call(
        arguments[0],
        &arguments[1..],
        context.dictionary_attribute(&attributes),
        effect_tokens,
        block,
        context,
        location,
    )
}

/// Builds the deliberately false `value < 0` predicate attached to diagnostic assertion custom calls whose safety
/// condition is evaluated host-side with checked operations: a nonnegative dimension value can never satisfy it, so
/// the callback always inspects the concrete observed operands, and the side-effecting call cannot be folded away by
/// a constant-true predicate.
pub(crate) fn lower_deliberately_false_assertion_predicate<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let zero = lower_static_index_constants(&[0], block, context, location)?[0];
    lower_compare_to_mlir(ComparisonDirection::LessThan, value, zero, block, location)
}

/// Validates the extent operands of a complete output signature against the static extents to which type inference
/// refined their declared dynamic axes. An exact operand of a shared dimension identity refines every occurrence of
/// that identity, so a symbolic operand of another occurrence must equal the refined extent at runtime. This emits one
/// assertion per distinct pair of operand value and expected extent, and none for operands whose exact extent already
/// proves it. Axes that remain dynamic need no check: distinct values that share a symbolic identity are equal by
/// construction, because each internal identity has a single defining value (i.e., `Region::type_identity_signature`
/// rejects a second definition) and entry refinements validate identities shared by boundary inputs. Returns the
/// conjunction of the emitted predicates, when there are any, so that foreign-call consumers can gate their
/// invocation on it.
///
/// # Parameters
///
///   - `name`: Name of the operation that owns the assertions, reported in their diagnostics.
///   - `declared_types`: Declared output types, which consume one extent operand per dynamic axis, in axis order.
///   - `input_values`: Compact extent operands of the declared dynamic axes.
///   - `input_types`: Dimension types of `input_values`.
///   - `output_types`: Refined output types, which must have the ranks of `declared_types`.
pub(crate) fn lower_output_extent_validation<'b, 'c: 'b, 't: 'c>(
    name: &str,
    declared_types: &[ArrayType],
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayIrType],
    output_types: &[ArrayType],
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Option<ValueRef<'b, 'c, 't>>, LoweringError> {
    let checks = output_extent_checks(declared_types, input_types, output_types)?;
    check_count!("input", input_values, input_types.len(), ProgramError);
    let mut checked = Vec::new();
    let mut validation = None;
    for (index, variable, extent) in checks {
        let value = input_values[index];
        if checked.contains(&(value, extent)) {
            continue;
        }
        checked.push((value, extent));
        let expected = lower_static_index_constants(&[extent], block, context, location)?.remove(0);
        let predicate = lower_compare_to_mlir(ComparisonDirection::Equal, value, expected, block, location)?;
        lower_assert_to_custom_call(
            name,
            &format!("input dimension `{variable}` expected {extent}"),
            &[variable.to_string()],
            None,
            &[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::I64)],
            &[predicate, value],
            effect_tokens,
            block,
            context,
            location,
        )?;
        validation = Some(match validation {
            None => predicate,
            Some(previous) => {
                block.append_operation(stable_hlo::and(previous, predicate, location)?)?.result(0).unwrap().as_ref()
            }
        });
    }
    Ok(validation)
}

/// Returns `true` if lowering an operation whose complete output signature declares `declared_types` emits runtime
/// assertions that validate its extent operands (refer to [`lower_output_extent_validation`]). Such assertions are
/// effects that lowering synthesizes, because the operation itself declares none, so the enclosing scopes must thread
/// their tokens (refer to [`lowered_effect_classes`](crate::operations::control_flow::lowered_effect_classes)). An
/// instruction with fewer inputs than the declared dynamic axes does not size them with extent operands (e.g., a
/// homogeneous array constructor), so lowering validates nothing for it.
///
/// # Parameters
///
///   - `declared_types`: Declared output types, which consume one extent operand per dynamic axis, in axis order.
///   - `input_types`: Types of all instruction inputs, which end with the extent operands of the declared dynamic axes.
///   - `output_types`: Refined output types, which correspond to `declared_types`.
pub(crate) fn validates_output_extents(
    declared_types: &[ArrayType],
    input_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
) -> Result<bool, LoweringError> {
    let dynamic_count = declared_types
        .iter()
        .flat_map(|r#type| r#type.shape().dimensions())
        .filter_map(Dimension::variable)
        .count();
    let Some(extent_offset) = input_types.len().checked_sub(dynamic_count) else {
        return Ok(false);
    };
    let output_types = output_types
        .iter()
        .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| LoweringError::Tracing(error.into()))?;
    Ok(!output_extent_checks(declared_types, &input_types[extent_offset..], &output_types)?.is_empty())
}

/// Returns the checks that [`lower_output_extent_validation`] emits before deduplicating them by operand value: for
/// each extent operand of a declared dynamic axis whose refined output extent its own type does not prove, its index
/// among the `input_types` of the extent operands, the declared identity of its axis, and the refined extent.
fn output_extent_checks<'d>(
    declared_types: &'d [ArrayType],
    input_types: &[ArrayIrType],
    output_types: &[ArrayType],
) -> Result<Vec<(usize, &'d DimensionVariable, usize)>, LoweringError> {
    check_count!("output", output_types, declared_types.len(), ProgramError);
    let dynamic_count = declared_types
        .iter()
        .flat_map(|r#type| r#type.shape().dimensions())
        .filter_map(Dimension::variable)
        .count();
    check_count!("input", input_types, dynamic_count, ProgramError);
    let mut inputs = input_types.iter().enumerate();
    let mut checks = Vec::new();
    let dimensions = declared_types
        .iter()
        .zip(output_types)
        .flat_map(|(declared, output)| declared.shape().dimensions().iter().zip(output.shape().dimensions()));
    for (declared, output) in dimensions {
        let Dimension::Dynamic(variable) = declared else { continue };
        let (index, input_type) = inputs.next().unwrap();
        let input_type =
            <&DimensionType>::try_from(input_type).map_err(|error| LoweringError::Tracing(error.into()))?;
        if let Dimension::Static(extent) = *output
            && input_type.extent() != Some(extent)
        {
            checks.push((index, variable, extent));
        }
    }
    Ok(checks)
}

/// Name of the typed XLA FFI custom call that reports failed runtime assertions.
pub(crate) const ASSERT_CUSTOM_CALL_TARGET: &str = "ryft.assert";

/// Backend-config attribute containing the canonical Ryft operation name that owns the assertion.
pub(crate) const ASSERT_ACTOR_ATTRIBUTE: &str = "actor";

/// Backend-config attribute containing kind-specific diagnostic detail such as bounds or a concatenation axis.
pub(crate) const ASSERT_DETAIL_ATTRIBUTE: &str = "detail";

/// Backend-config attribute containing the diagnostic name of the first observed extent.
pub(crate) const ASSERT_LEFT_ATTRIBUTE: &str = "left";

/// Optional backend-config attribute containing the diagnostic name of the second observed extent.
pub(crate) const ASSERT_RIGHT_ATTRIBUTE: &str = "right";

/// Backend-config attribute identifying the formatting contract used for a failed assertion.
pub(crate) const ASSERT_KIND_ATTRIBUTE: &str = "kind";

/// Formatting kind for a general assertion with named scalar observations.
pub(crate) const ASSERT_GENERIC_KIND: &str = "generic";

/// Formatting kind for bounded reporting of failed assertion elements.
pub(crate) const ASSERT_ELEMENTS_KIND: &str = "elements";

/// Backend-config attribute containing the maximum number of failed elements to report.
pub(crate) const ASSERT_FAILURE_LIMIT_ATTRIBUTE: &str = "failure_limit";

/// Backend-config attribute containing the rank of the predicate and of its logical extents.
pub(crate) const ASSERT_PREDICATE_RANK_ATTRIBUTE: &str = "predicate_rank";

/// Backend-config attribute containing a general assertion's literal diagnostic message.
pub(crate) const ASSERT_MESSAGE_ATTRIBUTE: &str = "message";

/// Backend-config attribute containing the number of `label_N` observation labels.
pub(crate) const ASSERT_LABEL_COUNT_ATTRIBUTE: &str = "label_count";

/// Formatting kind used for explicit dimension bounds.
pub(crate) const ASSERT_BOUNDS_KIND: &str = "bounds";

/// Formatting kind used for a dynamic concatenation result-extent check.
pub(crate) const ASSERT_CONCATENATE_KIND: &str = "concatenate";

/// Formatting kind used for a padding result-extent check.
pub(crate) const ASSERT_PAD_KIND: &str = "pad";

/// Formatting kind used for a reshape element-count check.
pub(crate) const ASSERT_RESHAPE_KIND: &str = "reshape";

/// Formatting kind used for a dynamic-shape-slice runtime bounds check.
// Keep this registered wire name stable when renaming the public operation family.
pub(crate) const ASSERT_DYNAMIC_SHAPE_SLICE_KIND: &str = "dynamic_shape_slice";

/// Formatting kind used for checked dimension addition.
pub(crate) const ASSERT_ADD_KIND: &str = "add";

/// Formatting kind used for checked dimension subtraction.
pub(crate) const ASSERT_SUB_KIND: &str = "sub";

/// Formatting kind used for checked dimension multiplication.
pub(crate) const ASSERT_MUL_KIND: &str = "mul";

/// Formatting kind used for checked dimension exponentiation.
pub(crate) const ASSERT_POW_KIND: &str = "pow";

/// Formatting kind used for a nonzero floor-division divisor.
pub(crate) const ASSERT_DIV_KIND: &str = "div";

/// Formatting kind used for a nonzero remainder divisor.
pub(crate) const ASSERT_REM_KIND: &str = "rem";

/// CUDA Driver API success result.
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
const CUDA_SUCCESS: i32 = 0;

#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
#[link(name = "cuda")]
unsafe extern "C" {
    /// Enqueues one device-to-host copy on a CUDA stream.
    fn cuMemcpyDtoHAsync_v2(
        destination: *mut std::ffi::c_void,
        source: u64,
        byte_count: usize,
        stream: FfiStream,
    ) -> i32;

    /// Waits for all previously enqueued work on a CUDA stream.
    fn cuStreamSynchronize(stream: FfiStream) -> i32;
}

/// Memory location of the scalar buffers passed to one assertion callback.
#[derive(Copy, Clone)]
enum AssertionBufferMemory {
    /// CPU buffers that the callback can read directly.
    Host,

    /// CUDA device buffers read back through the invocation's stream.
    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    Cuda(FfiStream),
}

/// Registers the [`ASSERT_CUSTOM_CALL_TARGET`] handler with the plugin backing `client`.
///
/// Registration is lazy and process-global because XLA's FFI registry rejects duplicate target registrations. The
/// CPU handler reads scalar operands directly. The CUDA handler copies them through the invocation's CUDA stream
/// before applying the same validation and diagnostic logic.
pub(crate) fn ensure_assertion_handler_registered(client: &Client<'_>) -> Result<(), Error> {
    let platform_name = client.platform_name()?.into_owned();
    if platform_name.eq_ignore_ascii_case("cpu") {
        static CPU_ASSERTION_HANDLER_REGISTRATION: OnceLock<Result<(), Error>> = OnceLock::new();
        return CPU_ASSERTION_HANDLER_REGISTRATION
            .get_or_init(|| {
                client.register_ffi_handler(
                    ASSERT_CUSTOM_CALL_TARGET,
                    platform_name,
                    FfiHandler::from(assertion_handler as XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                )
            })
            .clone();
    }
    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    if platform_name.eq_ignore_ascii_case("cuda") {
        static CUDA_ASSERTION_HANDLER_REGISTRATION: OnceLock<Result<(), Error>> = OnceLock::new();
        return CUDA_ASSERTION_HANDLER_REGISTRATION
            .get_or_init(|| {
                client.register_ffi_handler(
                    ASSERT_CUSTOM_CALL_TARGET,
                    platform_name,
                    FfiHandler::from(cuda_assertion_handler as XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                )
            })
            .clone();
    }
    Err(Error::unimplemented(format!(
        "compiled runtime assertions are not supported on XLA platform `{platform_name}`",
    )))
}

/// XLA FFI handler for host-resident assertion operands.
unsafe extern "C" fn assertion_handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    unsafe { assertion_handler_for_memory(call_frame, |_| Ok(AssertionBufferMemory::Host)) }
}

/// XLA FFI handler for CUDA-resident assertion operands.
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
unsafe extern "C" fn cuda_assertion_handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    unsafe {
        assertion_handler_for_memory(call_frame, |call_frame| {
            Ok(AssertionBufferMemory::Cuda(call_frame.context()?.stream()?))
        })
    }
}

/// Decodes one XLA FFI invocation and evaluates it using the memory location returned by `memory`.
unsafe fn assertion_handler_for_memory(
    call_frame: *mut XLA_FFI_CallFrame,
    memory: impl FnOnce(&FfiCallFrame<'_>) -> Result<AssertionBufferMemory, FfiError>,
) -> *mut XLA_FFI_Error {
    let _callback_guard = crate::experimental::domains::EffectCallbackGuard::enter();
    // SAFETY: XLA owns the call frame for this invocation. All access is localized in `FfiCallFrame` and the checked
    // scalar-buffer readers below.
    unsafe {
        match FfiCallFrame::from_c_api(call_frame) {
            Err(_) => std::ptr::null_mut(),
            Ok(call_frame) if call_frame.register_metadata(FfiTypeId::default()) => std::ptr::null_mut(),
            Ok(call_frame) if call_frame.stage() != FfiExecutionStage::Execution => std::ptr::null_mut(),
            Ok(call_frame) => match call_frame.api() {
                Err(_) => std::ptr::null_mut(),
                Ok(api) => {
                    match memory(&call_frame).and_then(|memory| handle_assertion_call_frame(&call_frame, memory)) {
                        Ok(()) => std::ptr::null_mut(),
                        Err(error) => error.to_c_api(api),
                    }
                }
            },
        }
    }
}

/// Copies `byte_count` bytes from one CUDA device allocation into host memory and waits for completion. The copy is
/// enqueued on the invoking FFI call's `stream`, so it observes every device write ordered before the custom call.
/// This is shared by the CUDA assertion handler and the CUDA print handler in [`crate::operations::debugging`].
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
pub(crate) fn copy_cuda_bytes(
    source: *mut std::ffi::c_void,
    destination: *mut std::ffi::c_void,
    byte_count: usize,
    stream: FfiStream,
) -> Result<(), FfiError> {
    let copy_result = unsafe { cuMemcpyDtoHAsync_v2(destination, source as usize as u64, byte_count, stream) };
    if copy_result != CUDA_SUCCESS {
        return Err(FfiError::internal(format!(
            "CUDA device-to-host operand copy failed with driver error {copy_result}"
        )));
    }
    let synchronize_result = unsafe { cuStreamSynchronize(stream) };
    if synchronize_result != CUDA_SUCCESS {
        return Err(FfiError::internal(format!(
            "CUDA operand copy stream synchronization failed with driver error {synchronize_result}",
        )));
    }
    Ok(())
}

/// Copies a group of buffers, draining queued transfers even when a later enqueue fails.
///
/// The destinations remain allocated if synchronization fails because completion of asynchronous writes cannot
/// then be established. This exceptional leak prevents a driver failure from becoming a host use-after-free.
#[cfg(any(test, feature = "cuda-12", feature = "cuda-13"))]
fn copy_buffer_group<E, S>(
    sources: &[(*mut std::ffi::c_void, usize)],
    mut enqueue: E,
    synchronize: S,
) -> Result<Vec<Vec<u8>>, FfiError>
where
    E: FnMut(*mut std::ffi::c_void, *mut std::ffi::c_void, usize) -> Result<(), FfiError>,
    S: FnOnce() -> Result<(), FfiError>,
{
    let mut destinations = sources.iter().map(|(_, size)| vec![0; *size]).collect::<Vec<_>>();
    if sources.iter().all(|(_, size)| *size == 0) {
        return Ok(destinations);
    }
    let copied = sources.iter().zip(&mut destinations).try_for_each(|((source, size), destination)| {
        if *size == 0 { Ok(()) } else { enqueue(*source, destination.as_mut_ptr().cast(), *size) }
    });
    if let Err(error) = synchronize() {
        std::mem::forget(destinations);
        return Err(error);
    }
    copied?;
    Ok(destinations)
}

/// Reads validated buffers together, using a single stream synchronization for CUDA memory.
fn buffer_bytes(
    sources: &[(*mut std::ffi::c_void, usize)],
    memory: AssertionBufferMemory,
) -> Result<Vec<Vec<u8>>, FfiError> {
    if sources.iter().any(|(source, size)| source.is_null() && *size != 0) {
        return Err(FfiError::invalid_argument(format!(
            "encountered a null scalar buffer in `{ASSERT_CUSTOM_CALL_TARGET}`",
        )));
    }
    match memory {
        AssertionBufferMemory::Host => Ok(sources
            .iter()
            .map(|(source, size)| {
                // SAFETY: Callers validate buffer types and sizes, and XLA owns these invocation-scoped addresses.
                if *size == 0 {
                    Vec::new()
                } else {
                    unsafe { std::slice::from_raw_parts(source.cast::<u8>(), *size).to_vec() }
                }
            })
            .collect::<Vec<_>>()),
        #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
        AssertionBufferMemory::Cuda(stream) => copy_buffer_group(
            sources,
            |source, destination, size| {
                let status = unsafe { cuMemcpyDtoHAsync_v2(destination, source as usize as u64, size, stream) };
                if status == CUDA_SUCCESS {
                    Ok(())
                } else {
                    Err(FfiError::internal(format!(
                        "CUDA device-to-host operand copy failed with driver error {status}",
                    )))
                }
            },
            || {
                let status = unsafe { cuStreamSynchronize(stream) };
                if status == CUDA_SUCCESS {
                    Ok(())
                } else {
                    Err(FfiError::internal(format!(
                        "CUDA operand copy stream synchronization failed with driver error {status}",
                    )))
                }
            },
        ),
    }
}

/// Reads rank-zero signed 64-bit extent buffers together.
fn scalar_extents(buffers: &[FfiBuffer<'_>], memory: AssertionBufferMemory) -> Result<Vec<i64>, FfiError> {
    let sources = buffers
        .iter()
        .map(|buffer| {
            if buffer.element_type() != FfiBufferType::I64 || buffer.rank() != 0 {
                return Err(FfiError::invalid_argument(format!(
                    "expected the `{ASSERT_CUSTOM_CALL_TARGET}` observed extent to be a rank-zero i64 buffer",
                )));
            }
            Ok((unsafe { buffer.data() }, size_of::<i64>()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(buffer_bytes(&sources, memory)?
        .into_iter()
        .map(|bytes| i64::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>())
}

/// Reads `BYTE_COUNT` bytes from a validated scalar assertion buffer.
fn scalar_bytes<const BYTE_COUNT: usize>(
    buffer: &FfiBuffer<'_>,
    memory: AssertionBufferMemory,
) -> Result<[u8; BYTE_COUNT], FfiError> {
    let source = unsafe { buffer.data() };
    if source.is_null() {
        return Err(FfiError::invalid_argument(format!(
            "encountered a null scalar buffer in `{ASSERT_CUSTOM_CALL_TARGET}`",
        )));
    }
    match memory {
        AssertionBufferMemory::Host => {
            let mut bytes = [0; BYTE_COUNT];
            // SAFETY: Scalar readers validate the type and rank before requesting this element's bytes.
            unsafe { std::ptr::copy_nonoverlapping(source.cast::<u8>(), bytes.as_mut_ptr(), BYTE_COUNT) };
            Ok(bytes)
        }
        #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
        AssertionBufferMemory::Cuda(_) => {
            let bytes = buffer_bytes(&[(source, BYTE_COUNT)], memory)?;
            Ok(bytes.into_iter().next().unwrap().try_into().unwrap())
        }
    }
}

/// Decodes and evaluates one assertion call frame.
fn handle_assertion_call_frame(call_frame: &FfiCallFrame<'_>, memory: AssertionBufferMemory) -> Result<(), FfiError> {
    let mut buffers = Vec::new();
    for input in call_frame.inputs() {
        let FfiInput::Buffer { buffer } = input?;
        if buffer.element_type() != FfiBufferType::Token {
            buffers.push(buffer);
        }
    }
    if buffers.is_empty() {
        return Err(FfiError::invalid_argument(format!(
            "expected the `{ASSERT_CUSTOM_CALL_TARGET}` custom call to receive a predicate"
        )));
    }
    let actor = string_attribute(call_frame, ASSERT_ACTOR_ATTRIBUTE)?;
    let kind = string_attribute(call_frame, ASSERT_KIND_ATTRIBUTE)?;
    if kind == ASSERT_ELEMENTS_KIND {
        return handle_assertion_elements(call_frame, &buffers, memory);
    }
    if kind == ASSERT_GENERIC_KIND {
        let count = string_attribute(call_frame, ASSERT_LABEL_COUNT_ATTRIBUTE)?
            .parse::<usize>()
            .map_err(|_| FfiError::invalid_argument("invalid assertion diagnostic label count"))?;
        if buffers.len().checked_sub(1) != Some(count) {
            return Err(FfiError::invalid_argument("assertion diagnostic label count does not match its inputs"));
        }
        let message = string_attribute(call_frame, ASSERT_MESSAGE_ATTRIBUTE)?;
        let labels = (0..count)
            .map(|index| string_attribute(call_frame, &format!("label_{index}")))
            .collect::<Result<Vec<_>, _>>()?;
        if scalar_predicate(&buffers[0], memory)? {
            return Ok(());
        }
        let observations =
            labels.into_iter().map(str::to_owned).zip(scalar_observations(&buffers[1..], memory)?).collect();
        return Err(FfiError::invalid_argument(
            ryft_core::AssertionError::Failed { message: message.to_owned(), observations }.to_string(),
        ));
    }
    if kind == ASSERT_RESHAPE_KIND {
        let input_rank = string_attribute(call_frame, ASSERT_DETAIL_ATTRIBUTE)?
            .parse::<usize>()
            .map_err(|_| FfiError::invalid_argument("invalid reshape assertion input rank"))?;
        let extents = scalar_extents(&buffers[1..], memory)?;
        if input_rank > extents.len() {
            return Err(FfiError::invalid_argument("reshape assertion input rank exceeds the number of extents"));
        }
        let (input, output) = extents.split_at(input_rank);
        return validate_reshape(actor, input, output).map_err(FfiError::invalid_argument);
    }
    if buffers.len() < 2 {
        return Err(FfiError::invalid_argument("assertion is missing observed extents"));
    }
    if kind == ASSERT_PAD_KIND {
        if buffers.len() != 3 {
            return Err(FfiError::invalid_argument("padding assertion requires input and output extents"));
        }
        let detail = string_attribute(call_frame, ASSERT_DETAIL_ATTRIBUTE)?;
        let fields = detail.split(':').collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err(FfiError::invalid_argument("invalid padding assertion configuration"));
        }
        let axis = fields[0].parse::<usize>().map_err(|_| FfiError::invalid_argument("invalid padding axis"))?;
        let low = fields[1].parse::<i64>().map_err(|_| FfiError::invalid_argument("invalid low padding"))?;
        let high = fields[2].parse::<i64>().map_err(|_| FfiError::invalid_argument("invalid high padding"))?;
        let interior =
            fields[3].parse::<usize>().map_err(|_| FfiError::invalid_argument("invalid interior padding"))?;
        let extents = scalar_extents(&buffers[1..], memory)?;
        return validate_pad(axis, extents[0], extents[1], low, high, interior).map_err(FfiError::invalid_argument);
    }
    if kind == ASSERT_CONCATENATE_KIND {
        // Concatenation is variadic, so its callback recomputes the checked sum from every input extent instead of
        // relying on a fixed-arity predicate produced in StableHLO.
        if buffers.len() < 3 {
            return Err(FfiError::invalid_argument(format!(
                "expected the `{ASSERT_CUSTOM_CALL_TARGET}` concatenate assertion to receive a result extent and at \
                 least one input extent"
            )));
        }
        let extents = scalar_extents(&buffers[1..], memory)?;
        let actual = extents[0];
        let input_extents = &extents[1..];
        let axis = string_attribute(call_frame, ASSERT_DETAIL_ATTRIBUTE)?;
        return validate_concatenate(actor, axis, actual, input_extents).map_err(FfiError::invalid_argument);
    }
    if kind == ASSERT_DYNAMIC_SHAPE_SLICE_KIND {
        if buffers.len() != 4 {
            return Err(FfiError::invalid_argument(format!(
                "expected the `{ASSERT_CUSTOM_CALL_TARGET}` dynamic-shape-slice assertion to receive the input \
                 extent, start, and size"
            )));
        }
        let detail = string_attribute(call_frame, ASSERT_DETAIL_ATTRIBUTE)?;
        let (axis, stride) = detail.split_once(':').ok_or_else(|| {
            FfiError::invalid_argument(format!(
                "expected the `{ASSERT_CUSTOM_CALL_TARGET}` dynamic-shape-slice detail to contain `axis:stride`"
            ))
        })?;
        let axis = axis.parse::<usize>().map_err(|_| {
            FfiError::invalid_argument(format!(
                "expected the `{ASSERT_CUSTOM_CALL_TARGET}` dynamic-shape-slice axis to be an unsigned integer"
            ))
        })?;
        let stride = stride.parse::<usize>().map_err(|_| {
            FfiError::invalid_argument(format!(
                "expected the `{ASSERT_CUSTOM_CALL_TARGET}` dynamic-shape-slice stride to be an unsigned integer"
            ))
        })?;
        let extents = scalar_extents(&buffers[1..], memory)?;
        let [input_size, start, size] = extents[..] else { unreachable!() };
        return validate_dynamic_slice_with_dimensions(axis, stride, input_size, start, size)
            .map_err(FfiError::invalid_argument);
    }

    let arithmetic = matches!(
        kind,
        ASSERT_ADD_KIND | ASSERT_SUB_KIND | ASSERT_MUL_KIND | ASSERT_POW_KIND | ASSERT_DIV_KIND | ASSERT_REM_KIND
    );
    if arithmetic {
        if buffers.len() != 3 {
            return Err(FfiError::invalid_argument(format!(
                "expected the `{ASSERT_CUSTOM_CALL_TARGET}` arithmetic assertion to receive two extents"
            )));
        }
        let left_name = string_attribute(call_frame, ASSERT_LEFT_ATTRIBUTE)?;
        let right_name = string_attribute(call_frame, ASSERT_RIGHT_ATTRIBUTE)?;
        let extents = scalar_extents(&buffers[1..], memory)?;
        let [left, right] = extents[..] else { unreachable!() };
        return validate_arithmetic(kind, left_name, left, right_name, right).map_err(FfiError::invalid_argument);
    }

    if kind != ASSERT_BOUNDS_KIND {
        return Err(FfiError::invalid_argument(format!(
            "unsupported `{ASSERT_KIND_ATTRIBUTE}` value `{kind}` for `{ASSERT_CUSTOM_CALL_TARGET}`"
        )));
    }
    if buffers.len() != 2 {
        return Err(FfiError::invalid_argument("dimension conversion assertion requires one observed extent"));
    }
    if scalar_predicate(&buffers[0], memory)? {
        return Ok(());
    }
    let left_name = string_attribute(call_frame, ASSERT_LEFT_ATTRIBUTE)?;
    let left = scalar_i64(&buffers[1], memory)?;
    let bounds = string_attribute(call_frame, ASSERT_DETAIL_ATTRIBUTE)?;
    Err(FfiError::invalid_argument(format!(
        "`{actor}` failed: input dimension `{left_name}` = {left} is outside its declared bounds {bounds}"
    )))
}

/// Returns a checked physical element count for an assertion buffer.
fn buffer_element_count(buffer: &FfiBuffer<'_>) -> Result<usize, FfiError> {
    buffer.dimensions().iter().try_fold(1_usize, |count, extent| {
        usize::try_from(*extent)
            .ok()
            .and_then(|extent| count.checked_mul(extent))
            .ok_or_else(|| FfiError::invalid_argument("invalid assertion buffer dimensions"))
    })
}

/// Computes the row-major offset of a logical element in a physical assertion buffer.
fn element_offset(index: &[usize], dimensions: &[i64]) -> Result<usize, FfiError> {
    if index.len() != dimensions.len() {
        return Err(FfiError::invalid_argument("assertion element index rank does not match its buffer"));
    }
    index.iter().zip(dimensions).try_fold(0_usize, |offset, (index, extent)| {
        let extent =
            usize::try_from(*extent).map_err(|_| FfiError::invalid_argument("negative assertion buffer extent"))?;
        if *index >= extent {
            return Err(FfiError::invalid_argument("assertion element index exceeds its physical buffer"));
        }
        offset
            .checked_mul(extent)
            .and_then(|offset| offset.checked_add(*index))
            .ok_or_else(|| FfiError::invalid_argument("assertion buffer offset overflows"))
    })
}

/// Reports a bounded set of failing logical elements, reading observations only after a failure is found.
fn handle_assertion_elements(
    call_frame: &FfiCallFrame<'_>,
    buffers: &[FfiBuffer<'_>],
    memory: AssertionBufferMemory,
) -> Result<(), FfiError> {
    let count = string_attribute(call_frame, ASSERT_LABEL_COUNT_ATTRIBUTE)?
        .parse::<usize>()
        .map_err(|_| FfiError::invalid_argument("invalid assertion diagnostic label count"))?;
    let rank = string_attribute(call_frame, ASSERT_PREDICATE_RANK_ATTRIBUTE)?
        .parse::<usize>()
        .map_err(|_| FfiError::invalid_argument("invalid assertion predicate rank"))?;
    let limit = string_attribute(call_frame, ASSERT_FAILURE_LIMIT_ATTRIBUTE)?
        .parse::<std::num::NonZeroUsize>()
        .map_err(|_| FfiError::invalid_argument("invalid assertion failure limit"))?
        .get();
    if count.checked_add(rank).and_then(|count| count.checked_add(1)) != Some(buffers.len()) {
        return Err(FfiError::invalid_argument("assertion element metadata does not match its inputs"));
    }
    let predicate = &buffers[0];
    if predicate.element_type() != FfiBufferType::Predicate || predicate.rank() != rank {
        return Err(FfiError::invalid_argument("assertion predicate has an invalid type or rank"));
    }
    let message = string_attribute(call_frame, ASSERT_MESSAGE_ATTRIBUTE)?;
    let labels = (0..count)
        .map(|index| string_attribute(call_frame, &format!("label_{index}")))
        .collect::<Result<Vec<_>, _>>()?;
    let mut sources = vec![(unsafe { predicate.data() }, buffer_element_count(predicate)?)];
    for buffer in &buffers[1 + count..] {
        if buffer.element_type() != FfiBufferType::I64 || buffer.rank() != 0 {
            return Err(FfiError::invalid_argument("assertion logical extents must be rank-zero i64 buffers"));
        }
        sources.push((unsafe { buffer.data() }, size_of::<i64>()));
    }
    let bytes = buffer_bytes(&sources, memory)?;
    let extents = bytes[1..]
        .iter()
        .map(|bytes| {
            usize::try_from(i64::from_ne_bytes(bytes.as_slice().try_into().unwrap()))
                .map_err(|_| FfiError::invalid_argument("negative assertion logical extent"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (logical, physical) in extents.iter().zip(predicate.dimensions()) {
        if *logical > usize::try_from(*physical).unwrap() {
            return Err(FfiError::invalid_argument("assertion logical extent exceeds its physical buffer"));
        }
    }
    let observations = &buffers[1..1 + count];
    let observation_sources = observations
        .iter()
        .map(|buffer| {
            if buffer.rank() != 0 {
                if buffer.rank() != rank
                    || extents.iter().zip(buffer.dimensions()).any(|(logical, physical)| {
                        usize::try_from(*physical).map_or(true, |physical| *logical > physical)
                    })
                {
                    return Err(FfiError::invalid_argument(
                        "assertion observation shape does not contain its logical elements",
                    ));
                }
            }
            let size = buffer_element_count(buffer)?
                .checked_mul(observation_byte_count(buffer.element_type())?)
                .ok_or_else(|| FfiError::invalid_argument("assertion observation byte count overflows"))?;
            Ok((unsafe { buffer.data() }, size))
        })
        .collect::<Result<Vec<_>, FfiError>>()?;
    let element_count = extents
        .iter()
        .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| FfiError::invalid_argument("assertion logical element count overflows"))?;
    let mut failed = Vec::new();
    let mut failure_count = 0;
    let mut index = vec![0; rank];
    for element in 0..element_count {
        let mut remaining = element;
        for axis in (0..rank).rev() {
            index[axis] = remaining % extents[axis];
            remaining /= extents[axis];
        }
        if bytes[0][element_offset(&index, predicate.dimensions())?] == 0 {
            failure_count += 1;
            if failed.len() < limit {
                failed.push(index.clone());
            }
        }
    }
    if failure_count == 0 {
        return Ok(());
    }
    let bytes = buffer_bytes(&observation_sources, memory)?;
    let failures = failed
        .into_iter()
        .map(|index| {
            let observations = labels
                .iter()
                .zip(observations)
                .zip(&bytes)
                .map(|((label, buffer), bytes)| {
                    let width = observation_byte_count(buffer.element_type())?;
                    let offset =
                        if buffer.rank() == 0 { 0 } else { element_offset(&index, buffer.dimensions())? } * width;
                    Ok((
                        (*label).to_owned(),
                        format_observation(buffer.element_type(), &bytes[offset..offset + width])?,
                    ))
                })
                .collect::<Result<Vec<_>, FfiError>>()?;
            Ok(ryft_core::AssertionFailure { index, observations })
        })
        .collect::<Result<Vec<_>, FfiError>>()?;
    let omitted = failure_count - failures.len();
    Err(FfiError::invalid_argument(
        ryft_core::AssertionError::FailedElements { message: message.to_owned(), failures, omitted }.to_string(),
    ))
}

/// Evaluates one checked dimension-arithmetic predicate and returns its eager-compatible diagnostic on failure.
fn validate_arithmetic(kind: &str, left_name: &str, left: i64, right_name: &str, right: i64) -> Result<(), String> {
    // This deliberately duplicates `ryft-core`'s private `checked_power` rather than widening that helper's
    // visibility: the eager helper computes over `usize` extents while this one validates the signed 64-bit values
    // that cross the FFI boundary, and the wording parity between the two paths is pinned by
    // `test_validate_arithmetic_matches_eager_checked_diagnostics` below.
    let checked_power = |mut base: i64, mut exponent: i64| {
        if base < 0 || exponent < 0 {
            return None;
        }
        let mut result = 1_i64;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.checked_mul(base)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.checked_mul(base)?;
            }
        }
        Some(result)
    };
    let valid = match kind {
        ASSERT_ADD_KIND => left.checked_add(right).is_some(),
        ASSERT_SUB_KIND => left >= right,
        ASSERT_MUL_KIND => left.checked_mul(right).is_some(),
        ASSERT_POW_KIND => checked_power(left, right).is_some(),
        ASSERT_DIV_KIND | ASSERT_REM_KIND => right > 0,
        _ => return Err(format!("unsupported arithmetic assertion kind `{kind}`")),
    };
    if valid {
        return Ok(());
    }
    // Static types label operands by their literal extents. Report operand roles instead, retaining meaningful
    // symbolic names as additional context without making the observed values look like numeric self-assignments.
    let observation = |role: &str, name: &str, value: i64| {
        if name == role || name.parse::<i64>().is_ok() {
            format!("{role}={value}")
        } else {
            format!("{role} (`{name}`)={value}")
        }
    };
    let observations = format!("{}, {}", observation("left", left_name, left), observation("right", right_name, right));
    Err(match kind {
        ASSERT_ADD_KIND | ASSERT_MUL_KIND | ASSERT_POW_KIND => {
            format!("dimension arithmetic overflow in `dimension_{kind}`; got {observations}",)
        }
        ASSERT_SUB_KIND => format!("`dimension_sub` requires left >= right; got {observations}"),
        ASSERT_DIV_KIND | ASSERT_REM_KIND => {
            format!("`dimension_{kind}` requires right > 0; got {observations}")
        }
        _ => unreachable!(),
    })
}

/// Validates one padding result extent using wide signed arithmetic before narrowing to a native size carrier.
fn validate_pad(axis: usize, input: i64, output: i64, low: i64, high: i64, interior: usize) -> Result<(), String> {
    if input < 0 || output < 0 {
        return Err(format!("`pad` dimensions must be nonnegative on axis {axis}"));
    }
    let expected = i128::from(input.saturating_sub(1).max(0))
        .checked_mul(interior as i128)
        .and_then(|value| value.checked_add(i128::from(input)))
        .and_then(|value| value.checked_add(i128::from(low)))
        .and_then(|value| value.checked_add(i128::from(high)))
        .ok_or_else(|| format!("`pad` result extent arithmetic overflows on axis {axis}"))?;
    if expected < 0 || expected > i128::from(i32::MAX) {
        return Err(format!("`pad` result extent {expected} is outside the native dimension range on axis {axis}"));
    }
    if expected != i128::from(output) {
        return Err(format!("`pad` result extent on axis {axis} must equal {expected} but got {output}"));
    }
    Ok(())
}

/// Validates equal logical element counts before a native reshape consumes its inputs.
fn validate_reshape(actor: &str, input: &[i64], output: &[i64]) -> Result<(), String> {
    let product = |extents: &[i64]| {
        if extents.iter().any(|extent| *extent < 0) {
            return Err(format!("`{actor}` dimensions must be nonnegative"));
        }
        // A zero-sized axis makes the product zero even if other factors would overflow independently.
        if extents.contains(&0) {
            return Ok(0_i64);
        }
        extents
            .iter()
            .try_fold(1_i64, |product, extent| product.checked_mul(*extent))
            .ok_or_else(|| format!("`{actor}` element count overflows the portable dimension range"))
    };
    let input_count = product(input)?;
    let output_count = product(output)?;
    if input_count == output_count {
        Ok(())
    } else {
        Err(format!("`{actor}` changes the number of elements from {input_count} to {output_count}"))
    }
}

/// Validates that `actual` equals the checked sum of `input_extents` for one concatenation axis.
fn validate_concatenate(actor: &str, axis: &str, actual: i64, input_extents: &[i64]) -> Result<(), String> {
    let expected = input_extents
        .iter()
        .try_fold(0_i64, |sum, extent| sum.checked_add(*extent))
        .ok_or_else(|| format!("`{actor}` input axis {axis} extent sum overflows the portable dimension range"))?;
    if expected == actual {
        Ok(())
    } else {
        Err(format!(
            "`{actor}` result extent must equal the sum of input axis {axis} extents; expected {expected} but got \
             {actual}",
        ))
    }
}

/// Validates one runtime dynamic-shape-slice axis using the same checked limit calculation as eager execution.
fn validate_dynamic_slice_with_dimensions(
    axis: usize,
    stride: usize,
    input_size: i64,
    start: i64,
    size: i64,
) -> Result<(), String> {
    let stride = i64::try_from(stride).map_err(|_| {
        format!(
            "`{DYNAMIC_SLICE_OPERATION_NAME}` stride is outside the portable dimension range on axis \
                 {axis}",
        )
    })?;
    let span = if size == 0 {
        0
    } else {
        size.checked_sub(1)
            .and_then(|size| size.checked_mul(stride))
            .and_then(|span| span.checked_add(1))
            .ok_or_else(|| format!("`{DYNAMIC_SLICE_OPERATION_NAME}` span overflows usize on axis {axis}"))?
    };
    let limit = start
        .checked_add(span)
        .ok_or_else(|| format!("`{DYNAMIC_SLICE_OPERATION_NAME}` limit overflows usize on axis {axis}"))?;
    if limit > input_size {
        return Err(format!(
            "`{DYNAMIC_SLICE_OPERATION_NAME}` limit {limit} exceeds input axis {axis} extent {input_size}",
        ));
    }
    Ok(())
}

/// Returns one required string attribute from `call_frame`.
fn string_attribute<'o>(call_frame: &FfiCallFrame<'o>, expected_name: &str) -> Result<&'o str, FfiError> {
    for attribute in call_frame.attributes() {
        let (name, attribute) = attribute?;
        if name == expected_name {
            return match attribute {
                FfiAttribute::String { string } => Ok(string),
                _ => Err(FfiError::invalid_argument(format!(
                    "expected the `{expected_name}` attribute of `{ASSERT_CUSTOM_CALL_TARGET}` to be a string"
                ))),
            };
        }
    }
    Err(FfiError::invalid_argument(format!(
        "missing required `{expected_name}` attribute in the `{ASSERT_CUSTOM_CALL_TARGET}` custom call"
    )))
}

/// Reads one rank-zero predicate buffer.
fn scalar_predicate(buffer: &FfiBuffer<'_>, memory: AssertionBufferMemory) -> Result<bool, FfiError> {
    if buffer.element_type() != FfiBufferType::Predicate || buffer.rank() != 0 {
        return Err(FfiError::invalid_argument(format!(
            "expected the `{ASSERT_CUSTOM_CALL_TARGET}` predicate to be a rank-zero predicate buffer"
        )));
    }
    Ok(scalar_bytes::<1>(buffer, memory)?[0] != 0)
}

/// Returns the byte width of a supported assertion observation.
fn observation_byte_count(r#type: FfiBufferType) -> Result<usize, FfiError> {
    match r#type {
        FfiBufferType::Predicate
        | FfiBufferType::I1
        | FfiBufferType::I2
        | FfiBufferType::I4
        | FfiBufferType::U1
        | FfiBufferType::U2
        | FfiBufferType::U4
        | FfiBufferType::I8
        | FfiBufferType::U8 => Ok(1),
        FfiBufferType::I16 | FfiBufferType::U16 | FfiBufferType::F16 | FfiBufferType::BF16 => Ok(2),
        FfiBufferType::I32 | FfiBufferType::U32 | FfiBufferType::F32 => Ok(4),
        FfiBufferType::I64 | FfiBufferType::U64 | FfiBufferType::F64 => Ok(8),
        _ => Err(FfiError::invalid_argument(format!("unsupported assertion observation type `{type}`"))),
    }
}

/// Reads and renders scalar observations together without narrowing integers or changing float precision.
fn scalar_observations(buffers: &[FfiBuffer<'_>], memory: AssertionBufferMemory) -> Result<Vec<String>, FfiError> {
    let sources = buffers
        .iter()
        .map(|buffer| {
            if buffer.rank() != 0 {
                return Err(FfiError::invalid_argument("assertion observations must be rank-zero buffers"));
            }
            Ok((unsafe { buffer.data() }, observation_byte_count(buffer.element_type())?))
        })
        .collect::<Result<Vec<_>, _>>()?;
    buffers
        .iter()
        .zip(buffer_bytes(&sources, memory)?)
        .map(|(buffer, bytes)| format_observation(buffer.element_type(), &bytes))
        .collect()
}

/// Renders the validated bytes of one scalar observation.
fn format_observation(r#type: FfiBufferType, bytes: &[u8]) -> Result<String, FfiError> {
    Ok(match r#type {
        FfiBufferType::Predicate => (bytes[0] != 0).to_string(),
        FfiBufferType::I1 => ((bytes[0] as i8) << 7 >> 7).to_string(),
        FfiBufferType::I2 => ((bytes[0] as i8) << 6 >> 6).to_string(),
        FfiBufferType::I4 => ((bytes[0] as i8) << 4 >> 4).to_string(),
        FfiBufferType::U1 => (bytes[0] & 1).to_string(),
        FfiBufferType::U2 => (bytes[0] & 3).to_string(),
        FfiBufferType::U4 => (bytes[0] & 15).to_string(),
        FfiBufferType::I8 => i8::from_ne_bytes(bytes.try_into().unwrap()).to_string(),
        FfiBufferType::I16 => i16::from_ne_bytes(bytes.try_into().unwrap()).to_string(),
        FfiBufferType::I32 => i32::from_ne_bytes(bytes.try_into().unwrap()).to_string(),
        FfiBufferType::I64 => i64::from_ne_bytes(bytes.try_into().unwrap()).to_string(),
        FfiBufferType::U8 => u8::from_ne_bytes(bytes.try_into().unwrap()).to_string(),
        FfiBufferType::U16 => u16::from_ne_bytes(bytes.try_into().unwrap()).to_string(),
        FfiBufferType::U32 => u32::from_ne_bytes(bytes.try_into().unwrap()).to_string(),
        FfiBufferType::U64 => u64::from_ne_bytes(bytes.try_into().unwrap()).to_string(),
        FfiBufferType::F16 => half::f16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32().to_string(),
        FfiBufferType::BF16 => {
            half::bf16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32().to_string()
        }
        FfiBufferType::F32 => f32::from_ne_bytes(bytes.try_into().unwrap()).to_string(),
        FfiBufferType::F64 => f64::from_ne_bytes(bytes.try_into().unwrap()).to_string(),
        r#type => return Err(FfiError::invalid_argument(format!("unsupported assertion observation type `{type}`"))),
    })
}

/// Reads one rank-zero signed 64-bit extent buffer.
fn scalar_i64(buffer: &FfiBuffer<'_>, memory: AssertionBufferMemory) -> Result<i64, FfiError> {
    if buffer.element_type() != FfiBufferType::I64 || buffer.rank() != 0 {
        return Err(FfiError::invalid_argument(format!(
            "expected the `{ASSERT_CUSTOM_CALL_TARGET}` observed extent to be a rank-zero i64 buffer"
        )));
    }
    Ok(i64::from_ne_bytes(scalar_bytes(buffer, memory)?))
}

/// Lowers the runtime check of a [`GuardOperation`] over `values`, whose actual types are `actual_types`, against
/// the guard's `declared_types` (refer to [`lower_type_relationship_checks`]).
///
/// Ordered assertions and ordered I/O normally use independent token chains. A guard instead joins the assertion
/// token into the ordered-I/O chain, so every later ordered I/O effect (including those inside the region whose input
/// it checks) is ordered after its check and after every earlier assertion on that chain, exactly as the input
/// validation that it replaces precedes the region. That includes a guard that emits no check of its own because its
/// declarations are proven. Ordinary assertions keep their independent chain.
///
/// # Parameters
///
///   - `boundary`: Diagnostic name of the guarded boundary.
///   - `declared_types`: Declared types of the guarded members.
///   - `values`: Lowered values of the guarded members.
///   - `actual_types`: Actual types of `values`, which refine `declared_types`.
pub(crate) fn lower_guard<'b, 'c: 'b, 't: 'c>(
    boundary: &str,
    declared_types: &[ArrayIrType],
    values: &[ValueRef<'b, 'c, 't>],
    actual_types: &[ArrayIrType],
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let guarded = lower_type_relationship_checks(
        GUARD_OPERATION_NAME,
        &format!("at the entry of `{boundary}`"),
        declared_types,
        values,
        actual_types,
        effect_tokens,
        block,
        context,
        location,
    )?;

    // Later ordered I/O of both ordered I/O classes, which share one token slot, follows the check: the assertion token
    // joins the I/O chain, which every scope that threads the assertion token also threads (refer to
    // `token_threaded_effects`). A guard that emits no check still joins the assertions that precede it, since its
    // ordered effect follows them. Unordered effects have no token and are not ordered after the check.
    let assertion = if guarded {
        Some(current_or_new_token(EffectClass::OrderedAssertion, effect_tokens, block, location)?)
    } else {
        effect_tokens.get(EffectClass::OrderedAssertion)
    };
    if let Some(assertion) = assertion {
        let joined = match effect_tokens.get(EffectClass::OrderedIo) {
            Some(io) => block
                .append_operation(stable_hlo::after_all(&[io, assertion], location)?)?
                .result(0)
                .expect("stablehlo.after_all should return one result")
                .as_ref(),
            None => assertion,
        };
        effect_tokens.set(EffectClass::OrderedIo, joined);
    }
    Ok(())
}

/// Lowers runtime assertions that `values`, whose actual types are `actual_types`, satisfy the relationships that
/// `declared_types` express, and returns whether it emitted any. Every occurrence of a declared dimension identity (an
/// array axis or a first-class dimension member) is observed as an `i64` extent, which is static when its actual type
/// fixes it and read at runtime otherwise, and an assertion requires each occurrence to equal the first. A declared
/// static extent whose actual dimension is symbolic is asserted to be that extent. Pairs that are both static were
/// already checked when the instruction was typed and emit nothing. Actual dimensions with proven equal extents
/// share one observation even across different values. The assertions advance only the ordered-assertion chain.
///
/// # Parameters
///
///   - `actor`: Name of the operation that owns the assertions, reported in their diagnostics.
///   - `site`: Description of where the relationships must hold, appended to the diagnostics (e.g., ``in `cast` ``).
///   - `declared_types`: Declared types that express the relationships.
///   - `values`: Lowered values whose runtime extents must satisfy them.
///   - `actual_types`: Actual types of `values`.
pub(crate) fn lower_type_relationship_checks<'b, 'c: 'b, 't: 'c>(
    actor: &str,
    site: &str,
    declared_types: &[ArrayIrType],
    values: &[ValueRef<'b, 'c, 't>],
    actual_types: &[ArrayIrType],
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<bool, LoweringError> {
    check_count!("input", values, declared_types.len(), ProgramError);
    check_count!("input", actual_types, declared_types.len(), ProgramError);
    let mut checked = false;
    let mut lower_check = |expected: ValueRef<'b, 'c, 't>,
                           actual: ValueRef<'b, 'c, 't>,
                           message: String,
                           effect_tokens: &mut EffectTokens<'b, 'c, 't>,
                           block: &mut BlockRef<'b, 'c, 't>|
     -> Result<(), LoweringError> {
        let predicate = lower_compare_to_mlir(ComparisonDirection::Equal, actual, expected, block, location)?;
        lower_assert_to_custom_call(
            actor,
            &message,
            &["expected".to_string(), "actual".to_string()],
            None,
            &[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::I64), ArrayType::scalar(DataType::I64)],
            &[predicate, expected, actual],
            effect_tokens,
            block,
            context,
            location,
        )?;
        checked = true;
        Ok(())
    };

    // Keep a runtime extent's original input and optional array axis until a real comparison needs it. Source/output
    // declarations can observe the same axis twice, or different values can carry the same actual identity. Retain
    // those actual dimensions so proven equalities require neither a query nor a host callback.
    let mut occurrences =
        Vec::<(&DimensionVariable, &Dimension, Result<usize, (ValueRef<'b, 'c, 't>, Option<usize>)>)>::new();
    let mut static_checks = Vec::<(usize, &Dimension, (ValueRef<'b, 'c, 't>, Option<usize>))>::new();
    for ((declared, &value), actual) in declared_types.iter().zip(values).zip(actual_types) {
        match (declared, actual) {
            (ArrayIrType::Array(declared), ArrayIrType::Array(actual)) if declared.rank() == actual.rank() => {
                for (axis, (declared, actual)) in
                    declared.shape().dimensions().iter().zip(actual.shape().dimensions()).enumerate()
                {
                    let extent = actual.bounds().extent().ok_or((value, Some(axis)));
                    match (declared, extent) {
                        (Dimension::Dynamic(variable), extent) => occurrences.push((variable, actual, extent)),
                        (Dimension::Static(_), Ok(_)) => {}
                        (Dimension::Static(expected), Err((value, Some(axis)))) => {
                            if static_checks.iter().any(|(extent, dimension, observed)| {
                                extent == expected
                                    && (*observed == (value, Some(axis)) || dimension.has_equal_extents(actual))
                            }) {
                                continue;
                            }
                            static_checks.push((*expected, actual, (value, Some(axis))));
                            let runtime = lower_runtime_dimension_size_i64(value, axis, block, context, location)?;
                            let expected = lower_static_index_constants(&[*expected], block, context, location)?[0];
                            let message = format!("input dimension `{actual}` must have the declared extent {site}");
                            lower_check(expected, runtime, message, effect_tokens, block)?;
                        }
                        (Dimension::Static(_), Err((_, None))) => unreachable!(),
                    }
                }
            }
            (ArrayIrType::Dimension(declared), ArrayIrType::Dimension(actual)) => {
                match (declared.variable(), declared.extent(), actual.extent()) {
                    (Some(variable), _, extent) => {
                        occurrences.push((variable, actual.dimension(), extent.ok_or((value, None))));
                    }
                    (None, Some(expected), None) => {
                        if static_checks.iter().any(|(extent, dimension, observed)| {
                            *extent == expected
                                && (*observed == (value, None) || dimension.has_equal_extents(actual.dimension()))
                        }) {
                            continue;
                        }
                        static_checks.push((expected, actual.dimension(), (value, None)));
                        let expected = lower_static_index_constants(&[expected], block, context, location)?[0];
                        let message = format!("input dimension `{actual}` must have the declared extent {site}");
                        lower_check(expected, value, message, effect_tokens, block)?;
                    }
                    (None, _, _) => {}
                }
            }
            (declared, actual) => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "`{actor}` cannot check actual type `{actual}` against declared type `{declared}` {site}"
                    ),
                });
            }
        }
    }

    let mut comparisons = Vec::<(usize, usize)>::new();
    let equivalent = |left: usize, right: usize| {
        occurrences[left].2 == occurrences[right].2 || occurrences[left].1.has_equal_extents(occurrences[right].1)
    };
    for (index, (variable, dimension, extent)) in occurrences.iter().enumerate() {
        let Some(first_index) = occurrences[..index].iter().position(|(previous, _, _)| previous == variable) else {
            continue;
        };
        let first = &occurrences[first_index].2;
        if (first.is_ok() && extent.is_ok())
            || static_checks.iter().any(|(expected, actual, observed)| {
                (first.as_ref().is_ok_and(|extent| extent == expected)
                    && (*extent == Err(*observed) || dimension.has_equal_extents(actual)))
                    || (extent.as_ref().is_ok_and(|extent| extent == expected)
                        && (*first == Err(*observed) || occurrences[first_index].1.has_equal_extents(actual)))
            })
            || occurrences[..index].iter().any(|(previous, actual, observed)| {
                previous == variable && (observed == extent || dimension.has_equal_extents(actual))
            })
            || comparisons.iter().any(|&(left, right)| {
                (equivalent(left, first_index) && equivalent(right, index))
                    || (equivalent(right, first_index) && equivalent(left, index))
            })
        {
            continue;
        }
        let lower_extent = |extent: &Result<usize, (ValueRef<'b, 'c, 't>, Option<usize>)>,
                            block: &mut BlockRef<'b, 'c, 't>| match extent {
            Ok(extent) => Ok(lower_static_index_constants(&[*extent], block, context, location)?.remove(0)),
            Err((value, None)) => Ok::<_, LoweringError>(*value),
            Err((value, Some(axis))) => lower_runtime_dimension_size_i64(*value, *axis, block, context, location),
        };
        let expected = lower_extent(first, block)?;
        let actual = lower_extent(extent, block)?;
        let message = format!("input dimension `{variable}` must have one extent {site}");
        lower_check(expected, actual, message, effect_tokens, block)?;
        comparisons.push((first_index, index));
    }
    Ok(checked)
}

impl LowerableOperation for GuardOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let declared_types = self.declared_types().iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>();
        let input_types = input_types.iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>();
        let output_types = output_types.iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>();
        lower_guard_with_outputs(self.boundary(), &declared_types, input_values, &input_types, &output_types, lowerer)
    }
}

impl LowerableOperation for GuardOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_guard_with_outputs(
            self.boundary(),
            self.declared_types(),
            input_values,
            input_types,
            output_types,
            lowerer,
        )
    }
}

/// Makes every returned value depend on all the supplied value producers and the optional assertion token. A single
/// barrier carries tensors and tokens together, so even conversions of otherwise unchanged values follow the check,
/// and a proof-carrying cast cannot discard the dependencies of its witnesses. The original effect token remains the
/// ordered-effect chain's current token; the barrier only supplies dependencies to its value consumers.
pub(crate) fn lower_dependency_barrier<'b, 'c: 'b, 't: 'c>(
    values: &[ValueRef<'b, 'c, 't>],
    assertion_token: Option<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if values.is_empty() || (values.len() == 1 && assertion_token.is_none()) {
        return Ok(values.to_vec());
    }
    let inputs = values.iter().copied().chain(assertion_token).collect::<Vec<_>>();
    let barrier = block.append_operation(stable_hlo::optimization_barrier(&inputs, location)?)?;
    Ok((0..values.len()).map(|index| barrier.result(index).unwrap().as_ref()).collect())
}

/// Lowers a [`GuardOperation`] instruction: emits its check with [`lower_guard`], makes every output
/// depend on the check, and converts each input's shape with [`lower_cast_shape`]. The conversions only
/// re-describe values that the check has related, including outputs that narrow some axes and give others a
/// [retained](GuardOperation::retained) identity. Guards preserve their inputs' layout and sharding.
fn lower_guard_with_outputs<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    boundary: &str,
    declared_types: &[ArrayIrType],
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
    lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("output", output_types, input_values.len(), ProgramError);
    lower_guard(
        boundary,
        declared_types,
        input_values,
        input_types,
        &mut lowerer.effect_tokens,
        &mut lowerer.block,
        lowerer.context,
        lowerer.location,
    )?;
    let values = lower_dependency_barrier(
        input_values,
        lowerer.effect_tokens.ordered_assertion,
        &mut lowerer.block,
        lowerer.location,
    )?;
    values
        .iter()
        .zip(input_types)
        .zip(output_types)
        .map(|((&value, input_type), output_type)| {
            lower_cast_shape(value, input_type, output_type, &mut lowerer.block, lowerer.context, lowerer.location)
        })
        .collect()
}

impl LowerableOperation for AssertOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_assert_to_custom_call(
            self.name(),
            self.message(),
            self.labels(),
            self.failure_limit(),
            input_types,
            input_values,
            &mut lowerer.effect_tokens,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        Ok(Vec::new())
    }
}

impl LowerableOperation for AssertOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let context = lowerer.context;
        let location = lowerer.location;
        let effect_tokens = &mut lowerer.effect_tokens;
        let block = &mut lowerer.block;
        lower_assert_to_custom_call(
            self.name(),
            self.message(),
            self.labels(),
            self.failure_limit(),
            &input_types
                .iter()
                .map(|input| match input {
                    ArrayIrType::Array(array) => array.clone(),
                    ArrayIrType::Dimension(_) => ArrayType::scalar(DataType::I64),
                    ArrayIrType::Reference(_) => {
                        unreachable!("assertion validation rejects reference observations")
                    }
                })
                .collect::<Vec<_>>(),
            input_values,
            effect_tokens,
            block,
            context,
            location,
        )?;
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayIrType, CompareOperation, DimensionBounds, DimensionError, DimensionPow, DimensionType, DimensionValue,
        DimensionVariable, Div, Placeholder, ProgramError, Rem, Shape, Sub,
    };
    use ryft_pjrt::extensions::ffi::XLA_FFI_Buffer;

    use crate::experimental::lowering::lower_mlir_module_for_program_with_reference_state;
    use crate::experimental::ops::{XlaConstant, XlaProgramBuilder as CompositeXlaProgramBuilder};
    use crate::operations::dot::RaggedDotLoweringStrategy;
    use crate::to_mlir_module_for_program;

    use super::*;

    #[test]
    fn test_executable_assertions_remain_local() {
        use ryft_core::{AssertOperation, DimensionFromScalarOperation};

        let scalar_type = ArrayType::scalar(DataType::I64);
        let left_variable = DimensionVariable::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_variable = DimensionVariable::new("right", DimensionBounds::new(1, Some(9)).unwrap());
        let mut builder = CompositeXlaProgramBuilder::new();
        let left = builder.add_input(scalar_type.clone().into());
        let right = builder.add_input(scalar_type.clone().into());
        let left_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(left_variable.into()), Vec::new(), vec![left], None)
            .unwrap()[0];
        let right_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(right_variable.into()), Vec::new(), vec![right], None)
            .unwrap()[0];
        let predicate = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::Equal),
                Vec::new(),
                vec![left_dimension, right_dimension],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new("dimensions must agree"),
                Vec::new(),
                vec![predicate],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![], vec![Placeholder, Placeholder], vec![])
            .unwrap();
        let lowered = lower_mlir_module_for_program_with_reference_state(
            &program,
            &[],
            &vec![scalar_type.clone(), scalar_type],
            &Vec::<ArrayType>::new(),
            "main",
            None,
            None,
            Some("cpu"),
            &[],
            RaggedDotLoweringStrategy::default(),
        )
        .unwrap();
        assert!(lowered.requires_assertion_handler);
        assert!(lowered.signature.requires_assertion_handler());
        assert!(lowered.signature.has_effects());
        assert!(!lowered.signature.has_ordered_io());
        assert!(!lowered.signature.has_unordered_io());
        assert_eq!(lowered.signature.ordered_io_input_index(), None);
        assert_eq!(lowered.signature.ordered_io_output_index(), None);
        assert_eq!(lowered.signature.physical_input_count(), 2);
        assert!(lowered.stable_hlo.contains("func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) {"));
        assert!(lowered.stable_hlo.contains("stablehlo.after_all"));
        assert!(lowered.stable_hlo.contains("stablehlo.custom_call @ryft.assert"));
    }

    #[test]
    fn test_to_mlir_module_for_program_separates_assertion_and_io_token_chains() {
        use ryft_core::{AssertOperation, DimensionFromScalarOperation, PrintOperation};

        let scalar_type = ArrayType::scalar(DataType::I64);
        let left_variable = DimensionVariable::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_variable = DimensionVariable::new("right", DimensionBounds::new(1, Some(9)).unwrap());

        let mut builder = CompositeXlaProgramBuilder::new();
        let left = builder.add_input(scalar_type.clone().into());
        let right = builder.add_input(scalar_type.clone().into());
        let left_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(left_variable.into()), Vec::new(), vec![left], None)
            .unwrap()[0];
        let right_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(right_variable.into()), Vec::new(), vec![right], None)
            .unwrap()[0];
        let predicate = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThanOrEqual),
                Vec::new(),
                vec![left_dimension, right_dimension],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new("`left_dimension` <= `right_dimension`")
                    .with_labels(vec!["left_dimension".to_owned(), "right_dimension".to_owned()]),
                Vec::new(),
                vec![predicate, left_dimension, right_dimension],
                None,
            )
            .unwrap();
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("value"), Vec::new(), vec![left], None)
            .unwrap();
        let predicate = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::Equal),
                Vec::new(),
                vec![left_dimension, right_dimension],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new("`left_dimension` == `right_dimension`")
                    .with_labels(vec!["left_dimension".to_owned(), "right_dimension".to_owned()]),
                Vec::new(),
                vec![predicate, left_dimension, right_dimension],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![left], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![scalar_type.clone(), scalar_type.clone()],
            &vec![scalar_type],
            "main",
            None,
            None,
        )
        .unwrap();

        // The two gateway bounds checks and two explicit requirements share the assertion chain. The print receives
        // a separately created token, so it has no artificial dependency on either neighboring assertion.
        assert_eq!(module.matches("stablehlo.after_all").count(), 2, "{module}");
        assert_eq!(module.matches("@ryft.assert").count(), 4, "{module}");
        assert_eq!(module.matches("@ryft.print").count(), 1, "{module}");
        let print = module.lines().find(|line| line.contains("@ryft.print")).unwrap();
        let first_assertion = module.lines().find(|line| line.contains("@ryft.assert")).unwrap();
        let print_token =
            print.split_once("@ryft.print(").unwrap().1.split_once(')').unwrap().0.rsplit_once(", ").unwrap().1;
        let assertion_token = first_assertion
            .split_once("@ryft.assert(")
            .unwrap()
            .1
            .split_once(')')
            .unwrap()
            .0
            .rsplit_once(", ")
            .unwrap()
            .1;
        assert_ne!(print_token, assertion_token, "{module}",);
    }

    #[test]
    fn test_lower_guard_orders_later_io_after_the_check() {
        use ryft_core::{DimensionType, PrintOperation, Shape};

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::operations::debugging::ensure_print_handler_registered;
        use crate::tests::execution_client;

        // `[dimension<n>, f32[n]]` entered with a symbolic `dimension<m>` and an exact `f32[2]` needs one runtime
        // check.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let declared: Vec<ArrayIrType> =
            vec![n.clone().into(), ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()])).into()];
        let array_type = ArrayType::new_static(DataType::F64, [2]);
        let mut builder = CompositeXlaProgramBuilder::new();
        let dimension = builder.add_input(m.into());
        let array = builder.add_input(array_type.clone().into());
        builder
            .add_instruction(GuardOperation::new("region", declared), Vec::new(), vec![dimension, array], None)
            .unwrap();
        let printed = builder
            .add_instruction(PrintOperation::<ArrayType>::new("value"), Vec::new(), vec![array], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![printed], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[ArrayType::scalar(DataType::I64), array_type.clone()],
            &[array_type],
            "main",
            None,
            None,
        )
        .unwrap();
        // The print's token is the guard assertion's output token, so the print cannot run before the check.
        let assertion = module.lines().find(|line| line.contains("@ryft.assert")).unwrap();
        let assertion_token = assertion.trim().split_once(" = ").unwrap().0;
        let print = module.lines().find(|line| line.contains("@ryft.print")).unwrap();
        let print_token =
            print.split_once("@ryft.print(").unwrap().1.split_once(')').unwrap().0.rsplit_once(", ").unwrap().1;
        assert_eq!(print_token, assertion_token, "{module}");
        assert!(assertion.contains(r#"actor = "guard""#), "{module}");

        let client = execution_client();
        ensure_assertion_handler_registered(&client).unwrap();
        ensure_print_handler_registered(&client).unwrap();
        let values = MixedValue::Array(vec![1.0, 2.0], vec![2]);
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Dimension(2), values.clone()], &[]),
            Ok(vec![values.clone()]),
        );
        let error = execute_mixed_program(&client, &program, &[MixedValue::Dimension(5), values], &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("input dimension `n` must have one extent at the entry of `region`"), "{error}");
    }

    #[test]
    fn test_lower_guard_widens_values_to_retained_identities() {
        use ryft_core::{AddOperation, Typed};

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        // `[f64[m], f64[2], f64[k]]` entering `[f64[n], f64[n], f64[n]]` with a retained `m` checks `m = k = 2` and
        // returns all three values at `f64[m]`: the exact value is padded to the bound of `m`, and the value of `k`,
        // whose bound is larger, is sliced to the checked extent before it is padded, so their sum is well typed.
        let wide = DimensionBounds::positive(Some(9)).unwrap();
        let n = DimensionType::new("n", wide);
        let m = DimensionType::new("m", DimensionBounds::positive(Some(5)).unwrap());
        let k = DimensionType::new("k", wide);
        let vector = |dimension: &DimensionType| -> ArrayIrType {
            ArrayType::new(DataType::F64, Shape::new(vec![dimension.simplified_dimension()])).into()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let first = builder.add_input(vector(&m));
        let second = builder.add_input(ArrayType::new_static(DataType::F64, [2]).into());
        let third = builder.add_input(vector(&k));
        let guard = GuardOperation::new("region", vec![vector(&n), vector(&n), vector(&n)])
            .with_retained(vec![m.to_variable()]);
        let outputs = builder.add_instruction(guard, Vec::new(), vec![first, second, third], None).unwrap().to_vec();
        assert_eq!(builder.atoms()[outputs[1].index()].r#type().as_ref(), &vector(&m));
        assert_eq!(builder.atoms()[outputs[2].index()].r#type().as_ref(), &vector(&m));
        let partial = builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![outputs[0], outputs[1]], None)
            .unwrap()[0];
        let sum = builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![partial, outputs[2]], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let client = execution_client();
        let inputs = [
            MixedValue::Array(vec![1.0, 2.0, 0.0, 0.0], vec![4]),
            MixedValue::Array(vec![3.0, 4.0], vec![2]),
            MixedValue::Array(vec![5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![8]),
        ];
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[2, 2]),
            Ok(vec![MixedValue::Array(vec![9.0, 12.0], vec![2]), MixedValue::Dimension(2)]),
        );
        let error = execute_mixed_program(&client, &program, &inputs, &[2, 3]).unwrap_err().to_string();
        assert!(error.contains("must have one extent at the entry of `region`"), "{error}");
    }

    #[test]
    fn test_lower_guard_renames_values_to_retained_identities() {
        use ryft_core::{AddOperation, Typed};

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        // `[f64[m], f64[k]]` entering `[f64[n], f64[n]]` with a retained `m` checks `m = k` without an exact extent and
        // returns both values at `f64[m]`: the value of `k`, whose physical bound is larger, is sliced down to the
        // bound of `m` and keeps its runtime size, so their sum is well typed.
        let wide = DimensionBounds::positive(Some(9)).unwrap();
        let n = DimensionType::new("n", wide);
        let m = DimensionType::new("m", DimensionBounds::positive(Some(5)).unwrap());
        let k = DimensionType::new("k", wide);
        let vector = |dimension: &DimensionType| -> ArrayIrType {
            ArrayType::new(DataType::F64, Shape::new(vec![dimension.simplified_dimension()])).into()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let first = builder.add_input(vector(&m));
        let second = builder.add_input(vector(&k));
        let guard = GuardOperation::new("region", vec![vector(&n), vector(&n)]).with_retained(vec![m.to_variable()]);
        let outputs = builder.add_instruction(guard, Vec::new(), vec![first, second], None).unwrap().to_vec();
        assert_eq!(builder.atoms()[outputs[1].index()].r#type().as_ref(), &vector(&m));
        let sum = builder.add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), outputs, None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let client = execution_client();
        let inputs = [
            MixedValue::Array(vec![1.0, 2.0, 0.0, 0.0], vec![4]),
            MixedValue::Array(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![8]),
        ];
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[2, 2]),
            Ok(vec![MixedValue::Array(vec![4.0, 6.0], vec![2]), MixedValue::Dimension(2)]),
        );
        let error = execute_mixed_program(&client, &program, &inputs, &[2, 3]).unwrap_err().to_string();
        assert!(error.contains("must have one extent at the entry of `region`"), "{error}");
    }

    #[test]
    fn test_lower_guard_orders_later_io_after_preceding_assertions() {
        use ryft_core::{AssertOperation, DimensionFromScalarOperation, PrintOperation};

        // A guard whose declarations are proven emits no check of its own, but it still joins the assertions that
        // precede it into the ordered-I/O chain, so a later print of a value that does not consume the guard's output
        // cannot run before those assertions.
        let scalar_type = ArrayType::scalar(DataType::I64);
        let array_type = ArrayType::new_static(DataType::F64, [3]);
        let left_variable = DimensionVariable::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_variable = DimensionVariable::new("right", DimensionBounds::new(1, Some(9)).unwrap());
        let mut builder = CompositeXlaProgramBuilder::new();
        let left = builder.add_input(scalar_type.clone().into());
        let right = builder.add_input(scalar_type.clone().into());
        let value = builder.add_input(array_type.clone().into());
        let row = builder.add_input(array_type.clone().into());
        let left_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(left_variable.into()), Vec::new(), vec![left], None)
            .unwrap()[0];
        let right_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(right_variable.into()), Vec::new(), vec![right], None)
            .unwrap()[0];
        let predicate = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThanOrEqual),
                Vec::new(),
                vec![left_dimension, right_dimension],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new("`left_dimension` <= `right_dimension`"),
                Vec::new(),
                vec![predicate],
                None,
            )
            .unwrap();
        let guarded = builder
            .add_instruction(
                GuardOperation::new("region", vec![ArrayIrType::from(array_type.clone())]),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap()[0];
        let printed = builder
            .add_instruction(PrintOperation::<ArrayType>::new("row"), Vec::new(), vec![row], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![guarded, printed],
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[scalar_type.clone(), scalar_type, array_type.clone(), array_type.clone()],
            &[array_type.clone(), array_type],
            "main",
            None,
            None,
        )
        .unwrap();
        // The guard emits no assertion, so the explicit requirement is the last one. The print's token derives from
        // its output token through the guard's chain join.
        let assertion = module.lines().filter(|line| line.contains("@ryft.assert")).last().unwrap();
        assert!(assertion.contains("`left_dimension` <= `right_dimension`"), "{module}");
        let assertion_token = assertion.trim().split_once(" = ").unwrap().0.to_owned();
        let print = module.lines().find(|line| line.contains("@ryft.print")).unwrap();
        let mut token = print
            .split_once("@ryft.print(")
            .unwrap()
            .1
            .split_once(')')
            .unwrap()
            .0
            .rsplit_once(", ")
            .unwrap()
            .1
            .to_owned();
        while token != assertion_token {
            let definition = module
                .lines()
                .find(|line| line.trim().starts_with(&format!("{token} = stablehlo.after_all")))
                .unwrap_or_else(|| panic!("print token `{token}` does not derive from the assertion:\n{module}"));
            // Follow the last operand of the join, which is the assertion token rather than the ordered-I/O chain's.
            let operands = definition.split_once("after_all").unwrap().1.split_once(':').unwrap().0;
            token = operands.split(',').map(str::trim).filter(|operand| !operand.is_empty()).last().unwrap().to_owned();
        }
    }

    #[test]
    fn test_lower_guard_orders_bounded_conversions_after_the_check() {
        use ryft_core::{
            AddOperation, ArrayReferenceDischarge, PartialValue, ReferenceReadOperation, ReferenceType, TracingContext,
        };

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::experimental::ops::XlaOperation;
        use crate::tests::execution_client;

        let client = execution_client();
        for composite in [false, true] {
            // Both members fit the declared bound. Only the guard proves that the wider member fits the retained
            // member's smaller physical buffer; an invalid extent of 6 must fail before it can be restored on bound 4.
            let wide = DimensionBounds::positive(Some(9)).unwrap();
            let declared = DimensionType::new("n", wide);
            let retained = DimensionType::new("m", DimensionBounds::positive(Some(5)).unwrap());
            let other = DimensionType::new("k", wide);
            let vector = |dimension: &DimensionType| {
                ArrayType::new(DataType::F64, Shape::new(vec![dimension.simplified_dimension()]))
            };
            let input_types = vec![vector(&retained), vector(&other)];
            let mut builder = CompositeXlaProgramBuilder::new();
            let first = builder.add_input(ReferenceType::new(input_types[0].clone()).into());
            let second = builder.add_input(ReferenceType::new(input_types[1].clone()).into());
            let reads = [first, second]
                .into_iter()
                .map(|reference| {
                    builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()
                        [0]
                })
                .collect();
            let guard: XlaOperation = if composite {
                GuardOperation::<ArrayIrType>::new("region", vec![vector(&declared).into(), vector(&declared).into()])
                    .with_retained(vec![retained.to_variable()])
                    .into()
            } else {
                GuardOperation::<ArrayType>::new("region", vec![vector(&declared), vector(&declared)])
                    .with_retained(vec![retained.to_variable()])
                    .into()
            };
            let guarded = builder.add_instruction(guard, Vec::new(), reads, None).unwrap().to_vec();
            let sum =
                builder.add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), guarded, None).unwrap()[0];
            let source = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            // Exercise the public specialization and reference-discharge boundaries before differentiating and
            // lowering. These read-only external bindings become the two explicit initial-state array arguments.
            let context = TracingContext::<XlaConstant, XlaOperation>::new();
            let inputs = source.input_types().into_iter().map(PartialValue::Unknown).collect::<Vec<_>>();
            let evaluation = source.partially_evaluate_in_context(&context, &inputs).unwrap();
            let discharged = evaluation.program().clone().discharge_references::<ArrayReferenceDischarge>(0).unwrap();
            assert_eq!(discharged.external_reference_bindings().len(), 2);
            assert!(discharged.external_reference_bindings().iter().all(|binding| binding.output_index().is_none()));
            let program = discharged.program();
            let module =
                to_mlir_module_for_program(program, &[], &input_types, &vec![vector(&retained)], "main", None, None)
                    .unwrap();
            let assertion = module.lines().find(|line| line.contains("@ryft.assert")).unwrap();
            let token = assertion.trim().split_once(" = ").unwrap().0;
            let barrier = module.lines().find(|line| line.contains("stablehlo.optimization_barrier")).unwrap();
            let entry_value = |index| {
                module
                    .lines()
                    .find(|line| line.contains(&format!("stablehlo.set_dimension_size %arg{index},")))
                    .unwrap()
                    .trim()
                    .split_once(" = ")
                    .unwrap()
                    .0
            };
            assert!(barrier.contains(&format!("{}, {}, {token}", entry_value(0), entry_value(1))), "{module}");
            let values = barrier.trim().split_once(':').unwrap().0;
            assert!(module.contains(&format!("stablehlo.get_dimension_size {values}#1")), "{module}");
            let addition = module.lines().find(|line| line.contains("stablehlo.add ")).unwrap();
            assert!(addition.contains(&format!("{values}#0")), "{module}");

            let inputs = [
                MixedValue::Array(vec![1.0, 2.0, 0.0, 0.0], vec![4]),
                MixedValue::Array(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0], vec![8]),
            ];
            assert_eq!(
                execute_mixed_program(&client, program, &inputs, &[2, 2]),
                Ok(vec![MixedValue::Array(vec![4.0, 6.0], vec![2]), MixedValue::Dimension(2)]),
            );
            let error = execute_mixed_program(&client, program, &inputs, &[2, 6]).unwrap_err().to_string();
            assert!(error.contains("must have one extent at the entry of `region`"), "{error}");

            // The tangent cast has its own data input. The converted primal proof must remain a dependency even
            // though it is not numerically used; that proof already depends on the assertion token.
            let jvp = program.jvp().unwrap();
            let tangent_module = to_mlir_module_for_program(
                &jvp,
                &[],
                &vec![input_types[0].clone(), input_types[1].clone(), input_types[0].clone(), input_types[1].clone()],
                &vec![vector(&retained), vector(&retained)],
                "main",
                None,
                None,
            )
            .unwrap();
            let assertion = tangent_module.lines().find(|line| line.contains("@ryft.assert")).unwrap();
            let token = assertion.trim().split_once(" = ").unwrap().0;
            let entry_value = |index| {
                tangent_module
                    .lines()
                    .find(|line| line.contains(&format!("stablehlo.set_dimension_size %arg{index},")))
                    .unwrap()
                    .trim()
                    .split_once(" = ")
                    .unwrap()
                    .0
            };
            let barrier = tangent_module
                .lines()
                .find(|line| {
                    line.contains(&format!("stablehlo.optimization_barrier {}, {},", entry_value(3), entry_value(1),))
                })
                .unwrap();
            let primal_barrier = tangent_module
                .lines()
                .find(|line| line.contains("stablehlo.optimization_barrier") && line.contains(token))
                .unwrap();
            let primal_values = primal_barrier.trim().split_once(':').unwrap().0;
            let extent = tangent_module
                .lines()
                .find(|line| line.contains(&format!("stablehlo.get_dimension_size {primal_values}#1")))
                .unwrap()
                .trim()
                .split_once(" = ")
                .unwrap()
                .0;
            let proof = tangent_module
                .lines()
                .find(|line| line.contains("stablehlo.set_dimension_size") && line.contains(&format!(", {extent},")))
                .unwrap()
                .trim()
                .split_once(" = ")
                .unwrap()
                .0;
            assert!(barrier.contains(&format!(", {proof} :")), "{tangent_module}");
            let values = barrier.trim().split_once(':').unwrap().0;
            assert!(tangent_module.contains(&format!("stablehlo.get_dimension_size {values}#0")), "{tangent_module}");
            let inputs = [inputs[0].clone(), inputs[1].clone(), inputs[0].clone(), inputs[1].clone()];
            assert_eq!(
                execute_mixed_program(&client, &jvp, &inputs, &[2, 2, 2, 2]),
                Ok(vec![
                    MixedValue::Array(vec![4.0, 6.0], vec![2]),
                    MixedValue::Array(vec![4.0, 6.0], vec![2]),
                    MixedValue::Dimension(2),
                    MixedValue::Dimension(2),
                ]),
            );
            let error = execute_mixed_program(&client, &jvp, &inputs, &[2, 6, 2, 6]).unwrap_err().to_string();
            assert!(error.contains("must have one extent at the entry of `region`"), "{error}");
        }
    }

    /// Builds an eager dimension value whose diagnostic variable name matches the FFI-reported operand name.
    fn eager_dimension(name: &str, extent: usize) -> DimensionValue {
        DimensionValue::new(DimensionType::new(name, DimensionBounds::unbounded()), extent).unwrap()
    }

    /// Extracts the checked-arithmetic diagnostic carried by an eager dimension error.
    fn eager_message(error: ProgramError) -> String {
        match error.downcast_custom::<DimensionError>() {
            Some(DimensionError::ArithmeticOverflow { message })
            | Some(DimensionError::RequirementViolation { message }) => message.clone(),
            other => panic!("unexpected eager dimension error: {other:?}"),
        }
    }

    #[test]
    fn test_copy_buffer_group() {
        for failed_enqueue in [None, Some(1)] {
            let queued = std::cell::RefCell::new(Vec::new());
            let synchronized = std::cell::Cell::new(0);
            let source = [4_u8, 7];
            let sources =
                source.iter().map(|value| (std::ptr::from_ref(value).cast_mut().cast(), 1)).collect::<Vec<_>>();
            let output = copy_buffer_group(
                &sources,
                |source, destination, size| {
                    if failed_enqueue == Some(queued.borrow().len()) {
                        return Err(FfiError::internal("mock enqueue failure"));
                    }
                    queued.borrow_mut().push((source, destination, size));
                    Ok(())
                },
                || {
                    synchronized.set(synchronized.get() + 1);
                    for (source, destination, size) in queued.borrow().iter().copied() {
                        unsafe { std::ptr::copy_nonoverlapping(source.cast::<u8>(), destination.cast::<u8>(), size) };
                    }
                    Ok(())
                },
            );
            assert_eq!(synchronized.get(), 1);
            if failed_enqueue.is_some() {
                assert!(output.is_err());
                assert_eq!(queued.borrow().len(), 1);
            } else {
                assert_eq!(output.unwrap(), vec![vec![4], vec![7]]);
            }
        }
        assert!(
            copy_buffer_group(&[], |_, _, _| panic!("unexpected copy"), || panic!("unexpected synchronization"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_copy_buffer_group_failed_synchronization() {
        let destination = std::cell::Cell::new(std::ptr::null_mut::<u8>());
        let source = 4_u8;
        let output = copy_buffer_group(
            &[(std::ptr::from_ref(&source).cast_mut().cast(), 1)],
            |_, buffer, _| {
                destination.set(buffer.cast());
                Ok(())
            },
            || Err(FfiError::internal("mock synchronization failure")),
        );
        assert!(output.is_err());
        // The failed synchronization leaves the allocation alive for a potentially outstanding asynchronous write.
        unsafe { destination.get().write(7) };
        assert_eq!(unsafe { destination.get().read() }, 7);
    }

    #[test]
    fn test_element_offset() {
        assert_eq!(element_offset(&[1, 2], &[3, 5]).unwrap(), 7);
        assert_eq!(element_offset(&[], &[]).unwrap(), 0);
        assert!(element_offset(&[1], &[3, 5]).is_err());
        assert!(element_offset(&[3, 2], &[3, 5]).is_err());
        assert!(element_offset(&[0], &[-1]).is_err());
    }

    #[test]
    fn test_scalar_observation() {
        for (r#type, bytes, expected) in [
            (FfiBufferType::Predicate, vec![1], "true"),
            (FfiBufferType::I1, vec![1], "-1"),
            (FfiBufferType::I2, vec![2], "-2"),
            (FfiBufferType::I4, vec![8], "-8"),
            (FfiBufferType::U4, vec![15], "15"),
            (FfiBufferType::U64, u64::MAX.to_ne_bytes().to_vec(), "18446744073709551615"),
            (FfiBufferType::I64, i64::MIN.to_ne_bytes().to_vec(), "-9223372036854775808"),
            (FfiBufferType::F16, half::f16::from_f32(-0.5).to_bits().to_ne_bytes().to_vec(), "-0.5"),
            (FfiBufferType::BF16, half::bf16::from_f32(0.5).to_bits().to_ne_bytes().to_vec(), "0.5"),
            (FfiBufferType::F32, (-0.0_f32).to_ne_bytes().to_vec(), "-0"),
            (FfiBufferType::F64, f64::INFINITY.to_ne_bytes().to_vec(), "inf"),
        ] {
            let buffer = XLA_FFI_Buffer {
                struct_size: size_of::<XLA_FFI_Buffer>(),
                extension_start: std::ptr::null_mut(),
                data_type: unsafe { r#type.to_c_api() },
                data: bytes.as_ptr().cast_mut().cast(),
                rank: 0,
                dimensions: std::ptr::null_mut(),
            };
            let buffer = unsafe { FfiBuffer::from_c_api(&buffer) }.unwrap();
            assert_eq!(scalar_observations(&[buffer], AssertionBufferMemory::Host).unwrap()[0], expected);
        }
    }

    #[test]
    fn test_validate_arithmetic_matches_eager_checked_diagnostics() {
        // The host callback recomputes each checked predicate, so its wording must match the eager path exactly for
        // the same inputs; these pairs fail on both paths and must render identically. Literal type labels from eager
        // dimensions must be reported as operand roles, just as when a program is traced from eager values.
        let compiled = |kind, left: i64, right: i64| {
            validate_arithmetic(kind, &left.to_string(), left, &right.to_string(), right).unwrap_err()
        };
        assert_eq!(
            compiled(ASSERT_SUB_KIND, 2, 5),
            eager_message(eager_dimension("left", 2).sub(&eager_dimension("right", 5)).unwrap_err()),
        );
        assert_eq!(
            compiled(ASSERT_DIV_KIND, 7, 0),
            eager_message(eager_dimension("left", 7).div(&eager_dimension("right", 0)).unwrap_err()),
        );
        assert_eq!(
            compiled(ASSERT_REM_KIND, 7, 0),
            eager_message(eager_dimension("left", 7).rem(&eager_dimension("right", 0)).unwrap_err()),
        );
        assert_eq!(
            compiled(ASSERT_POW_KIND, 3, 64),
            eager_message(eager_dimension("left", 3).dimension_pow(&eager_dimension("right", 64)).unwrap_err()),
        );

        // These addition and multiplication pairs exceed the signed backend limit but fit the eager `usize`
        // computation on 64-bit hosts, so their overflow wording is pinned directly.
        assert_eq!(
            compiled(ASSERT_ADD_KIND, i64::MAX - 1, 2),
            "dimension arithmetic overflow in `dimension_add`; got left=9223372036854775806, right=2",
        );
        assert_eq!(
            compiled(ASSERT_MUL_KIND, i64::MAX / 2, 3),
            "dimension arithmetic overflow in `dimension_mul`; got left=4611686018427387903, right=3",
        );
    }

    #[test]
    fn test_validate_arithmetic() {
        assert_eq!(validate_arithmetic(ASSERT_ADD_KIND, "left", 2, "right", 3), Ok(()));
        assert_eq!(validate_arithmetic(ASSERT_SUB_KIND, "left", 3, "right", 2), Ok(()));
        assert_eq!(validate_arithmetic(ASSERT_MUL_KIND, "left", 2, "right", 3), Ok(()));
        assert_eq!(validate_arithmetic(ASSERT_POW_KIND, "left", 2, "right", 3), Ok(()));
        assert_eq!(validate_arithmetic(ASSERT_DIV_KIND, "left", 7, "right", 3), Ok(()));
        assert_eq!(validate_arithmetic(ASSERT_REM_KIND, "left", 7, "right", 3), Ok(()));

        assert_eq!(
            validate_arithmetic(ASSERT_ADD_KIND, "left", i64::MAX, "right", 1),
            Err("dimension arithmetic overflow in `dimension_add`; got left=9223372036854775807, right=1".to_string(),),
        );
        assert_eq!(
            validate_arithmetic(ASSERT_SUB_KIND, "left", 2, "right", 3),
            Err("`dimension_sub` requires left >= right; got left=2, right=3".to_string()),
        );
        assert_eq!(
            validate_arithmetic(ASSERT_MUL_KIND, "left", i64::MAX, "right", 2),
            Err("dimension arithmetic overflow in `dimension_mul`; got left=9223372036854775807, right=2".to_string(),),
        );
        assert_eq!(
            validate_arithmetic(ASSERT_POW_KIND, "left", i64::MAX, "right", 2),
            Err("dimension arithmetic overflow in `dimension_pow`; got left=9223372036854775807, right=2".to_string(),),
        );
        assert_eq!(
            validate_arithmetic(ASSERT_DIV_KIND, "left", 7, "right", 0),
            Err("`dimension_div` requires right > 0; got left=7, right=0".to_string()),
        );
        assert_eq!(
            validate_arithmetic(ASSERT_REM_KIND, "left", 7, "right", 0),
            Err("`dimension_rem` requires right > 0; got left=7, right=0".to_string()),
        );
        assert_eq!(
            validate_arithmetic(ASSERT_SUB_KIND, "rows", 2, "columns", 3),
            Err("`dimension_sub` requires left >= right; got left (`rows`)=2, right (`columns`)=3".to_string()),
        );
        assert_eq!(
            validate_arithmetic("unknown", "left", 1, "right", 1),
            Err("unsupported arithmetic assertion kind `unknown`".to_string()),
        );
    }

    #[test]
    fn test_validate_pad() {
        assert_eq!(validate_pad(0, 3, 6, -1, 2, 1), Ok(()));
        assert_eq!(validate_pad(0, 0, 3, 1, 2, usize::MAX), Ok(()));
        assert_eq!(validate_pad(0, 1, 4, 1, 2, usize::MAX), Ok(()));
        assert_eq!(
            validate_pad(0, 3, 5, -1, 2, 1),
            Err("`pad` result extent on axis 0 must equal 6 but got 5".to_string())
        );
        assert_eq!(
            validate_pad(1, 0, 0, -1, 0, 0),
            Err("`pad` result extent -1 is outside the native dimension range on axis 1".to_string())
        );
        assert_eq!(
            validate_pad(0, i64::from(i32::MAX), 0, 1, 0, 0),
            Err("`pad` result extent 2147483648 is outside the native dimension range on axis 0".to_string()),
        );
        assert_eq!(validate_pad(0, -1, 0, 0, 0, 0), Err("`pad` dimensions must be nonnegative on axis 0".to_string()));
        assert_eq!(
            validate_pad(0, 2, 0, i64::MIN, i64::MAX, 0),
            Err("`pad` result extent on axis 0 must equal 1 but got 0".to_string())
        );
    }

    #[test]
    fn test_validate_reshape() {
        assert_eq!(validate_reshape("reshape", &[4], &[2, 2]), Ok(()));
        assert_eq!(validate_reshape("reshape", &[], &[1]), Ok(()));
        assert_eq!(validate_reshape("reshape", &[0, i64::MAX, 2], &[2, 0]), Ok(()));
        assert_eq!(
            validate_reshape("reshape", &[4], &[3]),
            Err("`reshape` changes the number of elements from 4 to 3".to_string())
        );
        assert_eq!(
            validate_reshape("reshape", &[4], &[0]),
            Err("`reshape` changes the number of elements from 4 to 0".to_string())
        );
        assert_eq!(
            validate_reshape("reshape", &[i64::MAX, 2], &[1]),
            Err("`reshape` element count overflows the portable dimension range".to_string())
        );
        assert_eq!(
            validate_reshape("reshape", &[0, -1], &[0]),
            Err("`reshape` dimensions must be nonnegative".to_string())
        );
    }

    #[test]
    fn test_validate_concatenate() {
        assert_eq!(validate_concatenate("concatenate", "1", 7, &[2, 3, 2]), Ok(()));
        assert_eq!(
            validate_concatenate("concatenate", "1", 8, &[2, 3, 2]),
            Err("`concatenate` result extent must equal the sum of input axis 1 extents; expected 7 but got 8"
                .to_string(),),
        );
        assert_eq!(
            validate_concatenate("concatenate", "1", 0, &[i64::MAX, 1]),
            Err("`concatenate` input axis 1 extent sum overflows the portable dimension range".to_string()),
        );
    }

    #[test]
    fn test_validate_dynamic_slice_with_dimensions() {
        assert_eq!(validate_dynamic_slice_with_dimensions(1, 2, 8, 1, 3), Ok(()));
        assert_eq!(validate_dynamic_slice_with_dimensions(1, 2, 8, 8, 0), Ok(()));
        assert_eq!(
            validate_dynamic_slice_with_dimensions(1, 2, 8, 3, 4),
            Err("`dynamic_slice` limit 10 exceeds input axis 1 extent 8".to_string()),
        );
        assert_eq!(
            validate_dynamic_slice_with_dimensions(1, usize::MAX, 8, 0, 2),
            Err("`dynamic_slice` stride is outside the portable dimension range on axis 1".to_string()),
        );
        assert_eq!(
            validate_dynamic_slice_with_dimensions(1, i64::MAX as usize, i64::MAX, 0, 3),
            Err("`dynamic_slice` span overflows usize on axis 1".to_string()),
        );
        assert_eq!(
            validate_dynamic_slice_with_dimensions(1, 1, i64::MAX, i64::MAX, 1),
            Err("`dynamic_slice` limit overflows usize on axis 1".to_string()),
        );
    }
}
