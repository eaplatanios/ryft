use std::sync::OnceLock;

use ryft_core::DYNAMIC_SHAPE_SLICE_OPERATION_NAME;
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
use ryft_pjrt::extensions::ffi::FfiStream;
use ryft_pjrt::extensions::ffi::{
    FfiAttribute, FfiBuffer, FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits,
    FfiInput, FfiTypeId, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Handler,
};
use ryft_pjrt::{Client, Error};

/// Name of the typed XLA FFI custom call that reports failed first-class-dimension requirements.
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

/// Formatting kind used for an equality requirement.
pub(crate) const ASSERT_EQUAL_KIND: &str = "equal";

/// Formatting kind used for a less-than-or-equal requirement.
pub(crate) const ASSERT_LESS_THAN_OR_EQUAL_KIND: &str = "less_than_or_equal";

/// Formatting kind used for a positive-divisibility requirement.
pub(crate) const ASSERT_DIVISIBLE_BY_KIND: &str = "divisible_by";

/// Formatting kind used for explicit dimension bounds.
pub(crate) const ASSERT_BOUNDS_KIND: &str = "bounds";

/// Formatting kind used for a dynamic concatenation result-extent check.
pub(crate) const ASSERT_CONCATENATE_KIND: &str = "concatenate";

/// Formatting kind used for a dynamic-shape-slice runtime bounds check.
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
pub(crate) const ASSERT_DIV_FLOOR_KIND: &str = "div_floor";

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

/// Copies `byte_count` bytes from one CUDA device allocation into host memory and waits for completion.
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
fn copy_cuda_bytes(
    source: *mut std::ffi::c_void,
    destination: *mut std::ffi::c_void,
    byte_count: usize,
    stream: FfiStream,
) -> Result<(), FfiError> {
    let copy_result = unsafe { cuMemcpyDtoHAsync_v2(destination, source as usize as u64, byte_count, stream) };
    if copy_result != CUDA_SUCCESS {
        return Err(FfiError::internal(format!("CUDA assertion operand copy failed with driver error {copy_result}",)));
    }
    let synchronize_result = unsafe { cuStreamSynchronize(stream) };
    if synchronize_result != CUDA_SUCCESS {
        return Err(FfiError::internal(format!(
            "CUDA assertion operand synchronization failed with driver error {synchronize_result}",
        )));
    }
    Ok(())
}

/// Reads `BYTE_COUNT` bytes from a validated scalar assertion buffer.
fn scalar_bytes<const BYTE_COUNT: usize>(
    buffer: &FfiBuffer<'_>,
    memory: AssertionBufferMemory,
) -> Result<[u8; BYTE_COUNT], FfiError> {
    // SAFETY: The scalar readers validate the element type and rank before calling this function, so XLA owns at
    // least `BYTE_COUNT` bytes at the returned invocation-scoped address.
    let source = unsafe { buffer.data() };
    if source.is_null() {
        return Err(FfiError::invalid_argument(format!(
            "encountered a null scalar buffer in `{ASSERT_CUSTOM_CALL_TARGET}`",
        )));
    }
    let mut bytes = [0; BYTE_COUNT];
    match memory {
        AssertionBufferMemory::Host => unsafe {
            std::ptr::copy_nonoverlapping(source.cast::<u8>(), bytes.as_mut_ptr(), BYTE_COUNT);
        },
        #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
        AssertionBufferMemory::Cuda(stream) => {
            copy_cuda_bytes(source, bytes.as_mut_ptr().cast(), BYTE_COUNT, stream)?;
        }
    }
    Ok(bytes)
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
    if buffers.len() < 2 {
        return Err(FfiError::invalid_argument(format!(
            "expected the `{ASSERT_CUSTOM_CALL_TARGET}` custom call to receive a predicate and observed extents"
        )));
    }
    let actor = string_attribute(call_frame, ASSERT_ACTOR_ATTRIBUTE)?;
    let kind = string_attribute(call_frame, ASSERT_KIND_ATTRIBUTE)?;
    if kind == ASSERT_CONCATENATE_KIND {
        // Concatenation is variadic, so its callback recomputes the checked sum from every input extent instead of
        // relying on a fixed-arity predicate produced in StableHLO.
        if buffers.len() < 3 {
            return Err(FfiError::invalid_argument(format!(
                "expected the `{ASSERT_CUSTOM_CALL_TARGET}` concatenate assertion to receive a result extent and at \
                 least one input extent"
            )));
        }
        let actual = scalar_i64(&buffers[1], memory)?;
        let input_extents =
            buffers[2..].iter().map(|buffer| scalar_i64(buffer, memory)).collect::<Result<Vec<_>, _>>()?;
        let axis = string_attribute(call_frame, ASSERT_DETAIL_ATTRIBUTE)?;
        return validate_concatenate(actor, axis, actual, input_extents.as_slice()).map_err(FfiError::invalid_argument);
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
        let input_size = scalar_i64(&buffers[1], memory)?;
        let start = scalar_i64(&buffers[2], memory)?;
        let size = scalar_i64(&buffers[3], memory)?;
        return validate_dynamic_shape_slice(axis, stride, input_size, start, size).map_err(FfiError::invalid_argument);
    }

    let arithmetic = matches!(
        kind,
        ASSERT_ADD_KIND | ASSERT_SUB_KIND | ASSERT_MUL_KIND | ASSERT_POW_KIND | ASSERT_DIV_FLOOR_KIND | ASSERT_REM_KIND
    );
    if arithmetic {
        if buffers.len() != 3 {
            return Err(FfiError::invalid_argument(format!(
                "expected the `{ASSERT_CUSTOM_CALL_TARGET}` arithmetic assertion to receive two extents"
            )));
        }
        let left_name = string_attribute(call_frame, ASSERT_LEFT_ATTRIBUTE)?;
        let left = scalar_i64(&buffers[1], memory)?;
        let right_name = string_attribute(call_frame, ASSERT_RIGHT_ATTRIBUTE)?;
        let right = scalar_i64(&buffers[2], memory)?;
        return validate_arithmetic(kind, left_name, left, right_name, right).map_err(FfiError::invalid_argument);
    }

    let expected_extent_count = match kind {
        ASSERT_EQUAL_KIND | ASSERT_LESS_THAN_OR_EQUAL_KIND | ASSERT_DIVISIBLE_BY_KIND => 2,
        ASSERT_BOUNDS_KIND => 1,
        _ => {
            return Err(FfiError::invalid_argument(format!(
                "unsupported `{ASSERT_KIND_ATTRIBUTE}` value `{kind}` for `{ASSERT_CUSTOM_CALL_TARGET}`"
            )));
        }
    };
    if buffers.len() != expected_extent_count + 1 {
        return Err(FfiError::invalid_argument(format!(
            "expected the `{ASSERT_CUSTOM_CALL_TARGET}` requirement assertion to receive {expected_extent_count} \
             extent(s)"
        )));
    }
    if scalar_predicate(&buffers[0], memory)? {
        return Ok(());
    }
    let left_name = string_attribute(call_frame, ASSERT_LEFT_ATTRIBUTE)?;
    let left = scalar_i64(&buffers[1], memory)?;
    let message = match kind {
        ASSERT_EQUAL_KIND | ASSERT_LESS_THAN_OR_EQUAL_KIND | ASSERT_DIVISIBLE_BY_KIND => {
            let right_name = string_attribute(call_frame, ASSERT_RIGHT_ATTRIBUTE)?;
            let right = scalar_i64(&buffers[2], memory)?;
            let requirement = match kind {
                ASSERT_EQUAL_KIND => format!("{left_name} == {right_name}"),
                ASSERT_LESS_THAN_OR_EQUAL_KIND => format!("{left_name} <= {right_name}"),
                ASSERT_DIVISIBLE_BY_KIND if right == 0 => format!("{right_name} > 0 for divisibility"),
                ASSERT_DIVISIBLE_BY_KIND => format!("{left_name} % {right_name} == 0"),
                _ => unreachable!(),
            };
            format!("`{actor}` failed: {requirement}; observed {left_name}={left}, {right_name}={right}")
        }
        ASSERT_BOUNDS_KIND => {
            let bounds = string_attribute(call_frame, ASSERT_DETAIL_ATTRIBUTE)?;
            format!("`{actor}` failed: input dimension `{left_name}` = {left} is outside its declared bounds {bounds}")
        }
        _ => unreachable!(),
    };
    Err(FfiError::invalid_argument(message))
}

/// Evaluates one checked dimension-arithmetic predicate and returns its eager-compatible diagnostic on failure.
fn validate_arithmetic(kind: &str, left_name: &str, left: i64, right_name: &str, right: i64) -> Result<(), String> {
    // This deliberately duplicates `ryft-core`'s crate-private `checked_power` rather than widening that helper's
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
        ASSERT_DIV_FLOOR_KIND | ASSERT_REM_KIND => right > 0,
        _ => return Err(format!("unsupported arithmetic assertion kind `{kind}`")),
    };
    if valid {
        return Ok(());
    }
    Err(match kind {
        ASSERT_ADD_KIND => format!(
            "dimension arithmetic overflow while adding dimensions with operands {left_name}={left}, \
             {right_name}={right}",
        ),
        ASSERT_SUB_KIND => format!("{left_name} >= {right_name}; observed {left_name}={left}, {right_name}={right}"),
        ASSERT_MUL_KIND => format!(
            "dimension arithmetic overflow while multiplying dimensions with operands {left_name}={left}, \
             {right_name}={right}",
        ),
        ASSERT_POW_KIND => format!(
            "dimension arithmetic overflow while raising a dimension to a dimension power with operands \
             {left_name}={left}, {right_name}={right}",
        ),
        ASSERT_DIV_FLOOR_KIND | ASSERT_REM_KIND => {
            format!("{right_name} > 0; observed {left_name}={left}, {right_name}={right}")
        }
        _ => unreachable!(),
    })
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
fn validate_dynamic_shape_slice(
    axis: usize,
    stride: usize,
    input_size: i64,
    start: i64,
    size: i64,
) -> Result<(), String> {
    let stride = i64::try_from(stride).map_err(|_| {
        format!(
            "`{DYNAMIC_SHAPE_SLICE_OPERATION_NAME}` stride is outside the portable dimension range on axis \
                 {axis}",
        )
    })?;
    let span = if size == 0 {
        0
    } else {
        size.checked_sub(1)
            .and_then(|size| size.checked_mul(stride))
            .and_then(|span| span.checked_add(1))
            .ok_or_else(|| format!("`{DYNAMIC_SHAPE_SLICE_OPERATION_NAME}` span overflows usize on axis {axis}"))?
    };
    let limit = start
        .checked_add(span)
        .ok_or_else(|| format!("`{DYNAMIC_SHAPE_SLICE_OPERATION_NAME}` limit overflows usize on axis {axis}"))?;
    if limit > input_size {
        return Err(format!(
            "`{DYNAMIC_SHAPE_SLICE_OPERATION_NAME}` limit {limit} exceeds input axis {axis} extent {input_size}",
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

/// Reads one rank-zero signed 64-bit extent buffer.
fn scalar_i64(buffer: &FfiBuffer<'_>, memory: AssertionBufferMemory) -> Result<i64, FfiError> {
    if buffer.element_type() != FfiBufferType::I64 || buffer.rank() != 0 {
        return Err(FfiError::invalid_argument(format!(
            "expected the `{ASSERT_CUSTOM_CALL_TARGET}` observed extent to be a rank-zero i64 buffer"
        )));
    }
    Ok(i64::from_ne_bytes(scalar_bytes(buffer, memory)?))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{
        DimensionBounds, DimensionError, DimensionPow, DimensionType, DimensionValue, DimensionVariable, Div,
        ProgramError, Rem, Sub,
    };

    use super::*;

    /// Builds an eager dimension value whose diagnostic variable name matches the FFI-reported operand name.
    fn eager_dimension(name: &str, extent: usize) -> DimensionValue {
        DimensionValue::new(DimensionType::new(DimensionVariable::new(name, DimensionBounds::unbounded())), extent)
            .unwrap()
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
    fn test_validate_arithmetic_matches_eager_checked_diagnostics() {
        // The host callback recomputes each checked predicate, so its wording must match the eager path exactly for
        // the same named operands; these pairs fail on both paths and must render identically.
        let compiled = |kind, left, right| validate_arithmetic(kind, "left", left, "right", right).unwrap_err();
        assert_eq!(
            compiled(ASSERT_SUB_KIND, 2, 5),
            eager_message(eager_dimension("left", 2).sub(&eager_dimension("right", 5)).unwrap_err()),
        );
        assert_eq!(
            compiled(ASSERT_DIV_FLOOR_KIND, 7, 0),
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

        // Addition and multiplication of two constructible extents cannot overflow the eager `usize` computation on
        // 64-bit hosts (the sum of two values at most `i64::MAX` fits in `usize`, and construction already rejects
        // larger extents with the backend-width diagnostic), so their overflow wording is pinned against the shared
        // template instead of a reproduced eager error.
        assert_eq!(
            compiled(ASSERT_ADD_KIND, i64::MAX - 1, 2),
            "dimension arithmetic overflow while adding dimensions with operands left=9223372036854775806, right=2",
        );
        assert_eq!(
            compiled(ASSERT_MUL_KIND, i64::MAX / 2, 3),
            "dimension arithmetic overflow while multiplying dimensions with operands left=4611686018427387903, \
             right=3",
        );
    }

    #[test]
    fn test_validate_arithmetic() {
        assert_eq!(validate_arithmetic(ASSERT_ADD_KIND, "left", 2, "right", 3), Ok(()));
        assert_eq!(validate_arithmetic(ASSERT_SUB_KIND, "left", 3, "right", 2), Ok(()));
        assert_eq!(validate_arithmetic(ASSERT_MUL_KIND, "left", 2, "right", 3), Ok(()));
        assert_eq!(validate_arithmetic(ASSERT_POW_KIND, "left", 2, "right", 3), Ok(()));
        assert_eq!(validate_arithmetic(ASSERT_DIV_FLOOR_KIND, "left", 7, "right", 3), Ok(()));
        assert_eq!(validate_arithmetic(ASSERT_REM_KIND, "left", 7, "right", 3), Ok(()));

        assert_eq!(
            validate_arithmetic(ASSERT_ADD_KIND, "left", i64::MAX, "right", 1),
            Err("dimension arithmetic overflow while adding dimensions with operands left=9223372036854775807, \
                 right=1"
                .to_string(),),
        );
        assert_eq!(
            validate_arithmetic(ASSERT_SUB_KIND, "left", 2, "right", 3),
            Err("left >= right; observed left=2, right=3".to_string()),
        );
        assert_eq!(
            validate_arithmetic(ASSERT_MUL_KIND, "left", i64::MAX, "right", 2),
            Err("dimension arithmetic overflow while multiplying dimensions with operands left=9223372036854775807, \
                 right=2"
                .to_string(),),
        );
        assert_eq!(
            validate_arithmetic(ASSERT_POW_KIND, "left", i64::MAX, "right", 2),
            Err("dimension arithmetic overflow while raising a dimension to a dimension power with operands \
                 left=9223372036854775807, right=2"
                .to_string(),),
        );
        assert_eq!(
            validate_arithmetic(ASSERT_DIV_FLOOR_KIND, "left", 7, "right", 0),
            Err("right > 0; observed left=7, right=0".to_string()),
        );
        assert_eq!(
            validate_arithmetic(ASSERT_REM_KIND, "left", 7, "right", 0),
            Err("right > 0; observed left=7, right=0".to_string()),
        );
        assert_eq!(
            validate_arithmetic("unknown", "left", 1, "right", 1),
            Err("unsupported arithmetic assertion kind `unknown`".to_string()),
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
    fn test_validate_dynamic_shape_slice() {
        assert_eq!(validate_dynamic_shape_slice(1, 2, 8, 1, 3), Ok(()));
        assert_eq!(validate_dynamic_shape_slice(1, 2, 8, 8, 0), Ok(()));
        assert_eq!(
            validate_dynamic_shape_slice(1, 2, 8, 3, 4),
            Err("`dynamic_shape_slice` limit 10 exceeds input axis 1 extent 8".to_string()),
        );
        assert_eq!(
            validate_dynamic_shape_slice(1, usize::MAX, 8, 0, 2),
            Err("`dynamic_shape_slice` stride is outside the portable dimension range on axis 1".to_string()),
        );
        assert_eq!(
            validate_dynamic_shape_slice(1, i64::MAX as usize, i64::MAX, 0, 3),
            Err("`dynamic_shape_slice` span overflows usize on axis 1".to_string()),
        );
        assert_eq!(
            validate_dynamic_shape_slice(1, 1, i64::MAX, i64::MAX, 1),
            Err("`dynamic_shape_slice` limit overflows usize on axis 1".to_string()),
        );
    }
}
