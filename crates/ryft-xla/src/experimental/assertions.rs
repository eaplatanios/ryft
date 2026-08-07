use std::sync::OnceLock;

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

/// Registers the [`ASSERT_CUSTOM_CALL_TARGET`] CPU handler with the plugin backing `client`.
///
/// Registration is lazy and process-global because XLA's FFI registry rejects duplicate target registrations. The
/// handler reads scalar operand buffers directly on the host and therefore must only be registered for CPU clients.
pub(crate) fn ensure_assertion_handler_registered(client: &Client<'_>) -> Result<(), Error> {
    static ASSERTION_HANDLER_REGISTRATION: OnceLock<Result<(), Error>> = OnceLock::new();
    ASSERTION_HANDLER_REGISTRATION
        .get_or_init(|| {
            let platform_name = client.platform_name()?.into_owned();
            client.register_ffi_handler(
                ASSERT_CUSTOM_CALL_TARGET,
                platform_name,
                FfiHandler::from(assertion_handler as XLA_FFI_Handler),
                FfiHandlerTraits::NONE,
            )
        })
        .clone()
}

/// XLA FFI handler for [`ASSERT_CUSTOM_CALL_TARGET`].
unsafe extern "C" fn assertion_handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    // SAFETY: XLA owns the call frame for this invocation. All access is localized in `FfiCallFrame` and the checked
    // scalar-buffer readers below.
    unsafe {
        match FfiCallFrame::from_c_api(call_frame) {
            Err(_) => std::ptr::null_mut(),
            Ok(call_frame) if call_frame.register_metadata(FfiTypeId::default()) => std::ptr::null_mut(),
            Ok(call_frame) if call_frame.stage() != FfiExecutionStage::Execution => std::ptr::null_mut(),
            Ok(call_frame) => match call_frame.api() {
                Err(_) => std::ptr::null_mut(),
                Ok(api) => match handle_assertion_call_frame(&call_frame) {
                    Ok(()) => std::ptr::null_mut(),
                    Err(error) => error.to_c_api(api),
                },
            },
        }
    }
}

/// Decodes and evaluates one assertion call frame.
fn handle_assertion_call_frame(call_frame: &FfiCallFrame<'_>) -> Result<(), FfiError> {
    let mut buffers = Vec::new();
    for input in call_frame.inputs() {
        let FfiInput::Buffer { buffer } = input?;
        if buffer.element_type() != FfiBufferType::Token {
            buffers.push(buffer);
        }
    }
    if buffers.len() < 2 {
        return Err(FfiError::invalid_argument(format!(
            "expected the '{ASSERT_CUSTOM_CALL_TARGET}' custom call to receive a predicate and observed extents"
        )));
    }
    let actor = string_attribute(call_frame, ASSERT_ACTOR_ATTRIBUTE)?;
    let kind = string_attribute(call_frame, ASSERT_KIND_ATTRIBUTE)?;
    if kind == ASSERT_CONCATENATE_KIND {
        // Concatenation is variadic, so its callback recomputes the checked sum from every input extent instead of
        // relying on a fixed-arity predicate produced in StableHLO.
        if buffers.len() < 3 {
            return Err(FfiError::invalid_argument(format!(
                "expected the '{ASSERT_CUSTOM_CALL_TARGET}' concatenate assertion to receive a result extent and at \
                 least one input extent"
            )));
        }
        let actual = scalar_i64(&buffers[1])?;
        let input_extents = buffers[2..].iter().map(scalar_i64).collect::<Result<Vec<_>, _>>()?;
        let axis = string_attribute(call_frame, ASSERT_DETAIL_ATTRIBUTE)?;
        return validate_concatenate(actor, axis, actual, input_extents.as_slice()).map_err(FfiError::invalid_argument);
    }

    let arithmetic = matches!(
        kind,
        ASSERT_ADD_KIND | ASSERT_SUB_KIND | ASSERT_MUL_KIND | ASSERT_POW_KIND | ASSERT_DIV_FLOOR_KIND | ASSERT_REM_KIND
    );
    if arithmetic {
        if buffers.len() != 3 {
            return Err(FfiError::invalid_argument(format!(
                "expected the '{ASSERT_CUSTOM_CALL_TARGET}' arithmetic assertion to receive two extents"
            )));
        }
        let left_name = string_attribute(call_frame, ASSERT_LEFT_ATTRIBUTE)?;
        let left = scalar_i64(&buffers[1])?;
        let right_name = string_attribute(call_frame, ASSERT_RIGHT_ATTRIBUTE)?;
        let right = scalar_i64(&buffers[2])?;
        return validate_arithmetic(kind, left_name, left, right_name, right).map_err(FfiError::invalid_argument);
    }

    let expected_extent_count = match kind {
        ASSERT_EQUAL_KIND | ASSERT_LESS_THAN_OR_EQUAL_KIND | ASSERT_DIVISIBLE_BY_KIND => 2,
        ASSERT_BOUNDS_KIND => 1,
        _ => {
            return Err(FfiError::invalid_argument(format!(
                "unsupported '{ASSERT_KIND_ATTRIBUTE}' value '{kind}' for '{ASSERT_CUSTOM_CALL_TARGET}'"
            )));
        }
    };
    if buffers.len() != expected_extent_count + 1 {
        return Err(FfiError::invalid_argument(format!(
            "expected the '{ASSERT_CUSTOM_CALL_TARGET}' requirement assertion to receive {expected_extent_count} \
             extent(s)"
        )));
    }
    if scalar_predicate(&buffers[0])? {
        return Ok(());
    }
    let left_name = string_attribute(call_frame, ASSERT_LEFT_ATTRIBUTE)?;
    let left = scalar_i64(&buffers[1])?;
    let message = match kind {
        ASSERT_EQUAL_KIND | ASSERT_LESS_THAN_OR_EQUAL_KIND | ASSERT_DIVISIBLE_BY_KIND => {
            let right_name = string_attribute(call_frame, ASSERT_RIGHT_ATTRIBUTE)?;
            let right = scalar_i64(&buffers[2])?;
            let requirement = match kind {
                ASSERT_EQUAL_KIND => format!("{left_name} == {right_name}"),
                ASSERT_LESS_THAN_OR_EQUAL_KIND => format!("{left_name} <= {right_name}"),
                ASSERT_DIVISIBLE_BY_KIND if right == 0 => format!("{right_name} > 0 for divisibility"),
                ASSERT_DIVISIBLE_BY_KIND => format!("{left_name} % {right_name} == 0"),
                _ => unreachable!(),
            };
            format!("'{actor}' failed: {requirement}; observed {left_name}={left}, {right_name}={right}")
        }
        ASSERT_BOUNDS_KIND => {
            let bounds = string_attribute(call_frame, ASSERT_DETAIL_ATTRIBUTE)?;
            format!("'{actor}' failed: input dimension `{left_name}` = {left} is outside its declared bounds {bounds}")
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
        _ => return Err(format!("unsupported arithmetic assertion kind '{kind}'")),
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
        .ok_or_else(|| format!("'{actor}' input axis {axis} extent sum overflows the portable dimension range"))?;
    if expected == actual {
        Ok(())
    } else {
        Err(format!(
            "'{actor}' result extent must equal the sum of input axis {axis} extents; expected {expected} but got \
             {actual}",
        ))
    }
}

/// Returns one required string attribute from `call_frame`.
fn string_attribute<'o>(call_frame: &FfiCallFrame<'o>, expected_name: &str) -> Result<&'o str, FfiError> {
    for attribute in call_frame.attributes() {
        let (name, attribute) = attribute?;
        if name == expected_name {
            return match attribute {
                FfiAttribute::String { string } => Ok(string),
                _ => Err(FfiError::invalid_argument(format!(
                    "expected the '{expected_name}' attribute of '{ASSERT_CUSTOM_CALL_TARGET}' to be a string"
                ))),
            };
        }
    }
    Err(FfiError::invalid_argument(format!(
        "missing required '{expected_name}' attribute in the '{ASSERT_CUSTOM_CALL_TARGET}' custom call"
    )))
}

/// Reads one rank-zero predicate buffer.
fn scalar_predicate(buffer: &FfiBuffer<'_>) -> Result<bool, FfiError> {
    if buffer.element_type() != FfiBufferType::Predicate || buffer.rank() != 0 {
        return Err(FfiError::invalid_argument(format!(
            "expected the '{ASSERT_CUSTOM_CALL_TARGET}' predicate to be a rank-zero predicate buffer"
        )));
    }
    // SAFETY: XLA reports a rank-zero predicate buffer, whose CPU allocation contains one byte for this invocation.
    let data = unsafe { buffer.data() as *const u8 };
    if data.is_null() {
        return Err(FfiError::invalid_argument(format!(
            "encountered a null predicate buffer in '{ASSERT_CUSTOM_CALL_TARGET}'"
        )));
    }
    Ok(unsafe { *data != 0 })
}

/// Reads one rank-zero signed 64-bit extent buffer.
fn scalar_i64(buffer: &FfiBuffer<'_>) -> Result<i64, FfiError> {
    if buffer.element_type() != FfiBufferType::I64 || buffer.rank() != 0 {
        return Err(FfiError::invalid_argument(format!(
            "expected the '{ASSERT_CUSTOM_CALL_TARGET}' observed extent to be a rank-zero i64 buffer"
        )));
    }
    // SAFETY: XLA reports a rank-zero i64 buffer, whose CPU allocation contains one `i64` for this invocation.
    let data = unsafe { buffer.data() as *const i64 };
    if data.is_null() {
        return Err(FfiError::invalid_argument(format!(
            "encountered a null extent buffer in '{ASSERT_CUSTOM_CALL_TARGET}'"
        )));
    }
    Ok(unsafe { *data })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::arrays::{DimensionBounds, DimensionError, DimensionType, DimensionValue, DimensionVariable};
    use ryft_core::operations::DimensionPow;
    use ryft_core::{Div, ProgramError, Rem, Sub};

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
            Err("unsupported arithmetic assertion kind 'unknown'".to_string()),
        );
    }

    #[test]
    fn test_validate_concatenate() {
        assert_eq!(validate_concatenate("concatenate", "1", 7, &[2, 3, 2]), Ok(()));
        assert_eq!(
            validate_concatenate("concatenate", "1", 8, &[2, 3, 2]),
            Err("'concatenate' result extent must equal the sum of input axis 1 extents; expected 7 but got 8"
                .to_string(),),
        );
        assert_eq!(
            validate_concatenate("concatenate", "1", 0, &[i64::MAX, 1]),
            Err("'concatenate' input axis 1 extent sum overflows the portable dimension range".to_string()),
        );
    }
}
