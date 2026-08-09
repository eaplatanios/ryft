//! Operand-boundary helpers shared by the higher-order tracing operations.
//!
//! [`CustomJvpOperation`](crate::CustomJvpOperation), [`CustomVjpOperation`](crate::CustomVjpOperation), and
//! [`RematerializeOperation`](crate::RematerializeOperation) all draw the same operand split: a leading group of
//! _nondifferentiated_ operands that parameterize the call, followed by the differentiated operands. This module owns
//! the two pieces of that contract they would otherwise duplicate — the split itself and the guard that rejects a live
//! tangent supplied for a nondifferentiated operand — so that neither operation module depends on its sibling.

use crate::differentiation::{DifferentiableType, DifferentiationDual};
use crate::programs::{ProgramError, TypeError, Typed, Value};

/// Splits `values` into the leading nondifferentiated group of `nondifferentiated_count` entries and the trailing
/// differentiated group. A count exceeding the operand list is a malformed operation payload rather than a recoverable
/// condition, and is reported with `name` identifying the owning operation.
///
/// # Parameters
///
///   - `name`: Canonical operation name included in the diagnostic.
///   - `nondifferentiated_count`: Number of leading operands that parameterize the call without being differentiated.
///   - `values`: Per-operand values (or types, or batch axes) in operand order.
pub(crate) fn split_nondifferentiated<'v, V>(
    name: &str,
    nondifferentiated_count: usize,
    values: &'v [V],
) -> Result<(&'v [V], &'v [V]), TypeError> {
    if nondifferentiated_count > values.len() {
        return Err(TypeError::invalid(format!(
            "{name} nondifferentiated operand count {nondifferentiated_count} exceeds input count {}",
            values.len(),
        )));
    }
    Ok(values.split_at(nondifferentiated_count))
}

/// Rejects a live tangent supplied for one of `nondifferentiated_inputs`, which are the leading operands that
/// parameterize a call without being differentiated. Such an operand has no tangent slot anywhere in the owning
/// operation's rule, so a live tangent reaching one would be silently dropped instead of propagated. Every
/// higher-order operation that draws this operand split shares the guard, with `name` selecting the reported
/// operation.
pub(crate) fn check_nondifferentiated_tangents_are_zero<V: Value>(
    name: &str,
    nondifferentiated_inputs: &[DifferentiationDual<V>],
) -> Result<(), ProgramError>
where
    V::Type: DifferentiableType,
{
    match nondifferentiated_inputs
        .iter()
        .find(|input| !input.tangent().is_zero() && !input.tangent().r#type().is_zero_space())
    {
        None => Ok(()),
        Some(input) => Err(ProgramError::UnsupportedOperation {
            message: format!(
                "{name} cannot propagate the nonzero tangent of type `{}` supplied for one of its {} leading \
                 nondifferentiated operands, because its rule has no tangent slot for them",
                input.tangent().r#type(),
                nondifferentiated_inputs.len(),
            ),
        }),
    }
}
