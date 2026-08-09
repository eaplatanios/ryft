//! Operand-boundary helpers shared by the higher-order tracing operations.
//!
//! [`CustomJvpOperation`](crate::CustomJvpOperation), [`CustomVjpOperation`](crate::CustomVjpOperation), and
//! [`RematerializeOperation`](crate::RematerializeOperation) all draw the same operand split: a leading group of
//! _non-differentiated_ operands that parameterize the call, followed by the differentiated operands. This module owns
//! the two pieces of that contract they would otherwise duplicate — the split itself and the guard that rejects a live
//! tangent supplied for a non-differentiated operand — so that neither operation module depends on its sibling.

use crate::differentiation::{DifferentiableType, DifferentiationDual};
use crate::programs::{ProgramError, TypeError, Typed, Value};

/// Splits `values` into the leading non-differentiated group of `non_differentiated_count` entries and the trailing
/// differentiated group. A count exceeding the operand list is a malformed operation payload rather than a recoverable
/// condition, and is reported with `name` identifying the owning operation.
///
/// # Parameters
///
///   - `name`: Canonical operation name included in the diagnostic.
///   - `non_differentiated_count`: Number of leading operands that parameterize the call without being differentiated.
///   - `values`: Per-operand values (or types, or batch axes) in operand order.
pub(crate) fn split_non_differentiated<'v, V>(
    name: &str,
    non_differentiated_count: usize,
    values: &'v [V],
) -> Result<(&'v [V], &'v [V]), TypeError> {
    if non_differentiated_count > values.len() {
        return Err(TypeError::invalid(format!(
            "{name} non-differentiated operand count {non_differentiated_count} exceeds input count {}",
            values.len(),
        )));
    }
    Ok(values.split_at(non_differentiated_count))
}

/// Rejects a live tangent supplied for one of `non_differentiated_inputs`, which are the leading operands that
/// parameterize a call without being differentiated. Such an operand has no tangent slot anywhere in the owning
/// operation's rule, so a live tangent reaching one would be silently dropped instead of propagated. Every
/// higher-order operation that draws this operand split shares the guard, with `name` selecting the reported
/// operation.
pub(crate) fn check_non_differentiated_tangents_are_zero<V: Value>(
    name: &str,
    non_differentiated_inputs: &[DifferentiationDual<V>],
) -> Result<(), ProgramError>
where
    V::Type: DifferentiableType,
{
    match non_differentiated_inputs
        .iter()
        .find(|input| !input.tangent().is_zero() && !input.tangent().r#type().is_zero_space())
    {
        None => Ok(()),
        Some(input) => Err(ProgramError::UnsupportedOperation {
            message: format!(
                "{name} cannot propagate the nonzero tangent of type `{}` supplied for one of its {} leading \
                 non-differentiated operands, because its rule has no tangent slot for them",
                input.tangent().r#type(),
                non_differentiated_inputs.len(),
            ),
        }),
    }
}
