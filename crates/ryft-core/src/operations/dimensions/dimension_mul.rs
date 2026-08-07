use crate::arrays::{DimensionBounds, DimensionError, DimensionType, MAX_DIMENSION_EXTENT};
use crate::macros::{check_count, define_arithmetic_dimension_operation};
use crate::operations::math::mul::{Mul, MulOperation};
use crate::parameters::Parameter;
use crate::programs::{OperationProvider, ProgramError};

use super::{bounds_overflow, maximum_extent, representable_extent_range};

/// Canonical operation name for [`DimensionMulOperation`].
pub const DIMENSION_MUL_OPERATION_NAME: &str = "dimension_mul";

define_arithmetic_dimension_operation!(
    /// Checked dimension-multiplication operation used by [`Mul`].
    DimensionMulOperation, DIMENSION_MUL_OPERATION_NAME,
    Mul, mul,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} * {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

impl OperationProvider<DimensionType> for MulOperation<DimensionType> {
    type Operation = DimensionMulOperation;

    fn provide(input_types: &[&DimensionType]) -> Result<Self::Operation, ProgramError> {
        check_count!("input", input_types, 2, ProgramError);
        Ok(DimensionMulOperation::new(input_types[0], input_types[1])?)
    }
}

/// Derives sound bounds for checked dimension multiplication and reports whether runtime overflow remains possible.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<(DimensionBounds, bool), DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    let overflow = || bounds_overflow(DIMENSION_MUL_OPERATION_NAME, left, right);
    let lower = left_lower.checked_mul(right_lower).ok_or_else(overflow)?;
    let maximum = left_maximum.saturating_mul(right_maximum).min(MAX_DIMENSION_EXTENT);
    let bounds = DimensionBounds::new(lower, maximum.checked_add(1))?;
    let requires_runtime_assertion = maximum_extent(left)
        .zip(maximum_extent(right))
        .and_then(|(left, right)| left.checked_mul(right))
        .is_none_or(|result| result > MAX_DIMENSION_EXTENT);
    Ok((bounds, requires_runtime_assertion))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionValue};
    use crate::operations::dimensions::test_dimension_type;
    use crate::operations::math::mul::Mul;

    use super::*;

    #[test]
    fn test_dimension_mul_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionMulOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_MUL_OPERATION_NAME);
        assert_eq!(operation.result_bounds(), DimensionBounds::new(2, Some(33)).unwrap());
        assert_eq!(
            DimensionValue::constant(7).unwrap().mul(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            21,
        );
    }
}
