use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::parameters::Parameter;
use crate::types::{DimensionBounds, DimensionError, DimensionType};

use super::{positive_divisor_lower_bound, representable_extent_range, requirement_violation};

/// Canonical operation name for [`DimensionRemainderOperation`].
pub const DIMENSION_REMAINDER_OPERATION_NAME: &str = "dimension_remainder";

define_arithmetic_dimension_operation!(
    /// Checked dimension-remainder operation used by [`DimensionRemainder`].
    ///
    /// Refer to [`DimensionRemainder`] for semantic details and an example.
    DimensionRemainderOperation, DIMENSION_REMAINDER_OPERATION_NAME,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} % {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
    evaluate = evaluate,
);

define_arithmetic_dimension_capability!(
    /// Computes the remainder of one runtime dimension divided by a positive runtime dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionRemainder, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(7)?.remainder_dimension(&DimensionValue::constant(3)?)?;
    /// assert_eq!(result.extent(), 1);
    /// # Ok(())
    /// # }
    /// ```
    DimensionRemainder,
    /// Returns `self % right`, failing when `right` is zero.
    remainder_dimension(right),
    DimensionRemainderOperation,
);

/// Derives sound bounds for checked dimension remainder.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<DimensionBounds, DimensionError> {
    let (_, left_maximum) = representable_extent_range(left.bounds())?;
    let (_, right_maximum) = representable_extent_range(right.bounds())?;
    positive_divisor_lower_bound(right, right_maximum)?;
    DimensionBounds::new(0, left_maximum.min(right_maximum - 1).checked_add(1))
}

/// Evaluates checked dimension remainder.
fn evaluate(
    left_type: &DimensionType,
    left: usize,
    right_type: &DimensionType,
    right: usize,
) -> Result<usize, DimensionError> {
    if right == 0 {
        Err(requirement_violation(format!("{} > 0", right_type.variable()), left_type, left, right_type, right))
    } else {
        Ok(left % right)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::DimensionValue;
    use crate::types::{DimensionBounds, DimensionError};

    use super::super::test_dimension_type;
    use super::*;

    #[test]
    fn test_dimension_remainder_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionRemainderOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_REMAINDER_OPERATION_NAME);
        assert_eq!(operation.result_type().bounds(), DimensionBounds::new(0, Some(4)).unwrap());
        assert_eq!(evaluate(&left, 7, &right, 3), Ok(1));
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .remainder_dimension(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            1,
        );
        assert_eq!(
            evaluate(&left, 7, &right, 0),
            Err(DimensionError::RequirementViolation { message: "right > 0; observed left=7, right=0".to_string() }),
        );
    }
}
