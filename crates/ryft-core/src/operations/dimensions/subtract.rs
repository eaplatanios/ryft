use crate::macros::define_arithmetic_dimension_operation;
use crate::parameters::Parameter;
use crate::types::{DimensionError, DimensionType};

use super::{bounds_from_extrema, representable_extent_range, requirement_violation};

/// Canonical operation name for [`DimensionSubtractOperation`].
pub const DIMENSION_SUBTRACT_OPERATION_NAME: &str = "dimension_subtract";

define_arithmetic_dimension_operation!(
    /// Subtracts one first-class runtime dimension from another and rejects negative results.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionSubtract, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(7)?.subtract_dimension(&DimensionValue::constant(3)?)?;
    /// assert_eq!(result.extent(), 4);
    /// assert!(DimensionValue::constant(3)?.subtract_dimension(&DimensionValue::constant(7)?).is_err());
    /// # Ok(())
    /// # }
    /// ```
    capability DimensionSubtract {
        /// Returns `self - right`, failing if `right` is greater than `self`.
        fn subtract_dimension;
    }
    /// Checked dimension-subtraction operation used by [`DimensionSubtract`].
    ///
    /// Refer to [`DimensionSubtract`] for semantic details and an example.
    operation DimensionSubtractOperation {
        name = DIMENSION_SUBTRACT_OPERATION_NAME,
        result_name = |left: &DimensionType, right: &DimensionType| {
            format!("{} - {}", left.variable(), right.variable())
        },
        infer_bounds = infer_bounds,
        evaluate = evaluate,
    }
);

/// Derives sound bounds for checked dimension subtraction.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<crate::DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    if left_maximum < right_lower {
        return Err(DimensionError::RequirementViolation {
            message: format!("{} >= {} is impossible from declared bounds", left.variable(), right.variable()),
        });
    }
    bounds_from_extrema(left_lower.saturating_sub(right_maximum), left_maximum - right_lower)
}

/// Evaluates checked dimension subtraction.
fn evaluate(
    left_type: &DimensionType,
    left: usize,
    right_type: &DimensionType,
    right: usize,
) -> Result<usize, DimensionError> {
    left.checked_sub(right).ok_or_else(|| {
        requirement_violation(
            format!("{} >= {}", left_type.variable(), right_type.variable()),
            left_type,
            left,
            right_type,
            right,
        )
    })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::DimensionValue;
    use crate::operations::dimensions::ArithmeticDimensionOperation;
    use crate::types::{DimensionBounds, DimensionError};

    use super::super::test_dimension_type;
    use super::*;

    #[test]
    fn test_dimension_subtract_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionSubtractOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_SUBTRACT_OPERATION_NAME);
        assert_eq!(operation.result_type().bounds(), DimensionBounds::new(0, Some(8)).unwrap());
        assert_eq!(operation.evaluate_extents(7, 3), Ok(4));
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .subtract_dimension(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            4,
        );
        assert_eq!(
            operation.evaluate_extents(1, 3),
            Err(DimensionError::RequirementViolation {
                message: "left >= right; observed left=1, right=3".to_string(),
            }),
        );
    }
}
