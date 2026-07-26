use crate::macros::define_arithmetic_dimension_operation;
use crate::parameters::Parameter;
use crate::types::{DimensionError, DimensionType};

use super::{bounds_from_extrema, positive_divisor_lower_bound, representable_extent_range, requirement_violation};

/// Canonical operation name for [`DimensionFloorDivideOperation`].
pub const DIMENSION_FLOOR_DIVIDE_OPERATION_NAME: &str = "dimension_floor_divide";

define_arithmetic_dimension_operation!(
    /// Floor-divides one runtime dimension by a positive runtime dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionFloorDivide, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(7)?.floor_divide_dimension(&DimensionValue::constant(3)?)?;
    /// assert_eq!(result.extent(), 2);
    /// # Ok(())
    /// # }
    /// ```
    capability DimensionFloorDivide {
        /// Returns `self // right`, failing when `right` is zero.
        fn floor_divide_dimension;
    }
    /// Checked dimension-floor-division operation used by [`DimensionFloorDivide`].
    ///
    /// Refer to [`DimensionFloorDivide`] for semantic details and an example.
    operation DimensionFloorDivideOperation {
        name = DIMENSION_FLOOR_DIVIDE_OPERATION_NAME,
        result_name = |left: &DimensionType, right: &DimensionType| {
            format!("{} // {}", left.variable(), right.variable())
        },
        infer_bounds = infer_bounds,
        evaluate = evaluate,
    }
);

/// Derives sound bounds for checked dimension floor division.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<crate::DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (_, right_maximum) = representable_extent_range(right.bounds())?;
    let positive_right_lower = positive_divisor_lower_bound(right, right_maximum)?;
    bounds_from_extrema(left_lower / right_maximum, left_maximum / positive_right_lower)
}

/// Evaluates checked dimension floor division.
fn evaluate(
    left_type: &DimensionType,
    left: usize,
    right_type: &DimensionType,
    right: usize,
) -> Result<usize, DimensionError> {
    if right == 0 {
        Err(requirement_violation(format!("{} > 0", right_type.variable()), left_type, left, right_type, right))
    } else {
        Ok(left / right)
    }
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
    fn test_dimension_floor_divide_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionFloorDivideOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_FLOOR_DIVIDE_OPERATION_NAME);
        assert_eq!(operation.result_type().bounds(), DimensionBounds::new(0, Some(9)).unwrap());
        assert_eq!(operation.evaluate_extents(7, 3), Ok(2));
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .floor_divide_dimension(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            2,
        );
        assert_eq!(
            operation.evaluate_extents(7, 0),
            Err(DimensionError::RequirementViolation { message: "right > 0; observed left=7, right=0".to_string() }),
        );
    }
}
