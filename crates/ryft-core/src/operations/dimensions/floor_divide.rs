use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::parameters::Parameter;
use crate::types::{DimensionBounds, DimensionError, DimensionType};

use super::{positive_divisor_lower_bound, representable_extent_range, requirement_violation};

/// Canonical operation name for [`DimensionFloorDivideOperation`].
pub const DIMENSION_FLOOR_DIVIDE_OPERATION_NAME: &str = "dimension_floor_divide";

define_arithmetic_dimension_operation!(
    /// Checked dimension-floor-division operation used by [`DimensionFloorDivide`].
    ///
    /// Refer to [`DimensionFloorDivide`] for semantic details and an example.
    DimensionFloorDivideOperation, DIMENSION_FLOOR_DIVIDE_OPERATION_NAME,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} // {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
    evaluate = evaluate,
);

define_arithmetic_dimension_capability!(
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
    DimensionFloorDivide,
    /// Returns `self // right`, failing when `right` is zero.
    floor_divide_dimension(right),
    DimensionFloorDivideOperation,
);

/// Derives sound bounds for checked dimension floor division.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (_, right_maximum) = representable_extent_range(right.bounds())?;
    let positive_right_lower = positive_divisor_lower_bound(right, right_maximum)?;
    DimensionBounds::new(left_lower / right_maximum, (left_maximum / positive_right_lower).checked_add(1))
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
        assert_eq!(evaluate(&left, 7, &right, 3), Ok(2));
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .floor_divide_dimension(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            2,
        );
        assert_eq!(
            evaluate(&left, 7, &right, 0),
            Err(DimensionError::RequirementViolation { message: "right > 0; observed left=7, right=0".to_string() }),
        );
    }
}
