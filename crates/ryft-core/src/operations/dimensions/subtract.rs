use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::parameters::Parameter;
use crate::types::{DimensionBounds, DimensionError, DimensionType};

use super::representable_extent_range;

/// Canonical operation name for [`DimensionSubtractOperation`].
pub const DIMENSION_SUBTRACT_OPERATION_NAME: &str = "dimension_subtract";

define_arithmetic_dimension_operation!(
    /// Checked dimension-subtraction operation used by [`DimensionSubtract`].
    ///
    /// Refer to [`DimensionSubtract`] for semantic details and an example.
    DimensionSubtractOperation, DIMENSION_SUBTRACT_OPERATION_NAME,
    DimensionSubtract, subtract_dimension_with,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} - {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

define_arithmetic_dimension_capability!(
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
    DimensionSubtract,
    /// Returns `self - right`, failing if `right` is greater than `self`.
    subtract_dimension(right),
    subtract_dimension_with(right, operation),
    DimensionSubtractOperation,
);

/// Derives sound bounds for checked dimension subtraction.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    if left_maximum < right_lower {
        return Err(DimensionError::RequirementViolation {
            message: format!("{} >= {} is impossible from declared bounds", left.variable(), right.variable()),
        });
    }
    DimensionBounds::new(left_lower.saturating_sub(right_maximum), (left_maximum - right_lower).checked_add(1))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::DimensionValue;
    use crate::types::DimensionBounds;

    use super::super::test_dimension_type;
    use super::*;

    #[test]
    fn test_dimension_subtract_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionSubtractOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_SUBTRACT_OPERATION_NAME);
        assert_eq!(operation.result_type().bounds(), DimensionBounds::new(0, Some(8)).unwrap());
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .subtract_dimension(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            4,
        );
    }
}
