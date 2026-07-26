use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::parameters::Parameter;
use crate::types::{DimensionBounds, DimensionError, DimensionType};

use super::representable_extent_range;

/// Canonical operation name for [`DimensionMinimumOperation`].
pub const DIMENSION_MINIMUM_OPERATION_NAME: &str = "dimension_minimum";

define_arithmetic_dimension_operation!(
    /// Dimension-minimum operation used by [`DimensionMinimum`].
    ///
    /// Refer to [`DimensionMinimum`] for semantic details and an example.
    DimensionMinimumOperation, DIMENSION_MINIMUM_OPERATION_NAME,
    DimensionMinimum, minimum_dimension_with,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("min({}, {})", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

define_arithmetic_dimension_capability!(
    /// Returns the smaller of two runtime dimensions.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionMinimum, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(7)?.minimum_dimension(&DimensionValue::constant(3)?)?;
    /// assert_eq!(result.extent(), 3);
    /// # Ok(())
    /// # }
    /// ```
    DimensionMinimum,
    /// Returns `min(self, right)`.
    minimum_dimension(right),
    minimum_dimension_with(right, operation),
    DimensionMinimumOperation,
);

/// Derives sound bounds for dimension minimum.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    DimensionBounds::new(left_lower.min(right_lower), left_maximum.min(right_maximum).checked_add(1))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::DimensionValue;
    use crate::types::DimensionBounds;

    use super::super::test_dimension_type;
    use super::*;

    #[test]
    fn test_dimension_minimum_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionMinimumOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_MINIMUM_OPERATION_NAME);
        assert_eq!(operation.result_type().bounds(), DimensionBounds::new(1, Some(5)).unwrap());
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .minimum_dimension(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            3,
        );
    }
}
