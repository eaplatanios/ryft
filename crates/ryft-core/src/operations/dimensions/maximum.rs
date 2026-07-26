use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::parameters::Parameter;
use crate::types::{DimensionBounds, DimensionError, DimensionType};

use super::representable_extent_range;

/// Canonical operation name for [`DimensionMaximumOperation`].
pub const DIMENSION_MAXIMUM_OPERATION_NAME: &str = "dimension_maximum";

define_arithmetic_dimension_operation!(
    /// Dimension-maximum operation used by [`DimensionMaximum`].
    ///
    /// Refer to [`DimensionMaximum`] for semantic details and an example.
    DimensionMaximumOperation, DIMENSION_MAXIMUM_OPERATION_NAME,
    DimensionMaximum, maximum_dimension_with,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("max({}, {})", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

define_arithmetic_dimension_capability!(
    /// Returns the larger of two runtime dimensions.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionMaximum, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(7)?.maximum_dimension(&DimensionValue::constant(3)?)?;
    /// assert_eq!(result.extent(), 7);
    /// # Ok(())
    /// # }
    /// ```
    DimensionMaximum,
    /// Returns `max(self, right)`.
    maximum_dimension(right),
    maximum_dimension_with(right, operation),
    DimensionMaximumOperation,
);

/// Derives sound bounds for dimension maximum.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    DimensionBounds::new(left_lower.max(right_lower), left_maximum.max(right_maximum).checked_add(1))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::DimensionValue;
    use crate::types::DimensionBounds;

    use super::super::test_dimension_type;
    use super::*;

    #[test]
    fn test_dimension_maximum_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionMaximumOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_MAXIMUM_OPERATION_NAME);
        assert_eq!(operation.result_type().bounds(), DimensionBounds::new(2, Some(9)).unwrap());
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .maximum_dimension(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            7,
        );
    }
}
