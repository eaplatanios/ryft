use crate::arrays::{DimensionBounds, DimensionError, DimensionType};
use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::parameters::Parameter;

use super::representable_extent_range;

/// Canonical operation name for [`DimensionMaxOperation`].
pub const DIMENSION_MAX_OPERATION_NAME: &str = "dimension_max";

define_arithmetic_dimension_operation!(
    /// Dimension-maximum operation used by [`DimensionMax`].
    ///
    /// Refer to [`DimensionMax`] for semantic details and an example.
    DimensionMaxOperation, DIMENSION_MAX_OPERATION_NAME,
    DimensionMax, dimension_max,
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
    /// # use ryft_core::{DimensionMax, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(7)?.dimension_max(&DimensionValue::constant(3)?)?;
    /// assert_eq!(result.extent(), 7);
    /// # Ok(())
    /// # }
    /// ```
    DimensionMax,
    /// Returns `max(self, right)`.
    dimension_max(right),
    DimensionMaxOperation,
);

/// Derives sound bounds for total dimension maximum.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<(DimensionBounds, bool), DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    let bounds = DimensionBounds::new(left_lower.max(right_lower), left_maximum.max(right_maximum).checked_add(1))?;
    Ok((bounds, false))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionValue};
    use crate::operations::dimensions::test_dimension_type;

    use super::*;

    #[test]
    fn test_dimension_max_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionMaxOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_MAX_OPERATION_NAME);
        assert_eq!(operation.result_bounds(), DimensionBounds::new(2, Some(9)).unwrap());
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .dimension_max(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            7,
        );
    }
}
