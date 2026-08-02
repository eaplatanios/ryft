use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::parameters::Parameter;
use crate::types::{DimensionBounds, DimensionError, DimensionType};

use super::representable_extent_range;

/// Canonical operation name for [`DimensionMinOperation`].
pub const DIMENSION_MIN_OPERATION_NAME: &str = "dimension_min";

define_arithmetic_dimension_operation!(
    /// Dimension-minimum operation used by [`DimensionMin`].
    ///
    /// Refer to [`DimensionMin`] for semantic details and an example.
    DimensionMinOperation, DIMENSION_MIN_OPERATION_NAME,
    DimensionMin, dimension_min,
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
    /// # use ryft_core::{DimensionMin, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(7)?.dimension_min(&DimensionValue::constant(3)?)?;
    /// assert_eq!(result.extent(), 3);
    /// # Ok(())
    /// # }
    /// ```
    DimensionMin,
    /// Returns `min(self, right)`.
    dimension_min(right),
    DimensionMinOperation,
);

/// Derives sound bounds for total dimension minimum.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<(DimensionBounds, bool), DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    let bounds = DimensionBounds::new(left_lower.min(right_lower), left_maximum.min(right_maximum).checked_add(1))?;
    Ok((bounds, false))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::DimensionValue;
    use crate::types::DimensionBounds;

    use super::super::test_dimension_type;
    use super::*;

    #[test]
    fn test_dimension_min_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionMinOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_MIN_OPERATION_NAME);
        assert_eq!(operation.result_bounds(), DimensionBounds::new(1, Some(5)).unwrap());
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .dimension_min(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            3,
        );
    }
}
