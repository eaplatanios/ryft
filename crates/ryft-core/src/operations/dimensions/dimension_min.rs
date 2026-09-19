use crate::arrays::{DimensionBounds, DimensionError, DimensionType};
use crate::macros::define_dimension_arithmetic_operation;
use crate::parameters::Parameter;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`DimensionMinOperation`].
pub const DIMENSION_MIN_OPERATION_NAME: &str = "dimension_min";

define_dimension_arithmetic_operation!(
    /// Dimension-minimum operation used by [`DimensionMin`].
    ///
    /// Refer to [`DimensionMin`] for semantic details and an example.
    DimensionMinOperation, DIMENSION_MIN_OPERATION_NAME,
    DimensionMin, dimension_min,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("min({}, {})", left.variable(), right.variable())
    },
    // Derives sound bounds for total dimension minimum.
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (right_lower, right_maximum) = right.bounds().representable_extent_range()?;
        let bounds = DimensionBounds::new(left_lower.min(right_lower), left_maximum.min(right_maximum).checked_add(1))?;
        Ok((bounds, false))
    },
    capability = {
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
        trait;
        /// Returns `min(self, right)`.
        fn(right);
    },
);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionValue};

    use super::*;

    #[test]
    fn test_dimension_min_operation() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionMinOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_MIN_OPERATION_NAME);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(1, Some(5)).unwrap());
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
