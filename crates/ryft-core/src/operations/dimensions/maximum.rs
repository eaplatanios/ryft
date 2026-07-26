use crate::macros::define_arithmetic_dimension_operation;
use crate::parameters::Parameter;
use crate::types::{DimensionError, DimensionType};

use super::{bounds_from_extrema, representable_extent_range};

/// Canonical operation name for [`DimensionMaximumOperation`].
pub const DIMENSION_MAXIMUM_OPERATION_NAME: &str = "dimension_maximum";

define_arithmetic_dimension_operation!(
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
    capability DimensionMaximum {
        /// Returns `max(self, right)`.
        fn maximum_dimension;
    }
    /// Dimension-maximum operation used by [`DimensionMaximum`].
    ///
    /// Refer to [`DimensionMaximum`] for semantic details and an example.
    operation DimensionMaximumOperation {
        name = DIMENSION_MAXIMUM_OPERATION_NAME,
        result_name = |left: &DimensionType, right: &DimensionType| {
            format!("max({}, {})", left.variable(), right.variable())
        },
        infer_bounds = infer_bounds,
        evaluate = evaluate,
    }
);

/// Derives sound bounds for dimension maximum.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<crate::DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    bounds_from_extrema(left_lower.max(right_lower), left_maximum.max(right_maximum))
}

/// Evaluates dimension maximum.
fn evaluate(
    _left_type: &DimensionType,
    left: usize,
    _right_type: &DimensionType,
    right: usize,
) -> Result<usize, DimensionError> {
    Ok(left.max(right))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::DimensionValue;
    use crate::operations::dimensions::ArithmeticDimensionOperation;
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
        assert_eq!(operation.evaluate_extents(7, 3), Ok(7));
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
