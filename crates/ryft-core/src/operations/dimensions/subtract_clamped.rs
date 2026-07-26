use crate::macros::define_arithmetic_dimension_operation;
use crate::parameters::Parameter;
use crate::types::{DimensionError, DimensionType};

use super::{bounds_from_extrema, representable_extent_range};

/// Canonical operation name for [`DimensionSubtractClampedOperation`].
pub const DIMENSION_SUBTRACT_CLAMPED_OPERATION_NAME: &str = "dimension_subtract_clamped";

define_arithmetic_dimension_operation!(
    /// Subtracts one runtime dimension from another and clamps negative results to zero.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionSubtractClamped, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(3)?
    ///     .subtract_dimension_clamped(&DimensionValue::constant(7)?)?;
    /// assert_eq!(result.extent(), 0);
    /// # Ok(())
    /// # }
    /// ```
    capability DimensionSubtractClamped {
        /// Returns `max(0, self - right)`.
        fn subtract_dimension_clamped;
    }
    /// Clamped dimension-subtraction operation used by [`DimensionSubtractClamped`].
    ///
    /// Refer to [`DimensionSubtractClamped`] for semantic details and an example.
    operation DimensionSubtractClampedOperation {
        name = DIMENSION_SUBTRACT_CLAMPED_OPERATION_NAME,
        result_name = |left: &DimensionType, right: &DimensionType| {
            format!("max(0, {} - {})", left.variable(), right.variable())
        },
        infer_bounds = infer_bounds,
        evaluate = evaluate,
    }
);

/// Derives sound bounds for clamped dimension subtraction.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<crate::DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    bounds_from_extrema(left_lower.saturating_sub(right_maximum), left_maximum.saturating_sub(right_lower))
}

/// Evaluates clamped dimension subtraction.
fn evaluate(
    _left_type: &DimensionType,
    left: usize,
    _right_type: &DimensionType,
    right: usize,
) -> Result<usize, DimensionError> {
    Ok(left.saturating_sub(right))
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
    fn test_dimension_subtract_clamped_operation() {
        let left = test_dimension_type("left", 1, 5);
        let right = test_dimension_type("right", 2, 9);
        let operation = DimensionSubtractClampedOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_SUBTRACT_CLAMPED_OPERATION_NAME);
        assert_eq!(operation.result_type().bounds(), DimensionBounds::new(0, Some(3)).unwrap());
        assert_eq!(operation.evaluate_extents(3, 7), Ok(0));
        assert_eq!(operation.evaluate_extents(4, 2), Ok(2));
        assert_eq!(
            DimensionValue::constant(3)
                .unwrap()
                .subtract_dimension_clamped(&DimensionValue::constant(7).unwrap())
                .unwrap()
                .extent(),
            0,
        );
    }
}
