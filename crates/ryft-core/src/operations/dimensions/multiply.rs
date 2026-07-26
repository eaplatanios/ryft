use crate::macros::define_arithmetic_dimension_operation;
use crate::parameters::Parameter;
use crate::types::{DimensionError, DimensionType, MAX_DIMENSION_EXTENT};

use super::{bounds_from_extrema, bounds_overflow, evaluation_overflow, representable_extent_range};

/// Canonical operation name for [`DimensionMultiplyOperation`].
pub const DIMENSION_MULTIPLY_OPERATION_NAME: &str = "dimension_multiply";

define_arithmetic_dimension_operation!(
    /// Multiplies two runtime dimensions using checked nonnegative integer arithmetic.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionMultiply, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(3)?.multiply_dimension(&DimensionValue::constant(4)?)?;
    /// assert_eq!(result.extent(), 12);
    /// # Ok(())
    /// # }
    /// ```
    capability DimensionMultiply {
        /// Returns the checked product of `self` and `right`.
        fn multiply_dimension;
    }
    /// Checked dimension-multiplication operation used by [`DimensionMultiply`].
    ///
    /// Refer to [`DimensionMultiply`] for semantic details and an example.
    operation DimensionMultiplyOperation {
        name = DIMENSION_MULTIPLY_OPERATION_NAME,
        result_name = |left: &DimensionType, right: &DimensionType| {
            format!("{} * {}", left.variable(), right.variable())
        },
        infer_bounds = infer_bounds,
        evaluate = evaluate,
    }
);

/// Derives sound bounds for checked dimension multiplication.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<crate::DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    let overflow = || bounds_overflow(DIMENSION_MULTIPLY_OPERATION_NAME, left, right);
    let lower = left_lower.checked_mul(right_lower).ok_or_else(overflow)?;
    let maximum = left_maximum.saturating_mul(right_maximum).min(MAX_DIMENSION_EXTENT);
    bounds_from_extrema(lower, maximum)
}

/// Evaluates checked dimension multiplication.
fn evaluate(
    left_type: &DimensionType,
    left: usize,
    right_type: &DimensionType,
    right: usize,
) -> Result<usize, DimensionError> {
    left.checked_mul(right)
        .ok_or_else(|| evaluation_overflow("multiplying runtime dimensions", left_type, left, right_type, right))
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
    fn test_dimension_multiply_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionMultiplyOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_MULTIPLY_OPERATION_NAME);
        assert_eq!(operation.result_type().bounds(), DimensionBounds::new(2, Some(33)).unwrap());
        assert_eq!(operation.evaluate_extents(7, 3), Ok(21));
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .multiply_dimension(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            21,
        );
    }
}
