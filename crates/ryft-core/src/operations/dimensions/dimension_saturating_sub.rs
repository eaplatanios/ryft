use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::parameters::Parameter;
use crate::types::{DimensionBounds, DimensionError, DimensionType};

use super::representable_extent_range;

/// Canonical operation name for [`DimensionSaturatingSubOperation`].
pub const DIMENSION_SATURATING_SUB_OPERATION_NAME: &str = "dimension_saturating_sub";

define_arithmetic_dimension_operation!(
    /// Saturating dimension-subtraction operation used by [`DimensionSaturatingSub`].
    ///
    /// Refer to [`DimensionSaturatingSub`] for semantic details and an example.
    DimensionSaturatingSubOperation, DIMENSION_SATURATING_SUB_OPERATION_NAME,
    DimensionSaturatingSub, dimension_saturating_sub,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("max(0, {} - {})", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

define_arithmetic_dimension_capability!(
    /// Subtracts one runtime dimension from another, saturating at zero instead of producing a negative result.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionSaturatingSub, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(3)?
    ///     .dimension_saturating_sub(&DimensionValue::constant(7)?)?;
    /// assert_eq!(result.extent(), 0);
    /// # Ok(())
    /// # }
    /// ```
    DimensionSaturatingSub,
    /// Returns `max(0, self - right)`.
    dimension_saturating_sub(right),
    DimensionSaturatingSubOperation,
);

/// Derives sound bounds for total, saturating dimension subtraction.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<(DimensionBounds, bool), DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    let bounds = DimensionBounds::new(
        left_lower.saturating_sub(right_maximum),
        left_maximum.saturating_sub(right_lower).checked_add(1),
    )?;
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
    fn test_dimension_saturating_sub_operation() {
        let left = test_dimension_type("left", 1, 5);
        let right = test_dimension_type("right", 2, 9);
        let operation = DimensionSaturatingSubOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_SATURATING_SUB_OPERATION_NAME);
        assert_eq!(operation.result_bounds(), DimensionBounds::new(0, Some(3)).unwrap());
        assert_eq!(
            DimensionValue::constant(3)
                .unwrap()
                .dimension_saturating_sub(&DimensionValue::constant(7).unwrap())
                .unwrap()
                .extent(),
            0,
        );
    }
}
