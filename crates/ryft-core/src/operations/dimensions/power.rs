use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::parameters::Parameter;
use crate::types::{DimensionBounds, DimensionError, DimensionType, MAX_DIMENSION_EXTENT};

use super::{bounds_overflow, checked_power, representable_extent_range};

/// Canonical operation name for [`DimensionPowerOperation`].
pub const DIMENSION_POWER_OPERATION_NAME: &str = "dimension_power";

define_arithmetic_dimension_operation!(
    /// Checked dimension-exponentiation operation used by [`DimensionPower`].
    ///
    /// Refer to [`DimensionPower`] for semantic details and an example.
    DimensionPowerOperation, DIMENSION_POWER_OPERATION_NAME,
    DimensionPower, raise_dimension_to_power_with,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} ^ {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

define_arithmetic_dimension_capability!(
    /// Raises one runtime dimension to another dimension's power using checked integer exponentiation.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionPower, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(3)?.raise_dimension_to_power(&DimensionValue::constant(4)?)?;
    /// assert_eq!(result.extent(), 81);
    /// # Ok(())
    /// # }
    /// ```
    DimensionPower,
    /// Returns `self` raised to the nonnegative integer power `right`.
    raise_dimension_to_power(right),
    raise_dimension_to_power_with(right, operation),
    DimensionPowerOperation,
);

/// Derives sound bounds for checked dimension exponentiation.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    let overflow = || bounds_overflow(DIMENSION_POWER_OPERATION_NAME, left, right);
    let lower = if right_maximum == 0 {
        1
    } else if left_lower == 0 {
        0
    } else if left_lower == 1 {
        1
    } else {
        checked_power(left_lower, right_lower).ok_or_else(overflow)?
    };
    let maximum = if right_maximum == 0 || left_maximum == 1 {
        1
    } else if left_maximum == 0 {
        usize::from(right_lower == 0)
    } else {
        checked_power(left_maximum, right_maximum).unwrap_or(usize::MAX).min(MAX_DIMENSION_EXTENT)
    };
    DimensionBounds::new(lower, maximum.checked_add(1))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::DimensionValue;
    use crate::types::DimensionBounds;

    use super::super::test_dimension_type;
    use super::*;

    #[test]
    fn test_dimension_power_operation() {
        let base = test_dimension_type("base", 0, 3);
        let exponent = test_dimension_type("exponent", 0, 3);
        let operation = DimensionPowerOperation::new(&base, &exponent).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_POWER_OPERATION_NAME);
        assert_eq!(operation.result_type().bounds(), DimensionBounds::new(0, Some(5)).unwrap());
        assert_eq!(
            DimensionValue::constant(3)
                .unwrap()
                .raise_dimension_to_power(&DimensionValue::constant(4).unwrap())
                .unwrap()
                .extent(),
            81,
        );
    }
}
