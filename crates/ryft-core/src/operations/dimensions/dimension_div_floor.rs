use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::operations::math::DivOperationFor;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::types::{DimensionBounds, DimensionError, DimensionType};

use super::{positive_divisor_lower_bound, representable_extent_range};

/// Canonical operation name for [`DimensionDivFloorOperation`].
pub const DIMENSION_DIV_FLOOR_OPERATION_NAME: &str = "dimension_div_floor";

define_arithmetic_dimension_operation!(
    /// Checked dimension-floor-division operation used by [`DimensionDivFloor`].
    ///
    /// Refer to [`DimensionDivFloor`] for semantic details and an example.
    DimensionDivFloorOperation, DIMENSION_DIV_FLOOR_OPERATION_NAME,
    DimensionDivFloor, dimension_div_floor,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} // {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

impl DivOperationFor for DimensionType {
    type Operation = DimensionDivFloorOperation;

    #[inline]
    fn operation(left_type: &Self, right_type: &Self) -> Result<Self::Operation, ProgramError> {
        Ok(DimensionDivFloorOperation::new(left_type, right_type)?)
    }
}

define_arithmetic_dimension_capability!(
    /// Floor-divides one runtime dimension by a positive runtime dimension.
    /// [`DimensionDivFloor::dimension_div_floor`] is the fallible counterpart to [`std::ops::Div`]; because dimensions
    /// are nonnegative integers, [`DimensionValue`](crate::DimensionValue)'s `/` operator has these same floor-division
    /// semantics.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionDivFloor, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(7)?.dimension_div_floor(&DimensionValue::constant(3)?)?;
    /// assert_eq!(result.extent(), 2);
    /// # Ok(())
    /// # }
    /// ```
    DimensionDivFloor,
    /// Returns `self // right`, failing when `right` is zero.
    dimension_div_floor(right),
    DimensionDivFloorOperation,
);

/// Derives sound bounds for checked dimension floor division.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (_, right_maximum) = representable_extent_range(right.bounds())?;
    let positive_right_lower = positive_divisor_lower_bound(right, right_maximum)?;
    DimensionBounds::new(left_lower / right_maximum, (left_maximum / positive_right_lower).checked_add(1))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::DimensionValue;
    use crate::types::DimensionBounds;

    use super::super::test_dimension_type;
    use super::*;

    #[test]
    fn test_dimension_div_floor_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionDivFloorOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_DIV_FLOOR_OPERATION_NAME);
        assert_eq!(operation.result_bounds(), DimensionBounds::new(0, Some(9)).unwrap());
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .dimension_div_floor(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            2,
        );
    }
}
