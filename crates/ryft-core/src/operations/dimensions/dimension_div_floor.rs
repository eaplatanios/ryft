use crate::arrays::{DimensionBounds, DimensionError, DimensionType};
use crate::macros::define_dimension_arithmetic_operation;
use crate::operations::math::div::{Div, DivOperation};
use crate::parameters::Parameter;

// TODO(eaplatanios): Review this module.

use super::positive_divisor_lower_bound;

/// Canonical operation name for [`DimensionDivFloorOperation`].
pub const DIMENSION_DIV_FLOOR_OPERATION_NAME: &str = "dimension_div_floor";

define_dimension_arithmetic_operation!(
    /// Checked dimension-floor-division operation used by [`Div`].
    DimensionDivFloorOperation, DIMENSION_DIV_FLOOR_OPERATION_NAME,
    Div, div,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} // {}", left.variable(), right.variable())
    },
    // Derives sound bounds for checked floor division and reports whether a zero runtime divisor remains possible.
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (_, right_maximum) = right.bounds().representable_extent_range()?;
        let positive_right_lower = positive_divisor_lower_bound(right, right_maximum)?;
        let bounds =
            DimensionBounds::new(left_lower / right_maximum, (left_maximum / positive_right_lower).checked_add(1))?;
        Ok((bounds, right.bounds().lower() == 0))
    },
    provider = DivOperation<DimensionType>,
);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionValue};
    use crate::operations::dimensions::test_dimension_type;
    use crate::operations::math::div::Div;

    use super::*;

    #[test]
    fn test_dimension_div_floor_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionDivFloorOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_DIV_FLOOR_OPERATION_NAME);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(0, Some(9)).unwrap());
        assert_eq!(
            DimensionValue::constant(7).unwrap().div(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            2,
        );
    }
}
