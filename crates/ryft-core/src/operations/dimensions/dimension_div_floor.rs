use crate::macros::check_count;
use crate::macros::define_arithmetic_dimension_operation;
use crate::operations::math::{Div, DivOperation};
use crate::parameters::Parameter;
use crate::programs::{OperationProvider, ProgramError};
use crate::types::{DimensionBounds, DimensionError, DimensionType};

use super::{positive_divisor_lower_bound, representable_extent_range};

/// Canonical operation name for [`DimensionDivFloorOperation`].
pub const DIMENSION_DIV_FLOOR_OPERATION_NAME: &str = "dimension_div_floor";

define_arithmetic_dimension_operation!(
    /// Checked dimension-floor-division operation used by [`Div`].
    DimensionDivFloorOperation, DIMENSION_DIV_FLOOR_OPERATION_NAME,
    Div, div,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} // {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

impl OperationProvider<DimensionType> for DivOperation {
    type Operation = DimensionDivFloorOperation;

    fn provide(input_types: &[&DimensionType]) -> Result<Self::Operation, ProgramError> {
        check_count!("input", input_types, 2, ProgramError);
        Ok(DimensionDivFloorOperation::new(input_types[0], input_types[1])?)
    }
}

/// Derives sound bounds for checked floor division and reports whether a zero runtime divisor remains possible.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<(DimensionBounds, bool), DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (_, right_maximum) = representable_extent_range(right.bounds())?;
    let positive_right_lower = positive_divisor_lower_bound(right, right_maximum)?;
    let bounds =
        DimensionBounds::new(left_lower / right_maximum, (left_maximum / positive_right_lower).checked_add(1))?;
    Ok((bounds, right.bounds().lower() == 0))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::DimensionValue;
    use crate::operations::math::Div;
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
            DimensionValue::constant(7).unwrap().div(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            2,
        );
    }
}
