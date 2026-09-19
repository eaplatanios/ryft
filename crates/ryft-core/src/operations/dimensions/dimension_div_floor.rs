use crate::arrays::{DimensionBounds, DimensionError, DimensionType, DimensionValue};
use crate::macros::define_dimension_arithmetic_operation;
use crate::operations::math::div::{Div, DivOperation};
use crate::parameters::Parameter;
use crate::programs::{Operation, ProgramError, Typed};

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

impl Div for DimensionValue {
    fn div(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionDivFloorOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
        let inputs = &[self.r#type().into_owned(), right.r#type().into_owned()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        if right.extent() == 0 {
            let left_variable = self.r#type().variable().to_string();
            let right_variable = right.r#type().variable().to_string();
            return Err(DimensionError::RequirementViolation {
                message: format!(
                    "{right_variable} > 0; observed {left_variable}={}, {right_variable}={}",
                    self.extent(),
                    right.extent(),
                ),
            }
            .into());
        }
        Ok(Self::new(result_type, self.extent() / right.extent())?)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionValue};
    use crate::operations::math::div::Div;

    use super::*;

    #[test]
    fn test_dimension_div_floor() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionDivFloorOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_DIV_FLOOR_OPERATION_NAME);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(0, Some(9)).unwrap());
        assert_eq!(
            DimensionValue::constant(7).unwrap().div(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            2,
        );
    }
}
