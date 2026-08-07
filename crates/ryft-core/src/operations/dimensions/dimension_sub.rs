use crate::arrays::{DimensionBounds, DimensionError, DimensionType};
use crate::macros::check_count;
use crate::macros::define_arithmetic_dimension_operation;
use crate::operations::math::{Sub, SubOperation};
use crate::parameters::Parameter;
use crate::programs::{OperationProvider, ProgramError};

use super::{maximum_extent, representable_extent_range};

/// Canonical operation name for [`DimensionSubOperation`].
pub const DIMENSION_SUB_OPERATION_NAME: &str = "dimension_sub";

define_arithmetic_dimension_operation!(
    /// Checked dimension-subtraction operation used by [`Sub`].
    DimensionSubOperation, DIMENSION_SUB_OPERATION_NAME,
    Sub, sub,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} - {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

impl OperationProvider<DimensionType> for SubOperation<DimensionType> {
    type Operation = DimensionSubOperation;

    fn provide(input_types: &[&DimensionType]) -> Result<Self::Operation, ProgramError> {
        check_count!("input", input_types, 2, ProgramError);
        Ok(DimensionSubOperation::new(input_types[0], input_types[1])?)
    }
}

/// Derives sound bounds for checked dimension subtraction and reports whether runtime underflow remains possible.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<(DimensionBounds, bool), DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    if left_maximum < right_lower {
        return Err(DimensionError::RequirementViolation {
            message: format!("{} >= {} is impossible from declared bounds", left.variable(), right.variable()),
        });
    }
    let bounds =
        DimensionBounds::new(left_lower.saturating_sub(right_maximum), (left_maximum - right_lower).checked_add(1))?;
    let requires_runtime_assertion = maximum_extent(right).is_none_or(|right| left.bounds().lower() < right);
    Ok((bounds, requires_runtime_assertion))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::DimensionBounds;
    use crate::backends::dimensions::DimensionValue;

    use super::super::test_dimension_type;
    use super::*;

    #[test]
    fn test_dimension_sub_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionSubOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_SUB_OPERATION_NAME);
        assert_eq!(operation.result_bounds(), DimensionBounds::new(0, Some(8)).unwrap());
        assert_eq!(
            DimensionValue::constant(7).unwrap().sub(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            4,
        );
    }
}
