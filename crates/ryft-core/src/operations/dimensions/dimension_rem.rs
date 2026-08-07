use crate::arrays::{DimensionBounds, DimensionError, DimensionType};
use crate::macros::{check_count, define_arithmetic_dimension_operation};
use crate::operations::math::rem::{Rem, RemOperation};
use crate::parameters::Parameter;
use crate::programs::{OperationProvider, ProgramError};

use super::{positive_divisor_lower_bound, representable_extent_range};

/// Canonical operation name for [`DimensionRemOperation`].
pub const DIMENSION_REM_OPERATION_NAME: &str = "dimension_rem";

define_arithmetic_dimension_operation!(
    /// Checked dimension-remainder operation used by [`Rem`].
    DimensionRemOperation, DIMENSION_REM_OPERATION_NAME,
    Rem, rem,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} % {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

impl OperationProvider<DimensionType> for RemOperation<DimensionType> {
    type Operation = DimensionRemOperation;

    fn provide(input_types: &[&DimensionType]) -> Result<Self::Operation, ProgramError> {
        check_count!("input", input_types, 2, ProgramError);
        Ok(DimensionRemOperation::new(input_types[0], input_types[1])?)
    }
}

/// Derives sound bounds for checked remainder and reports whether a zero runtime divisor remains possible.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<(DimensionBounds, bool), DimensionError> {
    let (_, left_maximum) = representable_extent_range(left.bounds())?;
    let (_, right_maximum) = representable_extent_range(right.bounds())?;
    positive_divisor_lower_bound(right, right_maximum)?;
    let bounds = DimensionBounds::new(0, left_maximum.min(right_maximum - 1).checked_add(1))?;
    Ok((bounds, right.bounds().lower() == 0))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionValue};
    use crate::operations::dimensions::test_dimension_type;

    use super::*;

    #[test]
    fn test_dimension_rem_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionRemOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_REM_OPERATION_NAME);
        assert_eq!(operation.result_bounds(), DimensionBounds::new(0, Some(4)).unwrap());
        assert_eq!(
            DimensionValue::constant(7).unwrap().rem(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            1,
        );
    }
}
