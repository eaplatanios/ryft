use crate::arrays::{DimensionBounds, DimensionError, DimensionType};
use crate::macros::define_dimension_arithmetic_operation;
use crate::operations::math::rem::{Rem, RemOperation};
use crate::parameters::Parameter;

// TODO(eaplatanios): Review this module.

use super::positive_divisor_lower_bound;

/// Canonical operation name for [`DimensionRemOperation`].
pub const DIMENSION_REM_OPERATION_NAME: &str = "dimension_rem";

define_dimension_arithmetic_operation!(
    /// Checked dimension-remainder operation used by [`Rem`].
    DimensionRemOperation, DIMENSION_REM_OPERATION_NAME,
    Rem, rem,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} % {}", left.variable(), right.variable())
    },
    // Derives sound bounds for checked remainder and reports whether a zero runtime divisor remains possible.
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (_, left_maximum) = left.bounds().representable_extent_range()?;
        let (_, right_maximum) = right.bounds().representable_extent_range()?;
        positive_divisor_lower_bound(right, right_maximum)?;
        let bounds = if let (Some(left), Some(right)) = (left.extent(), right.extent()) {
            let remainder = left % right;
            DimensionBounds::new(remainder, remainder.checked_add(1))?
        } else {
            DimensionBounds::new(0, left_maximum.min(right_maximum - 1).checked_add(1))?
        };
        Ok((bounds, right.bounds().lower() == 0))
    },
    provider = RemOperation<DimensionType>,
);

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
        assert_eq!(operation.output_bounds(), DimensionBounds::new(0, Some(4)).unwrap());
        assert_eq!(
            DimensionValue::constant(7).unwrap().rem(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            1,
        );
    }
}
