use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::operations::math::RemOperationFor;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::types::{DimensionBounds, DimensionError, DimensionType};

use super::{positive_divisor_lower_bound, representable_extent_range};

/// Canonical operation name for [`DimensionRemOperation`].
pub const DIMENSION_REM_OPERATION_NAME: &str = "dimension_rem";

define_arithmetic_dimension_operation!(
    /// Checked dimension-remainder operation used by [`DimensionRem`].
    ///
    /// Refer to [`DimensionRem`] for semantic details and an example.
    DimensionRemOperation, DIMENSION_REM_OPERATION_NAME,
    DimensionRem, rem,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} % {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

impl RemOperationFor for DimensionType {
    type Operation = DimensionRemOperation;

    #[inline]
    fn operation(left_type: &Self, right_type: &Self) -> Result<Self::Operation, ProgramError> {
        Ok(DimensionRemOperation::new(left_type, right_type)?)
    }
}

define_arithmetic_dimension_capability!(
    /// Computes the remainder of one runtime dimension divided by a positive runtime dimension.
    /// [`DimensionRem::rem`] is the fallible counterpart to [`std::ops::Rem`];
    /// [`DimensionValue`](crate::DimensionValue) supports `%` as panicking convenience syntax.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionRem, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(7)?.rem(&DimensionValue::constant(3)?)?;
    /// assert_eq!(result.extent(), 1);
    /// # Ok(())
    /// # }
    /// ```
    DimensionRem,
    /// Returns `self % right`, failing when `right` is zero.
    rem(right),
    DimensionRemOperation,
);

/// Derives sound bounds for checked dimension remainder.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<DimensionBounds, DimensionError> {
    let (_, left_maximum) = representable_extent_range(left.bounds())?;
    let (_, right_maximum) = representable_extent_range(right.bounds())?;
    positive_divisor_lower_bound(right, right_maximum)?;
    DimensionBounds::new(0, left_maximum.min(right_maximum - 1).checked_add(1))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::DimensionValue;
    use crate::types::DimensionBounds;

    use super::super::test_dimension_type;
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
