use crate::macros::{define_arithmetic_dimension_capability, define_arithmetic_dimension_operation};
use crate::operations::math::MulOperationFor;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::types::{DimensionBounds, DimensionError, DimensionType, MAX_DIMENSION_EXTENT};

use super::{bounds_overflow, representable_extent_range};

/// Canonical operation name for [`DimensionMulOperation`].
pub const DIMENSION_MUL_OPERATION_NAME: &str = "dimension_mul";

define_arithmetic_dimension_operation!(
    /// Checked dimension-multiplication operation used by [`DimensionMul`].
    ///
    /// Refer to [`DimensionMul`] for semantic details and an example.
    DimensionMulOperation, DIMENSION_MUL_OPERATION_NAME,
    DimensionMul, mul,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} * {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

impl MulOperationFor for DimensionType {
    type Operation = DimensionMulOperation;

    #[inline]
    fn operation(left_type: &Self, right_type: &Self) -> Result<Self::Operation, ProgramError> {
        Ok(DimensionMulOperation::new(left_type, right_type)?)
    }
}

define_arithmetic_dimension_capability!(
    /// Multiplies two runtime dimensions using checked nonnegative integer arithmetic. [`DimensionMul::mul`] is
    /// the fallible counterpart to [`std::ops::Mul`]; [`DimensionValue`](crate::DimensionValue) supports `*` as
    /// panicking convenience syntax.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionMul, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(3)?.mul(&DimensionValue::constant(4)?)?;
    /// assert_eq!(result.extent(), 12);
    /// # Ok(())
    /// # }
    /// ```
    DimensionMul,
    /// Returns the checked product of `self` and `right`.
    mul(right),
    DimensionMulOperation,
);

/// Derives sound bounds for checked dimension multiplication.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    let overflow = || bounds_overflow(DIMENSION_MUL_OPERATION_NAME, left, right);
    let lower = left_lower.checked_mul(right_lower).ok_or_else(overflow)?;
    let maximum = left_maximum.saturating_mul(right_maximum).min(MAX_DIMENSION_EXTENT);
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
    fn test_dimension_mul_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionMulOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_MUL_OPERATION_NAME);
        assert_eq!(operation.result_bounds(), DimensionBounds::new(2, Some(33)).unwrap());
        assert_eq!(
            DimensionValue::constant(7).unwrap().mul(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            21,
        );
    }
}
