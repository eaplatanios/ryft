use crate::arrays::{DimensionBounds, DimensionError, DimensionType, DimensionValue};
use crate::macros::define_dimension_arithmetic_operation;
use crate::parameters::Parameter;
use crate::programs::{Operation, ProgramError, Typed};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`DimensionSaturatingSubOperation`].
pub const DIMENSION_SATURATING_SUB_OPERATION_NAME: &str = "dimension_saturating_sub";

define_dimension_arithmetic_operation!(
    /// Saturating dimension-subtraction operation used by [`DimensionSaturatingSub`].
    ///
    /// Refer to [`DimensionSaturatingSub`] for semantic details and an example.
    DimensionSaturatingSubOperation, DIMENSION_SATURATING_SUB_OPERATION_NAME,
    DimensionSaturatingSub, dimension_saturating_sub,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("max(0, {} - {})", left.variable(), right.variable())
    },
    // Derives sound bounds for total, saturating dimension subtraction.
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (right_lower, right_maximum) = right.bounds().representable_extent_range()?;
        let bounds = DimensionBounds::new(
            left_lower.saturating_sub(right_maximum),
            left_maximum.saturating_sub(right_lower).checked_add(1),
        )?;
        Ok((bounds, false))
    },
    capability = {
        /// Subtracts one runtime dimension from another, saturating at zero instead of producing a negative result.
        ///
        /// # Example
        ///
        /// ```rust
        /// # use ryft_core::{DimensionSaturatingSub, DimensionValue, ProgramError};
        /// # fn main() -> Result<(), ProgramError> {
        /// let result = DimensionValue::constant(3)?
        ///     .dimension_saturating_sub(&DimensionValue::constant(7)?)?;
        /// assert_eq!(result.extent(), 0);
        /// # Ok(())
        /// # }
        /// ```
        trait;
        /// Returns `max(0, self - right)`.
        fn(right);
    },
);

impl DimensionSaturatingSub for DimensionValue {
    fn dimension_saturating_sub(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionSaturatingSubOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
        let inputs = &[self.r#type().into_owned(), right.r#type().into_owned()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        Ok(Self::new(result_type, self.extent().saturating_sub(right.extent()))?)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionValue};

    use super::*;

    #[test]
    fn test_dimension_saturating_sub() {
        let left = DimensionType::new("left", DimensionBounds::new(1, Some(5)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(2, Some(9)).unwrap());
        let operation = DimensionSaturatingSubOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_SATURATING_SUB_OPERATION_NAME);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(0, Some(3)).unwrap());
        assert_eq!(
            DimensionValue::constant(3)
                .unwrap()
                .dimension_saturating_sub(&DimensionValue::constant(7).unwrap())
                .unwrap()
                .extent(),
            0,
        );
    }
}
