use crate::arrays::{DimensionBounds, DimensionError, DimensionType, DimensionValue};
use crate::macros::define_dimension_arithmetic_operation;
use crate::operations::math::sub::{Sub, SubOperation};
use crate::parameters::Parameter;
use crate::programs::{Operation, ProgramError, Typed};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`DimensionSubOperation`].
pub const DIMENSION_SUB_OPERATION_NAME: &str = "dimension_sub";

define_dimension_arithmetic_operation!(
    /// Checked dimension-subtraction operation used by [`Sub`].
    DimensionSubOperation, DIMENSION_SUB_OPERATION_NAME,
    Sub, sub,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} - {}", left.variable(), right.variable())
    },
    // Derives sound bounds for checked dimension subtraction and reports whether runtime underflow remains possible.
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (right_lower, right_maximum) = right.bounds().representable_extent_range()?;
        if left_maximum < right_lower {
            return Err(DimensionError::RequirementViolation {
                message: format!("{} >= {} is impossible from declared bounds", left.variable(), right.variable()),
            });
        }
        let bounds =
            DimensionBounds::new(left_lower.saturating_sub(right_maximum), (left_maximum - right_lower).checked_add(1))?;
        let requires_runtime_assertion = right.maximum_extent().is_none_or(|right| left.bounds().lower() < right);
        Ok((bounds, requires_runtime_assertion))
    },
    provider = SubOperation<DimensionType>,
);

impl Sub for DimensionValue {
    fn sub(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionSubOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
        let inputs = &[self.r#type().into_owned(), right.r#type().into_owned()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        let extent = self.extent().checked_sub(right.extent()).ok_or_else(|| {
            let left_variable = self.r#type().variable().to_string();
            let right_variable = right.r#type().variable().to_string();
            DimensionError::RequirementViolation {
                message: format!(
                    "{left_variable} >= {right_variable}; observed {left_variable}={}, {right_variable}={}",
                    self.extent(),
                    right.extent(),
                ),
            }
        })?;
        Ok(Self::new(result_type, extent)?)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionValue};

    use super::*;

    #[test]
    fn test_dimension_sub() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionSubOperation::new(&left, &right).unwrap();
        // These operand bounds admit an underflow, so the rendering carries the runtime-assertion classification that
        // makes this instruction effectful.
        assert_eq!(operation.to_string(), format!("{DIMENSION_SUB_OPERATION_NAME} [requires_runtime_assertion=true]"));
        assert_eq!(operation.output_bounds(), DimensionBounds::new(0, Some(8)).unwrap());
        assert_eq!(
            DimensionValue::constant(7).unwrap().sub(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            4,
        );
    }
}
