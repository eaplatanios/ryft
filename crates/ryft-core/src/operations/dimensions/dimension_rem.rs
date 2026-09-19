use crate::arrays::{DimensionBounds, DimensionError, DimensionType, DimensionValue};
use crate::macros::define_dimension_arithmetic_operation;
use crate::operations::math::rem::{Rem, RemOperation};
use crate::parameters::Parameter;
use crate::programs::{Operation, ProgramError, Typed};

// TODO(eaplatanios): Review this module.

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
        if right_maximum == 0 {
            return Err(DimensionError::RequirementViolation {
                message: format!("{} > 0 is impossible from declared bounds", right.variable()),
            });
        }
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

impl Rem for DimensionValue {
    fn rem(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionRemOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
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
        Ok(Self::new(result_type, self.extent() % right.extent())?)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionValue};
    use crate::programs::{EffectClass, EffectClasses};

    use super::*;

    #[test]
    fn test_dimension_rem() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionRemOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_REM_OPERATION_NAME);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(0, Some(4)).unwrap());
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        let maybe_zero = DimensionType::new("maybe_zero", DimensionBounds::new(0, Some(5)).unwrap());
        assert_eq!(
            DimensionRemOperation::new(&left, &maybe_zero).unwrap().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion),
        );

        // Specializing the input bounds recomputes this operation's result bounds.
        let declared_left = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionRemOperation::new(&declared_left, &declared_right).unwrap();
        let exact_left = DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap());
        let exact_right = DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap());
        let result = operation.infer_output_types(&[exact_left, exact_right], &[]).unwrap();
        assert_eq!(result[0].extent(), Some(0));

        assert_eq!(
            DimensionValue::constant(7).unwrap().rem(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            1,
        );
    }
}
