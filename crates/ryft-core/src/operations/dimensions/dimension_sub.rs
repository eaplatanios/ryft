use crate::arrays::{ArrayIrOperation, ArrayType, DimensionBounds, DimensionError, DimensionType, DimensionValue};
use crate::macros::define_dimension_arithmetic_operation;
use crate::operations::{Sub, SubOperation};
use crate::parameters::Parameter;
use crate::programs::{Operation, ProgramError, Typed, Value};

/// Canonical operation name for [`DimensionSubOperation`].
pub const DIMENSION_SUB_OPERATION_NAME: &str = "dimension_sub";

define_dimension_arithmetic_operation!(
    /// Checked dimension-subtraction operation used by [`Sub`] for [`DimensionValue`]s.
    DimensionSubOperation,
    DIMENSION_SUB_OPERATION_NAME,
    Sub,
    sub,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} - {}", left.variable(), right.variable())
    },
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (right_lower, right_maximum) = right.bounds().representable_extent_range()?;
        if left_maximum < right_lower {
            return Err(DimensionError::RequirementViolation {
                message: format!("{} >= {} is impossible from declared bounds", left.variable(), right.variable()),
            });
        }
        let bounds = DimensionBounds::new(
            left_lower.saturating_sub(right_maximum),
            (left_maximum - right_lower).checked_add(1),
        )?;
        let requires_runtime_assertion = right.maximum_extent().is_none_or(|right| left.bounds().lower() < right);
        Ok((bounds, requires_runtime_assertion))
    },
    provider = SubOperation<DimensionType>,
);

impl<A: Value<Type = ArrayType>> From<DimensionSubOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: DimensionSubOperation) -> Self {
        Self::Dimension(operation.into())
    }
}

impl Sub for DimensionValue {
    fn sub(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionSubOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
        let inputs = &[self.r#type().into_owned(), right.r#type().into_owned()];
        let output_type = operation.infer_output_types(inputs, &[])?.remove(0);
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
        Ok(Self::new(output_type, extent)?)
    }
}

impl std::ops::Sub for DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn sub(self, right: DimensionValue) -> Self::Output {
        Sub::sub(&self, &right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Sub<&DimensionValue> for DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn sub(self, right: &DimensionValue) -> Self::Output {
        Sub::sub(&self, right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Sub<DimensionValue> for &DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn sub(self, right: DimensionValue) -> Self::Output {
        Sub::sub(self, &right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Sub<&DimensionValue> for &DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn sub(self, right: &DimensionValue) -> Self::Output {
        Sub::sub(self, right).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionValue};
    use crate::programs::{EffectClass, EffectClasses};

    use super::*;

    #[test]
    fn test_dimension_sub() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionSubOperation::new(&left, &right).unwrap();

        // These input bounds admit an underflow, so the rendering carries the runtime-assertion classification
        // that makes this instruction effectful.
        assert_eq!(operation.to_string(), format!("{DIMENSION_SUB_OPERATION_NAME} [requires_runtime_assertion=true]"));
        assert_eq!(operation.output_bounds(), DimensionBounds::new(0, Some(8)).unwrap());
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let safe_minuend = DimensionType::new("safe_minuend", DimensionBounds::new(5, Some(9)).unwrap());
        let safe_subtrahend = DimensionType::new("safe_subtrahend", DimensionBounds::new(1, Some(4)).unwrap());
        assert_eq!(
            DimensionSubOperation::new(&safe_minuend, &safe_subtrahend).unwrap().effects().classes(),
            EffectClasses::NONE,
        );

        // Specializing the input bounds recomputes this operation's output bounds.
        let declared_left = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionSubOperation::new(&declared_left, &declared_right).unwrap();
        let exact_left = DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap());
        let exact_right = DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap());
        let output = operation.infer_output_types(&[exact_left, exact_right], &[]).unwrap();
        assert_eq!(output[0].extent(), Some(4));

        // Refinement retains the original conservative assertion effect.
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));

        assert_eq!(
            DimensionValue::constant(7).unwrap().sub(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            4,
        );

        let left = DimensionValue::constant(7).unwrap();
        let right = DimensionValue::constant(3).unwrap();
        assert_eq!((left.clone() - right.clone()).extent(), 4);
        assert_eq!((left.clone() - &right).extent(), 4);
        assert_eq!((&left - right.clone()).extent(), 4);
        assert_eq!((&left - &right).extent(), 4);
    }

    #[test]
    #[should_panic(expected = "left >= right; observed left=1, right=3")]
    fn test_dimension_sub_operator_panics_on_capability_error() {
        let left_type = DimensionType::new("left", DimensionBounds::new(0, Some(10)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(0, Some(10)).unwrap());
        let left = DimensionValue::new(left_type, 1).unwrap();
        let right = DimensionValue::new(right_type, 3).unwrap();
        let _ = left - right;
    }
}
