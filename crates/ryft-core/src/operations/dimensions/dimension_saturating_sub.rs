use crate::arrays::{ArrayIrOperation, ArrayType, DimensionBounds, DimensionError, DimensionType, DimensionValue};
use crate::macros::define_dimension_arithmetic_operation;
use crate::parameters::Parameter;
use crate::programs::{Operation, OperationFoldOutput, ProgramError, Typed, Value};

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
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (right_lower, right_maximum) = right.bounds().representable_extent_range()?;
        let bounds = DimensionBounds::new(
            left_lower.saturating_sub(right_maximum),
            left_maximum.saturating_sub(right_lower).checked_add(1),
        )?;
        Ok((bounds, false))
    },
    fold = |left: &DimensionType, right: &DimensionType| {
        if right.extent() == Some(0) || left.extent() == Some(0) {
            Some(OperationFoldOutput::Input(0))
        } else {
            None
        }
    },
    capability = {
        /// Subtracts one runtime dimension from another, saturating at zero instead of producing a negative output.
        ///
        /// # Example
        ///
        /// ```rust
        /// # use ryft_core::{DimensionSaturatingSub, DimensionValue, ProgramError};
        /// # fn main() -> Result<(), ProgramError> {
        /// let output = DimensionValue::constant(3)?.dimension_saturating_sub(&DimensionValue::constant(7)?)?;
        /// assert_eq!(output.extent(), 0);
        /// # Ok(())
        /// # }
        /// ```
        trait;
        /// Returns `max(0, self - other)`.
        fn(other);
    },
);

impl<A: Value<Type = ArrayType>> From<DimensionSaturatingSubOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: DimensionSaturatingSubOperation) -> Self {
        Self::Dimension(operation.into())
    }
}

impl DimensionSaturatingSub for DimensionValue {
    fn dimension_saturating_sub(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionSaturatingSubOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
        let inputs = &[self.r#type().into_owned(), right.r#type().into_owned()];
        let output_type = operation.infer_output_types(inputs, &[])?.remove(0);
        Ok(Self::new(output_type, self.extent().saturating_sub(right.extent()))?)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionOperation, DimensionValue};
    use crate::programs::EffectClasses;
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_dimension_saturating_sub() {
        let left = DimensionType::new("left", DimensionBounds::new(1, Some(5)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(2, Some(9)).unwrap());
        let operation = DimensionSaturatingSubOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_SATURATING_SUB_OPERATION_NAME);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(0, Some(3)).unwrap());
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);

        // Specializing the input bounds recomputes this operation's output bounds.
        let declared_left = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionSaturatingSubOperation::new(&declared_left, &declared_right).unwrap();
        let exact_left = DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap());
        let exact_right = DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap());
        let output = operation.infer_output_types(&[exact_left, exact_right], &[]).unwrap();
        assert_eq!(output[0].extent(), Some(4));

        assert_eq!(
            DimensionValue::constant(3)
                .unwrap()
                .dimension_saturating_sub(&DimensionValue::constant(7).unwrap())
                .unwrap()
                .extent(),
            0,
        );
    }

    #[test]
    fn test_dimension_saturating_sub_identity() {
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(value, zero, _one)| {
                Ok(vec![value.dimension_saturating_sub(&zero)?, zero.dimension_saturating_sub(&value)?])
            },
            (
                DimensionType::new("value", DimensionBounds::new(2, Some(9)).unwrap()),
                DimensionValue::constant(0).unwrap().r#type().into_owned(),
                DimensionValue::constant(1).unwrap().r#type().into_owned(),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<value ∈ [2, 9)>, %1:dimension<0>, %2:dimension<1> .
                in (%0, %1)"},
        );
    }
}
