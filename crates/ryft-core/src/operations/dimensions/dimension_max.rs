use crate::arrays::{ArrayIrOperation, ArrayType, DimensionBounds, DimensionError, DimensionType, DimensionValue};
use crate::macros::define_dimension_arithmetic_operation;
use crate::parameters::Parameter;
use crate::programs::{Operation, ProgramError, Typed, Value};

/// Canonical operation name for [`DimensionMaxOperation`].
pub const DIMENSION_MAX_OPERATION_NAME: &str = "dimension_max";

define_dimension_arithmetic_operation!(
    /// Dimension-maximum operation used by [`DimensionMax`].
    ///
    /// Refer to [`DimensionMax`] for semantic details and an example.
    DimensionMaxOperation,
    DIMENSION_MAX_OPERATION_NAME,
    DimensionMax,
    dimension_max,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("max({}, {})", left.variable(), right.variable())
    },
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (right_lower, right_maximum) = right.bounds().representable_extent_range()?;
        let bounds = DimensionBounds::new(left_lower.max(right_lower), left_maximum.max(right_maximum).checked_add(1))?;
        Ok((bounds, false))
    },
    capability = {
        /// Returns the larger of two [`DimensionValue`]s.
        ///
        /// # Example
        ///
        /// ```rust
        /// # use ryft_core::{DimensionMax, DimensionValue, ProgramError};
        /// # fn main() -> Result<(), ProgramError> {
        /// let output = DimensionValue::constant(7)?.dimension_max(&DimensionValue::constant(3)?)?;
        /// assert_eq!(output.extent(), 7);
        /// # Ok(())
        /// # }
        /// ```
        trait;
        /// Returns the larger value between `self` and `other`.
        fn(other);
    },
);

impl<A: Value<Type = ArrayType>> From<DimensionMaxOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: DimensionMaxOperation) -> Self {
        Self::Dimension(operation.into())
    }
}

impl DimensionMax for DimensionValue {
    fn dimension_max(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionMaxOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
        let inputs = &[self.r#type().into_owned(), right.r#type().into_owned()];
        let output_type = operation.infer_output_types(inputs, &[])?.remove(0);
        Ok(Self::new(output_type, self.extent().max(right.extent()))?)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionValue};
    use crate::programs::EffectClasses;

    use super::*;

    #[test]
    fn test_dimension_max() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionMaxOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_MAX_OPERATION_NAME);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(2, Some(9)).unwrap());
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);

        // Specializing the input bounds recomputes this operation's output bounds.
        let declared_left = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionMaxOperation::new(&declared_left, &declared_right).unwrap();
        let exact_left = DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap());
        let exact_right = DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap());
        let output = operation.infer_output_types(&[exact_left, exact_right], &[]).unwrap();
        assert_eq!(output[0].extent(), Some(6));

        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .dimension_max(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            7,
        );
    }
}
