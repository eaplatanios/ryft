use crate::arrays::{
    ArrayIrOperation, ArrayType, DimensionBounds, DimensionError, DimensionType, DimensionValue, MAX_DIMENSION_EXTENT,
};
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
    fold = |left: &DimensionType, right: &DimensionType| {
        if left.variable() == right.variable()
            || left.bounds().lower() >= right.maximum_extent().unwrap_or(MAX_DIMENSION_EXTENT)
        {
            Some(vec![0])
        } else if right.bounds().lower() >= left.maximum_extent().unwrap_or(MAX_DIMENSION_EXTENT) {
            Some(vec![1])
        } else {
            None
        }
    },
    capability = {
        /// Returns the larger of two first-class dimensions.
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
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionOperation, DimensionValue};
    use crate::contexts::StagingContext;
    use crate::programs::EffectClasses;
    use crate::tracing::TracingContext;

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

    #[test]
    fn test_dimension_max_identity() {
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(value, _zero, one)| {
                Ok(vec![value.dimension_max(&value)?, value.dimension_max(&one)?, one.dimension_max(&value)?])
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
                in (%0, %0, %0)"},
        );
    }

    #[test]
    fn test_dimension_max_identity_bounds() {
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(left, right)| left.dimension_max(&right),
            (
                DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap()),
                DimensionType::new("right", DimensionBounds::new(1, Some(9)).unwrap()),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<left ∈ [1, 9)>, %1:dimension<right ∈ [1, 9)> .
                let %2:dimension<max(left, right) ∈ [1, 9)> = dimension_max %0 %1
                in (%2)"},
        );

        // Inclusive ordering also proves the boundary case with unbounded positive dimensions.
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(positive, one)| Ok((positive.dimension_max(&one)?, one.dimension_max(&positive)?)),
            (
                DimensionType::new("positive", DimensionBounds::new(1, None).unwrap()),
                DimensionValue::constant(1).unwrap().r#type().into_owned(),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<positive ∈ [1, ∞)>, %1:dimension<1> .
                in (%0, %0)"},
        );

        // Even identical inputs must validate their portable extent range before being reused.
        if let Some(invalid_extent) = MAX_DIMENSION_EXTENT.checked_add(1) {
            let context = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::new();
            let invalid =
                context.input(DimensionType::new("invalid", DimensionBounds::new(invalid_extent, None).unwrap()));
            let error = invalid.dimension_max(&invalid).unwrap_err();
            assert_eq!(
                error.downcast_custom::<DimensionError>(),
                Some(&DimensionError::ExtentExceedsBackendWidth {
                    value: invalid_extent,
                    maximum: MAX_DIMENSION_EXTENT,
                }),
            );
        }
    }
}
