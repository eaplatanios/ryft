use crate::arrays::{
    ArrayIrOperation, ArrayType, DimensionBounds, DimensionError, DimensionType, DimensionValue, MAX_DIMENSION_EXTENT,
};
use crate::macros::define_dimension_arithmetic_operation;
use crate::parameters::Parameter;
use crate::programs::{Operation, ProgramError, Typed, Value};

/// Canonical operation name for [`DimensionMinOperation`].
pub const DIMENSION_MIN_OPERATION_NAME: &str = "dimension_min";

define_dimension_arithmetic_operation!(
    /// Dimension-minimum operation used by [`DimensionMin`].
    ///
    /// Refer to [`DimensionMin`] for semantic details and an example.
    DimensionMinOperation, DIMENSION_MIN_OPERATION_NAME,
    DimensionMin, dimension_min,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("min({}, {})", left.variable(), right.variable())
    },
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (right_lower, right_maximum) = right.bounds().representable_extent_range()?;
        let bounds = DimensionBounds::new(left_lower.min(right_lower), left_maximum.min(right_maximum).checked_add(1))?;
        Ok((bounds, false))
    },
    fold = |left: &DimensionType, right: &DimensionType| {
        if left.variable() == right.variable()
            || left.maximum_extent().unwrap_or(MAX_DIMENSION_EXTENT) <= right.bounds().lower()
        {
            Some(vec![0])
        } else if right.maximum_extent().unwrap_or(MAX_DIMENSION_EXTENT) <= left.bounds().lower() {
            Some(vec![1])
        } else {
            None
        }
    },
    capability = {
        /// Returns the smaller of two first-class dimensions.
        ///
        /// # Example
        ///
        /// ```rust
        /// # use ryft_core::{DimensionMin, DimensionValue, ProgramError};
        /// # fn main() -> Result<(), ProgramError> {
        /// let output = DimensionValue::constant(7)?.dimension_min(&DimensionValue::constant(3)?)?;
        /// assert_eq!(output.extent(), 3);
        /// # Ok(())
        /// # }
        /// ```
        trait;
        /// Returns the smaller value between `self` and `other`.
        fn(other);
    },
);

impl<A: Value<Type = ArrayType>> From<DimensionMinOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: DimensionMinOperation) -> Self {
        Self::Dimension(operation.into())
    }
}

impl DimensionMin for DimensionValue {
    fn dimension_min(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionMinOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
        let inputs = &[self.r#type().into_owned(), right.r#type().into_owned()];
        let output_type = operation.infer_output_types(inputs, &[])?.remove(0);
        Ok(Self::new(output_type, self.extent().min(right.extent()))?)
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
    fn test_dimension_min() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionMinOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_MIN_OPERATION_NAME);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(1, Some(5)).unwrap());
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);

        // Specializing the input bounds recomputes this operation's output bounds.
        let declared_left = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionMinOperation::new(&declared_left, &declared_right).unwrap();
        let exact_left = DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap());
        let exact_right = DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap());
        let output = operation.infer_output_types(&[exact_left, exact_right], &[]).unwrap();
        assert_eq!(output[0].extent(), Some(2));

        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .dimension_min(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            3,
        );
    }

    #[test]
    fn test_dimension_min_identity() {
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(value, _zero, one)| {
                Ok(vec![value.dimension_min(&value)?, value.dimension_min(&one)?, one.dimension_min(&value)?])
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
                in (%0, %2, %2)"},
        );
    }

    #[test]
    fn test_dimension_min_identity_bounds() {
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(left, right)| left.dimension_min(&right),
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
                let %2:dimension<min(left, right) ∈ [1, 9)> = dimension_min %0 %1
                in (%2)"},
        );

        // Inclusive ordering also proves the boundary case with unbounded positive dimensions.
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(positive, one)| Ok((positive.dimension_min(&one)?, one.dimension_min(&positive)?)),
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
                in (%1, %1)"},
        );

        // Even identical inputs must validate their portable extent range before being reused.
        if let Some(invalid_extent) = MAX_DIMENSION_EXTENT.checked_add(1) {
            let context = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::new();
            let invalid =
                context.input(DimensionType::new("invalid", DimensionBounds::new(invalid_extent, None).unwrap()));
            let error = invalid.dimension_min(&invalid).unwrap_err();
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
