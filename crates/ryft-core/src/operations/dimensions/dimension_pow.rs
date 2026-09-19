use crate::arrays::{
    ArrayIrOperation, ArrayType, DimensionBounds, DimensionError, DimensionType, DimensionValue, MAX_DIMENSION_EXTENT,
};
use crate::macros::define_dimension_arithmetic_operation;
use crate::parameters::Parameter;
use crate::programs::{Operation, ProgramError, Typed, Value};

/// Canonical operation name for [`DimensionPowOperation`].
pub const DIMENSION_POW_OPERATION_NAME: &str = "dimension_pow";

define_dimension_arithmetic_operation!(
    /// Checked dimension-exponentiation operation used by [`DimensionPow`].
    ///
    /// Refer to [`DimensionPow`] for semantic details and an example.
    DimensionPowOperation,
    DIMENSION_POW_OPERATION_NAME,
    DimensionPow,
    dimension_pow,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} ^ {}", left.variable(), right.variable())
    },
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (right_lower, right_maximum) = right.bounds().representable_extent_range()?;
        let lower = if right_maximum == 0 {
            1
        } else if left_lower == 0 {
            0
        } else if left_lower == 1 {
            1
        } else {
            checked_power(left_lower, right_lower).ok_or_else(|| DimensionError::ArithmeticOverflow {
                message: format!(
                    "dimension arithmetic overflow while deriving `{DIMENSION_POW_OPERATION_NAME}` output bounds \
                     with operands `{left}` and `{right}`",
                ),
            })?
        };
        let maximum = if right_maximum == 0 || left_maximum == 1 {
            1
        } else if left_maximum == 0 {
            usize::from(right_lower == 0)
        } else {
            checked_power(left_maximum, right_maximum).unwrap_or(usize::MAX).min(MAX_DIMENSION_EXTENT)
        };
        let bounds = DimensionBounds::new(lower, maximum.checked_add(1))?;
        let requires_runtime_assertion = left.maximum_extent()
            .zip(right.maximum_extent())
            .and_then(|(left, right)| checked_power(left, right))
            .is_none_or(|output| output > MAX_DIMENSION_EXTENT);
        Ok((bounds, requires_runtime_assertion))
    },
    fold = |left: &DimensionType, right: &DimensionType| {
        if right.extent() == Some(1)
            || left.extent() == Some(1)
            || (left.extent() == Some(0) && right.bounds().lower() > 0)
        {
            Some(vec![0])
        } else {
            None
        }
    },
    capability = {
        /// Raises one [`DimensionValue`] to another [`DimensionValue`]'s power using checked integer exponentiation.
        ///
        /// # Example
        ///
        /// ```rust
        /// # use ryft_core::{DimensionPow, DimensionValue, ProgramError};
        /// # fn main() -> Result<(), ProgramError> {
        /// let output = DimensionValue::constant(3)?.dimension_pow(&DimensionValue::constant(4)?)?;
        /// assert_eq!(output.extent(), 81);
        /// # Ok(())
        /// # }
        /// ```
        trait;
        /// Returns `self` raised to the non-negative integer power `other`.
        fn(other);
    },
);

impl<A: Value<Type = ArrayType>> From<DimensionPowOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: DimensionPowOperation) -> Self {
        Self::Dimension(operation.into())
    }
}

impl DimensionPow for DimensionValue {
    fn dimension_pow(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionPowOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
        let inputs = &[self.r#type().into_owned(), right.r#type().into_owned()];
        let output_type = operation.infer_output_types(inputs, &[])?.remove(0);
        let extent =
            checked_power(self.extent(), right.extent()).ok_or_else(|| DimensionError::ArithmeticOverflow {
                message: format!(
                    "dimension arithmetic overflow while raising a dimension to a dimension power with operands \
                     {}={}, {}={}",
                    self.r#type().variable(),
                    self.extent(),
                    right.r#type().variable(),
                    right.extent(),
                ),
            })?;
        Ok(Self::new(output_type, extent)?)
    }
}

/// Computes `base.pow(exponent)` without narrowing `exponent`.
fn checked_power(mut base: usize, mut exponent: usize) -> Option<usize> {
    let mut output = 1usize;
    while exponent != 0 {
        if exponent & 1 != 0 {
            output = output.checked_mul(base)?;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base.checked_mul(base)?;
        }
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionOperation, DimensionValue};
    use crate::programs::{EffectClass, EffectClasses};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_dimension_pow() {
        let base = DimensionType::new("base", DimensionBounds::new(0, Some(3)).unwrap());
        let exponent = DimensionType::new("exponent", DimensionBounds::new(0, Some(3)).unwrap());
        let operation = DimensionPowOperation::new(&base, &exponent).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_POW_OPERATION_NAME);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(0, Some(5)).unwrap());
        let bounded_left = DimensionType::new("bounded_left", DimensionBounds::new(2, Some(9)).unwrap());
        let bounded_right = DimensionType::new("bounded_right", DimensionBounds::new(1, Some(5)).unwrap());
        assert_eq!(
            DimensionPowOperation::new(&bounded_left, &bounded_right).unwrap().effects().classes(),
            EffectClasses::NONE,
        );
        let unbounded = DimensionType::new("unbounded", DimensionBounds::unbounded());
        assert_eq!(
            DimensionPowOperation::new(&unbounded, &bounded_right).unwrap().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion),
        );

        // Specializing the input bounds recomputes this operation's output bounds.
        let declared_left = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionPowOperation::new(&declared_left, &declared_right).unwrap();
        let exact_left = DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap());
        let exact_right = DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap());
        let output = operation.infer_output_types(&[exact_left, exact_right], &[]).unwrap();
        assert_eq!(output[0].extent(), Some(36));

        assert_eq!(
            DimensionValue::constant(3)
                .unwrap()
                .dimension_pow(&DimensionValue::constant(4).unwrap())
                .unwrap()
                .extent(),
            81,
        );
    }

    #[test]
    fn test_dimension_pow_identity() {
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(value, zero, one)| {
                Ok(vec![value.dimension_pow(&one)?, one.dimension_pow(&value)?, zero.dimension_pow(&value)?])
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
                in (%0, %2, %1)"},
        );
    }

    #[test]
    fn test_dimension_pow_identity_preserves_zero_exponent() {
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |zero| zero.dimension_pow(&zero),
            DimensionValue::constant(0).unwrap().r#type().into_owned(),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<0> .
                let %1:dimension<1> = dimension_pow %0 %0
                in (%1)"},
        );
        assert_eq!(program.interpret(DimensionValue::constant(0).unwrap()).unwrap().extent(), 1);
    }
}
