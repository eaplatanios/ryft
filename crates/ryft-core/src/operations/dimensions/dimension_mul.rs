use crate::arrays::{
    ArrayIrOperation, ArrayType, DimensionBounds, DimensionError, DimensionType, DimensionValue, MAX_DIMENSION_EXTENT,
};
use crate::macros::define_dimension_arithmetic_operation;
use crate::operations::{Mul, MulOperation};
use crate::parameters::Parameter;
use crate::programs::{Operation, OperationFoldOutput, ProgramError, Typed, Value};

/// Canonical operation name for [`DimensionMulOperation`].
pub const DIMENSION_MUL_OPERATION_NAME: &str = "dimension_mul";

define_dimension_arithmetic_operation!(
    /// Checked dimension-multiplication operation used by [`Mul`] for [`DimensionValue`]s.
    DimensionMulOperation,
    DIMENSION_MUL_OPERATION_NAME,
    Mul,
    mul,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} * {}", left.variable(), right.variable())
    },
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (right_lower, right_maximum) = right.bounds().representable_extent_range()?;
        let lower = left_lower.checked_mul(right_lower).ok_or_else(|| DimensionError::ArithmeticOverflow {
            message: format!(
                "dimension arithmetic overflow while deriving `{DIMENSION_MUL_OPERATION_NAME}` output bounds \
                 with operands `{left}` and `{right}`",
            ),
        })?;
        let maximum = left_maximum.saturating_mul(right_maximum).min(MAX_DIMENSION_EXTENT);
        let bounds = DimensionBounds::new(lower, maximum.checked_add(1))?;
        let requires_runtime_assertion = left.maximum_extent()
            .zip(right.maximum_extent())
            .and_then(|(left, right)| left.checked_mul(right))
            .is_none_or(|output| output > MAX_DIMENSION_EXTENT);
        Ok((bounds, requires_runtime_assertion))
    },
    fold = |left: &DimensionType, right: &DimensionType| {
        if left.extent() == Some(0) || right.extent() == Some(1) {
            Some(OperationFoldOutput::Input(0))
        } else if right.extent() == Some(0) || left.extent() == Some(1) {
            Some(OperationFoldOutput::Input(1))
        } else {
            None
        }
    },
    provider = MulOperation<DimensionType>,
);

impl<A: Value<Type = ArrayType>> From<DimensionMulOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: DimensionMulOperation) -> Self {
        Self::Dimension(operation.into())
    }
}

impl Mul for DimensionValue {
    fn mul(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionMulOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
        let inputs = &[self.r#type().into_owned(), right.r#type().into_owned()];
        let output_type = operation.infer_output_types(inputs, &[])?.remove(0);
        let extent = self.extent().checked_mul(right.extent()).ok_or_else(|| DimensionError::ArithmeticOverflow {
            message: format!(
                "dimension arithmetic overflow while multiplying dimensions with operands {}={}, {}={}",
                self.r#type().variable(),
                self.extent(),
                right.r#type().variable(),
                right.extent(),
            ),
        })?;
        Ok(Self::new(output_type, extent)?)
    }
}

impl std::ops::Mul for DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn mul(self, right: DimensionValue) -> Self::Output {
        Mul::mul(&self, &right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Mul<&DimensionValue> for DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn mul(self, right: &DimensionValue) -> Self::Output {
        Mul::mul(&self, right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Mul<DimensionValue> for &DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn mul(self, right: DimensionValue) -> Self::Output {
        Mul::mul(self, &right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Mul<&DimensionValue> for &DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn mul(self, right: &DimensionValue) -> Self::Output {
        Mul::mul(self, right).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionOperation, DimensionValue};
    use crate::contexts::StagingContext;
    use crate::operations::arithmetic::Mul;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EffectClass, EffectClasses, ProgramBuilder};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_dimension_mul() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionMulOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_MUL_OPERATION_NAME);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(2, Some(33)).unwrap());
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        let unbounded = DimensionType::new("unbounded", DimensionBounds::unbounded());
        assert_eq!(
            DimensionMulOperation::new(&unbounded, &right).unwrap().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion),
        );

        // Specializing the input bounds recomputes this operation's output bounds.
        let declared_left = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionMulOperation::new(&declared_left, &declared_right).unwrap();
        let exact_left = DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap());
        let exact_right = DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap());
        let output = operation.infer_output_types(&[exact_left, exact_right], &[]).unwrap();
        assert_eq!(output[0].extent(), Some(12));

        assert_eq!(
            DimensionValue::constant(7).unwrap().mul(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            21,
        );

        let left = DimensionValue::constant(7).unwrap();
        let right = DimensionValue::constant(3).unwrap();
        assert_eq!((left.clone() * right.clone()).extent(), 21);
        assert_eq!((left.clone() * &right).extent(), 21);
        assert_eq!((&left * right.clone()).extent(), 21);
        assert_eq!((&left * &right).extent(), 21);
    }

    #[test]
    fn test_dimension_mul_partial_evaluation_identity() {
        let input_types = [
            DimensionType::new("value", DimensionBounds::new(2, Some(9)).unwrap()),
            DimensionValue::constant(0).unwrap().r#type().into_owned(),
            DimensionValue::constant(1).unwrap().r#type().into_owned(),
        ];
        let mut builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = [(0, 2), (2, 0), (0, 1), (1, 0)]
            .into_iter()
            .map(|(left, right)| {
                builder
                    .add_instruction(
                        DimensionMulOperation::new(&input_types[left], &input_types[right]).unwrap(),
                        Vec::new(),
                        vec![inputs[left], inputs[right]],
                        None,
                    )
                    .unwrap()[0]
            })
            .collect::<Vec<_>>();
        let output_count = outputs.len();
        let program = builder
            .build::<Vec<DimensionValue>, Vec<DimensionValue>>(
                outputs,
                vec![Placeholder; 3],
                vec![Placeholder; output_count],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<value ∈ [2, 9)>, %1:dimension<0>, %2:dimension<1> .
                let %3:dimension<value * 1 ∈ [2, 9)> = dimension_mul %0 %2
                    %4:dimension<1 * value ∈ [2, 9)> = dimension_mul %2 %0
                    %5:dimension<0> = dimension_mul %0 %1
                    %6:dimension<0> = dimension_mul %1 %0
                in (%3, %4, %5, %6)"},
        );
        let evaluation = program
            .partially_evaluate(&program.input_types().into_iter().map(PartialValue::Unknown).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:dimension<value ∈ [2, 9)>, %1:dimension<0>, %2:dimension<1> .
                in (%0, %0, %1, %1)"},
        );
    }

    #[test]
    fn test_dimension_mul_identity_validates_discarded_input() {
        if let Some(invalid_extent) = MAX_DIMENSION_EXTENT.checked_add(1) {
            let context = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::new();
            let invalid =
                context.input(DimensionType::new("invalid", DimensionBounds::new(invalid_extent, None).unwrap()));
            let zero = context.input(DimensionValue::constant(0).unwrap().r#type().into_owned());
            assert_eq!(
                invalid.mul(&zero).unwrap_err().downcast_custom::<DimensionError>(),
                Some(&DimensionError::ExtentExceedsBackendWidth {
                    value: invalid_extent,
                    maximum: MAX_DIMENSION_EXTENT,
                }),
            );
            assert_eq!(
                zero.mul(&invalid).unwrap_err().downcast_custom::<DimensionError>(),
                Some(&DimensionError::ExtentExceedsBackendWidth {
                    value: invalid_extent,
                    maximum: MAX_DIMENSION_EXTENT,
                }),
            );
        }
    }
}
