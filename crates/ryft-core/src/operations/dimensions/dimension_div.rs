use crate::arrays::{ArrayIrOperation, ArrayType, DimensionBounds, DimensionError, DimensionType, DimensionValue};
use crate::macros::define_dimension_arithmetic_operation;
use crate::operations::{Div, DivOperation};
use crate::parameters::Parameter;
use crate::programs::{Operation, OperationFoldOutput, ProgramError, Typed, Value};

/// Canonical operation name for [`DimensionDivOperation`].
pub const DIMENSION_DIV_OPERATION_NAME: &str = "dimension_div";

define_dimension_arithmetic_operation!(
    /// Checked integer division of non-negative [`DimensionValue`]s used by [`Div`] that returns the quotient rounded
    /// down (equivalently, truncated towards zero for non-negative operands). Division by zero returns an error.
    /// Construction rejects bounds that admit only a zero divisor. Otherwise, a divisor that may be zero requires
    /// a runtime assertion.
    DimensionDivOperation,
    DIMENSION_DIV_OPERATION_NAME,
    Div,
    div,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} / {}", left.variable(), right.variable())
    },
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (_, right_maximum) = right.bounds().representable_extent_range()?;
        if right_maximum == 0 {
            return Err(DimensionError::RequirementViolation {
                message: format!("{} > 0 is impossible from declared bounds", right.variable()),
            });
        }
        let positive_right_lower = right.bounds().lower().max(1);
        let bounds = DimensionBounds::new(
            left_lower / right_maximum,
            (left_maximum / positive_right_lower).checked_add(1),
        )?;
        Ok((bounds, right.bounds().lower() == 0))
    },
    fold = |left: &DimensionType, right: &DimensionType| {
        if right.extent() == Some(1) || (left.extent() == Some(0) && right.bounds().lower() > 0) {
            Some(OperationFoldOutput::Input(0))
        } else {
            None
        }
    },
    provider = DivOperation<DimensionType>,
);

impl<A: Value<Type = ArrayType>> From<DimensionDivOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: DimensionDivOperation) -> Self {
        Self::Dimension(operation.into())
    }
}

impl Div for DimensionValue {
    fn div(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionDivOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
        let inputs = &[self.r#type().into_owned(), right.r#type().into_owned()];
        let output_type = operation.infer_output_types(inputs, &[])?.remove(0);
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
        Ok(Self::new(output_type, self.extent() / right.extent())?)
    }
}

impl std::ops::Div for DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn div(self, right: DimensionValue) -> Self::Output {
        Div::div(&self, &right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Div<&DimensionValue> for DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn div(self, right: &DimensionValue) -> Self::Output {
        Div::div(&self, right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Div<DimensionValue> for &DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn div(self, right: DimensionValue) -> Self::Output {
        Div::div(self, &right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Div<&DimensionValue> for &DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn div(self, right: &DimensionValue) -> Self::Output {
        Div::div(self, right).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionOperation, DimensionValue};
    use crate::operations::math::div::Div;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EffectClass, EffectClasses, ProgramBuilder};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_dimension_div() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionDivOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_DIV_OPERATION_NAME);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(0, Some(9)).unwrap());
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        let maybe_zero = DimensionType::new("maybe_zero", DimensionBounds::new(0, Some(5)).unwrap());
        assert_eq!(
            DimensionDivOperation::new(&left, &maybe_zero).unwrap().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion),
        );

        // Specializing the input bounds recomputes this operation's output bounds.
        let declared_left = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionDivOperation::new(&declared_left, &declared_right).unwrap();
        let exact_left = DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap());
        let exact_right = DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap());
        let output = operation.infer_output_types(&[exact_left, exact_right], &[]).unwrap();
        assert_eq!(output[0].extent(), Some(3));

        assert_eq!(
            DimensionValue::constant(7).unwrap().div(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            2,
        );

        let left = DimensionValue::constant(7).unwrap();
        let right = DimensionValue::constant(3).unwrap();
        assert_eq!((left.clone() / right.clone()).extent(), 2);
        assert_eq!((left.clone() / &right).extent(), 2);
        assert_eq!((&left / right.clone()).extent(), 2);
        assert_eq!((&left / &right).extent(), 2);
    }

    #[test]
    fn test_dimension_div_partial_evaluation_identity() {
        let input_types = [
            DimensionType::new("value", DimensionBounds::new(2, Some(9)).unwrap()),
            DimensionValue::constant(0).unwrap().r#type().into_owned(),
            DimensionValue::constant(1).unwrap().r#type().into_owned(),
        ];
        let mut builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = [(0, 2), (1, 0)]
            .into_iter()
            .map(|(left, right)| {
                builder
                    .add_instruction(
                        DimensionDivOperation::new(&input_types[left], &input_types[right]).unwrap(),
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
                let %3:dimension<value / 1 ∈ [2, 9)> = dimension_div %0 %2
                    %4:dimension<0> = dimension_div %1 %0
                in (%3, %4)"},
        );
        let evaluation = program
            .partially_evaluate(&program.input_types().into_iter().map(PartialValue::Unknown).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:dimension<value ∈ [2, 9)>, %1:dimension<0>, %2:dimension<1> .
                in (%0, %1)"},
        );
    }

    #[test]
    fn test_dimension_div_identity_retains_zero_divisor_check() {
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(zero, divisor)| zero.div(&divisor),
            (
                DimensionValue::constant(0).unwrap().r#type().into_owned(),
                DimensionType::new("divisor", DimensionBounds::new(0, Some(9)).unwrap()),
            ),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<0>, %1:dimension<divisor ∈ [0, 9)> .
                let %2:dimension<0> = dimension_div [requires_runtime_assertion=true] %0 %1
                in (%2)"},
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let evaluation = program
            .partially_evaluate(&program.input_types().into_iter().map(PartialValue::Unknown).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(evaluation.program().to_string(), program.to_string());
        assert_eq!(evaluation.program().effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let error = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |zero| zero.div(&zero),
            DimensionValue::constant(0).unwrap().r#type().into_owned(),
        )
        .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "0 > 0 is impossible from declared bounds".to_owned(),
            }),
        );
    }
}
