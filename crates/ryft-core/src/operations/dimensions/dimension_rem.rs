use crate::arrays::{ArrayIrOperation, ArrayType, DimensionBounds, DimensionError, DimensionType, DimensionValue};
use crate::macros::define_dimension_arithmetic_operation;
use crate::operations::{Rem, RemOperation};
use crate::parameters::Parameter;
use crate::programs::{Operation, OperationFoldOutput, ProgramError, Typed, Value};

/// Canonical operation name for [`DimensionRemOperation`].
pub const DIMENSION_REM_OPERATION_NAME: &str = "dimension_rem";

define_dimension_arithmetic_operation!(
    /// Checked dimension-remainder operation used by [`Rem`].
    DimensionRemOperation, DIMENSION_REM_OPERATION_NAME,
    Rem, rem,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} % {}", left.variable(), right.variable())
    },
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (_, left_maximum) = left.bounds().representable_extent_range()?;
        let (_, right_maximum) = right.bounds().representable_extent_range()?;
        if right_maximum == 0 {
            return Err(DimensionError::RequirementViolation {
                message: format!("{} > 0 is impossible from declared bounds", right.variable()),
            });
        }
        let bounds = if right.bounds().lower() > 0
            && (left.variable() == right.variable() || left_maximum == 0 || right.extent() == Some(1))
        {
            DimensionBounds::new(0, Some(1))?
        } else if let (Some(left), Some(right)) = (left.extent(), right.extent()) {
            let remainder = left % right;
            DimensionBounds::new(remainder, remainder.checked_add(1))?
        } else {
            DimensionBounds::new(0, left_maximum.min(right_maximum - 1).checked_add(1))?
        };
        Ok((bounds, right.bounds().lower() == 0))
    },
    fold = |left: &DimensionType, right: &DimensionType| {
        if left.maximum_extent().is_some_and(|maximum| maximum < right.bounds().lower()) {
            Some(OperationFoldOutput::Input(0))
        } else {
            None
        }
    },
    provider = RemOperation<DimensionType>,
);

impl<A: Value<Type = ArrayType>> From<DimensionRemOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: DimensionRemOperation) -> Self {
        Self::Dimension(operation.into())
    }
}

impl Rem for DimensionValue {
    fn rem(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionRemOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
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
        Ok(Self::new(output_type, self.extent() % right.extent())?)
    }
}

impl std::ops::Rem for DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn rem(self, right: DimensionValue) -> Self::Output {
        Rem::rem(&self, &right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Rem<&DimensionValue> for DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn rem(self, right: &DimensionValue) -> Self::Output {
        Rem::rem(&self, right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Rem<DimensionValue> for &DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn rem(self, right: DimensionValue) -> Self::Output {
        Rem::rem(self, &right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Rem<&DimensionValue> for &DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn rem(self, right: &DimensionValue) -> Self::Output {
        Rem::rem(self, right).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayIrType, ArrayIrValue, DimensionBounds, DimensionOperation, DimensionValue};
    use crate::operations::assertions::AssertOperation;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EffectClass, EffectClasses, ProgramBuilder};
    use crate::tracing::TracingContext;

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

        // Specializing the input bounds recomputes this operation's output bounds.
        let declared_left = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionRemOperation::new(&declared_left, &declared_right).unwrap();
        let exact_left = DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap());
        let exact_right = DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap());
        let output = operation.infer_output_types(&[exact_left, exact_right], &[]).unwrap();
        assert_eq!(output[0].extent(), Some(0));

        let positive = DimensionType::new("positive", DimensionBounds::new(1, None).unwrap());
        let one = DimensionValue::constant(1).unwrap();
        let zero = DimensionValue::constant(0).unwrap();
        for (left, right) in
            [(&positive, &positive), (&positive, one.r#type().as_ref()), (zero.r#type().as_ref(), &positive)]
        {
            assert_eq!(
                DimensionRemOperation::new(left, right).unwrap().output_bounds(),
                DimensionBounds::new(0, Some(1)).unwrap()
            );
        }
        assert_eq!(
            DimensionRemOperation::new(&maybe_zero, &maybe_zero).unwrap().output_bounds(),
            DimensionBounds::new(0, Some(4)).unwrap(),
        );

        assert_eq!(
            DimensionValue::constant(7).unwrap().rem(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            1,
        );

        let left = DimensionValue::constant(7).unwrap();
        let right = DimensionValue::constant(3).unwrap();
        assert_eq!((left.clone() % right.clone()).extent(), 1);
        assert_eq!((left.clone() % &right).extent(), 1);
        assert_eq!((&left % right.clone()).extent(), 1);
        assert_eq!((&left % &right).extent(), 1);
    }
    #[test]
    fn test_dimension_rem_partial_evaluation_retains_unproven_congruence() {
        let extent = DimensionType::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let four = DimensionValue::constant(4).unwrap();
        let two = DimensionValue::constant(2).unwrap();
        let multiplication = DimensionMulOperation::new(&extent, four.r#type().as_ref()).unwrap();
        let product_type = multiplication
            .infer_output_types(&[extent.clone(), four.r#type().into_owned()], &[])
            .unwrap()
            .remove(0);
        let remainder = DimensionRemOperation::new(&product_type, two.r#type().as_ref()).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(extent.clone().into());
        let four = builder.add_constant(ArrayIrValue::Dimension(four));
        let two = builder.add_constant(ArrayIrValue::Dimension(two));
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let product = builder.add_instruction(multiplication, Vec::new(), vec![input, four], None).unwrap()[0];
        let remainder = builder.add_instruction(remainder, Vec::new(), vec![product, two], None).unwrap()[0];
        let predicate = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::Equal),
                Vec::new(),
                vec![remainder, zero],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new("product must be divisible by `2`"),
                Vec::new(),
                vec![predicate],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![product],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // Intervals and identities do not reconstruct modular congruences, even though multiplying by four makes
        // this assertion mathematically true. The general predicate remains an ordered runtime assertion.
        let residual = program.partially_evaluate(&[PartialValue::Unknown(extent.into())]).unwrap();
        assert_eq!(
            residual.program().to_string(),
            indoc! {r#"
                lambda %0:dimension<extent ∈ [1, 9)> .
                let %1:dimension<4> = const 4
                    %2:dimension<extent * 4 ∈ [4, 33)> = dimension_mul %0 %1
                    %3:dimension<2> = const 2
                    %4:dimension<extent * 4 % 2 ∈ [0, 2)> = dimension_rem %2 %3
                    %5:dimension<0> = const 0
                    %6:bool[] = compare [direction=Equal] %4 %5
                    () = assert [message="product must be divisible by `2`", labels=[]] %6
                in (%2)
            "#}
            .trim_end(),
        );
        assert_eq!(residual.program().effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
    }

    #[test]
    fn test_dimension_rem_partial_evaluation_identity() {
        let input_types = [
            DimensionType::new("value", DimensionBounds::new(2, Some(9)).unwrap()),
            DimensionValue::constant(0).unwrap().r#type().into_owned(),
            DimensionValue::constant(1).unwrap().r#type().into_owned(),
        ];
        let mut builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = [(1, 0), (2, 0)]
            .into_iter()
            .map(|(left, right)| {
                builder
                    .add_instruction(
                        DimensionRemOperation::new(&input_types[left], &input_types[right]).unwrap(),
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
                let %3:dimension<0> = dimension_rem %1 %0
                    %4:dimension<1> = dimension_rem %2 %0
                in (%3, %4)"},
        );
        let evaluation = program
            .partially_evaluate(&program.input_types().into_iter().map(PartialValue::Unknown).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:dimension<value ∈ [2, 9)>, %1:dimension<0>, %2:dimension<1> .
                in (%1, %2)"},
        );
    }

    #[test]
    fn test_dimension_rem_identity_retains_zero_divisor_check() {
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(zero, divisor)| zero.rem(&divisor),
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
                let %2:dimension<0> = dimension_rem [requires_runtime_assertion=true] %0 %1
                in (%2)"},
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let evaluation = program
            .partially_evaluate(&program.input_types().into_iter().map(PartialValue::Unknown).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(evaluation.program().to_string(), program.to_string());
        assert_eq!(evaluation.program().effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let error = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |zero| zero.rem(&zero),
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
