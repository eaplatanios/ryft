use crate::arrays::{
    ArrayIrOperation, ArrayType, DimensionBounds, DimensionError, DimensionType, DimensionValue, MAX_DIMENSION_EXTENT,
};
use crate::macros::define_dimension_arithmetic_operation;
use crate::operations::{Add, AddOperation};
use crate::parameters::Parameter;
use crate::programs::{Operation, ProgramError, Typed, Value};

/// Canonical operation name for [`DimensionAddOperation`].
pub const DIMENSION_ADD_OPERATION_NAME: &str = "dimension_add";

define_dimension_arithmetic_operation!(
    /// Checked dimension-addition operation used by [`Add`] for [`DimensionValue`]s.
    DimensionAddOperation,
    DIMENSION_ADD_OPERATION_NAME,
    Add,
    add,
    output_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} + {}", left.variable(), right.variable())
    },
    infer_bounds = |left: &DimensionType, right: &DimensionType| -> Result<(DimensionBounds, bool), DimensionError> {
        let (left_lower, left_maximum) = left.bounds().representable_extent_range()?;
        let (right_lower, right_maximum) = right.bounds().representable_extent_range()?;
        let lower = left_lower.checked_add(right_lower).ok_or_else(|| DimensionError::ArithmeticOverflow {
            message: format!(
                "dimension arithmetic overflow while deriving `{DIMENSION_ADD_OPERATION_NAME}` output bounds \
                 with operands `{left}` and `{right}`",
            ),
        })?;
        let maximum = left_maximum.saturating_add(right_maximum).min(MAX_DIMENSION_EXTENT);
        let bounds = DimensionBounds::new(lower, maximum.checked_add(1))?;
        let requires_runtime_assertion = left.maximum_extent()
            .zip(right.maximum_extent())
            .and_then(|(left, right)| left.checked_add(right))
            .is_none_or(|output| output > MAX_DIMENSION_EXTENT);
        Ok((bounds, requires_runtime_assertion))
    },
    fold = |left: &DimensionType, right: &DimensionType| {
        if right.extent() == Some(0) {
            Some(vec![0])
        } else if left.extent() == Some(0) {
            Some(vec![1])
        } else {
            None
        }
    },
    provider = AddOperation<DimensionType>,
);

impl<A: Value<Type = ArrayType>> From<DimensionAddOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: DimensionAddOperation) -> Self {
        Self::Dimension(operation.into())
    }
}

impl Add for DimensionValue {
    fn add(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionAddOperation::new(self.r#type().as_ref(), right.r#type().as_ref())?;
        let inputs = &[self.r#type().into_owned(), right.r#type().into_owned()];
        let output_type = operation.infer_output_types(inputs, &[])?.remove(0);
        let extent = self.extent().checked_add(right.extent()).ok_or_else(|| DimensionError::ArithmeticOverflow {
            message: format!(
                "dimension arithmetic overflow while adding dimensions with operands {}={}, {}={}",
                self.r#type().variable(),
                self.extent(),
                right.r#type().variable(),
                right.extent(),
            ),
        })?;
        Ok(Self::new(output_type, extent)?)
    }
}

impl std::ops::Add for DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn add(self, right: DimensionValue) -> Self::Output {
        Add::add(&self, &right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Add<&DimensionValue> for DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn add(self, right: &DimensionValue) -> Self::Output {
        Add::add(&self, right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Add<DimensionValue> for &DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn add(self, right: DimensionValue) -> Self::Output {
        Add::add(self, &right).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Add<&DimensionValue> for &DimensionValue {
    type Output = DimensionValue;

    #[inline]
    fn add(self, right: &DimensionValue) -> Self::Output {
        Add::add(self, right).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionOperation, DimensionValue};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationOutput, PartialEvaluationValue, PartialValue};
    use crate::programs::{EffectClass, EffectClasses, ProgramBuilder, RegionInterface, TypeError, Typed};
    use crate::tracing::{Trace, TracingContext};

    use super::*;

    #[test]
    fn test_dimension_add() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionAddOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_ADD_OPERATION_NAME);
        assert_eq!(operation.left_type(), &left);
        assert_eq!(operation.right_type(), &right);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(3, Some(13)).unwrap());
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(
                &[left.clone(), right.clone()],
                &[RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        let unbounded = DimensionType::new("unbounded", DimensionBounds::unbounded());
        assert_eq!(
            DimensionAddOperation::new(&unbounded, &right).unwrap().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion),
        );

        // Specializing the input bounds recomputes this operation's output bounds.
        let declared_left = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionAddOperation::new(&declared_left, &declared_right).unwrap();
        let exact_left = DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap());
        let exact_right = DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap());
        let output = operation.infer_output_types(&[exact_left, exact_right], &[]).unwrap();
        assert_eq!(output[0].extent(), Some(8));

        assert_eq!(
            DimensionValue::constant(7).unwrap().add(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            10,
        );

        let left_type = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionAddOperation::new(&left_type, &right_type).unwrap();

        let mut builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let left = builder.add_input(left_type.clone());
        let right = builder.add_input(right_type.clone());
        let output = builder.add_instruction(operation, Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<DimensionValue>, Vec<DimensionValue>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // A dimension-only program renders with the same `lambda ... let ... in (...)` grammar as an array program;
        // each dimension type shows its diagnostic name beside the bounds inferred for it.
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<left ∈ [1, 9)>, %1:dimension<right ∈ [1, 5)> .
                let %2:dimension<left + right ∈ [2, 13)> = dimension_add %0 %1
                in (%2)"},
        );

        // Relocating the program into a fresh region renames the internal output identity but renders identically.
        let mut relocated_builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let relocated_inputs =
            vec![relocated_builder.add_input(left_type.clone()), relocated_builder.add_input(right_type.clone())];
        let relocated_outputs = relocated_builder.splice_program(&program, &relocated_inputs).unwrap();
        let relocated = relocated_builder
            .build::<Vec<DimensionValue>, Vec<DimensionValue>>(
                relocated_outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(relocated.to_string(), program.to_string());

        let output_type = program.output_types().remove(0);
        assert_ne!(output_type.variable(), left_type.variable());
        assert_ne!(output_type.variable(), right_type.variable());
        let left = DimensionValue::new(left_type.clone(), 7).unwrap();
        let right = DimensionValue::new(right_type.clone(), 3).unwrap();
        assert_eq!(program.interpret(vec![left.clone(), right.clone()]).unwrap()[0].extent(), 10);

        let evaluation = program.partially_evaluate(&[PartialValue::Known(left), PartialValue::Known(right)]).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        let PartialEvaluationOutput::Known(output) = &evaluation.outputs()[0] else {
            panic!("expected the dimension addition to fold to a known value");
        };
        assert_eq!(output.extent(), 10);
        assert_eq!(output.r#type().bounds(), output_type.bounds());

        let (traced_type, traced_program) = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |left| {
                let right = left.context().lift(DimensionValue::constant(2)?)?;
                left.add(&right)?.add(&right)
            },
            left_type.clone(),
        )
        .unwrap();
        assert_eq!(traced_type.bounds(), DimensionBounds::new(5, Some(13)).unwrap());
        assert_eq!(traced_program.interpret(DimensionValue::new(left_type, 6).unwrap()).unwrap().extent(), 10,);

        let left = DimensionValue::constant(7).unwrap();
        let right = DimensionValue::constant(3).unwrap();
        assert_eq!((left.clone() + right.clone()).extent(), 10);
        assert_eq!((left.clone() + &right).extent(), 10);
        assert_eq!((&left + right.clone()).extent(), 10);
        assert_eq!((&left + &right).extent(), 10);
    }

    #[test]
    fn test_dimension_add_folding() {
        use crate::operations::dimensions::DimensionDivOperation;

        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |divisor| {
                let context = divisor.context();
                let zero = context.lift(DimensionValue::constant(0)?)?;
                let quotient = context
                    .bind(
                        DimensionDivOperation::new(zero.r#type().as_ref(), divisor.r#type().as_ref())?,
                        Vec::new(),
                        &[zero.clone(), divisor.clone()],
                    )?
                    .remove(0);
                quotient.add(&zero)
            },
            DimensionType::new("divisor", DimensionBounds::new(0, Some(9)).unwrap()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:dimension<divisor ∈ [0, 9)> .
            let %1:dimension<0> = const 0
                %2:dimension<0> = dimension_div [requires_runtime_assertion=true] %1 %0
            in (%2)"}
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let left_context = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let right_context = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let value = left_context.input(DimensionType::new("value", DimensionBounds::new(1, Some(9)).unwrap()));
        let zero = right_context.input(DimensionValue::constant(0).unwrap().r#type().into_owned());
        assert_eq!(value.add(&zero).unwrap_err(), ProgramError::MismatchedProgramBuilders);
    }

    #[test]
    fn test_dimension_add_partial_evaluation_identity() {
        let input_types = [
            DimensionType::new("value", DimensionBounds::new(2, Some(9)).unwrap()),
            DimensionValue::constant(0).unwrap().r#type().into_owned(),
            DimensionValue::constant(1).unwrap().r#type().into_owned(),
        ];
        let mut builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = [(0, 1), (1, 0)]
            .into_iter()
            .map(|(left, right)| {
                builder
                    .add_instruction(
                        DimensionAddOperation::new(&input_types[left], &input_types[right]).unwrap(),
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
                let %3:dimension<value + 0 ∈ [2, 9)> = dimension_add %0 %1
                    %4:dimension<0 + value ∈ [2, 9)> = dimension_add %1 %0
                in (%3, %4)"},
        );
        let evaluation = program
            .partially_evaluate(&program.input_types().into_iter().map(PartialValue::Unknown).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:dimension<value ∈ [2, 9)>, %1:dimension<0>, %2:dimension<1> .
                in (%0, %0)"},
        );
    }

    #[test]
    fn test_dimension_add_partial_evaluation_refined_identity() {
        let value = DimensionType::new("value", DimensionBounds::unbounded());
        let offset = DimensionType::new("offset", DimensionBounds::new(0, Some(2)).unwrap());
        let (_, program) = TracingContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(value, offset)| value.add(&offset),
            (value.clone(), offset),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(value),
                PartialValue::Unknown(DimensionValue::constant(0).unwrap().r#type().into_owned()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:dimension<value ∈ [0, ∞)>, %1:dimension<0> .
                in (%0)"},
        );
        assert_eq!(evaluation.program().effects().classes(), EffectClasses::NONE);
    }
    #[test]
    fn test_dimension_add_partial_evaluation_failure_order() {
        let left = DimensionType::new("left", DimensionBounds::unbounded());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(3)).unwrap());
        let operation = DimensionAddOperation::new(&left, &right).unwrap();
        let context =
            PartialEvaluationContext::new(EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::new());
        let unknown = context.unknown_input(left.clone(), 0);
        let zero = PartialEvaluationValue::known(DimensionValue::constant(0).unwrap());
        assert_eq!(
            context.fold_or_residualize(operation.clone(), Vec::new(), &[unknown, zero]).unwrap_err(),
            ProgramError::Type(TypeError::invalid(format!(
                "`dimension_add` input 1 has type `dimension<0>` but the operation was constructed for type `{right}`",
            )))
        );
        let outputs = context
            .fold_or_residualize(
                operation,
                Vec::new(),
                &[
                    PartialEvaluationValue::known(DimensionValue::new(left, 1).unwrap()),
                    PartialEvaluationValue::known(DimensionValue::new(right, 1).unwrap()),
                ],
            )
            .unwrap();
        assert!(outputs[0].is_unknown());
    }
}
