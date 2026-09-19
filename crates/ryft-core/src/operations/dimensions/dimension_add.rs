use crate::arrays::{DimensionBounds, DimensionError, DimensionType, MAX_DIMENSION_EXTENT};
use crate::macros::define_dimension_arithmetic_operation;
use crate::operations::math::add::{Add, AddOperation};
use crate::parameters::Parameter;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`DimensionAddOperation`].
pub const DIMENSION_ADD_OPERATION_NAME: &str = "dimension_add";

define_dimension_arithmetic_operation!(
    /// Checked dimension-addition operation used by [`Add`].
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
                "dimension arithmetic overflow while deriving `{DIMENSION_ADD_OPERATION_NAME}` result bounds \
                 with operands `{left}` and `{right}`",
            ),
        })?;
        let maximum = left_maximum.saturating_add(right_maximum).min(MAX_DIMENSION_EXTENT);
        let bounds = DimensionBounds::new(lower, maximum.checked_add(1))?;
        let requires_runtime_assertion = left.maximum_extent()
            .zip(right.maximum_extent())
            .and_then(|(left, right)| left.checked_add(right))
            .is_none_or(|result| result > MAX_DIMENSION_EXTENT);
        Ok((bounds, requires_runtime_assertion))
    },
    provider = AddOperation<DimensionType>,
);

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionBounds, DimensionOperation, DimensionValue};
    use crate::contexts::{Context, EagerContext};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::{ProgramBuilder, Typed};
    use crate::tracing::Trace;

    use super::*;

    #[test]
    fn test_dimension_add_operation() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionAddOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_ADD_OPERATION_NAME);
        assert_eq!(operation.left_type(), &left);
        assert_eq!(operation.right_type(), &right);
        assert_eq!(operation.output_bounds(), DimensionBounds::new(3, Some(13)).unwrap());
        assert_eq!(
            DimensionValue::constant(7).unwrap().add(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            10,
        );
    }

    #[test]
    fn test_dimension_add_program() {
        let left_type = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionAddOperation::new(&left_type, &right_type).unwrap();

        let mut builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let left = builder.add_input(left_type.clone());
        let right = builder.add_input(right_type.clone());
        let result = builder.add_instruction(operation, Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<DimensionValue>, Vec<DimensionValue>>(
                vec![result],
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

        // Relocating the program into a fresh region renames the internal result identity but renders identically.
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

        let result_type = program.output_types().remove(0);
        assert_ne!(result_type.variable(), left_type.variable());
        assert_ne!(result_type.variable(), right_type.variable());
        let left = DimensionValue::new(left_type.clone(), 7).unwrap();
        let right = DimensionValue::new(right_type.clone(), 3).unwrap();
        assert_eq!(program.interpret(vec![left.clone(), right.clone()]).unwrap()[0].extent(), 10);

        let evaluation = program.partially_evaluate(&[PartialValue::Known(left), PartialValue::Known(right)]).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        let PartialEvaluationOutput::Known(output) = &evaluation.outputs()[0] else {
            panic!("expected the dimension addition to fold to a known value");
        };
        assert_eq!(output.extent(), 10);
        assert_eq!(output.r#type().bounds(), result_type.bounds());

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
    }
}
