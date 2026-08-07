use crate::arrays::{DimensionBounds, DimensionError, DimensionType, MAX_DIMENSION_EXTENT};
use crate::macros::check_count;
use crate::macros::define_arithmetic_dimension_operation;
use crate::operations::math::{Add, AddOperation};
use crate::parameters::Parameter;
use crate::programs::{OperationProvider, ProgramError};

use super::{bounds_overflow, maximum_extent, representable_extent_range};

/// Canonical operation name for [`DimensionAddOperation`].
pub const DIMENSION_ADD_OPERATION_NAME: &str = "dimension_add";

define_arithmetic_dimension_operation!(
    /// Checked dimension-addition operation used by [`Add`].
    DimensionAddOperation,
    DIMENSION_ADD_OPERATION_NAME,
    Add,
    add,
    result_name = |left: &DimensionType, right: &DimensionType| {
        format!("{} + {}", left.variable(), right.variable())
    },
    infer_bounds = infer_bounds,
);

impl OperationProvider<DimensionType> for AddOperation<DimensionType> {
    type Operation = DimensionAddOperation;

    fn provide(input_types: &[&DimensionType]) -> Result<Self::Operation, ProgramError> {
        check_count!("input", input_types, 2, ProgramError);
        Ok(DimensionAddOperation::new(input_types[0], input_types[1])?)
    }
}

/// Derives sound bounds for checked dimension addition and reports whether runtime overflow remains possible.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<(DimensionBounds, bool), DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    let overflow = || bounds_overflow(DIMENSION_ADD_OPERATION_NAME, left, right);
    let lower = left_lower.checked_add(right_lower).ok_or_else(overflow)?;
    let maximum = left_maximum.saturating_add(right_maximum).min(MAX_DIMENSION_EXTENT);
    let bounds = DimensionBounds::new(lower, maximum.checked_add(1))?;
    let requires_runtime_assertion = maximum_extent(left)
        .zip(maximum_extent(right))
        .and_then(|(left, right)| left.checked_add(right))
        .is_none_or(|result| result > MAX_DIMENSION_EXTENT);
    Ok((bounds, requires_runtime_assertion))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::DimensionBounds;
    use crate::backends::dimensions::{DimensionOperation, DimensionValue};
    use crate::contexts::{Context, EagerContext};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::ProgramBuilder;
    use crate::tracing::Trace;

    use super::super::test_dimension_type;
    use super::*;

    #[test]
    fn test_dimension_add_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionAddOperation::new(&left, &right).unwrap();
        assert_eq!(operation.to_string(), DIMENSION_ADD_OPERATION_NAME);
        assert_eq!(operation.left_type(), &left);
        assert_eq!(operation.right_type(), &right);
        assert_eq!(operation.result_bounds(), DimensionBounds::new(3, Some(13)).unwrap());
        assert_eq!(
            DimensionValue::constant(7).unwrap().add(&DimensionValue::constant(3).unwrap()).unwrap().extent(),
            10,
        );
    }

    #[test]
    fn test_dimension_add_program() {
        let left_type = test_dimension_type("left", 1, 9);
        let right_type = test_dimension_type("right", 1, 5);
        let operation = DimensionAddOperation::new(&left_type, &right_type).unwrap();

        let mut builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let left = builder.add_input(left_type.clone());
        let right = builder.add_input(right_type.clone());
        let result = builder.add_instruction(operation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<Vec<DimensionValue>, Vec<DimensionValue>>(
                vec![result],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

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
