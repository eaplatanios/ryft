use crate::macros::define_arithmetic_dimension_operation;
use crate::parameters::Parameter;
use crate::types::{DimensionError, DimensionType, MAX_DIMENSION_EXTENT};

use super::{bounds_from_extrema, bounds_overflow, evaluation_overflow, representable_extent_range};

/// Canonical operation name for [`DimensionAddOperation`].
pub const DIMENSION_ADD_OPERATION_NAME: &str = "dimension_add";

define_arithmetic_dimension_operation!(
    /// Adds two first-class runtime dimensions using checked nonnegative integer arithmetic.
    ///
    /// The result owns a fresh dimension identity whose bounds contain every representable sum admitted by the
    /// operands. Addition fails when either inferred bounds or a concrete result exceeds Ryft's portable dimension
    /// representation.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DimensionAdd, DimensionValue, ProgramError};
    /// # fn main() -> Result<(), ProgramError> {
    /// let result = DimensionValue::constant(3)?.add_dimension(&DimensionValue::constant(4)?)?;
    /// assert_eq!(result.extent(), 7);
    /// # Ok(())
    /// # }
    /// ```
    capability DimensionAdd {
        /// Returns the checked sum of `self` and `right`.
        fn add_dimension;
    }
    /// Checked dimension-addition operation used by [`DimensionAdd`].
    ///
    /// Refer to [`DimensionAdd`] for semantic details and an example.
    operation DimensionAddOperation {
        name = DIMENSION_ADD_OPERATION_NAME,
        result_name = |left: &DimensionType, right: &DimensionType| {
            format!("{} + {}", left.variable(), right.variable())
        },
        infer_bounds = infer_bounds,
        evaluate = evaluate,
    }
);

/// Derives sound bounds for checked dimension addition.
fn infer_bounds(left: &DimensionType, right: &DimensionType) -> Result<crate::DimensionBounds, DimensionError> {
    let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
    let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
    let overflow = || bounds_overflow(DIMENSION_ADD_OPERATION_NAME, left, right);
    let lower = left_lower.checked_add(right_lower).ok_or_else(overflow)?;
    let maximum = left_maximum.saturating_add(right_maximum).min(MAX_DIMENSION_EXTENT);
    bounds_from_extrema(lower, maximum)
}

/// Evaluates checked dimension addition.
fn evaluate(
    left_type: &DimensionType,
    left: usize,
    right_type: &DimensionType,
    right: usize,
) -> Result<usize, DimensionError> {
    left.checked_add(right)
        .ok_or_else(|| evaluation_overflow("adding runtime dimensions", left_type, left, right_type, right))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::dimensions::{DimensionOperation, DimensionValue};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::operations::dimensions::ArithmeticDimensionOperation;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::ProgramBuilder;
    use crate::programs::types::Typed;
    use crate::tracing::Trace;
    use crate::types::DimensionBounds;

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
        assert_ne!(operation.result_type().variable(), left.variable());
        assert_ne!(operation.result_type().variable(), right.variable());
        assert_eq!(operation.result_type().bounds(), DimensionBounds::new(3, Some(13)).unwrap());
        assert_eq!(operation.evaluate_extents(7, 3), Ok(10));
        assert_eq!(
            DimensionValue::constant(7)
                .unwrap()
                .add_dimension(&DimensionValue::constant(3).unwrap())
                .unwrap()
                .extent(),
            10,
        );
    }

    #[test]
    fn test_dimension_add_program() {
        let left_type = test_dimension_type("left", 1, 9);
        let right_type = test_dimension_type("right", 1, 5);
        let operation = DimensionAddOperation::new(&left_type, &right_type).unwrap();
        let result_type = operation.result_type();

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

        assert_eq!(program.output_types(), vec![result_type.clone()]);
        assert_ne!(result_type.variable(), left_type.variable());
        assert_ne!(result_type.variable(), right_type.variable());
        let left = DimensionValue::new(left_type.clone(), 7).unwrap();
        let right = DimensionValue::new(right_type.clone(), 3).unwrap();
        assert_eq!(program.interpret(vec![left.clone(), right.clone()]).unwrap()[0].extent(), 10);

        let evaluation = program.partially_evaluate(&[PartialValue::Known(left), PartialValue::Known(right)]).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        assert_eq!(
            evaluation.outputs(),
            &[PartialEvaluationOutput::Known(DimensionValue::new(result_type, 10).unwrap())],
        );

        let (traced_type, traced_program) = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |left| {
                let right = left.context().lift(DimensionValue::constant(2)?)?;
                let operation = DimensionAddOperation::new(&left.r#type().into_owned(), &right.r#type().into_owned())?;
                let mut outputs = left.context().stage_operation(operation, Vec::new(), &[&left, &right])?;
                Ok(outputs.remove(0))
            },
            left_type.clone(),
        )
        .unwrap();
        assert_eq!(traced_type.bounds(), DimensionBounds::new(3, Some(11)).unwrap());
        assert_eq!(traced_program.interpret(DimensionValue::new(left_type, 6).unwrap()).unwrap().extent(), 8,);
    }
}
