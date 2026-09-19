//! Dimension operation integration tests.

// TODO(eaplatanios): Review this module.

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, DimensionOperation};
    use crate::arrays::types::dimensions::{DimensionBounds, DimensionError, DimensionType};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::contexts::EagerContext;
    use crate::operations::{
        Add, DimensionMax, DimensionMin, DimensionPow, DimensionSaturatingSub, Div, Mul, Rem, Sub,
    };
    use crate::programs::ValueProjection;
    use crate::tracing::Trace;

    #[test]
    fn test_dimension_tracer_projection() {
        let rows = DimensionType::new("rows", DimensionBounds::new(5, Some(9)).unwrap());
        let columns = DimensionType::new("columns", DimensionBounds::new(1, Some(5)).unwrap());
        let (output_type, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(rows, columns)| {
                let rows = ValueProjection::<DimensionType>::into_projected(rows)?;
                let columns = ValueProjection::<DimensionType>::into_projected(columns)?;
                let padded = rows.mul(&columns)?.add(&columns)?;
                let trimmed = padded.dimension_saturating_sub(&rows)?;
                Ok((
                    trimmed.div(&columns)?.into_value(),
                    trimmed.rem(&columns)?.into_value(),
                    rows.dimension_pow(&columns)?.into_value(),
                    rows.dimension_max(&columns)?.sub(&rows.dimension_min(&columns)?)?.into_value(),
                ))
            },
            (ArrayIrType::Dimension(rows), ArrayIrType::Dimension(columns)),
        )
        .unwrap();
        assert_eq!(output_type.0.to_string(), "dimension<max(0, rows * columns + columns - rows) / columns ∈ [0, 32)>",);
        let expected = indoc! {"
            lambda %0:dimension<rows ∈ [5, 9)>, %1:dimension<columns ∈ [1, 5)> .
            let %2:dimension<rows * columns ∈ [5, 33)> = dimension_mul %0 %1
                %3:dimension<rows * columns + columns ∈ [6, 37)> = dimension_add %2 %1
                %4:dimension<max(0, rows * columns + columns - rows) ∈ [0, 32)> = dimension_saturating_sub %3 %0
                %5:dimension<max(0, rows * columns + columns - rows) / columns ∈ [0, 32)> = dimension_div %4 %1
                %6:dimension<max(0, rows * columns + columns - rows) % columns ∈ [0, 4)> = dimension_rem %4 %1
                %7:dimension<rows ^ columns ∈ [5, 4097)> = dimension_pow %0 %1
                %8:dimension<max(rows, columns) ∈ [5, 9)> = dimension_max %0 %1
                %9:dimension<min(rows, columns) ∈ [1, 5)> = dimension_min %0 %1
                %10:dimension<max(rows, columns) - min(rows, columns) ∈ [1, 8)> = dimension_sub %8 %9
            in (%5, %6, %7, %10)
        "};
        assert_eq!(program.to_string(), expected.trim_end());
    }

    #[test]
    fn test_dimension_tracer_operators() {
        let left_type = DimensionType::new("left", DimensionBounds::new(3, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(4)).unwrap());
        let (output_type, program) = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(left, right)| {
                let sum = left.clone() + right.clone();
                let product = sum * right.clone();
                let difference = product - left.clone();
                let quotient = difference / right.clone();
                Ok(quotient % left)
            },
            (left_type.clone(), right_type.clone()),
        )
        .unwrap();

        assert_eq!(output_type.bounds(), DimensionBounds::new(0, Some(8)).unwrap());
        assert_eq!(
            program
                .interpret((DimensionValue::new(left_type, 7).unwrap(), DimensionValue::new(right_type, 3).unwrap(),))
                .unwrap()
                .extent(),
            0,
        );
        assert!(matches!(program.instructions()[0].operation(), DimensionOperation::Add(_)));
        assert!(matches!(program.instructions()[1].operation(), DimensionOperation::Mul(_)));
        assert!(matches!(program.instructions()[2].operation(), DimensionOperation::Sub(_)));
        assert!(matches!(program.instructions()[3].operation(), DimensionOperation::Div(_)));
        assert!(matches!(program.instructions()[4].operation(), DimensionOperation::Rem(_)));
    }

    #[test]
    fn test_dimension_tracer_operator_propagates_construction_error() {
        let left_type = DimensionType::new("left", DimensionBounds::new(0, Some(2)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(3, Some(5)).unwrap());
        let result = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(left, right)| Ok(left - right),
            (left_type, right_type),
        );

        let Err(error) = result else {
            panic!("expected impossible subtraction bounds to fail tracing");
        };
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "left >= right is impossible from declared bounds".to_string(),
            }),
        );
    }
}
