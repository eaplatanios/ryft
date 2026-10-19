use ryft_core::macros::check_count;
use ryft_core::{ArrayType, DataType, Dimension, SelectOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, Context as MlirContext, Location, Operation as MlirOperation, TensorTypeRef, Type, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::types::{
    lower_physical_bound_value, lower_restore_dynamic_dimensions, lower_signed_one_bit_to_mlir, lower_tensor_type,
    physical_bound_type,
};
use crate::operations::constants::lower_layout_constraint;
use crate::operations::normalize_elementwise_operand;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Normalizes a select condition and both branches to the exact tensor descriptors required by StableHLO.
pub(crate) fn normalize_select_operands<'b, 'c: 'b, 't: 'c, B, L>(
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<[ValueRef<'b, 'c, 't>; 3], LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    check_count!("input", input_values, 3, ProgramError);
    check_count!("input", input_types, 3, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let output_type = &output_types[0];
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    let multiple_dynamic_axes = output_type
        .shape()
        .dimensions()
        .iter()
        .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        .count()
        > 1;
    // Each dynamic output extent is inherited from a right-aligned input dimension. Cross-broadcasts such as
    // [rows, 1] and [1, columns] need different inputs for different axes, rather than one full-shape exemplar.
    let mut sources = vec![(input_values[0], 0); output_type.rank()];
    if multiple_dynamic_axes {
        for (output_axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
            if matches!(dimension, Dimension::Static(_)) {
                continue;
            }
            let source = input_types.iter().enumerate().find_map(|(index, input_type)| {
                let offset = output_type.rank().checked_sub(input_type.rank())?;
                let input_axis = output_axis.checked_sub(offset)?;
                (input_type.shape().dimensions().get(input_axis) == Some(dimension))
                    .then_some((input_values[index], input_axis))
            });
            sources[output_axis] = source.ok_or_else(|| LoweringError::UnsupportedOp {
                op: format!("cannot recover select output axis {output_axis} extent from its inputs"),
            })?;
        }
    }
    let mut inputs = [input_values[0], input_values[1], input_values[2]];
    for index in 0..3 {
        let target_type =
            if index == 0 { output_type.clone().with_data_type(DataType::Boolean) } else { output_type.clone() };
        // StableHLO stores I1 in a predicate carrier. Recover its signed value before promoting a branch,
        // while equal-I1 branches keep their original carrier and the Boolean condition stays a predicate.
        if index != 0 && input_types[index].data_type() == DataType::I1 && output_type.data_type() != DataType::I1 {
            inputs[index] = lower_signed_one_bit_to_mlir(inputs[index], block, context, location)?;
        }
        let input_tensor_type = inputs[index].r#type()?.cast::<TensorTypeRef>();
        if multiple_dynamic_axes
            && input_tensor_type.is_some_and(|r#type| r#type.dimensions().ne(output_tensor_type.dimensions()))
        {
            // Native broadcast supports at most one bounded dynamic result axis. Convert at the input shape,
            // expose its physical capacity, broadcast that capacity, then restore the composed logical extents.
            let converted_type = input_types[index].clone().with_data_type(target_type.data_type());
            let converted = normalize_elementwise_operand(inputs[index], &converted_type, block, context, location)?;
            let physical_input = lower_physical_bound_value(
                converted,
                &converted_type,
                0.0,
                &mut block.as_ref(),
                context,
                location.as_ref(),
            )?;
            let physical_type = physical_bound_type(&target_type)?;
            let first_axis =
                target_type.rank().checked_sub(converted_type.rank()).ok_or_else(|| LoweringError::UnsupportedOp {
                    op: format!(
                        "cannot broadcast rank-{} select input to rank-{} output",
                        converted_type.rank(),
                        target_type.rank(),
                    ),
                })?;
            let output_axes = (first_axis..target_type.rank()).collect::<Vec<_>>();
            let broadcast = block.append_operation(stable_hlo::broadcast(
                physical_input,
                lower_tensor_type(&physical_type, context, location)?,
                &output_axes,
                location,
            )?)?;
            let result = lower_layout_constraint(
                broadcast.result(0).unwrap().as_ref(),
                &physical_type,
                &mut block.as_ref(),
                location.as_ref(),
            )?;
            inputs[index] = lower_restore_dynamic_dimensions(
                result,
                &target_type,
                &sources,
                &mut block.as_ref(),
                context,
                location.as_ref(),
            )?;
        } else {
            inputs[index] = normalize_elementwise_operand(inputs[index], &target_type, block, context, location)?;
        }
    }
    Ok(inputs)
}

impl LowerableOperation for SelectOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [condition, on_true, on_false] = normalize_select_operands(
            input_values,
            input_types,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result =
            lowerer
                .block
                .append_operation(stable_hlo::select(condition, on_true, on_false, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.select should return one result").as_ref()])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{CompareOperation, ComparisonDirection, Placeholder, SelectOperation, Shape};

    use crate::experimental::lowering::tests::{
        MixedValue, XlaProgramBuilder, dynamic_dimension, execute_mixed_program, unproject_plain_program,
    };
    use crate::experimental::ops::XlaArrayConstant;
    use crate::tests::execution_client;

    use super::*;

    #[test]
    fn test_select_lowering_broadcasts_scalars_over_multiple_dynamic_axes() {
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![dynamic_dimension("rows", Some(5)), dynamic_dimension("columns", Some(5))]),
        );
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let on_true = builder.add_input(ArrayType::scalar(DataType::F64));
        let on_false = builder.add_input(ArrayType::scalar(DataType::F64));
        let condition = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![input, on_false],
                None,
            )
            .unwrap()[0];
        let selected = builder
            .add_instruction(SelectOperation::new(), Vec::new(), vec![condition, on_true, on_false], None)
            .unwrap()[0];
        let scalar_condition = builder.add_constant(XlaArrayConstant::Boolean(true));
        let identity = builder
            .add_instruction(SelectOperation::new(), Vec::new(), vec![scalar_condition, input, on_false], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![selected, identity],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let program = unproject_plain_program(program);
        let client = execution_client();
        let storage = vec![1., -8., 99., 99., -9., 4., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99.];
        let inputs = [
            MixedValue::Array(storage, vec![4, 4]),
            MixedValue::Array(vec![5.], Vec::new()),
            MixedValue::Array(vec![-7.], Vec::new()),
        ];
        // Scalar branches inherit both logical extents from a matrix condition, and a scalar condition inherits
        // them from a matrix branch. Capacity padding must not appear in either selected result.
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[2, 2]),
            Ok(vec![
                MixedValue::Array(vec![5., -7., -7., 5.], vec![2, 2]),
                MixedValue::Array(vec![1., -8., -9., 4.], vec![2, 2]),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
            ]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[0, 2]),
            Ok(vec![
                MixedValue::Array(Vec::new(), vec![0, 2]),
                MixedValue::Array(Vec::new(), vec![0, 2]),
                MixedValue::Dimension(0),
                MixedValue::Dimension(2),
                MixedValue::Dimension(0),
                MixedValue::Dimension(2),
            ]),
        );
    }

    #[test]
    fn test_select_lowering_cross_broadcasts_multiple_dynamic_axes() {
        let rows = dynamic_dimension("rows", Some(5));
        let columns = dynamic_dimension("columns", Some(5));
        let mut builder = XlaProgramBuilder::new();
        let row_values = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![rows, 1.into()])));
        let on_true = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![1.into(), columns])));
        let on_false = builder.add_input(ArrayType::scalar(DataType::F64));
        let condition = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![row_values, on_false],
                None,
            )
            .unwrap()[0];
        let selected = builder
            .add_instruction(SelectOperation::new(), Vec::new(), vec![condition, on_true, on_false], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![selected],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let program = unproject_plain_program(program);
        let client = execution_client();
        let inputs = [
            MixedValue::Array(vec![1., -8., 99., 99.], vec![4, 1]),
            MixedValue::Array(vec![10., 20., 30., 99.], vec![1, 4]),
            MixedValue::Array(vec![-7.], Vec::new()),
        ];
        // No input has the output shape: the row extent comes from the condition and the column extent from
        // the true branch. Both stretched inputs must broadcast their bounded storage before restoring sizes.
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[2, 3]),
            Ok(vec![
                MixedValue::Array(vec![10., 20., 30., -7., -7., -7.], vec![2, 3]),
                MixedValue::Dimension(2),
                MixedValue::Dimension(3),
            ]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[2, 0]),
            Ok(vec![MixedValue::Array(Vec::new(), vec![2, 0]), MixedValue::Dimension(2), MixedValue::Dimension(0),]),
        );
    }
}
