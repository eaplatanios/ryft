use ryft_core::macros::check_count;
use ryft_core::{ArrayType, CompareOperation, DataType, ProgramError};
use ryft_mlir::ValueRef;

use crate::lowering::types::{ComparisonDirectionLoweringExtension, lower_signed_one_bit_to_mlir};
use crate::operations::normalize_binary_elementwise_inputs;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, Type};

impl LowerableOperation for CompareOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let mut operand_type = comparison_operand_type(input_types, output_types)?;
        // StableHLO treats `i1` as a predicate, even when its logical data type is signed. Recover each signed input
        // before promotion, and compare two I1 inputs in the same signed byte carrier.
        let mut inputs = [input_values[0], input_values[1]];
        for (input, input_type) in inputs.iter_mut().zip(input_types) {
            if input_type.data_type() == DataType::I1 {
                *input =
                    lower_signed_one_bit_to_mlir(*input, &mut lowerer.block(), lowerer.context(), lowerer.location())?;
            }
        }
        if operand_type.data_type() == DataType::I1 {
            operand_type = operand_type.with_data_type(DataType::I8);
        }
        let [left, right] = normalize_binary_elementwise_inputs(
            &inputs,
            &[operand_type],
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        let value = self.direction().lower_comparison(left, right, &mut lowerer.block(), lowerer.location())?;
        Ok(vec![value])
    }
}

impl LowerableOperation for CompareOperation<Type> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let location = lowerer.location();
        let block = &mut lowerer.block();
        let [left, right] = input_values else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into());
        };
        // Dimension atoms already lower to scalar `i64` SSA values, so the ordinary comparison lowering can
        // consume them directly without a data gateway or host-side shape reconstruction.
        Ok(vec![self.direction().lower_comparison(*left, *right, block, location)?])
    }
}

/// Returns the promoted numeric operand descriptor for one comparison output descriptor.
fn comparison_operand_type(input_types: &[ArrayType], output_types: &[ArrayType]) -> Result<ArrayType, Error> {
    check_count!("input", input_types, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let data_type = DataType::promoted(&[input_types[0].data_type(), input_types[1].data_type()])?;
    Ok(output_types[0].clone().with_data_type(data_type))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{Compare, ComparisonDirection, DataTypeError, Error as CoreError, Typed};

    use crate::sessions::tests::{domain_mesh, f32_vector};
    use crate::tests::{cpu_client, execution_client, replicated_array, values_from_bytes, values_to_bytes};
    use crate::{Options, Session};

    use super::*;

    #[test]
    fn test_compare_operation_lower() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let left = replicated_array(&session, &mesh, DataType::F32, &[4], &values_to_bytes(&[1f32, 5.0, 3.0, 8.0]));
        let right = replicated_array(&session, &mesh, DataType::F32, &[4], &values_to_bytes(&[4f32, 2.0, 3.0, 9.0]));
        assert_eq!(
            values_from_bytes::<bool>(
                &left.compare(&right, ComparisonDirection::LessThan).unwrap().copy_to_host_bytes().unwrap(),
            ),
            [true, false, false, true],
        );
    }

    #[test]
    fn test_compare_operation_lower_one_bit_integers() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);

        // Every pair of bits is compared in every direction. I1 interprets the set bit as -1, while U1 and Boolean
        // interpret it as +1.
        for data_type in [DataType::Boolean, DataType::I1, DataType::U1] {
            let left = replicated_array(&session, &mesh, data_type, &[4], &[0, 0, 1, 1]);
            let right = replicated_array(&session, &mesh, data_type, &[4], &[0, 1, 0, 1]);
            let signed = data_type == DataType::I1;
            for (direction, expected) in [
                (ComparisonDirection::Equal, [true, false, false, true]),
                (ComparisonDirection::NotEqual, [false, true, true, false]),
                (ComparisonDirection::LessThan, [false, !signed, signed, false]),
                (ComparisonDirection::LessThanOrEqual, [true, !signed, signed, true]),
                (ComparisonDirection::GreaterThan, [false, signed, !signed, false]),
                (ComparisonDirection::GreaterThanOrEqual, [true, signed, !signed, true]),
            ] {
                assert_eq!(
                    left.compare(&right, direction).unwrap().copy_to_host().unwrap().elements::<bool>().unwrap(),
                    expected,
                    "{data_type} {direction:?}",
                );
            }
        }

        // Implicit promotion must preserve the signed set bit before comparing against wider numeric inputs.
        let signed = replicated_array(&session, &mesh, DataType::I1, &[2], &[1, 0]);
        for (data_type, bytes) in [
            (DataType::Boolean, vec![0, 0]),
            (DataType::U1, vec![0, 0]),
            (DataType::I8, vec![0, 0]),
            (DataType::I32, values_to_bytes(&[0i32, 0])),
            (DataType::F32, values_to_bytes(&[0f32, 0.0])),
        ] {
            let zero = replicated_array(&session, &mesh, data_type, &[2], &bytes);
            for (direction, expected) in [
                (ComparisonDirection::Equal, [false, true]),
                (ComparisonDirection::NotEqual, [true, false]),
                (ComparisonDirection::LessThan, [true, false]),
                (ComparisonDirection::LessThanOrEqual, [true, true]),
                (ComparisonDirection::GreaterThan, [false, false]),
                (ComparisonDirection::GreaterThanOrEqual, [false, true]),
            ] {
                assert_eq!(
                    signed.compare(&zero, direction).unwrap().copy_to_host().unwrap().elements::<bool>().unwrap(),
                    expected,
                    "{data_type} {direction:?}",
                );
            }
        }
    }

    #[test]
    fn test_compare_operation_lower_executes_per_element() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let values = f32_vector(&session, &mesh, &[1., -2., 3., -4.]);
        let limits = f32_vector(&session, &mesh, &[0., 0., 0., 0.]);
        let staged = crate::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| vec![inputs[0].greater_than(&inputs[1]).unwrap()],
            vec![values.r#type().into_owned(), limits.r#type().into_owned()],
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let lowered = session.lower_program(staged.source_program().program(), 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        assert!(!compiled.requires_assertion_handler());
        let outputs = session.execute_program(&compiled, vec![values, limits]).unwrap();
        assert_eq!(
            outputs[0]
                .addressable_shards()
                .next()
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap(),
            vec![1, 0, 1, 0]
        );
    }

    #[test]
    fn test_comparison_operand_type() {
        // The inputs compare at their promoted data type, with the shape of the output.
        assert_eq!(
            comparison_operand_type(
                &[ArrayType::new_static(DataType::I8, [2]), ArrayType::new_static(DataType::I32, [2])],
                &[ArrayType::new_static(DataType::Boolean, [2])],
            )
            .unwrap(),
            ArrayType::new_static(DataType::I32, [2]),
        );
    }

    #[test]
    fn test_comparison_operand_type_preserves_promotion_error() {
        assert!(matches!(
            comparison_operand_type(
                &[ArrayType::scalar(DataType::F6E2M3FN), ArrayType::scalar(DataType::F32)],
                &[ArrayType::scalar(DataType::Boolean)],
            ),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::DataType(DataTypeError::InvalidPromotion { message, .. })
                    if *message == "cannot promote types `f6e2m3fn` and `f32` to a common type",
            ),
        ));
    }
}
