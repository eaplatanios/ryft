use ryft_core::macros::check_count;
use ryft_core::{ArrayIrType, ArrayType, CompareOperation, ComparisonDirection, DataType, ProgramError};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, FloatTypeRef, IntegerTypeRef, Location, LocationRef, Operation as MlirOperation, TensorTypeRef,
    Type, TypeRef, Value as MlirValue, ValueRef,
};

use crate::lowering::types::lower_signed_one_bit_to_mlir;
use crate::operations::normalize_binary_elementwise_operands;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, ToMlir};

impl ToMlir for ComparisonDirection {
    type Output<'c, 't: 'c> = stable_hlo::ComparisonDirection;

    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(&self, location: L) -> Result<Self::Output<'c, 't>, ryft_mlir::Error> {
        let _ = location;
        Ok(match self {
            ComparisonDirection::Equal => stable_hlo::ComparisonDirection::Equal,
            ComparisonDirection::NotEqual => stable_hlo::ComparisonDirection::NotEqual,
            ComparisonDirection::LessThan => stable_hlo::ComparisonDirection::LessThan,
            ComparisonDirection::LessThanOrEqual => stable_hlo::ComparisonDirection::LessThanOrEqual,
            ComparisonDirection::GreaterThan => stable_hlo::ComparisonDirection::GreaterThan,
            ComparisonDirection::GreaterThanOrEqual => stable_hlo::ComparisonDirection::GreaterThanOrEqual,
        })
    }
}

/// Lowers one comparison of two values in the provided `direction` to a single `stablehlo.compare`. This serves
/// [`CompareOperation`] lowerings, scalar dimension comparisons, and the comparisons inside of combiner and comparator
/// bodies. Both operands must already share one tensor type, because this function neither promotes nor broadcasts
/// them, and the result has their shape with a Boolean element type. The comparison type (i.e., `FLOAT`, `SIGNED`, or
/// `UNSIGNED`) follows the element type of the left-hand side input, as [`comparison_type_for_mlir_type`] describes.
pub(crate) fn lower_compare_to_mlir<'b, 'c: 'b, 't: 'c>(
    direction: ComparisonDirection,
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let direction = direction.to_mlir(location)?;
    let left_type = left.r#type()?;
    let comparison_type = comparison_type_for_mlir_type(left_type)?;
    let result = block.append_operation(stable_hlo::compare(left, right, direction, comparison_type, location)?)?;
    Ok(result.result(0).unwrap().as_ref())
}

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
        let [left, right] = normalize_binary_elementwise_operands(
            &inputs,
            &[operand_type],
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        let value = lower_compare_to_mlir(self.direction(), left, right, &mut lowerer.block(), lowerer.location())?;
        Ok(vec![value])
    }
}

impl LowerableOperation for CompareOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let location = lowerer.location();
        let block = &mut lowerer.block();
        let [left, right] = input_values else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into());
        };
        // Dimension atoms already lower to scalar `i64` SSA values, so the ordinary comparison lowering can
        // consume them directly without a data gateway or host-side shape reconstruction.
        Ok(vec![lower_compare_to_mlir(self.direction(), *left, *right, block, location)?])
    }
}

/// Returns the promoted numeric operand descriptor for one comparison result descriptor.
fn comparison_operand_type(input_types: &[ArrayType], output_types: &[ArrayType]) -> Result<ArrayType, Error> {
    check_count!("input", input_types, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let data_type = DataType::promoted(&[input_types[0].data_type(), input_types[1].data_type()])?;
    Ok(output_types[0].clone().with_data_type(data_type))
}

/// Picks the right StableHLO comparison semantic based on the element type of an MLIR value.
///
/// Tensor values are unwrapped to their element type, while scalar types that are not tensors are inspected directly.
/// Floating-point types route to [`stable_hlo::ComparisonType::Float`], explicitly unsigned integers and one-bit
/// predicate carriers route to [`stable_hlo::ComparisonType::Unsigned`], and other integers route to
/// [`stable_hlo::ComparisonType::Signed`]. Every other element type (e.g., complex) routes to
/// [`stable_hlo::ComparisonType::Float`]. Logical signed one-bit inputs must be widened before calling this function,
/// because their MLIR carrier alone cannot distinguish them from Boolean inputs.
fn comparison_type_for_mlir_type<'c, 't>(r#type: TypeRef<'c, 't>) -> Result<stable_hlo::ComparisonType, Error> {
    let element_type = if let Some(tensor) = r#type.cast::<TensorTypeRef>() { tensor.element_type()? } else { r#type };
    if element_type.is::<FloatTypeRef>() {
        return Ok(stable_hlo::ComparisonType::Float);
    }
    if let Some(integer) = element_type.cast::<IntegerTypeRef>() {
        if integer.is_unsigned() || integer.bit_width() == 1 {
            return Ok(stable_hlo::ComparisonType::Unsigned);
        }
        return Ok(stable_hlo::ComparisonType::Signed);
    }
    // Other element types compare as floating-point values, which matches StableHLO's lenient handling of numeric
    // types that are neither integers nor floating-point types (e.g., complex types).
    Ok(stable_hlo::ComparisonType::Float)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{Compare, DataTypeError, Error as CoreError, Typed};
    use ryft_mlir::Context as MlirContext;

    use crate::domains::tests::{domain_mesh, f32_vector};
    use crate::tests::{cpu_client, execution_client, replicated_array, values_from_bytes, values_to_bytes};
    use crate::{Options, Session};

    use super::*;

    #[test]
    fn test_comparison_direction_to_mlir() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        for (direction, expected) in [
            (ComparisonDirection::Equal, stable_hlo::ComparisonDirection::Equal),
            (ComparisonDirection::NotEqual, stable_hlo::ComparisonDirection::NotEqual),
            (ComparisonDirection::LessThan, stable_hlo::ComparisonDirection::LessThan),
            (ComparisonDirection::LessThanOrEqual, stable_hlo::ComparisonDirection::LessThanOrEqual),
            (ComparisonDirection::GreaterThan, stable_hlo::ComparisonDirection::GreaterThan),
            (ComparisonDirection::GreaterThanOrEqual, stable_hlo::ComparisonDirection::GreaterThanOrEqual),
        ] {
            assert_eq!(direction.to_mlir(location), Ok(expected));
        }
    }

    #[test]
    fn test_compare_operation_lower() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let left = replicated_array(&domain, &mesh, DataType::F32, &[4], &values_to_bytes(&[1f32, 5.0, 3.0, 8.0]));
        let right = replicated_array(&domain, &mesh, DataType::F32, &[4], &values_to_bytes(&[4f32, 2.0, 3.0, 9.0]));
        assert_eq!(
            values_from_bytes::<bool>(
                &left.compare(&right, ComparisonDirection::LessThan).unwrap().copy_to_host().unwrap(),
            ),
            [true, false, false, true],
        );
    }

    #[test]
    fn test_compare_operation_lower_one_bit_integers() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Every pair of bits is compared in every direction. I1 interprets the set bit as -1, while U1 and Boolean
        // interpret it as +1.
        for data_type in [DataType::Boolean, DataType::I1, DataType::U1] {
            let left = replicated_array(&domain, &mesh, data_type, &[4], &[0, 0, 1, 1]);
            let right = replicated_array(&domain, &mesh, data_type, &[4], &[0, 1, 0, 1]);
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
                    values_from_bytes::<bool>(&left.compare(&right, direction).unwrap().copy_to_host().unwrap()),
                    expected,
                    "{data_type} {direction:?}",
                );
            }
        }

        // Implicit promotion must preserve the signed set bit before comparing against wider numeric inputs.
        let signed = replicated_array(&domain, &mesh, DataType::I1, &[2], &[1, 0]);
        for (data_type, bytes) in [
            (DataType::Boolean, vec![0, 0]),
            (DataType::U1, vec![0, 0]),
            (DataType::I8, vec![0, 0]),
            (DataType::I32, values_to_bytes(&[0i32, 0])),
            (DataType::F32, values_to_bytes(&[0f32, 0.0])),
        ] {
            let zero = replicated_array(&domain, &mesh, data_type, &[2], &bytes);
            for (direction, expected) in [
                (ComparisonDirection::Equal, [false, true]),
                (ComparisonDirection::NotEqual, [true, false]),
                (ComparisonDirection::LessThan, [true, false]),
                (ComparisonDirection::LessThanOrEqual, [true, true]),
                (ComparisonDirection::GreaterThan, [false, false]),
                (ComparisonDirection::GreaterThanOrEqual, [false, true]),
            ] {
                assert_eq!(
                    values_from_bytes::<bool>(&signed.compare(&zero, direction).unwrap().copy_to_host().unwrap()),
                    expected,
                    "{data_type} {direction:?}",
                );
            }
        }
    }

    #[test]
    fn test_compare_operation_lower_executes_per_element() {
        let client = cpu_client(1);
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let values = f32_vector(&domain, &mesh, &[1., -2., 3., -4.]);
        let limits = f32_vector(&domain, &mesh, &[0., 0., 0., 0.]);
        let staged = crate::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| vec![inputs[0].greater_than(&inputs[1]).unwrap()],
            vec![values.r#type().into_owned(), limits.r#type().into_owned()],
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let lowered = domain.lower_program(staged.source_program().program(), 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        assert!(!compiled.requires_assertion_handler());
        let outputs = domain.execute_program(&compiled, vec![values, limits]).unwrap();
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
        // The operands compare at their promoted data type, with the shape of the result.
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
