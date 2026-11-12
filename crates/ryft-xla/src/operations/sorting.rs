use ryft_core::{ArrayType, DataType, ProgramError, SORT_OPERATION_NAME, SortDirection, SortOperation, SortOrdering};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::types::{lower_f64_constant_splat, lower_tensor_type};
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

/// Lowers one traced sort to a `stablehlo.sort` with a synthesized comparator region: two scalar block arguments
/// per operand, comparing the first `key_count` operand pairs lexicographically with the sort's direction (`LT`
/// for ascending and `GT` for descending) so that non-key operands ride along as passengers. Each key contributes one
/// comparison component, except for complex keys, which contribute their real part followed by their imaginary part.
/// For components `0..N`, the synthesized result is `cmp_0 OR (eq_0 AND (cmp_1 OR (eq_1 AND … cmp_{N-1})))`, built
/// right to left, where `cmp_i` is the direction comparison of component pair `i` and `eq_i` its equality comparison.
/// Each component derives its own comparison type from its data type: floating-point components compare with
/// `TOTALORDER` semantics (for both the direction and the equality comparison, so NaN ties fall through
/// deterministically), Boolean and unsigned-integer components compare `UNSIGNED`, and signed-integer components
/// compare `SIGNED`. Under [`SortOrdering::Canonical`], every floating-point component is first canonicalized exactly
/// like JAX's `lax.sort` comparator does it (i.e., `select(x == 0, 0, x)` followed by `select(x != x, NaN, ·)`), so
/// that signed zeros and all NaNs compare equal, with each step omitted for data types that cannot need it (i.e., that
/// have no negative zero or a single NaN encoding). The emitted sort is always stable, which is what routes ranking
/// ties to the lowest index.
pub(crate) fn lower_sort_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &SortOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    if output_types.is_empty() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!("`{SORT_OPERATION_NAME}` needs at least one input"),
        }
        .into());
    }
    let mut comparator_arguments = Vec::with_capacity(2 * output_types.len());
    for output_type in output_types {
        let scalar_type = lower_tensor_type(&ArrayType::scalar(output_type.data_type()), context, location)?;
        comparator_arguments.push((scalar_type, location));
        comparator_arguments.push((scalar_type, location));
    }
    let comparator_block = context.block(comparator_arguments.as_slice());
    let mut comparator = context.region();
    let mut comparator_block = comparator.append_block(comparator_block)?;
    let comparison_direction = match operation.direction() {
        SortDirection::Ascending => stable_hlo::ComparisonDirection::LessThan,
        SortDirection::Descending => stable_hlo::ComparisonDirection::GreaterThan,
    };

    // Each key contributes its comparison components in lexicographic order as left and right scalar values together
    // with their comparison type.
    let mut components = Vec::with_capacity(operation.key_count().get());
    for key_index in 0..operation.key_count().get() {
        let left_key = comparator_block.argument(2 * key_index)?.as_ref();
        let right_key = comparator_block.argument(2 * key_index + 1)?.as_ref();
        let data_type = output_types[key_index].data_type();
        let comparison_type = match data_type {
            DataType::Boolean
            | DataType::U1
            | DataType::U2
            | DataType::U4
            | DataType::U8
            | DataType::U16
            | DataType::U32
            | DataType::U64 => stable_hlo::ComparisonType::Unsigned,
            DataType::I1
            | DataType::I2
            | DataType::I4
            | DataType::I8
            | DataType::I16
            | DataType::I32
            | DataType::I64 => stable_hlo::ComparisonType::Signed,
            _ => stable_hlo::ComparisonType::TotalOrder,
        };
        if !data_type.is_complex() {
            if data_type.is_floating_point() && operation.ordering() == SortOrdering::Canonical {
                components.push((
                    lower_canonical_sort_key(left_key, data_type, &mut comparator_block, context, location)?,
                    lower_canonical_sort_key(right_key, data_type, &mut comparator_block, context, location)?,
                    comparison_type,
                ));
            } else {
                components.push((left_key, right_key, comparison_type));
            }
            continue;
        }
        // Type inference rejects complex keys under the total ordering, so complex keys are canonical here.
        let part_data_type = if data_type == DataType::C64 { DataType::F32 } else { DataType::F64 };
        let left_real = comparator_block.append_operation(stable_hlo::real(left_key, location)?)?;
        let right_real = comparator_block.append_operation(stable_hlo::real(right_key, location)?)?;
        let left_imaginary = comparator_block.append_operation(stable_hlo::imag(left_key, location)?)?;
        let right_imaginary = comparator_block.append_operation(stable_hlo::imag(right_key, location)?)?;
        for (left_part, right_part) in [
            (left_real.result(0).unwrap().as_ref(), right_real.result(0).unwrap().as_ref()),
            (left_imaginary.result(0).unwrap().as_ref(), right_imaginary.result(0).unwrap().as_ref()),
        ] {
            components.push((
                lower_canonical_sort_key(left_part, part_data_type, &mut comparator_block, context, location)?,
                lower_canonical_sort_key(right_part, part_data_type, &mut comparator_block, context, location)?,
                comparison_type,
            ));
        }
    }

    // The lexicographic chain builds right to left: the innermost term is the last component's direction comparison,
    // and every earlier component wraps the accumulated tail as `cmp_i OR (eq_i AND tail)`.
    let mut compared = None;
    for (left, right, comparison_type) in components.into_iter().rev() {
        let directed = comparator_block.append_operation(stable_hlo::compare(
            left,
            right,
            comparison_direction,
            comparison_type,
            location,
        )?)?;
        let directed = directed.result(0).expect("stablehlo.compare should return one result").as_ref();
        compared = Some(match compared {
            None => directed,
            Some(tail) => {
                let equal = comparator_block.append_operation(stable_hlo::compare(
                    left,
                    right,
                    stable_hlo::ComparisonDirection::Equal,
                    comparison_type,
                    location,
                )?)?;
                let equal = equal.result(0).expect("stablehlo.compare should return one result").as_ref();
                let tied = comparator_block.append_operation(stable_hlo::and(equal, tail, location)?)?;
                let tied = tied.result(0).expect("stablehlo.and should return one result").as_ref();
                let chained = comparator_block.append_operation(stable_hlo::or(directed, tied, location)?)?;
                chained.result(0).expect("stablehlo.or should return one result").as_ref()
            }
        });
    }
    // A `SortOperation` always carries at least one key, so the chain is never empty.
    let compared = compared.unwrap();
    comparator_block.append_operation(stable_hlo::r#return(&[compared], location)?)?;
    let sorted =
        block.append_operation(stable_hlo::sort(input_values, operation.axis(), true, comparator, location)?)?;
    Ok((0..output_types.len())
        .map(|index| sorted.result(index).expect("stablehlo.sort should return one result per operand").as_ref())
        .collect())
}

/// Canonicalizes one scalar floating-point sort key `value` of `data_type` inside a sort comparator for
/// [`SortOrdering::Canonical`]: negative zero becomes positive zero and every NaN becomes the same positive quiet NaN,
/// so that the subsequent `TOTALORDER` comparisons treat signed zeros as equal and all NaNs as equal and greater than
/// `+∞`. A step is omitted when `data_type` cannot need it: the `fnuz` formats and `f8e8m0fnu` have no negative zero,
/// while the `f4`/`f6` formats have no NaN and the `fnuz` formats and `f8e8m0fnu` have a single NaN encoding.
fn lower_canonical_sort_key<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    data_type: DataType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let (has_negative_zero, has_multiple_nan_encodings) = match data_type {
        DataType::F4E2M1FN | DataType::F6E2M3FN | DataType::F6E3M2FN => (true, false),
        DataType::F8E4M3FNUZ | DataType::F8E4M3B11FNUZ | DataType::F8E5M2FNUZ | DataType::F8E8M0FNU => (false, false),
        _ => (true, true),
    };
    let scalar_type = ArrayType::scalar(data_type);
    let scalar_tensor_type = lower_tensor_type(&scalar_type, context, location)?;
    let mut canonical = value;
    if has_negative_zero {
        let zero = lower_f64_constant_splat(0.0, &scalar_type, scalar_tensor_type, block, context, location)?;
        let is_zero = block.append_operation(stable_hlo::compare(
            value,
            zero,
            stable_hlo::ComparisonDirection::Equal,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?;
        let selected = block.append_operation(stable_hlo::select(
            is_zero.result(0).unwrap().as_ref(),
            zero,
            canonical,
            location,
        )?)?;
        canonical = selected.result(0).unwrap().as_ref();
    }
    if has_multiple_nan_encodings {
        let nan = lower_f64_constant_splat(f64::NAN, &scalar_type, scalar_tensor_type, block, context, location)?;
        let is_nan = block.append_operation(stable_hlo::compare(
            value,
            value,
            stable_hlo::ComparisonDirection::NotEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?;
        let selected = block.append_operation(stable_hlo::select(
            is_nan.result(0).unwrap().as_ref(),
            nan,
            canonical,
            location,
        )?)?;
        canonical = selected.result(0).unwrap().as_ref();
    }
    Ok(canonical)
}

impl LowerableOperation for SortOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        lower_sort_to_mlir(self, input_values, output_types, &mut lowerer.block, lowerer.context, lowerer.location)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;
    use ryft_core::{Array as CoreArray, Dimension as CoreDimension, Placeholder, Shape, Sort, TopK};

    use crate::domains::tests::domain_mesh;
    use crate::experimental::lowering::tests::{ProgramBuilder, unproject_plain_program};
    use crate::experimental::ops::ArrayConstant;
    use crate::tests::{execution_client, replicated_array, values_from_bytes, values_to_bytes};
    use crate::{Session, to_mlir_module_for_program};

    use super::*;

    #[test]
    fn test_to_mlir_module_for_program_lowers_sort_with_synthesized_comparator() {
        // A two-operand descending total-order sort (i.e., the top-k idiom) lowers to a stable `stablehlo.sort`
        // whose synthesized comparator compares the key pair with `GT` and `TOTALORDER` semantics, and whose index
        // passenger rides along unexamined.
        let key_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let index_type = ArrayType::new(DataType::I32, Shape::new(vec![CoreDimension::Static(4)]));
        let mut builder = ProgramBuilder::new();
        let keys = builder.add_input(key_type.clone());
        let indices = builder.add_input(index_type.clone());
        let outputs = builder
            .add_instruction(
                SortOperation::new(0, SortDirection::Descending).with_ordering(SortOrdering::Total),
                Vec::new(),
                vec![keys, indices],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let program = unproject_plain_program(program);
        let input_types = vec![key_type.clone(), index_type.clone()];
        let output_types = vec![key_type, index_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
                    %0:2 = "stablehlo.sort"(%arg0, %arg1) <{dimension = 0 : i64, is_stable = true}> ({
                    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
                      %1 = stablehlo.compare GT, %arg2, %arg3, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      stablehlo.return %1 : tensor<i1>
                    }) : (tensor<4xf32>, tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>)
                    return %0#0, %0#1 : tensor<4xf32>, tensor<4xi32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_sort_to_mlir_matches_reference_backend() {
        // The canonical ordering ties signed zeros and all NaNs, while the total ordering separates them, so both the
        // sorted keys (compared bit for bit) and the payloads expose any disagreement in tie handling.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let key_values = [3f32, f32::NAN, 1.0, 3.0, -f32::NAN, -0.0, 0.0, 2.0];
        let payload_values = [10f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let keys = replicated_array(&domain, &mesh, DataType::F32, &[8], &values_to_bytes(&key_values));
        let payloads = replicated_array(&domain, &mesh, DataType::F32, &[8], &values_to_bytes(&payload_values));
        let reference_keys = CoreArray::vector(key_values.to_vec()).unwrap();
        let reference_payloads = CoreArray::vector(payload_values.to_vec()).unwrap();
        for ordering in [SortOrdering::Canonical, SortOrdering::Total] {
            for direction in [SortDirection::Ascending, SortDirection::Descending] {
                let sorted =
                    Sort::sort_with_ordering(&[keys.clone(), payloads.clone()], 0, 1, direction, ordering).unwrap();
                let reference_sorted = Sort::sort_with_ordering(
                    &[reference_keys.clone(), reference_payloads.clone()],
                    0,
                    1,
                    direction,
                    ordering,
                )
                .unwrap();
                assert_eq!(
                    sorted.iter().map(|output| output.copy_to_host().unwrap()).collect::<Vec<_>>(),
                    reference_sorted.iter().map(CoreArray::logical_bytes).collect::<Vec<_>>(),
                    "{ordering} {direction}",
                );
            }
        }
    }

    #[test]
    fn test_lower_sort_to_mlir_top_k() {
        // Ties select the lowest index, so both threes appear before the two and index 0 precedes index 2.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let input =
            replicated_array(&domain, &mesh, DataType::F32, &[6], &values_to_bytes(&[3f32, 1.0, 3.0, -0.0, 0.0, 2.0]));
        let (values, indices) = input.top_k(3, 0).unwrap();
        assert_eq!(values_from_bytes::<f32>(&values.copy_to_host().unwrap()), vec![3.0, 3.0, 2.0]);
        assert_eq!(values_from_bytes::<i32>(&indices.copy_to_host().unwrap()), vec![0, 2, 5]);
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_multi_key_sort_with_lexicographic_comparator() {
        // A two-key ascending canonical sort lowers to a stable `stablehlo.sort` whose synthesized comparator chains
        // the key comparisons lexicographically as `cmp_0 OR (eq_0 AND cmp_1)`, with each key's comparison type
        // derived from that key's data type (`TOTALORDER` for the `f32` primary key, including its equality comparison
        // so NaN ties fall through deterministically, and `SIGNED` for the `i32` secondary key), while the passenger
        // rides along unexamined. The `f32` key is first canonicalized like JAX's `lax.sort` comparator does it, so
        // that signed zeros and all NaNs compare equal.
        let primary_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let secondary_type = ArrayType::new(DataType::I32, Shape::new(vec![CoreDimension::Static(4)]));
        let passenger_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let mut builder = ProgramBuilder::new();
        let primary = builder.add_input(primary_type.clone());
        let secondary = builder.add_input(secondary_type.clone());
        let passenger = builder.add_input(passenger_type.clone());
        let operation = SortOperation::new(0, SortDirection::Ascending).with_key_count(NonZeroUsize::new(2).unwrap());
        let outputs = builder
            .add_instruction(operation, Vec::new(), vec![primary, secondary, passenger], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();
        let program = unproject_plain_program(program);
        let input_types = vec![primary_type.clone(), secondary_type.clone(), passenger_type.clone()];
        let output_types = vec![primary_type, secondary_type, passenger_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xi32>, tensor<4xf32>) {
                    %0:3 = "stablehlo.sort"(%arg0, %arg1, %arg2) <{dimension = 0 : i64, is_stable = true}> ({
                    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<i32>, %arg6: tensor<i32>, %arg7: tensor<f32>, %arg8: tensor<f32>):
                      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %1 = stablehlo.compare EQ, %arg3, %cst, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %2 = stablehlo.select %1, %cst, %arg3 : tensor<i1>, tensor<f32>
                      %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
                      %3 = stablehlo.compare NE, %arg3, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %4 = stablehlo.select %3, %cst_0, %2 : tensor<i1>, tensor<f32>
                      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %5 = stablehlo.compare EQ, %arg4, %cst_1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %6 = stablehlo.select %5, %cst_1, %arg4 : tensor<i1>, tensor<f32>
                      %cst_2 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
                      %7 = stablehlo.compare NE, %arg4, %arg4, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %8 = stablehlo.select %7, %cst_2, %6 : tensor<i1>, tensor<f32>
                      %9 = stablehlo.compare LT, %arg5, %arg6, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                      %10 = stablehlo.compare LT, %4, %8, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %11 = stablehlo.compare EQ, %4, %8, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %12 = stablehlo.and %11, %9 : tensor<i1>
                      %13 = stablehlo.or %10, %12 : tensor<i1>
                      stablehlo.return %13 : tensor<i1>
                    }) : (tensor<4xf32>, tensor<4xi32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xi32>, tensor<4xf32>)
                    return %0#0, %0#1, %0#2 : tensor<4xf32>, tensor<4xi32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_sort_to_mlir_multiple_keys() {
        // Duplicates in the `i32` primary key fall through to the `f32` secondary key, full ties keep their original
        // order because the sort is stable, and the passenger co-permutes.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let primary = replicated_array(&domain, &mesh, DataType::I32, &[6], &values_to_bytes(&[2i32, 1, 2, 1, 2, 1]));
        let secondary = replicated_array(
            &domain,
            &mesh,
            DataType::F32,
            &[6],
            &values_to_bytes(&[0.5f32, 3.0, -1.0, 1.0, 0.5, 2.0]),
        );
        let passenger = replicated_array(
            &domain,
            &mesh,
            DataType::F32,
            &[6],
            &values_to_bytes(&[10f32, 20.0, 30.0, 40.0, 50.0, 60.0]),
        );
        let cases = [
            // Ascending: primary 1s precede 2s, ties resolve by the secondary key, and the full `(2, 0.5)` tie keeps
            // element 0 before element 4, as the passenger shows.
            (
                SortDirection::Ascending,
                vec![1, 1, 1, 2, 2, 2],
                vec![1f32, 2.0, 3.0, -1.0, 0.5, 0.5],
                vec![40f32, 60.0, 20.0, 30.0, 10.0, 50.0],
            ),
            // Descending reverses both key comparisons while keeping the full tie in its original order.
            (
                SortDirection::Descending,
                vec![2, 2, 2, 1, 1, 1],
                vec![0.5f32, 0.5, -1.0, 3.0, 2.0, 1.0],
                vec![10f32, 50.0, 30.0, 20.0, 60.0, 40.0],
            ),
        ];
        for (direction, expected_primary, expected_secondary, expected_passenger) in cases {
            let sorted =
                Sort::sort_with_key_count(&[primary.clone(), secondary.clone(), passenger.clone()], 0, 2, direction)
                    .unwrap();
            assert_eq!(sorted.len(), 3);
            assert_eq!(values_from_bytes::<i32>(&sorted[0].copy_to_host().unwrap()), expected_primary);
            assert_eq!(values_from_bytes::<f32>(&sorted[1].copy_to_host().unwrap()), expected_secondary);
            assert_eq!(values_from_bytes::<f32>(&sorted[2].copy_to_host().unwrap()), expected_passenger);
        }
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_complex_sort_with_canonical_comparator() {
        // A complex key contributes its real part and then its imaginary part as two lexicographic comparison
        // components, each canonicalized like a real floating-point key.
        let key_type = ArrayType::new(DataType::C64, Shape::new(vec![CoreDimension::Static(4)]));
        let mut builder = ProgramBuilder::new();
        let keys = builder.add_input(key_type.clone());
        let outputs = builder
            .add_instruction(SortOperation::new(0, SortDirection::Ascending), Vec::new(), vec![keys], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        let types = vec![key_type];
        let module = to_mlir_module_for_program(&program, &[], &types, &types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
                    %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
                    ^bb0(%arg1: tensor<complex<f32>>, %arg2: tensor<complex<f32>>):
                      %1 = stablehlo.real %arg1 : (tensor<complex<f32>>) -> tensor<f32>
                      %2 = stablehlo.real %arg2 : (tensor<complex<f32>>) -> tensor<f32>
                      %3 = stablehlo.imag %arg1 : (tensor<complex<f32>>) -> tensor<f32>
                      %4 = stablehlo.imag %arg2 : (tensor<complex<f32>>) -> tensor<f32>
                      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %5 = stablehlo.compare EQ, %1, %cst, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %6 = stablehlo.select %5, %cst, %1 : tensor<i1>, tensor<f32>
                      %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
                      %7 = stablehlo.compare NE, %1, %1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %8 = stablehlo.select %7, %cst_0, %6 : tensor<i1>, tensor<f32>
                      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %9 = stablehlo.compare EQ, %2, %cst_1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %10 = stablehlo.select %9, %cst_1, %2 : tensor<i1>, tensor<f32>
                      %cst_2 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
                      %11 = stablehlo.compare NE, %2, %2, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %12 = stablehlo.select %11, %cst_2, %10 : tensor<i1>, tensor<f32>
                      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %13 = stablehlo.compare EQ, %3, %cst_3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %14 = stablehlo.select %13, %cst_3, %3 : tensor<i1>, tensor<f32>
                      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
                      %15 = stablehlo.compare NE, %3, %3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %16 = stablehlo.select %15, %cst_4, %14 : tensor<i1>, tensor<f32>
                      %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %17 = stablehlo.compare EQ, %4, %cst_5, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %18 = stablehlo.select %17, %cst_5, %4 : tensor<i1>, tensor<f32>
                      %cst_6 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
                      %19 = stablehlo.compare NE, %4, %4, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %20 = stablehlo.select %19, %cst_6, %18 : tensor<i1>, tensor<f32>
                      %21 = stablehlo.compare LT, %16, %20, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %22 = stablehlo.compare LT, %8, %12, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %23 = stablehlo.compare EQ, %8, %12, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %24 = stablehlo.and %23, %21 : tensor<i1>
                      %25 = stablehlo.or %22, %24 : tensor<i1>
                      stablehlo.return %25 : tensor<i1>
                    }) : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
                    return %0 : tensor<4xcomplex<f32>>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_sort_to_mlir_complex_keys() {
        // Canonical complex keys order lexicographically by their canonicalized real and imaginary parts, so that
        // signed-zero real parts tie and NaN parts order last, as in JAX's `lax.sort`.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let keys = replicated_array(
            &domain,
            &mesh,
            DataType::C64,
            &[6],
            &values_to_bytes(&[
                Complex::new(1f32, f32::NAN),
                Complex::new(f32::NAN, 0.0),
                Complex::new(1.0, 2.0),
                Complex::new(1.0, 1.0),
                Complex::new(-0.0, 1.0),
                Complex::new(0.0, 1.0),
            ]),
        );
        let positions =
            replicated_array(&domain, &mesh, DataType::F32, &[6], &values_to_bytes(&[0f32, 1.0, 2.0, 3.0, 4.0, 5.0]));
        let sorted = Sort::sort(&[keys, positions], 0, SortDirection::Ascending).unwrap();
        assert_eq!(sorted.len(), 2);
        assert_eq!(values_from_bytes::<f32>(&sorted[1].copy_to_host().unwrap()), vec![4.0, 5.0, 3.0, 2.0, 0.0, 1.0]);
    }
}
