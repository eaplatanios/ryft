use ryft_core::macros::check_count;
use ryft_core::{ArrayType, ClampOperation, DataType, MaxOperation, MinOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, LocationRef, Operation as MlirOperation, TensorTypeRef, Type, Value as MlirValue, ValueRef,
};

use crate::operations::{normalize_binary_elementwise_operands, normalize_elementwise_operand};
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for MaxOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        if output_types[0].data_type().is_complex() {
            return Ok(vec![lower_extremum_to_mlir(
                true,
                output_types[0].data_type(),
                left,
                right,
                &mut lowerer.block,
                lowerer.location,
            )?]);
        }
        let result = lowerer.block.append_operation(stable_hlo::maximum(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.maximum should return one result").as_ref()])
    }
}

impl LowerableOperation for MinOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        if output_types[0].data_type().is_complex() {
            return Ok(vec![lower_extremum_to_mlir(
                false,
                output_types[0].data_type(),
                left,
                right,
                &mut lowerer.block,
                lowerer.location,
            )?]);
        }
        let result = lowerer.block.append_operation(stable_hlo::minimum(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.minimum should return one result").as_ref()])
    }
}

impl LowerableOperation for ClampOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 3, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let output_type = &output_types[0];
        let input = normalize_elementwise_operand(
            input_values[1],
            output_type,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let data_type = output_type.data_type();
        if data_type.is_complex() || data_type.is_floating_point() {
            // Floating-point and complex values compose `min(max(x, lower), upper)` explicitly. XLA's native `clamp`
            // does not propagate NaN operands on every backend (e.g., the CPU backend returns the lower bound for a
            // NaN input), and StableHLO leaves the ordering of complex `clamp` unspecified, whereas the extremum
            // lowerings below propagate NaNs and order complex values lexicographically like the reference backend.
            let [lower, upper] = [input_values[0], input_values[2]].map(|bound| {
                normalize_elementwise_operand(bound, output_type, &mut lowerer.block, lowerer.context, lowerer.location)
            });
            let (lower, upper) = (lower?, upper?);
            if data_type.is_floating_point() {
                let clamped_below =
                    lowerer.block.append_operation(stable_hlo::maximum(input, lower, lowerer.location)?)?;
                let clamped_below =
                    clamped_below.result(0).expect("stablehlo.maximum should return one result").as_ref();
                let result =
                    lowerer.block.append_operation(stable_hlo::minimum(clamped_below, upper, lowerer.location)?)?;
                return Ok(vec![result.result(0).expect("stablehlo.minimum should return one result").as_ref()]);
            }
            let clamped_below =
                lower_extremum_to_mlir(true, data_type, input, lower, &mut lowerer.block, lowerer.location)?;
            return Ok(vec![lower_extremum_to_mlir(
                false,
                data_type,
                clamped_below,
                upper,
                &mut lowerer.block,
                lowerer.location,
            )?]);
        }

        // Integer and Boolean values use one `stablehlo.clamp`, which accepts scalar bounds or bounds with the
        // operand's shape, so rank-zero bounds keep their rank and only convert their element type, while every other
        // bound is broadcast to the result type.
        let mut bounds = [input_values[0], input_values[2]];
        for bound in &mut bounds {
            let is_scalar = bound.r#type()?.cast::<TensorTypeRef>().is_some_and(|r#type| r#type.rank() == 0);
            let bound_type = if is_scalar { ArrayType::scalar(output_type.data_type()) } else { output_type.clone() };
            *bound = normalize_elementwise_operand(
                *bound,
                &bound_type,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
        }
        let result =
            lowerer.block.append_operation(stable_hlo::clamp(bounds[0], input, bounds[1], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.clamp should return one result").as_ref()])
    }
}

/// Lowers one JAX-compatible minimum or maximum. Integer and Boolean values use the corresponding StableHLO
/// operation. Floating-point values use explicit total-order comparisons and NaN propagation so every XLA backend
/// preserves IEEE signed-zero and NaN semantics. Complex values use JAX's lexicographic `(real, imaginary)`
/// compare/select sequence.
pub(crate) fn lower_extremum_to_mlir<'b, 'c: 'b, 't: 'c>(
    maximum: bool,
    element_type: DataType,
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    if !element_type.is_complex() && !element_type.is_floating_point() {
        let result = if maximum {
            block.append_operation(stable_hlo::maximum(left, right, location)?)?
        } else {
            block.append_operation(stable_hlo::minimum(left, right, location)?)?
        };
        return Ok(result.result(0).expect("stablehlo extremum should return one result").as_ref());
    }
    if element_type.is_floating_point() {
        let direction = if maximum {
            stable_hlo::ComparisonDirection::GreaterThan
        } else {
            stable_hlo::ComparisonDirection::LessThan
        };
        let ordered = block.append_operation(stable_hlo::compare(
            left,
            right,
            direction,
            stable_hlo::ComparisonType::TotalOrder,
            location,
        )?)?;
        let ordered = ordered.result(0).expect("stablehlo.compare should return one result").as_ref();
        let result = block.append_operation(stable_hlo::select(ordered, left, right, location)?)?;
        let result = result.result(0).expect("stablehlo.select should return one result").as_ref();

        // Some XLA backends implement native min/max using instructions that drop NaNs or do not distinguish signed
        // zeros. StableHLO and JAX require IEEE maximum/minimum, so make both behaviors explicit and portable.
        let left_is_nan = block.append_operation(stable_hlo::compare(
            left,
            left,
            stable_hlo::ComparisonDirection::NotEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?;
        let left_is_nan = left_is_nan.result(0).expect("stablehlo.compare should return one result").as_ref();
        let right_is_nan = block.append_operation(stable_hlo::compare(
            right,
            right,
            stable_hlo::ComparisonDirection::NotEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?;
        let right_is_nan = right_is_nan.result(0).expect("stablehlo.compare should return one result").as_ref();
        let right_or_result = block.append_operation(stable_hlo::select(right_is_nan, right, result, location)?)?;
        let right_or_result = right_or_result.result(0).expect("stablehlo.select should return one result").as_ref();
        let result = block.append_operation(stable_hlo::select(left_is_nan, left, right_or_result, location)?)?;
        return Ok(result.result(0).expect("stablehlo.select should return one result").as_ref());
    }

    let left_real = block.append_operation(stable_hlo::real(left, location)?)?;
    let left_real = left_real.result(0).expect("stablehlo.real should return one result").as_ref();
    let right_real = block.append_operation(stable_hlo::real(right, location)?)?;
    let right_real = right_real.result(0).expect("stablehlo.real should return one result").as_ref();
    let left_imaginary = block.append_operation(stable_hlo::imag(left, location)?)?;
    let left_imaginary = left_imaginary.result(0).expect("stablehlo.imag should return one result").as_ref();
    let right_imaginary = block.append_operation(stable_hlo::imag(right, location)?)?;
    let right_imaginary = right_imaginary.result(0).expect("stablehlo.imag should return one result").as_ref();
    let real_equal = block.append_operation(stable_hlo::compare(
        left_real,
        right_real,
        stable_hlo::ComparisonDirection::Equal,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let real_equal = real_equal.result(0).expect("stablehlo.compare should return one result").as_ref();
    let direction =
        if maximum { stable_hlo::ComparisonDirection::GreaterThan } else { stable_hlo::ComparisonDirection::LessThan };
    let real_ordered = block.append_operation(stable_hlo::compare(
        left_real,
        right_real,
        direction,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let real_ordered = real_ordered.result(0).expect("stablehlo.compare should return one result").as_ref();
    let imaginary_ordered = block.append_operation(stable_hlo::compare(
        left_imaginary,
        right_imaginary,
        direction,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let imaginary_ordered = imaginary_ordered.result(0).expect("stablehlo.compare should return one result").as_ref();
    let ordered = block.append_operation(stable_hlo::select(real_equal, imaginary_ordered, real_ordered, location)?)?;
    let ordered = ordered.result(0).expect("stablehlo.select should return one result").as_ref();
    let result = block.append_operation(stable_hlo::select(ordered, left, right, location)?)?;
    Ok(result.result(0).expect("stablehlo.select should return one result").as_ref())
}

#[cfg(test)]
mod tests {
    use num_complex::Complex;
    use pretty_assertions::assert_eq;
    use ryft_core::{Clamp, DeviceMesh, Max, Min, Placeholder, Shape};

    use crate::domains::tests::domain_mesh;
    use crate::experimental::lowering::tests::{MixedValue, dynamic_dimension, execute_mixed_program};
    use crate::experimental::ops::{Constant, ProgramBuilder as ArrayIrProgramBuilder};
    use crate::tests::{execution_client, replicated_array, values_to_bytes};
    use crate::{Array, Domain, Session};

    use super::*;

    /// Uploads complex operands whose lexicographic order is decided by the real components, by the imaginary
    /// components, by signed zeros that compare equal, and by NaN payloads in either component.
    fn complex_operands<'c>(
        domain: &Domain<'c>,
        mesh: &DeviceMesh,
    ) -> ([Complex<f32>; 5], [Complex<f32>; 5], Array<'c>, Array<'c>) {
        let left = [
            Complex::new(1f32, 100.0),
            Complex::new(1.0, 2.0),
            Complex::new(-0.0, 0.0),
            Complex::new(f32::from_bits(0x7fc0_1234), 1.0),
            Complex::new(1.0, f32::from_bits(0x7fc0_5678)),
        ];
        let right = [
            Complex::new(2f32, 0.0),
            Complex::new(1.0, 3.0),
            Complex::new(0.0, -0.0),
            Complex::new(2.0, 3.0),
            Complex::new(1.0, 4.0),
        ];
        let left_array = replicated_array(domain, mesh, DataType::C64, &[5], values_to_bytes(&left).as_slice());
        let right_array = replicated_array(domain, mesh, DataType::C64, &[5], values_to_bytes(&right).as_slice());
        (left, right, left_array, right_array)
    }

    #[test]
    fn test_max_operation_lower() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Boolean maxima are disjunctions.
        let booleans = replicated_array(&domain, &mesh, DataType::Boolean, &[2], &[0, 1]);
        let truth = replicated_array(&domain, &mesh, DataType::Boolean, &[], &[1]);
        assert_eq!(booleans.max(&truth).unwrap().copy_to_host().unwrap(), values_to_bytes(&[true, true]));
    }

    #[test]
    fn test_max_operation_lower_complex() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Complete operands compare lexicographically, and ties and unordered comparisons select the right operand.
        // Exact byte comparisons also check signed zeros and NaN payloads in both operand orders.
        let (left, right, left_array, right_array) = complex_operands(&domain, &mesh);
        assert_eq!(
            left_array.max(&right_array).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[right[0], right[1], right[2], right[3], right[4]]),
        );
        assert_eq!(
            right_array.max(&left_array).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[right[0], right[1], left[2], left[3], left[4]]),
        );

        // A real rank-0 operand broadcasts into a complex vector after promotion to complex double precision.
        let complex = replicated_array(
            &domain,
            &mesh,
            DataType::C128,
            &[2],
            values_to_bytes(&[Complex::new(1f64, 2.0), Complex::new(3.0, 4.0)]).as_slice(),
        );
        let real = replicated_array(&domain, &mesh, DataType::F64, &[], values_to_bytes(&[2f64]).as_slice());
        let output = complex.max(&real).unwrap();
        assert_eq!(output.data_type(), DataType::C128);
        assert_eq!(output.shape().dimensions(), &[2]);
        assert_eq!(output.copy_to_host().unwrap(), values_to_bytes(&[Complex::new(2f64, 0.0), Complex::new(3.0, 4.0)]));
    }

    #[test]
    fn test_min_operation_lower() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Boolean minima are conjunctions.
        let booleans = replicated_array(&domain, &mesh, DataType::Boolean, &[2], &[0, 1]);
        let truth = replicated_array(&domain, &mesh, DataType::Boolean, &[], &[1]);
        assert_eq!(booleans.min(&truth).unwrap().copy_to_host().unwrap(), values_to_bytes(&[false, true]));
    }

    #[test]
    fn test_min_operation_lower_complex() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Complete operands compare lexicographically, and ties and unordered comparisons select the right operand.
        // Exact byte comparisons also check signed zeros and NaN payloads in both operand orders.
        let (left, right, left_array, right_array) = complex_operands(&domain, &mesh);
        assert_eq!(
            left_array.min(&right_array).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[left[0], left[1], right[2], right[3], right[4]]),
        );
        assert_eq!(
            right_array.min(&left_array).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[left[0], left[1], left[2], left[3], left[4]]),
        );

        // A real rank-0 operand broadcasts into a complex vector after promotion to complex double precision.
        let complex = replicated_array(
            &domain,
            &mesh,
            DataType::C128,
            &[2],
            values_to_bytes(&[Complex::new(1f64, 2.0), Complex::new(3.0, 4.0)]).as_slice(),
        );
        let real = replicated_array(&domain, &mesh, DataType::F64, &[], values_to_bytes(&[2f64]).as_slice());
        let output = complex.min(&real).unwrap();
        assert_eq!(output.data_type(), DataType::C128);
        assert_eq!(output.shape().dimensions(), &[2]);
        assert_eq!(output.copy_to_host().unwrap(), values_to_bytes(&[Complex::new(1f64, 2.0), Complex::new(2.0, 0.0)]));
    }

    #[test]
    fn test_clamp_operation_lower() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Floating-point values below, inside, and above the interval keep their exact bytes, NaN inputs propagate,
        // and crossed bounds produce the upper bound.
        let input = replicated_array(
            &domain,
            &mesh,
            DataType::F32,
            &[4],
            values_to_bytes(&[-2.5f32, 0.5, 2.5, f32::NAN]).as_slice(),
        );
        let lower = replicated_array(&domain, &mesh, DataType::F32, &[], values_to_bytes(&[-1f32]).as_slice());
        let upper = replicated_array(&domain, &mesh, DataType::F32, &[], values_to_bytes(&[1f32]).as_slice());
        assert_eq!(
            input.clamp(&lower, &upper).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[-1f32, 0.5, 1.0, f32::NAN]),
        );
        assert_eq!(
            input.clamp(&upper, &lower).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[-1f32, -1.0, -1.0, f32::NAN]),
        );

        // Array bounds broadcast to the result.
        let bounds =
            replicated_array(&domain, &mesh, DataType::F32, &[4], values_to_bytes(&[0f32, 1.0, 2.0, 3.0]).as_slice());
        assert_eq!(
            input.clamp(&lower, &bounds).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[-1f32, 0.5, 2.0, f32::NAN]),
        );

        // Integer inputs clamp against rank-0 bounds.
        let integers =
            replicated_array(&domain, &mesh, DataType::I32, &[3], values_to_bytes(&[-5i32, 2, 9]).as_slice());
        let integer_lower = replicated_array(&domain, &mesh, DataType::I32, &[], values_to_bytes(&[0i32]).as_slice());
        let integer_upper = replicated_array(&domain, &mesh, DataType::I32, &[], values_to_bytes(&[5i32]).as_slice());
        assert_eq!(
            integers.clamp(&integer_lower, &integer_upper).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[0i32, 2, 5]),
        );

        // Boolean inputs clamp with conjunction and disjunction.
        let booleans = replicated_array(&domain, &mesh, DataType::Boolean, &[2], &[0, 1]);
        let truth = replicated_array(&domain, &mesh, DataType::Boolean, &[], &[1]);
        assert_eq!(booleans.clamp(&truth, &truth).unwrap().copy_to_host().unwrap(), values_to_bytes(&[true, true]));

        // All three inputs take part in promotion before selection, preserving integers that would round in `f32`.
        let input = replicated_array(&domain, &mesh, DataType::I64, &[], values_to_bytes(&[16777217i64]).as_slice());
        let lower = replicated_array(&domain, &mesh, DataType::F32, &[], values_to_bytes(&[0f32]).as_slice());
        let upper = replicated_array(&domain, &mesh, DataType::F64, &[], values_to_bytes(&[2e7f64]).as_slice());
        let output = input.clamp(&lower, &upper).unwrap();
        assert_eq!(output.data_type(), DataType::F64);
        assert_eq!(output.copy_to_host().unwrap(), values_to_bytes(&[16777217f64]));

        // Complex inputs compose the lexicographic extrema, so the imaginary part decides between equal real parts.
        let complex = replicated_array(
            &domain,
            &mesh,
            DataType::C64,
            &[3],
            values_to_bytes(&[Complex::new(1f32, -1.0), Complex::new(1.0, 1.0), Complex::new(1.0, 3.0)]).as_slice(),
        );
        let complex_lower = replicated_array(
            &domain,
            &mesh,
            DataType::C64,
            &[],
            values_to_bytes(&[Complex::new(1f32, 0.0)]).as_slice(),
        );
        let complex_upper = replicated_array(
            &domain,
            &mesh,
            DataType::C64,
            &[],
            values_to_bytes(&[Complex::new(1f32, 2.0)]).as_slice(),
        );
        assert_eq!(
            complex.clamp(&complex_lower, &complex_upper).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[Complex::new(1f32, 0.0), Complex::new(1.0, 1.0), Complex::new(1.0, 2.0)]),
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_extrema_dynamic_differentiation() {
        let client = execution_client();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("size", Some(5))]));
        let mut builder = ArrayIrProgramBuilder::new();
        let left = builder.add_input(input_type.clone().into());
        let right = builder.add_input(input_type.into());
        let minimum = builder.add_instruction(MinOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let maximum = builder.add_instruction(MaxOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![minimum, maximum], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
            .jvp()
            .unwrap();

        // The fourth element is padding. Both selection and tie weights must preserve the runtime extent.
        assert_eq!(
            execute_mixed_program(
                &client,
                &program,
                &[
                    MixedValue::Array(vec![1.0, 3.0, 2.0, 0.0], vec![4]),
                    MixedValue::Array(vec![2.0, 1.0, 2.0, 0.0], vec![4]),
                    MixedValue::Array(vec![2.0, 4.0, 6.0, 0.0], vec![4]),
                    MixedValue::Array(vec![8.0, 10.0, 12.0, 0.0], vec![4]),
                ],
                &[3, 3, 3, 3],
            )
            .unwrap(),
            vec![
                MixedValue::Array(vec![1.0, 1.0, 2.0], vec![3]),
                MixedValue::Array(vec![2.0, 3.0, 2.0], vec![3]),
                MixedValue::Array(vec![2.0, 10.0, 9.0], vec![3]),
                MixedValue::Array(vec![8.0, 4.0, 9.0], vec![3]),
                MixedValue::Dimension(3),
                MixedValue::Dimension(3),
                MixedValue::Dimension(3),
                MixedValue::Dimension(3),
            ],
        );
    }
}
