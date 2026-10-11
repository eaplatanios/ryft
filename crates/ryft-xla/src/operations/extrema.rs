use ryft_core::macros::check_count;
use ryft_core::{ArrayType, ClampOperation, DataType, MaxOperation, MinOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, LocationRef, Operation as MlirOperation, TensorTypeRef, Type, Value as MlirValue, ValueRef,
};

use crate::operations::{normalize_binary_elementwise_operands, normalize_elementwise_operand};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for MaxOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
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
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
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
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
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
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
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
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{Placeholder, ReductionKind, Shape};

    use crate::experimental::lowering::tests::{
        MixedValue, dynamic_dimension, execute_mixed_program, lowered_reduce_module,
    };
    use crate::experimental::ops::{XlaConstant, XlaProgramBuilder as CompositeXlaProgramBuilder};
    use crate::tests::execution_client;

    use super::*;

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_boolean_and_complex_extrema() {
        assert_eq!(
            lowered_reduce_module(DataType::Boolean, ReductionKind::Max, vec![0], vec![0]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<0xi1>) -> tensor<i1> {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.maximum across dimensions = [0] : (tensor<0xi1>, tensor<i1>) -> tensor<i1>
                    return %0 : tensor<i1>
                  }
                }
            "#},
        );

        // StableHLO has no primitive with JAX's complex ordering contract, so the reduction body explicitly compares
        // real components first and imaginary components only when the real components are equal.
        let stablehlo = lowered_reduce_module(DataType::C64, ReductionKind::Max, vec![0], vec![0]).unwrap();
        assert!(stablehlo.contains("tensor<complex<f32>>"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.reduce"), "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.real").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.imag").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.compare").count(), 3, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.select").count(), 2, "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_extrema_dynamic_differentiation() {
        let client = execution_client();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("size", Some(5))]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let left = builder.add_input(input_type.clone().into());
        let right = builder.add_input(input_type.into());
        let minimum = builder.add_instruction(MinOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let maximum = builder.add_instruction(MaxOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![minimum, maximum],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
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
            ),
            Ok(vec![
                MixedValue::Array(vec![1.0, 1.0, 2.0], vec![3]),
                MixedValue::Array(vec![2.0, 3.0, 2.0], vec![3]),
                MixedValue::Array(vec![2.0, 10.0, 9.0], vec![3]),
                MixedValue::Array(vec![8.0, 4.0, 9.0], vec![3]),
                MixedValue::Dimension(3),
                MixedValue::Dimension(3),
                MixedValue::Dimension(3),
                MixedValue::Dimension(3),
            ]),
        );
    }
}
