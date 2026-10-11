use ryft_core::{ArrayType, Atan2Operation, CosOperation, DataType, SinOperation, TanOperation, TanhOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::Accuracy;
use ryft_mlir::{Block, Context as MlirContext, Location, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::experimental::lowering::types::{lower_f64_constant_splat, lower_tensor_type};
use crate::operations::{lower_accuracy, normalize_binary_elementwise_operands};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Lowers complex sine or cosine through real operations while preserving an exact zero component for a zero real
/// input. The direct StableHLO complex trigonometric operations can evaluate a mathematically zero product as
/// `0 * inf`, yielding NaN when the imaginary component is large enough to overflow the hyperbolic factor. Small
/// imaginary components use `expm1` to avoid cancellation; large components scale the trigonometric factor before
/// forming the exponential product to avoid intermediate overflow.
///
/// # Parameters
///
///   - `input`: Complex tensor whose elementwise sine or cosine is lowered.
///   - `output_type`: Inferred complex output type, used to construct same-shaped real constants.
///   - `sine`: If `true`, lowers sine; otherwise, lowers cosine.
///   - `block`: Destination block for the StableHLO decomposition.
///   - `context`: MLIR context that owns all emitted types and attributes.
///   - `location`: Source location attached to emitted operations.
fn lower_complex_sine_or_cosine<'b, 'c: 'b, 't: 'c, B, L>(
    input: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    sine: bool,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let part_data_type = match output_type.data_type() {
        DataType::C64 => DataType::F32,
        DataType::C128 => DataType::F64,
        _ => unreachable!(),
    };
    let part_type = ArrayType::new(part_data_type, output_type.shape().clone());
    let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
    let zero = lower_f64_constant_splat(0.0, &part_type, part_tensor_type, block, context, location)?;
    let two = lower_f64_constant_splat(2.0, &part_type, part_tensor_type, block, context, location)?;

    let real = block.append_operation(stable_hlo::real(input, location)?)?;
    let real = real.result(0).expect("stablehlo.real should return one result").as_ref();
    let imaginary = block.append_operation(stable_hlo::imag(input, location)?)?;
    let imaginary = imaginary.result(0).expect("stablehlo.imag should return one result").as_ref();
    let real_is_zero = block.append_operation(stable_hlo::compare(
        real,
        zero,
        stable_hlo::ComparisonDirection::Equal,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let real_is_zero = real_is_zero.result(0).expect("stablehlo.compare should return one result").as_ref();
    let real_sine = block.append_operation(stable_hlo::sine(real, Accuracy::Default, location)?)?;
    let real_sine = real_sine.result(0).expect("stablehlo.sine should return one result").as_ref();
    let real_cosine = block.append_operation(stable_hlo::cosine(real, Accuracy::Default, location)?)?;
    let real_cosine = real_cosine.result(0).expect("stablehlo.cosine should return one result").as_ref();
    let negative_imaginary = block.append_operation(stable_hlo::negate(imaginary, location)?)?;
    let negative_imaginary = negative_imaginary.result(0).expect("stablehlo.negate should return one result").as_ref();
    let positive_exponential_minus_one =
        block.append_operation(stable_hlo::exponential_minus_one(imaginary, Accuracy::Default, location)?)?;
    let positive_exponential_minus_one = positive_exponential_minus_one
        .result(0)
        .expect("stablehlo.exponential_minus_one should return one result")
        .as_ref();
    let negative_exponential_minus_one =
        block.append_operation(stable_hlo::exponential_minus_one(negative_imaginary, Accuracy::Default, location)?)?;
    let negative_exponential_minus_one = negative_exponential_minus_one
        .result(0)
        .expect("stablehlo.exponential_minus_one should return one result")
        .as_ref();
    let sinh_numerator = block.append_operation(stable_hlo::subtract(
        positive_exponential_minus_one,
        negative_exponential_minus_one,
        location,
    )?)?;
    let sinh_numerator = sinh_numerator.result(0).expect("stablehlo.subtract should return one result").as_ref();
    let sinh = block.append_operation(stable_hlo::divide(sinh_numerator, two, location)?)?;
    let sinh = sinh.result(0).expect("stablehlo.divide should return one result").as_ref();
    let cosh_without_two = block.append_operation(stable_hlo::add(
        positive_exponential_minus_one,
        negative_exponential_minus_one,
        location,
    )?)?;
    let cosh_without_two = cosh_without_two.result(0).expect("stablehlo.add should return one result").as_ref();
    let cosh_numerator = block.append_operation(stable_hlo::add(cosh_without_two, two, location)?)?;
    let cosh_numerator = cosh_numerator.result(0).expect("stablehlo.add should return one result").as_ref();
    let cosh = block.append_operation(stable_hlo::divide(cosh_numerator, two, location)?)?;
    let cosh = cosh.result(0).expect("stablehlo.divide should return one result").as_ref();

    // Scale the trigonometric factor before the second exponential multiplication. This keeps representable
    // components finite even when `exp(abs(imaginary))`, or the hyperbolic factor itself, would overflow. Above
    // magnitude twenty, dropping the decaying exponential changes the factor by less than one `f64` ulp.
    let magnitude = block.append_operation(stable_hlo::abs(imaginary, location)?)?.result(0).unwrap().as_ref();
    let half = lower_f64_constant_splat(0.5, &part_type, part_tensor_type, block, context, location)?;
    let threshold = lower_f64_constant_splat(20.0, &part_type, part_tensor_type, block, context, location)?;
    let infinity = lower_f64_constant_splat(f64::INFINITY, &part_type, part_tensor_type, block, context, location)?;
    let log_two =
        lower_f64_constant_splat(std::f64::consts::LN_2, &part_type, part_tensor_type, block, context, location)?;
    let large = block
        .append_operation(stable_hlo::compare(
            magnitude,
            threshold,
            stable_hlo::ComparisonDirection::GreaterThan,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let half_magnitude = block
        .append_operation(stable_hlo::multiply(magnitude, half, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let half_exponential = block
        .append_operation(stable_hlo::exponential(half_magnitude, Accuracy::Default, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let half_overflows = block
        .append_operation(stable_hlo::compare(
            half_exponential,
            infinity,
            stable_hlo::ComparisonDirection::Equal,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let negative = block
        .append_operation(stable_hlo::compare(
            imaginary,
            zero,
            stable_hlo::ComparisonDirection::LessThan,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let mut large_component = |factor: ValueRef<'b, 'c, 't>,
                               odd: bool|
     -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
        let factor = if odd {
            let negated = block.append_operation(stable_hlo::negate(factor, location)?)?.result(0).unwrap().as_ref();
            block
                .append_operation(stable_hlo::select(negative, negated, factor, location)?)?
                .result(0)
                .unwrap()
                .as_ref()
        } else {
            factor
        };
        let first = block
            .append_operation(stable_hlo::multiply(factor, half_exponential, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let first = block.append_operation(stable_hlo::multiply(first, half, location)?)?.result(0).unwrap().as_ref();
        let scaled = block
            .append_operation(stable_hlo::multiply(first, half_exponential, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        // Extremely small trigonometric factors can still produce finite components when the half exponential
        // overflows. Combine exponents for that rare case; retain exact zeros before taking a logarithm.
        let absolute = block.append_operation(stable_hlo::abs(factor, location)?)?.result(0).unwrap().as_ref();
        let logarithm = block
            .append_operation(stable_hlo::log(absolute, Accuracy::Default, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let exponent = block
            .append_operation(stable_hlo::add(magnitude, logarithm, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let exponent = block
            .append_operation(stable_hlo::subtract(exponent, log_two, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let fallback = block
            .append_operation(stable_hlo::exponential(exponent, Accuracy::Default, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let sign = block.append_operation(stable_hlo::sign(factor, location)?)?.result(0).unwrap().as_ref();
        let fallback =
            block.append_operation(stable_hlo::multiply(sign, fallback, location)?)?.result(0).unwrap().as_ref();
        let scaled = block
            .append_operation(stable_hlo::select(half_overflows, fallback, scaled, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let factor_is_zero = block
            .append_operation(stable_hlo::compare(
                factor,
                zero,
                stable_hlo::ComparisonDirection::Equal,
                stable_hlo::ComparisonType::Float,
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref();
        Ok(block
            .append_operation(stable_hlo::select(factor_is_zero, factor, scaled, location)?)?
            .result(0)
            .unwrap()
            .as_ref())
    };
    let large_real = large_component(if sine { real_sine } else { real_cosine }, false)?;
    let large_imaginary = large_component(if sine { real_cosine } else { real_sine }, true)?;
    let large_imaginary = if sine {
        large_imaginary
    } else {
        block.append_operation(stable_hlo::negate(large_imaginary, location)?)?.result(0).unwrap().as_ref()
    };

    let (real_result, imaginary_result) = if sine {
        // Mask the overflowing factor before multiplication as well as selecting the final complex result. Some XLA
        // optimization paths otherwise speculate the unselected `0 * inf` expression and preserve its NaN.
        let safe_cosh = block.append_operation(stable_hlo::select(real_is_zero, zero, cosh, location)?)?;
        let safe_cosh = safe_cosh.result(0).expect("stablehlo.select should return one result").as_ref();
        let real_result = block.append_operation(stable_hlo::multiply(real_sine, safe_cosh, location)?)?;
        let real_result = real_result.result(0).expect("stablehlo.multiply should return one result").as_ref();
        let imaginary_result = block.append_operation(stable_hlo::multiply(real_cosine, sinh, location)?)?;
        let imaginary_result =
            imaginary_result.result(0).expect("stablehlo.multiply should return one result").as_ref();
        (real_result, imaginary_result)
    } else {
        let real_result = block.append_operation(stable_hlo::multiply(real_cosine, cosh, location)?)?;
        let real_result = real_result.result(0).expect("stablehlo.multiply should return one result").as_ref();
        let negative_real_sine = block.append_operation(stable_hlo::negate(real_sine, location)?)?;
        let negative_real_sine =
            negative_real_sine.result(0).expect("stablehlo.negate should return one result").as_ref();
        let safe_sinh = block.append_operation(stable_hlo::select(real_is_zero, zero, sinh, location)?)?;
        let safe_sinh = safe_sinh.result(0).expect("stablehlo.select should return one result").as_ref();
        let imaginary_result =
            block.append_operation(stable_hlo::multiply(negative_real_sine, safe_sinh, location)?)?;
        let imaginary_result =
            imaginary_result.result(0).expect("stablehlo.multiply should return one result").as_ref();
        (real_result, imaginary_result)
    };
    let real_result = block
        .append_operation(stable_hlo::select(large, large_real, real_result, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let imaginary_result = block
        .append_operation(stable_hlo::select(large, large_imaginary, imaginary_result, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let ordinary_result = block.append_operation(stable_hlo::complex(real_result, imaginary_result, location)?)?;
    let ordinary_result = ordinary_result.result(0).expect("stablehlo.complex should return one result").as_ref();
    let zero_real_result = if sine {
        block.append_operation(stable_hlo::complex(zero, imaginary_result, location)?)?
    } else {
        block.append_operation(stable_hlo::complex(real_result, zero, location)?)?
    };
    let zero_real_result = zero_real_result.result(0).expect("stablehlo.complex should return one result").as_ref();
    let result =
        block.append_operation(stable_hlo::select(real_is_zero, zero_real_result, ordinary_result, location)?)?;
    Ok(result.result(0).expect("stablehlo.select should return one result").as_ref())
}

impl LowerableOperation for SinOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        if output_types[0].data_type().is_complex() {
            return Ok(vec![lower_complex_sine_or_cosine(
                input_values[0],
                &output_types[0],
                true,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?]);
        }
        let result = lowerer.block.append_operation(stable_hlo::sine(
            input_values[0],
            lower_accuracy(self.accuracy()),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.sine should return one result").as_ref()])
    }
}

impl LowerableOperation for CosOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        if output_types[0].data_type().is_complex() {
            return Ok(vec![lower_complex_sine_or_cosine(
                input_values[0],
                &output_types[0],
                false,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?]);
        }
        let result = lowerer.block.append_operation(stable_hlo::cosine(
            input_values[0],
            lower_accuracy(self.accuracy()),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.cosine should return one result").as_ref()])
    }
}

impl LowerableOperation for TanOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::tan(
            input_values[0],
            lower_accuracy(self.accuracy()),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for Atan2Operation<ArrayType> {
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
        let result = lowerer.block.append_operation(stable_hlo::atan2(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.atan2 should return one result").as_ref()])
    }
}

impl LowerableOperation for TanhOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::tanh(
            input_values[0],
            lower_accuracy(self.accuracy()),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.tanh should return one result").as_ref()])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayOperation, Placeholder};

    use crate::experimental::lowering::tests::{XlaProgramBuilder, lowered_unary_module};
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::XlaArrayConstant;

    use super::*;

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_complex_sine_and_cosine() {
        let complex_type = ArrayType::scalar(DataType::C64);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(complex_type);
        let sine = builder
            .add_instruction(ArrayOperation::Sin(SinOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let cosine = builder
            .add_instruction(ArrayOperation::Cos(CosOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![sine, cosine],
                vec![Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(stablehlo.matches("stablehlo.real").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.imag").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.exponential_minus_one").count(), 4, "{stablehlo}");
        // Select the small-argument, scaled, and logarithmic paths while preserving exact zero components.
        assert_eq!(stablehlo.matches("stablehlo.select").count(), 18, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.complex").count(), 4, "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_tangent() {
        assert_eq!(
            lowered_unary_module(ArrayOperation::Tan(TanOperation::new()), DataType::F32, vec![2]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.tan %arg0 : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "#},
        );
        assert_eq!(
            lowered_unary_module(ArrayOperation::Tan(TanOperation::new()), DataType::C64, vec![2]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
                    %0 = stablehlo.tan %arg0 : tensor<2xcomplex<f32>>
                    return %0 : tensor<2xcomplex<f32>>
                  }
                }
            "#},
        );
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Tan(TanOperation::new().with_accuracy(ryft_core::Accuracy::Highest)),
                DataType::F32,
                vec![2],
            )
            .unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.tan %arg0 {result_accuracy = #stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "#},
        );
    }
}
