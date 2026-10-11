use ryft_core::{
    ArrayType, DataType, ExpOperation, Ln1pOperation, LogAddExpOperation, LogOperation, LogisticOperation,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::Accuracy;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::{
    lower_f64_constant_splat, lower_tensor_type, lower_unplaced_constant_output,
};
use crate::operations::complex::lower_select_principal_phase_to_mlir;
use crate::operations::extrema::lower_extremum_to_mlir;
use crate::operations::{lower_accuracy, normalize_binary_elementwise_operands};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for ExpOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::exponential(
            input_values[0],
            lower_accuracy(self.accuracy()),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.exponential should return one result").as_ref()])
    }
}

impl LowerableOperation for LogOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::log(
            input_values[0],
            lower_accuracy(self.accuracy()),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.log should return one result").as_ref()])
    }
}

impl LowerableOperation for Ln1pOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        if output_types[0].data_type().is_complex() {
            return Ok(vec![lower_complex_ln_1p_to_mlir(
                input_values[0],
                &output_types[0],
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?]);
        }
        let result = lowerer.block.append_operation(stable_hlo::log_plus_one(
            input_values[0],
            lower_accuracy(self.accuracy()),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.log_plus_one should return one result").as_ref()])
    }
}

// StableHLO has no `logaddexp` primitive, so the operation lowers by expanding the guarded construction that
// `ryft-core` pins, shared with the log-sum-exp `cumulative` reducer body through `lower_log_add_exp_to_mlir`.
impl LowerableOperation for LogAddExpOperation<ArrayType> {
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
        let value = lower_log_add_exp_to_mlir(
            left,
            right,
            &output_types[0],
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        Ok(vec![value])
    }
}

impl LowerableOperation for LogisticOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::logistic(
            input_values[0],
            lower_accuracy(self.accuracy()),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.logistic should return one result").as_ref()])
    }
}

/// Lowers a principal complex `ln_1p` without cancellation near zero or magnitude underflow near `-1`.
/// The ordinary complex logarithm supplies its scaled-magnitude implementation away from zero; near zero,
/// `ln_1p(real * (2 + real) + imaginary²) / 2` retains the perturbation of the squared magnitude from one.
fn lower_complex_ln_1p_to_mlir<'b, 'c: 'b, 't: 'c>(
    input: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let part_type = output_type.clone().with_data_type(if output_type.data_type() == DataType::C64 {
        DataType::F32
    } else {
        DataType::F64
    });
    let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
    let one = lower_f64_constant_splat(1.0, &part_type, part_tensor_type, block, context, location)?;
    let two = lower_f64_constant_splat(2.0, &part_type, part_tensor_type, block, context, location)?;
    let half = lower_f64_constant_splat(0.5, &part_type, part_tensor_type, block, context, location)?;
    let nan = lower_f64_constant_splat(f64::NAN, &part_type, part_tensor_type, block, context, location)?;
    let real = block.append_operation(stable_hlo::real(input, location)?)?.result(0).unwrap().as_ref();
    let imaginary = block.append_operation(stable_hlo::imag(input, location)?)?.result(0).unwrap().as_ref();
    let shifted_real = block.append_operation(stable_hlo::add(one, real, location)?)?.result(0).unwrap().as_ref();
    let shifted = block
        .append_operation(stable_hlo::complex(shifted_real, imaginary, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let ordinary = block
        .append_operation(stable_hlo::log(shifted, Accuracy::Default, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let ordinary_real = block.append_operation(stable_hlo::real(ordinary, location)?)?.result(0).unwrap().as_ref();

    // Only the near-zero branch forms the small squared-magnitude perturbation; neither branch divides
    // by the magnitude, so the logarithmic singularity and signed branch-cut arguments remain well defined.
    let shifted_two = block.append_operation(stable_hlo::add(two, real, location)?)?.result(0).unwrap().as_ref();
    let real_term = block
        .append_operation(stable_hlo::multiply(real, shifted_two, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let imaginary_term = block
        .append_operation(stable_hlo::multiply(imaginary, imaginary, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let perturbation = block
        .append_operation(stable_hlo::add(real_term, imaginary_term, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let near_log = block
        .append_operation(stable_hlo::log_plus_one(perturbation, Accuracy::Default, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let near_real =
        block.append_operation(stable_hlo::multiply(half, near_log, location)?)?.result(0).unwrap().as_ref();
    let real_magnitude = block.append_operation(stable_hlo::abs(real, location)?)?.result(0).unwrap().as_ref();
    let imaginary_magnitude =
        block.append_operation(stable_hlo::abs(imaginary, location)?)?.result(0).unwrap().as_ref();
    let near_real_axis = block
        .append_operation(stable_hlo::compare(
            real_magnitude,
            half,
            stable_hlo::ComparisonDirection::LessThan,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let near_imaginary_axis = block
        .append_operation(stable_hlo::compare(
            imaginary_magnitude,
            half,
            stable_hlo::ComparisonDirection::LessThan,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let near = block
        .append_operation(stable_hlo::and(near_real_axis, near_imaginary_axis, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let output_real = block
        .append_operation(stable_hlo::select(near, near_real, ordinary_real, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let output_imaginary = block
        .append_operation(stable_hlo::atan2(imaginary, shifted_real, location)?)?
        .result(0)
        .unwrap()
        .as_ref();

    // Make NaN propagation explicit even when the other component is infinite.
    let real_nan = block
        .append_operation(stable_hlo::compare(
            real,
            real,
            stable_hlo::ComparisonDirection::NotEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let imaginary_nan = block
        .append_operation(stable_hlo::compare(
            imaginary,
            imaginary,
            stable_hlo::ComparisonDirection::NotEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let is_nan = block
        .append_operation(stable_hlo::or(real_nan, imaginary_nan, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let output_real = block
        .append_operation(stable_hlo::select(is_nan, nan, output_real, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let output_imaginary = block
        .append_operation(stable_hlo::select(is_nan, nan, output_imaginary, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    Ok(block
        .append_operation(stable_hlo::complex(output_real, output_imaginary, location)?)?
        .result(0)
        .unwrap()
        .as_ref())
}

/// Lowers one elementwise `log(exp(a) + exp(b))` by expanding the guarded construction that `ryft-core`'s
/// [`LogAddExpOperation`] pins, since StableHLO has no `logaddexp` primitive:
///
/// ```text
/// log_add_exp(a, b) = select(isnan(a - b), a + b, max(a, b) + ln_1p(exp(-|a - b|)))
/// ```
///
/// Factoring the larger operand out of the sum keeps `exp` from overflowing anywhere on the real line, and the
/// `isnan(a - b)` arm routes the cases whose difference is undefined through `a + b`, which pins `(+∞, +∞) ↦ +∞`,
/// `(-∞, -∞) ↦ -∞`, and NaN propagation. Both operands must already carry the same tensor type, which is what the
/// callers guarantee: the elementwise operation normalizes its broadcast operands, and the
/// log-sum-exp `cumulative` reducer body applies the same expansion to two scalar block arguments.
/// Narrow real formats evaluate the entire composition in `f32` before converting back. Complex values factor out the
/// operand `p` with the larger real component, as `p + ln_1p(exp(q - p))`, and form the ratio `exp(q - p)` by rotating
/// by `p`'s phase rather than by subtracting the imaginary components from each other, exactly like `ryft-core`'s
/// reference kernel. The imaginary part lies in `[-pi, pi]`, reduced through `atan2` when it leaves that range.
pub(crate) fn lower_log_add_exp_to_mlir<'b, 'c: 'b, 't: 'c>(
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    let data_type = output_type.data_type();
    let narrow = !data_type.is_complex() && !matches!(data_type, DataType::F32 | DataType::F64);
    let (left, right) = if narrow {
        // Evaluate the entire composite in one working type, then round only its final output.
        let working_type = lower_tensor_type(&output_type.clone().with_data_type(DataType::F32), context, location)?;
        let left = block
            .append_operation(stable_hlo::convert(left, working_type, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let right = block
            .append_operation(stable_hlo::convert(right, working_type, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        (left, right)
    } else {
        (left, right)
    };
    if data_type.is_complex() {
        // Factor out the operand `p` with the larger real component, so that the output is `p + ln_1p(exp(q - p))`,
        // which keeps small corrections exact. The ratio `exp(q - p)` is formed as `exp(q - re(p)) · exp(-i im(p))`,
        // which rotates by `p`'s phase instead of subtracting the imaginary components from each other.
        let larger = lower_extremum_to_mlir(true, data_type, left, right, block, location)?;
        let smaller = lower_extremum_to_mlir(false, data_type, left, right, block, location)?;
        let part_type =
            output_type
                .clone()
                .with_data_type(if data_type == DataType::C64 { DataType::F32 } else { DataType::F64 });
        let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
        let zero = lower_f64_constant_splat(0.0, &part_type, part_tensor_type, block, context, location)?;
        let larger_real = block.append_operation(stable_hlo::real(larger, location)?)?.result(0).unwrap().as_ref();
        let larger_imaginary = block.append_operation(stable_hlo::imag(larger, location)?)?.result(0).unwrap().as_ref();
        let cosine = block
            .append_operation(stable_hlo::cosine(larger_imaginary, Accuracy::Default, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let sine = block
            .append_operation(stable_hlo::sine(larger_imaginary, Accuracy::Default, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let negated_sine = block.append_operation(stable_hlo::negate(sine, location)?)?.result(0).unwrap().as_ref();
        let rotation =
            block.append_operation(stable_hlo::complex(cosine, sine, location)?)?.result(0).unwrap().as_ref();
        let inverse_rotation = block
            .append_operation(stable_hlo::complex(cosine, negated_sine, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let shift = block
            .append_operation(stable_hlo::complex(larger_real, zero, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let shifted =
            block.append_operation(stable_hlo::subtract(smaller, shift, location)?)?.result(0).unwrap().as_ref();
        let exponential = block
            .append_operation(stable_hlo::exponential(shifted, Accuracy::Default, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let ratio = block
            .append_operation(stable_hlo::multiply(exponential, inverse_rotation, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let correction = lower_complex_ln_1p_to_mlir(ratio, output_type, block, context, location)?;
        let correction_real =
            block.append_operation(stable_hlo::real(correction, location)?)?.result(0).unwrap().as_ref();
        let correction_imaginary =
            block.append_operation(stable_hlo::imag(correction, location)?)?.result(0).unwrap().as_ref();
        let real = block
            .append_operation(stable_hlo::add(larger_real, correction_real, location)?)?
            .result(0)
            .unwrap()
            .as_ref();

        // A principal phase is kept as the plain sum, which preserves tiny corrections. Any other phase is reduced
        // through `atan2` of the rotated sum `exp(i im(p)) · (1 + ratio)`, whose components come from a sine and cosine
        // rather than from a floating-point remainder by `2π`.
        let imaginary = block
            .append_operation(stable_hlo::add(larger_imaginary, correction_imaginary, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let one = lower_unplaced_constant_output(&[output_type.clone()], 1, block, context, location)?[0];
        let unit = block.append_operation(stable_hlo::add(ratio, one, location)?)?.result(0).unwrap().as_ref();
        let unit = block.append_operation(stable_hlo::multiply(rotation, unit, location)?)?.result(0).unwrap().as_ref();
        let unit_real = block.append_operation(stable_hlo::real(unit, location)?)?.result(0).unwrap().as_ref();
        let unit_imaginary = block.append_operation(stable_hlo::imag(unit, location)?)?.result(0).unwrap().as_ref();
        let reduced = block
            .append_operation(stable_hlo::atan2(unit_imaginary, unit_real, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let imaginary = lower_select_principal_phase_to_mlir(
            imaginary,
            reduced,
            &part_type,
            part_tensor_type,
            block,
            context,
            location,
        )?;
        return Ok(block
            .append_operation(stable_hlo::complex(real, imaginary, location)?)?
            .result(0)
            .unwrap()
            .as_ref());
    }
    let difference = block.append_operation(stable_hlo::subtract(left, right, location)?)?;
    let difference = difference.result(0).expect("stablehlo.subtract should return one result").as_ref();
    let is_nan = block.append_operation(stable_hlo::compare(
        difference,
        difference,
        stable_hlo::ComparisonDirection::NotEqual,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let is_nan = is_nan.result(0).expect("stablehlo.compare should return one result").as_ref();
    let naive = block.append_operation(stable_hlo::add(left, right, location)?)?;
    let naive = naive.result(0).expect("stablehlo.add should return one result").as_ref();
    let magnitude = block.append_operation(stable_hlo::abs(difference, location)?)?;
    let magnitude = magnitude.result(0).expect("stablehlo.abs should return one result").as_ref();
    let negated = block.append_operation(stable_hlo::negate(magnitude, location)?)?;
    let negated = negated.result(0).expect("stablehlo.negate should return one result").as_ref();
    let exponential = block.append_operation(stable_hlo::exponential(negated, Accuracy::Default, location)?)?;
    let exponential = exponential.result(0).expect("stablehlo.exponential should return one result").as_ref();
    let correction = block.append_operation(stable_hlo::log_plus_one(exponential, Accuracy::Default, location)?)?;
    let correction = correction.result(0).expect("stablehlo.log_plus_one should return one result").as_ref();
    // The guard already routes every NaN operand through the `a + b` arm, so the shift can use the plain StableHLO
    // maximum rather than the NaN-propagating total-order expansion that `lower_extremum_to_mlir` emits.
    let shift = block.append_operation(stable_hlo::maximum(left, right, location)?)?;
    let shift = shift.result(0).expect("stablehlo.maximum should return one result").as_ref();
    let stable = block.append_operation(stable_hlo::add(shift, correction, location)?)?;
    let stable = stable.result(0).expect("stablehlo.add should return one result").as_ref();
    let result = block.append_operation(stable_hlo::select(is_nan, naive, stable, location)?)?;
    let output = result.result(0).unwrap().as_ref();
    Ok(if narrow {
        block
            .append_operation(stable_hlo::convert(output, output_tensor_type, location)?)?
            .result(0)
            .unwrap()
            .as_ref()
    } else {
        output
    })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayOperation, Dimension, Placeholder, Shape};

    use crate::experimental::lowering::tests::{XlaProgramBuilder, lowered_unary_module};
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::XlaArrayConstant;

    use super::*;

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_ln_1p_and_log_add_exp() {
        // `ln_1p` has a StableHLO primitive of its own, so it lowers to exactly one operation.
        assert_eq!(
            lowered_unary_module(ArrayOperation::Ln1p(Ln1pOperation::new()), DataType::F32, vec![4]).unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.log_plus_one %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );

        // `log_add_exp` has none, so it expands into the guarded max-shifted construction: the `NE` self-comparison
        // is the `isnan(a - b)` test that routes same-sign infinities and NaN operands through the plain `a + b`
        // sum, while the ordinary arm shifts by the larger operand so that the exponential never overflows.
        let mut builder = XlaProgramBuilder::new();
        let left = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)])));
        let right = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)])));
        let output = builder
            .add_instruction(ArrayOperation::LogAddExp(LogAddExpOperation::new()), Vec::new(), vec![left, right], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xf32>
                    %1 = stablehlo.compare NE, %0, %0, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
                    %2 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
                    %3 = stablehlo.abs %0 : tensor<4xf32>
                    %4 = stablehlo.negate %3 : tensor<4xf32>
                    %5 = stablehlo.exponential %4 : tensor<4xf32>
                    %6 = stablehlo.log_plus_one %5 : tensor<4xf32>
                    %7 = stablehlo.maximum %arg0, %arg1 : tensor<4xf32>
                    %8 = stablehlo.add %7, %6 : tensor<4xf32>
                    %9 = stablehlo.select %1, %2, %8 : tensor<4xi1>, tensor<4xf32>
                    return %9 : tensor<4xf32>
                  }
                }
            "},
        );
    }
}
