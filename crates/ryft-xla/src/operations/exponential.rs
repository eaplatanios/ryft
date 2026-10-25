use ryft_core::{
    ArrayType, DataType, Dimension, ExpM1Operation, ExpOperation, Ln1pOperation, LogAddExpOperation, LogOperation,
    LogisticOperation,
};
use ryft_mlir::dialects::stable_hlo::Accuracy;
use ryft_mlir::dialects::{stable_hlo, tensor};
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::{
    lower_explicit_shape, lower_f64_constant_splat, lower_runtime_dimension_size_i64, lower_static_index_constants,
    lower_tensor_type, lower_unplaced_constant_output,
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

impl LowerableOperation for ExpM1Operation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        if output_types[0].data_type().is_complex() {
            return Ok(vec![lower_complex_exp_m1_to_mlir(
                input_values[0],
                &output_types[0],
                lower_accuracy(self.accuracy()),
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?]);
        }
        let result = lowerer.block.append_operation(stable_hlo::exponential_minus_one(
            input_values[0],
            lower_accuracy(self.accuracy()),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.exponential_minus_one should return one result").as_ref()])
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
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let output_type = &output_types[0];
        // Native logistic expands to `1 / (1 + exp(-x))`, whose exponential overflows before a negative
        // sigmoid tail underflows. Select the exponential's sign so its real component stays nonpositive.
        let complex = output_type.data_type().is_complex();
        let narrow = !complex && !matches!(output_type.data_type(), DataType::F32 | DataType::F64);
        let working_type = if narrow { output_type.clone().with_data_type(DataType::F32) } else { output_type.clone() };
        let working_tensor_type = lower_tensor_type(&working_type, lowerer.context, lowerer.location)?;
        let input = if narrow {
            lowerer
                .block
                .append_operation(stable_hlo::convert(input_values[0], working_tensor_type, lowerer.location)?)?
                .result(0)
                .unwrap()
                .as_ref()
        } else {
            input_values[0]
        };
        let part_type = working_type.clone().with_data_type(match working_type.data_type() {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            data_type => data_type,
        });
        let real = if complex {
            lowerer
                .block
                .append_operation(stable_hlo::real(input, lowerer.location)?)?
                .result(0)
                .unwrap()
                .as_ref()
        } else {
            input
        };
        let one = lower_exponential_constant_like(
            1.0,
            real,
            &part_type,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let zero = lower_exponential_constant_like(
            0.0,
            real,
            &part_type,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let nonnegative = lowerer
            .block
            .append_operation(stable_hlo::compare(
                real,
                zero,
                stable_hlo::ComparisonDirection::GreaterThanOrEqual,
                stable_hlo::ComparisonType::Float,
                lowerer.location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref();
        let one = if complex {
            lowerer
                .block
                .append_operation(stable_hlo::complex(one, zero, lowerer.location)?)?
                .result(0)
                .unwrap()
                .as_ref()
        } else {
            one
        };
        let negative_input = lowerer
            .block
            .append_operation(stable_hlo::negate(input, lowerer.location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let exponential_input = lowerer
            .block
            .append_operation(stable_hlo::select(nonnegative, negative_input, input, lowerer.location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let exponential = lowerer
            .block
            .append_operation(stable_hlo::exponential(
                exponential_input,
                lower_accuracy(self.accuracy()),
                lowerer.location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref();
        let denominator = lowerer
            .block
            .append_operation(stable_hlo::add(one, exponential, lowerer.location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let positive = lowerer
            .block
            .append_operation(stable_hlo::divide(one, denominator, lowerer.location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let negative = lowerer
            .block
            .append_operation(stable_hlo::divide(exponential, denominator, lowerer.location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let output = lowerer
            .block
            .append_operation(stable_hlo::select(nonnegative, positive, negative, lowerer.location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        Ok(vec![if narrow {
            lowerer
                .block
                .append_operation(stable_hlo::convert(
                    output,
                    lower_tensor_type(output_type, lowerer.context, lowerer.location)?,
                    lowerer.location,
                )?)?
                .result(0)
                .unwrap()
                .as_ref()
        } else {
            output
        }])
    }
}

/// Broadcasts a real scalar literal to an exemplar's runtime shape, retaining bounded dynamic dimensions.
fn lower_exponential_constant_like<'b, 'c: 'b, 't: 'c>(
    factor: f64,
    input: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    if output_type.static_shape().is_some() {
        return lower_f64_constant_splat(factor, output_type, output_tensor_type, block, context, location);
    }
    let scalar_type = ArrayType::scalar(output_type.data_type());
    let scalar = lower_f64_constant_splat(
        factor,
        &scalar_type,
        lower_tensor_type(&scalar_type, context, location)?,
        block,
        context,
        location,
    )?;
    let dimensions = output_type
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .map(|(axis, dimension)| match dimension {
            Dimension::Static(extent) => {
                Ok(lower_static_index_constants(&[*extent], block, context, location)?.remove(0))
            }
            Dimension::Dynamic(_) => lower_runtime_dimension_size_i64(input, axis, block, context, location),
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let shape = lower_explicit_shape(&dimensions, block, context, location)?;
    let broadcast = block
        .append_operation(stable_hlo::dynamic_broadcast(scalar, shape, &[], None, None, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    Ok(block
        .append_operation(tensor::cast(broadcast, output_tensor_type, location)?)?
        .result(0)
        .unwrap()
        .as_ref())
}

/// Lowers complex `exp_m1` without recovering `exp(real)` by adding one to a rounded `exp_m1(real)`.
fn lower_complex_exp_m1_to_mlir<'b, 'c: 'b, 't: 'c>(
    input: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    accuracy: Accuracy,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let part_type = output_type.clone().with_data_type(if output_type.data_type() == DataType::C64 {
        DataType::F32
    } else {
        DataType::F64
    });
    let two = lower_exponential_constant_like(2.0, input, &part_type, block, context, location)?;
    let half = lower_exponential_constant_like(0.5, input, &part_type, block, context, location)?;
    let real_input = block.append_operation(stable_hlo::real(input, location)?)?.result(0).unwrap().as_ref();
    let imaginary = block.append_operation(stable_hlo::imag(input, location)?)?.result(0).unwrap().as_ref();
    let exponential_minus_one = block
        .append_operation(stable_hlo::exponential_minus_one(real_input, accuracy, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let cosine = block
        .append_operation(stable_hlo::cosine(imaginary, accuracy, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let half_imaginary = block
        .append_operation(stable_hlo::multiply(imaginary, half, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let half_sine = block
        .append_operation(stable_hlo::sine(half_imaginary, accuracy, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let squared_half_sine = block
        .append_operation(stable_hlo::multiply(half_sine, half_sine, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let correction = block
        .append_operation(stable_hlo::multiply(two, squared_half_sine, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let real = block
        .append_operation(stable_hlo::multiply(exponential_minus_one, cosine, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let real = block
        .append_operation(stable_hlo::subtract(real, correction, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    // This independent exponential retains the imaginary tail after the real output has rounded to minus one.
    let exponential = block
        .append_operation(stable_hlo::exponential(real_input, accuracy, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let sine = block
        .append_operation(stable_hlo::sine(imaginary, accuracy, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let imaginary = block
        .append_operation(stable_hlo::multiply(exponential, sine, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    Ok(block.append_operation(stable_hlo::complex(real, imaginary, location)?)?.result(0).unwrap().as_ref())
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

    use crate::experimental::lowering::tests::{XlaProgramBuilder, dynamic_dimension, lowered_unary_module};
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::XlaArrayConstant;

    use super::*;

    /// Lowers one unary operation while retaining its bounded dynamic input and result shape.
    fn lowered_dynamic_unary_module(operation: ArrayOperation<XlaArrayConstant>, data_type: DataType) -> String {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(ArrayType::new(data_type, Shape::new(vec![dynamic_dimension("size", Some(5))])));
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        to_mlir_module_for_plain_program(&program, "main").unwrap()
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_exp_m1() {
        assert_eq!(
            lowered_unary_module(ArrayOperation::ExpM1(ExpM1Operation::new()), DataType::F32, vec![4]).unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.exponential_minus_one %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );
        assert_eq!(
            lowered_unary_module(ArrayOperation::ExpM1(ExpM1Operation::new()), DataType::C64, vec![]).unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<complex<f32>>) -> tensor<complex<f32>> {
                    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
                    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
                    %0 = stablehlo.real %arg0 : (tensor<complex<f32>>) -> tensor<f32>
                    %1 = stablehlo.imag %arg0 : (tensor<complex<f32>>) -> tensor<f32>
                    %2 = stablehlo.exponential_minus_one %0 : tensor<f32>
                    %3 = stablehlo.cosine %1 : tensor<f32>
                    %4 = stablehlo.multiply %1, %cst_0 : tensor<f32>
                    %5 = stablehlo.sine %4 : tensor<f32>
                    %6 = stablehlo.multiply %5, %5 : tensor<f32>
                    %7 = stablehlo.multiply %cst, %6 : tensor<f32>
                    %8 = stablehlo.multiply %2, %3 : tensor<f32>
                    %9 = stablehlo.subtract %8, %7 : tensor<f32>
                    %10 = stablehlo.exponential %0 : tensor<f32>
                    %11 = stablehlo.sine %1 : tensor<f32>
                    %12 = stablehlo.multiply %10, %11 : tensor<f32>
                    %13 = stablehlo.complex %9, %12 : tensor<complex<f32>>
                    return %13 : tensor<complex<f32>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_logistic() {
        assert_eq!(
            lowered_unary_module(ArrayOperation::Logistic(LogisticOperation::new()), DataType::F64, vec![]).unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.compare GE, %arg0, %cst_0, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
                    %1 = stablehlo.negate %arg0 : tensor<f64>
                    %2 = stablehlo.select %0, %1, %arg0 : tensor<i1>, tensor<f64>
                    %3 = stablehlo.exponential %2 : tensor<f64>
                    %4 = stablehlo.add %cst, %3 : tensor<f64>
                    %5 = stablehlo.divide %cst, %4 : tensor<f64>
                    %6 = stablehlo.divide %3, %4 : tensor<f64>
                    %7 = stablehlo.select %0, %5, %6 : tensor<i1>, tensor<f64>
                    return %7 : tensor<f64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_logistic_dynamic_shape() {
        assert_eq!(
            lowered_dynamic_unary_module(ArrayOperation::Logistic(LogisticOperation::new()), DataType::F64),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<?xf64, #stablehlo.bounds<4>>) -> tensor<?xf64, #stablehlo.bounds<4>> {
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf64, #stablehlo.bounds<4>>) -> tensor<i32>
                    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
                    %2 = stablehlo.reshape %1 : (tensor<i64>) -> tensor<1xi64>
                    %3 = stablehlo.dynamic_broadcast_in_dim %cst, %2, dims = [] : (tensor<f64>, tensor<1xi64>) -> tensor<?xf64>
                    %cast = tensor.cast %3 : tensor<?xf64> to tensor<?xf64, #stablehlo.bounds<4>>
                    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %4 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf64, #stablehlo.bounds<4>>) -> tensor<i32>
                    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                    %6 = stablehlo.reshape %5 : (tensor<i64>) -> tensor<1xi64>
                    %7 = stablehlo.dynamic_broadcast_in_dim %cst_0, %6, dims = [] : (tensor<f64>, tensor<1xi64>) -> tensor<?xf64>
                    %cast_1 = tensor.cast %7 : tensor<?xf64> to tensor<?xf64, #stablehlo.bounds<4>>
                    %8 = stablehlo.compare GE, %arg0, %cast_1, FLOAT : (tensor<?xf64, #stablehlo.bounds<4>>, tensor<?xf64, #stablehlo.bounds<4>>) -> tensor<?xi1, #stablehlo.bounds<4>>
                    %9 = stablehlo.negate %arg0 : tensor<?xf64, #stablehlo.bounds<4>>
                    %10 = stablehlo.select %8, %9, %arg0 : tensor<?xi1, #stablehlo.bounds<4>>, tensor<?xf64, #stablehlo.bounds<4>>
                    %11 = stablehlo.exponential %10 : tensor<?xf64, #stablehlo.bounds<4>>
                    %12 = stablehlo.add %cast, %11 : tensor<?xf64, #stablehlo.bounds<4>>
                    %13 = stablehlo.divide %cast, %12 : tensor<?xf64, #stablehlo.bounds<4>>
                    %14 = stablehlo.divide %11, %12 : tensor<?xf64, #stablehlo.bounds<4>>
                    %15 = stablehlo.select %8, %13, %14 : tensor<?xi1, #stablehlo.bounds<4>>, tensor<?xf64, #stablehlo.bounds<4>>
                    return %15 : tensor<?xf64, #stablehlo.bounds<4>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_exp_m1_dynamic_shape() {
        assert_eq!(
            lowered_dynamic_unary_module(ArrayOperation::ExpM1(ExpM1Operation::new()), DataType::C128),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<?xcomplex<f64>, #stablehlo.bounds<4>>) -> tensor<?xcomplex<f64>, #stablehlo.bounds<4>> {
                    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
                    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xcomplex<f64>, #stablehlo.bounds<4>>) -> tensor<i32>
                    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
                    %2 = stablehlo.reshape %1 : (tensor<i64>) -> tensor<1xi64>
                    %3 = stablehlo.dynamic_broadcast_in_dim %cst, %2, dims = [] : (tensor<f64>, tensor<1xi64>) -> tensor<?xf64>
                    %cast = tensor.cast %3 : tensor<?xf64> to tensor<?xf64, #stablehlo.bounds<4>>
                    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
                    %4 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xcomplex<f64>, #stablehlo.bounds<4>>) -> tensor<i32>
                    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                    %6 = stablehlo.reshape %5 : (tensor<i64>) -> tensor<1xi64>
                    %7 = stablehlo.dynamic_broadcast_in_dim %cst_0, %6, dims = [] : (tensor<f64>, tensor<1xi64>) -> tensor<?xf64>
                    %cast_1 = tensor.cast %7 : tensor<?xf64> to tensor<?xf64, #stablehlo.bounds<4>>
                    %8 = stablehlo.real %arg0 : (tensor<?xcomplex<f64>, #stablehlo.bounds<4>>) -> tensor<?xf64, #stablehlo.bounds<4>>
                    %9 = stablehlo.imag %arg0 : (tensor<?xcomplex<f64>, #stablehlo.bounds<4>>) -> tensor<?xf64, #stablehlo.bounds<4>>
                    %10 = stablehlo.exponential_minus_one %8 : tensor<?xf64, #stablehlo.bounds<4>>
                    %11 = stablehlo.cosine %9 : tensor<?xf64, #stablehlo.bounds<4>>
                    %12 = stablehlo.multiply %9, %cast_1 : tensor<?xf64, #stablehlo.bounds<4>>
                    %13 = stablehlo.sine %12 : tensor<?xf64, #stablehlo.bounds<4>>
                    %14 = stablehlo.multiply %13, %13 : tensor<?xf64, #stablehlo.bounds<4>>
                    %15 = stablehlo.multiply %cast, %14 : tensor<?xf64, #stablehlo.bounds<4>>
                    %16 = stablehlo.multiply %10, %11 : tensor<?xf64, #stablehlo.bounds<4>>
                    %17 = stablehlo.subtract %16, %15 : tensor<?xf64, #stablehlo.bounds<4>>
                    %18 = stablehlo.exponential %8 : tensor<?xf64, #stablehlo.bounds<4>>
                    %19 = stablehlo.sine %9 : tensor<?xf64, #stablehlo.bounds<4>>
                    %20 = stablehlo.multiply %18, %19 : tensor<?xf64, #stablehlo.bounds<4>>
                    %21 = stablehlo.complex %17, %20 : tensor<?xcomplex<f64>, #stablehlo.bounds<4>>
                    return %21 : tensor<?xcomplex<f64>, #stablehlo.bounds<4>>
                  }
                }
            "},
        );
    }

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
