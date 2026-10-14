use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, CastOperation, ConvertElementTypeOperation, DataType, Dimension, Operation,
    ReducePrecisionOperation, ReductionKind, Shape,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::{
    lower_f64_constant_splat, lower_physical_bound_value, lower_restore_dynamic_dimensions,
    lower_signed_one_bit_to_mlir, lower_tensor_type, lower_unplaced_constant_output, physical_bound_type,
    static_dimensions,
};
use crate::operations::assertions::lower_dependency_barrier;
use crate::operations::control_flow::{lower_refined_region_input, lower_refined_region_outputs};
use crate::operations::reductions::lower_reduce_to_mlir;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for ConvertElementTypeOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        check_count!("input", input_types, 1, ProgramError);
        let input_type = &input_types[0];
        let mut input_data_type = input_type.data_type();
        let output_data_type = output_types[0].data_type();
        let output_type = lower_tensor_type(&output_types[0], lowerer.context, lowerer.location)?;
        let mut input = input_values[0];
        if self.bitcast() {
            return Ok(vec![lower_bitcast_to_mlir(
                input,
                input_type,
                &output_types[0],
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?]);
        }

        // StableHLO's complex-to-predicate conversion discards the imaginary part. Ryft Boolean conversion tests
        // the entire value for nonzero, including purely imaginary values and NaNs in either component.
        if input_data_type.is_complex() && output_data_type == DataType::Boolean {
            let real = lowerer.block.append_operation(stable_hlo::real(input, lowerer.location)?)?;
            let imaginary = lowerer.block.append_operation(stable_hlo::imag(input, lowerer.location)?)?;
            let real = lowerer.block.append_operation(stable_hlo::convert(
                real.result(0).unwrap().as_ref(),
                output_type,
                lowerer.location,
            )?)?;
            let imaginary = lowerer.block.append_operation(stable_hlo::convert(
                imaginary.result(0).unwrap().as_ref(),
                output_type,
                lowerer.location,
            )?)?;
            let result = lowerer.block.append_operation(stable_hlo::or(
                real.result(0).unwrap().as_ref(),
                imaginary.result(0).unwrap().as_ref(),
                lowerer.location,
            )?)?;
            return Ok(vec![result.result(0).unwrap().as_ref()]);
        }

        // Floating destinations use the real component, retaining its precision for the rounding adjustment below.
        if input_data_type.is_complex() && output_data_type.is_floating_point() {
            let real = lowerer.block.append_operation(stable_hlo::real(input, lowerer.location)?)?;
            input = real.result(0).unwrap().as_ref();
            input_data_type = if input_data_type == DataType::C128 { DataType::F64 } else { DataType::F32 };
        }

        // The shared i1 carrier has predicate semantics in StableHLO. Recover the signed numeric value before
        // widening an I1, while preserving its bit directly when converting between one-bit carriers.
        if matches!(input_data_type, DataType::Boolean | DataType::I1 | DataType::U1)
            && matches!(output_data_type, DataType::Boolean | DataType::I1 | DataType::U1)
        {
            return Ok(vec![input]);
        }
        if input_data_type == DataType::I1 {
            input = lower_signed_one_bit_to_mlir(input, &mut lowerer.block, lowerer.context, lowerer.location)?;
        }
        if matches!(
            output_data_type,
            DataType::I1 | DataType::I2 | DataType::I4 | DataType::U1 | DataType::U2 | DataType::U4
        ) {
            // Integer narrowing retains low bits. Real inputs instead saturate to the logical integer range;
            // using a byte carrier first also gives NaNs the integer conversion's zero value. The final mask
            // preserves one-bit signed encodings rather than invoking predicate truthiness.
            let carrier_data_type = if matches!(output_data_type, DataType::I1 | DataType::I2 | DataType::I4) {
                DataType::I8
            } else {
                DataType::U8
            };
            let carrier_type = ArrayType::new(carrier_data_type, input_type.shape().clone());
            let carrier_tensor_type = lower_tensor_type(&carrier_type, lowerer.context, lowerer.location)?;
            let converted =
                lowerer.block.append_operation(stable_hlo::convert(input, carrier_tensor_type, lowerer.location)?)?;
            let mut converted = converted.result(0).unwrap().as_ref();
            if input_data_type.is_floating_point() || input_data_type.is_complex() {
                let (minimum, maximum) = match output_data_type {
                    DataType::I1 => (-1, 0),
                    DataType::I2 => (-2, 1),
                    DataType::I4 => (-8, 7),
                    DataType::U1 => (0, 1),
                    DataType::U2 => (0, 3),
                    _ => (0, 15),
                };
                let minimum = lower_unplaced_constant_output(
                    &[carrier_type.clone()],
                    minimum,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?[0];
                let maximum = lower_unplaced_constant_output(
                    &[carrier_type.clone()],
                    maximum,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?[0];
                let clamped = lowerer.block.append_operation(stable_hlo::clamp(
                    minimum,
                    converted,
                    maximum,
                    lowerer.location,
                )?)?;
                converted = clamped.result(0).unwrap().as_ref();
            }
            let mask = lower_unplaced_constant_output(
                &[carrier_type],
                match output_data_type {
                    DataType::I1 | DataType::U1 => 1,
                    DataType::I2 | DataType::U2 => 3,
                    _ => 15,
                },
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?[0];
            let masked = lowerer.block.append_operation(stable_hlo::and(converted, mask, lowerer.location)?)?;
            input = masked.result(0).unwrap().as_ref();
        }
        if (matches!(input_data_type, DataType::I32 | DataType::U32 | DataType::I64 | DataType::U64)
            && matches!(output_data_type, DataType::BF16 | DataType::F8E8M0FNU))
            || (input_data_type == DataType::F64
                && output_data_type.is_floating_point()
                && !matches!(output_data_type, DataType::F32 | DataType::F64))
        {
            input = lower_conversion_round_to_odd(
                input,
                &input_type.clone().with_data_type(input_data_type),
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
        }
        if matches!(output_data_type, DataType::F4E2M1FN | DataType::F6E2M3FN | DataType::F6E3M2FN) {
            // These formats have no NaN or infinity encodings. Normalize exceptional values and clamp finite
            // overflow before native conversion, whose handling otherwise differs across CPU and GPU backends.
            let float_type = input_type.clone().with_data_type(DataType::F32);
            let float_tensor_type = lower_tensor_type(&float_type, lowerer.context, lowerer.location)?;
            let converted =
                lowerer.block.append_operation(stable_hlo::convert(input, float_tensor_type, lowerer.location)?)?;
            input = converted.result(0).unwrap().as_ref();
            let maximum = match output_data_type {
                DataType::F4E2M1FN => 6.0,
                DataType::F6E2M3FN => 7.5,
                _ => 28.0,
            };
            let minimum = lower_f64_constant_splat(
                -maximum,
                &float_type,
                float_tensor_type,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let maximum = lower_f64_constant_splat(
                maximum,
                &float_type,
                float_tensor_type,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let is_nan = lowerer.block.append_operation(stable_hlo::compare(
                input,
                input,
                stable_hlo::ComparisonDirection::NotEqual,
                stable_hlo::ComparisonType::Float,
                lowerer.location,
            )?)?;
            let clamped =
                lowerer.block.append_operation(stable_hlo::clamp(minimum, input, maximum, lowerer.location)?)?;
            let normalized = lowerer.block.append_operation(stable_hlo::select(
                is_nan.result(0).unwrap().as_ref(),
                maximum,
                clamped.result(0).unwrap().as_ref(),
                lowerer.location,
            )?)?;
            input = normalized.result(0).unwrap().as_ref();
        }
        let result = lowerer.block.append_operation(stable_hlo::convert(input, output_type, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.convert should return one result").as_ref()])
    }
}

impl LowerableOperation for ReducePrecisionOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 1, ProgramError);
        let result = lowerer.block.append_operation(stable_hlo::reduce_precision(
            input_values[0],
            self.exponent_bits(),
            self.mantissa_bits(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for CastOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("output", output_types, 1, ProgramError);
        let (Some(_), Some(input_type)) = (input_values.first(), input_types.first()) else {
            return Err(LoweringError::UnsupportedOp { op: format!("`{}` expects at least 1 input", self.name()) });
        };
        // The cast only converts the first input, but its proof witnesses and preceding assertion must remain real
        // dependencies. This also orders tangent conversions whose witnesses are unconverted primal inputs.
        let values = lower_dependency_barrier(
            input_values,
            lowerer.effect_tokens.ordered_assertion,
            &mut lowerer.block,
            lowerer.location,
        )?;
        Ok(vec![lower_cast(
            values[0],
            &input_type.clone().into(),
            &output_types[0].clone().into(),
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?])
    }
}

impl LowerableOperation for CastOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("output", output_types, 1, ProgramError);
        let (Some(_), Some(input_type)) = (input_values.first(), input_types.first()) else {
            return Err(LoweringError::UnsupportedOp { op: format!("`{}` expects at least 1 input", self.name()) });
        };
        // Witness values can carry a preceding check even when their geometry is already represented by the types.
        // Keep those producers, and the current assertion token, ahead of the proof-carrying conversion.
        let values = lower_dependency_barrier(
            input_values,
            lowerer.effect_tokens.ordered_assertion,
            &mut lowerer.block,
            lowerer.location,
        )?;
        Ok(vec![lower_cast(
            values[0],
            input_type,
            &output_types[0],
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?])
    }
}

/// Converts the lowered `value` of type `input_type` to the lowered representation of the compatible `output_type`
/// (refer to [`CastOperation`] and [`GuardOperation`](ryft_core::GuardOperation)). First-class dimensions lower to `i64` scalars whatever
/// their type, and types that differ only in metadata lower to the same tensor type, so those values are returned
/// unchanged. The value is converted through the type whose axes are static wherever either type is static: narrowing
/// a bounded dynamic axis to a static extent slices the physical buffer (refer to [`lower_refined_region_outputs`]),
/// and then widening a static axis to a bounded dynamic one, or moving a dynamic axis to another identity with another
/// physical bound, pads or slices it and restores its runtime size (refer to [`lower_refined_region_input`]). Neither
/// conversion checks runtime extents, which is why ryft-core emits them only behind a check such as an entry guard.
pub(crate) fn lower_cast<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    input_type: &ArrayIrType,
    output_type: &ArrayIrType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let intermediate_type = match (input_type, output_type) {
        (ArrayIrType::Array(input), ArrayIrType::Array(output)) if input.rank() == output.rank() => {
            let dimensions = input
                .shape()
                .dimensions()
                .iter()
                .zip(output.shape().dimensions())
                .map(|(input, output)| match (input, output) {
                    (Dimension::Dynamic(_), Dimension::Static(_)) => output.clone(),
                    _ => input.clone(),
                })
                .collect();
            ArrayIrType::Array(input.clone().with_shape(Shape::new(dimensions)))
        }
        _ => input_type.clone(),
    };
    let value = lower_refined_region_outputs(
        vec![value],
        std::slice::from_ref(input_type),
        std::slice::from_ref(&intermediate_type),
        block,
        context,
        location,
    )?
    .remove(0);
    lower_refined_region_input(value, &intermediate_type, output_type, block, context, location)
}

/// Preserves rounding information in an intermediate `f32` before conversion to a lower-precision float.
/// An inexact intermediate is truncated toward zero and its low significand bit is set, preventing a second
/// nearest-even rounding from mistaking a value adjacent to a destination midpoint for the midpoint itself.
fn lower_conversion_round_to_odd<'b, 'c: 'b, 't: 'c>(
    input: ValueRef<'b, 'c, 't>,
    input_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let float_type = lower_tensor_type(&input_type.clone().with_data_type(DataType::F32), context, location)?;
    let float = block.append_operation(stable_hlo::convert(input, float_type, location)?)?;
    let float = float.result(0).unwrap().as_ref();
    // The intermediate rounding is observable here. Prevent excess-precision optimizations from cancelling
    // the narrowing/widening pair and incorrectly proving that the original conversion was exact.
    let float = block.append_operation(stable_hlo::optimization_barrier(&[float], location)?)?;
    let float = float.result(0).unwrap().as_ref();
    let recovered = block.append_operation(stable_hlo::convert(
        float,
        lower_tensor_type(input_type, context, location)?,
        location,
    )?)?;
    let recovered = recovered.result(0).unwrap().as_ref();
    let comparison_type = if input_type.data_type().is_floating_point() {
        stable_hlo::ComparisonType::Float
    } else if input_type.data_type().is_signed() {
        stable_hlo::ComparisonType::Signed
    } else {
        stable_hlo::ComparisonType::Unsigned
    };
    let inexact = block.append_operation(stable_hlo::compare(
        input,
        recovered,
        stable_hlo::ComparisonDirection::NotEqual,
        comparison_type,
        location,
    )?)?;
    let inexact = inexact.result(0).unwrap().as_ref();
    let increased = block.append_operation(stable_hlo::compare(
        recovered,
        input,
        stable_hlo::ComparisonDirection::GreaterThan,
        comparison_type,
        location,
    )?)?;
    let mut increased = increased.result(0).unwrap().as_ref();
    if input_type.data_type().is_signed() || input_type.data_type().is_floating_point() {
        let zero = lower_unplaced_constant_output(&[input_type.clone()], 0, block, context, location)?[0];
        let negative = block.append_operation(stable_hlo::compare(
            input,
            zero,
            stable_hlo::ComparisonDirection::LessThan,
            comparison_type,
            location,
        )?)?;
        let decreased = block.append_operation(stable_hlo::compare(
            recovered,
            input,
            stable_hlo::ComparisonDirection::LessThan,
            comparison_type,
            location,
        )?)?;
        let direction = block.append_operation(stable_hlo::select(
            negative.result(0).unwrap().as_ref(),
            decreased.result(0).unwrap().as_ref(),
            increased,
            location,
        )?)?;
        increased = direction.result(0).unwrap().as_ref();
    }
    let bits_type = input_type.clone().with_data_type(DataType::U32);
    let bits = block.append_operation(stable_hlo::bitcast_convert(
        float,
        lower_tensor_type(&bits_type, context, location)?,
        location,
    )?)?;
    let bits = bits.result(0).unwrap().as_ref();
    let one = lower_unplaced_constant_output(&[bits_type], 1, block, context, location)?[0];
    let previous = block.append_operation(stable_hlo::subtract(bits, one, location)?)?;
    let truncated =
        block.append_operation(stable_hlo::select(increased, previous.result(0).unwrap().as_ref(), bits, location)?)?;
    let odd = block.append_operation(stable_hlo::or(truncated.result(0).unwrap().as_ref(), one, location)?)?;
    let rounded =
        block.append_operation(stable_hlo::select(inexact, odd.result(0).unwrap().as_ref(), bits, location)?)?;
    let mut rounded = rounded.result(0).unwrap().as_ref();
    if input_type.data_type() == DataType::F64 {
        // CPU floating-point modes may flush subnormal f32 results during the initial narrowing. Construct these
        // encodings directly from f64 bits instead. Shift the exact significand to f32 subnormal precision and
        // jam every discarded nonzero bit into its low bit, without any floating-point arithmetic.
        let wide_type = input_type.clone().with_data_type(DataType::U64);
        let wide_tensor_type = lower_tensor_type(&wide_type, context, location)?;
        let source_bits = block.append_operation(stable_hlo::bitcast_convert(input, wide_tensor_type, location)?)?;
        let source_bits = source_bits.result(0).unwrap().as_ref();
        let magnitude_mask =
            lower_unplaced_constant_output(&[wide_type.clone()], i64::MAX, block, context, location)?[0];
        let exponent_shift = lower_unplaced_constant_output(&[wide_type.clone()], 52, block, context, location)?[0];
        // The smallest normal f32 has f64 biased exponent 1023 - 126 = 897. Its subnormal unit is 2^-149,
        // so an f64 significand at biased exponent `e` needs a right shift of 1023 + 52 - 149 - e = 926 - e.
        let smallest_normal_exponent =
            lower_unplaced_constant_output(&[wide_type.clone()], 897, block, context, location)?[0];
        let shift_offset = lower_unplaced_constant_output(&[wide_type.clone()], 926, block, context, location)?[0];
        let mantissa_mask =
            lower_unplaced_constant_output(&[wide_type.clone()], (1_i64 << 52) - 1, block, context, location)?[0];
        let implicit_bit =
            lower_unplaced_constant_output(&[wide_type.clone()], 1_i64 << 52, block, context, location)?[0];
        let wide_one = lower_unplaced_constant_output(&[wide_type.clone()], 1, block, context, location)?[0];
        let wide_zero = lower_unplaced_constant_output(&[wide_type.clone()], 0, block, context, location)?[0];
        let sign_shift = lower_unplaced_constant_output(&[wide_type.clone()], 32, block, context, location)?[0];
        let sign_mask = lower_unplaced_constant_output(&[wide_type.clone()], 1_i64 << 31, block, context, location)?[0];
        let magnitude = block.append_operation(stable_hlo::and(source_bits, magnitude_mask, location)?)?;
        let magnitude = magnitude.result(0).unwrap().as_ref();
        let exponent = block.append_operation(stable_hlo::shift_right_logical(magnitude, exponent_shift, location)?)?;
        let exponent = exponent.result(0).unwrap().as_ref();
        let mantissa = block.append_operation(stable_hlo::and(magnitude, mantissa_mask, location)?)?;
        let mantissa = mantissa.result(0).unwrap().as_ref();
        let significand = block.append_operation(stable_hlo::or(mantissa, implicit_bit, location)?)?;
        let significand = significand.result(0).unwrap().as_ref();
        let shift = block.append_operation(stable_hlo::subtract(shift_offset, exponent, location)?)?;
        let shift = shift.result(0).unwrap().as_ref();
        let truncated = block.append_operation(stable_hlo::shift_right_logical(significand, shift, location)?)?;
        let truncated = truncated.result(0).unwrap().as_ref();
        let discarded_mask = block.append_operation(stable_hlo::shift_left(wide_one, shift, location)?)?;
        let discarded_mask = discarded_mask.result(0).unwrap().as_ref();
        let discarded_mask = block.append_operation(stable_hlo::subtract(discarded_mask, wide_one, location)?)?;
        let discarded_mask = discarded_mask.result(0).unwrap().as_ref();
        let discarded = block.append_operation(stable_hlo::and(significand, discarded_mask, location)?)?;
        let discarded = discarded.result(0).unwrap().as_ref();
        let sticky = block.append_operation(stable_hlo::compare(
            discarded,
            wide_zero,
            stable_hlo::ComparisonDirection::NotEqual,
            stable_hlo::ComparisonType::Unsigned,
            location,
        )?)?;
        let sticky = sticky.result(0).unwrap().as_ref();
        let sticky = block.append_operation(stable_hlo::convert(sticky, wide_tensor_type, location)?)?;
        let sticky = sticky.result(0).unwrap().as_ref();
        let subnormal = block.append_operation(stable_hlo::or(truncated, sticky, location)?)?;
        let subnormal = subnormal.result(0).unwrap().as_ref();
        // A shift beyond 63 bits gives zero and an all-ones discarded mask, so every nonzero tiny f64 jams to
        // the smallest f32 subnormal. Clear that sticky bit only for an exact signed zero.
        let is_zero = block.append_operation(stable_hlo::compare(
            magnitude,
            wide_zero,
            stable_hlo::ComparisonDirection::Equal,
            stable_hlo::ComparisonType::Unsigned,
            location,
        )?)?;
        let is_zero = is_zero.result(0).unwrap().as_ref();
        let subnormal = block.append_operation(stable_hlo::select(is_zero, wide_zero, subnormal, location)?)?;
        let subnormal = subnormal.result(0).unwrap().as_ref();
        let sign = block.append_operation(stable_hlo::shift_right_logical(source_bits, sign_shift, location)?)?;
        let sign = sign.result(0).unwrap().as_ref();
        let sign = block.append_operation(stable_hlo::and(sign, sign_mask, location)?)?;
        let sign = sign.result(0).unwrap().as_ref();
        let subnormal = block.append_operation(stable_hlo::or(subnormal, sign, location)?)?;
        let subnormal = subnormal.result(0).unwrap().as_ref();
        let subnormal = block.append_operation(stable_hlo::convert(
            subnormal,
            lower_tensor_type(&input_type.clone().with_data_type(DataType::U32), context, location)?,
            location,
        )?)?;
        let subnormal = subnormal.result(0).unwrap().as_ref();
        let is_subnormal = block.append_operation(stable_hlo::compare(
            exponent,
            smallest_normal_exponent,
            stable_hlo::ComparisonDirection::LessThan,
            stable_hlo::ComparisonType::Unsigned,
            location,
        )?)?;
        let is_subnormal = is_subnormal.result(0).unwrap().as_ref();
        let corrected = block.append_operation(stable_hlo::select(is_subnormal, subnormal, rounded, location)?)?;
        let corrected = corrected.result(0).unwrap().as_ref();
        rounded = corrected;
    }
    // Saturating cast-back cannot detect the inexact representation of the largest positive integer. Its rounded
    // power of two is nevertheless already the correct result for both destination formats, far from a midpoint.
    let result = block.append_operation(stable_hlo::bitcast_convert(rounded, float_type, location)?)?;
    Ok(result.result(0).unwrap().as_ref())
}

/// Reinterprets physical encoding bits, including logical one-bit integers whose XLA predicate carrier occupies a
/// full byte. Dynamic prefix extents are restored after operating on fixed physical bounds because XLA's padding
/// pass does not support width-changing bitcasts of dynamic tensors.
fn lower_bitcast_to_mlir<'b, 'c: 'b, 't: 'c>(
    input: ValueRef<'b, 'c, 't>,
    input_type: &ArrayType,
    output_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let input_one_bit = matches!(input_type.data_type(), DataType::I1 | DataType::U1);
    let output_one_bit = matches!(output_type.data_type(), DataType::I1 | DataType::U1);
    if input_type.data_type() == output_type.data_type() || input_one_bit && output_one_bit {
        return Ok(input);
    }
    // Equal-width bitcasts preserve every dimension and can operate directly on unbounded dynamic tensors.
    // Type inference adds or removes one trailing axis for every width-changing bitcast.
    if input_type.rank() == output_type.rank() {
        let result = block.append_operation(stable_hlo::bitcast_convert(
            input,
            lower_tensor_type(output_type, context, location)?,
            location,
        )?)?;
        return Ok(result.result(0).unwrap().as_ref());
    }
    // Width-changing bitcasts need physical bounds for XLA's dynamic padding pass.
    let physical_input_type = physical_bound_type(input_type).map_err(|_| LoweringError::UnsupportedOp {
        op: "width-changing bitcast with an unbounded dynamic dimension".to_string(),
    })?;
    let padding = if input_type.data_type() == DataType::F8E8M0FNU { 1.0 } else { 0.0 };
    let carrier_input_type =
        if input_one_bit { input_type.clone().with_data_type(DataType::Boolean) } else { input_type.clone() };
    let physical_input = lower_physical_bound_value(input, &carrier_input_type, padding, block, context, location)?;
    let physical_output_type = physical_bound_type(output_type)?;
    let output_tensor_type = lower_tensor_type(&physical_output_type, context, location)?;
    let result = if matches!(input_type.data_type(), DataType::F6E2M3FN | DataType::F6E3M2FN) {
        // XLA stores FP6 in a byte and native rank-changing bitcasts expose eight bits. Recover the six logical
        // encoding bits explicitly before splitting them into one- or two-bit groups.
        let bits = lower_fp6_to_bits(physical_input, &physical_input_type, block, context, location)?;
        let group_width = if output_one_bit { 1 } else { 2 };
        let expanded_type = physical_output_type.clone().with_data_type(DataType::U8);
        let expanded_tensor_type = lower_tensor_type(&expanded_type, context, location)?;
        let expanded = block.append_operation(stable_hlo::broadcast(
            bits,
            expanded_tensor_type,
            &(0..input_type.rank()).collect::<Vec<_>>(),
            location,
        )?)?;
        let offsets =
            block.append_operation(stable_hlo::iota(expanded_tensor_type, output_type.rank() - 1, location)?)?;
        let width = lower_unplaced_constant_output(&[expanded_type.clone()], group_width, block, context, location)?[0];
        let offsets =
            block.append_operation(stable_hlo::multiply(offsets.result(0).unwrap().as_ref(), width, location)?)?;
        let shifted = block.append_operation(stable_hlo::shift_right_logical(
            expanded.result(0).unwrap().as_ref(),
            offsets.result(0).unwrap().as_ref(),
            location,
        )?)?;
        let mask =
            lower_unplaced_constant_output(&[expanded_type], (1 << group_width) - 1, block, context, location)?[0];
        let groups = block.append_operation(stable_hlo::and(shifted.result(0).unwrap().as_ref(), mask, location)?)?;
        let result = block.append_operation(stable_hlo::convert(
            groups.result(0).unwrap().as_ref(),
            output_tensor_type,
            location,
        )?)?;
        result.result(0).unwrap().as_ref()
    } else if matches!(output_type.data_type(), DataType::F6E2M3FN | DataType::F6E3M2FN) {
        let group_width = if input_one_bit { 1 } else { 2 };
        let groups_type = physical_input_type.clone().with_data_type(DataType::U8);
        let groups_tensor_type = lower_tensor_type(&groups_type, context, location)?;
        let groups = block.append_operation(stable_hlo::convert(physical_input, groups_tensor_type, location)?)?;
        let mask =
            lower_unplaced_constant_output(&[groups_type.clone()], (1 << group_width) - 1, block, context, location)?
                [0];
        let groups = block.append_operation(stable_hlo::and(groups.result(0).unwrap().as_ref(), mask, location)?)?;
        let offsets = block.append_operation(stable_hlo::iota(groups_tensor_type, input_type.rank() - 1, location)?)?;
        let width = lower_unplaced_constant_output(&[groups_type], group_width, block, context, location)?[0];
        let offsets =
            block.append_operation(stable_hlo::multiply(offsets.result(0).unwrap().as_ref(), width, location)?)?;
        let shifted = block.append_operation(stable_hlo::shift_left(
            groups.result(0).unwrap().as_ref(),
            offsets.result(0).unwrap().as_ref(),
            location,
        )?)?;
        let bits = lower_reduce_to_mlir(
            ReductionKind::Sum,
            &[input_type.rank() - 1],
            shifted.result(0).unwrap().as_ref(),
            &physical_output_type.clone().with_data_type(DataType::U8),
            block,
            context,
            location,
        )?;
        lower_bits_to_fp6(bits, &physical_output_type, block, context, location)?
    } else if output_one_bit {
        // Split into two-bit groups first: every supported non-one-bit numeric encoding has an even width, including
        // six-bit floats. Expanding each group to two predicates avoids treating each predicate as eight bits.
        let mut shape = static_dimensions(&physical_output_type)?;
        let width = shape.pop().unwrap();
        if width > 2 {
            shape.push(width / 2);
        }
        let groups_type =
            ArrayType::new(DataType::U2, Shape::new(shape.iter().copied().map(Dimension::Static).collect()));
        let groups = block.append_operation(stable_hlo::bitcast_convert(
            physical_input,
            lower_tensor_type(&groups_type, context, location)?,
            location,
        )?)?;
        let groups = block.append_operation(stable_hlo::convert(
            groups.result(0).unwrap().as_ref(),
            lower_tensor_type(&groups_type.clone().with_data_type(DataType::U8), context, location)?,
            location,
        )?)?;
        let group_rank = shape.len();
        shape.push(2);
        let expanded_type =
            ArrayType::new(DataType::U8, Shape::new(shape.iter().copied().map(Dimension::Static).collect()));
        let expanded_tensor_type = lower_tensor_type(&expanded_type, context, location)?;
        let expanded = block.append_operation(stable_hlo::broadcast(
            groups.result(0).unwrap().as_ref(),
            expanded_tensor_type,
            &(0..group_rank).collect::<Vec<_>>(),
            location,
        )?)?;
        let offsets = block.append_operation(stable_hlo::iota(expanded_tensor_type, group_rank, location)?)?;
        let shifted = block.append_operation(stable_hlo::shift_right_logical(
            expanded.result(0).unwrap().as_ref(),
            offsets.result(0).unwrap().as_ref(),
            location,
        )?)?;
        let one = lower_unplaced_constant_output(&[expanded_type.clone()], 1, block, context, location)?[0];
        let bits = block.append_operation(stable_hlo::and(shifted.result(0).unwrap().as_ref(), one, location)?)?;
        let bits = block.append_operation(stable_hlo::convert(
            bits.result(0).unwrap().as_ref(),
            lower_tensor_type(&expanded_type.with_data_type(DataType::Boolean), context, location)?,
            location,
        )?)?;
        let result = block.append_operation(stable_hlo::reshape(
            bits.result(0).unwrap().as_ref(),
            &static_dimensions(&physical_output_type)?,
            location,
        )?)?;
        result.result(0).unwrap().as_ref()
    } else if input_one_bit {
        // Pack neighboring predicates into two-bit groups with disjoint shifts; their sum is their bitwise union.
        let mut shape = static_dimensions(&physical_input_type)?;
        let width = shape.pop().unwrap();
        if width > 2 {
            shape.push(width / 2);
        }
        shape.push(2);
        let reshaped = block.append_operation(stable_hlo::reshape(physical_input, &shape, location)?)?;
        let expanded_type =
            ArrayType::new(DataType::U8, Shape::new(shape.iter().copied().map(Dimension::Static).collect()));
        let expanded_tensor_type = lower_tensor_type(&expanded_type, context, location)?;
        let expanded = block.append_operation(stable_hlo::convert(
            reshaped.result(0).unwrap().as_ref(),
            expanded_tensor_type,
            location,
        )?)?;
        let bit_axis = shape.len() - 1;
        let offsets = block.append_operation(stable_hlo::iota(expanded_tensor_type, bit_axis, location)?)?;
        let shifted = block.append_operation(stable_hlo::shift_left(
            expanded.result(0).unwrap().as_ref(),
            offsets.result(0).unwrap().as_ref(),
            location,
        )?)?;
        shape.pop();
        let groups_type = ArrayType::new(DataType::U8, Shape::new(shape.into_iter().map(Dimension::Static).collect()));
        let groups = lower_reduce_to_mlir(
            ReductionKind::Sum,
            &[bit_axis],
            shifted.result(0).unwrap().as_ref(),
            &groups_type,
            block,
            context,
            location,
        )?;
        let groups = block.append_operation(stable_hlo::convert(
            groups,
            lower_tensor_type(&groups_type.with_data_type(DataType::U2), context, location)?,
            location,
        )?)?;
        let result = block.append_operation(stable_hlo::bitcast_convert(
            groups.result(0).unwrap().as_ref(),
            output_tensor_type,
            location,
        )?)?;
        result.result(0).unwrap().as_ref()
    } else {
        let result =
            block.append_operation(stable_hlo::bitcast_convert(physical_input, output_tensor_type, location)?)?;
        result.result(0).unwrap().as_ref()
    };
    let sources = (0..output_type.rank()).map(|axis| (input, axis)).collect::<Vec<_>>();
    lower_restore_dynamic_dimensions(result, output_type, &sources, block, context, location)
}

/// Extracts a finite FP6 encoding through its exact `f32` representation, preserving the sign of zero.
fn lower_fp6_to_bits<'b, 'c: 'b, 't: 'c>(
    input: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let (mantissa_bits, bias) = if r#type.data_type() == DataType::F6E2M3FN { (3, 1) } else { (2, 3) };
    let integer_type = r#type.clone().with_data_type(DataType::U32);
    let integer_tensor_type = lower_tensor_type(&integer_type, context, location)?;
    let float_type = r#type.clone().with_data_type(DataType::F32);
    let float_tensor_type = lower_tensor_type(&float_type, context, location)?;
    // The native dependency patches the finite-only FP6 NaN predicate, making this exact widening conversion valid.
    let float = block.append_operation(stable_hlo::convert(input, float_tensor_type, location)?)?;
    let float = float.result(0).unwrap().as_ref();
    let bits = block.append_operation(stable_hlo::bitcast_convert(float, integer_tensor_type, location)?)?;
    let bits = bits.result(0).unwrap().as_ref();
    let sign_shift = lower_unplaced_constant_output(&[integer_type.clone()], 26, block, context, location)?[0];
    let sign_mask = lower_unplaced_constant_output(&[integer_type.clone()], 32, block, context, location)?[0];
    let exponent_shift = lower_unplaced_constant_output(&[integer_type.clone()], 23, block, context, location)?[0];
    let exponent_mask = lower_unplaced_constant_output(&[integer_type.clone()], 255, block, context, location)?[0];
    let mantissa_shift =
        lower_unplaced_constant_output(&[integer_type.clone()], 23 - mantissa_bits, block, context, location)?[0];
    let mantissa_mask =
        lower_unplaced_constant_output(&[integer_type.clone()], (1 << mantissa_bits) - 1, block, context, location)?[0];
    let bias_offset = lower_unplaced_constant_output(&[integer_type.clone()], 127 - bias, block, context, location)?[0];
    let smallest_normal_exponent =
        lower_unplaced_constant_output(&[integer_type.clone()], 128 - bias, block, context, location)?[0];
    let encoding_shift =
        lower_unplaced_constant_output(&[integer_type.clone()], mantissa_bits, block, context, location)?[0];
    let sign = block.append_operation(stable_hlo::shift_right_logical(bits, sign_shift, location)?)?;
    let sign = sign.result(0).unwrap().as_ref();
    let sign = block.append_operation(stable_hlo::and(sign, sign_mask, location)?)?;
    let sign = sign.result(0).unwrap().as_ref();
    let exponent = block.append_operation(stable_hlo::shift_right_logical(bits, exponent_shift, location)?)?;
    let exponent = exponent.result(0).unwrap().as_ref();
    let exponent = block.append_operation(stable_hlo::and(exponent, exponent_mask, location)?)?;
    let exponent = exponent.result(0).unwrap().as_ref();
    let mantissa = block.append_operation(stable_hlo::shift_right_logical(bits, mantissa_shift, location)?)?;
    let mantissa = mantissa.result(0).unwrap().as_ref();
    let mantissa = block.append_operation(stable_hlo::and(mantissa, mantissa_mask, location)?)?;
    let mantissa = mantissa.result(0).unwrap().as_ref();
    let biased = block.append_operation(stable_hlo::subtract(exponent, bias_offset, location)?)?;
    let biased = biased.result(0).unwrap().as_ref();
    let biased = block.append_operation(stable_hlo::shift_left(biased, encoding_shift, location)?)?;
    let biased = biased.result(0).unwrap().as_ref();
    let normal = block.append_operation(stable_hlo::or(biased, mantissa, location)?)?;
    let normal = normal.result(0).unwrap().as_ref();
    let scale = lower_f64_constant_splat(
        2.0_f64.powi((bias + mantissa_bits - 1) as i32),
        &float_type,
        float_tensor_type,
        block,
        context,
        location,
    )?;
    let magnitude = block.append_operation(stable_hlo::abs(float, location)?)?;
    let magnitude = magnitude.result(0).unwrap().as_ref();
    let scaled = block.append_operation(stable_hlo::multiply(magnitude, scale, location)?)?;
    let scaled = scaled.result(0).unwrap().as_ref();
    let subnormal = block.append_operation(stable_hlo::convert(scaled, integer_tensor_type, location)?)?;
    let subnormal = subnormal.result(0).unwrap().as_ref();
    let is_subnormal = block.append_operation(stable_hlo::compare(
        exponent,
        smallest_normal_exponent,
        stable_hlo::ComparisonDirection::LessThan,
        stable_hlo::ComparisonType::Unsigned,
        location,
    )?)?;
    let is_subnormal = is_subnormal.result(0).unwrap().as_ref();
    let encoding = block.append_operation(stable_hlo::select(is_subnormal, subnormal, normal, location)?)?;
    let encoding = encoding.result(0).unwrap().as_ref();
    let encoding = block.append_operation(stable_hlo::or(encoding, sign, location)?)?;
    let encoding = encoding.result(0).unwrap().as_ref();
    let result = block.append_operation(stable_hlo::convert(
        encoding,
        lower_tensor_type(&r#type.clone().with_data_type(DataType::U8), context, location)?,
        location,
    )?)?;
    let result = result.result(0).unwrap().as_ref();
    Ok(result)
}

/// Decodes six packed logical bits into an exactly representable `f32` value before constructing the FP6 result.
fn lower_bits_to_fp6<'b, 'c: 'b, 't: 'c>(
    input: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let (mantissa_bits, bias) = if r#type.data_type() == DataType::F6E2M3FN { (3, 1) } else { (2, 3) };
    let integer_type = r#type.clone().with_data_type(DataType::U32);
    let integer_tensor_type = lower_tensor_type(&integer_type, context, location)?;
    let float_type = r#type.clone().with_data_type(DataType::F32);
    let float_tensor_type = lower_tensor_type(&float_type, context, location)?;
    let bits = block.append_operation(stable_hlo::convert(input, integer_tensor_type, location)?)?;
    let bits = bits.result(0).unwrap().as_ref();
    let sign_mask = lower_unplaced_constant_output(&[integer_type.clone()], 32, block, context, location)?[0];
    let sign_shift = lower_unplaced_constant_output(&[integer_type.clone()], 26, block, context, location)?[0];
    let encoding_shift =
        lower_unplaced_constant_output(&[integer_type.clone()], mantissa_bits, block, context, location)?[0];
    let exponent_mask = lower_unplaced_constant_output(
        &[integer_type.clone()],
        (1 << (5 - mantissa_bits)) - 1,
        block,
        context,
        location,
    )?[0];
    let mantissa_mask =
        lower_unplaced_constant_output(&[integer_type.clone()], (1 << mantissa_bits) - 1, block, context, location)?[0];
    let bias_offset = lower_unplaced_constant_output(&[integer_type.clone()], 127 - bias, block, context, location)?[0];
    let exponent_shift = lower_unplaced_constant_output(&[integer_type.clone()], 23, block, context, location)?[0];
    let mantissa_shift =
        lower_unplaced_constant_output(&[integer_type.clone()], 23 - mantissa_bits, block, context, location)?[0];
    let zero = lower_unplaced_constant_output(&[integer_type.clone()], 0, block, context, location)?[0];
    let sign = block.append_operation(stable_hlo::and(bits, sign_mask, location)?)?;
    let sign = sign.result(0).unwrap().as_ref();
    let sign = block.append_operation(stable_hlo::shift_left(sign, sign_shift, location)?)?;
    let sign = sign.result(0).unwrap().as_ref();
    let exponent = block.append_operation(stable_hlo::shift_right_logical(bits, encoding_shift, location)?)?;
    let exponent = exponent.result(0).unwrap().as_ref();
    let exponent = block.append_operation(stable_hlo::and(exponent, exponent_mask, location)?)?;
    let exponent = exponent.result(0).unwrap().as_ref();
    let mantissa = block.append_operation(stable_hlo::and(bits, mantissa_mask, location)?)?;
    let mantissa = mantissa.result(0).unwrap().as_ref();
    let biased = block.append_operation(stable_hlo::add(exponent, bias_offset, location)?)?;
    let biased = biased.result(0).unwrap().as_ref();
    let biased = block.append_operation(stable_hlo::shift_left(biased, exponent_shift, location)?)?;
    let biased = biased.result(0).unwrap().as_ref();
    let normal_mantissa = block.append_operation(stable_hlo::shift_left(mantissa, mantissa_shift, location)?)?;
    let normal_mantissa = normal_mantissa.result(0).unwrap().as_ref();
    let normal = block.append_operation(stable_hlo::or(biased, normal_mantissa, location)?)?;
    let normal = normal.result(0).unwrap().as_ref();
    let scale = lower_f64_constant_splat(
        2.0_f64.powi((1 - bias - mantissa_bits) as i32),
        &float_type,
        float_tensor_type,
        block,
        context,
        location,
    )?;
    let subnormal = block.append_operation(stable_hlo::convert(mantissa, float_tensor_type, location)?)?;
    let subnormal = subnormal.result(0).unwrap().as_ref();
    let subnormal = block.append_operation(stable_hlo::multiply(subnormal, scale, location)?)?;
    let subnormal = subnormal.result(0).unwrap().as_ref();
    let subnormal = block.append_operation(stable_hlo::bitcast_convert(subnormal, integer_tensor_type, location)?)?;
    let subnormal = subnormal.result(0).unwrap().as_ref();
    let is_subnormal = block.append_operation(stable_hlo::compare(
        exponent,
        zero,
        stable_hlo::ComparisonDirection::Equal,
        stable_hlo::ComparisonType::Unsigned,
        location,
    )?)?;
    let is_subnormal = is_subnormal.result(0).unwrap().as_ref();
    let encoding = block.append_operation(stable_hlo::select(is_subnormal, subnormal, normal, location)?)?;
    let encoding = encoding.result(0).unwrap().as_ref();
    let encoding = block.append_operation(stable_hlo::or(encoding, sign, location)?)?;
    let encoding = encoding.result(0).unwrap().as_ref();
    let float = block.append_operation(stable_hlo::bitcast_convert(encoding, float_tensor_type, location)?)?;
    let float = float.result(0).unwrap().as_ref();
    let result =
        block.append_operation(stable_hlo::convert(float, lower_tensor_type(r#type, context, location)?, location)?)?;
    let result = result.result(0).unwrap().as_ref();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{DimensionBounds, DimensionType, GuardOperation, Placeholder};

    use crate::experimental::lowering::tests::{XlaProgramBuilder, dynamic_dimension};
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::{XlaArrayConstant, XlaConstant, XlaProgramBuilder as CompositeXlaProgramBuilder};

    use super::*;

    #[test]
    fn test_lower_bitcast_to_mlir_unbounded_dimensions() {
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("size", None)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::I32, true),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<?xf32>) -> tensor<?xi32> {
                %0 = stablehlo.bitcast_convert %arg0 : (tensor<?xf32>) -> tensor<?xi32>
                return %0 : tensor<?xi32>
              }
            }
        "#},
        );
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::U16, true),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main"),
            Err(LoweringError::UnsupportedOp {
                op: "width-changing bitcast with an unbounded dynamic dimension".to_string(),
            }),
        );
    }

    #[test]
    fn test_lower_reduce_precision() {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [3]));
        let output = builder
            .add_instruction(ReducePrecisionOperation::<ArrayType>::new(8, 7), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
                    %0 = stablehlo.reduce_precision %arg0, format = e8m7 : tensor<3xf32>
                    return %0 : tensor<3xf32>
                  }
                }
            "#},
        );
    }
    #[test]
    fn test_lower_cast_converts_bounded_dynamic_values() {
        use ryft_core::{AddOperation, Shape};

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        // An entry guard returns its inputs narrowed behind its check, so that computations consume static values.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()])).into();
        let exact: ArrayIrType = ArrayType::new_static(DataType::F64, [2]).into();
        let mut builder = CompositeXlaProgramBuilder::new();
        let first = builder.add_input(exact);
        let second = builder.add_input(symbolic.clone());
        let narrowed = builder
            .add_instruction(
                GuardOperation::new("region", vec![symbolic.clone(), symbolic]),
                Vec::new(),
                vec![first, second],
                None,
            )
            .unwrap()
            .to_vec();
        let sum = builder.add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), narrowed, None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let client = execution_client();
        let inputs = [MixedValue::Array(vec![1.0, 2.0], vec![2]), MixedValue::Array(vec![3.0, 4.0, 0.0, 0.0], vec![4])];
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[2]),
            Ok(vec![MixedValue::Array(vec![4.0, 6.0], vec![2])]),
        );
        let error = execute_mixed_program(&client, &program, &inputs, &[3]).unwrap_err().to_string();
        assert!(error.contains("input dimension `n` must have one extent at the entry of `region`"), "{error}");

        // The forward-mode derivative narrows the bounded dynamic tangent with a `cast`, which slices its physical
        // buffer to the static extent.
        let jvp = program.jvp().unwrap();
        assert!(jvp.to_string().contains("cast"), "{jvp}");
        let tangents =
            [MixedValue::Array(vec![1.0, 1.0], vec![2]), MixedValue::Array(vec![10.0, 20.0, 0.0, 0.0], vec![4])];
        assert_eq!(
            execute_mixed_program(
                &client,
                &jvp,
                &[inputs[0].clone(), inputs[1].clone(), tangents[0].clone(), tangents[1].clone()],
                &[2, 2],
            ),
            Ok(vec![MixedValue::Array(vec![4.0, 6.0], vec![2]), MixedValue::Array(vec![11.0, 21.0], vec![2])]),
        );

        // The pullback widens the static cotangent back to the bounded dynamic input with a `cast` that the residual
        // primal input witnesses, which pads its physical buffer and restores the logical extent.
        let pullback = program.linearize().unwrap().tangent().transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert!(pullback.to_string().contains("cast"), "{pullback}");
        assert_eq!(
            execute_mixed_program(
                &client,
                &pullback,
                &[MixedValue::Array(vec![1.0, 2.0], vec![2]), inputs[1].clone()],
                &[2],
            ),
            Ok(vec![
                MixedValue::Array(vec![1.0, 2.0], vec![2]),
                MixedValue::Array(vec![1.0, 2.0], vec![2]),
                MixedValue::Dimension(2),
            ]),
        );
    }
}
