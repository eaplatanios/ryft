use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, CAST_OPERATION_NAME, CastOperation, CastType, ConvertElementTypeOperation, DataType,
    Dimension, MeshAxisType, Operation, ProgramError, ReducePrecisionOperation, ReductionKind, Shape, Sharding,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::{
    lower_f64_constant_splat, lower_physical_bound_value, lower_physical_storage, lower_restore_dynamic_dimensions,
    lower_signed_one_bit_to_mlir, lower_tensor_type, lower_unplaced_constant_output, physical_bound_type,
    static_dimensions,
};
use crate::operations::assertions::{lower_dependency_barrier, lower_type_relationship_checks};
use crate::operations::constants::lower_layout_constraint;
use crate::operations::control_flow::{lower_refined_region_input, lower_refined_region_outputs};
use crate::operations::reductions::lower_reduce_to_mlir;
use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
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
        lower_cast_operation(self, lowerer, input_values, input_types, output_types)
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
        lower_cast_operation(self, lowerer, input_values, input_types, output_types)
    }
}

/// Lowers the same complete cast contract for native and composite type families. Witness producers are explicit
/// dependencies even for unchecked casts. A [checked](CastOperation::is_checked) cast is also ordered like an
/// assertion, so its dependency barrier consumes the current assertion token whether or not it emits a check of its
/// own (it emits none when the actual input types prove its relationship), whereas an unchecked cast never does.
fn lower_cast_operation<'b, 'c: 'b, 't: 'c, T: CastType + Into<ArrayIrType>, O: LowerableOperationFamily>(
    operation: &CastOperation<T>,
    lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[T],
    output_types: &[T],
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("output", output_types, 1, ProgramError);
    if input_values.is_empty() || input_types.is_empty() {
        return Err(LoweringError::UnsupportedOp { op: format!("`{}` expects at least 1 input", operation.name()) });
    }
    // A checked cast is ordered like an assertion even when the actual types prove its relationship and no check is
    // emitted (binding normalizes proven casts to unchecked ones, so this only affects casts staged directly through
    // a program builder); the dependency barrier therefore follows the mode rather than the emitted check.
    lower_cast_check(operation, input_values, input_types, lowerer)?;
    let values = lower_dependency_barrier(
        input_values,
        if operation.is_checked() { lowerer.effect_tokens.ordered_assertion } else { None },
        &mut lowerer.block,
        lowerer.location,
    )?;
    Ok(vec![lower_cast(values[0], &input_types[0].clone().into(), &output_types[0].clone().into(), lowerer)?])
}

/// Lowers the runtime check of a [checked](CastOperation::is_checked) cast whose inputs (the value followed by the
/// witnesses) are `input_values` of the `input_types`: the value at the cast's output type and the witnesses at their
/// declared types must satisfy their shared identity relationships. The source and output declarations observe the
/// same first input, preserving its original nominal bindings. Unchecked casts emit nothing.
fn lower_cast_check<'b, 'c: 'b, 't: 'c, T: CastType + Into<ArrayIrType>, O: LowerableOperationFamily>(
    operation: &CastOperation<T>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[T],
    lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
) -> Result<(), LoweringError> {
    if !operation.is_checked()
        || T::infer_cast_output_type(operation, input_types).map_err(ProgramError::from)?.1.is_some()
    {
        return Ok(());
    }
    let declared_types = std::iter::once(operation.output_type())
        .chain(operation.input_types())
        .cloned()
        .map(Into::into)
        .collect::<Vec<_>>();
    let values = std::iter::once(input_values[0]).chain(input_values.iter().copied()).collect::<Vec<_>>();
    let actual_types = std::iter::once(input_types[0].clone())
        .chain(input_types.iter().cloned())
        .map(Into::into)
        .collect::<Vec<_>>();
    lower_type_relationship_checks(
        CAST_OPERATION_NAME,
        &format!("in `{CAST_OPERATION_NAME}`"),
        &declared_types,
        &values,
        &actual_types,
        &mut lowerer.effect_tokens,
        &mut lowerer.block,
        lowerer.context,
        lowerer.location,
    )
    .map(|_| ())
}

/// Converts a cast's shape and enforces its requested physical layout and explicit sharding.
fn lower_cast<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    value: ValueRef<'b, 'c, 't>,
    input_type: &ArrayIrType,
    output_type: &ArrayIrType,
    lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let value =
        lower_cast_shape(value, input_type, output_type, &mut lowerer.block, lowerer.context, lowerer.location)?;
    let (ArrayIrType::Array(input_type), ArrayIrType::Array(output_type)) = (input_type, output_type) else {
        return Ok(value);
    };
    let mut result = value;
    if output_type.layout().is_some() && output_type.layout() != input_type.layout() {
        // LayoutConstraint takes a static physical buffer. Preserve logical sizes on the converted value while the
        // buffer is exposed, then restore them after constraining its allocation. No padding becomes logical data.
        // An input that already declares the requested layout was constrained by its producer.
        let physical_type = physical_bound_type(output_type)?;
        let physical =
            lower_physical_storage(value, output_type, &mut lowerer.block, lowerer.context, lowerer.location)?;
        result = lower_layout_constraint(physical, &physical_type, &mut lowerer.block, lowerer.location)?;
        let sources = (0..output_type.rank()).map(|axis| (value, axis)).collect::<Vec<_>>();
        result = lower_restore_dynamic_dimensions(
            result,
            output_type,
            &sources,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
    }
    if let Some(sharding) = output_type.sharding()
        && cast_changes_explicit_placement(input_type, output_type)
    {
        result = lower_sharding_constraint(
            &[result],
            sharding,
            &lowerer.collective_state.bound_manual_axes,
            &mut lowerer.block,
            lowerer.location,
        )?[0];
    }
    Ok(result)
}

/// Returns whether a cast from `input_type` to `output_type` changes the placement that the type system tracks over
/// explicit mesh axes and therefore emits a sharding constraint carrying its complete output sharding, like
/// [`ReshardOperation`](ryft_core::ReshardOperation). Carrying an existing sharding forward requests no
/// redistribution, and an output without a sharding is untracked rather than replicated. Type inference guarantees
/// that the placement over auto and manual axes is unchanged (refer to [`CastOperation`]), so the emitted constraint
/// never contradicts the type the program was checked against.
pub(crate) fn cast_changes_explicit_placement(input_type: &ArrayType, output_type: &ArrayType) -> bool {
    let Some(output_sharding) = output_type.sharding() else {
        return false;
    };
    let has_explicit_axis =
        |sharding: &Sharding| sharding.mesh().axes().iter().any(|axis| axis.r#type() == MeshAxisType::Explicit);
    match input_type.sharding() {
        None => has_explicit_axis(output_sharding),
        Some(input_sharding) if input_sharding.mesh() != output_sharding.mesh() => {
            has_explicit_axis(input_sharding) || has_explicit_axis(output_sharding)
        }
        Some(input_sharding) => input_sharding.conflicts_on_explicit_axes_with(output_sharding),
    }
}

/// Converts the shape of the lowered `value` of type `input_type` to that of the compatible `output_type` (refer to
/// [`CastOperation`] and [`GuardOperation`](ryft_core::GuardOperation)). First-class dimensions lower to `i64` scalars
/// whatever their type. The value is converted through the type whose axes are static wherever either type is static:
/// narrowing a bounded dynamic axis to a static extent slices the physical buffer (refer to
/// [`lower_refined_region_outputs`]), and then widening a static axis to a bounded dynamic one, or moving a dynamic
/// axis to another identity with another physical bound, pads or slices it and restores its runtime size (refer to
/// [`lower_refined_region_input`]). Neither conversion checks runtime extents. Checked casts and guards validate the
/// relationship first; unchecked casts from transforms carry a proof of the relationship and dependencies on the values
/// that justify it. Both conversions operate on the static physical buffer of the whole value, so a conversion of any
/// axis is rejected when either type has an unbounded dynamic axis, whereas a value whose tensor type does not change
/// is returned unchanged even when it has unbounded axes.
pub(crate) fn lower_cast_shape<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    input_type: &ArrayIrType,
    output_type: &ArrayIrType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    // Converting any axis materializes the static physical buffer of the whole value, which needs a finite physical
    // bound for every dynamic axis of both types, including axes that this conversion leaves unchanged.
    if let (ArrayIrType::Array(input), ArrayIrType::Array(output)) = (input_type, output_type) {
        let dimension_pairs = input.shape().dimensions().iter().zip(output.shape().dimensions());
        let changed_axis = dimension_pairs.clone().position(|pair| match pair {
            (Dimension::Static(_), Dimension::Static(_)) => false,
            (Dimension::Dynamic(input), Dimension::Dynamic(output)) => {
                input.bounds().upper() != output.bounds().upper()
            }
            _ => true,
        });
        let unbounded_dimension = dimension_pairs
            .enumerate()
            .flat_map(|(axis, (input, output))| [(axis, input), (axis, output)])
            .find(|(_, dimension)| matches!(dimension, Dimension::Dynamic(variable) if variable.bounds().upper().is_none()));
        if let (Some(changed_axis), Some((axis, dimension))) = (changed_axis, unbounded_dimension) {
            return Err(LoweringError::UnsupportedOp {
                op: format!(
                    "shape conversion along axis {changed_axis} requires finite physical bounds, but axis {axis} has \
                     unbounded dimension `{dimension}`",
                ),
            });
        }
    }
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
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;

    use ryft_core::{DimensionBounds, DimensionType, GuardOperation, Placeholder, TypeError};

    use crate::experimental::lowering::tests::{XlaProgramBuilder, dynamic_dimension};
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::{
        XlaArrayConstant, XlaConstant, XlaOperation, XlaProgramBuilder as CompositeXlaProgramBuilder,
    };
    use crate::to_mlir_module_for_program;

    use super::*;

    /// Compiles `module` with one SPMD partition per CPU device, executes it once with one `f64` input buffer of
    /// `dimensions` per device holding that device's entry of `shards`, and returns every device's single `f64` output
    /// as its elements in logical row-major order together with its physical layout.
    #[allow(deprecated)] // Inspect the physical result layout, independently of the declared array type.
    fn execute_sharded_module(
        module: &str,
        shards: &[Vec<f64>],
        dimensions: &[u64],
    ) -> Result<Vec<(Vec<f64>, ryft_pjrt::Layout)>, ryft_pjrt::Error> {
        use std::sync::Arc;

        use ryft_pjrt::{
            BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
            load_cpu_plugin,
        };

        use crate::experimental::lowering::tests::test_spmd_compilation_options;
        use crate::tests::{values_from_bytes, values_to_bytes};

        let plugin = load_cpu_plugin()?;
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(shards.len()), ..Default::default() }))?;
        let executable = client.compile(
            &PjrtProgram::Mlir { bytecode: module.as_bytes().to_vec() },
            &test_spmd_compilation_options(shards.len()),
        )?;
        let devices = executable.addressable_devices()?;
        assert_eq!(devices.len(), shards.len());
        let inputs = devices
            .iter()
            .zip(shards)
            .map(|(device, shard)| {
                let buffer = client.buffer(
                    values_to_bytes(shard).as_slice(),
                    BufferType::F64,
                    dimensions,
                    None,
                    device.clone(),
                    None,
                )?;
                Ok(vec![ExecutionInput { buffer: Arc::new(buffer), donatable: false }])
            })
            .collect::<Result<Vec<_>, ryft_pjrt::Error>>()?;
        let outputs = executable
            .execute(
                inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
            )?
            .block_until_ready()?;
        outputs
            .iter()
            .map(|output| {
                assert_eq!(output.outputs.len(), 1);
                let output = &output.outputs[0];
                let layout = output.layout()?;
                let bytes = output
                    .copy_to_host(Some(ryft_pjrt::Layout::dense_major_to_minor(output.dimensions()?.len())))?
                    .r#await()?;
                Ok((values_from_bytes::<f64>(bytes.as_slice()), layout))
            })
            .collect()
    }

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
        std::fs::write("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/converts_jvp.txt", jvp.to_string()).unwrap();
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
        std::fs::write("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/converts_pullback.txt", pullback.to_string()).unwrap();
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

    #[test]
    fn test_lower_checked_cast() {
        use ryft_core::{Add, Cast, Shape, Tracer, TracingContext};

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        // A checked cast asserts its relationship before converting: narrowing `f64[n]` to `f64[2]` requires the
        // runtime extent 2, and widening `f64[2]` to `f64[n]` next to a witness of type `f64[n]` requires that the
        // witness has extent 2. An unchecked narrowing assumes the extent instead.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()])).into();
        let exact: ArrayIrType = ArrayType::new_static(DataType::F64, [2]).into();
        let client = execution_client();
        let symbolic_input = MixedValue::Array(vec![1.0, 2.0, 0.0, 0.0], vec![4]);
        for checked in [true, false] {
            let (_, program) = TracingContext::<XlaConstant, XlaOperation>::trace(
                |inputs: Vec<Tracer<_>>| {
                    Ok(vec![if checked {
                        inputs[0].cast(exact.clone(), &[])?
                    } else {
                        inputs[0].cast_unchecked(exact.clone(), &[])?
                    }])
                },
                vec![symbolic.clone()],
            )
            .unwrap();
            assert_eq!(
                execute_mixed_program(&client, &program, std::slice::from_ref(&symbolic_input), &[2]),
                Ok(vec![MixedValue::Array(vec![1.0, 2.0], vec![2])]),
            );
            let result = execute_mixed_program(&client, &program, std::slice::from_ref(&symbolic_input), &[3]);
            if checked {
                let error = result.unwrap_err().to_string();
                assert!(error.contains("input dimension `n` must have the declared extent in `cast`"), "{error}");
            } else {
                assert!(result.is_ok(), "{result:?}");
            }
        }

        let (_, program) = TracingContext::<XlaConstant, XlaOperation>::trace(
            |inputs: Vec<Tracer<_>>| {
                let widened = inputs[0].cast(symbolic.clone(), std::slice::from_ref(&inputs[1]))?;
                Ok(vec![widened.add(&inputs[1])?])
            },
            vec![exact, symbolic.clone()],
        )
        .unwrap();
        let inputs = [MixedValue::Array(vec![1.0, 2.0], vec![2]), MixedValue::Array(vec![3.0, 4.0, 5.0, 0.0], vec![4])];
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[2]),
            Ok(vec![MixedValue::Array(vec![4.0, 6.0], vec![2]), MixedValue::Dimension(2)]),
        );
        let error = execute_mixed_program(&client, &program, &inputs, &[3]).unwrap_err().to_string();
        assert!(error.contains("input dimension `n` must have one extent in `cast`"), "{error}");
    }

    #[test]
    fn test_lower_checked_cast_dimension() {
        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let mut builder = CompositeXlaProgramBuilder::new();
        let value = builder.add_input(n.clone().into());
        let output = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::checked(DimensionType::from(Dimension::Static(2)).into(), vec![n.into()])
                    .unwrap(),
                vec![],
                vec![value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let client = execution_client();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Dimension(2)], &[]),
            Ok(vec![MixedValue::Dimension(2)]),
        );
        let error = execute_mixed_program(&client, &program, &[MixedValue::Dimension(3)], &[]).unwrap_err().to_string();
        assert!(error.contains("must have the declared extent in `cast`"), "{error}");
    }

    #[test]
    fn test_lower_cast_omits_proven_relationship_checks() {
        use ryft_core::{Layout, Shape, TiledLayout};

        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let [a, b] = ["a", "b"].map(|name| DimensionType::new(name, bounds));
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let output_type = input_type.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let mut builder = XlaProgramBuilder::new();
        let value = builder.add_input(input_type.clone());
        // Keep an explicitly checked primitive to exercise backend proof, independently of public normalization.
        let output = builder
            .add_instruction(
                CastOperation::checked(output_type, vec![input_type.clone()]).unwrap(),
                vec![],
                vec![value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        std::fs::write("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/relationship_1.txt", &module).unwrap();
        assert!(!module.contains("@ryft.assert"), "{module}");

        let declared = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(); 2]));
        let actual = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let value = builder.add_input(actual);
        let output = builder
            .add_instruction(
                CastOperation::checked(declared.clone(), vec![declared]).unwrap(),
                vec![],
                vec![value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        std::fs::write("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/relationship_2.txt", &module).unwrap();
        // The stored repeated-identity signature proves itself, but actual [a, 3] still needs a == 3. Source/output
        // declarations repeat both observations; that one real comparison must not emit duplicate callbacks.
        assert_eq!(module.matches("@ryft.assert").count(), 1, "{module}");

        let fixed = DimensionType::new("fixed", DimensionBounds::new(2, Some(3)).unwrap());
        let source =
            ArrayType::new(DataType::F32, Shape::new(vec![fixed.dimension().clone(), b.simplified_dimension()]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(source.clone());
        let output = builder
            .add_instruction(
                CastOperation::checked(ArrayType::new_static(DataType::F32, [2, 3]), vec![source]).unwrap(),
                vec![],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        std::fs::write("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/relationship_3.txt", &module).unwrap();
        assert_eq!(module.matches("@ryft.assert").count(), 1, "{module}");

        let source =
            ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let target =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), a.simplified_dimension()]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(source.clone());
        let output = builder
            .add_instruction(CastOperation::checked(target, vec![source]).unwrap(), vec![], vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        std::fs::write("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/relationship_4.txt", &module).unwrap();
        // Swapping axes states a == b twice, but requires just one runtime comparison.
        assert_eq!(module.matches("@ryft.assert").count(), 1, "{module}");
    }

    #[test]
    fn test_lower_cast_different_bounds() {
        use ryft_core::Shape;

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let source = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()]));
        let target = ArrayType::new(DataType::F64, Shape::new(vec![m.simplified_dimension()]));
        let client = execution_client();
        for constructor in [CastOperation::<ArrayIrType>::checked, CastOperation::<ArrayIrType>::unchecked] {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(source.clone().into());
            let witness = builder.add_input(target.clone().into());
            let output = builder
                .add_instruction(
                    constructor(target.clone().into(), vec![source.clone().into(), target.clone().into()]).unwrap(),
                    vec![],
                    vec![input, witness],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            assert_eq!(
                execute_mixed_program(
                    &client,
                    &program,
                    &[MixedValue::Array(vec![1.0, 2.0, 0.0, 0.0], vec![4]), MixedValue::Array(vec![0.0; 7], vec![7]),],
                    &[2, 2]
                ),
                Ok(vec![MixedValue::Array(vec![1.0, 2.0], vec![2]), MixedValue::Dimension(2)])
            );
            if program.instructions()[0]
                .operation()
                .effects()
                .classes()
                .contains(ryft_core::EffectClass::OrderedAssertion)
            {
                let error = execute_mixed_program(
                    &client,
                    &program,
                    &[MixedValue::Array(vec![1.0; 4], vec![4]), MixedValue::Array(vec![0.0; 7], vec![7])],
                    &[2, 3],
                )
                .unwrap_err()
                .to_string();
                assert!(error.contains("in `cast`"), "{error}");
            }
        }
    }

    #[test]
    fn test_lower_cast_different_bounds_narrowing_bound() {
        use ryft_core::Shape;

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        // Moving a dynamic axis to an identity with a smaller physical bound slices the physical buffer from the
        // source bound to the target bound and restores the runtime extent, which the witness shares. Only the checked
        // cast verifies that the extents agree.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let m = DimensionType::new("m", DimensionBounds::positive(Some(5)).unwrap());
        let source = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()]));
        let target = ArrayType::new(DataType::F64, Shape::new(vec![m.simplified_dimension()]));
        let client = execution_client();
        for constructor in [CastOperation::<ArrayIrType>::checked, CastOperation::<ArrayIrType>::unchecked] {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(source.clone().into());
            let witness = builder.add_input(target.clone().into());
            let output = builder
                .add_instruction(
                    constructor(target.clone().into(), vec![source.clone().into(), target.clone().into()]).unwrap(),
                    vec![],
                    vec![input, witness],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            let checked = program.instructions()[0]
                .operation()
                .effects()
                .classes()
                .contains(ryft_core::EffectClass::OrderedAssertion);
            if !checked {
                assert_eq!(
                    to_mlir_module_for_program(
                        &program,
                        &[],
                        &[source.clone(), target.clone()],
                        &[target.clone()],
                        "main",
                        None,
                        None,
                    )
                    .unwrap(),
                    // The witness is a dependency of the unchecked cast, so both inputs pass through one barrier. The
                    // negative high padding slices the physical buffer from the source bound to the target bound.
                    indoc! {"
                        module {
                          func.func @main(%arg0: tensor<7xf64>, %arg1: tensor<4xf64>, %arg2: tensor<i32>, \
                              %arg3: tensor<i32>) -> (tensor<?xf64, #stablehlo.bounds<4>>, tensor<i64>) {
                            %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : \
                                (tensor<7xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<7>>
                            %1 = stablehlo.set_dimension_size %arg1, %arg3, dim = 0 : \
                                (tensor<4xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<4>>
                            %2:2 = stablehlo.optimization_barrier %0, %1 : \
                                tensor<?xf64, #stablehlo.bounds<7>>, tensor<?xf64, #stablehlo.bounds<4>>
                            %3 = stablehlo.get_dimension_size %2#0, dim = 0 : \
                                (tensor<?xf64, #stablehlo.bounds<7>>) -> tensor<i32>
                            %c = stablehlo.constant dense<7> : tensor<i32>
                            %4 = stablehlo.set_dimension_size %2#0, %c, dim = 0 : \
                                (tensor<?xf64, #stablehlo.bounds<7>>, tensor<i32>) -> tensor<7xf64>
                            %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                            %5 = stablehlo.pad %4, %cst, low = [0], high = [-3], interior = [0] : \
                                (tensor<7xf64>, tensor<f64>) -> tensor<4xf64>
                            %6 = stablehlo.set_dimension_size %5, %3, dim = 0 : \
                                (tensor<4xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<4>>
                            %7 = stablehlo.get_dimension_size %6, dim = 0 : \
                                (tensor<?xf64, #stablehlo.bounds<4>>) -> tensor<i32>
                            %8 = stablehlo.convert %7 : (tensor<i32>) -> tensor<i64>
                            return %6, %8 : tensor<?xf64, #stablehlo.bounds<4>>, tensor<i64>
                          }
                        }
                    "},
                );
            }
            let inputs = [
                MixedValue::Array(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0], vec![7]),
                MixedValue::Array(vec![0.0; 4], vec![4]),
            ];
            assert_eq!(
                execute_mixed_program(&client, &program, &inputs, &[3, 3]),
                Ok(vec![MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]), MixedValue::Dimension(3)]),
            );
            let result = execute_mixed_program(&client, &program, &inputs, &[3, 2]);
            if checked {
                let error = result.unwrap_err().to_string();
                assert!(error.contains("in `cast`"), "{error}");
            } else {
                assert_eq!(result, Ok(vec![MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]), MixedValue::Dimension(3)]));
            }
        }
    }

    #[test]
    fn test_lower_cast_check_dependencies() {
        use ryft_core::{PrintOperation, Shape};
        use ryft_mlir::{Location, OperationResultRef};

        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block(&[
            (lower_tensor_type(&symbolic, &context, location).unwrap(), location),
            (lower_tensor_type(&exact, &context, location).unwrap(), location),
        ]);
        let source = block.argument(0).unwrap().as_ref();
        let unrelated = block.argument(1).unwrap().as_ref();
        let mut lowerer: XlaLowerer<'_, '_, '_, XlaOperation> =
            XlaLowerer::new(block.as_ref(), &context, location.as_ref());
        let guarded = CastOperation::checked(exact.clone(), vec![symbolic.clone()])
            .unwrap()
            .lower(&mut lowerer, &[source], &[symbolic], &[exact.clone()])
            .unwrap()[0];
        assert!(lowerer.effect_tokens.ordered_assertion.is_some());
        let unchanged = CastOperation::unchecked(exact.clone(), vec![exact.clone()])
            .unwrap()
            .lower(&mut lowerer, &[unrelated], &[exact.clone()], &[exact.clone()])
            .unwrap();
        assert_eq!(unchanged, vec![unrelated]);
        // Inspect actual operand edges: the narrowing must consume the barrier, not merely follow it in the block.
        let mut pending = vec![guarded];
        let mut found = false;
        while let Some(value) = pending.pop() {
            if let Some(result) = value.cast::<OperationResultRef>() {
                let operation = result.operation().unwrap();
                if operation.name().as_str().unwrap() == "stablehlo.optimization_barrier" {
                    assert_eq!(operation.operand(0).unwrap().value().unwrap(), source);
                    assert_eq!(
                        operation.operand(1).unwrap().value().unwrap(),
                        lowerer.effect_tokens.ordered_assertion.unwrap()
                    );
                    found = true;
                    break;
                }
                pending.extend(
                    (0..operation.operand_count()).map(|index| operation.operand(index).unwrap().value().unwrap()),
                );
            }
        }
        assert!(found, "{block}");
        // A later print of the converted value consumes the barrier-dependent result. An unrelated print keeps
        // its independent I/O token chain, as tested in the assertion module.
        let printed = PrintOperation::<ArrayType>::new("cast result")
            .lower(&mut lowerer, &[guarded], &[exact.clone()], &[exact])
            .unwrap();
        assert_eq!(printed, vec![guarded]);
        let print =
            lowerer.effect_tokens.ordered_io.unwrap().cast::<OperationResultRef>().unwrap().operation().unwrap();
        assert_eq!(print.operand(0).unwrap().value().unwrap(), guarded);
    }

    #[test]
    fn test_lower_checked_cast_token_ordering() {
        use ryft_core::{AssertOperation, Layout, TiledLayout};

        // Binding normalizes a proven cast to an unchecked one, so a checked cast whose relationship the actual types
        // prove only arises when staged directly through a program builder. It emits no check of its own but is still
        // ordered like an assertion: its dependency barrier consumes the token of the preceding assertion, whereas
        // its unchecked twin converts the input directly and leaves the token chain alone.
        let input_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let output_type = input_type.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let assertion = "%0 = stablehlo.after_all  : !stablehlo.token\n    \
            %1 = stablehlo.custom_call @ryft.assert(%arg0, %0) {api_version = 4 : i32, \
            backend_config = {actor = \"assert\", kind = \"generic\", label_count = \"0\", \
            message = \"precondition\"}, has_side_effect = true} : (tensor<i1>, !stablehlo.token) -> !stablehlo.token";
        let cases = [
            (
                CastOperation::checked as fn(ArrayType, Vec<ArrayType>) -> Result<CastOperation<ArrayType>, TypeError>,
                formatdoc! {"
                    module {{
                      func.func @main(%arg0: tensor<i1>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {{
                        {assertion}
                        %2:2 = stablehlo.optimization_barrier %arg1, %1 : tensor<2x3xf32>, !stablehlo.token
                        %3 = stablehlo.custom_call @LayoutConstraint(%2#0) {{\
                            operand_layouts = [dense<[0, 1]> : tensor<2xindex>], \
                            result_layouts = [dense<[0, 1]> : tensor<2xindex>]}} : (tensor<2x3xf32>) -> tensor<2x3xf32>
                        return %3 : tensor<2x3xf32>
                      }}
                    }}
                "},
            ),
            (
                CastOperation::unchecked,
                formatdoc! {"
                    module {{
                      func.func @main(%arg0: tensor<i1>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {{
                        {assertion}
                        %2 = stablehlo.custom_call @LayoutConstraint(%arg1) {{\
                            operand_layouts = [dense<[0, 1]> : tensor<2xindex>], \
                            result_layouts = [dense<[0, 1]> : tensor<2xindex>]}} : (tensor<2x3xf32>) -> tensor<2x3xf32>
                        return %2 : tensor<2x3xf32>
                      }}
                    }}
                "},
            ),
        ];
        for (constructor, expected) in cases {
            let mut builder = XlaProgramBuilder::new();
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
            let input = builder.add_input(input_type.clone());
            builder
                .add_instruction(AssertOperation::<ArrayType>::new("precondition"), Vec::new(), vec![predicate], None)
                .unwrap();
            let output = builder
                .add_instruction(
                    constructor(output_type.clone(), vec![input_type.clone()]).unwrap(),
                    Vec::new(),
                    vec![input],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap();
            assert_eq!(to_mlir_module_for_plain_program(&program, "main").unwrap(), expected);
        }
    }

    #[test]
    fn test_lower_cast_unbounded_shape() {
        use ryft_core::Shape;
        use ryft_mlir::Location;

        // Converting any axis materializes the static physical buffer of the whole value, so it is rejected when the
        // converted axis or another axis that the conversion leaves unchanged has an unbounded dimension. A cast that
        // converts no axis returns its input unchanged, even when the input has unbounded dimensions.
        let n = DimensionType::new("n", DimensionBounds::positive(None).unwrap());
        let m = DimensionType::new("m", DimensionBounds::positive(Some(5)).unwrap());
        let unbounded = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let mixed = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension(), m.simplified_dimension()]));
        let partially_exact =
            ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension(), Dimension::Static(2)]));
        let cases = [
            (
                unbounded.clone(),
                ArrayType::new_static(DataType::F32, [2]),
                Err(LoweringError::UnsupportedOp {
                    op: "shape conversion along axis 0 requires finite physical bounds, but axis 0 has unbounded \
                         dimension `n`"
                        .to_owned(),
                }),
            ),
            (
                mixed.clone(),
                partially_exact,
                Err(LoweringError::UnsupportedOp {
                    op: "shape conversion along axis 1 requires finite physical bounds, but axis 0 has unbounded \
                         dimension `n`"
                        .to_owned(),
                }),
            ),
            (mixed.clone(), mixed, Ok(())),
        ];
        for (input_type, output_type, expected) in cases {
            let context = MlirContext::new();
            let location = context.unknown_location();
            let block = context.block(&[(lower_tensor_type(&input_type, &context, location).unwrap(), location)]);
            let source = block.argument(0).unwrap().as_ref();
            let mut lowerer: XlaLowerer<'_, '_, '_, XlaOperation> =
                XlaLowerer::new(block.as_ref(), &context, location.as_ref());
            assert_eq!(
                CastOperation::unchecked(output_type.clone(), vec![input_type.clone()]).unwrap().lower(
                    &mut lowerer,
                    &[source],
                    &[input_type],
                    &[output_type],
                ),
                expected.map(|()| vec![source]),
            );
        }
    }

    #[test]
    fn test_lower_checked_cast_in_shard_map() {
        use ryft_core::{
            Cast, ConvertElementType, DimensionFromScalar, DimensionToScalar, LogicalMesh, MeshAxis, MeshAxisType,
            Sharding, TracedShardMap, ValueProjection, trace_shard_map,
        };

        use crate::XlaDomain;
        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 0);
        let scalar = ArrayType::scalar(DataType::F64);
        let extent = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |input| {
                let integer = input.convert_element_type(DataType::I64).unwrap();
                let dimension = integer.to_dimension(extent.to_variable()).unwrap();
                let checked = dimension.cast(DimensionType::from(Dimension::Static(2)).into(), &[]).unwrap();
                let scalar = checked.to_scalar().unwrap().convert_element_type(DataType::F64).unwrap();
                scalar.into_projected().unwrap()
            },
            scalar.clone(),
            mesh,
            sharding.clone(),
            sharding,
        )
        .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(scalar.into());
        let body = builder.import_region(traced.body().entry_region_ref());
        let output = builder.add_instruction(traced.operation().clone(), vec![body], vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let client = execution_client();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(vec![2.0], vec![])], &[]),
            Ok(vec![MixedValue::Array(vec![2.0], vec![])])
        );
        let error = execute_mixed_program(&client, &program, &[MixedValue::Array(vec![3.0], vec![])], &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("must have the declared extent in `cast`"), "{error}");
    }

    #[test]
    fn test_lower_checked_cast_refined_representation() {
        use ryft_core::{LogicalMesh, MeshAxis, Sharding, ShardingDimension};

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let declared = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = declared
            .clone()
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let output = ArrayType::new_static(DataType::F64, [2]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let value = builder.add_input(input.clone().into());
        let result = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::checked(output.clone().into(), vec![declared.clone().into()]).unwrap(),
                vec![],
                vec![value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![result], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            formatdoc! {"
                lambda %0:{input}, %1:{input} .
                let %2:{output} = cast [
                    type={output},
                    inputs=[{declared}],
                    checked=true,
                ] %0
                    %3:{input} = cast [
                        type={input},
                        inputs=[{input}, {output}],
                    ] %0 %2
                    %4:{output} = cast [
                        type={output},
                        inputs=[{input}, {input}],
                    ] %1 %3
                in (%2, %4)"},
        );
        // Distinct explicit shardings on one device exercise representation changes without moving physical buffers.
        // The tangent conversion must remain behind the primal shape check.
        let client = execution_client();
        let primal = MixedValue::Array(vec![1.0, 2.0, 0.0, 0.0], vec![4]);
        let tangent = MixedValue::Array(vec![3.0, 4.0, 0.0, 0.0], vec![4]);
        assert_eq!(
            execute_mixed_program(&client, &jvp, &[primal.clone(), tangent], &[2, 2]),
            Ok(vec![MixedValue::Array(vec![1.0, 2.0], vec![2]), MixedValue::Array(vec![3.0, 4.0], vec![2])]),
        );
        let wrong = MixedValue::Array(vec![1.0, 2.0, 3.0, 0.0], vec![4]);
        let error = execute_mixed_program(&client, &jvp, &[wrong.clone(), wrong], &[3, 3]).unwrap_err().to_string();
        assert!(error.contains("in `cast`"), "{error}");

        let pullback = program.linearize().unwrap().tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            formatdoc! {"
                lambda %0:{output}, %1:{input} .
                let %2:{input} = cast [
                    type={input},
                    inputs=[{output}, {input}],
                ] %0 %1
                in (%2)"},
        );
        assert_eq!(
            execute_mixed_program(&client, &pullback, &[MixedValue::Array(vec![5.0, 6.0], vec![2]), primal], &[2],),
            Ok(vec![MixedValue::Array(vec![5.0, 6.0], vec![2]), MixedValue::Dimension(2)]),
        );
    }

    #[test]
    fn test_lower_cast_output_placement() {
        use ryft_core::{Layout, LogicalMesh, MeshAxis, Sharding, ShardingDimension, TiledLayout};

        let input_type = ArrayType::new_static(DataType::F64, [2, 3]);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let output_type = input_type
            .clone()
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        let expected = indoc! {"
            module {
              sdy.mesh @mesh = <[\"x\"=2]>
              func.func @main(%arg0: tensor<2x3xf64>) -> tensor<2x3xf64> {
                %0 = stablehlo.custom_call @LayoutConstraint(%arg0) {\
                    operand_layouts = [dense<[0, 1]> : tensor<2xindex>], \
                    result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<2x3xf64>) -> tensor<2x3xf64>
                %1 = sdy.sharding_constraint %0 <@mesh, [{\"x\"}, {}]> : tensor<2x3xf64>
                return %1 : tensor<2x3xf64>
              }
            }
        "};
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(input_type.clone());
            let output = builder
                .add_instruction(
                    constructor(output_type.clone(), vec![input_type.clone()]).unwrap(),
                    vec![],
                    vec![input],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            assert_eq!(to_mlir_module_for_plain_program(&program, "main").unwrap(), expected);
            let program = program.into_unprojected::<XlaConstant, XlaOperation>().unwrap();
            assert_eq!(
                to_mlir_module_for_program(
                    &program,
                    &[],
                    &vec![input_type.clone()],
                    &vec![output_type.clone()],
                    "main",
                    None,
                    None,
                )
                .unwrap(),
                indoc! {"
                    module {
                      sdy.mesh @mesh = <[\"x\"=2]>
                      func.func @main(%arg0: tensor<2x3xf64>) \
                          -> (tensor<2x3xf64> {mhlo.layout_mode = \"{0,1}\"}) {
                        %0 = stablehlo.custom_call @LayoutConstraint(%arg0) {\
                            operand_layouts = [dense<[0, 1]> : tensor<2xindex>], \
                            result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<2x3xf64>) -> tensor<2x3xf64>
                        %1 = sdy.sharding_constraint %0 <@mesh, [{\"x\"}, {}]> : tensor<2x3xf64>
                        return %1 : tensor<2x3xf64>
                      }
                    }
                "},
            );
        }
    }

    #[test]
    fn test_lower_cast_explicit_sharding_multiple_devices() {
        use ryft_core::{
            Cast, LogicalMesh, MeshAxis, Sharding, ShardingDimension, TracedShardMap, Typed, trace_shard_map,
        };

        use crate::XlaDomain;
        use crate::operations::sharding::shard_map::to_mlir_module as to_mlir_module_for_shard_map;

        // A cast whose output moves the placement over an explicit axis emits its complete output sharding as a
        // constraint, which the partitioner realizes by gathering the shards: every device returns the whole array.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let replicated = Sharding::replicated(mesh, 2);
        let input_type = ArrayType::new_static(DataType::F64, [4, 2]).with_sharding(sharded.clone()).unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [4, 2]).with_sharding(replicated.clone()).unwrap();
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(
                CastOperation::unchecked(output_type.clone(), vec![input_type.clone()]).unwrap(),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .into_unprojected::<XlaConstant, XlaOperation>()
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[input_type],
            &[output_type],
            "main",
            Some(std::slice::from_ref(&sharded)),
            Some(std::slice::from_ref(&replicated)),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4x2xf64> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) \
                      -> (tensor<4x2xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={\"x\"}>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {}], replicated={\"x\"}> : tensor<4x2xf64>
                    return %0 : tensor<4x2xf64>
                  }
                }
            "},
        );
        let shards = [vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let values = (1..=8).map(f64::from).collect::<Vec<_>>();
        let outputs = execute_sharded_module(&module, &shards, &[2, 2]).unwrap();
        assert_eq!(outputs.iter().map(|(values, _)| values.clone()).collect::<Vec<_>>(), vec![values.clone(), values]);

        // Inside a manual region, the constraint names only the axes that the region does not bind: the bound manual
        // axis `x` is projected out of the emitted constraint, which still refers to the module's full mesh symbol.
        // Replicating the body value over the explicit axis `z` gathers it on every device of the manual shard.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let in_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "z"])]).unwrap();
        let out_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |input| {
                let local_type = input.r#type().into_owned();
                let replicated =
                    local_type.sharding().unwrap().with_dimensions(vec![ShardingDimension::replicated()]).unwrap();
                input.cast(local_type.with_sharding(replicated).unwrap(), &[]).unwrap()
            },
            ArrayType::new_static(DataType::F64, [8]),
            mesh,
            in_sharding,
            out_sharding,
        )
        .unwrap();
        let module = to_mlir_module_for_shard_map(&traced, "main").unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"z\"=2]>
                  func.func @main(%arg0: tensor<8xf64> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\", \"z\"}]>}) \
                      -> (tensor<8xf64> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}], replicated={\"z\"}>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\", \"z\", ?}]>] \
                        out_shardings=[<@mesh, [{\"x\", ?}], replicated={\"z\"}>] manual_axes={\"x\"} \
                        (%arg1: tensor<4xf64>) {
                      %1 = sdy.sharding_constraint %arg1 <@mesh, [{}], replicated={\"z\"}> : tensor<4xf64>
                      sdy.return %1 : tensor<4xf64>
                    } : (tensor<8xf64>) -> tensor<8xf64>
                    return %0 : tensor<8xf64>
                  }
                }
            "},
        );
        let shards =
            (0..4).map(|device| vec![f64::from(2 * device + 1), f64::from(2 * device + 2)]).collect::<Vec<_>>();
        let outputs = execute_sharded_module(&module, &shards, &[2]).unwrap();
        assert_eq!(
            outputs.iter().map(|(values, _)| values.clone()).collect::<Vec<_>>(),
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0],
                vec![5.0, 6.0, 7.0, 8.0],
            ],
        );
    }

    #[test]
    fn test_lower_cast_dynamic_output_explicit_sharding() {
        use ryft_core::{LogicalMesh, MeshAxis, Shape, Sharding, ShardingDimension};
        use ryft_pjrt::{ClientOptions, CpuClientOptions, Program as PjrtProgram, load_cpu_plugin};

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program, test_spmd_compilation_options};
        use crate::tests::execution_client;

        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let symbolic = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()]));
        let program_and_types = |mesh: LogicalMesh| {
            let replicated = Sharding::replicated(mesh.clone(), 1);
            let sharded = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
            let input_type = symbolic.clone().with_sharding(replicated.clone()).unwrap();
            let output_type = symbolic.clone().with_sharding(sharded.clone()).unwrap();
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(input_type.clone().into());
            let output = builder
                .add_instruction(
                    CastOperation::<ArrayIrType>::unchecked(
                        output_type.clone().into(),
                        vec![input_type.clone().into()],
                    )
                    .unwrap(),
                    Vec::new(),
                    vec![input],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            (program, input_type, output_type, replicated, sharded)
        };

        // The constraint lowers on the bounded-dynamic tensor (the executable boundary carries the physical buffer and
        // its runtime extent separately), but Shardy propagation rejects the dynamic tensor that the constraint
        // places, so the module cannot be compiled for a multi-device mesh.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let (program, input_type, output_type, replicated, sharded) = program_and_types(mesh);
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[input_type],
            &[output_type],
            "main",
            Some(std::slice::from_ref(&replicated)),
            Some(std::slice::from_ref(&sharded)),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf64> \
                      {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={\"x\"}>}, \
                      %arg1: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={\"x\"}>}) \
                      -> (tensor<?xf64, #stablehlo.bounds<4>> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}, \
                      tensor<i64> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={\"x\"}>}) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : \
                        (tensor<4xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<4>>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{\"x\"}]> : tensor<?xf64, #stablehlo.bounds<4>>
                    %2 = stablehlo.get_dimension_size %1, dim = 0 : (tensor<?xf64, #stablehlo.bounds<4>>) -> tensor<i32>
                    %3 = stablehlo.convert %2 : (tensor<i32>) -> tensor<i64>
                    return %1, %3 : tensor<?xf64, #stablehlo.bounds<4>>, tensor<i64>
                  }
                }
            "},
        );
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let Err(error) =
            client.compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
        else {
            panic!("bounded-dynamic sharded module unexpectedly compiled for two devices");
        };
        let error = error.to_string();
        assert!(error.contains("Shardy propagation only supports ranked tensors with a static shape"), "{error}");

        // A single-device explicit mesh needs no partitioning, so the constrained dynamic value executes.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let (program, _, _, _, _) = program_and_types(mesh);
        assert_eq!(
            execute_mixed_program(
                &execution_client(),
                &program,
                &[MixedValue::Array(vec![1.0, 2.0, 0.0, 0.0], vec![4])],
                &[2],
            ),
            Ok(vec![MixedValue::Array(vec![1.0, 2.0], vec![2]), MixedValue::Dimension(2)]),
        );
    }

    #[test]
    fn test_lower_cast_preserved_placement() {
        use ryft_core::{Layout, LogicalMesh, MeshAxis, Sharding, ShardingDimension, TiledLayout};

        // Carrying an existing sharding forward requests no redistribution: a cast that only changes the layout emits
        // no constraint, whereas a changed explicit placement emits the complete output sharding, including the
        // preserved auto placement, like `reshard`. The plain-program entry point declares the mesh of every sharded
        // atom, so the mesh declaration below does not depend on the cast.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let placed = |dimensions: Vec<ShardingDimension>| {
            ArrayType::new_static(DataType::F64, [4, 4])
                .with_sharding(Sharding::new(mesh.clone(), dimensions).unwrap())
                .unwrap()
        };
        let input_type = placed(vec![ShardingDimension::sharded(["a"]), ShardingDimension::replicated()]);
        let cases = [
            (
                input_type.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]))),
                indoc! {"
                    module {
                      sdy.mesh @mesh = <[\"x\"=2, \"a\"=2]>
                      func.func @main(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
                        %0 = stablehlo.custom_call @LayoutConstraint(%arg0) {\
                            operand_layouts = [dense<[0, 1]> : tensor<2xindex>], \
                            result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x4xf64>) -> tensor<4x4xf64>
                        return %0 : tensor<4x4xf64>
                      }
                    }
                "},
                indoc! {"
                    module {
                      func.func @main(%arg0: tensor<4x4xf64>) -> (tensor<4x4xf64> {mhlo.layout_mode = \"{0,1}\"}) {
                        %0 = stablehlo.custom_call @LayoutConstraint(%arg0) {\
                            operand_layouts = [dense<[0, 1]> : tensor<2xindex>], \
                            result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x4xf64>) -> tensor<4x4xf64>
                        return %0 : tensor<4x4xf64>
                      }
                    }
                "},
            ),
            (
                placed(vec![ShardingDimension::sharded(["x", "a"]), ShardingDimension::replicated()]),
                indoc! {"
                    module {
                      sdy.mesh @mesh = <[\"x\"=2, \"a\"=2]>
                      func.func @main(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
                        %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\", \"a\"}, {}]> : tensor<4x4xf64>
                        return %0 : tensor<4x4xf64>
                      }
                    }
                "},
                indoc! {"
                    module {
                      sdy.mesh @mesh = <[\"x\"=2, \"a\"=2]>
                      func.func @main(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
                        %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\", \"a\"}, {}]> : tensor<4x4xf64>
                        return %0 : tensor<4x4xf64>
                      }
                    }
                "},
            ),
        ];
        for (output_type, expected, expected_executable) in cases {
            assert!(!cast_changes_explicit_placement(&input_type, &input_type));
            assert_eq!(
                cast_changes_explicit_placement(&input_type, &output_type),
                output_type.sharding() != input_type.sharding(),
            );
            for constructor in [CastOperation::checked, CastOperation::unchecked] {
                let mut builder = XlaProgramBuilder::new();
                let input = builder.add_input(input_type.clone());
                let output = builder
                    .add_instruction(
                        constructor(output_type.clone(), vec![input_type.clone()]).unwrap(),
                        vec![],
                        vec![input],
                        None,
                    )
                    .unwrap()[0];
                let program = builder
                    .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                        vec![output],
                        vec![Placeholder],
                        vec![Placeholder],
                    )
                    .unwrap();
                assert_eq!(to_mlir_module_for_plain_program(&program, "main").unwrap(), expected);
                // The executable entry point declares meshes only for casts that emit a constraint; the input's
                // own placement is declared through the executable signature instead.
                let program = program.into_unprojected::<XlaConstant, XlaOperation>().unwrap();
                assert_eq!(
                    to_mlir_module_for_program(
                        &program,
                        &[],
                        &vec![input_type.clone()],
                        &vec![output_type.clone()],
                        "main",
                        None,
                        None,
                    )
                    .unwrap(),
                    expected_executable,
                );
            }
        }
    }

    #[test]
    fn test_lower_cast_preserved_placement_executes() {
        use ryft_core::{Layout, LogicalMesh, MeshAxis, Sharding, ShardingDimension, TiledLayout};

        use crate::ToPjrt;

        // Carrying the auto and explicit placement forward emits no constraint, so every device keeps its own shard
        // and only the physical layout of the result changes.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["a"]), ShardingDimension::replicated()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [4, 4]).with_sharding(sharding.clone()).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let output_type = input_type.clone().with_layout(layout.clone());
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(
                CastOperation::checked(output_type.clone(), vec![input_type.clone()]).unwrap(),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .into_unprojected::<XlaConstant, XlaOperation>()
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[input_type],
            &[output_type],
            "main",
            Some(std::slice::from_ref(&sharding)),
            Some(std::slice::from_ref(&sharding)),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"a\"=2]>
                  func.func @main(%arg0: tensor<4x4xf64> \
                      {sdy.sharding = #sdy.sharding<@mesh, [{\"a\"}, {}], replicated={\"x\"}>}) \
                      -> (tensor<4x4xf64> {mhlo.layout_mode = \"{0,1}\", \
                      sdy.sharding = #sdy.sharding<@mesh, [{\"a\"}, {}], replicated={\"x\"}>}) {
                    %0 = stablehlo.custom_call @LayoutConstraint(%arg0) {\
                        operand_layouts = [dense<[0, 1]> : tensor<2xindex>], \
                        result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x4xf64>) -> tensor<4x4xf64>
                    return %0 : tensor<4x4xf64>
                  }
                }
            "},
        );
        // Devices are ordered along the mesh axes, so the device at `(x, a)` holds rows `2a..2a + 2` whatever `x` is.
        let shards = [0, 1, 0, 1].map(|a| (8 * a + 1..=8 * a + 8).map(f64::from).collect::<Vec<_>>());
        let outputs = execute_sharded_module(&module, &shards, &[2, 4]).unwrap();
        let expected = shards.iter().map(|shard| (shard.clone(), layout.to_pjrt().unwrap())).collect::<Vec<_>>();
        assert_eq!(outputs, expected);
    }

    #[test]
    fn test_lower_cast_output_layout_dynamic() {
        use ryft_core::{Layout, TiledLayout};

        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("n", Some(5)), 3.into()]));
        let output_type = input_type.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(input_type.clone());
            let output = builder
                .add_instruction(
                    constructor(output_type.clone(), vec![input_type.clone()]).unwrap(),
                    vec![],
                    vec![input],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            assert_eq!(
                to_mlir_module_for_plain_program(&program, "main").unwrap(),
                indoc! {"
                    module {
                      func.func @main(%arg0: tensor<?x3xf64, #stablehlo.bounds<4, ?>>) \
                          -> tensor<?x3xf64, #stablehlo.bounds<4, ?>> {
                        %c = stablehlo.constant dense<4> : tensor<i32>
                        %0 = stablehlo.set_dimension_size %arg0, %c, dim = 0 : \
                            (tensor<?x3xf64, #stablehlo.bounds<4, ?>>, tensor<i32>) -> tensor<4x3xf64>
                        %1 = stablehlo.custom_call @LayoutConstraint(%0) {\
                            operand_layouts = [dense<[0, 1]> : tensor<2xindex>], \
                            result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x3xf64>) -> tensor<4x3xf64>
                        %2 = stablehlo.get_dimension_size %arg0, dim = 0 : \
                            (tensor<?x3xf64, #stablehlo.bounds<4, ?>>) -> tensor<i32>
                        %3 = stablehlo.set_dimension_size %1, %2, dim = 0 : \
                            (tensor<4x3xf64>, tensor<i32>) -> tensor<?x3xf64, #stablehlo.bounds<4, ?>>
                        return %3 : tensor<?x3xf64, #stablehlo.bounds<4, ?>>
                      }
                    }
                "},
            );
            let program = program.into_unprojected::<XlaConstant, XlaOperation>().unwrap();
            assert_eq!(
                crate::to_mlir_module_for_program(
                    &program,
                    &[],
                    &vec![input_type.clone()],
                    &vec![output_type.clone()],
                    "main",
                    None,
                    None,
                ),
                Err(LoweringError::UnsupportedOp {
                    op: "non-row-major bounded-dynamic output layout `tiled{0,1}`".to_string(),
                }),
            );
        }
    }

    #[test]
    fn test_lower_cast_widening_with_output_layout() {
        use ryft_core::{Layout, Shape, TiledLayout};

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        // Widening a static axis to a bounded dynamic one pads the physical buffer to the bound, and the requested
        // layout constrains that whole physical buffer before the runtime extent is restored on the constrained value.
        // A second cast drops the layout again so that the executable boundary stays row-major.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let exact = ArrayType::new_static(DataType::F64, [2, 3]);
        let symbolic = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension(), Dimension::Static(3)]));
        let laid_out = symbolic.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(exact.clone().into());
        let witness = builder.add_input(symbolic.clone().into());
        let widened = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::checked(
                    laid_out.clone().into(),
                    vec![exact.clone().into(), symbolic.clone().into()],
                )
                .unwrap(),
                Vec::new(),
                vec![input, witness],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::unchecked(symbolic.clone().into(), vec![laid_out.into()]).unwrap(),
                Vec::new(),
                vec![widened],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let module =
            to_mlir_module_for_program(&program, &[], &[exact, symbolic.clone()], &[symbolic], "main", None, None)
                .unwrap();
        std::fs::write("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/widening_1.txt", &module).unwrap();
        assert_eq!(module.matches("@ryft.assert").count(), 1, "{module}");
        assert_eq!(module.matches("@LayoutConstraint").count(), 1, "{module}");
        let client = execution_client();
        let inputs = [
            MixedValue::Array((1..=6).map(f64::from).collect(), vec![2, 3]),
            MixedValue::Array(vec![0.0; 12], vec![4, 3]),
        ];
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[2]),
            Ok(vec![MixedValue::Array((1..=6).map(f64::from).collect(), vec![2, 3]), MixedValue::Dimension(2)]),
        );
        let error = execute_mixed_program(&client, &program, &inputs, &[3]).unwrap_err().to_string();
        assert!(error.contains("input dimension `n` must have one extent in `cast`"), "{error}");
    }

    #[test]
    fn test_lower_cast_output_layout_unsupported() {
        use ryft_core::{Layout, StridedLayout, Tile, TileDimension, TiledLayout};

        let input_type = ArrayType::new_static(DataType::F64, [2]);
        for (layout, kind) in [
            (Layout::Strided(StridedLayout::new(vec![8])), "strided"),
            (Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])), "tiled"),
        ] {
            let output_type = input_type.clone().with_layout(layout.clone());
            let expected = Err(LoweringError::UnsupportedOp { op: format!("{kind} output layout `{layout}`") });
            for constructor in [CastOperation::checked, CastOperation::unchecked] {
                let mut builder = XlaProgramBuilder::new();
                let input = builder.add_input(input_type.clone());
                let output = builder
                    .add_instruction(
                        constructor(output_type.clone(), vec![input_type.clone()]).unwrap(),
                        vec![],
                        vec![input],
                        None,
                    )
                    .unwrap()[0];
                let program = builder
                    .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                        vec![output],
                        vec![Placeholder],
                        vec![Placeholder],
                    )
                    .unwrap();
                assert_eq!(to_mlir_module_for_plain_program(&program, "main"), expected);
                let program = program.into_unprojected::<XlaConstant, XlaOperation>().unwrap();
                assert_eq!(
                    to_mlir_module_for_program(
                        &program,
                        &[],
                        &vec![input_type.clone()],
                        &vec![output_type.clone()],
                        "main",
                        None,
                        None,
                    ),
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_lower_cast_repeated_static_check() {
        use ryft_core::Shape;

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension(); 2]));
        let output_type = ArrayType::new_static(DataType::F64, [2, 2]);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(
                CastOperation::checked(output_type.clone(), vec![input_type]).unwrap(),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[n, n] .
                let %1:f64[2, 2] = cast [type=f64[2, 2], inputs=[f64[n, n]], checked=true] %0
                in (%1)"},
        );
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        std::fs::write("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/repeated_1.txt", &module).unwrap();
        assert_eq!(module.matches("@ryft.assert").count(), 1, "{module}");
        let program = program.into_unprojected::<XlaConstant, XlaOperation>().unwrap();
        let input = MixedValue::Array(vec![1f64; 16], vec![4, 4]);
        let client = execution_client();
        assert_eq!(
            execute_mixed_program(&client, &program, std::slice::from_ref(&input), &[2, 2]),
            Ok(vec![MixedValue::Array(vec![1f64; 4], vec![2, 2])]),
        );
        assert!(execute_mixed_program(&client, &program, &[input], &[3, 3]).is_err());
    }

    #[test]
    fn test_lower_cast_omits_proven_witness_checks() {
        use ryft_core::Shape;

        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let [a, b] = ["a", "b"].map(|name| DimensionType::new(name, bounds));
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let value = builder.add_input(input_type.clone());
        let witness = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(
                CastOperation::checked(output_type, vec![input_type.clone(), input_type]).unwrap(),
                Vec::new(),
                vec![value, witness],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[a, b], %1:f32[a, b] .
                let %2:f32[a, 2] = cast [type=f32[a, 2], inputs=[f32[a, b], f32[a, b]], checked=true] %0 %1
                in (%2)"},
        );
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        std::fs::write("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/witness_1.txt", &module).unwrap();
        // Only b == 2 is conditional. Both input arrays already carry the same a and b identities.
        assert_eq!(module.matches("@ryft.assert").count(), 1, "{module}");
    }

    #[test]
    fn test_lower_cast_dimension_witness_checks() {
        use ryft_core::Shape;

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let [a, b, c] = ["a", "b", "c"].map(|name| DimensionType::new(name, bounds));
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![a.simplified_dimension(), Dimension::Static(2)]));
        let client = execution_client();
        // An array axis and a first-class dimension can share an actual identity. Equal non-singleton bounds on an
        // independent identity prove nothing, so that witness must retain its comparison and fail for extent 3.
        for (witness_type, comparisons) in [(b.clone(), 1), (c, 2)] {
            let mut builder = CompositeXlaProgramBuilder::new();
            let witness = builder.add_input(witness_type.clone().into());
            let value = builder.add_input(input_type.clone().into());
            let output = builder
                .add_instruction(
                    CastOperation::<ArrayIrType>::checked(
                        output_type.clone().into(),
                        vec![input_type.clone().into(), b.clone().into()],
                    )
                    .unwrap(),
                    Vec::new(),
                    vec![value, witness],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            let expected = formatdoc! {"
            lambda %0:{witness_type}, %1:f64[a, b] .
            let %2:f64[a, 2] = cast [type=f64[a, 2], inputs=[f64[a, b], dimension<b ∈ [1, 5)>], checked=true] %1 %0
            in (%2)"};
            assert_eq!(program.to_string(), expected);
            let module = to_mlir_module_for_program(
                &program,
                &[],
                &vec![ArrayType::scalar(DataType::I64), input_type.clone()],
                &vec![output_type.clone()],
                "main",
                None,
                None,
            )
            .unwrap();
            std::fs::write(format!("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/dimension_witness_{comparisons}.txt"), &module).unwrap();
            assert_eq!(module.matches("@ryft.assert").count(), comparisons, "{module}");
            let input = MixedValue::Array(vec![1.0; 16], vec![4, 4]);
            assert_eq!(
                execute_mixed_program(&client, &program, &[MixedValue::Dimension(2), input.clone()], &[2, 2]),
                Ok(vec![MixedValue::Array(vec![1.0; 4], vec![2, 2]), MixedValue::Dimension(2)]),
            );
            if comparisons == 2 {
                let error = execute_mixed_program(&client, &program, &[MixedValue::Dimension(3), input], &[2, 2])
                    .unwrap_err()
                    .to_string();
                assert!(error.contains("input dimension `b` must have one extent in `cast`"), "{error}");
            }
        }
    }

    #[test]
    fn test_lower_unchecked_cast_with_witness() {
        use ryft_core::{Cast, Shape, TracingContext};

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()])).into();
        let (_, program) = TracingContext::<XlaConstant, XlaOperation>::trace(
            |(value, witness)| value.cast_unchecked(symbolic.clone(), &[witness]),
            (ArrayIrType::from(ArrayType::new_static(DataType::F64, [2])), symbolic.clone()),
        )
        .unwrap();
        assert_eq!(
            execute_mixed_program(
                &execution_client(),
                &program.to_flat_program(),
                &[MixedValue::Array(vec![1.0, 2.0], vec![2]), MixedValue::Array(vec![3.0, 4.0, 0.0, 0.0], vec![4]),],
                &[2]
            ),
            Ok(vec![MixedValue::Array(vec![1.0, 2.0], vec![2]), MixedValue::Dimension(2)]),
        );
    }

    #[test]
    fn test_lower_checked_cast_region_checks() {
        use ryft_core::{ConditionOperation, ScanOperation, Shape, WhileOperation};

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::operations::jit_call::JitCallOperation;
        use crate::tests::execution_client;

        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()])).into();
        let body = |index, checked| {
            let mut builder = CompositeXlaProgramBuilder::new();
            if index {
                builder.add_input(ArrayType::scalar(DataType::I64).into());
            }
            let value = builder.add_input(symbolic.clone());
            if checked {
                builder
                    .add_instruction(
                        CastOperation::<ArrayIrType>::checked(
                            ArrayType::new_static(DataType::F64, [2]).into(),
                            vec![symbolic.clone()],
                        )
                        .unwrap(),
                        vec![],
                        vec![value],
                        None,
                    )
                    .unwrap();
            }
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![value],
                    vec![Placeholder; if index { 2 } else { 1 }],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let client = execution_client();
        // Exercise repeated calls, selected and unselected branches, zero-trip loops, a rolled scan and a fully
        // unrolled scan. The checked cast's unused result must never make its precondition disappear.
        for wrapper in 0..5 {
            for entered in [false, true] {
                if wrapper == 0 && !entered {
                    continue;
                }
                let mut builder = CompositeXlaProgramBuilder::new();
                let input = builder.add_input(symbolic.clone());
                let region = builder.import_program(body(wrapper >= 3, true));
                let output = match wrapper {
                    0 => {
                        for _ in 0..2 {
                            builder
                                .add_instruction(
                                    XlaOperation::JitCall(JitCallOperation::new(0)),
                                    vec![region],
                                    vec![input],
                                    None,
                                )
                                .unwrap();
                        }
                        input
                    }
                    1 => {
                        let other = builder.import_program(body(false, false));
                        let predicate = builder.add_constant(XlaConstant::Boolean(entered));
                        builder
                            .add_instruction(
                                XlaOperation::Condition(ConditionOperation::new()),
                                vec![region, other],
                                vec![predicate, input],
                                None,
                            )
                            .unwrap()[0]
                    }
                    2 => {
                        let mut condition = CompositeXlaProgramBuilder::new();
                        condition.add_input(symbolic.clone());
                        let predicate = condition.add_constant(XlaConstant::Boolean(entered));
                        let condition = condition
                            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                                vec![predicate],
                                vec![Placeholder],
                                vec![Placeholder],
                            )
                            .unwrap();
                        let condition = builder.import_program(condition);
                        builder
                            .add_instruction(
                                XlaOperation::While(WhileOperation::new().with_iteration_bound(1).unwrap()),
                                vec![condition, region],
                                vec![input],
                                None,
                            )
                            .unwrap()[0]
                    }
                    _ => builder
                        .add_instruction(
                            XlaOperation::Scan(
                                ScanOperation::new(1, Dimension::Static(if entered { 2 } else { 0 }))
                                    .with_unroll(if wrapper == 4 { 3 } else { 1 })
                                    .unwrap(),
                            ),
                            vec![region],
                            vec![input],
                            None,
                        )
                        .unwrap()[0],
                };
                let program = builder
                    .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                    .unwrap();
                for extent in [2, 3] {
                    let result = execute_mixed_program(
                        &client,
                        &program,
                        &[MixedValue::Array(vec![1.0, 2.0, 3.0, 0.0], vec![4])],
                        &[extent],
                    );
                    if entered && extent == 3 {
                        let error = result.unwrap_err().to_string();
                        assert!(error.contains("must have the declared extent in `cast`"), "{error}");
                    } else {
                        assert_eq!(
                            result,
                            Ok(vec![
                                MixedValue::Array(
                                    (1..=extent).map(|value| value as f64).collect(),
                                    vec![extent as u64]
                                ),
                                MixedValue::Dimension(extent as i64),
                            ]),
                            "wrapper={wrapper}, entered={entered}, extent={extent}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_lower_guarded_cast() {
        use ryft_core::{
            BatchAxis, BatchedProgram, Layout, ProgramBatchingOutputAxesPolicy, Shape, ShardingDimension, TiledLayout,
        };

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let bounds = DimensionBounds::non_negative(Some(5)).unwrap();
        let [a, b, p] = ["a", "b", "p"].map(|name| DimensionType::new(name, bounds));
        // A declared physical layout disappears when batching inserts an axis. The guard and cast must validate
        // the source declaration before lifting instead of requiring that layout on reconstructed per-item types.
        let array = ArrayType::new(DataType::F64, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]))
            .with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![])));
        let relation = ArrayType::new(DataType::F64, Shape::new(vec![p.simplified_dimension(); 2]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(array.clone().into());
        let input = builder
            .add_instruction(
                GuardOperation::new("square", vec![relation.clone()])
                    .with_retained_identities(vec![a.to_variable(), b.to_variable()]),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        // Exercise the homogeneous-array rule inside the composite program, including empty dense shapes.
        let output = builder
            .add_instruction(CastOperation::checked(array.clone(), vec![array]).unwrap(), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let client = execution_client();
        let input = MixedValue::Array(vec![1f64; 16], vec![4, 4]);
        assert_eq!(
            execute_mixed_program(&client, &program, std::slice::from_ref(&input), &[2, 2]),
            Ok(vec![MixedValue::Array(vec![1f64; 4], vec![2, 2]), MixedValue::Dimension(2), MixedValue::Dimension(2)]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, std::slice::from_ref(&input), &[0, 0]),
            Ok(vec![MixedValue::Array(Vec::new(), vec![0, 0]), MixedValue::Dimension(0), MixedValue::Dimension(0)]),
        );
        let error = execute_mixed_program(&client, &program, std::slice::from_ref(&input), &[2, 3])
            .unwrap_err()
            .to_string();
        assert!(error.contains("input dimension `p` must have one extent at the entry of `square`"), "{error}");
        let error = execute_mixed_program(&client, &program, std::slice::from_ref(&input), &[0, 3])
            .unwrap_err()
            .to_string();
        assert!(error.contains("input dimension `p` must have one extent at the entry of `square`"), "{error}");

        // Differentiation retains the checked primal relation even when the tangent already has the output type.
        let differentiated = program.jvp().unwrap();
        let inputs = [input.clone(), input];
        assert_eq!(
            execute_mixed_program(&client, &differentiated, &inputs, &[2, 2, 2, 2]),
            Ok(vec![
                MixedValue::Array(vec![1f64; 4], vec![2, 2]),
                MixedValue::Array(vec![1f64; 4], vec![2, 2]),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
            ]),
        );
        let error = execute_mixed_program(&client, &differentiated, &inputs, &[2, 3, 2, 3]).unwrap_err().to_string();
        assert!(error.contains("input dimension `p` must have one extent at the entry of `square`"), "{error}");

        // Batching adds a separate leading dimension to the guard and cast. The two item axes must still
        // agree, including when the mapped extent is zero and there are no elements to inspect.
        let batched = program
            .batched_with_threaded_extent(
                DimensionType::new("batch", DimensionBounds::non_negative(Some(4)).unwrap()),
                ShardingDimension::replicated(),
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let inputs = [MixedValue::Dimension(2), MixedValue::Array(vec![1f64; 48], vec![3, 4, 4])];
        assert_eq!(
            execute_mixed_program(&client, &batched, &inputs, &[2, 2, 2]),
            Ok(vec![
                MixedValue::Dimension(2),
                MixedValue::Array(vec![1f64; 8], vec![2, 2, 2]),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
            ]),
        );
        let error = execute_mixed_program(&client, &batched, &inputs, &[2, 2, 3]).unwrap_err().to_string();
        assert!(error.contains("must have one extent at the entry of `square`"), "{error}");
        let inputs = [MixedValue::Dimension(0), inputs[1].clone()];
        let error = execute_mixed_program(&client, &batched, &inputs, &[0, 2, 3]).unwrap_err().to_string();
        assert!(error.contains("must have one extent at the entry of `square`"), "{error}");
    }

    #[test]
    fn test_lower_checked_cast_layout_and_sharding() {
        use ryft_core::{
            Layout, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension, StridedLayout, TiledLayout,
        };

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let plain = ArrayType::new_static(DataType::F32, [2]);
        let contiguous = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let strided = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = plain
            .clone()
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let replicated = plain
            .clone()
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap())
            .unwrap();

        // Different specified representations are admitted before lowering materializes the requested output.
        for (input_type, output_type) in [(contiguous, strided), (sharded, replicated)] {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(input_type.clone().into());
            let output = builder
                .add_instruction(
                    CastOperation::<ArrayIrType>::checked(output_type.clone().into(), vec![input_type.into()]).unwrap(),
                    Vec::new(),
                    vec![input],
                    None,
                )
                .unwrap();
            assert_eq!(output.len(), 1);
        }

        // Independent layout and sharding declarations can describe the same value, without a representation change.
        let plain = ArrayType::new_static(DataType::F64, [2]);
        let input_type = plain.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![])));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let output_type = plain
            .clone()
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap())
            .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::checked(output_type.into(), vec![input_type.into()]).unwrap(),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let client = execution_client();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(vec![1.0, 2.0], vec![2])], &[],),
            Ok(vec![MixedValue::Array(vec![1.0, 2.0], vec![2])]),
        );
    }

    #[test]
    fn test_lower_checked_cast_ambient_identities() {
        use ryft_core::Shape;

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let [a, b] = ["a", "b"].map(|name| DimensionType::new(name, bounds));
        let array = ArrayType::new(DataType::F64, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let swapped =
            ArrayType::new(DataType::F64, Shape::new(vec![b.simplified_dimension(), a.simplified_dimension()]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(array.clone().into());
        let output = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::checked(swapped.into(), vec![array.clone().into()]).unwrap(),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let client = execution_client();
        let input = MixedValue::Array(vec![1f64; 16], vec![4, 4]);
        assert_eq!(
            execute_mixed_program(&client, &program, std::slice::from_ref(&input), &[2, 2]),
            Ok(vec![MixedValue::Array(vec![1f64; 4], vec![2, 2]), MixedValue::Dimension(2), MixedValue::Dimension(2)]),
        );
        let error = execute_mixed_program(&client, &program, std::slice::from_ref(&input), &[2, 3])
            .unwrap_err()
            .to_string();
        assert!(error.contains("must have one extent in `cast`"), "{error}");

        // The primal of a differentiated cast retains that same source declaration. Tangent casts reverse neither
        // the ambient assignments nor their dependence on the checked primal output.
        let differentiated = program.jvp().unwrap();
        assert_eq!(
            execute_mixed_program(&client, &differentiated, &[input.clone(), input.clone()], &[2, 2, 2, 2]),
            Ok(vec![
                MixedValue::Array(vec![1f64; 4], vec![2, 2]),
                MixedValue::Array(vec![1f64; 4], vec![2, 2]),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
            ]),
        );
        let error = execute_mixed_program(&client, &differentiated, &[input.clone(), input.clone()], &[2, 3, 2, 3])
            .unwrap_err()
            .to_string();
        assert!(error.contains("must have one extent in `cast`"), "{error}");

        // Keeping the original source and output types accepts rectangular shapes.
        let mut builder = CompositeXlaProgramBuilder::new();
        let value = builder.add_input(array.clone().into());
        let output = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::checked(array.clone().into(), vec![array.into()]).unwrap(),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            execute_mixed_program(&client, &program, &[input], &[2, 3]),
            Ok(vec![MixedValue::Array(vec![1f64; 6], vec![2, 3]), MixedValue::Dimension(2), MixedValue::Dimension(3)]),
        );
    }

    #[test]
    fn test_lower_checked_cast_mixed_dimensions() {
        use ryft_core::Shape;

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let [a, b] = ["a", "b"].map(|name| DimensionType::new(name, bounds));
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![b.simplified_dimension(), Dimension::Static(3)]));
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), a.simplified_dimension()]));
        let witness_type = ArrayType::new(DataType::F64, Shape::new(vec![a.simplified_dimension()]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let witness = builder.add_input(witness_type.clone().into());
        let output = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::checked(output_type.into(), vec![input_type.into(), witness_type.into()])
                    .unwrap(),
                Vec::new(),
                vec![input, witness],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let client = execution_client();
        let inputs = [MixedValue::Array(vec![1f64; 12], vec![4, 3]), MixedValue::Array(vec![1f64; 4], vec![4])];
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[3, 3]),
            Ok(vec![MixedValue::Array(vec![1f64; 9], vec![3, 3]), MixedValue::Dimension(3)]),
        );
        let error = execute_mixed_program(&client, &program, &inputs, &[4, 3]).unwrap_err().to_string();
        assert!(error.contains("must have the declared extent in `cast`"), "{error}");
        let error = execute_mixed_program(&client, &program, &inputs, &[3, 2]).unwrap_err().to_string();
        assert!(error.contains("must have one extent in `cast`"), "{error}");
    }

    #[test]
    fn test_lower_checked_cast_static_witnesses() {
        use ryft_core::Shape;

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        // A static witness establishes ambient b = 3 before type inference compares the source and target dimensions.
        // The symbolic input still needs its runtime check, before conversion slices its physical bound to 3.
        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let b = DimensionType::new("b", bounds);
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![b.simplified_dimension(), Dimension::Static(3)]));
        let output_type = ArrayType::new_static(DataType::F64, [3, 3]);
        let witness_type = ArrayType::new(DataType::F64, Shape::new(vec![b.simplified_dimension()]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let witness = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let output = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::checked(
                    output_type.into(),
                    vec![input_type.clone().into(), witness_type.into()],
                )
                .unwrap(),
                Vec::new(),
                vec![input, witness],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[input_type, ArrayType::new_static(DataType::F64, [3])],
            &[ArrayType::new_static(DataType::F64, [3, 3])],
            "main",
            None,
            None,
        )
        .unwrap();
        std::fs::write("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/static_witnesses_1.txt", &module).unwrap();
        assert_eq!(module.matches("@ryft.assert").count(), 1, "{module}");
        let client = execution_client();
        let inputs = [MixedValue::Array(vec![1f64; 12], vec![4, 3]), MixedValue::Array(vec![1f64; 3], vec![3])];
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[3]),
            Ok(vec![MixedValue::Array(vec![1f64; 9], vec![3, 3])]),
        );
        let error = execute_mixed_program(&client, &program, &inputs, &[4]).unwrap_err().to_string();
        assert!(error.contains("must have the declared extent in `cast`"), "{error}");
    }

    #[test]
    fn test_lower_guarded_cast_discarded_output() {
        use ryft_core::Shape;

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::tests::execution_client;

        let bounds = DimensionBounds::non_negative(Some(5)).unwrap();
        let [a, b, p] = ["a", "b", "p"].map(|name| DimensionType::new(name, bounds));
        let array = ArrayType::new(DataType::F64, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let relation = ArrayType::new(DataType::F64, Shape::new(vec![p.simplified_dimension(); 2]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(array.clone().into());
        let input = builder
            .add_instruction(
                GuardOperation::new("square", vec![relation.clone()])
                    .with_retained_identities(vec![a.to_variable(), b.to_variable()]),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(CastOperation::checked(array.clone(), vec![array]).unwrap(), Vec::new(), vec![input], None)
            .unwrap();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        let client = execution_client();
        let input = MixedValue::Array(vec![1f64; 16], vec![4, 4]);
        assert_eq!(execute_mixed_program(&client, &program, std::slice::from_ref(&input), &[2, 2]), Ok(Vec::new()));
        let error = execute_mixed_program(&client, &program, &[input], &[2, 3]).unwrap_err().to_string();
        assert!(error.contains("input dimension `p` must have one extent at the entry of `square`"), "{error}");
    }

    #[test]
    fn test_lower_cast_layout_after_bound_change() {
        use std::sync::OnceLock;

        use ryft_core::{CustomCallOperation, Layout, TiledLayout};
        use ryft_pjrt::extensions::ffi::{
            FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits, FfiInput, FfiOutput, FfiTypeId,
            XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Handler,
        };
        use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};

        const TARGET: &str = "ryft.test.physical_f64_copy";
        static REGISTRATION: OnceLock<Result<(), ryft_pjrt::Error>> = OnceLock::new();

        /// Copies the physical storage of the single `f64` operand, in memory order, into the rank-1 `f64` result.
        unsafe extern "C" fn handler(raw: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
            // SAFETY: XLA provides a valid call frame, operand buffer, and result allocation for this invocation, and
            // the program below declares a result with as many elements as its operand.
            unsafe {
                let Ok(frame) = FfiCallFrame::from_c_api(raw) else { return std::ptr::null_mut() };
                if frame.register_metadata(FfiTypeId::default()) || frame.stage() != FfiExecutionStage::Execution {
                    return std::ptr::null_mut();
                }
                let Ok(api) = frame.api() else { return std::ptr::null_mut() };
                let (Ok(FfiInput::Buffer { buffer: input }), Ok(FfiOutput::Buffer { buffer: output })) =
                    (frame.input(0), frame.output(0))
                else {
                    return FfiError::invalid_argument("expected one operand and one result buffer").to_c_api(api);
                };
                let length = output.dimensions().iter().product::<i64>() as usize;
                std::ptr::copy_nonoverlapping(input.data() as *const f64, output.data() as *mut f64, length);
                std::ptr::null_mut()
            }
        }

        // The first cast constrains the physical buffer to its new column-major layout. The second cast keeps that
        // layout but narrows the bounded dynamic axis, which slices a new buffer that it does not constrain again.
        // Layouts are only observable where a value crosses a boundary (executable inputs and outputs, and custom-call
        // operands and results), and every such boundary states the layout of its own declared type. XLA rejects
        // bounded dynamic custom-call operands and non-row-major bounded dynamic executable outputs, so only a static
        // result of such a cast can be observed, and its consumer still receives the declared column-major storage.
        let client = load_cpu_plugin()
            .unwrap()
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        REGISTRATION
            .get_or_init(|| {
                client.register_ffi_handler(
                    TARGET,
                    client.platform_name()?.into_owned(),
                    FfiHandler::from(handler as XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                )
            })
            .clone()
            .unwrap();
        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let source = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension(), Dimension::Static(3)]));
        let laid_out = source.clone().with_layout(layout.clone());
        let narrowed = ArrayType::new_static(DataType::F64, [2, 3]).with_layout(layout);
        let copied = ArrayType::new_static(DataType::F64, [6]);
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(source.clone().into());
        let value = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::unchecked(laid_out.clone().into(), vec![source.clone().into()]).unwrap(),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let value = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::unchecked(narrowed.into(), vec![laid_out.into()]).unwrap(),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(CustomCallOperation::new(TARGET, vec![copied.clone()]), Vec::new(), vec![value], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(&program, &[], &[source], &[copied], "main", None, None).unwrap();
        std::fs::write("/private/tmp/claude-501/-Users-eaplatanios-Development-Repositories-ryft-1/16f4a8a6-5def-45fa-88f5-0b02429cd832/scratchpad/groupG3/dumps/layout_after_bound_change.txt", &module).unwrap();
        assert_eq!(
            execute_mixed_program(
                &client,
                &program,
                &[MixedValue::Array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![4, 3])],
                &[2],
            ),
            Ok(vec![MixedValue::Array(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![6])]),
        );
    }

    #[test]
    fn test_lower_cast_rejects_bound_manual_output_placement() {
        use ryft_core::{
            Cast, LogicalMesh, MeshAxis, Sharding, ShardingDimension, TracedShardMap, Typed, trace_shard_map,
        };

        use crate::XlaDomain;
        use crate::operations::sharding::shard_map::to_mlir_module as to_mlir_module_for_shard_map;

        // A body value that does not vary along the bound manual axis `x` preserves no definition of it, so a cast may
        // name a mesh that redefines `x` as explicit. Sharding a local dimension over `x` would request a placement
        // that only the region boundary owns, and lowering rejects it instead of stripping `x` from the constraint.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |input| {
                let local_type = input.r#type().into_owned();
                let explicit_type = local_type
                    .clone()
                    .with_sharding(Sharding::new(explicit_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
                    .unwrap();
                input.cast(explicit_type, &[]).unwrap().cast(local_type, &[]).unwrap()
            },
            ArrayType::new_static(DataType::F64, [4]),
            mesh,
            replicated.clone(),
            replicated,
        )
        .unwrap();
        assert!(matches!(
            to_mlir_module_for_shard_map(&traced, "main"),
            Err(LoweringError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == "`cast` cannot assign bound manual mesh axes to local output dimensions",
        ));
    }

    #[test]
    fn test_cast_changes_explicit_placement() {
        use ryft_core::{LogicalMesh, MeshAxis, ShardingDimension};

        let explicit_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let resized_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let extended_auto_mesh = LogicalMesh::new(vec![
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("b", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let unsharded = ArrayType::new_static(DataType::F64, [4, 4]);
        let placed = |mesh: &LogicalMesh, first: ShardingDimension| {
            unsharded
                .clone()
                .with_sharding(Sharding::new(mesh.clone(), vec![first, ShardingDimension::replicated()]).unwrap())
                .unwrap()
        };
        let sharded = |axes: &[&str]| ShardingDimension::sharded(axes.iter().copied());
        let replicated = ShardingDimension::replicated;

        // An output without a sharding stays untracked, whatever the input placement.
        assert!(!cast_changes_explicit_placement(&unsharded, &unsharded));
        assert!(!cast_changes_explicit_placement(&placed(&explicit_mesh, sharded(&["x"])), &unsharded));

        // An unsharded input acquires a tracked placement only when the output mesh has explicit axes.
        assert!(cast_changes_explicit_placement(&unsharded, &placed(&explicit_mesh, replicated())));
        assert!(!cast_changes_explicit_placement(&unsharded, &placed(&auto_mesh, sharded(&["a"]))));

        // Carrying a mesh forward requests a redistribution only when the placement over explicit axes conflicts.
        assert!(!cast_changes_explicit_placement(
            &placed(&explicit_mesh, sharded(&["x", "a"])),
            &placed(&explicit_mesh, sharded(&["x", "a"])),
        ));
        assert!(cast_changes_explicit_placement(
            &placed(&explicit_mesh, sharded(&["a"])),
            &placed(&explicit_mesh, sharded(&["x", "a"])),
        ));

        // A mesh change requests one whenever either mesh has explicit axes, but never between auto-only meshes.
        assert!(cast_changes_explicit_placement(
            &placed(&explicit_mesh, sharded(&["a"])),
            &placed(&resized_mesh, sharded(&["a"])),
        ));
        assert!(cast_changes_explicit_placement(
            &placed(&explicit_mesh, sharded(&["a"])),
            &placed(&auto_mesh, sharded(&["a"])),
        ));
        assert!(!cast_changes_explicit_placement(
            &placed(&auto_mesh, sharded(&["a"])),
            &placed(&extended_auto_mesh, sharded(&["a"])),
        ));
    }
}
