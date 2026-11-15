use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, ComparisonDirection, DOT_OPERATION_NAME, DataType, DotAlgorithmPreset, DotDimensionNumbers,
    DotInputPrecision, DotOperation, DotPrecision, ProgramError, RAGGED_DOT_OPERATION_NAME, RaggedDotMode,
    RaggedDotOperation, Shape,
};
use ryft_mlir::dialects::stable_hlo::{DotAlgorithmAttributeRef, Precision};
use ryft_mlir::dialects::{chlo, stable_hlo};
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, Float8TypeRef, FloatTypeRef, LocationRef, Operation as MlirOperation,
    Region, Type as MlirType, Value as MlirValue, ValueRef,
};

use crate::lowering::types::{
    lower_element_type, lower_physical_bound_value, lower_restore_dynamic_dimensions, lower_tensor_type,
    lower_u64_scalar_constant, lower_unplaced_constant_output, physical_bound_type,
};
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError};

/// XLA representation used when lowering ragged dots. The selection applies to a whole compilation and is made with
/// [`Options::with_ragged_dot_lowering_strategy`](crate::Options::with_ragged_dot_lowering_strategy).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum RaggedDotLoweringStrategy {
    /// Expand ragged dots into masking plus ordinary dense generalized dots. This is the default used by
    /// [`Options::new`](crate::Options::new).
    #[default]
    Decomposition,

    /// Emit dedicated CHLO `ragged_dot` instructions.
    Instruction,
}

impl LowerableOperation for DotOperation {
    // Lower a dot to one `stablehlo.dot_general` that carries its per-input precisions and its algorithm. An algorithm
    // may compute with other element types than the inputs and the output have, in which case the inputs are converted
    // before the contraction and its result is converted after it, exactly like JAX's `_dot_general_lower`.
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 2, ProgramError);
        check_count!("input", input_types, 2, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let precision = self.precision();

        // XLA's CPU compiler silently ignores the algorithm of a dot, so a CPU target accepts only the presets whose
        // results it reproduces and rejects every other algorithm instead of computing something else. This is the
        // allowlist of JAX's `_handle_dot_precision`, without its `DEFAULT` preset, which is the default precision.
        let targets_cpu = lowerer
            .collective_state()
            .target_platform()
            .is_some_and(|platform| platform.eq_ignore_ascii_case("cpu"));
        let supported_on_cpu = matches!(
            precision,
            DotPrecision::Inputs { .. }
                | DotPrecision::Preset(
                    DotAlgorithmPreset::F16_F16_F16
                        | DotAlgorithmPreset::F32_F32_F32
                        | DotAlgorithmPreset::F64_F64_F64
                        | DotAlgorithmPreset::BF16_BF16_F32
                        | DotAlgorithmPreset::BF16_BF16_F32_X3
                        | DotAlgorithmPreset::BF16_BF16_F32_X6,
                ),
        );
        if targets_cpu && !supported_on_cpu {
            return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                operation: DOT_OPERATION_NAME.to_string(),
                reason: format!("the algorithm `{precision}` is not supported on CPU"),
            }));
        }

        // The precision selects the element types that the contraction computes with. Per-input precisions keep the
        // types of the operation, so they emit no conversions.
        let output_type = &output_types[0];
        let (left_type, right_type, result_type) = precision.computation_types(
            input_types[0].data_type(),
            input_types[1].data_type(),
            output_type.data_type(),
        );
        let mut inputs = [input_values[0], input_values[1]];
        for ((input, input_type), computation_type) in inputs.iter_mut().zip(input_types).zip([left_type, right_type]) {
            if input_type.data_type() != computation_type {
                let converted_type = lowerer.lower_tensor_type(&input_type.clone().with_data_type(computation_type))?;
                let converted = lowerer.block().append_operation(stable_hlo::convert(
                    *input,
                    converted_type,
                    lowerer.location(),
                )?)?;
                *input = converted.result(0).unwrap().as_ref();
            }
        }

        let dimension_numbers = self.dimensions();
        let dimensions = lowerer.context().stable_hlo_dot_dimensions(
            dimension_numbers.lhs_batching_dimensions(),
            dimension_numbers.rhs_batching_dimensions(),
            dimension_numbers.lhs_contracting_dimensions(),
            dimension_numbers.rhs_contracting_dimensions(),
        )?;
        let lower_input_precision = |precision: DotInputPrecision| match precision {
            DotInputPrecision::Default => Precision::Default,
            DotInputPrecision::High => Precision::High,
            DotInputPrecision::Highest => Precision::Highest,
        };
        let (left_precision, right_precision) = precision.input_precisions();
        let algorithm = lower_dot_algorithm(precision, left_type, right_type, lowerer.context())?;
        let result_tensor_type = lowerer.lower_tensor_type(&output_type.clone().with_data_type(result_type))?;
        let result = lowerer.block().append_operation(stable_hlo::dot_general(
            inputs[0],
            inputs[1],
            dimensions,
            Some((lower_input_precision(left_precision), lower_input_precision(right_precision))),
            algorithm,
            result_tensor_type,
            lowerer.location(),
        )?)?;
        let result = result.result(0).unwrap().as_ref();
        if result_type == output_type.data_type() {
            return Ok(vec![result]);
        }
        let output_tensor_type = lowerer.lower_tensor_type(output_type)?;
        let converted =
            lowerer
                .block()
                .append_operation(stable_hlo::convert(result, output_tensor_type, lowerer.location())?)?;
        Ok(vec![converted.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for RaggedDotOperation {
    // Lower a ragged dot either to its portable masked dense decomposition or to the opt-in CHLO instruction.
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 3, ProgramError);
        check_count!("input", input_types, 3, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let dimensions = self.dimensions();
        if lowerer.collective_state().ragged_dot_lowering_strategy() == RaggedDotLoweringStrategy::Instruction {
            // JAX applies an explicit transpose of the right-hand side input only for its TPU instruction path as a
            // backend layout workaround. Ryft emits the canonical CHLO contract and leaves target specific layout
            // selection to legalization.
            let dot = dimensions.dot_dimensions();
            let attribute = lowerer.context().chlo_ragged_dot_dimensions(
                dot.lhs_batching_dimensions(),
                dot.rhs_batching_dimensions(),
                dot.lhs_contracting_dimensions(),
                dot.rhs_contracting_dimensions(),
                &[dimensions.lhs_ragged_dimension()],
                dimensions.rhs_group_dimension().as_slice(),
            )?;
            let result = lowerer.block().append_operation(chlo::ragged_dot(
                input_values[0],
                input_values[1],
                input_values[2],
                attribute,
                Some((chlo::Precision::Default, chlo::Precision::Default)),
                lower_tensor_type(&output_types[0], lowerer.context(), lowerer.location())?,
                lowerer.location(),
            )?)?;
            return Ok(vec![result.result(0).unwrap().as_ref()]);
        }

        let left_type = &input_types[0];
        let right_type = &input_types[1];
        let group_sizes_type = &input_types[2];
        let mode = dimensions.mode(left_type.rank()).map_err(ProgramError::from)?;
        if mode == RaggedDotMode::Batch {
            return Ok(vec![lower_dense_ragged_dot(
                input_values[0],
                input_values[1],
                dimensions.dot_dimensions(),
                &output_types[0],
                &mut lowerer.block(),
                lowerer.context(),
                lowerer.location(),
            )?]);
        }
        let left_ragged_axis = dimensions.lhs_ragged_dimension();
        let left_prefix = dimensions.group_sizes_prefix_dimensions(left_type.rank()).map_err(ProgramError::from)?;
        let physical_ragged_extent = physical_bound_type(left_type)?.dimension(left_ragged_axis).value().unwrap();
        let (group_starts, group_ends) = lower_ragged_dot_group_intervals(
            input_values[2],
            group_sizes_type,
            physical_ragged_extent,
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        let left = lower_ragged_dot_expansion(
            input_values[0],
            left_type,
            group_starts,
            group_ends,
            group_sizes_type,
            left_ragged_axis,
            left_prefix.as_slice(),
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        let shift = |axes: &[usize]| axes.iter().map(|axis| axis + 1).collect::<Vec<_>>();
        let dot = dimensions.dot_dimensions();
        let dense_dimensions = match mode {
            RaggedDotMode::NonContracting => {
                let mut left_contracting = shift(dot.lhs_contracting_dimensions());
                left_contracting.push(0);
                let mut right_contracting = dot.rhs_contracting_dimensions().to_vec();
                right_contracting.push(dimensions.rhs_group_dimension().unwrap());
                DotDimensionNumbers::new(
                    left_contracting,
                    right_contracting,
                    shift(dot.lhs_batching_dimensions()),
                    dot.rhs_batching_dimensions().to_vec(),
                )
            }
            RaggedDotMode::Contracting => {
                let position =
                    dot.lhs_contracting_dimensions().iter().position(|axis| *axis == left_ragged_axis).unwrap();
                let right_ragged_axis = dot.rhs_contracting_dimensions()[position];
                let right_prefix = dot
                    .rhs_batching_dimensions()
                    .iter()
                    .copied()
                    .chain(dot.rhs_contracting_dimensions()[..position].iter().copied())
                    .collect::<Vec<_>>();
                let right = lower_ragged_dot_expansion(
                    input_values[1],
                    right_type,
                    group_starts,
                    group_ends,
                    group_sizes_type,
                    right_ragged_axis,
                    right_prefix.as_slice(),
                    &mut lowerer.block(),
                    lowerer.context(),
                    lowerer.location(),
                )?;
                let mut left_batching = vec![0];
                left_batching.extend(shift(dot.lhs_batching_dimensions()));
                let mut right_batching = vec![0];
                right_batching.extend(shift(dot.rhs_batching_dimensions()));
                return Ok(vec![lower_dense_ragged_dot(
                    left,
                    right,
                    &DotDimensionNumbers::new(
                        shift(dot.lhs_contracting_dimensions()),
                        shift(dot.rhs_contracting_dimensions()),
                        left_batching,
                        right_batching,
                    ),
                    &output_types[0],
                    &mut lowerer.block(),
                    lowerer.context(),
                    lowerer.location(),
                )?]);
            }
            RaggedDotMode::Batch => unreachable!(),
        };
        Ok(vec![lower_dense_ragged_dot(
            left,
            input_values[1],
            &dense_dimensions,
            &output_types[0],
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?])
    }
}

/// Lowers the algorithm that `precision` requests to a StableHLO `algorithm` attribute for a `dot_general` whose
/// inputs have the computation element types `left_type` and `right_type`, which [`DotPrecision::computation_types`]
/// selects. Per-input precisions request no algorithm.
fn lower_dot_algorithm<'c, 't>(
    precision: DotPrecision,
    left_type: DataType,
    right_type: DataType,
    context: &'c MlirContext<'t>,
) -> Result<Option<DotAlgorithmAttributeRef<'c, 't>>, Error> {
    let float_type = |data_type: DataType| {
        lower_element_type(data_type, context)?.cast::<FloatTypeRef>().ok_or_else(|| {
            Error::Lowering(LoweringError::UnsupportedOperation {
                operation: DOT_OPERATION_NAME.to_string(),
                reason: format!("the algorithm type `{data_type}` is not a floating-point type"),
            })
        })
    };
    let float8_type = |data_type: DataType| {
        lower_element_type(data_type, context)?.cast::<Float8TypeRef>().ok_or_else(|| {
            Error::Lowering(LoweringError::UnsupportedOperation {
                operation: DOT_OPERATION_NAME.to_string(),
                reason: format!("the algorithm input type `{data_type}` is not an 8-bit floating-point type"),
            })
        })
    };
    let preset = match precision {
        DotPrecision::Inputs { .. } => return Ok(None),
        DotPrecision::Algorithm(algorithm) => {
            return Ok(Some(context.stable_hlo_dot_algorithm(
                float_type(algorithm.lhs_precision_type())?,
                float_type(algorithm.rhs_precision_type())?,
                float_type(algorithm.accumulation_type())?,
                algorithm.lhs_component_count(),
                algorithm.rhs_component_count(),
                algorithm.primitive_operation_count(),
                algorithm.allow_imprecise_accumulation(),
            )?));
        }
        DotPrecision::Preset(preset) => preset,
    };

    // The `Any_F8` presets name the 8-bit input types of the operation and always declare `f32` accumulation, which
    // is what JAX emits for all four of them. The fast variants allow imprecise accumulation. JAX sets that flag only
    // for the `F32` fast variant, which makes its `ANY` fast variant identical to the precise one.
    let preset = match preset {
        DotAlgorithmPreset::Any_F8_Any_F8_F32 => stable_hlo::DotAlgorithmPreset::Any_F8_Any_F8_F32 {
            lhs_precision_type: float8_type(left_type)?,
            rhs_precision_type: float8_type(right_type)?,
        },
        DotAlgorithmPreset::Any_F8_Any_F8_F32_Fast_Accumulation => {
            stable_hlo::DotAlgorithmPreset::Any_F8_Any_F8_F32_Fast_Accumulation {
                lhs_precision_type: float8_type(left_type)?,
                rhs_precision_type: float8_type(right_type)?,
            }
        }
        DotAlgorithmPreset::Any_F8_Any_F8_Any => stable_hlo::DotAlgorithmPreset::Any_F8_Any_F8_Any {
            lhs_precision_type: float8_type(left_type)?,
            rhs_precision_type: float8_type(right_type)?,
            accumulation_type: float_type(DataType::F32)?,
        },
        DotAlgorithmPreset::Any_F8_Any_F8_Any_Fast_Accumulation => {
            stable_hlo::DotAlgorithmPreset::Any_F8_Any_F8_Any_Fast_Accumulation {
                lhs_precision_type: float8_type(left_type)?,
                rhs_precision_type: float8_type(right_type)?,
                accumulation_type: float_type(DataType::F32)?,
            }
        }
        DotAlgorithmPreset::F16_F16_F16 => stable_hlo::DotAlgorithmPreset::F16_F16_F16,
        DotAlgorithmPreset::F16_F16_F32 => stable_hlo::DotAlgorithmPreset::F16_F16_F32,
        DotAlgorithmPreset::BF16_BF16_BF16 => stable_hlo::DotAlgorithmPreset::BF16_BF16_BF16,
        DotAlgorithmPreset::BF16_BF16_F32 => stable_hlo::DotAlgorithmPreset::BF16_BF16_F32,
        DotAlgorithmPreset::BF16_BF16_F32_X3 => stable_hlo::DotAlgorithmPreset::BF16_BF16_F32_X3,
        DotAlgorithmPreset::BF16_BF16_F32_X6 => stable_hlo::DotAlgorithmPreset::BF16_BF16_F32_X6,
        DotAlgorithmPreset::BF16_BF16_F32_X9 => stable_hlo::DotAlgorithmPreset::BF16_BF16_F32_X9,
        DotAlgorithmPreset::TF32_TF32_F32 => stable_hlo::DotAlgorithmPreset::TF32_TF32_F32,
        DotAlgorithmPreset::TF32_TF32_F32_X3 => stable_hlo::DotAlgorithmPreset::TF32_TF32_F32_X3,
        DotAlgorithmPreset::F32_F32_F32 => stable_hlo::DotAlgorithmPreset::F32_F32_F32,
        DotAlgorithmPreset::F64_F64_F64 => stable_hlo::DotAlgorithmPreset::F64_F64_F64,
    };
    Ok(context.stable_hlo_dot_algorithm_from_preset(preset)?)
}

/// Builds capped cumulative group starts and ends in `u64`. Signed inputs are clamped to zero before widening, bounded
/// dynamic prefixes are materialized once, and the cumulative reducer cannot overflow before applying the physical
/// ragged axis bound.
fn lower_ragged_dot_group_intervals<'b, 'c: 'b, 't: 'c>(
    group_sizes: ValueRef<'b, 'c, 't>,
    group_sizes_type: &ArrayType,
    physical_ragged_extent: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(ValueRef<'b, 'c, 't>, ValueRef<'b, 'c, 't>), Error> {
    let group_axis = group_sizes_type.rank() - 1;
    let group_count = group_sizes_type.dimension(group_axis).value().ok_or_else(|| {
        Error::Lowering(LoweringError::UnsupportedOperation {
            operation: RAGGED_DOT_OPERATION_NAME.to_string(),
            reason: "`ragged_dot` with a dynamically sized group axis".to_string(),
        })
    })?;
    let group_sizes = if group_sizes_type.data_type().is_signed() {
        // Subtracting the value from itself constructs a shape-identical zero in the source integer type. That avoids
        // unsigned sign extension without introducing a scalar constant that would need explicit broadcasting across
        // static or bounded dynamic metadata prefixes.
        let zero = block.append_operation(stable_hlo::subtract(group_sizes, group_sizes, location)?)?;
        let zero = zero.result(0).unwrap().as_ref();
        let nonnegative = block.append_operation(stable_hlo::maximum(group_sizes, zero, location)?)?;
        nonnegative.result(0).unwrap().as_ref()
    } else {
        group_sizes
    };
    let wide_group_sizes_type = group_sizes_type.clone().with_data_type(DataType::U64);
    let wide_group_sizes_tensor_type = lower_tensor_type(&wide_group_sizes_type, context, location)?;
    let wide_group_sizes = if group_sizes_type.data_type() == DataType::U64 {
        group_sizes
    } else {
        let converted =
            block.append_operation(stable_hlo::convert(group_sizes, wide_group_sizes_tensor_type, location)?)?;
        converted.result(0).unwrap().as_ref()
    };
    let wide_group_sizes =
        lower_physical_bound_value(wide_group_sizes, &wide_group_sizes_type, 0.0, block, context, location)?;
    let physical_ragged_extent = u64::try_from(physical_ragged_extent).unwrap();
    if group_count == 0 {
        return Ok((wide_group_sizes, wide_group_sizes));
    }

    let zero = lower_u64_scalar_constant(0, block, context, location)?;
    let extent = lower_u64_scalar_constant(physical_ragged_extent, block, context, location)?;
    // StableHLO may reassociate a reduction, so cap every leaf first to ensure every reducer operand is at most the
    // physical extent and the saturating reducer remains associative.
    let capped_group_sizes = block.append_operation(stable_hlo::clamp(zero, wide_group_sizes, extent, location)?)?;
    let capped_group_sizes = capped_group_sizes.result(0).unwrap().as_ref();
    let scalar_tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::U64), context, location)?;
    let reducer_block = context.block(&[(scalar_tensor_type, location), (scalar_tensor_type, location)]);
    let mut reducer_region = context.region();
    let mut reducer_block = reducer_region.append_block(reducer_block)?;
    let left = reducer_block.argument(0)?.as_ref();
    let right = reducer_block.argument(1)?.as_ref();
    let reducer_extent = lower_u64_scalar_constant(physical_ragged_extent, &mut reducer_block, context, location)?;
    let remaining = reducer_block.append_operation(stable_hlo::subtract(reducer_extent, left, location)?)?;
    let remaining = remaining.result(0).unwrap().as_ref();
    let increment = reducer_block.append_operation(stable_hlo::minimum(right, remaining, location)?)?;
    let increment = increment.result(0).unwrap().as_ref();
    let end = reducer_block.append_operation(stable_hlo::add(left, increment, location)?)?;
    let end = end.result(0).unwrap().as_ref();
    reducer_block.append_operation(stable_hlo::r#return(&[end], location)?)?;

    let rank = group_sizes_type.rank();
    let mut window_dimensions = vec![1; rank];
    window_dimensions[group_axis] = group_count;
    let mut padding = vec![(0, 0); rank];
    padding[group_axis] = (group_count - 1, 0);
    let group_ends = block.append_operation(stable_hlo::reduce_window(
        &[capped_group_sizes],
        &[zero],
        &window_dimensions,
        None,
        None,
        None,
        Some(&padding),
        reducer_region,
        location,
    )?)?;
    let group_ends = group_ends.result(0).unwrap().as_ref();
    let mut edge_padding_low = vec![0; rank];
    edge_padding_low[group_axis] = 1;
    let mut edge_padding_high = vec![0; rank];
    edge_padding_high[group_axis] = -1;
    let interior_padding = vec![0; rank];
    let group_starts = block.append_operation(stable_hlo::pad(
        group_ends,
        zero,
        edge_padding_low.as_slice(),
        edge_padding_high.as_slice(),
        interior_padding.as_slice(),
        location,
    )?)?;
    let group_starts = group_starts.result(0).unwrap().as_ref();
    Ok((group_starts, group_ends))
}

/// Expands one grouped dot operand into a dense group axis whose elements outside each group are masked to zero. Group
/// intervals are accumulated in `u64` and capped at the physical ragged extent so metadata arithmetic and iota indices
/// cannot wrap. The intermediate order keeps the `group_sizes` prefix axes first so every broadcast dimension is
/// monotonic, and a final transpose, when necessary, produces `[group, original axes...]` for the dense dot.
fn lower_ragged_dot_expansion<'b, 'c: 'b, 't: 'c>(
    input: ValueRef<'b, 'c, 't>,
    input_type: &ArrayType,
    group_starts: ValueRef<'b, 'c, 't>,
    group_ends: ValueRef<'b, 'c, 't>,
    group_sizes_type: &ArrayType,
    ragged_axis: usize,
    prefix_axes: &[usize],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let group_axis = group_sizes_type.rank() - 1;
    let group_count = group_sizes_type.dimension(group_axis);

    let remaining_axes = (0..input_type.rank()).filter(|axis| !prefix_axes.contains(axis)).collect::<Vec<_>>();
    let input_permutation = prefix_axes.iter().copied().chain(remaining_axes.iter().copied()).collect::<Vec<_>>();
    let input = if input_permutation.iter().enumerate().all(|(axis, source)| axis == *source) {
        input
    } else {
        let transpose =
            block.append_operation(stable_hlo::transpose(input, input_permutation.as_slice(), location)?)?;
        transpose.result(0).unwrap().as_ref()
    };
    let mut intermediate_dimensions = prefix_axes.iter().map(|axis| input_type.dimension(*axis)).collect::<Vec<_>>();
    intermediate_dimensions.push(group_count.clone());
    intermediate_dimensions.extend(remaining_axes.iter().map(|axis| input_type.dimension(*axis)));
    let intermediate_type = input_type.clone().with_shape(Shape::new(intermediate_dimensions));
    let intermediate_tensor_type = lower_tensor_type(&intermediate_type, context, location)?;
    let input_broadcast_dimensions =
        (0..prefix_axes.len()).chain((prefix_axes.len() + 1)..intermediate_type.rank()).collect::<Vec<_>>();
    let input = block.append_operation(stable_hlo::broadcast(
        input,
        intermediate_tensor_type,
        input_broadcast_dimensions.as_slice(),
        location,
    )?)?;
    let input = input.result(0).unwrap().as_ref();

    let metadata_broadcast_dimensions =
        if group_sizes_type.rank() == 1 { vec![prefix_axes.len()] } else { (0..=prefix_axes.len()).collect() };
    let metadata_type = intermediate_type.clone().with_data_type(DataType::U64);
    let physical_metadata_type = physical_bound_type(&metadata_type)?;
    let physical_metadata_tensor_type = lower_tensor_type(&physical_metadata_type, context, location)?;
    let dimension_sources = (0..metadata_type.rank()).map(|axis| (input, axis)).collect::<Vec<_>>();
    let group_starts = block.append_operation(stable_hlo::broadcast(
        group_starts,
        physical_metadata_tensor_type,
        metadata_broadcast_dimensions.as_slice(),
        location,
    )?)?;
    let group_starts = group_starts.result(0).unwrap().as_ref();
    let group_ends = block.append_operation(stable_hlo::broadcast(
        group_ends,
        physical_metadata_tensor_type,
        metadata_broadcast_dimensions.as_slice(),
        location,
    )?)?;
    let group_ends = group_ends.result(0).unwrap().as_ref();
    let ragged_intermediate_axis =
        prefix_axes.len() + 1 + remaining_axes.iter().position(|axis| *axis == ragged_axis).unwrap();
    let indices =
        block.append_operation(stable_hlo::iota(physical_metadata_tensor_type, ragged_intermediate_axis, location)?)?;
    let indices = indices.result(0).unwrap().as_ref();
    let after_start =
        lower_compare_to_mlir(ComparisonDirection::LessThanOrEqual, group_starts, indices, block, location)?;
    let before_end = lower_compare_to_mlir(ComparisonDirection::LessThan, indices, group_ends, block, location)?;
    let mask = block.append_operation(stable_hlo::and(after_start, before_end, location)?)?;
    let mask = mask.result(0).unwrap().as_ref();
    let mask = lower_restore_dynamic_dimensions(
        mask,
        &metadata_type.with_data_type(DataType::Boolean),
        dimension_sources.as_slice(),
        block,
        context,
        location,
    )?;
    let zero = if intermediate_type.static_shape().is_some() {
        let zero =
            lower_unplaced_constant_output(&[ArrayType::scalar(input_type.data_type())], 0, block, context, location)?
                [0];
        let zero = block.append_operation(stable_hlo::broadcast(zero, intermediate_tensor_type, &[], location)?)?;
        zero.result(0).unwrap().as_ref()
    } else {
        let false_mask = block.append_operation(stable_hlo::xor(mask, mask, location)?)?;
        let false_mask = false_mask.result(0).unwrap().as_ref();
        let zero = block.append_operation(stable_hlo::convert(false_mask, intermediate_tensor_type, location)?)?;
        zero.result(0).unwrap().as_ref()
    };
    let masked = block.append_operation(stable_hlo::select(mask, input, zero, location)?)?;
    let masked = masked.result(0).unwrap().as_ref();

    let labels = prefix_axes
        .iter()
        .copied()
        .map(Some)
        .chain(std::iter::once(None))
        .chain(remaining_axes.iter().copied().map(Some))
        .collect::<Vec<_>>();
    let final_permutation = std::iter::once(labels.iter().position(Option::is_none).unwrap())
        .chain((0..input_type.rank()).map(|axis| labels.iter().position(|label| *label == Some(axis)).unwrap()))
        .collect::<Vec<_>>();
    if final_permutation.iter().enumerate().all(|(axis, source)| axis == *source) {
        Ok(masked)
    } else {
        let expanded =
            block.append_operation(stable_hlo::transpose(masked, final_permutation.as_slice(), location)?)?;
        Ok(expanded.result(0).unwrap().as_ref())
    }
}

/// Emits one ordinary dense dot after the grouped operands have been expanded.
fn lower_dense_ragged_dot<'b, 'c: 'b, 't: 'c>(
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    dimensions: &DotDimensionNumbers,
    output_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let dimensions = context.stable_hlo_dot_dimensions(
        dimensions.lhs_batching_dimensions(),
        dimensions.rhs_batching_dimensions(),
        dimensions.lhs_contracting_dimensions(),
        dimensions.rhs_contracting_dimensions(),
    )?;
    let dot = block.append_operation(stable_hlo::dot_general(
        left,
        right,
        dimensions,
        Some((Precision::Default, Precision::Default)),
        None,
        lower_tensor_type(output_type, context, location)?,
        location,
    )?)?;
    Ok(dot.result(0).unwrap().as_ref())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array as CoreArray, ArrayIrType, ArrayOperation, Complex, Dimension as CoreDimension, Dot, DotAlgorithm,
        DotDimensionNumbers, DotOperation, Einsum, EinsumOptions, EinsumPathStep, EinsumPathStrategy,
        EinsumSpecification, Operation as CoreOperation, Placeholder, ProjectedContext, RaggedDot,
        RaggedDotDimensionNumbers, Sharding, Typed, bf16, differentiate_at, einsum, f8e4m3fn, f8e5m2, f16,
    };
    use ryft_pjrt::{BufferType, Error as PjrtError, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::domains::tests::domain_mesh;
    use crate::lowering::tests::{
        ProgramBuilder, dynamic_dimension, ragged_dot_cpu_compilation_options, unproject_plain_program,
    };
    use crate::lowering::{
        lower_mlir_module_for_program, to_mlir_module_for_plain_program_with_ragged_dot_lowering_strategy,
    };
    use crate::operations::ProgramBuilder as ArrayIrProgramBuilder;
    use crate::tests::{cpu_client, execution_client, replicated_array, values_from_bytes, values_to_bytes};
    use crate::tracing::{ArrayTracer, TracedProgram, trace};
    use crate::values::ArrayConstant;
    use crate::{Constant, Domain, Options, Session, ToPjrt, to_mlir_module_for_program};

    use super::*;

    // Lowers one dense dot of inputs with the provided types for the PJRT platform named `target_platform`.
    fn lowered_dot_module(
        operation: DotOperation,
        left_type: ArrayType,
        right_type: ArrayType,
        target_platform: Option<&str>,
    ) -> Result<String, Error> {
        let output_types = operation.infer_output_types(&[left_type.clone(), right_type.clone()], &[]).unwrap();
        let mut builder = ProgramBuilder::new();
        let left = builder.add_input(left_type.clone());
        let right = builder.add_input(right_type.clone());
        let output = builder.add_instruction(operation, Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        let input_types = vec![left_type, right_type];
        lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, target_platform)
            .map(|module| module.into_stable_hlo())
    }

    // Lowers one grouped dot signature using `strategy`.
    fn lowered_ragged_dot_module_for(
        dimensions: RaggedDotDimensionNumbers,
        left_type: ArrayType,
        right_type: ArrayType,
        group_sizes_type: ArrayType,
        strategy: RaggedDotLoweringStrategy,
    ) -> Result<String, Error> {
        let mut builder = ProgramBuilder::new();
        let left = builder.add_input(left_type);
        let right = builder.add_input(right_type);
        let group_sizes = builder.add_input(group_sizes_type);
        let output = builder
            .add_instruction(
                ArrayOperation::RaggedDot(RaggedDotOperation::new(dimensions)),
                Vec::new(),
                vec![left, right, group_sizes],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        to_mlir_module_for_plain_program_with_ragged_dot_lowering_strategy(&program, "main", strategy)
    }

    // Lowers the basic non-contracting grouped matrix form using `strategy`.
    fn lowered_ragged_dot_module(strategy: RaggedDotLoweringStrategy) -> Result<String, Error> {
        lowered_ragged_dot_module_for(
            RaggedDotDimensionNumbers::matmul(),
            ArrayType::new_static(DataType::F32, [5, 2]),
            ArrayType::new_static(DataType::F32, [3, 2, 1]),
            ArrayType::new_static(DataType::I32, [3]),
            strategy,
        )
    }

    // Compiles and executes one grouped matrix decomposition on the CPU plugin.
    fn execute_ragged_dot_decomposition_on_cpu(
        dimensions: RaggedDotDimensionNumbers,
        left_type: ArrayType,
        right_type: ArrayType,
        group_sizes_type: ArrayType,
        left_values: &[f32],
        right_values: &[f32],
        group_sizes_data: &[u8],
    ) -> Vec<f32> {
        let module = lowered_ragged_dot_module_for(
            dimensions,
            left_type.clone(),
            right_type.clone(),
            group_sizes_type.clone(),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();
        let left_shape = left_type.static_shape().unwrap();
        let left_dimensions = left_shape.dimensions().iter().map(|dimension| *dimension as u64).collect::<Vec<_>>();
        let right_shape = right_type.static_shape().unwrap();
        let right_dimensions = right_shape.dimensions().iter().map(|dimension| *dimension as u64).collect::<Vec<_>>();
        let group_sizes_shape = group_sizes_type.static_shape().unwrap();
        let group_sizes_dimensions =
            group_sizes_shape.dimensions().iter().map(|dimension| *dimension as u64).collect::<Vec<_>>();
        let client = cpu_client(1);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        let left = client
            .buffer(
                values_to_bytes(left_values).as_slice(),
                BufferType::F32,
                left_dimensions.as_slice(),
                None,
                device.clone(),
                None,
            )
            .unwrap();
        let right = client
            .buffer(
                values_to_bytes(right_values).as_slice(),
                BufferType::F32,
                right_dimensions.as_slice(),
                None,
                device.clone(),
                None,
            )
            .unwrap();
        let group_sizes = client
            .buffer(
                group_sizes_data,
                group_sizes_type.data_type().to_pjrt(),
                group_sizes_dimensions.as_slice(),
                None,
                device,
                None,
            )
            .unwrap();
        let inputs = [left, right, group_sizes]
            .into_iter()
            .map(|buffer| ExecutionInput { buffer: Arc::new(buffer), donatable: false })
            .collect::<Vec<_>>();
        let mut executions = executable
            .execute(
                vec![ExecutionDeviceInputs { inputs: inputs.as_slice(), ..Default::default() }],
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        let output = executions.remove(0).outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        values_from_bytes::<f32>(output.as_slice())
    }

    #[test]
    fn test_dot_operation_lower_with_accumulation_type() {
        // A dot with an accumulation type lowers to a `stablehlo.dot_general` whose result type is the accumulation
        // type, as XLA's `preferred_element_type` contract requires, with the operands kept at their narrow element
        // type.
        let operand_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![CoreDimension::Static(2), CoreDimension::Static(2)]));
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2), CoreDimension::Static(2)]));
        let mut builder = ProgramBuilder::new();
        let left = builder.add_input(operand_type.clone());
        let right = builder.add_input(operand_type.clone());
        let operation = DotOperation::new(DotDimensionNumbers::matmul()).with_accumulation_type(DataType::F32);
        let output = builder.add_instruction(operation, Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        let input_types = vec![operand_type.clone(), operand_type];
        let output_types = vec![output_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x2xf8E4M3FN>, %arg1: tensor<2x2xf8E4M3FN>) -> tensor<2x2xf32> {
                    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf8E4M3FN>, tensor<2x2xf8E4M3FN>) -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_dot_operation_lower_with_accumulation_type_matches_reference_backend() {
        // The operands stay `f8e4m3fn` on device and the contraction accumulates at `f32`. Every value is exactly
        // representable in `f8e4m3fn` and every product and sum is exact in `f32`, so both backends are exact.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let operand_type = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        let reference_left = CoreArray::from_elements(
            operand_type.clone(),
            &[0.5f64, 1.0, 1.5, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let reference_right = CoreArray::from_elements(
            operand_type,
            &[1f64, 0.5, 0.5, 1.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let left = replicated_array(&domain, &mesh, DataType::F8E4M3FN, &[2, 2], &reference_left.logical_bytes());
        let right = replicated_array(&domain, &mesh, DataType::F8E4M3FN, &[2, 2], &reference_right.logical_bytes());

        let product = left.dot_with_accumulation_type(&right, &DotDimensionNumbers::matmul(), DataType::F32).unwrap();
        let reference_product = reference_left
            .dot_with_accumulation_type(&reference_right, &DotDimensionNumbers::matmul(), DataType::F32)
            .unwrap();
        assert_eq!(
            product.r#type().into_owned(),
            ArrayType::new_static(DataType::F32, [2, 2])
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
                .unwrap(),
        );
        assert_eq!(product.copy_to_host().unwrap(), values_to_bytes(&[1f32, 1.25, 2.5, 2.75]));
        assert_eq!(product.copy_to_host().unwrap(), reference_product.logical_bytes());
    }

    #[test]
    fn test_dot_operation_lower_with_narrower_accumulation_type_matches_reference_backend() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // The three terms `2⁻¹²` sum to three quarters of the distance from `1` to the next `f16` value, so a sum
        // that accumulates at `f32` and rounds once is `1 + 2⁻¹⁰`.
        let reference_left = CoreArray::vector(vec![1.0f32, 0.000244140625, 0.000244140625, 0.000244140625]).unwrap();
        let reference_right = CoreArray::vector(vec![1.0f32; 4]).unwrap();
        let left = replicated_array(&domain, &mesh, DataType::F32, &[4], &reference_left.logical_bytes());
        let right = replicated_array(&domain, &mesh, DataType::F32, &[4], &reference_right.logical_bytes());
        let dimensions = DotDimensionNumbers::inner_product();
        let product = left.dot_with_accumulation_type(&right, &dimensions, DataType::F16).unwrap();
        let reference_product =
            reference_left.dot_with_accumulation_type(&reference_right, &dimensions, DataType::F16).unwrap();
        assert_eq!(product.copy_to_host().unwrap(), values_to_bytes(&[f16::from_f32(1.0009765625)]));
        assert_eq!(product.copy_to_host().unwrap(), reference_product.logical_bytes());
    }

    #[test]
    fn test_dot_operation_lower_with_distinct_f8_inputs_matches_reference_backend() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Two distinct 8-bit input types stay as they are on device and the contraction accumulates at the requested
        // type. Every value is exactly representable, so `[0.5, 1.0] · [2.0, 4.0] = 5` on both backends.
        let reference_left = CoreArray::from_elements(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[0.5f64, 1.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let reference_right = CoreArray::from_elements(
            ArrayType::new_static(DataType::F8E5M2, [2]),
            &[2.0f64, 4.0].map(|value| f8e5m2::from_f64(value).unwrap()),
        )
        .unwrap();
        let left = replicated_array(&domain, &mesh, DataType::F8E4M3FN, &[2], &reference_left.logical_bytes());
        let right = replicated_array(&domain, &mesh, DataType::F8E5M2, &[2], &reference_right.logical_bytes());
        let dimensions = DotDimensionNumbers::inner_product();
        let product = left.dot_with_accumulation_type(&right, &dimensions, DataType::BF16).unwrap();
        let reference_product =
            reference_left.dot_with_accumulation_type(&reference_right, &dimensions, DataType::BF16).unwrap();
        assert_eq!(product.copy_to_host().unwrap(), values_to_bytes(&[bf16::from_f32(5.0)]));
        assert_eq!(product.copy_to_host().unwrap(), reference_product.logical_bytes());
    }

    #[test]
    fn test_dot_operation_lower_with_precision() {
        // Per-input precisions lower to the `precision_config` of the contraction without any conversions.
        let f32_type = ArrayType::new_static(DataType::F32, [2, 2]);
        assert_eq!(
            lowered_dot_module(
                DotOperation::matmul().with_precision((DotInputPrecision::High, DotInputPrecision::Highest)),
                f32_type.clone(),
                f32_type.clone(),
                None,
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], \
                    precision = [HIGH, HIGHEST] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );

        // A preset converts the inputs that it does not compute with, accumulates at its accumulation type, and
        // converts the result to the output type.
        let f64_type = ArrayType::new_static(DataType::F64, [2, 2]);
        assert_eq!(
            lowered_dot_module(
                DotOperation::matmul().with_precision(DotAlgorithmPreset::BF16_BF16_F32),
                f64_type.clone(),
                f64_type,
                None,
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xf64> {
                    %0 = stablehlo.convert %arg0 : (tensor<2x2xf64>) -> tensor<2x2xbf16>
                    %1 = stablehlo.convert %arg1 : (tensor<2x2xf64>) -> tensor<2x2xbf16>
                    %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0], \
                    precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = bf16, \
                    rhs_precision_type = bf16, accumulation_type = f32, lhs_component_count = 1, \
                    rhs_component_count = 1, num_primitive_operations = 1, \
                    allow_imprecise_accumulation = false> : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xf32>
                    %3 = stablehlo.convert %2 : (tensor<2x2xf32>) -> tensor<2x2xf64>
                    return %3 : tensor<2x2xf64>
                  }
                }
            "},
        );

        // The `Any_F8` presets name the two 8-bit input types, declare `f32` accumulation, and produce a listed
        // output type directly.
        assert_eq!(
            lowered_dot_module(
                DotOperation::matmul()
                    .with_precision(DotAlgorithmPreset::Any_F8_Any_F8_F32_Fast_Accumulation)
                    .with_accumulation_type(DataType::BF16),
                ArrayType::new_static(DataType::F8E4M3FN, [2, 2]),
                ArrayType::new_static(DataType::F8E5M2, [2, 2]),
                None,
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x2xf8E4M3FN>, %arg1: tensor<2x2xf8E5M2>) -> tensor<2x2xbf16> {
                    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], \
                    precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = f8E4M3FN, \
                    rhs_precision_type = f8E5M2, accumulation_type = f32, lhs_component_count = 1, \
                    rhs_component_count = 1, num_primitive_operations = 1, \
                    allow_imprecise_accumulation = true> : (tensor<2x2xf8E4M3FN>, tensor<2x2xf8E5M2>) -> \
                    tensor<2x2xbf16>
                    return %0 : tensor<2x2xbf16>
                  }
                }
            "},
        );

        // An explicit algorithm converts the inputs to its precision types and lowers all of its fields.
        assert_eq!(
            lowered_dot_module(
                DotOperation::matmul().with_precision(
                    DotAlgorithm::new(DataType::BF16, DataType::BF16, DataType::F32).with_primitive_operation_count(3),
                ),
                f32_type.clone(),
                f32_type.clone(),
                None,
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.convert %arg0 : (tensor<2x2xf32>) -> tensor<2x2xbf16>
                    %1 = stablehlo.convert %arg1 : (tensor<2x2xf32>) -> tensor<2x2xbf16>
                    %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0], \
                    precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = bf16, \
                    rhs_precision_type = bf16, accumulation_type = f32, lhs_component_count = 1, \
                    rhs_component_count = 1, num_primitive_operations = 3, \
                    allow_imprecise_accumulation = false> : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xf32>
                    return %2 : tensor<2x2xf32>
                  }
                }
            "},
        );

        // The TensorFloat-32 presets lower to the `tf32` precision type, which no array element type represents.
        assert_eq!(
            lowered_dot_module(
                DotOperation::matmul().with_precision(DotAlgorithmPreset::TF32_TF32_F32_X3),
                f32_type.clone(),
                f32_type.clone(),
                Some("cuda"),
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], \
                    precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = tf32, \
                    rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, \
                    rhs_component_count = 1, num_primitive_operations = 3, \
                    allow_imprecise_accumulation = false> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );

        // StableHLO verifies that an algorithm is known to be supported on some hardware, so an explicit algorithm
        // outside that set fails the verification of the lowered module.
        assert!(matches!(
            lowered_dot_module(
                DotOperation::matmul().with_precision(
                    DotAlgorithm::new(DataType::F16, DataType::BF16, DataType::F32).with_component_counts(2, 3),
                ),
                f32_type.clone(),
                f32_type,
                None,
            ),
            Err(Error::Lowering(LoweringError::VerificationFailure)),
        ));
    }

    #[test]
    fn test_dot_operation_lower_with_precision_matches_reference_backend() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // The `bf16` preset rounds its `f64` inputs to `bf16`, where `1 + 2⁻¹⁰` becomes `1`, accumulates at `f32`,
        // and converts the result to the `f64` output type. Every remaining value is exact, so both backends are.
        let reference_left = CoreArray::vector(vec![1.0009765625f64, 2.0]).unwrap();
        let reference_right = CoreArray::vector(vec![1.0f64, 1.0]).unwrap();
        let left = replicated_array(&domain, &mesh, DataType::F64, &[2], &reference_left.logical_bytes());
        let right = replicated_array(&domain, &mesh, DataType::F64, &[2], &reference_right.logical_bytes());
        let dimensions = DotDimensionNumbers::inner_product();
        let product = left.dot_with_precision(&right, &dimensions, DotAlgorithmPreset::BF16_BF16_F32.into()).unwrap();
        let reference_product = reference_left
            .dot_with_precision(&reference_right, &dimensions, DotAlgorithmPreset::BF16_BF16_F32.into())
            .unwrap();
        assert_eq!(product.copy_to_host().unwrap(), values_to_bytes(&[3.0f64]));
        assert_eq!(product.copy_to_host().unwrap(), reference_product.logical_bytes());

        // Per-input precisions are hints that the CPU backend ignores.
        let product = left.dot_with_precision(&right, &dimensions, DotInputPrecision::Highest.into()).unwrap();
        assert_eq!(product.copy_to_host().unwrap(), values_to_bytes(&[3.0009765625f64]));
    }

    #[test]
    fn test_dot_operation_lower_rejects_unsupported_algorithms_on_cpu() {
        let f32_type = ArrayType::new_static(DataType::F32, [2, 2]);
        let f8_type = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);

        // The CPU compiler ignores algorithms, so a CPU target accepts only the presets whose results it reproduces.
        for preset in [
            DotAlgorithmPreset::F16_F16_F16,
            DotAlgorithmPreset::F32_F32_F32,
            DotAlgorithmPreset::F64_F64_F64,
            DotAlgorithmPreset::BF16_BF16_F32,
            DotAlgorithmPreset::BF16_BF16_F32_X3,
            DotAlgorithmPreset::BF16_BF16_F32_X6,
        ] {
            let operation = DotOperation::matmul().with_precision(preset);
            lowered_dot_module(operation, f32_type.clone(), f32_type.clone(), Some("cpu")).unwrap();
        }
        for preset in [
            DotAlgorithmPreset::F16_F16_F32,
            DotAlgorithmPreset::BF16_BF16_BF16,
            DotAlgorithmPreset::BF16_BF16_F32_X9,
            DotAlgorithmPreset::TF32_TF32_F32,
            DotAlgorithmPreset::TF32_TF32_F32_X3,
        ] {
            let operation = DotOperation::matmul().with_precision(preset);
            assert!(matches!(
                lowered_dot_module(operation, f32_type.clone(), f32_type.clone(), Some("cpu")),
                Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                    if operation == DOT_OPERATION_NAME
                        && reason == format!("the algorithm `{preset}` is not supported on CPU"),
            ));
        }
        for preset in [
            DotAlgorithmPreset::Any_F8_Any_F8_F32,
            DotAlgorithmPreset::Any_F8_Any_F8_F32_Fast_Accumulation,
            DotAlgorithmPreset::Any_F8_Any_F8_Any,
            DotAlgorithmPreset::Any_F8_Any_F8_Any_Fast_Accumulation,
        ] {
            let operation = DotOperation::matmul().with_precision(preset).with_accumulation_type(DataType::F32);
            assert!(matches!(
                lowered_dot_module(operation, f8_type.clone(), f8_type.clone(), Some("cpu")),
                Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                    if operation == DOT_OPERATION_NAME
                        && reason == format!("the algorithm `{preset}` is not supported on CPU"),
            ));
        }

        // Explicit algorithms are rejected on CPU as well, while per-input precisions are always accepted.
        let algorithm = DotAlgorithm::new(DataType::F32, DataType::F32, DataType::F32);
        assert!(matches!(
            lowered_dot_module(
                DotOperation::matmul().with_precision(algorithm),
                f32_type.clone(),
                f32_type.clone(),
                Some("cpu"),
            ),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == DOT_OPERATION_NAME
                    && reason == format!("the algorithm `{algorithm}` is not supported on CPU"),
        ));
        let operation = DotOperation::matmul().with_precision(DotInputPrecision::Highest);
        lowered_dot_module(operation, f32_type.clone(), f32_type, Some("cpu")).unwrap();
    }

    #[test]
    fn test_ragged_dot_operation_lower() {
        let decomposition = lowered_ragged_dot_module(RaggedDotLoweringStrategy::Decomposition).unwrap();
        assert_eq!(
            decomposition,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<5x2xf32>, \
                  %arg1: tensor<3x2x1xf32>, %arg2: tensor<3xi32>) -> tensor<5x1xf32> {
                    %0 = stablehlo.subtract %arg2, %arg2 : tensor<3xi32>
                    %1 = stablehlo.maximum %arg2, %0 : tensor<3xi32>
                    %2 = stablehlo.convert %1 : (tensor<3xi32>) -> tensor<3xui64>
                    %c = stablehlo.constant dense<0> : tensor<ui64>
                    %c_0 = stablehlo.constant dense<5> : tensor<ui64>
                    %3 = stablehlo.clamp %c, %2, %c_0 : \
                    (tensor<ui64>, tensor<3xui64>, tensor<ui64>) -> tensor<3xui64>
                    %4 = \"stablehlo.reduce_window\"(%3, %c) \
                    <{padding = dense<[[2, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 3>}> ({
                    ^bb0(%arg3: tensor<ui64>, %arg4: tensor<ui64>):
                      %c_1 = stablehlo.constant dense<5> : tensor<ui64>
                      %16 = stablehlo.subtract %c_1, %arg3 : tensor<ui64>
                      %17 = stablehlo.minimum %arg4, %16 : tensor<ui64>
                      %18 = stablehlo.add %arg3, %17 : tensor<ui64>
                      stablehlo.return %18 : tensor<ui64>
                    }) : (tensor<3xui64>, tensor<ui64>) -> tensor<3xui64>
                    %5 = stablehlo.pad %4, %c, low = [1], high = [-1], interior = [0] : \
                    (tensor<3xui64>, tensor<ui64>) -> tensor<3xui64>
                    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<5x2xf32>) -> tensor<3x5x2xf32>
                    %7 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<3xui64>) -> tensor<3x5x2xui64>
                    %8 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<3xui64>) -> tensor<3x5x2xui64>
                    %9 = stablehlo.iota dim = 1 : tensor<3x5x2xui64>
                    %10 = stablehlo.compare LE, %7, %9, UNSIGNED : \
                    (tensor<3x5x2xui64>, tensor<3x5x2xui64>) -> tensor<3x5x2xi1>
                    %11 = stablehlo.compare LT, %9, %8, UNSIGNED : \
                    (tensor<3x5x2xui64>, tensor<3x5x2xui64>) -> tensor<3x5x2xi1>
                    %12 = stablehlo.and %10, %11 : tensor<3x5x2xi1>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x5x2xf32>
                    %14 = stablehlo.select %12, %6, %13 : tensor<3x5x2xi1>, tensor<3x5x2xf32>
                    %15 = stablehlo.dot_general %14, %arg1, contracting_dims = [2, 0] x [1, 0], \
                    precision = [DEFAULT, DEFAULT] : \
                    (tensor<3x5x2xf32>, tensor<3x2x1xf32>) -> tensor<5x1xf32>
                    return %15 : tensor<5x1xf32>
                  }
                }
            "},
        );

        let instruction = lowered_ragged_dot_module(RaggedDotLoweringStrategy::Instruction).unwrap();
        assert_eq!(
            instruction,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<5x2xf32>, \
                  %arg1: tensor<3x2x1xf32>, %arg2: tensor<3xi32>) -> tensor<5x1xf32> {
                    %0 = \"chlo.ragged_dot\"(%arg0, %arg1, %arg2) \
                    <{precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>], \
                    ragged_dot_dimension_numbers = \
                    #chlo.ragged_dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1], \
                    lhs_ragged_dimensions = [0], rhs_group_dimensions = [0]>}> : \
                    (tensor<5x2xf32>, tensor<3x2x1xf32>, tensor<3xi32>) -> tensor<5x1xf32>
                    return %0 : tensor<5x1xf32>
                  }
                }
            "},
        );

        let contracting_dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()), 1, None);
        let contracting = lowered_ragged_dot_module_for(
            contracting_dimensions.clone(),
            ArrayType::new_static(DataType::F32, [2, 5]),
            ArrayType::new_static(DataType::F32, [5, 1]),
            ArrayType::new_static(DataType::I32, [3]),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();
        assert_eq!(
            contracting,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x5xf32>, \
                  %arg1: tensor<5x1xf32>, %arg2: tensor<3xi32>) -> tensor<3x2x1xf32> {
                    %0 = stablehlo.subtract %arg2, %arg2 : tensor<3xi32>
                    %1 = stablehlo.maximum %arg2, %0 : tensor<3xi32>
                    %2 = stablehlo.convert %1 : (tensor<3xi32>) -> tensor<3xui64>
                    %c = stablehlo.constant dense<0> : tensor<ui64>
                    %c_0 = stablehlo.constant dense<5> : tensor<ui64>
                    %3 = stablehlo.clamp %c, %2, %c_0 : \
                    (tensor<ui64>, tensor<3xui64>, tensor<ui64>) -> tensor<3xui64>
                    %4 = \"stablehlo.reduce_window\"(%3, %c) \
                    <{padding = dense<[[2, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 3>}> ({
                    ^bb0(%arg3: tensor<ui64>, %arg4: tensor<ui64>):
                      %c_2 = stablehlo.constant dense<5> : tensor<ui64>
                      %25 = stablehlo.subtract %c_2, %arg3 : tensor<ui64>
                      %26 = stablehlo.minimum %arg4, %25 : tensor<ui64>
                      %27 = stablehlo.add %arg3, %26 : tensor<ui64>
                      stablehlo.return %27 : tensor<ui64>
                    }) : (tensor<3xui64>, tensor<ui64>) -> tensor<3xui64>
                    %5 = stablehlo.pad %4, %c, low = [1], high = [-1], interior = [0] : \
                    (tensor<3xui64>, tensor<ui64>) -> tensor<3xui64>
                    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<2x5xf32>) -> tensor<3x2x5xf32>
                    %7 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<3xui64>) -> tensor<3x2x5xui64>
                    %8 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<3xui64>) -> tensor<3x2x5xui64>
                    %9 = stablehlo.iota dim = 2 : tensor<3x2x5xui64>
                    %10 = stablehlo.compare LE, %7, %9, UNSIGNED : \
                    (tensor<3x2x5xui64>, tensor<3x2x5xui64>) -> tensor<3x2x5xi1>
                    %11 = stablehlo.compare LT, %9, %8, UNSIGNED : \
                    (tensor<3x2x5xui64>, tensor<3x2x5xui64>) -> tensor<3x2x5xi1>
                    %12 = stablehlo.and %10, %11 : tensor<3x2x5xi1>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x2x5xf32>
                    %14 = stablehlo.select %12, %6, %13 : tensor<3x2x5xi1>, tensor<3x2x5xf32>
                    %15 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<5x1xf32>) -> tensor<3x5x1xf32>
                    %16 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<3xui64>) -> tensor<3x5x1xui64>
                    %17 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<3xui64>) -> tensor<3x5x1xui64>
                    %18 = stablehlo.iota dim = 1 : tensor<3x5x1xui64>
                    %19 = stablehlo.compare LE, %16, %18, UNSIGNED : \
                    (tensor<3x5x1xui64>, tensor<3x5x1xui64>) -> tensor<3x5x1xi1>
                    %20 = stablehlo.compare LT, %18, %17, UNSIGNED : \
                    (tensor<3x5x1xui64>, tensor<3x5x1xui64>) -> tensor<3x5x1xi1>
                    %21 = stablehlo.and %19, %20 : tensor<3x5x1xi1>
                    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %22 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<3x5x1xf32>
                    %23 = stablehlo.select %21, %15, %22 : tensor<3x5x1xi1>, tensor<3x5x1xf32>
                    %24 = stablehlo.dot_general %14, %23, batching_dims = [0] x [0], \
                    contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : \
                    (tensor<3x2x5xf32>, tensor<3x5x1xf32>) -> tensor<3x2x1xf32>
                    return %24 : tensor<3x2x1xf32>
                  }
                }
            "},
        );
        let contracting_instruction = lowered_ragged_dot_module_for(
            contracting_dimensions,
            ArrayType::new_static(DataType::F32, [2, 5]),
            ArrayType::new_static(DataType::F32, [5, 1]),
            ArrayType::new_static(DataType::I32, [3]),
            RaggedDotLoweringStrategy::Instruction,
        )
        .unwrap();
        assert_eq!(
            contracting_instruction,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x5xf32>, \
                  %arg1: tensor<5x1xf32>, %arg2: tensor<3xi32>) -> tensor<3x2x1xf32> {
                    %0 = \"chlo.ragged_dot\"(%arg0, %arg1, %arg2) \
                    <{precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>], \
                    ragged_dot_dimension_numbers = \
                    #chlo.ragged_dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0], \
                    lhs_ragged_dimensions = [1]>}> : \
                    (tensor<2x5xf32>, tensor<5x1xf32>, tensor<3xi32>) -> tensor<3x2x1xf32>
                    return %0 : tensor<3x2x1xf32>
                  }
                }
            "},
        );

        let batch_dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]), 0, None);
        let batch = lowered_ragged_dot_module_for(
            batch_dimensions.clone(),
            ArrayType::new_static(DataType::F32, [5, 2]),
            ArrayType::new_static(DataType::F32, [5, 2, 1]),
            ArrayType::new_static(DataType::I32, [3]),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();
        assert_eq!(
            batch,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<5x2xf32>, \
                  %arg1: tensor<5x2x1xf32>, %arg2: tensor<3xi32>) -> tensor<5x1xf32> {
                    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], \
                    contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : \
                    (tensor<5x2xf32>, tensor<5x2x1xf32>) -> tensor<5x1xf32>
                    return %0 : tensor<5x1xf32>
                  }
                }
            "},
        );
        let batch_instruction = lowered_ragged_dot_module_for(
            batch_dimensions,
            ArrayType::new_static(DataType::F32, [5, 2]),
            ArrayType::new_static(DataType::F32, [5, 2, 1]),
            ArrayType::new_static(DataType::I32, [3]),
            RaggedDotLoweringStrategy::Instruction,
        )
        .unwrap();
        assert_eq!(
            batch_instruction,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<5x2xf32>, \
                  %arg1: tensor<5x2x1xf32>, %arg2: tensor<3xi32>) -> tensor<5x1xf32> {
                    %0 = \"chlo.ragged_dot\"(%arg0, %arg1, %arg2) \
                    <{precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>], \
                    ragged_dot_dimension_numbers = \
                    #chlo.ragged_dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], \
                    lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1], \
                    lhs_ragged_dimensions = [0]>}> : \
                    (tensor<5x2xf32>, tensor<5x2x1xf32>, tensor<3xi32>) -> tensor<5x1xf32>
                    return %0 : tensor<5x1xf32>
                  }
                }
            "},
        );

        let complex = lowered_ragged_dot_module_for(
            RaggedDotDimensionNumbers::matmul(),
            ArrayType::new_static(DataType::C64, [5, 2]),
            ArrayType::new_static(DataType::C64, [3, 2, 1]),
            ArrayType::new_static(DataType::I32, [3]),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();
        assert_eq!(
            complex,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<5x2xcomplex<f32>>, \
                  %arg1: tensor<3x2x1xcomplex<f32>>, %arg2: tensor<3xi32>) -> tensor<5x1xcomplex<f32>> {
                    %0 = stablehlo.subtract %arg2, %arg2 : tensor<3xi32>
                    %1 = stablehlo.maximum %arg2, %0 : tensor<3xi32>
                    %2 = stablehlo.convert %1 : (tensor<3xi32>) -> tensor<3xui64>
                    %c = stablehlo.constant dense<0> : tensor<ui64>
                    %c_0 = stablehlo.constant dense<5> : tensor<ui64>
                    %3 = stablehlo.clamp %c, %2, %c_0 : \
                    (tensor<ui64>, tensor<3xui64>, tensor<ui64>) -> tensor<3xui64>
                    %4 = \"stablehlo.reduce_window\"(%3, %c) \
                    <{padding = dense<[[2, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 3>}> ({
                    ^bb0(%arg3: tensor<ui64>, %arg4: tensor<ui64>):
                      %c_2 = stablehlo.constant dense<5> : tensor<ui64>
                      %17 = stablehlo.subtract %c_2, %arg3 : tensor<ui64>
                      %18 = stablehlo.minimum %arg4, %17 : tensor<ui64>
                      %19 = stablehlo.add %arg3, %18 : tensor<ui64>
                      stablehlo.return %19 : tensor<ui64>
                    }) : (tensor<3xui64>, tensor<ui64>) -> tensor<3xui64>
                    %5 = stablehlo.pad %4, %c, low = [1], high = [-1], interior = [0] : \
                    (tensor<3xui64>, tensor<ui64>) -> tensor<3xui64>
                    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : \
                    (tensor<5x2xcomplex<f32>>) -> tensor<3x5x2xcomplex<f32>>
                    %7 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<3xui64>) -> tensor<3x5x2xui64>
                    %8 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<3xui64>) -> tensor<3x5x2xui64>
                    %9 = stablehlo.iota dim = 1 : tensor<3x5x2xui64>
                    %10 = stablehlo.compare LE, %7, %9, UNSIGNED : \
                    (tensor<3x5x2xui64>, tensor<3x5x2xui64>) -> tensor<3x5x2xi1>
                    %11 = stablehlo.compare LT, %9, %8, UNSIGNED : \
                    (tensor<3x5x2xui64>, tensor<3x5x2xui64>) -> tensor<3x5x2xi1>
                    %12 = stablehlo.and %10, %11 : tensor<3x5x2xi1>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %13 = stablehlo.complex %cst, %cst_1 : tensor<complex<f32>>
                    %14 = stablehlo.broadcast_in_dim %13, dims = [] : \
                    (tensor<complex<f32>>) -> tensor<3x5x2xcomplex<f32>>
                    %15 = stablehlo.select %12, %6, %14 : tensor<3x5x2xi1>, tensor<3x5x2xcomplex<f32>>
                    %16 = stablehlo.dot_general %15, %arg1, contracting_dims = [2, 0] x [1, 0], \
                    precision = [DEFAULT, DEFAULT] : \
                    (tensor<3x5x2xcomplex<f32>>, tensor<3x2x1xcomplex<f32>>) -> tensor<5x1xcomplex<f32>>
                    return %16 : tensor<5x1xcomplex<f32>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_ragged_dot_operation_lower_strategy() {
        let client = cpu_client(1);
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let left_type = ArrayType::new_static(DataType::F32, [5, 2]);
        let right_type = ArrayType::new_static(DataType::F32, [3, 2, 1]);
        let group_sizes_type = ArrayType::new_static(DataType::I32, [3]);
        let mut builder = ArrayIrProgramBuilder::new();
        let left = builder.add_input(ArrayIrType::Array(left_type));
        let right = builder.add_input(ArrayIrType::Array(right_type));
        let group_sizes = builder.add_input(ArrayIrType::Array(group_sizes_type));
        let output = builder
            .add_instruction(
                RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
                Vec::new(),
                vec![left, right, group_sizes],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let decomposition_options = Options::new(mesh.clone());
        let instruction_options =
            Options::new(mesh).with_ragged_dot_lowering_strategy(RaggedDotLoweringStrategy::Instruction);
        let decomposition = domain.lower_program(&program, 0, &decomposition_options).unwrap();
        let instruction = domain.lower_program(&program, 0, &instruction_options).unwrap();

        assert!(decomposition.stable_hlo().contains("stablehlo.dot_general"));
        assert!(!decomposition.stable_hlo().contains("chlo.ragged_dot"));
        assert_eq!(
            instruction.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<5x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<3x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg2: tensor<3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<5x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
                    %0 = "chlo.ragged_dot"(%arg0, %arg1, %arg2) <{precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>], ragged_dot_dimension_numbers = #chlo.ragged_dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1], lhs_ragged_dimensions = [0], rhs_group_dimensions = [0]>}> : (tensor<5x2xf32>, tensor<3x2x1xf32>, tensor<3xi32>) -> tensor<5x1xf32>
                    return %0 : tensor<5x1xf32>
                  }
                }
            "#},
        );
        assert_ne!(decomposition.stable_hlo(), instruction.stable_hlo());
        assert_ne!(decomposition_options, instruction_options);
        let options_by_representation = HashMap::from([
            (decomposition_options.clone(), "decomposition"),
            (instruction_options.clone(), "instruction"),
        ]);
        assert_eq!(options_by_representation.len(), 2);
        assert_eq!(options_by_representation.get(&decomposition_options), Some(&"decomposition"));
        assert_eq!(options_by_representation.get(&instruction_options), Some(&"instruction"));
    }

    #[test]
    fn test_ragged_dot_operation_lower_preserves_prefix_shaped_group_sizes() {
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]), 1, Some(0));
        let module = lowered_ragged_dot_module_for(
            dimensions.clone(),
            ArrayType::new_static(DataType::F32, [2, 3, 1]),
            ArrayType::new_static(DataType::F32, [2, 2, 1, 1]),
            ArrayType::new_static(DataType::I32, [2, 2]),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x3x1xf32>, \
                  %arg1: tensor<2x2x1x1xf32>, %arg2: tensor<2x2xi32>) -> tensor<2x3x1xf32> {
                    %0 = stablehlo.subtract %arg2, %arg2 : tensor<2x2xi32>
                    %1 = stablehlo.maximum %arg2, %0 : tensor<2x2xi32>
                    %2 = stablehlo.convert %1 : (tensor<2x2xi32>) -> tensor<2x2xui64>
                    %c = stablehlo.constant dense<0> : tensor<ui64>
                    %c_0 = stablehlo.constant dense<3> : tensor<ui64>
                    %3 = stablehlo.clamp %c, %2, %c_0 : \
                    (tensor<ui64>, tensor<2x2xui64>, tensor<ui64>) -> tensor<2x2xui64>
                    %4 = \"stablehlo.reduce_window\"(%3, %c) \
                    <{padding = dense<[[0, 0], [1, 0]]> : tensor<2x2xi64>, \
                    window_dimensions = array<i64: 1, 2>}> ({
                    ^bb0(%arg3: tensor<ui64>, %arg4: tensor<ui64>):
                      %c_1 = stablehlo.constant dense<3> : tensor<ui64>
                      %17 = stablehlo.subtract %c_1, %arg3 : tensor<ui64>
                      %18 = stablehlo.minimum %arg4, %17 : tensor<ui64>
                      %19 = stablehlo.add %arg3, %18 : tensor<ui64>
                      stablehlo.return %19 : tensor<ui64>
                    }) : (tensor<2x2xui64>, tensor<ui64>) -> tensor<2x2xui64>
                    %5 = stablehlo.pad %4, %c, low = [0, 1], high = [0, -1], interior = [0, 0] : \
                    (tensor<2x2xui64>, tensor<ui64>) -> tensor<2x2xui64>
                    %6 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3] : \
                    (tensor<2x3x1xf32>) -> tensor<2x2x3x1xf32>
                    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : \
                    (tensor<2x2xui64>) -> tensor<2x2x3x1xui64>
                    %8 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : \
                    (tensor<2x2xui64>) -> tensor<2x2x3x1xui64>
                    %9 = stablehlo.iota dim = 2 : tensor<2x2x3x1xui64>
                    %10 = stablehlo.compare LE, %7, %9, UNSIGNED : \
                    (tensor<2x2x3x1xui64>, tensor<2x2x3x1xui64>) -> tensor<2x2x3x1xi1>
                    %11 = stablehlo.compare LT, %9, %8, UNSIGNED : \
                    (tensor<2x2x3x1xui64>, tensor<2x2x3x1xui64>) -> tensor<2x2x3x1xi1>
                    %12 = stablehlo.and %10, %11 : tensor<2x2x3x1xi1>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : \
                    (tensor<f32>) -> tensor<2x2x3x1xf32>
                    %14 = stablehlo.select %12, %6, %13 : \
                    tensor<2x2x3x1xi1>, tensor<2x2x3x1xf32>
                    %15 = stablehlo.transpose %14, dims = [1, 0, 2, 3] : \
                    (tensor<2x2x3x1xf32>) -> tensor<2x2x3x1xf32>
                    %16 = stablehlo.dot_general %15, %arg1, batching_dims = [1] x [1], \
                    contracting_dims = [3, 0] x [2, 0], precision = [DEFAULT, DEFAULT] : \
                    (tensor<2x2x3x1xf32>, tensor<2x2x1x1xf32>) -> tensor<2x3x1xf32>
                    return %16 : tensor<2x3x1xf32>
                  }
                }
            "},
        );

        let module = lowered_ragged_dot_module_for(
            dimensions,
            ArrayType::new_static(DataType::F32, [2, 3, 1]),
            ArrayType::new_static(DataType::F32, [2, 2, 1, 1]),
            ArrayType::new_static(DataType::I32, [2, 2]),
            RaggedDotLoweringStrategy::Instruction,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x3x1xf32>, \
                  %arg1: tensor<2x2x1x1xf32>, %arg2: tensor<2x2xi32>) -> tensor<2x3x1xf32> {
                    %0 = \"chlo.ragged_dot\"(%arg0, %arg1, %arg2) \
                    <{precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>], \
                    ragged_dot_dimension_numbers = \
                    #chlo.ragged_dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [1], \
                    lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2], \
                    lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}> : \
                    (tensor<2x3x1xf32>, tensor<2x2x1x1xf32>, tensor<2x2xi32>) -> tensor<2x3x1xf32>
                    return %0 : tensor<2x3x1xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_ragged_dot_operation_lower_instruction_is_unsupported_on_cpu() {
        let module = lowered_ragged_dot_module(RaggedDotLoweringStrategy::Instruction).unwrap();
        let client = cpu_client(1);
        assert!(matches!(
            client.compile(
                &PjrtProgram::Mlir { bytecode: module.into_bytes() },
                &ragged_dot_cpu_compilation_options(),
            ),
            Err(PjrtError::Unimplemented { message, .. })
                if message
                    == "HLO opcode `ragged-dot` is not supported by XLA:CPU ThunkEmitter. Standalone elemental and \
                        reduction operations must be wrapped into fusions by FusionWrapper.",
        ));
    }

    #[test]
    fn test_ragged_dot_operation_lower_decomposition_executes_on_cpu() {
        let left_values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let right_values = [10.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let group_sizes_values = [1i32, 0, 3];
        let left = CoreArray::matrix(5, 2, left_values.to_vec()).unwrap();
        let right =
            CoreArray::from_elements::<f32>(ArrayType::new_static(DataType::F32, [3, 2, 1]), &right_values).unwrap();
        let eager = left
            .ragged_dot(
                &right,
                &CoreArray::vector(group_sizes_values.to_vec()).unwrap(),
                &RaggedDotDimensionNumbers::matmul(),
            )
            .unwrap();
        let expected = eager.to_f64s().into_iter().map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(expected, vec![12.0, 32.0, 50.0, 68.0, 0.0]);
        let group_sizes_data = values_to_bytes(&group_sizes_values);
        assert_eq!(
            execute_ragged_dot_decomposition_on_cpu(
                RaggedDotDimensionNumbers::matmul(),
                ArrayType::new_static(DataType::F32, [5, 2]),
                ArrayType::new_static(DataType::F32, [3, 2, 1]),
                ArrayType::new_static(DataType::I32, [3]),
                &left_values,
                &right_values,
                group_sizes_data.as_slice(),
            ),
            expected,
        );
    }

    #[test]
    fn test_ragged_dot_operation_lower_contracting_decomposition_executes_on_cpu() {
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()), 1, None);
        let left_values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let right_values = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let group_sizes_values = [2i32, 0, 3];
        let left = CoreArray::matrix(2, 5, left_values.to_vec()).unwrap();
        let right = CoreArray::matrix(5, 1, right_values.to_vec()).unwrap();
        let eager = left
            .ragged_dot(&right, &CoreArray::vector(group_sizes_values.to_vec()).unwrap(), &dimensions)
            .unwrap();
        let expected = eager.to_f64s().into_iter().map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(expected, vec![50.0, 200.0, 0.0, 0.0, 500.0, 1_100.0]);
        let group_sizes_data = values_to_bytes(&group_sizes_values);
        assert_eq!(
            execute_ragged_dot_decomposition_on_cpu(
                dimensions,
                ArrayType::new_static(DataType::F32, [2, 5]),
                ArrayType::new_static(DataType::F32, [5, 1]),
                ArrayType::new_static(DataType::I32, [3]),
                &left_values,
                &right_values,
                group_sizes_data.as_slice(),
            ),
            expected,
        );
    }

    #[test]
    fn test_ragged_dot_operation_lower_decomposition_clamps_signed_group_sizes_with_dynamic_prefix() {
        let batch = dynamic_dimension("batch", Some(4));
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]), 1, Some(0));
        let module = lowered_ragged_dot_module_for(
            dimensions,
            ArrayType::new(DataType::F32, Shape::new(vec![batch.clone(), 3.into(), 1.into()])),
            ArrayType::new(DataType::F32, Shape::new(vec![2.into(), batch.clone(), 1.into(), 1.into()])),
            ArrayType::new(DataType::I32, Shape::new(vec![batch, 2.into()])),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<?x3x1xf32, #stablehlo.bounds<3, ?, ?>>, %arg1: tensor<2x?x1x1xf32, \
                      #stablehlo.bounds<?, 3, ?, ?>>, %arg2: tensor<?x2xi32, #stablehlo.bounds<3, ?>>) -> \
                      tensor<?x3x1xf32, #stablehlo.bounds<3, ?, ?>> {
                    %0 = stablehlo.subtract %arg2, %arg2 : tensor<?x2xi32, #stablehlo.bounds<3, ?>>
                    %1 = stablehlo.maximum %arg2, %0 : tensor<?x2xi32, #stablehlo.bounds<3, ?>>
                    %2 = stablehlo.convert %1 : (tensor<?x2xi32, #stablehlo.bounds<3, ?>>) -> tensor<?x2xui64, \
                        #stablehlo.bounds<3, ?>>
                    %3 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<?x2xui64, #stablehlo.bounds<3, ?>>) -> \
                        tensor<i32>
                    %c = stablehlo.constant dense<3> : tensor<i32>
                    %4 = stablehlo.set_dimension_size %2, %c, dim = 0 : (tensor<?x2xui64, #stablehlo.bounds<3, ?>>, \
                        tensor<i32>) -> tensor<3x2xui64>
                    %5 = stablehlo.iota dim = 0 : tensor<3x2xi32>
                    %6 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<3x2xi32>
                    %7 = stablehlo.compare LT, %5, %6, SIGNED : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi1>
                    %c_0 = stablehlo.constant dense<0> : tensor<ui64>
                    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui64>) -> tensor<3x2xui64>
                    %9 = stablehlo.select %7, %4, %8 : tensor<3x2xi1>, tensor<3x2xui64>
                    %c_1 = stablehlo.constant dense<0> : tensor<ui64>
                    %c_2 = stablehlo.constant dense<3> : tensor<ui64>
                    %10 = stablehlo.clamp %c_1, %9, %c_2 : (tensor<ui64>, tensor<3x2xui64>, tensor<ui64>) -> \
                        tensor<3x2xui64>
                    %11 = \"stablehlo.reduce_window\"(%10, %c_1) <{padding = dense<[[0, 0], [1, 0]]> : \
                        tensor<2x2xi64>, window_dimensions = array<i64: 1, 2>}> ({
                    ^bb0(%arg3: tensor<ui64>, %arg4: tensor<ui64>):
                      %c_3 = stablehlo.constant dense<3> : tensor<ui64>
                      %27 = stablehlo.subtract %c_3, %arg3 : tensor<ui64>
                      %28 = stablehlo.minimum %arg4, %27 : tensor<ui64>
                      %29 = stablehlo.add %arg3, %28 : tensor<ui64>
                      stablehlo.return %29 : tensor<ui64>
                    }) : (tensor<3x2xui64>, tensor<ui64>) -> tensor<3x2xui64>
                    %12 = stablehlo.pad %11, %c_1, low = [0, 1], high = [0, -1], interior = [0, 0] : \
                        (tensor<3x2xui64>, tensor<ui64>) -> tensor<3x2xui64>
                    %13 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3] : (tensor<?x3x1xf32, #stablehlo.bounds<3, \
                        ?, ?>>) -> tensor<?x2x3x1xf32, #stablehlo.bounds<3, ?, ?, ?>>
                    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<3x2xui64>) -> tensor<3x2x3x1xui64>
                    %15 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<3x2xui64>) -> tensor<3x2x3x1xui64>
                    %16 = stablehlo.iota dim = 2 : tensor<3x2x3x1xui64>
                    %17 = stablehlo.compare LE, %14, %16, UNSIGNED : (tensor<3x2x3x1xui64>, tensor<3x2x3x1xui64>) -> \
                        tensor<3x2x3x1xi1>
                    %18 = stablehlo.compare LT, %16, %15, UNSIGNED : (tensor<3x2x3x1xui64>, tensor<3x2x3x1xui64>) -> \
                        tensor<3x2x3x1xi1>
                    %19 = stablehlo.and %17, %18 : tensor<3x2x3x1xi1>
                    %20 = stablehlo.get_dimension_size %13, dim = 0 : (tensor<?x2x3x1xf32, #stablehlo.bounds<3, ?, ?, \
                        ?>>) -> tensor<i32>
                    %21 = stablehlo.set_dimension_size %19, %20, dim = 0 : (tensor<3x2x3x1xi1>, tensor<i32>) -> \
                        tensor<?x2x3x1xi1, #stablehlo.bounds<3, ?, ?, ?>>
                    %22 = stablehlo.xor %21, %21 : tensor<?x2x3x1xi1, #stablehlo.bounds<3, ?, ?, ?>>
                    %23 = stablehlo.convert %22 : (tensor<?x2x3x1xi1, #stablehlo.bounds<3, ?, ?, ?>>) -> \
                        tensor<?x2x3x1xf32, #stablehlo.bounds<3, ?, ?, ?>>
                    %24 = stablehlo.select %21, %13, %23 : tensor<?x2x3x1xi1, #stablehlo.bounds<3, ?, ?, ?>>, \
                        tensor<?x2x3x1xf32, #stablehlo.bounds<3, ?, ?, ?>>
                    %25 = stablehlo.transpose %24, dims = [1, 0, 2, 3] : (tensor<?x2x3x1xf32, #stablehlo.bounds<3, ?, \
                        ?, ?>>) -> tensor<2x?x3x1xf32, #stablehlo.bounds<?, 3, ?, ?>>
                    %26 = stablehlo.dot_general %25, %arg1, batching_dims = [1] x [1], contracting_dims = [3, 0] x [2, \
                        0], precision = [DEFAULT, DEFAULT] : (tensor<2x?x3x1xf32, #stablehlo.bounds<?, 3, ?, ?>>, \
                        tensor<2x?x1x1xf32, #stablehlo.bounds<?, 3, ?, ?>>) -> tensor<?x3x1xf32, #stablehlo.bounds<3, \
                        ?, ?>>
                    return %26 : tensor<?x3x1xf32, #stablehlo.bounds<3, ?, ?>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_ragged_dot_operation_lower_decomposition_clamps_negative_group_sizes_before_widening_on_cpu() {
        let left_values = [1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let right_values = [2.0f32, 0.0, 3.0, 0.0, 5.0, 0.0];
        let group_sizes_values = [1i32, -1, 2];
        let group_sizes_data = values_to_bytes(&group_sizes_values);
        assert_eq!(
            execute_ragged_dot_decomposition_on_cpu(
                RaggedDotDimensionNumbers::matmul(),
                ArrayType::new_static(DataType::F32, [4, 2]),
                ArrayType::new_static(DataType::F32, [3, 2, 1]),
                ArrayType::new_static(DataType::I32, [3]),
                &left_values,
                &right_values,
                group_sizes_data.as_slice(),
            ),
            vec![2.0, 10.0, 15.0, 0.0],
        );
    }

    #[test]
    fn test_ragged_dot_operation_lower_decomposition_broadcasts_shared_group_sizes_across_dynamic_prefix() {
        let batch = dynamic_dimension("batch", Some(4));
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]), 1, Some(0));
        let module = lowered_ragged_dot_module_for(
            dimensions,
            ArrayType::new(DataType::F32, Shape::new(vec![batch.clone(), 3.into(), 1.into()])),
            ArrayType::new(DataType::F32, Shape::new(vec![2.into(), batch, 1.into(), 1.into()])),
            ArrayType::new_static(DataType::I32, [2]),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<?x3x1xf32, #stablehlo.bounds<3, ?, ?>>, %arg1: tensor<2x?x1x1xf32, \
                      #stablehlo.bounds<?, 3, ?, ?>>, %arg2: tensor<2xi32>) -> tensor<?x3x1xf32, #stablehlo.bounds<3, \
                      ?, ?>> {
                    %0 = stablehlo.subtract %arg2, %arg2 : tensor<2xi32>
                    %1 = stablehlo.maximum %arg2, %0 : tensor<2xi32>
                    %2 = stablehlo.convert %1 : (tensor<2xi32>) -> tensor<2xui64>
                    %c = stablehlo.constant dense<0> : tensor<ui64>
                    %c_0 = stablehlo.constant dense<3> : tensor<ui64>
                    %3 = stablehlo.clamp %c, %2, %c_0 : (tensor<ui64>, tensor<2xui64>, tensor<ui64>) -> tensor<2xui64>
                    %4 = \"stablehlo.reduce_window\"(%3, %c) <{padding = dense<[[1, 0]]> : tensor<1x2xi64>, \
                        window_dimensions = array<i64: 2>}> ({
                    ^bb0(%arg3: tensor<ui64>, %arg4: tensor<ui64>):
                      %c_1 = stablehlo.constant dense<3> : tensor<ui64>
                      %20 = stablehlo.subtract %c_1, %arg3 : tensor<ui64>
                      %21 = stablehlo.minimum %arg4, %20 : tensor<ui64>
                      %22 = stablehlo.add %arg3, %21 : tensor<ui64>
                      stablehlo.return %22 : tensor<ui64>
                    }) : (tensor<2xui64>, tensor<ui64>) -> tensor<2xui64>
                    %5 = stablehlo.pad %4, %c, low = [1], high = [-1], interior = [0] : (tensor<2xui64>, tensor<ui64>) \
                        -> tensor<2xui64>
                    %6 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3] : (tensor<?x3x1xf32, #stablehlo.bounds<3, \
                        ?, ?>>) -> tensor<?x2x3x1xf32, #stablehlo.bounds<3, ?, ?, ?>>
                    %7 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<2xui64>) -> tensor<3x2x3x1xui64>
                    %8 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<2xui64>) -> tensor<3x2x3x1xui64>
                    %9 = stablehlo.iota dim = 2 : tensor<3x2x3x1xui64>
                    %10 = stablehlo.compare LE, %7, %9, UNSIGNED : (tensor<3x2x3x1xui64>, tensor<3x2x3x1xui64>) -> \
                        tensor<3x2x3x1xi1>
                    %11 = stablehlo.compare LT, %9, %8, UNSIGNED : (tensor<3x2x3x1xui64>, tensor<3x2x3x1xui64>) -> \
                        tensor<3x2x3x1xi1>
                    %12 = stablehlo.and %10, %11 : tensor<3x2x3x1xi1>
                    %13 = stablehlo.get_dimension_size %6, dim = 0 : (tensor<?x2x3x1xf32, #stablehlo.bounds<3, ?, ?, \
                        ?>>) -> tensor<i32>
                    %14 = stablehlo.set_dimension_size %12, %13, dim = 0 : (tensor<3x2x3x1xi1>, tensor<i32>) -> \
                        tensor<?x2x3x1xi1, #stablehlo.bounds<3, ?, ?, ?>>
                    %15 = stablehlo.xor %14, %14 : tensor<?x2x3x1xi1, #stablehlo.bounds<3, ?, ?, ?>>
                    %16 = stablehlo.convert %15 : (tensor<?x2x3x1xi1, #stablehlo.bounds<3, ?, ?, ?>>) -> \
                        tensor<?x2x3x1xf32, #stablehlo.bounds<3, ?, ?, ?>>
                    %17 = stablehlo.select %14, %6, %16 : tensor<?x2x3x1xi1, #stablehlo.bounds<3, ?, ?, ?>>, \
                        tensor<?x2x3x1xf32, #stablehlo.bounds<3, ?, ?, ?>>
                    %18 = stablehlo.transpose %17, dims = [1, 0, 2, 3] : (tensor<?x2x3x1xf32, #stablehlo.bounds<3, ?, \
                        ?, ?>>) -> tensor<2x?x3x1xf32, #stablehlo.bounds<?, 3, ?, ?>>
                    %19 = stablehlo.dot_general %18, %arg1, batching_dims = [1] x [1], contracting_dims = [3, 0] x [2, \
                        0], precision = [DEFAULT, DEFAULT] : (tensor<2x?x3x1xf32, #stablehlo.bounds<?, 3, ?, ?>>, \
                        tensor<2x?x1x1xf32, #stablehlo.bounds<?, 3, ?, ?>>) -> tensor<?x3x1xf32, #stablehlo.bounds<3, \
                        ?, ?>>
                    return %19 : tensor<?x3x1xf32, #stablehlo.bounds<3, ?, ?>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_ragged_dot_operation_lower_decomposition_executes_prefix_below_dynamic_bound() {
        let batch = dynamic_dimension("batch", Some(4));
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]), 1, Some(0));
        let dynamic_module = lowered_ragged_dot_module_for(
            dimensions,
            ArrayType::new(DataType::F32, Shape::new(vec![batch.clone(), 3.into(), 1.into()])),
            ArrayType::new(DataType::F32, Shape::new(vec![2.into(), batch.clone(), 1.into(), 1.into()])),
            ArrayType::new(DataType::I32, Shape::new(vec![batch, 2.into()])),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();
        // CPU PJRT cannot expose bounded dynamic parameters at the executable boundary because it lacks XLA's
        // `PadToStatic` custom call. A static wrapper creates logical size two inside the module, so this executes the
        // generated decomposition with a runtime prefix below its physical bound of three.
        let inner_module = dynamic_module.strip_suffix("}\n").unwrap().replacen(
            "func.func @main",
            "func.func private @ragged_dot_dynamic",
            1,
        );
        let module = format!(
            "{inner_module}{}",
            indoc! {"
                  func.func @main(
                      %arg0: tensor<3x3x1xf32>,
                      %arg1: tensor<2x3x1x1xf32>,
                      %arg2: tensor<3x2xi32>
                  ) -> tensor<3x3x1xf32> {
                    %c2 = stablehlo.constant dense<2> : tensor<i32>
                    %0 = stablehlo.set_dimension_size %arg0, %c2, dim = 0 :
                      (tensor<3x3x1xf32>, tensor<i32>) -> tensor<?x3x1xf32, #stablehlo.bounds<3, ?, ?>>
                    %1 = stablehlo.set_dimension_size %arg1, %c2, dim = 1 :
                      (tensor<2x3x1x1xf32>, tensor<i32>) -> tensor<2x?x1x1xf32, #stablehlo.bounds<?, 3, ?, ?>>
                    %2 = stablehlo.set_dimension_size %arg2, %c2, dim = 0 :
                      (tensor<3x2xi32>, tensor<i32>) -> tensor<?x2xi32, #stablehlo.bounds<3, ?>>
                    %3 = func.call @ragged_dot_dynamic(%0, %1, %2) :
                      (tensor<?x3x1xf32, #stablehlo.bounds<3, ?, ?>>,
                       tensor<2x?x1x1xf32, #stablehlo.bounds<?, 3, ?, ?>>,
                       tensor<?x2xi32, #stablehlo.bounds<3, ?>>) ->
                      tensor<?x3x1xf32, #stablehlo.bounds<3, ?, ?>>
                    %c3 = stablehlo.constant dense<3> : tensor<i32>
                    %4 = stablehlo.set_dimension_size %3, %c3, dim = 0 :
                      (tensor<?x3x1xf32, #stablehlo.bounds<3, ?, ?>>, tensor<i32>) -> tensor<3x3x1xf32>
                    return %4 : tensor<3x3x1xf32>
                  }
                }
            "},
        );

        let client = cpu_client(1);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        // The physical buffers deliberately carry garbage rather than zeros in every slot beyond the live prefix of the
        // bounded dynamic batch axis. Each batch position computes independently, so the asserted live prefix can
        // only stay correct if the lowering itself confines the padded positions, rather than relying on the caller
        // zeroing them.
        let inputs = [
            client
                .buffer(
                    values_to_bytes(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).as_slice(),
                    BufferType::F32,
                    &[3, 3, 1],
                    None,
                    device.clone(),
                    None,
                )
                .unwrap(),
            client
                .buffer(
                    values_to_bytes(&[10.0f32, 2.0, -9.0, 20.0, 3.0, -11.0]).as_slice(),
                    BufferType::F32,
                    &[2, 3, 1, 1],
                    None,
                    device.clone(),
                    None,
                )
                .unwrap(),
            client
                .buffer(
                    values_to_bytes(&[1i32, 2, 2, 1, 5, 7]).as_slice(),
                    BufferType::I32,
                    &[3, 2],
                    None,
                    device,
                    None,
                )
                .unwrap(),
        ]
        .into_iter()
        .map(|buffer| ExecutionInput { buffer: Arc::new(buffer), donatable: false })
        .collect::<Vec<_>>();
        let mut executions = executable
            .execute(
                vec![ExecutionDeviceInputs { inputs: inputs.as_slice(), ..Default::default() }],
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        let output = executions.remove(0).outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        let output = values_from_bytes::<f32>(output.as_slice());
        assert_eq!(&output[..6], &[10.0, 40.0, 60.0, 8.0, 10.0, 18.0]);
    }

    #[test]
    fn test_ragged_dot_operation_lower_decomposition_widens_narrow_group_metadata_and_iota() {
        let module = lowered_ragged_dot_module_for(
            RaggedDotDimensionNumbers::matmul(),
            ArrayType::new_static(DataType::F32, [5, 2]),
            ArrayType::new_static(DataType::F32, [2, 2, 1]),
            ArrayType::new_static(DataType::U2, [2]),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<5x2xf32>, \
                  %arg1: tensor<2x2x1xf32>, %arg2: tensor<2xui2>) -> tensor<5x1xf32> {
                    %0 = stablehlo.convert %arg2 : (tensor<2xui2>) -> tensor<2xui64>
                    %c = stablehlo.constant dense<0> : tensor<ui64>
                    %c_0 = stablehlo.constant dense<5> : tensor<ui64>
                    %1 = stablehlo.clamp %c, %0, %c_0 : \
                    (tensor<ui64>, tensor<2xui64>, tensor<ui64>) -> tensor<2xui64>
                    %2 = \"stablehlo.reduce_window\"(%1, %c) \
                    <{padding = dense<[[1, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 2>}> ({
                    ^bb0(%arg3: tensor<ui64>, %arg4: tensor<ui64>):
                      %c_1 = stablehlo.constant dense<5> : tensor<ui64>
                      %14 = stablehlo.subtract %c_1, %arg3 : tensor<ui64>
                      %15 = stablehlo.minimum %arg4, %14 : tensor<ui64>
                      %16 = stablehlo.add %arg3, %15 : tensor<ui64>
                      stablehlo.return %16 : tensor<ui64>
                    }) : (tensor<2xui64>, tensor<ui64>) -> tensor<2xui64>
                    %3 = stablehlo.pad %2, %c, low = [1], high = [-1], interior = [0] : \
                    (tensor<2xui64>, tensor<ui64>) -> tensor<2xui64>
                    %4 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<5x2xf32>) -> tensor<2x5x2xf32>
                    %5 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xui64>) -> tensor<2x5x2xui64>
                    %6 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xui64>) -> tensor<2x5x2xui64>
                    %7 = stablehlo.iota dim = 1 : tensor<2x5x2xui64>
                    %8 = stablehlo.compare LE, %5, %7, UNSIGNED : \
                    (tensor<2x5x2xui64>, tensor<2x5x2xui64>) -> tensor<2x5x2xi1>
                    %9 = stablehlo.compare LT, %7, %6, UNSIGNED : \
                    (tensor<2x5x2xui64>, tensor<2x5x2xui64>) -> tensor<2x5x2xi1>
                    %10 = stablehlo.and %8, %9 : tensor<2x5x2xi1>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x5x2xf32>
                    %12 = stablehlo.select %10, %4, %11 : tensor<2x5x2xi1>, tensor<2x5x2xf32>
                    %13 = stablehlo.dot_general %12, %arg1, contracting_dims = [2, 0] x [1, 0], \
                    precision = [DEFAULT, DEFAULT] : \
                    (tensor<2x5x2xf32>, tensor<2x2x1xf32>) -> tensor<5x1xf32>
                    return %13 : tensor<5x1xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_ragged_dot_operation_lower_decomposition_saturates_overflowing_group_intervals_on_cpu() {
        let left_values = [1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let right_values = [2.0f32, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0];
        let group_sizes_values = [u64::MAX, u64::MAX, 2, 1];
        let left = CoreArray::matrix(4, 2, left_values.to_vec()).unwrap();
        let right =
            CoreArray::from_elements::<f32>(ArrayType::new_static(DataType::F32, [4, 2, 1]), &right_values).unwrap();
        let eager = left
            .ragged_dot(
                &right,
                &CoreArray::vector(group_sizes_values.to_vec()).unwrap(),
                &RaggedDotDimensionNumbers::matmul(),
            )
            .unwrap();
        let expected = eager.to_f64s().into_iter().map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(expected, vec![2.0, 4.0, 6.0, 8.0]);
        let group_sizes_data = values_to_bytes(&group_sizes_values);
        assert_eq!(
            execute_ragged_dot_decomposition_on_cpu(
                RaggedDotDimensionNumbers::matmul(),
                ArrayType::new_static(DataType::F32, [4, 2]),
                ArrayType::new_static(DataType::F32, [4, 2, 1]),
                ArrayType::new_static(DataType::U64, [4]),
                &left_values,
                &right_values,
                group_sizes_data.as_slice(),
            ),
            expected,
        );
    }

    #[test]
    fn test_einsum_matches_reference_backend() {
        // An einsum is staged as dots, sums, diagonals, squeezes, and transposes, which lower and execute like the
        // reference backend. Every input holds small integers, so all results are exact on both backends. Elements
        // are compared as values, because a dot without contracting dimensions keeps the sign of a zero product on
        // XLA, while the reference kernel adds every product to a positive zero.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let cases: &[(&str, &[&[usize]])] = &[
            ("ij,jk->ik", &[&[2, 3], &[3, 4]]),
            ("ijk,kl->jil", &[&[2, 3, 4], &[4, 2]]),
            ("...ij,...jk->...ik", &[&[2, 2, 3], &[2, 3, 2]]),
            ("ij,jk,kl->il", &[&[2, 3], &[3, 4], &[4, 2]]),
            ("ii->", &[&[3, 3]]),
            ("iij,j->i", &[&[2, 2, 3], &[3]]),
            ("ij,ij->ij", &[&[2, 3], &[1, 3]]),
            ("ij,k->i", &[&[2, 3], &[4]]),
            ("ab,bc,cd,de,ef->af", &[&[2, 3], &[3, 2], &[2, 3], &[3, 2], &[2, 2]]),
            ("ij->ji", &[&[2, 3]]),
        ];
        for (subscripts, shapes) in cases {
            let mut inputs = Vec::new();
            let mut reference_inputs = Vec::new();
            for (index, shape) in shapes.iter().enumerate() {
                let count = shape.iter().product::<usize>();
                let values = (0..count).map(|value| ((value * (index + 2)) % 7) as f32 - 3.0).collect::<Vec<_>>();
                reference_inputs.push(
                    CoreArray::from_elements(ArrayType::new_static(DataType::F32, shape.to_vec()), &values).unwrap(),
                );
                inputs.push(replicated_array(&domain, &mesh, DataType::F32, shape, &values_to_bytes(&values)));
            }
            let output = Einsum::einsum(subscripts, inputs.iter()).unwrap();
            let reference_output = CoreArray::einsum(subscripts, reference_inputs.iter()).unwrap();
            assert_eq!(output.r#type().shape(), reference_output.r#type().shape(), "{subscripts}");
            assert_eq!(
                values_from_bytes::<f32>(&output.copy_to_host().unwrap()),
                reference_output.elements::<f32>().unwrap(),
                "{subscripts}",
            );
        }

        // The options select the path, the precision, and the accumulation type of the staged contractions.
        let values = [3.0f32, 0.5, 2.0, 4.0].map(bf16::from_f32);
        let reference = CoreArray::matrix(2, 2, values.to_vec()).unwrap();
        let input = replicated_array(&domain, &mesh, DataType::BF16, &[2, 2], &reference.logical_bytes());
        let specification = "ij,jk,kl->il".parse::<EinsumSpecification>().unwrap();
        let options = EinsumOptions::new()
            .with_path(EinsumPathStrategy::Unoptimized)
            .with_precision(DotAlgorithmPreset::BF16_BF16_F32)
            .with_accumulation_type(DataType::F32);
        let output = Einsum::einsum_with_options(&specification, [&input, &input, &input], &options).unwrap();
        let reference_output =
            CoreArray::einsum_with_options(&specification, [&reference, &reference, &reference], &options).unwrap();
        assert_eq!(reference_output.r#type().data_type(), DataType::F32);
        assert_eq!(output.copy_to_host().unwrap(), reference_output.logical_bytes());

        // Boolean and complex inputs, accumulation types of another kind than the inputs, algorithms over integers,
        // reductions in a path, and the dynamic programming search agree with the reference backend as well. Every result is exact and no contraction is
        // an outer product of signed zeros, so bytes are compared.
        let booleans = CoreArray::matrix(2, 2, vec![false, true, true, true]).unwrap();
        let fractions = CoreArray::matrix(2, 2, vec![1.5f32, 2.5, 0.5, 0.5]).unwrap();
        let integers = CoreArray::matrix(2, 2, vec![1u8, 2, 3, 4]).unwrap();
        let counted = EinsumOptions::new().with_accumulation_type(DataType::I32);
        let reduced = EinsumPathStrategy::Explicit(vec![
            EinsumPathStep::Reduction(0),
            EinsumPathStep::Contraction(0, 1),
            EinsumPathStep::Contraction(0, 1),
        ]);
        let complex = [(1.5f32, 0.5f32), (2.5, 0.25), (0.5, -1.0), (2.0, 0.0)];
        let complex = complex.map(|(real, imaginary)| Complex::new(real, imaginary));
        let complex = CoreArray::matrix(2, 2, complex.to_vec()).unwrap();
        let narrow = [1.0078125f32, 1.4140625, 0.5, 2.0].map(bf16::from_f32);
        let narrow = CoreArray::matrix(2, 2, narrow.to_vec()).unwrap();
        let truncated = EinsumOptions::new()
            .with_accumulation_type(DataType::I32)
            .with_precision(DotAlgorithmPreset::F32_F32_F32);
        let cases: Vec<(&str, &CoreArray, EinsumOptions)> = vec![
            ("ij,jk->ik", &booleans, EinsumOptions::new()),
            ("ij,jk,kl->il", &booleans, EinsumOptions::new()),
            ("ij,jk,kl->", &booleans, EinsumOptions::new()),
            ("ij->i", &booleans, EinsumOptions::new()),
            ("ij,jk,kl->il", &booleans, counted.clone()),
            ("ij,jk,kl->il", &fractions, counted.clone()),
            ("ij,jk->ik", &fractions, counted.clone()),
            ("ij,jk,kl->il", &integers, counted),
            ("ij,jk,kl->il", &fractions, EinsumOptions::new().with_path(EinsumPathStrategy::DynamicProgramming)),
            ("ij,jk,kl->l", &fractions, EinsumOptions::new().with_path(reduced)),
            ("ij,jk->ik", &complex, EinsumOptions::new().with_accumulation_type(DataType::F32)),
            ("ij,jk->ik", &complex, EinsumOptions::new().with_accumulation_type(DataType::I32)),
            ("ij,jk->ik", &narrow, EinsumOptions::new().with_accumulation_type(DataType::C64)),
            ("ij,jk->ik", &narrow, truncated),
            ("ij,jk->ik", &integers, EinsumOptions::new().with_precision(DotAlgorithmPreset::F32_F32_F32)),
            ("ij,jk->ik", &booleans, EinsumOptions::new().with_precision(DotAlgorithmPreset::F32_F32_F32)),
        ];
        for (subscripts, reference, options) in cases {
            let input =
                replicated_array(&domain, &mesh, reference.r#type().data_type(), &[2, 2], &reference.logical_bytes());
            let specification = subscripts.parse::<EinsumSpecification>().unwrap();
            let count = specification.inputs().len();
            let output = Einsum::einsum_with_options(&specification, vec![&input; count], &options).unwrap();
            let reference_output =
                CoreArray::einsum_with_options(&specification, vec![reference; count], &options).unwrap();
            assert_eq!(output.r#type().data_type(), reference_output.r#type().data_type(), "{subscripts}");
            assert_eq!(output.copy_to_host().unwrap(), reference_output.logical_bytes(), "{subscripts} {options:?}");
        }
    }

    #[test]
    fn test_einsum_composite_matches_reference_backend() {
        // An einsum function with multiple steps executes each call as one eager dispatch of its whole body, where
        // the capability dispatches every step separately. Both agree with the reference backend.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let shapes: [&[usize]; 3] = [&[2, 3], &[3, 4], &[4, 2]];
        let mut inputs = Vec::new();
        let mut reference_inputs = Vec::new();
        for (index, shape) in shapes.iter().enumerate() {
            let count = shape.iter().product::<usize>();
            let values = (0..count).map(|value| ((value * (index + 2)) % 7) as f32 - 3.0).collect::<Vec<_>>();
            reference_inputs
                .push(CoreArray::from_elements(ArrayType::new_static(DataType::F32, shape.to_vec()), &values).unwrap());
            inputs.push(replicated_array(&domain, &mesh, DataType::F32, shape, &values_to_bytes(&values)));
        }
        let function = einsum::<ProjectedContext<Domain<'_>, ArrayType>>("ij,jk,kl->il").unwrap();
        let reference_output = CoreArray::einsum("ij,jk,kl->il", reference_inputs.iter()).unwrap();
        let dispatches = || {
            let statistics = domain.eager_dispatch_statistics();
            (statistics.misses, statistics.hits)
        };

        let first = function.call(inputs.clone()).unwrap();
        assert_eq!(dispatches(), (1, 0));
        let second = function.call(inputs.clone()).unwrap();
        assert_eq!(dispatches(), (1, 1));
        for output in [first, second] {
            assert_eq!(output.r#type().shape(), reference_output.r#type().shape());
            assert_eq!(
                values_from_bytes::<f32>(&output.copy_to_host().unwrap()),
                reference_output.elements::<f32>().unwrap(),
            );
        }
        let inline = Einsum::einsum("ij,jk,kl->il", inputs.iter()).unwrap();
        assert_eq!(dispatches(), (3, 1));
        assert_eq!(
            values_from_bytes::<f32>(&inline.copy_to_host().unwrap()),
            reference_output.elements::<f32>().unwrap(),
        );

        // The derived rules differentiate the body of the call.
        let gradient = differentiate_at(inputs[0].clone())
            .with_captures((inputs[1].clone(), inputs[2].clone()))
            .gradient(|left, (middle, right)| function.call(vec![left, middle, right])?.einsum_with("il->", []))
            .unwrap();
        let reference_gradient = differentiate_at(reference_inputs[0].clone())
            .with_captures((reference_inputs[1].clone(), reference_inputs[2].clone()))
            .gradient(|left, (middle, right)| Einsum::einsum("ij,jk,kl->", [&left, &middle, &right]))
            .unwrap();
        assert_eq!(
            values_from_bytes::<f32>(&gradient.copy_to_host().unwrap()),
            reference_gradient.elements::<f32>().unwrap(),
        );
    }

    #[test]
    fn test_einsum_composite_lowering_shares_one_function() {
        // Calls of one einsum function with equal input types lower to calls of one shared function, which holds the
        // decomposition once.
        let function = einsum::<ProjectedContext<Domain<'static>, ArrayType>>("ij,jk,kl->il").unwrap();
        let traced: TracedProgram<Vec<ArrayType>, ArrayType> = trace(
            move |inputs: Vec<ArrayTracer>| {
                let first = function.call(inputs.clone()).unwrap();
                function.call(vec![first, inputs[1].clone(), inputs[2].clone()]).unwrap()
            },
            vec![ArrayType::new_static(DataType::F32, [2, 2]); 3],
        )
        .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.einsum.v1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                    %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                    return %1 : tensor<2x2xf32>
                  }
                  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = call @ryft.einsum.v1(%arg0, %arg1, %arg2) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                    %1 = call @ryft.einsum.v1(%0, %arg1, %arg2) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                    return %1 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }
}
