use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, ComparisonDirection, DataType, DotDimensionNumbers, DotOperation, ProgramError, RaggedDotMode,
    RaggedDotOperation, Shape,
};
use ryft_mlir::dialects::stable_hlo::Precision;
use ryft_mlir::dialects::{chlo, stable_hlo};
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::types::{
    lower_physical_bound_value, lower_restore_dynamic_dimensions, lower_tensor_type, lower_u64_scalar_constant,
    lower_unplaced_constant_output, physical_bound_type,
};
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// XLA representation used when lowering grouped generalized dots. Selection is compilation-wide through
/// [`XlaOptions::with_ragged_dot_lowering_strategy`](crate::XlaOptions::with_ragged_dot_lowering_strategy).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum RaggedDotLoweringStrategy {
    /// Expand grouped dots into masking plus ordinary dense generalized dots. This is the default used by
    /// [`XlaOptions::new`](crate::XlaOptions::new).
    #[default]
    Decomposition,

    /// Emit dedicated CHLO `ragged_dot` instructions.
    Instruction,
}

impl LowerableOperation for DotOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
        let dimension_numbers = self.dimensions();
        let dimensions = lowerer.context.stable_hlo_dot_dimensions(
            dimension_numbers.lhs_batching_dimensions(),
            dimension_numbers.rhs_batching_dimensions(),
            dimension_numbers.lhs_contracting_dimensions(),
            dimension_numbers.rhs_contracting_dimensions(),
        )?;
        let result = lowerer.block.append_operation(stable_hlo::dot_general(
            input_values[0],
            input_values[1],
            dimensions,
            Some((Precision::Default, Precision::Default)),
            None,
            output_tensor_type,
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
    }
}

/// Builds capped cumulative group starts and ends in `u64`. Signed inputs are clamped to zero before widening, bounded
/// dynamic prefixes are materialized once, and the cumulative reducer cannot overflow before applying the physical
/// ragged-axis bound.
fn lower_ragged_dot_group_intervals<'b, 'c: 'b, 't: 'c>(
    group_sizes: ValueRef<'b, 'c, 't>,
    group_sizes_type: &ArrayType,
    physical_ragged_extent: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(ValueRef<'b, 'c, 't>, ValueRef<'b, 'c, 't>), LoweringError> {
    let group_axis = group_sizes_type.rank() - 1;
    let group_count = group_sizes_type.dimension(group_axis).value().ok_or_else(|| LoweringError::UnsupportedOp {
        op: "`ragged_dot` with a dynamically sized group axis".to_string(),
    })?;
    let group_sizes = if group_sizes_type.data_type().is_signed() {
        // Subtracting the value from itself constructs a shape-identical zero in the source integer type. That avoids
        // unsigned sign extension without introducing a scalar constant that would need explicit broadcasting across
        // static or bounded-dynamic metadata prefixes.
        let zero = block.append_operation(stable_hlo::subtract(group_sizes, group_sizes, location)?)?;
        let zero = zero.result(0).expect("stablehlo.subtract should return one result").as_ref();
        let nonnegative = block.append_operation(stable_hlo::maximum(group_sizes, zero, location)?)?;
        nonnegative.result(0).expect("stablehlo.maximum should return one result").as_ref()
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
        converted.result(0).expect("stablehlo.convert should return one result").as_ref()
    };
    let wide_group_sizes =
        lower_physical_bound_value(wide_group_sizes, &wide_group_sizes_type, 0.0, block, context, location)?;
    let physical_ragged_extent = u64::try_from(physical_ragged_extent).map_err(|_| LoweringError::UnsupportedOp {
        op: "`ragged_dot` whose physical ragged extent cannot be represented as `u64`".to_string(),
    })?;
    if group_count == 0 {
        return Ok((wide_group_sizes, wide_group_sizes));
    }

    let zero = lower_u64_scalar_constant(0, block, context, location)?;
    let extent = lower_u64_scalar_constant(physical_ragged_extent, block, context, location)?;
    // StableHLO may reassociate a reduction, so cap every leaf first to ensure every reducer operand is at most the
    // physical extent and the saturating reducer remains associative.
    let capped_group_sizes = block.append_operation(stable_hlo::clamp(zero, wide_group_sizes, extent, location)?)?;
    let capped_group_sizes = capped_group_sizes.result(0).expect("stablehlo.clamp should return one result").as_ref();
    let scalar_tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::U64), context, location)?;
    let reducer_block = context.block(&[(scalar_tensor_type, location), (scalar_tensor_type, location)]);
    let mut reducer_region = context.region();
    let mut reducer_block = reducer_region.append_block(reducer_block)?;
    let left = reducer_block.argument(0)?.as_ref();
    let right = reducer_block.argument(1)?.as_ref();
    let reducer_extent = lower_u64_scalar_constant(physical_ragged_extent, &mut reducer_block, context, location)?;
    let remaining = reducer_block.append_operation(stable_hlo::subtract(reducer_extent, left, location)?)?;
    let remaining = remaining.result(0).expect("stablehlo.subtract should return one result").as_ref();
    let increment = reducer_block.append_operation(stable_hlo::minimum(right, remaining, location)?)?;
    let increment = increment.result(0).expect("stablehlo.minimum should return one result").as_ref();
    let end = reducer_block.append_operation(stable_hlo::add(left, increment, location)?)?;
    let end = end.result(0).expect("stablehlo.add should return one result").as_ref();
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
    let group_ends = group_ends.result(0).expect("stablehlo.reduce_window should return one result").as_ref();
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
    let group_starts = group_starts.result(0).expect("stablehlo.pad should return one result").as_ref();
    Ok((group_starts, group_ends))
}

/// Expands one grouped-dot operand into a zero-masked dense group axis. Group intervals are accumulated in `u64` and
/// capped at the physical ragged extent so metadata arithmetic and iota indices cannot wrap. The intermediate order
/// keeps the `group_sizes` prefix axes first so every broadcast dimension is monotonic; a final transpose, when
/// necessary, produces `[group, original axes...]` for the dense dot.
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
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let group_axis = group_sizes_type.rank() - 1;
    let group_count = group_sizes_type.dimension(group_axis);

    let remaining_axes = (0..input_type.rank()).filter(|axis| !prefix_axes.contains(axis)).collect::<Vec<_>>();
    let input_permutation = prefix_axes.iter().copied().chain(remaining_axes.iter().copied()).collect::<Vec<_>>();
    let input = if input_permutation.iter().enumerate().all(|(axis, source)| axis == *source) {
        input
    } else {
        let transpose =
            block.append_operation(stable_hlo::transpose(input, input_permutation.as_slice(), location)?)?;
        transpose.result(0).expect("stablehlo.transpose should return one result").as_ref()
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
    let input = input.result(0).expect("stablehlo.broadcast should return one result").as_ref();

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
    let group_starts = group_starts.result(0).expect("stablehlo.broadcast should return one result").as_ref();
    let group_ends = block.append_operation(stable_hlo::broadcast(
        group_ends,
        physical_metadata_tensor_type,
        metadata_broadcast_dimensions.as_slice(),
        location,
    )?)?;
    let group_ends = group_ends.result(0).expect("stablehlo.broadcast should return one result").as_ref();
    let ragged_intermediate_axis =
        prefix_axes.len() + 1 + remaining_axes.iter().position(|axis| *axis == ragged_axis).unwrap();
    let indices =
        block.append_operation(stable_hlo::iota(physical_metadata_tensor_type, ragged_intermediate_axis, location)?)?;
    let indices = indices.result(0).expect("stablehlo.iota should return one result").as_ref();
    let after_start =
        lower_compare_to_mlir(ComparisonDirection::LessThanOrEqual, group_starts, indices, block, location)?;
    let before_end = lower_compare_to_mlir(ComparisonDirection::LessThan, indices, group_ends, block, location)?;
    let mask = block.append_operation(stable_hlo::and(after_start, before_end, location)?)?;
    let mask = mask.result(0).expect("stablehlo.and should return one result").as_ref();
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
        zero.result(0).expect("stablehlo.broadcast should return one result").as_ref()
    } else {
        let false_mask = block.append_operation(stable_hlo::xor(mask, mask, location)?)?;
        let false_mask = false_mask.result(0).expect("stablehlo.xor should return one result").as_ref();
        let zero = block.append_operation(stable_hlo::convert(false_mask, intermediate_tensor_type, location)?)?;
        zero.result(0).expect("stablehlo.convert should return one result").as_ref()
    };
    let masked = block.append_operation(stable_hlo::select(mask, input, zero, location)?)?;
    let masked = masked.result(0).expect("stablehlo.select should return one result").as_ref();

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
        Ok(expanded.result(0).expect("stablehlo.transpose should return one result").as_ref())
    }
}

/// Emits one ordinary dense dot after the grouped operands have been expanded.
fn lower_dense_ragged_dot<'b, 'c: 'b, 't: 'c>(
    lhs: ValueRef<'b, 'c, 't>,
    rhs: ValueRef<'b, 'c, 't>,
    dimensions: &DotDimensionNumbers,
    output_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let dimensions = context.stable_hlo_dot_dimensions(
        dimensions.lhs_batching_dimensions(),
        dimensions.rhs_batching_dimensions(),
        dimensions.lhs_contracting_dimensions(),
        dimensions.rhs_contracting_dimensions(),
    )?;
    let dot = block.append_operation(stable_hlo::dot_general(
        lhs,
        rhs,
        dimensions,
        Some((Precision::Default, Precision::Default)),
        None,
        lower_tensor_type(output_type, context, location)?,
        location,
    )?)?;
    Ok(dot.result(0).expect("stablehlo.dot_general should return one result").as_ref())
}

impl LowerableOperation for RaggedDotOperation {
    // Lower grouped dot either to its portable masked-dense decomposition or to the opt-in CHLO instruction.
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 3, ProgramError);
        check_count!("input", input_types, 3, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let dimensions = self.dimensions();
        if lowerer.collective_state.ragged_dot_lowering_strategy() == RaggedDotLoweringStrategy::Instruction {
            // JAX applies an explicit RHS transpose only for its TPU instruction path as a backend layout workaround.
            // Ryft emits the canonical CHLO contract and leaves target-specific layout selection to legalization.
            let dot = dimensions.dot_dimensions();
            let attribute = lowerer.context.chlo_ragged_dot_dimensions(
                dot.lhs_batching_dimensions(),
                dot.rhs_batching_dimensions(),
                dot.lhs_contracting_dimensions(),
                dot.rhs_contracting_dimensions(),
                dimensions.lhs_ragged_dimensions(),
                dimensions.rhs_group_dimensions(),
            )?;
            let result = lowerer.block.append_operation(chlo::ragged_dot(
                input_values[0],
                input_values[1],
                input_values[2],
                attribute,
                Some((chlo::Precision::Default, chlo::Precision::Default)),
                lower_tensor_type(&output_types[0], lowerer.context, lowerer.location)?,
                lowerer.location,
            )?)?;
            return Ok(vec![result.result(0).expect("chlo.ragged_dot should return one result").as_ref()]);
        }

        let lhs_type = &input_types[0];
        let rhs_type = &input_types[1];
        let group_sizes_type = &input_types[2];
        let mode = dimensions.mode(lhs_type.rank()).map_err(ProgramError::from)?;
        if mode == RaggedDotMode::Batch {
            return Ok(vec![lower_dense_ragged_dot(
                input_values[0],
                input_values[1],
                dimensions.dot_dimensions(),
                &output_types[0],
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?]);
        }
        let lhs_ragged_axis = dimensions.lhs_ragged_dimensions()[0];
        let lhs_prefix = dimensions.group_sizes_prefix_dimensions(lhs_type.rank()).map_err(ProgramError::from)?;
        let physical_ragged_extent = physical_bound_type(lhs_type)?.dimension(lhs_ragged_axis).value().unwrap();
        let (group_starts, group_ends) = lower_ragged_dot_group_intervals(
            input_values[2],
            group_sizes_type,
            physical_ragged_extent,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let lhs = lower_ragged_dot_expansion(
            input_values[0],
            lhs_type,
            group_starts,
            group_ends,
            group_sizes_type,
            lhs_ragged_axis,
            lhs_prefix.as_slice(),
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let shift = |axes: &[usize]| axes.iter().map(|axis| axis + 1).collect::<Vec<_>>();
        let dot = dimensions.dot_dimensions();
        let dense_dimensions = match mode {
            RaggedDotMode::NonContracting => {
                let mut lhs_contracting = shift(dot.lhs_contracting_dimensions());
                lhs_contracting.push(0);
                let mut rhs_contracting = dot.rhs_contracting_dimensions().to_vec();
                rhs_contracting.push(dimensions.rhs_group_dimensions()[0]);
                DotDimensionNumbers::new(
                    lhs_contracting,
                    rhs_contracting,
                    shift(dot.lhs_batching_dimensions()),
                    dot.rhs_batching_dimensions().to_vec(),
                )
            }
            RaggedDotMode::Contracting => {
                let position =
                    dot.lhs_contracting_dimensions().iter().position(|axis| *axis == lhs_ragged_axis).unwrap();
                let rhs_ragged_axis = dot.rhs_contracting_dimensions()[position];
                let rhs_prefix = dot
                    .rhs_batching_dimensions()
                    .iter()
                    .copied()
                    .chain(dot.rhs_contracting_dimensions()[..position].iter().copied())
                    .collect::<Vec<_>>();
                let rhs = lower_ragged_dot_expansion(
                    input_values[1],
                    rhs_type,
                    group_starts,
                    group_ends,
                    group_sizes_type,
                    rhs_ragged_axis,
                    rhs_prefix.as_slice(),
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let mut lhs_batching = vec![0];
                lhs_batching.extend(shift(dot.lhs_batching_dimensions()));
                let mut rhs_batching = vec![0];
                rhs_batching.extend(shift(dot.rhs_batching_dimensions()));
                return Ok(vec![lower_dense_ragged_dot(
                    lhs,
                    rhs,
                    &DotDimensionNumbers::new(
                        shift(dot.lhs_contracting_dimensions()),
                        shift(dot.rhs_contracting_dimensions()),
                        lhs_batching,
                        rhs_batching,
                    ),
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?]);
            }
            RaggedDotMode::Batch => unreachable!(),
        };
        Ok(vec![lower_dense_ragged_dot(
            lhs,
            input_values[1],
            &dense_dimensions,
            &output_types[0],
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{Array, ArrayOperation, Dimension, Placeholder, RaggedDot};

    use crate::experimental::lowering::tests::{
        XlaProgramBuilder, dynamic_dimension, ragged_dot_cpu_compilation_options, unproject_plain_program,
    };
    use crate::experimental::lowering::to_mlir_module_for_plain_program_with_ragged_dot_lowering_strategy;
    use crate::experimental::ops::XlaArrayConstant;
    use crate::tests::values_to_bytes;
    use crate::{ToPjrt, to_mlir_module_for_program};

    use super::*;

    #[cfg(test)]
    use ryft_core::RaggedDotDimensionNumbers;

    // Lowers one grouped-dot signature using `strategy`.
    fn lowered_ragged_dot_module_for(
        dimensions: RaggedDotDimensionNumbers,
        lhs_type: ArrayType,
        rhs_type: ArrayType,
        group_sizes_type: ArrayType,
        strategy: RaggedDotLoweringStrategy,
    ) -> Result<String, LoweringError> {
        let mut builder = XlaProgramBuilder::new();
        let lhs = builder.add_input(lhs_type);
        let rhs = builder.add_input(rhs_type);
        let group_sizes = builder.add_input(group_sizes_type);
        let output = builder
            .add_instruction(
                ArrayOperation::RaggedDot(RaggedDotOperation::new(dimensions)),
                Vec::new(),
                vec![lhs, rhs, group_sizes],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        to_mlir_module_for_plain_program_with_ragged_dot_lowering_strategy(&program, "main", strategy)
    }

    // Lowers the basic non-contracting grouped-matrix form using `strategy`.
    fn lowered_ragged_dot_module(strategy: RaggedDotLoweringStrategy) -> Result<String, LoweringError> {
        lowered_ragged_dot_module_for(
            RaggedDotDimensionNumbers::matmul(),
            ArrayType::new_static(DataType::F32, [5, 2]),
            ArrayType::new_static(DataType::F32, [3, 2, 1]),
            ArrayType::new_static(DataType::I32, [3]),
            strategy,
        )
    }

    #[test]
    fn test_ragged_dot_lowering_strategies_preserve_prefix_shaped_group_sizes() {
        let dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]),
            vec![1],
            vec![0],
        );
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
    fn test_ragged_dot_decomposition_clamps_signed_group_sizes_with_dynamic_prefix() {
        let batch = dynamic_dimension("batch", Some(4));
        let dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]),
            vec![1],
            vec![0],
        );
        let module = lowered_ragged_dot_module_for(
            dimensions,
            ArrayType::new(DataType::F32, Shape::new(vec![batch.clone(), 3.into(), 1.into()])),
            ArrayType::new(DataType::F32, Shape::new(vec![2.into(), batch.clone(), 1.into(), 1.into()])),
            ArrayType::new(DataType::I32, Shape::new(vec![batch, 2.into()])),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();

        assert!(module.contains("stablehlo.subtract %arg2, %arg2"), "{module}");
        assert!(module.contains("stablehlo.maximum %arg2"), "{module}");
        assert!(module.contains("stablehlo.clamp"), "{module}");
    }

    #[test]
    fn test_ragged_dot_decomposition_broadcasts_shared_group_sizes_across_dynamic_prefix() {
        let batch = dynamic_dimension("batch", Some(4));
        let dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]),
            vec![1],
            vec![0],
        );
        let module = lowered_ragged_dot_module_for(
            dimensions,
            ArrayType::new(DataType::F32, Shape::new(vec![batch.clone(), 3.into(), 1.into()])),
            ArrayType::new(DataType::F32, Shape::new(vec![2.into(), batch, 1.into(), 1.into()])),
            ArrayType::new_static(DataType::I32, [2]),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();

        assert!(!module.contains("stablehlo.dynamic_broadcast_in_dim"), "{module}");
    }

    #[test]
    fn test_ragged_dot_decomposition_executes_prefix_below_dynamic_bound() {
        use std::sync::Arc;

        use ryft_pjrt::{
            BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
            load_cpu_plugin,
        };

        use crate::tests::values_from_bytes;

        let batch = dynamic_dimension("batch", Some(4));
        let dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]),
            vec![1],
            vec![0],
        );
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

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        // The physical buffers deliberately carry garbage (not zeros) in every slot beyond the live prefix of the
        // bounded-dynamic batch axis. Each batch position computes independently, so the asserted live prefix can
        // only stay correct if the lowering itself confines the padded positions, rather than relying on the caller
        // zeroing them.
        let inputs = [
            client
                .buffer(
                    values_to_bytes(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).as_slice(),
                    BufferType::F32,
                    &[3, 3, 1],
                    None,
                    device.clone(),
                    None,
                )
                .unwrap(),
            client
                .buffer(
                    values_to_bytes(&[10.0_f32, 2.0, -9.0, 20.0, 3.0, -11.0]).as_slice(),
                    BufferType::F32,
                    &[2, 3, 1, 1],
                    None,
                    device.clone(),
                    None,
                )
                .unwrap(),
            client
                .buffer(
                    values_to_bytes(&[1_i32, 2, 2, 1, 5, 7]).as_slice(),
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
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        let output = executions.remove(0).outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        let output = values_from_bytes::<f32>(output.as_slice());
        assert_eq!(&output[..6], &[10.0, 40.0, 60.0, 8.0, 10.0, 18.0]);
    }

    #[test]
    fn test_ragged_dot_decomposition_widens_narrow_group_metadata_and_iota() {
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
    fn test_ragged_dot_lowering_strategies() {
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

        let contracting_dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
            vec![1],
            Vec::new(),
        );
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

        let batch_dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
            vec![0],
            Vec::new(),
        );
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

    // Compiles and executes one grouped-matrix decomposition on the CPU plugin.
    fn execute_ragged_dot_decomposition_on_cpu(
        dimensions: RaggedDotDimensionNumbers,
        lhs_type: ArrayType,
        rhs_type: ArrayType,
        group_sizes_type: ArrayType,
        lhs_values: &[f32],
        rhs_values: &[f32],
        group_sizes_data: &[u8],
    ) -> Vec<f32> {
        use std::sync::Arc;

        use ryft_pjrt::{
            BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
            load_cpu_plugin,
        };

        use crate::tests::values_from_bytes;

        let module = lowered_ragged_dot_module_for(
            dimensions,
            lhs_type.clone(),
            rhs_type.clone(),
            group_sizes_type.clone(),
            RaggedDotLoweringStrategy::Decomposition,
        )
        .unwrap();
        let lhs_shape = lhs_type.static_shape().unwrap();
        let lhs_dimensions = lhs_shape.dimensions().iter().map(|dimension| *dimension as u64).collect::<Vec<_>>();
        let rhs_shape = rhs_type.static_shape().unwrap();
        let rhs_dimensions = rhs_shape.dimensions().iter().map(|dimension| *dimension as u64).collect::<Vec<_>>();
        let group_sizes_shape = group_sizes_type.static_shape().unwrap();
        let group_sizes_dimensions =
            group_sizes_shape.dimensions().iter().map(|dimension| *dimension as u64).collect::<Vec<_>>();
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        let lhs = client
            .buffer(
                values_to_bytes(lhs_values).as_slice(),
                BufferType::F32,
                lhs_dimensions.as_slice(),
                None,
                device.clone(),
                None,
            )
            .unwrap();
        let rhs = client
            .buffer(
                values_to_bytes(rhs_values).as_slice(),
                BufferType::F32,
                rhs_dimensions.as_slice(),
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
        let inputs = [lhs, rhs, group_sizes]
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
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        let output = executions.remove(0).outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        values_from_bytes::<f32>(output.as_slice())
    }

    #[test]
    fn test_ragged_dot_decomposition_executes_on_cpu() {
        let lhs_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let rhs_values = [10.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let group_sizes_values = [1_i32, 0, 3];
        let lhs = Array::matrix(5, 2, lhs_values.to_vec()).unwrap();
        let rhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [3, 2, 1]), &rhs_values).unwrap();
        let eager = lhs.ragged_dot(&rhs, &Array::vector(group_sizes_values.to_vec()).unwrap()).unwrap();
        let expected = eager.to_f64s().into_iter().map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(expected, vec![12.0, 32.0, 50.0, 68.0, 0.0]);
        let group_sizes_data = values_to_bytes(&group_sizes_values);
        assert_eq!(
            execute_ragged_dot_decomposition_on_cpu(
                RaggedDotDimensionNumbers::matmul(),
                ArrayType::new_static(DataType::F32, [5, 2]),
                ArrayType::new_static(DataType::F32, [3, 2, 1]),
                ArrayType::new_static(DataType::I32, [3]),
                &lhs_values,
                &rhs_values,
                group_sizes_data.as_slice(),
            ),
            expected,
        );
    }

    #[test]
    fn test_contracting_ragged_dot_decomposition_executes_on_cpu() {
        let dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
            vec![1],
            Vec::new(),
        );
        let lhs_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let rhs_values = [10.0_f32, 20.0, 30.0, 40.0, 50.0];
        let group_sizes_values = [2_i32, 0, 3];
        let lhs = Array::matrix(2, 5, lhs_values.to_vec()).unwrap();
        let rhs = Array::matrix(5, 1, rhs_values.to_vec()).unwrap();
        let eager = lhs
            .ragged_dot_general(&rhs, &Array::vector(group_sizes_values.to_vec()).unwrap(), &dimensions)
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
                &lhs_values,
                &rhs_values,
                group_sizes_data.as_slice(),
            ),
            expected,
        );
    }

    #[test]
    fn test_ragged_dot_decomposition_saturates_overflowing_group_intervals_on_cpu() {
        let lhs_values = [1.0_f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let rhs_values = [2.0_f32, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0];
        let group_sizes_values = [u64::MAX, u64::MAX, 2, 1];
        let lhs = Array::matrix(4, 2, lhs_values.to_vec()).unwrap();
        let rhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [4, 2, 1]), &rhs_values).unwrap();
        let eager = lhs.ragged_dot(&rhs, &Array::vector(group_sizes_values.to_vec()).unwrap()).unwrap();
        let expected = eager.to_f64s().into_iter().map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(expected, vec![2.0, 4.0, 6.0, 8.0]);
        let group_sizes_data = values_to_bytes(&group_sizes_values);
        assert_eq!(
            execute_ragged_dot_decomposition_on_cpu(
                RaggedDotDimensionNumbers::matmul(),
                ArrayType::new_static(DataType::F32, [4, 2]),
                ArrayType::new_static(DataType::F32, [4, 2, 1]),
                ArrayType::new_static(DataType::U64, [4]),
                &lhs_values,
                &rhs_values,
                group_sizes_data.as_slice(),
            ),
            expected,
        );
    }

    #[test]
    fn test_ragged_dot_decomposition_clamps_negative_group_sizes_before_widening_on_cpu() {
        let lhs_values = [1.0_f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let rhs_values = [2.0_f32, 0.0, 3.0, 0.0, 5.0, 0.0];
        let group_sizes_values = [1_i32, -1, 2];
        let group_sizes_data = values_to_bytes(&group_sizes_values);
        assert_eq!(
            execute_ragged_dot_decomposition_on_cpu(
                RaggedDotDimensionNumbers::matmul(),
                ArrayType::new_static(DataType::F32, [4, 2]),
                ArrayType::new_static(DataType::F32, [3, 2, 1]),
                ArrayType::new_static(DataType::I32, [3]),
                &lhs_values,
                &rhs_values,
                group_sizes_data.as_slice(),
            ),
            vec![2.0, 10.0, 15.0, 0.0],
        );
    }

    #[test]
    fn test_ragged_dot_instruction_is_explicitly_unsupported_on_cpu() {
        use ryft_pjrt::{ClientOptions, CpuClientOptions, Error, Program as PjrtProgram, load_cpu_plugin};

        let module = lowered_ragged_dot_module(RaggedDotLoweringStrategy::Instruction).unwrap();
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        assert!(matches!(
            client.compile(
                &PjrtProgram::Mlir { bytecode: module.into_bytes() },
                &ragged_dot_cpu_compilation_options(),
            ),
            Err(Error::Unimplemented { message, .. })
                if message == "HLO opcode `ragged-dot` is not supported by XLA:CPU ThunkEmitter",
        ));
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_accumulation_typed_dot() {
        use ryft_core::{DotDimensionNumbers, DotOperation};

        // An accumulation-typed dot lowers to a `stablehlo.dot_general` whose result type is the accumulation type
        // (XLA's `preferred_element_type` contract), with the operands kept at their narrow element type.
        let operand_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let lhs = builder.add_input(operand_type.clone());
        let rhs = builder.add_input(operand_type.clone());
        let operation = DotOperation::new(DotDimensionNumbers::matmul()).with_accumulation_type(DataType::F32);
        let output = builder.add_instruction(operation, Vec::new(), vec![lhs, rhs], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
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
}
