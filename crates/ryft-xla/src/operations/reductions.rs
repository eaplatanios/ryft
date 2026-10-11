use ryft_core::macros::check_count;
use ryft_core::{
    ArgMaxOperation, ArgMinOperation, ArrayType, ComparisonDirection, DataType, REDUCE_OPERATION_NAME, ReduceOperation,
    ReductionKind,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::Accuracy;
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef, LocationRef,
    Operation as MlirOperation, Region, Size as MlirSize, TensorTypeRef, Type, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::{
    lower_constant_elements_attribute, lower_f64_constant_splat, lower_f64_scalar_elements_attribute,
    lower_restore_dynamic_dimensions, lower_signed_one_bit_to_mlir, lower_tensor_type, lower_unplaced_constant_output,
    physical_bound_type, signed_integer_width, unsigned_integer_width,
};
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::operations::extrema::lower_extremum_to_mlir;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Lowers an [`ArgMaxOperation`] (when `maximize` is `true`) or an [`ArgMinOperation`] along `axis` to a variadic
/// `stablehlo.reduce` over the input and a `stablehlo.iota` of its indices along `axis`, returning the reduced indices.
/// The reduction starts from the extremum identity of the input data type (i.e., the lowest value for `argmax` and the
/// highest one for `argmin`) paired with index zero, and its comparator (refer to
/// [`build_index_reduction_body_region`]) is commutative and associative, so that the result does not depend on the
/// order in which XLA combines elements. Narrow floating-point inputs compare in `f32`, into which they convert exactly
/// (including their NaNs and signed zeros), and signed one-bit inputs and indices use a signed byte carrier, because
/// StableHLO treats `i1` as a predicate.
///
/// Bounded-dynamic inputs are supported as well: XLA's bounded-dynamic legalizer pads every input of a variadic
/// `stablehlo.reduce` along a dynamic reduced axis with that input's own initial value, so padded positions hold the
/// extremum identity paired with index zero. Such a pair never strictly beats a live element, and it ties with live
/// elements only when every live element equals the identity, in which case index zero is the correct result. Dynamic
/// axes that are not reduced pass through the reduction unchanged.
pub(crate) fn lower_index_reduction_to_mlir<'b, 'c: 'b, 't: 'c>(
    maximize: bool,
    axis: usize,
    input_value: ValueRef<'b, 'c, 't>,
    input_type: &ArrayType,
    output_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let (value, value_data_type) = match input_type.data_type() {
        DataType::I1 => (lower_signed_one_bit_to_mlir(input_value, block, context, location)?, DataType::I8),
        data_type if data_type.is_floating_point() && !matches!(data_type, DataType::F32 | DataType::F64) => {
            let working_type =
                lower_tensor_type(&ArrayType::new(DataType::F32, input_type.shape().clone()), context, location)?;
            let converted = block.append_operation(stable_hlo::convert(input_value, working_type, location)?)?;
            (converted.result(0).expect("stablehlo.convert should return one result").as_ref(), DataType::F32)
        }
        data_type => (input_value, data_type),
    };
    let index_data_type = match output_type.data_type() {
        DataType::I1 => DataType::I8,
        data_type => data_type,
    };

    // `stablehlo.reduce` requires its inputs to share one shape, so the index iota takes the bounded-dynamic type of
    // the input: it is built at the physical bound of every dynamic axis and then receives the runtime extents of the
    // input. Statically shaped inputs need no refinement.
    let index_type = ArrayType::new(index_data_type, input_type.shape().clone());
    let physical_index_tensor_type = lower_tensor_type(&physical_bound_type(&index_type)?, context, location)?;
    let indices = block.append_operation(stable_hlo::iota(physical_index_tensor_type, axis, location)?)?;
    let indices = indices.result(0).expect("stablehlo.iota should return one result").as_ref();
    let extent_sources = (0..index_type.rank()).map(|axis| (input_value, axis)).collect::<Vec<_>>();
    let indices = lower_restore_dynamic_dimensions(indices, &index_type, &extent_sources, block, context, location)?;
    let identity_kind = if maximize { ReductionKind::Max } else { ReductionKind::Min };
    let value_identity = build_reduction_identity_constant(identity_kind, value_data_type, block, context, location)?;
    let scalar_index_type = ArrayType::scalar(index_data_type);
    let scalar_index_tensor_type = lower_tensor_type(&scalar_index_type, context, location)?;
    let index_identity =
        lower_f64_constant_splat(0.0, &scalar_index_type, scalar_index_tensor_type, block, context, location)?;
    let body = build_index_reduction_body_region(maximize, value_data_type, index_data_type, context, location)?;
    let reduced = block.append_operation(stable_hlo::reduce(
        &[value, indices],
        &[value_identity, index_identity],
        &[axis],
        body,
        location,
    )?)?;
    let reduced_indices = reduced.result(1).expect("stablehlo.reduce should return one result per input").as_ref();
    if index_data_type == output_type.data_type() {
        return Ok(reduced_indices);
    }
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    let converted = block.append_operation(stable_hlo::convert(reduced_indices, output_tensor_type, location)?)?;
    Ok(converted.result(0).expect("stablehlo.convert should return one result").as_ref())
}

/// Builds the comparator region of [`lower_index_reduction_to_mlir`], which combines two `(value, index)` pairs by
/// picking the left pair when its value is strictly better than the right one, or equivalent to it with a lower index.
/// For `argmax` (when `maximize` is `true`), a value is strictly better when it is greater, and for `argmin` when it is
/// smaller, comparing floating-point values with `FLOAT` semantics (so that `-0.0` and `+0.0` are equivalent).
/// A floating-point NaN is also strictly better than every ordered value for both directions, and two NaNs are
/// equivalent. Unlike JAX's comparator, which picks the left pair whenever its value is a NaN, this one is commutative
/// and associative, so every combination order selects the first NaN and the lowest index among ties.
fn build_index_reduction_body_region<'c, 't>(
    maximize: bool,
    value_data_type: DataType,
    index_data_type: DataType,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ryft_mlir::DetachedRegion<'c, 't>, LoweringError> {
    let value_tensor_type = lower_tensor_type(&ArrayType::scalar(value_data_type), context, location)?;
    let index_tensor_type = lower_tensor_type(&ArrayType::scalar(index_data_type), context, location)?;
    let block = context.block(&[
        (value_tensor_type, location),
        (index_tensor_type, location),
        (value_tensor_type, location),
        (index_tensor_type, location),
    ]);
    let mut region = context.region();
    let mut block_ref = region.append_block(block)?;
    let left_value = block_ref.argument(0)?.as_ref();
    let left_index = block_ref.argument(1)?.as_ref();
    let right_value = block_ref.argument(2)?.as_ref();
    let right_index = block_ref.argument(3)?.as_ref();
    let direction = if maximize { ComparisonDirection::GreaterThan } else { ComparisonDirection::LessThan };
    let mut better = lower_compare_to_mlir(direction, left_value, right_value, &mut block_ref, location)?;
    let mut equivalent =
        lower_compare_to_mlir(ComparisonDirection::Equal, left_value, right_value, &mut block_ref, location)?;
    if value_data_type.is_floating_point() {
        let left_is_nan =
            lower_compare_to_mlir(ComparisonDirection::NotEqual, left_value, left_value, &mut block_ref, location)?;
        let right_is_nan =
            lower_compare_to_mlir(ComparisonDirection::NotEqual, right_value, right_value, &mut block_ref, location)?;
        let right_is_ordered =
            lower_compare_to_mlir(ComparisonDirection::Equal, right_value, right_value, &mut block_ref, location)?;
        let nan_beats_ordered =
            block_ref.append_operation(stable_hlo::and(left_is_nan, right_is_ordered, location)?)?;
        let nan_beats_ordered = nan_beats_ordered.result(0).expect("stablehlo.and should return one result").as_ref();
        let both_nan = block_ref.append_operation(stable_hlo::and(left_is_nan, right_is_nan, location)?)?;
        let both_nan = both_nan.result(0).expect("stablehlo.and should return one result").as_ref();
        better = block_ref
            .append_operation(stable_hlo::or(better, nan_beats_ordered, location)?)?
            .result(0)
            .expect("stablehlo.or should return one result")
            .as_ref();
        equivalent = block_ref
            .append_operation(stable_hlo::or(equivalent, both_nan, location)?)?
            .result(0)
            .expect("stablehlo.or should return one result")
            .as_ref();
    }
    let lower_index =
        lower_compare_to_mlir(ComparisonDirection::LessThan, left_index, right_index, &mut block_ref, location)?;
    let tie = block_ref.append_operation(stable_hlo::and(equivalent, lower_index, location)?)?;
    let tie = tie.result(0).expect("stablehlo.and should return one result").as_ref();
    let pick_left = block_ref.append_operation(stable_hlo::or(better, tie, location)?)?;
    let pick_left = pick_left.result(0).expect("stablehlo.or should return one result").as_ref();
    let value = block_ref.append_operation(stable_hlo::select(pick_left, left_value, right_value, location)?)?;
    let value = value.result(0).expect("stablehlo.select should return one result").as_ref();
    let index = block_ref.append_operation(stable_hlo::select(pick_left, left_index, right_index, location)?)?;
    let index = index.result(0).expect("stablehlo.select should return one result").as_ref();
    block_ref.append_operation(stable_hlo::r#return(&[value, index], location)?)?;
    Ok(region)
}

/// Lowers the scalar product combiner of a `reduce` or `cumulative` body over `element_type`. Complex operands return
/// the other operand when one of them is exactly the `1 + 0i` identity that seeds every reduction and pads every
/// window, because multiplying by it is not exact under complex arithmetic: its zero imaginary component meets the
/// other operand's infinite components as `0 · ∞ = NaN`. This is the same rule that `ryft-core`'s reference kernels
/// apply to every product (including inputs whose elements are themselves exactly `1 + 0i`), so the two backends agree.
/// Real products multiply directly.
pub(crate) fn lower_product_combiner_to_mlir<'b, 'c: 'b, 't: 'c>(
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    element_type: DataType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let product = block.append_operation(stable_hlo::multiply(left, right, location)?)?.result(0).unwrap().as_ref();
    if !element_type.is_complex() {
        return Ok(product);
    }
    let one = lower_unplaced_constant_output(&[ArrayType::scalar(element_type)], 1, block, context, location)?[0];
    let mut identity_masks = Vec::with_capacity(2);
    for operand in [left, right] {
        let mask = block.append_operation(stable_hlo::compare(
            operand,
            one,
            stable_hlo::ComparisonDirection::Equal,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?;
        identity_masks.push(mask.result(0).unwrap().as_ref());
    }
    let guarded = block
        .append_operation(stable_hlo::select(identity_masks[1], left, product, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    Ok(block
        .append_operation(stable_hlo::select(identity_masks[0], right, guarded, location)?)?
        .result(0)
        .unwrap()
        .as_ref())
}

/// Builds a reduction-body region for [`stable_hlo::reduce`] over the given scalar `element_type`. The generated
/// region has one block taking two scalar tensor arguments of `tensor<{element_type}>` and produces one scalar result
/// through the combiner matching the reduction kind.
pub(crate) fn build_reduce_body_region<'c, 't>(
    kind: ReductionKind,
    element_type: DataType,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ryft_mlir::DetachedRegion<'c, 't>, LoweringError> {
    let scalar_array_type = ArrayType::scalar(element_type);
    let scalar_tensor_type = lower_tensor_type(&scalar_array_type, context, location)?;
    let block = context.block(&[(scalar_tensor_type, location), (scalar_tensor_type, location)]);
    let mut region = context.region();
    let mut block_ref = region.append_block(block)?;
    let lhs = block_ref.argument(0)?.as_ref();
    let rhs = block_ref.argument(1)?.as_ref();
    let body_value = match kind {
        ReductionKind::LogSumExp => {
            return Err(LoweringError::UnsupportedOp {
                op: format!(
                    "`{REDUCE_OPERATION_NAME}` with kind `{}` requires a maximum-shifted expansion instead of a scalar \
                     combiner",
                    ReductionKind::LogSumExp,
                ),
            });
        }
        ReductionKind::Sum | ReductionKind::Mean => block_ref
            .append_operation(stable_hlo::add(lhs, rhs, location)?)?
            .result(0)
            .expect("stablehlo.add should return one result")
            .as_ref(),
        ReductionKind::Product => {
            lower_product_combiner_to_mlir(lhs, rhs, element_type, &mut block_ref, context, location)?
        }
        ReductionKind::Max | ReductionKind::Min => {
            lower_extremum_to_mlir(kind == ReductionKind::Max, element_type, lhs, rhs, &mut block_ref, location)?
        }
        ReductionKind::Any => block_ref
            .append_operation(stable_hlo::or(lhs, rhs, location)?)?
            .result(0)
            .expect("stablehlo.or should return one result")
            .as_ref(),
        ReductionKind::All => block_ref
            .append_operation(stable_hlo::and(lhs, rhs, location)?)?
            .result(0)
            .expect("stablehlo.and should return one result")
            .as_ref(),
    };
    block_ref.append_operation(stable_hlo::r#return(&[body_value], location)?)?;
    Ok(region)
}

/// Lowers an [`ArrayOperation::Reduce`](ryft_core::ArrayOperation::Reduce) dispatch using the reduction kind's scalar
/// combiner and identity. [`ReductionKind::Mean`] divides a sum by the reduced element count, while
/// [`ReductionKind::LogSumExp`] uses the guarded maximum-shifted expansion in [`lower_log_sum_exp_to_mlir`].
pub(crate) fn lower_reduce_to_mlir<'b, 'c: 'b, 't: 'c>(
    kind: ReductionKind,
    axes: &[usize],
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let element_type = output_array_type.data_type();
    if matches!(kind, ReductionKind::Sum | ReductionKind::Product | ReductionKind::Mean | ReductionKind::LogSumExp)
        && element_type.is_floating_point()
        && !matches!(element_type, DataType::F32 | DataType::F64)
        && (element_type != DataType::F8E8M0FNU || kind == ReductionKind::Product)
        && (kind != ReductionKind::LogSumExp
            || matches!(
                element_type,
                DataType::BF16 | DataType::F16 | DataType::F8E3M4 | DataType::F8E4M3 | DataType::F8E5M2,
            ))
    {
        // Keep both accumulation and mean normalization in `f32`; rounding the sum or count first can overflow.
        let input_type = input_value.r#type()?.cast::<TensorTypeRef>().unwrap();
        let working_input_type = context.tensor_type(
            context.float32_type(),
            &input_type.dimensions().collect::<Vec<_>>(),
            None,
            location,
        )?;
        let input = block
            .append_operation(stable_hlo::convert(input_value, working_input_type, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        let output = lower_reduce_to_mlir(
            kind,
            axes,
            input,
            &output_array_type.clone().with_data_type(DataType::F32),
            block,
            context,
            location,
        )?;
        return Ok(block
            .append_operation(stable_hlo::convert(
                output,
                lower_tensor_type(output_array_type, context, location)?,
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref());
    }
    if kind == ReductionKind::LogSumExp {
        return lower_log_sum_exp_to_mlir(axes, input_value, output_array_type, block, context, location);
    }
    let initial_value = build_reduction_identity_constant(kind, element_type, block, context, location)?;
    let body_region = build_reduce_body_region(kind, element_type, context, location)?;
    let reduce_op = stable_hlo::reduce(&[input_value], &[initial_value], axes, body_region, location)?;
    let result = block.append_operation(reduce_op)?;
    let reduced = result.result(0).expect("stablehlo.reduce should return one result").as_ref();
    if !matches!(kind, ReductionKind::Mean) {
        return Ok(reduced);
    }
    // Floating-point empty means use `0 / 0`, producing NaN. Integer means retain the zero identity by clamping
    // the count, since integer division has no NaN representation.
    let input_type = input_value.r#type()?;
    let input_tensor_type = input_type.cast::<TensorTypeRef>().ok_or_else(|| LoweringError::UnsupportedOp {
        op: format!("`reduce` operand has non-tensor MLIR type `{input_type}`"),
    })?;
    let dimensions = input_tensor_type.dimensions().collect::<Vec<_>>();
    let mut count = 1usize;
    for axis in axes {
        match dimensions.get(*axis) {
            Some(MlirSize::Static(size)) => count *= size,
            _ => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("`reduce` mean over dynamically sized axis {axis}"),
                });
            }
        }
    }
    let output_tensor_type = lower_tensor_type(output_array_type, context, location)?;
    let count = if element_type.is_integer() { count.max(1) } else { count };
    let divisor = if element_type.is_complex() {
        let part_type = output_array_type.clone().with_data_type(if element_type == DataType::C64 {
            DataType::F32
        } else {
            DataType::F64
        });
        let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
        let real = lower_f64_constant_splat(count as f64, &part_type, part_tensor_type, block, context, location)?;
        let imaginary = lower_f64_constant_splat(0.0, &part_type, part_tensor_type, block, context, location)?;
        block.append_operation(stable_hlo::complex(real, imaginary, location)?)?.result(0).unwrap().as_ref()
    } else {
        lower_f64_constant_splat(count as f64, output_array_type, output_tensor_type, block, context, location)?
    };
    let mean = block.append_operation(stable_hlo::divide(reduced, divisor, location)?)?;
    Ok(mean.result(0).expect("stablehlo.divide should return one result").as_ref())
}

/// Lowers [`ReductionKind::LogSumExp`] using a guarded maximum-shifted construction, since StableHLO has no
/// `logsumexp` primitive:
///
/// ```text
/// m      = reduce_max(real(x))                // with a -∞ initial value
/// safe_m = select(is_finite(m), m, 0)
/// result = log(reduce_sum(exp(x - broadcast(safe_m)))) + safe_m
/// ```
///
/// Shifting by the maximum is what keeps the exponentials in range, and the `safe_m` substitution is what keeps the
/// shift itself defined: a raw maximum of `-∞` (the maximum's identity, and therefore the result for an all-`-∞`
/// slice or an empty reduction) would compute `-∞ - -∞ = NaN`, while substituting zero there leaves `log(0) + 0`,
/// which is the correct `-∞`. The `+∞` maximum is guarded the same way, and NaN propagates through the sum. Complex
/// operands take the maximum of their real components, which alone determine the magnitudes of their exponentials,
/// and shift by it as a complex value with a zero imaginary component. The shift is broadcast back over the reduced
/// axes with the identity map from the output's axes to the operand's kept axes, so both reductions see the operand's
/// own shape.
fn lower_log_sum_exp_to_mlir<'b, 'c: 'b, 't: 'c>(
    axes: &[usize],
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let element_type = output_array_type.data_type();
    if !matches!(
        element_type,
        DataType::BF16
            | DataType::F16
            | DataType::F32
            | DataType::F64
            | DataType::F8E3M4
            | DataType::F8E4M3
            | DataType::F8E5M2
            | DataType::C64
            | DataType::C128
    ) {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "`{REDUCE_OPERATION_NAME}` with kind `{}` requires a format with negative infinity but got \
                 `{element_type}`",
                ReductionKind::LogSumExp,
            ),
        });
    }

    let input_type = input_value.r#type()?;
    let input_tensor_type = input_type.cast::<TensorTypeRef>().ok_or_else(|| LoweringError::UnsupportedOp {
        op: format!(
            "`{REDUCE_OPERATION_NAME}` with kind `{}` input has non-tensor MLIR type `{input_type}`",
            ReductionKind::LogSumExp,
        ),
    })?;
    let (shift_element_type, maximum_input) = match element_type {
        DataType::C64 | DataType::C128 => (
            if element_type == DataType::C64 { DataType::F32 } else { DataType::F64 },
            block.append_operation(stable_hlo::real(input_value, location)?)?.result(0).unwrap().as_ref(),
        ),
        _ => (element_type, input_value),
    };
    let maximum_initial_value =
        build_reduction_identity_constant(ReductionKind::Max, shift_element_type, block, context, location)?;
    let maximum_body_region = build_reduce_body_region(ReductionKind::Max, shift_element_type, context, location)?;
    let maximum = block.append_operation(stable_hlo::reduce(
        &[maximum_input],
        &[maximum_initial_value],
        axes,
        maximum_body_region,
        location,
    )?)?;
    let maximum = maximum.result(0).expect("stablehlo.reduce should return one result").as_ref();
    let finite = block.append_operation(stable_hlo::is_finite(maximum, location)?)?;
    let finite = finite.result(0).expect("stablehlo.is_finite should return one result").as_ref();
    let shift_type = output_array_type.clone().with_data_type(shift_element_type);
    let shift_tensor_type = lower_tensor_type(&shift_type, context, location)?;
    let zero = lower_f64_constant_splat(0.0, &shift_type, shift_tensor_type, block, context, location)?;
    let shift = block.append_operation(stable_hlo::select(finite, maximum, zero, location)?)?;
    let shift = shift.result(0).expect("stablehlo.select should return one result").as_ref();
    let shift = match element_type.is_complex() {
        true => block.append_operation(stable_hlo::complex(shift, zero, location)?)?.result(0).unwrap().as_ref(),
        false => shift,
    };
    let kept_axes = (0..input_tensor_type.rank()).filter(|axis| !axes.contains(axis)).collect::<Vec<_>>();
    let broadcast_shift =
        block.append_operation(stable_hlo::broadcast(shift, input_tensor_type, kept_axes.as_slice(), location)?)?;
    let broadcast_shift = broadcast_shift.result(0).expect("stablehlo.broadcast should return one result").as_ref();
    let shifted = block.append_operation(stable_hlo::subtract(input_value, broadcast_shift, location)?)?;
    let shifted = shifted.result(0).expect("stablehlo.subtract should return one result").as_ref();
    let exponentials = block.append_operation(stable_hlo::exponential(shifted, Accuracy::Default, location)?)?;
    let exponentials = exponentials.result(0).expect("stablehlo.exponential should return one result").as_ref();
    let sum_initial_value =
        build_reduction_identity_constant(ReductionKind::Sum, element_type, block, context, location)?;
    let sum_body_region = build_reduce_body_region(ReductionKind::Sum, element_type, context, location)?;
    let total = block.append_operation(stable_hlo::reduce(
        &[exponentials],
        &[sum_initial_value],
        axes,
        sum_body_region,
        location,
    )?)?;
    let total = total.result(0).expect("stablehlo.reduce should return one result").as_ref();
    let logarithm = block.append_operation(stable_hlo::log(total, Accuracy::Default, location)?)?;
    let logarithm = logarithm.result(0).expect("stablehlo.log should return one result").as_ref();
    let result = block.append_operation(stable_hlo::add(logarithm, shift, location)?)?;
    Ok(result.result(0).expect("stablehlo.add should return one result").as_ref())
}

/// Builds a scalar constant equal to the identity element for the given reduction kind, returned
/// as an MLIR `tensor<{element_type}>` value. Used as the `initial_values` argument of
/// `stablehlo.reduce`.
pub(crate) fn build_reduction_identity_constant<'b, 'c: 'b, 't: 'c>(
    kind: ReductionKind,
    element_type: DataType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let scalar_array_type = ArrayType::scalar(element_type);
    let scalar_tensor_type = lower_tensor_type(&scalar_array_type, context, location)?;
    if element_type.is_complex() {
        let real = match kind {
            ReductionKind::Sum | ReductionKind::Mean => 0.0,
            ReductionKind::Product => 1.0,
            ReductionKind::Max => f64::NEG_INFINITY,
            ReductionKind::Min => f64::INFINITY,
            ReductionKind::LogSumExp | ReductionKind::Any | ReductionKind::All => {
                return Err(LoweringError::UnsupportedDataType { data_type: element_type });
            }
        };
        let part_data_type = if element_type == DataType::C64 { DataType::F32 } else { DataType::F64 };
        let part_type = ArrayType::scalar(part_data_type);
        let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
        let real_value = lower_f64_constant_splat(real, &part_type, part_tensor_type, block, context, location)?;
        // Products use `1 + 0i`; lexicographic extrema need both components at the same bound.
        let imaginary_value = if kind == ReductionKind::Product {
            lower_f64_constant_splat(0.0, &part_type, part_tensor_type, block, context, location)?
        } else {
            real_value
        };
        let complex = block.append_operation(stable_hlo::complex(real_value, imaginary_value, location)?)?;
        return Ok(complex.result(0).expect("stablehlo.complex should return one result").as_ref());
    }
    let attribute = build_reduction_identity_attribute(kind, element_type, scalar_tensor_type, context)?;
    let result = block.append_operation(stable_hlo::constant(attribute, location)?)?;
    Ok(result.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Builds a dense-elements attribute holding the identity element of the given reduction kind at the given element
/// type. `Sum` and `Mean` use zero, `Product` uses one, and `Max` and `Min` use the bounds returned by
/// [`float_reduction_identity_bounds`] at float element types and the bounds returned by
/// [`integer_reduction_identity_bounds`] at integer element types. Boolean `Any`/`Max` use `false`, while Boolean
/// `All`/`Min` use `true`. Every other combination fails with [`LoweringError::UnsupportedDataType`].
fn build_reduction_identity_attribute<'c, 't>(
    kind: ReductionKind,
    element_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    if let Some((minimum, maximum)) = float_reduction_identity_bounds(element_type) {
        let identity = match kind {
            ReductionKind::Sum | ReductionKind::Mean if element_type != DataType::F8E8M0FNU => 0.0,
            ReductionKind::Product => 1.0,
            ReductionKind::Max => minimum,
            ReductionKind::Min => maximum,
            ReductionKind::Sum | ReductionKind::Mean => {
                return Err(LoweringError::UnsupportedDataType { data_type: element_type });
            }
            ReductionKind::LogSumExp | ReductionKind::Any | ReductionKind::All => {
                return Err(LoweringError::UnsupportedDataType { data_type: element_type });
            }
        };
        return lower_f64_scalar_elements_attribute(element_type, tensor_type, identity, context);
    }
    if let Some((minimum, maximum)) = integer_reduction_identity_bounds(element_type) {
        let identity = match kind {
            ReductionKind::Sum | ReductionKind::Mean => 0,
            ReductionKind::Product => 1,
            ReductionKind::Max => minimum,
            ReductionKind::Min => maximum,
            ReductionKind::LogSumExp | ReductionKind::Any | ReductionKind::All => {
                return Err(LoweringError::UnsupportedDataType { data_type: element_type });
            }
        };
        return lower_constant_elements_attribute(element_type, tensor_type, identity, context);
    }
    match (kind, element_type) {
        (ReductionKind::Any | ReductionKind::Max, DataType::Boolean) => context
            .dense_bool_elements_attribute(tensor_type, &[false])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        (ReductionKind::All | ReductionKind::Min, DataType::Boolean) => context
            .dense_bool_elements_attribute(tensor_type, &[true])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        _ => Err(LoweringError::UnsupportedDataType { data_type: element_type }),
    }
}

/// Returns the `(maximum identity, minimum identity)` pair for the given floating-point data type. Formats with
/// infinities use `(-inf, +inf)`, finite signed formats use their finite bounds, and [`DataType::F8E8M0FNU`] uses its
/// smallest and largest positive powers of two. Returns `None` for non-floating-point data types.
fn float_reduction_identity_bounds(data_type: DataType) -> Option<(f64, f64)> {
    match data_type {
        DataType::BF16
        | DataType::F16
        | DataType::F32
        | DataType::F64
        | DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E5M2 => Some((f64::NEG_INFINITY, f64::INFINITY)),
        DataType::F4E2M1FN => Some((-6.0, 6.0)),
        DataType::F6E2M3FN => Some((-7.5, 7.5)),
        DataType::F6E3M2FN => Some((-28.0, 28.0)),
        DataType::F8E4M3FN => Some((-448.0, 448.0)),
        DataType::F8E4M3FNUZ => Some((-240.0, 240.0)),
        DataType::F8E4M3B11FNUZ => Some((-30.0, 30.0)),
        DataType::F8E5M2FNUZ => Some((-57344.0, 57344.0)),
        DataType::F8E8M0FNU => Some((2f64.powi(-127), 2f64.powi(127))),
        _ => None,
    }
}

/// Returns the (minimum, maximum) values representable by the given integer data type, used as the `Max` and `Min`
/// reduction identities. The values are raw `i64` payloads for [`lower_constant_elements_attribute`], which encodes
/// them at the type's width and signedness — [`DataType::U64`]'s maximum therefore wraps to `-1`, whose
/// two's-complement bits are exactly `u64::MAX`. Returns `None` for non-integer data types.
fn integer_reduction_identity_bounds(data_type: DataType) -> Option<(i64, i64)> {
    if data_type.is_signed() {
        let width = signed_integer_width(data_type).unwrap();
        Some((i64::MIN >> (64 - width), i64::MAX >> (64 - width)))
    } else if data_type.is_unsigned() {
        let width = unsigned_integer_width(data_type).unwrap();
        Some((0, (u64::MAX >> (64 - width)) as i64))
    } else {
        None
    }
}

impl LowerableOperation for ReduceOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("output", output_types, 1, ProgramError);
        let value = lower_reduce_to_mlir(
            self.kind(),
            self.axes(),
            input_values[0],
            &output_types[0],
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        Ok(vec![value])
    }
}

impl LowerableOperation for ArgMaxOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        Ok(vec![lower_index_reduction_to_mlir(
            true,
            self.axis(),
            input_values[0],
            &input_types[0],
            &output_types[0],
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?])
    }
}

impl LowerableOperation for ArgMinOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        Ok(vec![lower_index_reduction_to_mlir(
            false,
            self.axis(),
            input_values[0],
            &input_types[0],
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
    use ryft_core::{
        ArrayOperation, Dimension, DimensionBounds, DimensionVariable, Instruction, Placeholder, ReduceOperation, Shape,
    };

    use crate::experimental::lowering::tests::{
        XlaProgramBuilder, lowered_reduce_module, lowered_unary_module, unproject_plain_program,
    };
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::XlaArrayConstant;
    use crate::to_mlir_module_for_program;

    use super::*;

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_log_sum_exp_as_a_guarded_max_shifted_reduction() {
        // The expansion reduces the maximum with a `-inf` identity, replaces a nonfinite maximum with zero so that
        // an all-`-inf` or empty slice shifts by zero instead of computing `-inf - -inf`, broadcasts the safe
        // maximum back over the reduced axis, and adds it to the logarithm of the shifted exponential sum.
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::LogSumExp)),
                DataType::F32,
                vec![2, 3],
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2xf32> {
                    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
                    %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [1] : \
                      (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
                     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
                      %10 = stablehlo.compare GT, %arg1, %arg2, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %11 = stablehlo.select %10, %arg1, %arg2 : tensor<i1>, tensor<f32>
                      %12 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %13 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %14 = stablehlo.select %13, %arg2, %11 : tensor<i1>, tensor<f32>
                      %15 = stablehlo.select %12, %arg1, %14 : tensor<i1>, tensor<f32>
                      stablehlo.return %15 : tensor<f32>
                    }
                    %1 = stablehlo.is_finite %0 : (tensor<2xf32>) -> tensor<2xi1>
                    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2xf32>
                    %3 = stablehlo.select %1, %0, %2 : tensor<2xi1>, tensor<2xf32>
                    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>
                    %5 = stablehlo.subtract %arg0, %4 : tensor<2x3xf32>
                    %6 = stablehlo.exponential %5 : tensor<2x3xf32>
                    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %7 = stablehlo.reduce(%6 init: %cst_1) applies stablehlo.add across dimensions = [1] : \
                      (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
                    %8 = stablehlo.log %7 : tensor<2xf32>
                    %9 = stablehlo.add %8, %3 : tensor<2xf32>
                    return %9 : tensor<2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_bf16_reduce_max() {
        assert_eq!(
            lowered_reduce_module(DataType::BF16, ReductionKind::Max, vec![1], vec![2, 3]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xbf16>) -> tensor<2xbf16> {
                    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
                    %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [1] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<2xbf16>
                     reducer(%arg1: tensor<bf16>, %arg2: tensor<bf16>)  {
                      %1 = stablehlo.compare GT, %arg1, %arg2, TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<bf16>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<bf16>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<bf16>
                      stablehlo.return %6 : tensor<bf16>
                    }
                    return %0 : tensor<2xbf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_bf16_reduce_sum() {
        assert_eq!(
            lowered_reduce_module(DataType::BF16, ReductionKind::Sum, vec![0], vec![2, 3]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xbf16>) -> tensor<3xbf16> {
                    %0 = stablehlo.convert %arg0 : (tensor<2x3xbf16>) -> tensor<2x3xf32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
                    %2 = stablehlo.convert %1 : (tensor<3xf32>) -> tensor<3xbf16>
                    return %2 : tensor<3xbf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_bf16_reduce_product() {
        assert_eq!(
            lowered_reduce_module(DataType::BF16, ReductionKind::Product, vec![0], vec![2, 3]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xbf16>) -> tensor<3xbf16> {
                    %0 = stablehlo.convert %arg0 : (tensor<2x3xbf16>) -> tensor<2x3xf32>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.multiply across dimensions = [0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
                    %2 = stablehlo.convert %1 : (tensor<3xf32>) -> tensor<3xbf16>
                    return %2 : tensor<3xbf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_log_sum_exp_requires_negative_infinity() {
        // A valid pairwise identity is insufficient for max-shifted padding, including in unchecked programs.
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F8E4M3FN, [4]));
        let output = builder.add_variable(ArrayType::scalar(DataType::F8E4M3FN));
        builder.add_instruction_unchecked(Instruction::new(
            ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::LogSumExp)),
            vec![input],
            vec![output],
            Vec::new(),
        ));
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main"),
            Err(LoweringError::UnsupportedOp {
                op: "`reduce` with kind `log_sum_exp` requires a format with negative infinity but got `f8e4m3fn`"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_preserves_dynamic_reduce_axes() {
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Sum)),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert!(stablehlo.contains("tensor<?x3xf32, #stablehlo.bounds<4, ?>>"), "{stablehlo}");
        assert!(stablehlo.contains("-> tensor<?xf32, #stablehlo.bounds<4>>"), "{stablehlo}");
        assert!(stablehlo.contains("across dimensions = [1]"), "{stablehlo}");

        // Mean needs the removed-axis runtime extent as a divisor. That future dimension residual is an explicit
        // Phase 6/7 boundary rather than an operand added to the primal reduce instruction.
        let reduced = DimensionVariable::new("reduced", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(reduced), Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Mean)),
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
            Err(LoweringError::UnsupportedOp { op: "`reduce` mean over dynamically sized axis 0".to_string() }),
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_bf16_reduce_mean() {
        assert_eq!(
            lowered_reduce_module(DataType::BF16, ReductionKind::Mean, vec![1], vec![2, 3]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xbf16>) -> tensor<2xbf16> {
                    %0 = stablehlo.convert %arg0 : (tensor<2x3xbf16>) -> tensor<2x3xf32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
                    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
                    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2xf32>
                    %3 = stablehlo.divide %1, %2 : tensor<2xf32>
                    %4 = stablehlo.convert %3 : (tensor<2xf32>) -> tensor<2xbf16>
                    return %4 : tensor<2xbf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_full_reduce_mean_without_broadcast() {
        assert_eq!(
            lowered_reduce_module(DataType::F32, ReductionKind::Mean, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<f32> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                    %cst_0 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                    %1 = stablehlo.divide %0, %cst_0 : tensor<f32>
                    return %1 : tensor<f32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_f16_reduce_min() {
        assert_eq!(
            lowered_reduce_module(DataType::F16, ReductionKind::Min, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf16>) -> tensor<f16> {
                    %cst = stablehlo.constant dense<0x7C00> : tensor<f16>
                    %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [0] : (tensor<4xf16>, tensor<f16>) -> tensor<f16>
                     reducer(%arg1: tensor<f16>, %arg2: tensor<f16>)  {
                      %1 = stablehlo.compare LT, %arg1, %arg2, TOTALORDER : (tensor<f16>, tensor<f16>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<f16>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<f16>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<f16>
                      stablehlo.return %6 : tensor<f16>
                    }
                    return %0 : tensor<f16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_f8e5m2_reduce_max_with_infinite_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::F8E5M2, ReductionKind::Max, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf8E5M2>) -> tensor<f8E5M2> {
                    %cst = stablehlo.constant dense<0xFC> : tensor<f8E5M2>
                    %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [0] : (tensor<4xf8E5M2>, tensor<f8E5M2>) -> tensor<f8E5M2>
                     reducer(%arg1: tensor<f8E5M2>, %arg2: tensor<f8E5M2>)  {
                      %1 = stablehlo.compare GT, %arg1, %arg2, TOTALORDER : (tensor<f8E5M2>, tensor<f8E5M2>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<f8E5M2>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f8E5M2>, tensor<f8E5M2>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<f8E5M2>, tensor<f8E5M2>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<f8E5M2>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<f8E5M2>
                      stablehlo.return %6 : tensor<f8E5M2>
                    }
                    return %0 : tensor<f8E5M2>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_f8e4m3fn_reduce_max_with_finite_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::F8E4M3FN, ReductionKind::Max, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf8E4M3FN>) -> tensor<f8E4M3FN> {
                    %cst = stablehlo.constant dense<-4.480000e+02> : tensor<f8E4M3FN>
                    %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [0] : (tensor<4xf8E4M3FN>, tensor<f8E4M3FN>) -> tensor<f8E4M3FN>
                     reducer(%arg1: tensor<f8E4M3FN>, %arg2: tensor<f8E4M3FN>)  {
                      %1 = stablehlo.compare GT, %arg1, %arg2, TOTALORDER : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<f8E4M3FN>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<f8E4M3FN>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<f8E4M3FN>
                      stablehlo.return %6 : tensor<f8E4M3FN>
                    }
                    return %0 : tensor<f8E4M3FN>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_f8e4m3fn_reduce_sum() {
        assert_eq!(
            lowered_reduce_module(DataType::F8E4M3FN, ReductionKind::Sum, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf8E4M3FN>) -> tensor<f8E4M3FN> {
                    %0 = stablehlo.convert %arg0 : (tensor<4xf8E4M3FN>) -> tensor<4xf32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                    %2 = stablehlo.convert %1 : (tensor<f32>) -> tensor<f8E4M3FN>
                    return %2 : tensor<f8E4M3FN>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_f4e2m1fn_reduce_min_with_finite_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::F4E2M1FN, ReductionKind::Min, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf4E2M1FN>) -> tensor<f4E2M1FN> {
                    %cst = stablehlo.constant dense<6.000000e+00> : tensor<f4E2M1FN>
                    %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [0] : (tensor<4xf4E2M1FN>, tensor<f4E2M1FN>) -> tensor<f4E2M1FN>
                     reducer(%arg1: tensor<f4E2M1FN>, %arg2: tensor<f4E2M1FN>)  {
                      %1 = stablehlo.compare LT, %arg1, %arg2, TOTALORDER : (tensor<f4E2M1FN>, tensor<f4E2M1FN>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<f4E2M1FN>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f4E2M1FN>, tensor<f4E2M1FN>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<f4E2M1FN>, tensor<f4E2M1FN>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<f4E2M1FN>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<f4E2M1FN>
                      stablehlo.return %6 : tensor<f4E2M1FN>
                    }
                    return %0 : tensor<f4E2M1FN>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_i32_reduce_max_with_min_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::I32, ReductionKind::Max, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xi32>) -> tensor<i32> {
                    %c = stablehlo.constant dense<-2147483648> : tensor<i32>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.maximum across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
                    return %0 : tensor<i32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_i64_reduce_sum() {
        assert_eq!(
            lowered_reduce_module(DataType::I64, ReductionKind::Sum, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xi64>) -> tensor<i64> {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
                    return %0 : tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_i64_reduce_product() {
        assert_eq!(
            lowered_reduce_module(DataType::I64, ReductionKind::Product, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xi64>) -> tensor<i64> {
                    %c = stablehlo.constant dense<1> : tensor<i64>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.multiply across dimensions = [0] : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
                    return %0 : tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_i32_reduce_mean_with_integer_division() {
        assert_eq!(
            lowered_reduce_module(DataType::I32, ReductionKind::Mean, vec![1], vec![2, 3]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
                    %c = stablehlo.constant dense<0> : tensor<i32>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.add across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
                    %c_0 = stablehlo.constant dense<3> : tensor<i32>
                    %1 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2xi32>
                    %2 = stablehlo.divide %0, %1 : tensor<2xi32>
                    return %2 : tensor<2xi32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_u8_reduce_min_with_max_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::U8, ReductionKind::Min, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xui8>) -> tensor<ui8> {
                    %c = stablehlo.constant dense<255> : tensor<ui8>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.minimum across dimensions = [0] : (tensor<4xui8>, tensor<ui8>) -> tensor<ui8>
                    return %0 : tensor<ui8>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_u64_reduce_min_with_max_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::U64, ReductionKind::Min, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xui64>) -> tensor<ui64> {
                    %c = stablehlo.constant dense<18446744073709551615> : tensor<ui64>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.minimum across dimensions = [0] : (tensor<4xui64>, tensor<ui64>) -> tensor<ui64>
                    return %0 : tensor<ui64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_supports_f8e8m0fnu_extrema_but_rejects_sum() {
        assert_eq!(
            lowered_reduce_module(DataType::F8E8M0FNU, ReductionKind::Sum, vec![0], vec![4]).unwrap_err(),
            LoweringError::UnsupportedDataType { data_type: DataType::F8E8M0FNU },
        );
        assert!(lowered_reduce_module(DataType::F8E8M0FNU, ReductionKind::Max, vec![0], vec![0]).is_ok());
        assert!(lowered_reduce_module(DataType::F8E8M0FNU, ReductionKind::Min, vec![0], vec![0]).is_ok());
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_argmax_with_variadic_reduce() {
        use ryft_core::ArgMaxOperation;

        // An argmax lowers to a variadic `stablehlo.reduce` over the input and an index iota, starting from the lowest
        // value paired with index zero. Its comparator picks the left pair when its value is greater or a NaN facing an
        // ordered value, or when the values are equivalent (i.e., equal under `FLOAT` semantics or both NaN) and the
        // left index is lower.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let outputs = builder
            .add_instruction(ArgMaxOperation::new(1, DataType::I32), Vec::new(), vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        let module =
            to_mlir_module_for_program(&program, &[], &[input_type], &[output_type], "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2xi32> {
                    %0 = stablehlo.iota dim = 1 : tensor<2x3xi32>
                    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
                    %c = stablehlo.constant dense<0> : tensor<i32>
                    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [1] : (tensor<2x3xf32>, tensor<2x3xi32>, tensor<f32>, tensor<i32>) -> (tensor<2xf32>, tensor<2xi32>)
                     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
                      %2 = stablehlo.compare GT, %arg1, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %3 = stablehlo.compare EQ, %arg1, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %5 = stablehlo.compare NE, %arg3, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %6 = stablehlo.compare EQ, %arg3, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %7 = stablehlo.and %4, %6 : tensor<i1>
                      %8 = stablehlo.and %4, %5 : tensor<i1>
                      %9 = stablehlo.or %2, %7 : tensor<i1>
                      %10 = stablehlo.or %3, %8 : tensor<i1>
                      %11 = stablehlo.compare LT, %arg2, %arg4, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                      %12 = stablehlo.and %10, %11 : tensor<i1>
                      %13 = stablehlo.or %9, %12 : tensor<i1>
                      %14 = stablehlo.select %13, %arg1, %arg3 : tensor<i1>, tensor<f32>
                      %15 = stablehlo.select %13, %arg2, %arg4 : tensor<i1>, tensor<i32>
                      stablehlo.return %14, %15 : tensor<f32>, tensor<i32>
                    }
                    return %1#1 : tensor<2xi32>
                  }
                }
            "#},
        );
    }
}
