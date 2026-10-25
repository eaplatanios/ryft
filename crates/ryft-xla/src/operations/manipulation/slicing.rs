use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, DYNAMIC_SLICE_OPERATION_NAME, DataType, Dimension, DimensionBounds, DimensionType,
    DynamicSliceBounds, DynamicSliceOperation, DynamicUpdateSliceOperation, ProgramError, Shape, SliceOperation,
    UpdateSliceOperation,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::EffectTokens;
use crate::experimental::lowering::types::{
    lower_constant_output, lower_physical_bound_value, lower_restore_dynamic_dimensions,
    lower_runtime_dimension_size_i64, lower_static_index_constants, lower_tensor_type, lower_unplaced_constant_output,
    physical_bound_type, reshape_dimension_i32, static_dimensions,
};
use crate::operations::assertions::{
    ASSERT_ACTOR_ATTRIBUTE, ASSERT_DETAIL_ATTRIBUTE, ASSERT_DYNAMIC_SHAPE_SLICE_KIND, ASSERT_KIND_ATTRIBUTE,
    lower_assertion_custom_call, lower_deliberately_false_assertion_predicate,
};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Normalizes scalar slice starts against logical input dimensions before native allocation-based clamping. Signed
/// starts count from the end of their axis once when `allow_negative_indices` is set, unsigned indices are widened
/// before clamping, and only then are bounded starts converted to the common signed i64 carrier.
pub(crate) fn lower_slice_start_indices<'b, 'c: 'b, 't: 'c>(
    input: ValueRef<'b, 'c, 't>,
    input_type: &ArrayType,
    starts: &[ValueRef<'b, 'c, 't>],
    start_types: &[ArrayType],
    sizes: &[usize],
    allow_negative_indices: bool,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", starts, input_type.rank(), ProgramError);
    check_count!("input", start_types, input_type.rank(), ProgramError);
    check_count!("input", sizes, input_type.rank(), ProgramError);
    // A signed start wraps only under the negative-index policy; unsigned starts cannot be negative and Boolean starts
    // are predicate carriers that never wrap.
    let wraps = |start_type: &ArrayType| {
        allow_negative_indices && !start_type.data_type().is_unsigned() && start_type.data_type() != DataType::I1
    };
    // Native clamping already has the right bounds for static inputs and uniformly typed ordinary signed indices.
    // Keep that common path direct, wrapping negative starts in their own type when the static extents fit it;
    // unsigned extremes, predicate carriers, mixed scalar types, and extents beyond the index type need normalization.
    if let Some(static_shape) = input_type.static_shape()
        && start_types.first().is_none_or(|first| {
            matches!(first.data_type(), DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64)
                && start_types.iter().all(|r#type| r#type.data_type() == first.data_type())
                && static_shape
                    .dimensions()
                    .iter()
                    .all(|extent| i64::try_from(*extent).is_ok_and(|extent| extent <= max_value(first.data_type())))
        })
    {
        if !allow_negative_indices {
            return Ok(starts.to_vec());
        }
        return starts
            .iter()
            .zip(start_types)
            .enumerate()
            .map(|(axis, (start, start_type))| {
                let scalar_type = ArrayType::scalar(start_type.data_type());
                let zero = lower_unplaced_constant_output(&[scalar_type.clone()], 0, block, context, location)?[0];
                let extent = lower_unplaced_constant_output(
                    &[scalar_type],
                    static_shape[axis] as i64,
                    block,
                    context,
                    location,
                )?[0];
                let negative = block
                    .append_operation(stable_hlo::compare(
                        *start,
                        zero,
                        stable_hlo::ComparisonDirection::LessThan,
                        stable_hlo::ComparisonType::Signed,
                        location,
                    )?)?
                    .result(0)
                    .unwrap()
                    .as_ref();
                let wrapped =
                    block.append_operation(stable_hlo::add(*start, extent, location)?)?.result(0).unwrap().as_ref();
                Ok(block
                    .append_operation(stable_hlo::select(negative, wrapped, *start, location)?)?
                    .result(0)
                    .unwrap()
                    .as_ref())
            })
            .collect();
    }
    let signed_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
    starts
        .iter()
        .zip(start_types)
        .zip(sizes)
        .enumerate()
        .map(|(axis, ((start, start_type), size))| {
            let data_type = if start_type.data_type().is_unsigned() { DataType::U64 } else { DataType::I64 };
            let scalar_type = ArrayType::scalar(data_type);
            let tensor_type = lower_tensor_type(&scalar_type, context, location)?;
            let mut start = block
                .append_operation(stable_hlo::convert(*start, tensor_type, location)?)?
                .result(0)
                .unwrap()
                .as_ref();
            if start_type.data_type() == DataType::I1 {
                start = block.append_operation(stable_hlo::negate(start, location)?)?.result(0).unwrap().as_ref();
            }
            let extent = lower_runtime_dimension_size_i64(input, axis, block, context, location)?;
            let zero = lower_unplaced_constant_output(&[scalar_type], 0, block, context, location)?[0];
            if wraps(start_type) {
                // The start is a signed i64 here, like the extent: `select(start < 0, start + extent, start)`.
                let negative = block
                    .append_operation(stable_hlo::compare(
                        start,
                        zero,
                        stable_hlo::ComparisonDirection::LessThan,
                        stable_hlo::ComparisonType::Signed,
                        location,
                    )?)?
                    .result(0)
                    .unwrap()
                    .as_ref();
                let wrapped =
                    block.append_operation(stable_hlo::add(start, extent, location)?)?.result(0).unwrap().as_ref();
                start = block
                    .append_operation(stable_hlo::select(negative, wrapped, start, location)?)?
                    .result(0)
                    .unwrap()
                    .as_ref();
            }
            let size = lower_static_index_constants(&[*size], block, context, location)?[0];
            let maximum =
                block.append_operation(stable_hlo::subtract(extent, size, location)?)?.result(0).unwrap().as_ref();
            let maximum = block
                .append_operation(stable_hlo::convert(maximum, tensor_type, location)?)?
                .result(0)
                .unwrap()
                .as_ref();
            let start =
                block.append_operation(stable_hlo::maximum(start, zero, location)?)?.result(0).unwrap().as_ref();
            let start =
                block.append_operation(stable_hlo::minimum(start, maximum, location)?)?.result(0).unwrap().as_ref();
            Ok(block
                .append_operation(stable_hlo::convert(start, signed_type, location)?)?
                .result(0)
                .unwrap()
                .as_ref())
        })
        .collect()
}

/// Returns the largest value a signed integer start of `data_type` can hold, so static extents can be checked to fit
/// the index type before a negative start is wrapped in that type.
fn max_value(data_type: DataType) -> i64 {
    match data_type {
        DataType::I8 => i64::from(i8::MAX),
        DataType::I16 => i64::from(i16::MAX),
        DataType::I32 => i64::from(i32::MAX),
        _ => i64::MAX,
    }
}

/// Lowers a slice. Static limits slice the input directly. A dynamic limit keeps its bounded dynamic axis whole, so
/// the input is first exposed as its static physical allocation (i.e., each dynamic axis is widened to its physical
/// bound), sliced with that bound as the limit along each such axis, and the logical extents of the output's dynamic
/// axes are then restored from the same axes of the input. The lanes beyond each logical extent are never masked,
/// because restoring the logical extents discards them again.
pub(crate) fn lower_slice_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &SliceOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    check_count!("input", input_types, 1, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let static_limits = operation.limits().iter().map(Dimension::value).collect::<Option<Vec<_>>>();
    if let Some(limits) = static_limits {
        let result = block.append_operation(stable_hlo::slice(
            input_values[0],
            operation.start_indices(),
            &limits,
            operation.strides(),
            location,
        )?)?;
        return Ok(vec![result.result(0).expect("stablehlo.slice should return one result").as_ref()]);
    }

    let input_type = &input_types[0];
    let physical_type = physical_bound_type(input_type)?;
    let size_type = lower_tensor_type(&ArrayType::scalar(DataType::I32), context, location)?;
    let mut physical = input_values[0];
    let mut physical_dimensions = input_type.shape().dimensions().to_vec();
    for (axis, dimension) in physical_type.shape().dimensions().iter().enumerate() {
        if physical_dimensions[axis] == *dimension {
            continue;
        }
        let extent = reshape_dimension_i32(dimension.value().unwrap())?;
        let elements = context
            .dense_i32_elements_attribute(size_type, &[extent])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::I32 })?;
        let size = block.append_operation(stable_hlo::constant(elements, location)?)?.result(0).unwrap().as_ref();
        physical_dimensions[axis] = dimension.clone();
        let physical_type = input_type.clone().with_shape(Shape::new(physical_dimensions.clone()));
        physical = block
            .append_operation(stable_hlo::set_dimension_size(
                physical,
                size,
                lower_tensor_type(&physical_type, context, location)?,
                axis,
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref();
    }
    let limits = operation
        .limits()
        .iter()
        .zip(physical_type.shape().dimensions())
        .map(|(limit, physical_extent)| limit.value().or(physical_extent.value()).unwrap())
        .collect::<Vec<_>>();
    let result = block.append_operation(stable_hlo::slice(
        physical,
        operation.start_indices(),
        &limits,
        operation.strides(),
        location,
    )?)?;
    let result = result.result(0).expect("stablehlo.slice should return one result").as_ref();
    let sources = (0..input_type.rank()).map(|axis| (input_values[0], axis)).collect::<Vec<_>>();
    Ok(vec![lower_restore_dynamic_dimensions(result, &output_types[0], &sources, block, context, location)?])
}

/// Lowers one runtime bounds check for an axis of a dynamic-shape-slice operation.
pub(crate) fn lower_dynamic_slice_assertion<'b, 'c: 'b, 't: 'c>(
    axis: usize,
    stride: usize,
    input_size: ValueRef<'b, 'c, 't>,
    start: ValueRef<'b, 'c, 't>,
    size: ValueRef<'b, 'c, 't>,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let predicate = lower_deliberately_false_assertion_predicate(start, block, context, location)?;
    let detail = format!("{axis}:{stride}");
    let backend_config = context.dictionary_attribute(&[
        context.named_attribute(
            context.identifier(ASSERT_ACTOR_ATTRIBUTE),
            context.string_attribute(DYNAMIC_SLICE_OPERATION_NAME),
        ),
        context.named_attribute(
            context.identifier(ASSERT_KIND_ATTRIBUTE),
            context.string_attribute(ASSERT_DYNAMIC_SHAPE_SLICE_KIND),
        ),
        context.named_attribute(context.identifier(ASSERT_DETAIL_ATTRIBUTE), context.string_attribute(detail.as_str())),
    ]);
    lower_assertion_custom_call(
        predicate,
        &[input_size, start, size],
        backend_config,
        effect_tokens,
        block,
        context,
        location,
    )
}

/// Physical lowering plan for one axis of a first-class dynamic shape slice.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct DynamicSliceAxisPlan {
    /// Maximum logical output size admitted by the size operand's type.
    pub(crate) size: usize,

    /// Maximum physical input span needed to realize [`Self::size`] elements at the static stride.
    pub(crate) span: usize,

    /// Whether declared bounds leave some runtime input/start/size combinations that need a checked assertion.
    pub(crate) needs_runtime_assertion: bool,
}

/// Plans one axis of a first-class dynamic shape slice from its declared input, start, and size bounds.
pub(crate) fn plan_dynamic_slice_axis(
    axis: usize,
    stride: usize,
    input_dimension: &Dimension,
    start_type: &DimensionType,
    size_type: &DimensionType,
) -> Result<DynamicSliceAxisPlan, LoweringError> {
    let Some(size_upper) = size_type.bounds().upper() else {
        return Err(LoweringError::UnsupportedOp {
            op: format!("{DYNAMIC_SLICE_OPERATION_NAME} size on axis {axis} needs a finite upper bound",),
        });
    };
    let size = size_upper - 1;
    let span = if size == 0 {
        0
    } else {
        (size - 1).checked_mul(stride).and_then(|size| size.checked_add(1)).ok_or_else(|| {
            LoweringError::UnsupportedOp {
                op: format!("{DYNAMIC_SLICE_OPERATION_NAME} physical span overflows on axis {axis}"),
            }
        })?
    };
    let (input_minimum, input_physical_size) = match input_dimension {
        Dimension::Static(extent) => (*extent, *extent),
        Dimension::Dynamic(variable) => {
            let Some(upper) = variable.bounds().upper() else {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("{DYNAMIC_SLICE_OPERATION_NAME} input on axis {axis} needs a finite upper bound",),
                });
            };
            (variable.bounds().lower(), upper - 1)
        }
    };
    let padded_size = input_physical_size.checked_add(span).ok_or_else(|| LoweringError::UnsupportedOp {
        op: format!("{DYNAMIC_SLICE_OPERATION_NAME} padded input size overflows on axis {axis}"),
    })?;
    reshape_dimension_i32(padded_size)?;
    let bounds_prove_runtime_in_bounds = start_type
        .bounds()
        .upper()
        .and_then(|upper| upper.checked_sub(1))
        .and_then(|start_maximum| start_maximum.checked_add(span))
        .is_some_and(|limit| limit <= input_minimum);
    Ok(DynamicSliceAxisPlan { size, span, needs_runtime_assertion: !bounds_prove_runtime_in_bounds })
}

impl LowerableOperation for SliceOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_slice_to_mlir(
            self,
            input_values,
            input_types,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

impl LowerableOperation for UpdateSliceOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 2, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let index_values =
            lower_static_index_constants(self.start_indices(), &mut lowerer.block, lowerer.context, lowerer.location)?;
        let result = lowerer.block.append_operation(stable_hlo::dynamic_update_slice(
            input_values[0],
            input_values[1],
            index_values.as_slice(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.dynamic_update_slice should return one result").as_ref()])
    }
}

impl LowerableOperation for DynamicSliceOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, self.sizes().len() + 1, ProgramError);
        check_count!("input", input_types, self.sizes().len() + 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let starts = lower_slice_start_indices(
            input_values[0],
            &input_types[0],
            &input_values[1..],
            &input_types[1..],
            self.sizes(),
            self.allows_negative_indices(),
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::dynamic_slice(
            input_values[0],
            &starts,
            self.sizes(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for DynamicUpdateSliceOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("output", output_types, 1, ProgramError);
        check_count!("input", input_values, output_types[0].rank() + 2, ProgramError);
        check_count!("input", input_types, output_types[0].rank() + 2, ProgramError);
        let sizes = static_dimensions(&input_types[1])?;
        let starts = lower_slice_start_indices(
            input_values[0],
            &input_types[0],
            &input_values[2..],
            &input_types[2..],
            &sizes,
            self.allows_negative_indices(),
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::dynamic_update_slice(
            input_values[0],
            input_values[1],
            &starts,
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for DynamicSliceOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let context = lowerer.context;
        let location = lowerer.location;
        let effect_tokens = &mut lowerer.effect_tokens;
        let block = &mut lowerer.block;
        let Some((input, bounds)) = input_values.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let input_type =
            <&ArrayType>::try_from(&input_types[0]).map_err(|error| LoweringError::Tracing(error.into()))?;
        let [output_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
        };
        let output_type = <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
        let rank = output_type.rank();
        if bounds.len() != 2 * rank {
            return Err(ProgramError::InvalidInputCount { expected: 1 + 2 * rank, actual: input_values.len() }.into());
        }
        let (starts, sizes) = bounds.split_at(rank);
        let mut starts = starts.to_vec();
        let start_types = &input_types[1..1 + rank];
        let size_types = &input_types[1 + rank..];
        let mut physical_sizes = Vec::with_capacity(rank);
        let mut physical_spans = Vec::with_capacity(rank);
        for axis in 0..rank {
            let start_type =
                <&DimensionType>::try_from(&start_types[axis]).map_err(|error| LoweringError::Tracing(error.into()))?;
            let size_type =
                <&DimensionType>::try_from(&size_types[axis]).map_err(|error| LoweringError::Tracing(error.into()))?;
            // Clamping is relative to the logical span, not the larger physical allocation window.
            // Normalize once and reuse the same origin for validation and extraction.
            let zero_start = DimensionType::new("start", DimensionBounds::new(0, Some(1)).unwrap());
            let start_type = if self.bounds() == DynamicSliceBounds::Clamp {
                let constants = lower_static_index_constants(&[0, 1, self.strides()[axis]], block, context, location)?;
                let difference = block
                    .append_operation(stable_hlo::subtract(sizes[axis], constants[1], location)?)?
                    .result(0)
                    .unwrap()
                    .as_ref();
                let steps = block
                    .append_operation(stable_hlo::maximum(difference, constants[0], location)?)?
                    .result(0)
                    .unwrap()
                    .as_ref();
                let steps = block
                    .append_operation(stable_hlo::multiply(steps, constants[2], location)?)?
                    .result(0)
                    .unwrap()
                    .as_ref();
                let occupied = block
                    .append_operation(stable_hlo::minimum(sizes[axis], constants[1], location)?)?
                    .result(0)
                    .unwrap()
                    .as_ref();
                let span =
                    block.append_operation(stable_hlo::add(steps, occupied, location)?)?.result(0).unwrap().as_ref();
                let input_size = lower_runtime_dimension_size_i64(*input, axis, block, context, location)?;
                let maximum_start = block
                    .append_operation(stable_hlo::subtract(input_size, span, location)?)?
                    .result(0)
                    .unwrap()
                    .as_ref();
                let maximum_start = block
                    .append_operation(stable_hlo::maximum(maximum_start, constants[0], location)?)?
                    .result(0)
                    .unwrap()
                    .as_ref();
                starts[axis] = block
                    .append_operation(stable_hlo::minimum(starts[axis], maximum_start, location)?)?
                    .result(0)
                    .unwrap()
                    .as_ref();
                &zero_start
            } else {
                start_type
            };
            let plan = plan_dynamic_slice_axis(
                axis,
                self.strides()[axis],
                &input_type.shape().dimensions()[axis],
                start_type,
                size_type,
            )?;
            if plan.needs_runtime_assertion {
                let input_size = lower_runtime_dimension_size_i64(*input, axis, block, context, location)?;
                lower_dynamic_slice_assertion(
                    axis,
                    self.strides()[axis],
                    input_size,
                    starts[axis],
                    sizes[axis],
                    effect_tokens,
                    block,
                    context,
                    location,
                )?;
            }
            physical_sizes.push(plan.size);
            physical_spans.push(plan.span);
        }

        // Reserve enough trailing storage for the maximum window at every valid runtime start. Without this,
        // native dynamic_slice clamps a small valid runtime slice near the end to fit the larger bound window.
        // Only the requested logical size is exposed, so padded values never become observable elements.
        let padding_value = if input_type.data_type() == DataType::F8E8M0FNU { 1 } else { 0 };
        let physical_input =
            lower_physical_bound_value(*input, input_type, padding_value as f64, block, context, location)?;
        let padding = lower_constant_output(
            &[ArrayType::scalar(input_type.data_type())],
            padding_value,
            block,
            context,
            location,
        )?[0];
        let high_padding = physical_spans.iter().map(|span| *span as i64).collect::<Vec<_>>();
        let padded = block
            .append_operation(stable_hlo::pad(
                physical_input,
                padding,
                &vec![0; rank],
                &high_padding,
                &vec![0; rank],
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref();
        let slice =
            block.append_operation(stable_hlo::dynamic_slice(padded, &starts, physical_spans.as_slice(), location)?)?;
        let mut result = slice.result(0).expect("stablehlo.dynamic_slice should return one result").as_ref();
        if self.strides().iter().any(|stride| *stride != 1) {
            let start_indices = vec![0; rank];
            let slice = block.append_operation(stable_hlo::slice(
                result,
                start_indices.as_slice(),
                physical_spans.as_slice(),
                self.strides(),
                location,
            )?)?;
            result = slice.result(0).expect("stablehlo.slice should return one result").as_ref();
        }

        let i32_scalar_type = context
            .tensor_type(context.signless_integer_type(32), &[], None, location)
            .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I32) })?;
        let mut refined_type = output_type
            .clone()
            .with_shape(Shape::new(physical_sizes.iter().copied().map(Dimension::Static).collect()));
        for (axis, dimension) in output_type.shape().dimensions().iter().cloned().enumerate() {
            if !matches!(dimension, Dimension::Dynamic(_)) {
                continue;
            }
            let size = block.append_operation(stable_hlo::convert(sizes[axis], i32_scalar_type, location)?)?;
            let size = size.result(0).expect("stablehlo.convert should return one result").as_ref();
            let mut dimensions = refined_type.shape().dimensions().to_vec();
            dimensions[axis] = dimension;
            refined_type = refined_type.with_shape(Shape::new(dimensions));
            let refined_tensor_type = lower_tensor_type(&refined_type, context, location)?;
            let refined = block.append_operation(stable_hlo::set_dimension_size(
                result,
                size,
                refined_tensor_type,
                axis,
                location,
            )?)?;
            result = refined.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
        }
        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array, ArrayIrType, ArrayOperation, Context, DimensionBounds, DimensionFromScalarOperation, DimensionVariable,
        DynamicSlice, DynamicSliceOperation, DynamicUpdateSliceOperation, EagerContext, Placeholder,
        ReverseModeDifferentiate, UpdateSliceOperation,
    };
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::experimental::lowering::tests::{
        XlaProgramBuilder, ragged_dot_cpu_compilation_options, test_matrix_type, unproject_plain_program,
    };
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::{
        FlatXlaProgram, XlaArrayConstant, XlaConstant, XlaProgramBuilder as ArrayIrXlaProgramBuilder,
    };
    use crate::operations::constants::dynamic_constructor_types;
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{XlaArray, to_mlir_module_for_program};

    use super::*;

    #[test]
    fn test_lower_dynamic_slice_indices() {
        let client = execution_client();
        for (index_type, index_bytes, negative) in [
            (DataType::I1, vec![1_u8], true),
            (DataType::I8, vec![127_u8], false),
            (DataType::U64, values_to_bytes(&[u64::MAX]), false),
        ] {
            let input_type = ArrayType::new(
                DataType::I64,
                Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                    "rows",
                    DimensionBounds::new(2, Some(5)).unwrap(),
                ))]),
            );
            let update_type = ArrayType::new_static(DataType::I64, [1]);
            let index_type = ArrayType::scalar(index_type);
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(input_type.clone());
            let update = builder.add_input(update_type.clone());
            let index = builder.add_input(index_type.clone());
            let sliced = builder
                .add_instruction(DynamicSliceOperation::new(vec![1]), Vec::new(), vec![input, index], None)
                .unwrap()[0];
            let updated = builder
                .add_instruction(DynamicUpdateSliceOperation::new(), Vec::new(), vec![input, update, index], None)
                .unwrap()[0];
            let program = unproject_plain_program(
                builder
                    .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                        vec![sliced, updated],
                        vec![Placeholder; 3],
                        vec![Placeholder; 2],
                    )
                    .unwrap(),
            );
            let module = to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type.clone(), update_type.clone(), index_type.clone()],
                &vec![update_type, input_type],
                "main",
                None,
                None,
            )
            .unwrap();
            let executable = client
                .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
                .unwrap();
            let device = executable.addressable_devices().unwrap().remove(0);
            let values = [i64::MAX, i64::MIN, 9_007_199_254_740_993, -9_007_199_254_740_993];
            for size in [2_i32, 4] {
                let inputs = [
                    (values_to_bytes(&values), BufferType::I64, vec![4]),
                    (values_to_bytes(&[42_i64]), BufferType::I64, vec![1]),
                    (index_bytes.clone(), XlaArray::physical_buffer_type(index_type.data_type()), vec![]),
                    (values_to_bytes(&[size]), BufferType::I32, vec![]),
                ]
                .into_iter()
                .map(|(bytes, data_type, shape)| ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(bytes.as_slice(), data_type, shape.as_slice(), None, device.clone(), None)
                            .unwrap(),
                    ),
                    donatable: false,
                })
                .collect::<Vec<_>>();
                let outputs = executable
                    .execute(
                        vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                        vec![],
                        0,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap()
                    .block_until_ready()
                    .unwrap()
                    .remove(0)
                    .outputs;
                let axis = if negative { 0 } else { size as usize - 1 };
                assert_eq!(
                    values_from_bytes::<i64>(&outputs[0].copy_to_host(None).unwrap().r#await().unwrap()),
                    vec![values[axis]],
                );
                let mut expected = values[..size as usize].to_vec();
                expected[axis] = 42;
                let bytes = outputs[1].copy_to_host(None).unwrap().r#await().unwrap();
                assert_eq!(values_from_bytes::<i64>(&bytes[..size as usize * 8]), expected);
                assert_eq!(
                    values_from_bytes::<i64>(&outputs[2].copy_to_host(None).unwrap().r#await().unwrap()),
                    vec![i64::from(size)],
                );
            }
        }
    }

    #[test]
    fn test_lower_dynamic_slice_runtime_windows() {
        let client = execution_client();
        crate::operations::assertions::ensure_assertion_handler_registered(&client).unwrap();
        let input_type = ArrayType::new_static(DataType::I64, [4]);
        let scalar_type = ArrayType::scalar(DataType::I64);
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let mut bounds = Vec::new();
        for name in ["start", "size"] {
            let scalar = builder.add_input(scalar_type.clone().into());
            bounds.push(
                builder
                    .add_instruction(
                        DimensionFromScalarOperation::new(
                            DimensionVariable::new(name, DimensionBounds::new(0, Some(5)).unwrap()).into(),
                        ),
                        Vec::new(),
                        vec![scalar],
                        None,
                    )
                    .unwrap()[0],
            );
        }
        let result = builder
            .add_instruction(
                DynamicSliceOperation::<ArrayIrType>::from_rank(1),
                Vec::new(),
                vec![input, bounds[0], bounds[1]],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![result], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let output_type = <&ArrayType>::try_from(&program.output_types()[0]).unwrap().clone();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type, scalar_type.clone(), scalar_type],
            &vec![output_type],
            "main",
            None,
            None,
        )
        .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        let values = [i64::MAX, i64::MIN, 9_007_199_254_740_993, -9_007_199_254_740_993];
        for (start, size) in [(3_i64, 1_i64), (2, 2), (0, 4), (4, 0), (3, 2)] {
            let inputs = [
                (values_to_bytes(&values), vec![4]),
                (values_to_bytes(&[start]), vec![]),
                (values_to_bytes(&[size]), vec![]),
            ]
            .into_iter()
            .map(|(bytes, shape)| ExecutionInput {
                buffer: Arc::new(
                    client
                        .buffer(bytes.as_slice(), BufferType::I64, shape.as_slice(), None, device.clone(), None)
                        .unwrap(),
                ),
                donatable: false,
            })
            .collect::<Vec<_>>();
            let result = executable
                .execute(
                    vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                    vec![],
                    0,
                    None,
                    None,
                    None,
                    None,
                )
                .and_then(|execution| execution.block_until_ready());
            if start + size > 4 {
                assert!(result.err().unwrap().to_string().contains("dynamic_slice"));
            } else {
                let outputs = result.unwrap().remove(0).outputs;
                let bytes = outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
                assert_eq!(
                    values_from_bytes::<i64>(&bytes[..size as usize * 8]),
                    &values[start as usize..(start + size) as usize],
                );
                assert_eq!(
                    values_from_bytes::<i64>(&outputs[1].copy_to_host(None).unwrap().r#await().unwrap()),
                    vec![size],
                );
            }
        }
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_slicing_operations() {
        let input_type = test_matrix_type(2, 3);
        let update_type = test_matrix_type(1, 2);
        let index_type = ArrayType::scalar(DataType::I32);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let update = builder.add_input(update_type);
        let index_0 = builder.add_input(index_type.clone());
        let index_1 = builder.add_input(index_type);
        let sliced = builder
            .add_instruction(
                SliceOperation::new(vec![1, 1], vec![2, 3]).with_strides(vec![1, 1]).unwrap(),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let updated = builder
            .add_instruction(UpdateSliceOperation::new(vec![0, 1]), Vec::new(), vec![input, update], None)
            .unwrap()[0];
        let dynamic_sliced = builder
            .add_instruction(DynamicSliceOperation::new(vec![1, 2]), Vec::new(), vec![input, index_0, index_1], None)
            .unwrap()[0];
        let dynamic_updated = builder
            .add_instruction(
                DynamicUpdateSliceOperation::new(),
                Vec::new(),
                vec![input, update, index_0, index_1],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![sliced, updated, dynamic_sliced, dynamic_updated],
                vec![Placeholder, Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder, Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<1x2xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<1x2xf32>, tensor<2x3xf32>, tensor<1x2xf32>, tensor<2x3xf32>) {
                    %0 = stablehlo.slice %arg0 [1:2, 1:3] : (tensor<2x3xf32>) -> tensor<1x2xf32>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %c_0 = stablehlo.constant dense<1> : tensor<i64>
                    %1 = stablehlo.dynamic_update_slice %arg0, %arg1, %c, %c_0 : (tensor<2x3xf32>, tensor<1x2xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
                    %c_1 = stablehlo.constant dense<0> : tensor<i32>
                    %c_2 = stablehlo.constant dense<2> : tensor<i32>
                    %2 = stablehlo.compare LT, %arg2, %c_1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                    %3 = stablehlo.add %arg2, %c_2 : tensor<i32>
                    %4 = stablehlo.select %2, %3, %arg2 : tensor<i1>, tensor<i32>
                    %c_3 = stablehlo.constant dense<0> : tensor<i32>
                    %c_4 = stablehlo.constant dense<3> : tensor<i32>
                    %5 = stablehlo.compare LT, %arg3, %c_3, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                    %6 = stablehlo.add %arg3, %c_4 : tensor<i32>
                    %7 = stablehlo.select %5, %6, %arg3 : tensor<i1>, tensor<i32>
                    %8 = stablehlo.dynamic_slice %arg0, %4, %7, sizes = [1, 2] : (tensor<2x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x2xf32>
                    %c_5 = stablehlo.constant dense<0> : tensor<i32>
                    %c_6 = stablehlo.constant dense<2> : tensor<i32>
                    %9 = stablehlo.compare LT, %arg2, %c_5, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                    %10 = stablehlo.add %arg2, %c_6 : tensor<i32>
                    %11 = stablehlo.select %9, %10, %arg2 : tensor<i1>, tensor<i32>
                    %c_7 = stablehlo.constant dense<0> : tensor<i32>
                    %c_8 = stablehlo.constant dense<3> : tensor<i32>
                    %12 = stablehlo.compare LT, %arg3, %c_7, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                    %13 = stablehlo.add %arg3, %c_8 : tensor<i32>
                    %14 = stablehlo.select %12, %13, %arg3 : tensor<i1>, tensor<i32>
                    %15 = stablehlo.dynamic_update_slice %arg0, %arg1, %11, %14 : (tensor<2x3xf32>, tensor<1x2xf32>, tensor<i32>, tensor<i32>) -> tensor<2x3xf32>
                    return %0, %1, %8, %15 : tensor<1x2xf32>, tensor<2x3xf32>, tensor<1x2xf32>, tensor<2x3xf32>
                  }
                }
            "#},
        );

        // Static `sizes` make the result exact even when an operand axis is dynamic; no first-class result extent or
        // dynamic-slice-size operand is needed.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(3)]));
        let index_type = ArrayType::scalar(DataType::I32);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let index_0 = builder.add_input(index_type.clone());
        let index_1 = builder.add_input(index_type);
        let output = builder
            .add_instruction(DynamicSliceOperation::new(vec![1, 2]), Vec::new(), vec![input, index_0, index_1], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert!(stablehlo.contains("stablehlo.dynamic_slice"), "{stablehlo}");
        assert!(stablehlo.contains("sizes = [1, 2]"), "{stablehlo}");
        assert!(stablehlo.contains("-> tensor<1x2xf32>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_whole_axis_dynamic_slice() {
        // A dynamic limit keeps its bounded dynamic axis whole: the lowering slices the physical allocation, whose
        // dynamic axis is widened to its bound, and then restores that axis's logical extent from the input.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(4)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                SliceOperation::new(vec![0, 1], vec![Dimension::Dynamic(batch), Dimension::Static(4)])
                    .with_strides(vec![1, 2])
                    .unwrap(),
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
                  func.func @main(%arg0: tensor<?x4xf32, #stablehlo.bounds<4, ?>>) -> tensor<?x2xf32, #stablehlo.bounds<4, ?>> {
                    %c = stablehlo.constant dense<4> : tensor<i32>
                    %0 = stablehlo.set_dimension_size %arg0, %c, dim = 0 : (tensor<?x4xf32, #stablehlo.bounds<4, ?>>, tensor<i32>) -> tensor<4x4xf32>
                    %1 = stablehlo.slice %0 [0:4, 1:4:2] : (tensor<4x4xf32>) -> tensor<4x2xf32>
                    %2 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x4xf32, #stablehlo.bounds<4, ?>>) -> tensor<i32>
                    %3 = stablehlo.set_dimension_size %1, %2, dim = 0 : (tensor<4x2xf32>, tensor<i32>) -> tensor<?x2xf32, #stablehlo.bounds<4, ?>>
                    return %3 : tensor<?x2xf32, #stablehlo.bounds<4, ?>>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_slicing_vjp_pullbacks_lower_to_stablehlo() {
        use ryft_core::Slice;

        // The static slice pullback writes the cotangent into a zero array at the static offsets via the
        // statically indexed update-slice, which lowers to `stablehlo.dynamic_update_slice` with constant indices.
        // The structural-zero destination is emitted as a `ZeroOperation` instruction in the pullback, which lowers
        // through the canonical zero path to a scalar constant broadcast to the array shape. The reverse path
        // stages the pullback over the primal operation family taking `[output_cotangents ++ residuals]`; this slice
        // pullback captures no residuals, so the pullback consumes only the single output cotangent.
        let (_, pullback): (Array, _) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(|x, ()| Ok(x.slice(&[1], &[3], &[1]).unwrap()), Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(), ())
            .unwrap();
        let (pullback, _residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf64>) -> tensor<4xf64> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
                    %c = stablehlo.constant dense<1> : tensor<i64>
                    %1 = stablehlo.dynamic_update_slice %0, %arg0, %c : (tensor<4xf64>, tensor<2xf64>, tensor<i64>) -> tensor<4xf64>
                    return %1 : tensor<4xf64>
                  }
                }
            "#},
        );

        // The strided slice pullback pads the cotangent with a zero scalar using the inverse padding
        // (`low = start`, `interior = stride - 1`), which lowers to `stablehlo.pad`.
        let (_, pullback): (Array, _) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |x, ()| Ok(x.slice(&[1], &[6], &[2]).unwrap()),
                Array::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
                (),
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<3xf64>) -> tensor<6xf64> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.pad %arg0, %cst, low = [1], high = [0], interior = [1] : (tensor<3xf64>, tensor<f64>) -> tensor<6xf64>
                    return %0 : tensor<6xf64>
                  }
                }
            "#},
        );

        // The pad pullback first edge-unpads the cotangent and then slices it with the non-unit interior stride for the
        // input. It constructs a Boolean padding-position mask for the padding-value cotangent so non-finite values at
        // input positions cannot contaminate its sum.
        let (_, pullback): (Array, _) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |(x, padding_value), ()| {
                    use ryft_core::Pad;
                    Ok(x.pad(&padding_value, &[1], &[2], &[1]).unwrap())
                },
                (Array::vector(vec![1.0, 2.0, 3.0]).unwrap(), Array::scalar(9.0).unwrap()),
                (),
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<8xf64>) -> (tensor<3xf64>, tensor<f64>) {
                    %0 = stablehlo.slice %arg0 [1:6:2] : (tensor<8xf64>) -> tensor<3xf64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %1 = stablehlo.pad %0, %cst, low = [0], high = [0], interior = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    %c_0 = stablehlo.constant dense<true> : tensor<i1>
                    %3 = stablehlo.pad %2, %c_0, low = [1], high = [2], interior = [1] : (tensor<3xi1>, tensor<i1>) -> tensor<8xi1>
                    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<8xf64>
                    %5 = stablehlo.select %3, %arg0, %4 : tensor<8xi1>, tensor<8xf64>
                    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %6 = stablehlo.reduce(%5 init: %cst_2) applies stablehlo.add across dimensions = [0] : (tensor<8xf64>, tensor<f64>) -> tensor<f64>
                    return %1, %6 : tensor<3xf64>, tensor<f64>
                  }
                }
            "#},
        );

        // The dynamic slice pullback scatters the cotangent at the captured index factors, which materialize as
        // integer constants through `lower_literal_value`.
        let (_, pullback): (Array, _) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |x, ()| {
                    let start = x
                        .context()
                        .lift(Array::from_elements::<i32>(ArrayType::scalar(DataType::I32), &[1]).unwrap())
                        .unwrap();
                    Ok(x.dynamic_slice(&[start], &[2]).unwrap())
                },
                Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                (),
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf64>) -> tensor<4xf64> {
                    %c = stablehlo.constant dense<1> : tensor<i32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
                    %c_0 = stablehlo.constant dense<0> : tensor<i32>
                    %c_1 = stablehlo.constant dense<4> : tensor<i32>
                    %1 = stablehlo.compare LT, %c, %c_0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                    %2 = stablehlo.add %c, %c_1 : tensor<i32>
                    %3 = stablehlo.select %1, %2, %c : tensor<i1>, tensor<i32>
                    %4 = stablehlo.dynamic_update_slice %0, %arg0, %3 : (tensor<4xf64>, tensor<2xf64>, tensor<i32>) -> tensor<4xf64>
                    return %4 : tensor<4xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_slicing_vjp_reference_pullback_lowers_without_dense_zero() {
        use ryft_core::CotangentDestinationKind;

        // The runtime index remains an input after transposition. Functionalizing the buffer update must keep
        // only the selected-block addition, without constructing a full-size zero cotangent first.
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(ArrayIrType::Array(ArrayType::new_static(DataType::F64, [5])));
        let start = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I32)));
        let output = builder
            .add_instruction(
                ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![2])),
                Vec::new(),
                vec![input, start],
                None,
            )
            .unwrap()[0];
        let program: FlatXlaProgram = builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        let discharged = pullback.discharge_references(0).unwrap();
        let stablehlo = to_mlir_module_for_program(
            discharged.program(),
            &[],
            &vec![
                ArrayType::new_static(DataType::F64, [2]),
                ArrayType::new_static(DataType::F64, [5]),
                ArrayType::scalar(DataType::I32),
            ],
            &vec![ArrayType::new_static(DataType::F64, [5])],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf64>, %arg1: tensor<5xf64>, %arg2: tensor<i32>) -> tensor<5xf64> {
                    %c = stablehlo.constant dense<0> : tensor<i32>
                    %c_0 = stablehlo.constant dense<5> : tensor<i32>
                    %0 = stablehlo.compare LT, %arg2, %c, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                    %1 = stablehlo.add %arg2, %c_0 : tensor<i32>
                    %2 = stablehlo.select %0, %1, %arg2 : tensor<i1>, tensor<i32>
                    %3 = stablehlo.dynamic_slice %arg1, %2, sizes = [2] : (tensor<5xf64>, tensor<i32>) -> tensor<2xf64>
                    %4 = stablehlo.add %3, %arg0 : tensor<2xf64>
                    %c_1 = stablehlo.constant dense<0> : tensor<i32>
                    %c_2 = stablehlo.constant dense<5> : tensor<i32>
                    %5 = stablehlo.compare LT, %arg2, %c_1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                    %6 = stablehlo.add %arg2, %c_2 : tensor<i32>
                    %7 = stablehlo.select %5, %6, %arg2 : tensor<i1>, tensor<i32>
                    %8 = stablehlo.dynamic_update_slice %arg1, %4, %7 : (tensor<5xf64>, tensor<2xf64>, tensor<i32>) -> tensor<5xf64>
                    return %8 : tensor<5xf64>
                  }
                }
            "#},
        );
    }

    /// Creates a first-class dimension type with the provided bounds.
    pub(crate) fn dimension_type(name: &str, bounds: DimensionBounds) -> DimensionType {
        DimensionType::new(name, bounds)
    }

    /// Extracts the unsupported-operation diagnostic produced by the physical slice planner.
    fn unsupported_operation(error: LoweringError) -> String {
        let LoweringError::UnsupportedOp { op } = error else {
            panic!("expected an unsupported-operation error but received {error}");
        };
        op
    }

    #[test]
    fn test_dynamic_constructor_types() {
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(5)).unwrap());
        let output_type = ArrayType::new(DataType::C64, Shape::new(vec![size.into(), 2.into()]));
        assert_eq!(
            dynamic_constructor_types("one_like", 1, &[output_type.clone().into()]),
            Ok((output_type, ArrayType::new_static(DataType::C64, [4, 2])))
        );

        let size = DimensionVariable::new("size", DimensionBounds::unbounded());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![size.into()]));
        assert_eq!(
            dynamic_constructor_types("zero_like", 1, &[output_type.into()]),
            Err(LoweringError::UnsupportedOp {
                op: "zero_like output dimension `size` needs a finite upper bound for physical buffer allocation"
                    .to_string()
            })
        );

        let bound = i32::MAX as usize + 1;
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(bound + 1)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![size.into()]));
        assert_eq!(
            dynamic_constructor_types("one_like", 1, &[output_type.into()]),
            Err(LoweringError::DynamicDimensionOutOfRange { value: bound })
        );
    }

    #[test]
    fn test_plan_dynamic_slice_axis() {
        let unbounded = dimension_type("unbounded", DimensionBounds::unbounded());
        let bounded_start = dimension_type("start", DimensionBounds::non_negative(Some(5)).unwrap());
        let bounded_size = dimension_type("size", DimensionBounds::non_negative(Some(5)).unwrap());

        assert_eq!(
            unsupported_operation(
                plan_dynamic_slice_axis(2, 1, &Dimension::Static(4), &bounded_start, &unbounded).unwrap_err(),
            ),
            "dynamic_slice size on axis 2 needs a finite upper bound",
        );

        let unbounded_input = Dimension::Dynamic(DimensionVariable::new("input", DimensionBounds::unbounded()));
        assert_eq!(
            unsupported_operation(
                plan_dynamic_slice_axis(1, 1, &unbounded_input, &bounded_start, &bounded_size).unwrap_err(),
            ),
            "dynamic_slice input on axis 1 needs a finite upper bound",
        );

        let overflowing_size = dimension_type("size", DimensionBounds::non_negative(Some(usize::MAX)).unwrap());
        assert_eq!(
            unsupported_operation(
                plan_dynamic_slice_axis(3, 2, &Dimension::Static(usize::MAX), &bounded_start, &overflowing_size,)
                    .unwrap_err(),
            ),
            "dynamic_slice physical span overflows on axis 3",
        );

        assert_eq!(
            plan_dynamic_slice_axis(0, 1, &Dimension::Static(3), &bounded_start, &bounded_size).unwrap(),
            DynamicSliceAxisPlan { size: 4, span: 4, needs_runtime_assertion: true },
        );

        let bounded_input =
            Dimension::Dynamic(DimensionVariable::new("input", DimensionBounds::positive(Some(5)).unwrap()));
        assert_eq!(
            plan_dynamic_slice_axis(0, 1, &bounded_input, &bounded_start, &bounded_size).unwrap(),
            DynamicSliceAxisPlan { size: 4, span: 4, needs_runtime_assertion: true },
        );

        let proven_start = dimension_type("start", DimensionBounds::non_negative(Some(1)).unwrap());
        let proven_size = dimension_type("size", DimensionBounds::non_negative(Some(4)).unwrap());
        assert_eq!(
            plan_dynamic_slice_axis(0, 1, &Dimension::Static(4), &proven_start, &proven_size).unwrap(),
            DynamicSliceAxisPlan { size: 3, span: 3, needs_runtime_assertion: false },
        );
    }
}
