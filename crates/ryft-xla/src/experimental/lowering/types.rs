use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, AtomId, CaptureReference, ComparisonDirection, DataType, Dimension, DimensionValue, Layout,
    ProgramError, Shape, Type as RyftType,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Attribute, AttributeRef, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef, FloatTypeRef,
    IntegerTypeRef, Location, LocationRef, Operation as MlirOperation, Size as MlirSize, TensorTypeRef, Type, TypeRef,
    Value as MlirValue, ValueRef,
};

use crate::ToPjrt;
use crate::experimental::lowering::{LoweringError, MlirLowerableValue};
use crate::experimental::ops::{XlaArrayConstant, XlaConstant};
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::operations::manipulation::memory::annotate_output_memory;

/// Widens a logical signed one-bit integer to `i8`, recovering `-1` from StableHLO's predicate carrier.
pub(crate) fn lower_signed_one_bit_to_mlir<'b, 'c: 'b, 't: 'c, B, L>(
    input: ValueRef<'b, 'c, 't>,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let input_type = input.r#type()?.cast::<TensorTypeRef>().unwrap();
    let dimensions = input_type.dimensions().collect::<Vec<_>>();
    let signed_type =
        context.tensor_type(context.signless_integer_type(8), &dimensions, input_type.encoding()?, location)?;
    let widened = block.append_operation(stable_hlo::convert(input, signed_type, location)?)?;
    let signed = block.append_operation(stable_hlo::negate(widened.result(0).unwrap().as_ref(), location)?)?;
    Ok(signed.result(0).unwrap().as_ref())
}

/// Reads one runtime tensor dimension and widens StableHLO's `i32` size result to Ryft's `i64` dimension ABI.
pub(crate) fn lower_runtime_dimension_size_i64<'b, 'c: 'b, 't: 'c>(
    input: ValueRef<'b, 'c, 't>,
    axis: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let size = block.append_operation(stable_hlo::get_dimension_size(input, axis, location)?)?;
    let size = size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref();
    let i64_scalar = context.tensor_type(context.signless_integer_type(64), &[], None, location)?;
    let size = block.append_operation(stable_hlo::convert(size, i64_scalar, location)?)?;
    Ok(size.result(0).expect("stablehlo.convert should return one result").as_ref())
}

/// Returns the statically shaped physical-bound carrier type of one bounded dynamic array type.
pub(crate) fn physical_bound_type(r#type: &ArrayType) -> Result<ArrayType, LoweringError> {
    let dimensions = r#type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
            Dimension::Dynamic(variable) => {
                variable.bounds().upper().and_then(|upper| upper.checked_sub(1)).map(Dimension::Static).ok_or_else(
                    || LoweringError::UnsupportedOp {
                        op: format!("dynamic dimension {variable} needs a finite positive physical bound"),
                    },
                )
            }
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    Ok(r#type.clone().with_shape(Shape::new(dimensions)))
}

/// Materializes the full physical buffer of one bounded dynamic value. The original runtime sizes are retained before
/// each dynamic axis is widened to its physical bound, and every newly exposed lane is replaced with `padding_value`.
pub(crate) fn lower_physical_bound_value<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    padding_value: f64,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    lower_physical_bound_value_with_padding(value, r#type, padding_value, None, block, context, location)
}

/// Exposes bounded storage, masking inactive lanes with either an exact scalar value or a numeric constant.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_physical_bound_value_with_padding<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    padding_value: f64,
    scalar_padding: Option<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if r#type.static_shape().is_some() {
        return Ok(value);
    }
    let runtime_sizes = r#type
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .map(|(axis, dimension)| {
            if matches!(dimension, Dimension::Static(_)) {
                return Ok(None);
            }
            let size = block.append_operation(stable_hlo::get_dimension_size(value, axis, location)?)?;
            Ok(Some(size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref()))
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let physical_type = physical_bound_type(r#type)?;
    let size_type = lower_tensor_type(&ArrayType::scalar(DataType::I32), context, location)?;
    let mut data = value;
    let mut physicalized_dimensions = r#type.shape().dimensions().to_vec();
    for (axis, dimension) in r#type.shape().dimensions().iter().enumerate() {
        if matches!(dimension, Dimension::Static(_)) {
            continue;
        }
        let physical_extent = physical_type.shape().dimensions()[axis].value().unwrap();
        let extent = reshape_dimension_i32(physical_extent)?;
        let elements = context
            .dense_i32_elements_attribute(size_type, &[extent])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::I32 })?;
        let size = block.append_operation(stable_hlo::constant(elements, location)?)?;
        let size = size.result(0).expect("stablehlo.constant should return one result").as_ref();
        physicalized_dimensions[axis] = Dimension::Static(physical_extent);
        let physicalized_type = r#type.clone().with_shape(Shape::new(physicalized_dimensions.clone()));
        let physicalized = block.append_operation(stable_hlo::set_dimension_size(
            data,
            size,
            lower_tensor_type(&physicalized_type, context, location)?,
            axis,
            location,
        )?)?;
        data = physicalized.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
    }
    let index_type = ArrayType::new(DataType::I32, physical_type.shape().clone());
    let index_tensor_type = lower_tensor_type(&index_type, context, location)?;
    let mut in_bounds = None;
    for (axis, dimension) in r#type.shape().dimensions().iter().enumerate() {
        if matches!(dimension, Dimension::Static(_)) {
            continue;
        }
        let indices = block.append_operation(stable_hlo::iota(index_tensor_type, axis, location)?)?;
        let indices = indices.result(0).expect("stablehlo.iota should return one result").as_ref();
        let size = runtime_sizes[axis].unwrap();
        let size = block.append_operation(stable_hlo::broadcast(size, index_tensor_type, &[], location)?)?;
        let size = size.result(0).expect("stablehlo.broadcast should return one result").as_ref();
        let axis_in_bounds = lower_compare_to_mlir(ComparisonDirection::LessThan, indices, size, block, location)?;
        in_bounds = Some(match in_bounds {
            None => axis_in_bounds,
            Some(in_bounds) => block
                .append_operation(stable_hlo::and(in_bounds, axis_in_bounds, location)?)?
                .result(0)
                .expect("stablehlo.and should return one result")
                .as_ref(),
        });
    }
    let physical_tensor_type = lower_tensor_type(&physical_type, context, location)?;
    let padding = if let Some(padding) = scalar_padding {
        block
            .append_operation(stable_hlo::broadcast(padding, physical_tensor_type, &[], location)?)?
            .result(0)
            .unwrap()
            .as_ref()
    } else if physical_type.data_type().is_complex() {
        // Padding belongs to the complex element type too; construct its real and imaginary components separately.
        let part_type = physical_type.clone().with_data_type(if physical_type.data_type() == DataType::C64 {
            DataType::F32
        } else {
            DataType::F64
        });
        let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
        let real = lower_f64_constant_splat(padding_value, &part_type, part_tensor_type, block, context, location)?;
        let imaginary = lower_f64_constant_splat(0.0, &part_type, part_tensor_type, block, context, location)?;
        block.append_operation(stable_hlo::complex(real, imaginary, location)?)?.result(0).unwrap().as_ref()
    } else {
        lower_f64_constant_splat(padding_value, &physical_type, physical_tensor_type, block, context, location)?
    };
    let masked = block.append_operation(stable_hlo::select(in_bounds.unwrap(), data, padding, location)?)?;
    Ok(masked.result(0).expect("stablehlo.select should return one result").as_ref())
}

/// Restores the dynamic dimensions of `output_type` on a physical-bound value. Each entry in `sources` identifies
/// the value and axis that define the corresponding output dimension.
pub(crate) fn lower_restore_dynamic_dimensions<'b, 'c: 'b, 't: 'c>(
    mut value: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    sources: &[(ValueRef<'b, 'c, 't>, usize)],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if sources.len() != output_type.rank() {
        return Err(ProgramError::InvalidInputCount { expected: output_type.rank(), actual: sources.len() }.into());
    }
    let mut dimensions = physical_bound_type(output_type)?.shape().dimensions().to_vec();
    for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
        if matches!(dimension, Dimension::Static(_)) {
            continue;
        }
        let size =
            block.append_operation(stable_hlo::get_dimension_size(sources[axis].0, sources[axis].1, location)?)?;
        let size = size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref();
        dimensions[axis] = dimension.clone();
        let refined_type = output_type.clone().with_shape(Shape::new(dimensions.clone()));
        let refined = block.append_operation(stable_hlo::set_dimension_size(
            value,
            size,
            lower_tensor_type(&refined_type, context, location)?,
            axis,
            location,
        )?)?;
        value = refined.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
    }
    Ok(value)
}

/// Exposes the physical storage of one bounded dynamic value as a statically shaped value of its
/// [`physical_bound_type`], by setting the runtime size of every dynamic axis to its physical bound. Unlike
/// [`lower_physical_bound_value`], this does not mask the lanes beyond the runtime sizes, so it is only suitable for
/// callers that discard those lanes again (e.g., by slicing to a static extent or by restoring the runtime sizes).
pub(crate) fn lower_physical_storage<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let physical_type = physical_bound_type(r#type)?;
    let mut dimensions = r#type.shape().dimensions().to_vec();
    let mut value = value;
    for (axis, dimension) in r#type.shape().dimensions().iter().enumerate() {
        if matches!(dimension, Dimension::Static(_)) {
            continue;
        }
        let bound = physical_type.shape().dimensions()[axis].value().unwrap();
        let size = lower_unplaced_constant_output(
            &[ArrayType::scalar(DataType::I32)],
            reshape_dimension_i64(bound)?,
            block,
            context,
            location,
        )?
        .remove(0);
        dimensions[axis] = Dimension::Static(bound);
        let exposed_type = r#type.clone().with_shape(Shape::new(dimensions.clone()));
        let exposed = block.append_operation(stable_hlo::set_dimension_size(
            value,
            size,
            lower_tensor_type(&exposed_type, context, location)?,
            axis,
            location,
        )?)?;
        value = exposed.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
    }
    Ok(value)
}

/// Converts one Ryft reshape dimension to StableHLO's signed shape element type.
pub(crate) fn reshape_dimension_i64(value: usize) -> Result<i64, LoweringError> {
    i64::try_from(value).map_err(|_| LoweringError::ReshapeDimensionOutOfRange { value, bit_width: 64 })
}

/// Converts one Ryft reshape dimension to the signed shape element type required by dynamic StableHLO reshape.
pub(crate) fn reshape_dimension_i32(value: usize) -> Result<i32, LoweringError> {
    i32::try_from(value).map_err(|_| LoweringError::ReshapeDimensionOutOfRange { value, bit_width: 32 })
}

/// Lowers static start indices to scalar `i64` StableHLO constants, as consumed by the index operands of
/// `stablehlo.dynamic_update_slice` when lowering the statically indexed `update_slice` operation (StableHLO has no
/// statically indexed update operation).
pub(crate) fn lower_static_index_constants<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    start_indices: &[usize],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let element_type = lower_element_type(DataType::I64, context)?;
    let tensor_type = context
        .tensor_type(element_type, &[], None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I64) })?;
    start_indices
        .iter()
        .map(|index| {
            let elements = lower_constant_elements_attribute(DataType::I64, tensor_type, *index as i64, context)?;
            let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
            Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
        })
        .collect()
}

pub(crate) fn lower_unplaced_constant_output<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    output_types: &[ArrayType],
    integer_value: i64,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("output", output_types, 1, ProgramError);
    let output_type = &output_types[0];
    let tensor_type = lower_tensor_type(output_type, context, location)?;
    let data_type = output_type.data_type();

    // Complex constants are composed as `value + 0i` from two real part constants through `stablehlo.complex`, so
    // that the dense-elements attribute helpers stay real-valued (MLIR has no complex scalar attribute to splat).
    if data_type.is_complex() {
        let part_data_type = if data_type == DataType::C64 { DataType::F32 } else { DataType::F64 };
        let part_tensor_type =
            context.tensor_type(lower_element_type(part_data_type, context)?, &[], None, location)?;
        let real_elements =
            lower_constant_elements_attribute(part_data_type, part_tensor_type, integer_value, context)?;
        let real = block.append_operation(stable_hlo::constant(real_elements, location)?)?;
        let imaginary_elements = lower_constant_elements_attribute(part_data_type, part_tensor_type, 0, context)?;
        let imaginary = block.append_operation(stable_hlo::constant(imaginary_elements, location)?)?;
        let complex = block.append_operation(stable_hlo::complex(
            real.result(0).expect("stablehlo.constant should return one result").as_ref(),
            imaginary.result(0).expect("stablehlo.constant should return one result").as_ref(),
            location,
        )?)?;
        let complex = complex.result(0).expect("stablehlo.complex should return one result").as_ref();
        if output_type.shape().dimensions().is_empty() {
            return Ok(vec![complex]);
        }
        let broadcast = block.append_operation(stable_hlo::broadcast(complex, tensor_type, &[], location)?)?;
        return Ok(vec![broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref()]);
    }

    if !output_type.shape().dimensions().is_empty() {
        let scalar_tensor_type = context.tensor_type(lower_element_type(data_type, context)?, &[], None, location)?;
        let scalar_elements = lower_constant_elements_attribute(data_type, scalar_tensor_type, integer_value, context)?;
        let scalar_constant = block.append_operation(stable_hlo::constant(scalar_elements, location)?)?;
        let broadcast = block.append_operation(stable_hlo::broadcast(
            scalar_constant.result(0).unwrap().as_ref(),
            tensor_type,
            &[],
            location,
        )?)?;
        return Ok(vec![broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref()]);
    }
    let elements = lower_constant_elements_attribute(data_type, tensor_type, integer_value, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    Ok(vec![constant.result(0).expect("stablehlo.constant should return one result").as_ref()])
}

/// Lowers one integer-valued constant and applies its declared non-device memory placement to the final value.
pub(crate) fn lower_constant_output<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    output_types: &[ArrayType],
    integer_value: i64,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let values = lower_unplaced_constant_output(output_types, integer_value, block, context, location)?;
    check_count!("output", output_types, 1, ProgramError);
    Ok(vec![annotate_output_memory(values[0], &output_types[0], block, context, location)?])
}

/// Returns the static dimensions for one tensor type.
pub(crate) fn static_dimensions(array_type: &ArrayType) -> Result<Vec<usize>, LoweringError> {
    array_type
        .shape()
        .dimensions()
        .iter()
        .map(|size| match size {
            Dimension::Static(value) => Ok(*value),
            Dimension::Dynamic(_) => Err(LoweringError::InvalidTensorType { array_type: array_type.clone() }),
        })
        .collect()
}

/// Lowers one concrete traced value to a StableHLO constant operation and applies its declared memory placement.
pub(crate) fn lower_literal_value<'b, 'c: 'b, 't: 'c, B, V, L>(
    value: &V,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    V: MlirLowerableValue,
    L: Copy + Location<'c, 't>,
{
    let value_type = value.r#type();
    let tensor_type = lower_tensor_type(&value_type, context, location)?;
    let elements = value.to_dense_elements_attribute(tensor_type, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    let constant = constant.result(0).expect("stablehlo.constant should return one result").as_ref();
    annotate_output_memory(constant, &value_type, block, context, location)
}

/// Lowers one captured constant reference by forwarding its runtime captured value.
pub(crate) fn lower_captured_constant<'b, 'c: 'b, 't: 'c, T: RyftType>(
    value: &CaptureReference<T>,
    captured_values: &[ValueRef<'b, 'c, 't>],
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    captured_values
        .get(value.index())
        .copied()
        .ok_or(LoweringError::MissingCapturedConstant { index: value.index() })
}

/// Lowers one first-class dimension extent to the scalar `i64` [`stablehlo.constant`](stable_hlo::constant) that
/// represents a dimension in StableHLO. Immediate [`XlaConstant::Dimension`] atoms and staged
/// [`DimensionOperation::Constant`](ryft_core::DimensionOperation::Constant) instructions must agree on this
/// representation, and so both lowering paths route through this function.
pub(crate) fn lower_dimension_extent<'b, 'c: 'b, 't: 'c, B, L>(
    value: &DimensionValue,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
    let extent = i64::try_from(value.extent()).map_err(|_| LoweringError::UnsupportedOp {
        op: format!("first-class dimension extent {} does not fit in an i64", value.extent()),
    })?;
    let elements = lower_constant_elements_attribute(DataType::I64, tensor_type, extent, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Lowers a traced constant atom to a StableHLO value and returns it. A captured constant forwards the hidden capture
/// argument that carries its runtime value, while an immediate dimension extent is materialized in place — which is
/// exactly what makes extents usable inside a `shard_map` manual region, where no capture table is reachable.
pub(crate) fn lower_constant<'b, 'c: 'b, 't: 'c, B, L>(
    _atom_id: AtomId,
    value: &XlaConstant,
    captured_values: &[ValueRef<'b, 'c, 't>],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    match value {
        XlaConstant::Captured(value) => lower_captured_constant(value, captured_values),
        XlaConstant::Dimension(value) => lower_dimension_extent(value, block, context, location),
        XlaConstant::Boolean(value) => {
            lower_literal_value(&XlaArrayConstant::Boolean(*value), block, context, location)
        }
    }
}

/// Lowers an exact `u64` value as a splatted constant of `output_type`.
fn lower_u64_constant_splat<'b, 'c: 'b, 't: 'c, B, L>(
    value: u64,
    output_type: &ArrayType,
    output_tensor_type: TensorTypeRef<'c, 't>,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let scalar_tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::U64), context, location)?;
    let attribute = context
        .dense_u64_elements_attribute(scalar_tensor_type, &[value])
        .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::U64 })?;
    let constant = block.append_operation(stable_hlo::constant(attribute, location)?)?;
    let constant = constant.result(0).expect("stablehlo.constant should return one result").as_ref();
    if output_type.shape().dimensions().is_empty() {
        return Ok(constant);
    }
    let broadcast = block.append_operation(stable_hlo::broadcast(constant, output_tensor_type, &[], location)?)?;
    Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref())
}

/// Lowers a scalar `u64` constant. Shared by the ragged-dot group-interval arithmetic, the physical
/// `parallel_ragged_all_to_all` lane arithmetic, and the [`AxisIndexOperation`](ryft_core::AxisIndexOperation)
/// coordinate arithmetic.
pub(crate) fn lower_u64_scalar_constant<'b, 'c: 'b, 't: 'c>(
    value: u64,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let output_type = ArrayType::scalar(DataType::U64);
    let output_tensor_type = lower_tensor_type(&output_type, context, location)?;
    lower_u64_constant_splat(value, &output_type, output_tensor_type, block, context, location)
}

/// Encodes an executable boundary's requested layout using PJRT's `mhlo.layout_mode` convention. Internal
/// `LayoutConstraint` calls alone do not constrain the parameter or result ABI.
pub(crate) fn lower_boundary_layout<'c, 't>(
    array_type: &ArrayType,
    kind: &str,
    context: &'c MlirContext<'t>,
) -> Result<Option<AttributeRef<'c, 't>>, LoweringError> {
    let Some(layout) = array_type.layout() else {
        return Ok(None);
    };
    let Layout::Tiled(tiled) = layout else {
        return Err(LoweringError::UnsupportedOp { op: format!("strided {kind} layout `{layout}`") });
    };
    // The pinned XLA SliceToDynamic emitter assumes row-major source/result layouts. Requesting a different
    // dynamic result layout can abort compilation rather than return an error.
    if kind == "output"
        && array_type.static_shape().is_none()
        && !tiled.minor_to_major().iter().copied().eq((0..array_type.rank()).rev())
    {
        return Err(LoweringError::UnsupportedOp {
            op: format!("non-row-major bounded-dynamic output layout `{layout}`"),
        });
    }
    Ok(Some(
        context
            .string_attribute(tiled.to_pjrt().map_err(ProgramError::custom)?.to_string().as_str())
            .as_ref(),
    ))
}

/// Lowers an [`ArrayType`] to a typed MLIR tensor type.
pub(crate) fn lower_tensor_type<'c, 't, L: Location<'c, 't>>(
    array_type: &ArrayType,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
    let element_type = lower_element_type(array_type.data_type(), context)?;
    let dimensions = array_type
        .shape()
        .dimensions()
        .iter()
        .map(|size| match size {
            Dimension::Static(value) => MlirSize::Static(*value),
            Dimension::Dynamic(_) => MlirSize::Dynamic,
        })
        .collect::<Vec<_>>();
    let bounds = array_type.shape().dimensions().iter().map(stable_hlo_dynamic_dimension_bound).collect::<Vec<_>>();
    let encoding = bounds
        .iter()
        .any(Option::is_some)
        .then(|| context.stable_hlo_tensor_type_extensions(bounds.as_slice()))
        .transpose()?
        .map(|attribute| attribute.as_ref());
    context
        .tensor_type(element_type, dimensions.as_slice(), encoding, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: array_type.clone() })
}

/// Converts a Ryft exclusive dynamic dimension bound to StableHLO's inclusive maximum dimension size.
pub(crate) fn stable_hlo_dynamic_dimension_bound(size: &Dimension) -> Option<usize> {
    match size {
        Dimension::Static(_) => None,
        Dimension::Dynamic(variable) => variable.bounds().upper().and_then(|upper| upper.checked_sub(1)),
    }
}

/// Lowers one [`DataType`] to the corresponding MLIR element type.
pub(crate) fn lower_element_type<'c, 't>(
    data_type: DataType,
    context: &'c MlirContext<'t>,
) -> Result<TypeRef<'c, 't>, LoweringError> {
    Ok(match data_type {
        DataType::Token => return Err(LoweringError::UnsupportedDataType { data_type }),
        // StableHLO has no zero-information element type. Use an i1 tensor as an unobservable physical carrier while
        // retaining the logical `DataType::Zero` in Ryft's lowered/compiled metadata.
        DataType::Zero => context.signless_integer_type(1).as_ref(),
        DataType::Boolean => context.signless_integer_type(1).as_ref(),
        DataType::I1 => context.signless_integer_type(1).as_ref(),
        DataType::I2 => context.signless_integer_type(2).as_ref(),
        DataType::I4 => context.signless_integer_type(4).as_ref(),
        DataType::I8 => context.signless_integer_type(8).as_ref(),
        DataType::I16 => context.signless_integer_type(16).as_ref(),
        DataType::I32 => context.signless_integer_type(32).as_ref(),
        DataType::I64 => context.signless_integer_type(64).as_ref(),
        // StableHLO admits signless `i1` as its Boolean/one-bit carrier but rejects `ui1`. U1's unsigned semantics
        // remain selected by Ryft operation metadata (e.g., comparison direction), while its sole bit is unchanged.
        DataType::U1 => context.signless_integer_type(1).as_ref(),
        DataType::U2 => context.unsigned_integer_type(2).as_ref(),
        DataType::U4 => context.unsigned_integer_type(4).as_ref(),
        DataType::U8 => context.unsigned_integer_type(8).as_ref(),
        DataType::U16 => context.unsigned_integer_type(16).as_ref(),
        DataType::U32 => context.unsigned_integer_type(32).as_ref(),
        DataType::U64 => context.unsigned_integer_type(64).as_ref(),
        DataType::BF16 => context.bfloat16_type().as_ref(),
        DataType::F16 => context.float16_type().as_ref(),
        DataType::F32 => context.float32_type().as_ref(),
        DataType::F64 => context.float64_type().as_ref(),
        DataType::F4E2M1FN => context.float4e2m1fn_type().as_ref(),
        DataType::F6E2M3FN => context.float6e2m3fn_type().as_ref(),
        DataType::F6E3M2FN => context.float6e3m2fn_type().as_ref(),
        DataType::F8E3M4 => context.float8e3m4_type().as_ref(),
        DataType::F8E4M3 => context.float8e4m3_type().as_ref(),
        DataType::F8E4M3FN => context.float8e4m3fn_type().as_ref(),
        DataType::F8E4M3FNUZ => context.float8e4m3fnuz_type().as_ref(),
        DataType::F8E4M3B11FNUZ => context.float8e4m3b11fnuz_type().as_ref(),
        DataType::F8E5M2 => context.float8e5m2_type().as_ref(),
        DataType::F8E5M2FNUZ => context.float8e5m2fnuz_type().as_ref(),
        DataType::F8E8M0FNU => context.float8e8m0fnu_type().as_ref(),
        DataType::C64 => context.complex_type(context.float32_type()).as_ref(),
        DataType::C128 => context.complex_type(context.float64_type()).as_ref(),
    })
}

/// Builds the dense-elements attribute for one traced splat constant.
/// Lowers an arbitrary `f64` factor into a splatted scalar StableHLO constant whose element type
/// matches `output_type`, then broadcasts that scalar to the full output shape. Used by the
/// zero/one synthesis and literal-constant lowering.
pub(crate) fn lower_f64_constant_splat<'b, 'c: 'b, 't: 'c, B, L>(
    factor: f64,
    output_type: &ArrayType,
    output_tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let data_type = output_type.data_type();
    let scalar_tensor_type = context
        .tensor_type(lower_element_type(data_type, context)?, &[], None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(data_type) })?;
    let elements = lower_f64_scalar_elements_attribute(data_type, scalar_tensor_type, factor, context)?;
    let scalar_constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    if output_type.shape().dimensions().is_empty() {
        return Ok(scalar_constant.result(0).expect("stablehlo.constant should return one result").as_ref());
    }
    let broadcast = block.append_operation(stable_hlo::broadcast(
        scalar_constant.result(0).unwrap().as_ref(),
        output_tensor_type,
        &[],
        location,
    )?)?;
    Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref())
}

/// Builds a splatted dense-elements attribute holding `factor` converted to the given `data_type`. Boolean splats
/// hold `factor != 0.0`, integer splats truncate `factor`, and float splats round `factor` to the nearest
/// representable value of the element type (covering every float format that [`lower_element_type`] supports,
/// including the `f8`/`f6`/`f4` families). [`DataType::Zero`] admits only a zero factor, lowered as its `i1` carrier.
/// Integer attributes use the physical element type from [`lower_element_type`], including signless `i1` for `U1`.
pub(crate) fn lower_f64_scalar_elements_attribute<'c, 't>(
    data_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    factor: f64,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    match data_type {
        DataType::Zero if factor == 0.0 => context
            .splatted_dense_attribute_elements_attribute(tensor_type, context.boolean_attribute(false))
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::Zero => Err(LoweringError::UnsupportedDataType { data_type }),
        DataType::Boolean => context
            .splatted_dense_attribute_elements_attribute(tensor_type, context.boolean_attribute(factor != 0.0))
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::I1 | DataType::I2 | DataType::I4 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.signless_integer_type(signed_integer_width(data_type)?),
                        factor as i64,
                    ),
                )
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::U1 | DataType::U2 | DataType::U4 | DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        lower_element_type(data_type, context)?.cast::<IntegerTypeRef>().unwrap(),
                        factor as i64,
                    ),
                )
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::BF16
        | DataType::F16
        | DataType::F32
        | DataType::F64
        | DataType::F4E2M1FN
        | DataType::F6E2M3FN
        | DataType::F6E3M2FN
        | DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E4M3FN
        | DataType::F8E4M3FNUZ
        | DataType::F8E4M3B11FNUZ
        | DataType::F8E5M2
        | DataType::F8E5M2FNUZ
        | DataType::F8E8M0FNU => {
            let element_type = lower_element_type(data_type, context)?
                .cast::<FloatTypeRef>()
                .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type })?;
            context
                .splatted_dense_attribute_elements_attribute(tensor_type, context.float_attribute(element_type, factor))
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::Token | DataType::C64 | DataType::C128 => Err(LoweringError::UnsupportedDataType { data_type }),
    }
}

/// Builds a splatted dense-elements attribute holding `integer_value` converted to the given `data_type`. Integer
/// splats hold the exact `i64` value; every other data type routes through
/// [`lower_f64_scalar_elements_attribute`] after a lossless widening of `integer_value` to `f64`. Integer attributes
/// use the physical element type from [`lower_element_type`], including signless `i1` for `U1`.
pub(crate) fn lower_constant_elements_attribute<'c, 't>(
    data_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    integer_value: i64,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    match data_type {
        DataType::I1 | DataType::I2 | DataType::I4 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.signless_integer_type(signed_integer_width(data_type)?),
                        integer_value,
                    ),
                )
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::U1 | DataType::U2 | DataType::U4 | DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        lower_element_type(data_type, context)?.cast::<IntegerTypeRef>().unwrap(),
                        integer_value,
                    ),
                )
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        _ => lower_f64_scalar_elements_attribute(data_type, tensor_type, integer_value as f64, context),
    }
}

/// Returns the bit width of a signed integer [`DataType`].
pub(crate) fn signed_integer_width(data_type: DataType) -> Result<usize, LoweringError> {
    Ok(match data_type {
        DataType::I1 => 1,
        DataType::I2 => 2,
        DataType::I4 => 4,
        DataType::I8 => 8,
        DataType::I16 => 16,
        DataType::I32 => 32,
        DataType::I64 => 64,
        _ => return Err(LoweringError::UnsupportedDataType { data_type }),
    })
}

/// Returns the bit width of an unsigned integer [`DataType`].
pub(crate) fn unsigned_integer_width(data_type: DataType) -> Result<usize, LoweringError> {
    Ok(match data_type {
        DataType::U1 => 1,
        DataType::U2 => 2,
        DataType::U4 => 4,
        DataType::U8 => 8,
        DataType::U16 => 16,
        DataType::U32 => 32,
        DataType::U64 => 64,
        _ => return Err(LoweringError::UnsupportedDataType { data_type }),
    })
}

/// Lowers a composite array IR type to its physical StableHLO tensor type.
pub(crate) fn lower_array_ir_type<'c, 't, L: Location<'c, 't>>(
    r#type: &ArrayIrType,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
    match r#type {
        ArrayIrType::Array(r#type) => lower_tensor_type(r#type, context, location),
        ArrayIrType::Dimension(_) => lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location),
        ArrayIrType::Reference(_) => Err(LoweringError::UnresolvedReference { construct: r#type.to_string() }),
    }
}

/// Packs scalar first-class dimension operands into the rank-one `i64` shape tensor consumed by dynamic StableHLO
/// shape operations.
pub(crate) fn lower_explicit_shape<'b, 'c: 'b, 't: 'c>(
    extents: &[ValueRef<'b, 'c, 't>],
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let shape_type = context
        .tensor_type(context.signless_integer_type(64), &[MlirSize::Static(extents.len())], None, location)
        .map_err(|_| LoweringError::InvalidTensorType {
            array_type: ArrayType::new(DataType::I64, Shape::new(vec![Dimension::Static(extents.len())])),
        })?;
    if extents.is_empty() {
        let elements = context
            .dense_i64_elements_attribute(shape_type, &[])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::I64 })?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        return Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref());
    }

    let dimensions = extents
        .iter()
        .map(|extent| {
            let reshape = block.append_operation(stable_hlo::reshape(*extent, &[1], location)?)?;
            Ok(reshape.result(0).expect("stablehlo.reshape should return one result").as_ref())
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    if let [dimension] = dimensions.as_slice() {
        return Ok(*dimension);
    }
    let shape = block.append_operation(stable_hlo::concatenate(dimensions.as_slice(), 0, location)?)?;
    Ok(shape.result(0).expect("stablehlo.concatenate should return one result").as_ref())
}

/// Packs scalar first-class dimension operands into the rank-one `i32` shape tensor required by
/// `stablehlo.dynamic_reshape`.
pub(crate) fn lower_explicit_reshape_shape<'b, 'c: 'b, 't: 'c>(
    extents: &[ValueRef<'b, 'c, 't>],
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let scalar_type = context
        .tensor_type(context.signless_integer_type(32), &[], None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I32) })?;
    let shape_type = context
        .tensor_type(context.signless_integer_type(32), &[MlirSize::Static(extents.len())], None, location)
        .map_err(|_| LoweringError::InvalidTensorType {
            array_type: ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(extents.len())])),
        })?;
    if extents.is_empty() {
        let elements = context
            .dense_i32_elements_attribute(shape_type, &[])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::I32 })?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        return Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref());
    }

    let dimensions = extents
        .iter()
        .map(|extent| {
            let extent = block.append_operation(stable_hlo::convert(*extent, scalar_type, location)?)?;
            let extent = extent.result(0).expect("stablehlo.convert should return one result").as_ref();
            let reshape = block.append_operation(stable_hlo::reshape(extent, &[1], location)?)?;
            Ok(reshape.result(0).expect("stablehlo.reshape should return one result").as_ref())
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    if let [dimension] = dimensions.as_slice() {
        return Ok(*dimension);
    }
    let shape = block.append_operation(stable_hlo::concatenate(dimensions.as_slice(), 0, location)?)?;
    Ok(shape.result(0).expect("stablehlo.concatenate should return one result").as_ref())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{ConvertElementTypeOperation, Placeholder};

    use crate::experimental::lowering::tests::{XlaProgramBuilder, dynamic_dimension};
    use crate::experimental::lowering::to_mlir_module_for_plain_program;

    use super::*;

    #[test]
    fn test_lower_signed_one_bit_to_mlir() {
        let mut builder = XlaProgramBuilder::new();
        let input =
            builder.add_input(ArrayType::new(DataType::I1, Shape::new(vec![dynamic_dimension("size", Some(6))])));
        let output = builder
            .add_instruction(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        // Widening retains the physical bound as well as the runtime dimension before recovering the signed value.
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
        module {
          func.func @main(%arg0: tensor<?xi1, #stablehlo.bounds<5>>) -> tensor<?xf32, #stablehlo.bounds<5>> {
            %0 = stablehlo.convert %arg0 : (tensor<?xi1, #stablehlo.bounds<5>>) -> tensor<?xi8, #stablehlo.bounds<5>>
            %1 = stablehlo.negate %0 : tensor<?xi8, #stablehlo.bounds<5>>
            %2 = stablehlo.convert %1 : (tensor<?xi8, #stablehlo.bounds<5>>) -> tensor<?xf32, #stablehlo.bounds<5>>
            return %2 : tensor<?xf32, #stablehlo.bounds<5>>
          }
        }
        "#},
        );
    }

    #[test]
    fn test_reshape_dimension_i64_rejects_out_of_range_values() {
        if usize::MAX <= i64::MAX as usize {
            return;
        }

        assert_eq!(
            reshape_dimension_i64(usize::MAX),
            Err(LoweringError::ReshapeDimensionOutOfRange { value: usize::MAX, bit_width: 64 }),
        );
    }

    #[test]
    fn test_reshape_dimension_i32_rejects_out_of_range_values() {
        let value = i32::MAX as usize + 1;
        assert_eq!(
            reshape_dimension_i32(value),
            Err(LoweringError::ReshapeDimensionOutOfRange { value, bit_width: 32 }),
        );
    }

    #[test]
    fn test_lower_boundary_layout() {
        use ryft_core::TiledLayout;

        let context = MlirContext::new();
        let plain = ArrayType::new_static(DataType::F32, [2, 3]);
        assert!(lower_boundary_layout(&plain, "input", &context).unwrap().is_none());
        let column_major = plain.with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let attribute = lower_boundary_layout(&column_major, "output", &context).unwrap().unwrap();
        assert_eq!(attribute.to_string(), "\"{0,1}\"");
    }

    #[test]
    fn test_stable_hlo_dynamic_dimension_bound_converts_exclusive_bounds() {
        assert_eq!(stable_hlo_dynamic_dimension_bound(&Dimension::Static(4)), None);
        assert_eq!(stable_hlo_dynamic_dimension_bound(&dynamic_dimension("unbounded", None)), None);
        assert_eq!(stable_hlo_dynamic_dimension_bound(&dynamic_dimension("bounded", Some(6))), Some(5));
    }

    #[test]
    fn test_lower_unsigned_constant_splats_use_physical_carrier() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        for data_type in [DataType::U1, DataType::U8] {
            let tensor_type = lower_tensor_type(&ArrayType::scalar(data_type), &context, location).unwrap();
            for integer_value in [0, 1, 2, 3] {
                let expected = match data_type {
                    DataType::U1 => format!("dense<{}> : tensor<i1>", integer_value & 1 != 0),
                    DataType::U8 => format!("dense<{integer_value}> : tensor<ui8>"),
                    _ => unreachable!(),
                };
                assert_eq!(
                    lower_constant_elements_attribute(data_type, tensor_type, integer_value, &context)
                        .unwrap()
                        .to_string(),
                    expected,
                );
                assert_eq!(
                    lower_f64_scalar_elements_attribute(data_type, tensor_type, integer_value as f64 + 0.75, &context,)
                        .unwrap()
                        .to_string(),
                    expected,
                );
            }
        }
    }
}
