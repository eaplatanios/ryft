use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, CaptureReference, ComparisonDirection, DataType, Dimension, DimensionError, DimensionValue, Layout,
    Memory, ProgramError, Shape, Type as CoreType,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::CustomCallApiVersion;
use ryft_mlir::{
    Attribute, AttributeRef, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef, FloatTypeRef,
    IntegerTypeRef, Location, LocationRef, Operation as MlirOperation, Size as MlirSize, TensorTypeRef,
    Type as MlirType, TypeRef, Value as MlirValue, ValueRef,
};

use crate::devices::MemoryExtension;
use crate::lowering::states::CollectiveLoweringState;
use crate::lowering::{LoweringError, MlirLowerableValue};
use crate::{ArrayConstant, ArrayError, Constant, Error, ToMlir, ToPjrt, Type};

// TODO(eaplatanios): Review this module.

impl ToMlir for ComparisonDirection {
    type Output<'c, 't: 'c> = stable_hlo::ComparisonDirection;

    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(&self, location: L) -> Result<Self::Output<'c, 't>, ryft_mlir::Error> {
        let _ = location;
        Ok(match self {
            ComparisonDirection::Equal => stable_hlo::ComparisonDirection::Equal,
            ComparisonDirection::NotEqual => stable_hlo::ComparisonDirection::NotEqual,
            ComparisonDirection::LessThan => stable_hlo::ComparisonDirection::LessThan,
            ComparisonDirection::LessThanOrEqual => stable_hlo::ComparisonDirection::LessThanOrEqual,
            ComparisonDirection::GreaterThan => stable_hlo::ComparisonDirection::GreaterThan,
            ComparisonDirection::GreaterThanOrEqual => stable_hlo::ComparisonDirection::GreaterThanOrEqual,
        })
    }
}

/// Lowers one comparison of two values in the provided `direction` to a single `stablehlo.compare`. This serves
/// [`CompareOperation`](ryft_core::CompareOperation) lowerings, scalar dimension comparisons, and the comparisons inside of combiner and comparator
/// bodies. Both operands must already share one tensor type, because this function neither promotes nor broadcasts
/// them, and the result has their shape with a Boolean element type. The comparison type (i.e., `FLOAT`, `SIGNED`, or
/// `UNSIGNED`) follows the element type of the left-hand side input, as [`comparison_type_for_mlir_type`] describes.
pub fn lower_compare_to_mlir<'b, 'c: 'b, 't: 'c>(
    direction: ComparisonDirection,
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let direction = direction.to_mlir(location)?;
    let left_type = left.r#type()?;
    let comparison_type = comparison_type_for_mlir_type(left_type)?;
    let result = block.append_operation(stable_hlo::compare(left, right, direction, comparison_type, location)?)?;
    Ok(result.result(0).unwrap().as_ref())
}

/// Validates that a program compiled for `target_platform` can take its arguments from `memory` and return its
/// results in it.
///
/// XLA's CUDA compiler does not place values in unpinned host memory, although its plugin can hold buffers there. It
/// reads an argument that is declared in that memory from device memory, and it returns a result that is declared in
/// that memory in device memory or, after a transfer, in pinned host memory. A program with such a boundary would
/// therefore reject its actual arguments and mislabel its results, so it is rejected here instead. Values inside a
/// program are not affected, because the compiler treats pinned and unpinned host memory alike there.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedBoundaryMemory`] if `memory` is unpinned host memory and `target_platform` is
/// CUDA.
pub(crate) fn validate_boundary_memory(memory: Memory, target_platform: Option<&str>) -> Result<(), Error> {
    if memory == (Memory::Host { pinned: false }) && target_platform == Some("cuda") {
        return Err(LoweringError::UnsupportedBoundaryMemory { memory, platform: "cuda".to_string() }.into());
    }
    Ok(())
}

/// Applies the non-default memory placement declared by `output_type` to `value`.
pub fn annotate_output_memory<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    value: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    if output_type.memory() == Memory::Device {
        return Ok(value);
    }
    Ok(lower_transfer_to_memory(output_type.memory(), &[value], block, context, location)?[0])
}

/// Lowers one staged [`TransferToMemoryOperation`](ryft_core::TransferToMemoryOperation) to the `stablehlo.custom_call @annotate_device_placement` annotation
/// that the XLA `ConvertMemoryPlacementToInternalAnnotations` and
/// [`HostOffloader`](https://openxla.org/xla/tools_and_passes/host_offloading) passes legalize into copies annotated
/// with memory spaces. The annotation uses API version 1, `has_side_effect = true`, an empty `backend_config`, and
/// the destination kind in `_xla_buffer_placement` inside `mhlo.frontend_attributes`.
///
/// These constraints are retained on CPU as well, because reference arrays only retag their types, but native arrays
/// must honor the requested physical placement on every platform. The current XLA offloading pass treats pinned and
/// unpinned host annotations alike internally, whereas function boundary attributes and PJRT memory validation retain
/// the distinction at runtime boundaries. A plugin must expose the requested memory kind for native allocation.
///
/// Placement does not affect the MLIR tensor type, so the result type is the operand's type unchanged. Identity
/// transfers (i.e., transfers whose destination equals the operand's current space) still lower to the annotation,
/// because placement round trips are meaningful to `HostOffloader` and must not be optimized away here.
pub fn lower_transfer_to_memory<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    destination: Memory,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    check_count!("input", input_values, 1, ProgramError);
    let empty_backend_configuration = context.string_attribute("");
    let mut operation = stable_hlo::custom_call(
        input_values,
        "annotate_device_placement",
        true,
        Some(empty_backend_configuration.as_ref()),
        CustomCallApiVersion::Original,
        &[],
        None,
        &[],
        None,
        &[input_values[0].r#type()?],
        location,
    )?;
    operation.set_discardable_attribute(
        "mhlo.frontend_attributes",
        context.dictionary_attribute(&[context.named_attribute(
            context.identifier("_xla_buffer_placement"),
            context.string_attribute(destination.placement_kind()),
        )]),
    );
    let operation = block.append_operation(operation)?;
    Ok(vec![operation.result(0).unwrap().as_ref()])
}

/// Widens a logical signed one-bit integer to `i8`, recovering `-1` from StableHLO's predicate carrier.
pub fn lower_signed_one_bit_to_mlir<'b, 'c: 'b, 't: 'c, B, L>(
    input: ValueRef<'b, 'c, 't>,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, Error>
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
pub fn lower_runtime_dimension_size_i64<'b, 'c: 'b, 't: 'c>(
    input: ValueRef<'b, 'c, 't>,
    axis: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let size = block.append_operation(stable_hlo::get_dimension_size(input, axis, location)?)?;
    let size = size.result(0).unwrap().as_ref();
    let i64_scalar = context.tensor_type(context.signless_integer_type(64), &[], None, location)?;
    let size = block.append_operation(stable_hlo::convert(size, i64_scalar, location)?)?;
    Ok(size.result(0).unwrap().as_ref())
}

/// Returns the statically shaped carrier type of one bounded dynamic array type, in which every dynamic dimension is
/// replaced by its physical bound (i.e., the largest extent that its exclusive upper bound admits).
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] when a dynamic dimension has no finite positive upper bound.
pub fn physical_bound_type(r#type: &ArrayType) -> Result<ArrayType, Error> {
    let dimensions = r#type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
            Dimension::Dynamic(variable) => {
                variable.bounds().upper().and_then(|upper| upper.checked_sub(1)).map(Dimension::Static).ok_or_else(
                    || {
                        Error::Lowering(LoweringError::UnsupportedOperation {
                            operation: "physical_bound_type".to_string(),
                            reason: format!("dynamic dimension `{variable}` needs a finite positive physical bound"),
                        })
                    },
                )
            }
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(r#type.clone().with_shape(Shape::new(dimensions)))
}

/// Materializes the full physical buffer of one bounded dynamic value. The original runtime sizes are retained before
/// each dynamic axis is widened to its physical bound, and every newly exposed position is replaced with
/// `padding_value`.
pub fn lower_physical_bound_value<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    padding_value: f64,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    lower_physical_bound_value_with_padding(value, r#type, padding_value, None, block, context, location)
}

/// Exposes the bounded storage of one bounded dynamic value like [`lower_physical_bound_value`], masking the inactive
/// positions with `scalar_padding`, which is an exact scalar value, when it is provided, and with the numeric constant
/// `padding_value` otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_physical_bound_value_with_padding<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    padding_value: f64,
    scalar_padding: Option<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
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
            Ok(Some(size.result(0).unwrap().as_ref()))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let physical_type = physical_bound_type(r#type)?;
    let size_type = lower_tensor_type(&ArrayType::scalar(DataType::I32), context, location)?;
    let mut data = value;
    let mut physicalized_dimensions = r#type.shape().dimensions().to_vec();
    for (axis, dimension) in r#type.shape().dimensions().iter().enumerate() {
        if matches!(dimension, Dimension::Static(_)) {
            continue;
        }
        let physical_extent = physical_type.shape().dimensions()[axis].value().unwrap();
        let extent = dimension_extent_i32(physical_extent)?;
        let elements = context.dense_i32_elements_attribute(size_type, &[extent])?;
        let size = block.append_operation(stable_hlo::constant(elements, location)?)?;
        let size = size.result(0).unwrap().as_ref();
        physicalized_dimensions[axis] = Dimension::Static(physical_extent);
        let physicalized_type = r#type.clone().with_shape(Shape::new(physicalized_dimensions.clone()));
        let physicalized = block.append_operation(stable_hlo::set_dimension_size(
            data,
            size,
            lower_tensor_type(&physicalized_type, context, location)?,
            axis,
            location,
        )?)?;
        data = physicalized.result(0).unwrap().as_ref();
    }
    let index_type = ArrayType::new(DataType::I32, physical_type.shape().clone());
    let index_tensor_type = lower_tensor_type(&index_type, context, location)?;
    let mut in_bounds = None;
    for (axis, dimension) in r#type.shape().dimensions().iter().enumerate() {
        if matches!(dimension, Dimension::Static(_)) {
            continue;
        }
        let indices = block.append_operation(stable_hlo::iota(index_tensor_type, axis, location)?)?;
        let indices = indices.result(0).unwrap().as_ref();
        let size = runtime_sizes[axis].unwrap();
        let size = block.append_operation(stable_hlo::broadcast(size, index_tensor_type, &[], location)?)?;
        let size = size.result(0).unwrap().as_ref();
        let axis_in_bounds = lower_compare_to_mlir(ComparisonDirection::LessThan, indices, size, block, location)?;
        in_bounds = Some(match in_bounds {
            None => axis_in_bounds,
            Some(in_bounds) => block
                .append_operation(stable_hlo::and(in_bounds, axis_in_bounds, location)?)?
                .result(0)
                .unwrap()
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
        // Padding belongs to the complex element type too, so construct its real and imaginary components separately.
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
    Ok(masked.result(0).unwrap().as_ref())
}

/// Restores the dynamic dimensions of `output_type` on a value of its [`physical_bound_type`]. Each entry in `sources`
/// identifies the value and axis that define the corresponding output dimension.
pub fn lower_restore_dynamic_dimensions<'b, 'c: 'b, 't: 'c>(
    mut value: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    sources: &[(ValueRef<'b, 'c, 't>, usize)],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
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
        let size = size.result(0).unwrap().as_ref();
        dimensions[axis] = dimension.clone();
        let refined_type = output_type.clone().with_shape(Shape::new(dimensions.clone()));
        let refined = block.append_operation(stable_hlo::set_dimension_size(
            value,
            size,
            lower_tensor_type(&refined_type, context, location)?,
            axis,
            location,
        )?)?;
        value = refined.result(0).unwrap().as_ref();
    }
    Ok(value)
}

/// Exposes the physical storage of one bounded dynamic value as a statically shaped value of its
/// [`physical_bound_type`], by setting the runtime size of every dynamic axis to its physical bound. Unlike
/// [`lower_physical_bound_value`], this does not mask the positions beyond the runtime sizes, so it is only suitable
/// for callers that discard those positions again (e.g., by slicing to a static extent or by restoring the runtime
/// sizes).
pub fn lower_physical_storage<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
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
            i64::from(dimension_extent_i32(bound)?),
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
        value = exposed.result(0).unwrap().as_ref();
    }
    Ok(value)
}

/// Converts one dimension extent to StableHLO's signed 64-bit shape representation.
pub fn dimension_extent_i64(value: usize) -> Result<i64, Error> {
    i64::try_from(value)
        .map_err(|_| DimensionError::ExtentExceedsBackendWidth { value, maximum: i64::MAX as usize }.into())
}

/// Converts one dimension extent to the signed 32-bit representation required by dynamic StableHLO shape operations.
pub fn dimension_extent_i32(value: usize) -> Result<i32, Error> {
    i32::try_from(value)
        .map_err(|_| DimensionError::ExtentExceedsBackendWidth { value, maximum: i32::MAX as usize }.into())
}

/// Lowers static start indices to scalar `i64` StableHLO constants, as consumed by the index operands of
/// `stablehlo.dynamic_update_slice` when lowering the statically indexed `update_slice` operation, because StableHLO
/// has no statically indexed update operation.
pub fn lower_static_index_constants<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    start_indices: &[usize],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let element_type = lower_element_type(DataType::I64, context)?;
    let tensor_type = context.tensor_type(element_type, &[], None, location)?;
    start_indices
        .iter()
        .map(|index| {
            let elements = lower_constant_elements_attribute(DataType::I64, tensor_type, *index as i64, context)?;
            let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
            Ok(constant.result(0).unwrap().as_ref())
        })
        .collect()
}

/// Lowers `integer_value` as a constant of the single type in `output_types`, without applying its memory placement
/// (refer to [`lower_constant_output`] for the placed variant). Scalars lower to one `stablehlo.constant`, while
/// shaped outputs and complex values are built from scalar constants and broadcast to the output shape.
///
/// # Errors
///
/// Returns an [`Error`] when `output_types` does not contain exactly one type, or when the type cannot be lowered or
/// represent `integer_value`.
pub fn lower_unplaced_constant_output<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    output_types: &[ArrayType],
    integer_value: i64,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    check_count!("output", output_types, 1, ProgramError);
    let output_type = &output_types[0];
    let tensor_type = lower_tensor_type(output_type, context, location)?;
    let data_type = output_type.data_type();

    // Complex constants are composed as `value + 0i` from two real part constants through `stablehlo.complex`, so that
    // the dense elements attribute helpers stay real valued, because MLIR has no complex scalar attribute to splat.
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
            real.result(0).unwrap().as_ref(),
            imaginary.result(0).unwrap().as_ref(),
            location,
        )?)?;
        let complex = complex.result(0).unwrap().as_ref();
        if output_type.shape().dimensions().is_empty() {
            return Ok(vec![complex]);
        }
        let broadcast = block.append_operation(stable_hlo::broadcast(complex, tensor_type, &[], location)?)?;
        return Ok(vec![broadcast.result(0).unwrap().as_ref()]);
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
        return Ok(vec![broadcast.result(0).unwrap().as_ref()]);
    }
    let elements = lower_constant_elements_attribute(data_type, tensor_type, integer_value, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    Ok(vec![constant.result(0).unwrap().as_ref()])
}

/// Lowers one constant with an integer value like [`lower_unplaced_constant_output`] and applies the declared memory
/// placement of its output type to the final value when that placement is not the device memory.
pub fn lower_constant_output<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    output_types: &[ArrayType],
    integer_value: i64,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let values = lower_unplaced_constant_output(output_types, integer_value, block, context, location)?;
    check_count!("output", output_types, 1, ProgramError);
    Ok(vec![annotate_output_memory(values[0], &output_types[0], block, context, location)?])
}

/// Returns the static dimensions for one tensor type.
pub fn static_dimensions(array_type: &ArrayType) -> Result<Vec<usize>, Error> {
    array_type
        .shape()
        .dimensions()
        .iter()
        .map(|size| match size {
            Dimension::Static(value) => Ok(*value),
            Dimension::Dynamic(_) => Err(Error::from(ArrayError::DynamicShape { shape: array_type.shape().clone() })),
        })
        .collect()
}

/// Lowers one concrete traced value to a StableHLO constant operation and applies its declared memory placement.
pub(crate) fn lower_literal_value<'b, 'c: 'b, 't: 'c, B, V, L>(
    value: &V,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, Error>
where
    B: Block<'b, 'c, 't>,
    V: MlirLowerableValue,
    L: Copy + Location<'c, 't>,
{
    let value_type = value.r#type();
    let tensor_type = lower_tensor_type(&value_type, context, location)?;
    let elements = value.to_dense_elements_attribute(tensor_type, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    let constant = constant.result(0).unwrap().as_ref();
    annotate_output_memory(constant, &value_type, block, context, location)
}

/// Lowers one captured constant reference by forwarding its runtime captured value.
pub(crate) fn lower_captured_constant<'b, 'c: 'b, 't: 'c, T: CoreType>(
    value: &CaptureReference<T>,
    captured_values: &[ValueRef<'b, 'c, 't>],
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    captured_values
        .get(value.index())
        .copied()
        .ok_or(Error::Lowering(LoweringError::MissingCapturedConstant { index: value.index() }))
}

/// Lowers one first-class dimension extent to the scalar `i64` [`stablehlo.constant`](stable_hlo::constant) that
/// represents a dimension in StableHLO. Immediate [`Constant::Dimension`] atoms and staged
/// [`DimensionOperation::Constant`](ryft_core::DimensionOperation::Constant) instructions must agree on this
/// representation, and so both lowering paths route through this function.
pub(crate) fn lower_dimension_extent<'b, 'c: 'b, 't: 'c, B, L>(
    value: &DimensionValue,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, Error>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
    let extent = dimension_extent_i64(value.extent())?;
    let elements = lower_constant_elements_attribute(DataType::I64, tensor_type, extent, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    Ok(constant.result(0).unwrap().as_ref())
}

/// Lowers a traced constant atom to a StableHLO value and returns it. A captured constant forwards the hidden capture
/// argument that carries its runtime value, and a session constant forwards the hidden entry argument that
/// `collective_state` assigns to it, while an immediate dimension extent is materialized in place. The latter is
/// exactly what makes extents usable inside a `shard_map` manual region, where no capture table is reachable.
pub(crate) fn lower_constant<'b, 'c: 'b, 't: 'c, B, L>(
    value: &Constant,
    captured_values: &[ValueRef<'b, 'c, 't>],
    collective_state: &CollectiveLoweringState,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, Error>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    match value {
        Constant::Captured(value) => lower_captured_constant(value, captured_values),
        Constant::Session(value) => collective_state.lower_session_constant(value, captured_values),
        Constant::Dimension(value) => lower_dimension_extent(value, block, context, location),
        Constant::Boolean(value) => lower_literal_value(&ArrayConstant::Boolean(*value), block, context, location),
    }
}

/// Lowers a scalar `u64` constant. It is shared by the group interval arithmetic of ragged dots, the device shard and
/// batch item arithmetic of `parallel_ragged_all_to_all`, and the coordinate arithmetic of
/// [`AxisIndexOperation`](ryft_core::AxisIndexOperation).
pub fn lower_u64_scalar_constant<'b, 'c: 'b, 't: 'c>(
    value: u64,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let output_type = ArrayType::scalar(DataType::U64);
    let output_tensor_type = lower_tensor_type(&output_type, context, location)?;
    lower_u64_constant_splat(value, &output_type, output_tensor_type, block, context, location)
}

/// Encodes the requested layout of an executable boundary value using the `mhlo.layout_mode` convention of PJRT, where
/// `kind` names the boundary side (i.e., `"input"` or `"output"`). Internal `LayoutConstraint` custom calls (refer to
/// [`lower_layout_constraint`](crate::operations::constants::lower_layout_constraint)) alone do not constrain the
/// parameter or result ABI.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] for strided layouts and for bounded dynamic outputs whose layout is
/// not row-major.
pub(crate) fn lower_boundary_layout<'c, 't>(
    array_type: &ArrayType,
    kind: &str,
    context: &'c MlirContext<'t>,
) -> Result<Option<AttributeRef<'c, 't>>, Error> {
    let Some(layout) = array_type.layout() else {
        return Ok(None);
    };
    let Layout::Tiled(tiled) = layout else {
        return Err(Error::Lowering(LoweringError::UnsupportedOperation {
            operation: "lower_boundary_layout".to_string(),
            reason: format!("strided {kind} layout `{layout}`"),
        }));
    };
    // The pinned XLA SliceToDynamic emitter assumes row-major source/result layouts. Requesting a different
    // dynamic result layout can abort compilation rather than return an error.
    if kind == "output"
        && array_type.static_shape().is_none()
        && !tiled.minor_to_major().iter().copied().eq((0..array_type.rank()).rev())
    {
        return Err(Error::Lowering(LoweringError::UnsupportedOperation {
            operation: "lower_boundary_layout".to_string(),
            reason: format!("non-row-major bounded-dynamic output layout `{layout}`"),
        }));
    }
    Ok(Some(
        context
            .string_attribute(tiled.to_pjrt().map_err(ProgramError::custom)?.to_string().as_str())
            .as_ref(),
    ))
}

/// Lowers an [`ArrayType`] to a typed MLIR tensor type.
pub fn lower_tensor_type<'c, 't, L: Location<'c, 't>>(
    array_type: &ArrayType,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, Error> {
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
    context.tensor_type(element_type, dimensions.as_slice(), encoding, location).map_err(Error::from)
}

/// Converts a Ryft exclusive dynamic dimension bound to StableHLO's inclusive maximum dimension size.
pub fn stable_hlo_dynamic_dimension_bound(size: &Dimension) -> Option<usize> {
    match size {
        Dimension::Static(_) => None,
        Dimension::Dynamic(variable) => variable.bounds().upper().and_then(|upper| upper.checked_sub(1)),
    }
}

/// Lowers one [`DataType`] to the corresponding MLIR element type.
pub fn lower_element_type<'c, 't>(data_type: DataType, context: &'c MlirContext<'t>) -> Result<TypeRef<'c, 't>, Error> {
    Ok(match data_type {
        DataType::Token => {
            return Err(Error::Lowering(LoweringError::UnsupportedElementType {
                data_type,
                context: "tensor element representation".to_string(),
            }));
        }
        // StableHLO has no element type that carries zero information. Use an `i1` tensor as an unobservable physical
        // carrier while retaining the logical `DataType::Zero` in the lowered and compiled metadata of Ryft.
        DataType::Zero => context.signless_integer_type(1).as_ref(),
        DataType::Boolean => context.signless_integer_type(1).as_ref(),
        DataType::I1 => context.signless_integer_type(1).as_ref(),
        DataType::I2 => context.signless_integer_type(2).as_ref(),
        DataType::I4 => context.signless_integer_type(4).as_ref(),
        DataType::I8 => context.signless_integer_type(8).as_ref(),
        DataType::I16 => context.signless_integer_type(16).as_ref(),
        DataType::I32 => context.signless_integer_type(32).as_ref(),
        DataType::I64 => context.signless_integer_type(64).as_ref(),
        // StableHLO admits signless `i1` as its Boolean and one-bit carrier but rejects `ui1`. The unsigned semantics
        // of `U1` remain selected by Ryft operation metadata (e.g., comparison direction), while its sole bit is
        // unchanged.
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

/// Lowers an arbitrary `f64` factor into a splatted scalar StableHLO constant whose element type matches
/// `output_type`, then broadcasts that scalar to the full output shape. It is used when synthesizing zeros, ones, and
/// padding values and when lowering literal constants.
pub fn lower_f64_constant_splat<'b, 'c: 'b, 't: 'c, B, L>(
    factor: f64,
    output_type: &ArrayType,
    output_tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, Error>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let data_type = output_type.data_type();
    let scalar_tensor_type = context.tensor_type(lower_element_type(data_type, context)?, &[], None, location)?;
    let elements = lower_f64_scalar_elements_attribute(data_type, scalar_tensor_type, factor, context)?;
    let scalar_constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    if output_type.shape().dimensions().is_empty() {
        return Ok(scalar_constant.result(0).unwrap().as_ref());
    }
    let broadcast = block.append_operation(stable_hlo::broadcast(
        scalar_constant.result(0).unwrap().as_ref(),
        output_tensor_type,
        &[],
        location,
    )?)?;
    Ok(broadcast.result(0).unwrap().as_ref())
}

/// Builds a splatted dense elements attribute holding `factor` converted to the given `data_type`. Boolean splats
/// hold `factor != 0.0`, integer splats truncate `factor`, and float splats round `factor` to the nearest
/// representable value of the element type, for every float format that [`lower_element_type`] supports (e.g., the
/// `f8`, `f6`, and `f4` families). [`DataType::Zero`] admits only a zero factor, lowered as its `i1` carrier.
/// Integer attributes use the physical element type from [`lower_element_type`], including signless `i1` for `U1`.
pub(crate) fn lower_f64_scalar_elements_attribute<'c, 't>(
    data_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    factor: f64,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
    match data_type {
        DataType::Zero if factor == 0.0 => context
            .splatted_dense_attribute_elements_attribute(tensor_type, context.boolean_attribute(false))
            .map_err(Error::from),
        DataType::Zero => Err(Error::Lowering(LoweringError::UnsupportedElementType {
            data_type,
            context: "floating-point splat constant".to_string(),
        })),
        DataType::Boolean => context
            .splatted_dense_attribute_elements_attribute(tensor_type, context.boolean_attribute(factor != 0.0))
            .map_err(Error::from),
        DataType::I1 | DataType::I2 | DataType::I4 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.signless_integer_type(signed_integer_width(data_type)?),
                        factor as i64,
                    ),
                )
                .map_err(Error::from)
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
                .map_err(Error::from)
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
            let element_type = lower_element_type(data_type, context)?.cast::<FloatTypeRef>().unwrap();
            context
                .splatted_dense_attribute_elements_attribute(tensor_type, context.float_attribute(element_type, factor))
                .map_err(Error::from)
        }
        DataType::Token | DataType::C64 | DataType::C128 => {
            Err(Error::Lowering(LoweringError::UnsupportedElementType {
                data_type,
                context: "floating-point splat constant".to_string(),
            }))
        }
    }
}

/// Builds a splatted dense elements attribute holding `integer_value` converted to the given `data_type`. Integer
/// splats hold the exact `i64` value, and every other data type routes through
/// [`lower_f64_scalar_elements_attribute`] after a widening of `integer_value` to `f64`. Integer attributes
/// use the physical element type from [`lower_element_type`], including signless `i1` for `U1`.
pub fn lower_constant_elements_attribute<'c, 't>(
    data_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    integer_value: i64,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
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
                .map_err(Error::from)
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
                .map_err(Error::from)
        }
        _ => lower_f64_scalar_elements_attribute(data_type, tensor_type, integer_value as f64, context),
    }
}

/// Returns the bit width of a signed integer [`DataType`].
pub(crate) fn signed_integer_width(data_type: DataType) -> Result<usize, Error> {
    Ok(match data_type {
        DataType::I1 => 1,
        DataType::I2 => 2,
        DataType::I4 => 4,
        DataType::I8 => 8,
        DataType::I16 => 16,
        DataType::I32 => 32,
        DataType::I64 => 64,
        _ => {
            return Err(Error::Lowering(LoweringError::UnsupportedElementType {
                data_type,
                context: "signed integer width".to_string(),
            }));
        }
    })
}

/// Returns the bit width of an unsigned integer [`DataType`].
pub(crate) fn unsigned_integer_width(data_type: DataType) -> Result<usize, Error> {
    Ok(match data_type {
        DataType::U1 => 1,
        DataType::U2 => 2,
        DataType::U4 => 4,
        DataType::U8 => 8,
        DataType::U16 => 16,
        DataType::U32 => 32,
        DataType::U64 => 64,
        _ => {
            return Err(Error::Lowering(LoweringError::UnsupportedElementType {
                data_type,
                context: "unsigned integer width".to_string(),
            }));
        }
    })
}

/// Lowers an array IR type to its physical StableHLO tensor type.
pub fn lower_array_ir_type<'c, 't, L: Location<'c, 't>>(
    r#type: &Type,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, Error> {
    match r#type {
        Type::Array(r#type) => lower_tensor_type(r#type, context, location),
        Type::Dimension(_) => lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location),
        Type::Reference(_) => {
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct: r#type.to_string() }))
        }
    }
}

/// Packs scalar first-class dimension operands into the rank-1 `i64` shape tensor consumed by dynamic StableHLO
/// shape operations.
pub fn lower_explicit_shape<'b, 'c: 'b, 't: 'c>(
    extents: &[ValueRef<'b, 'c, 't>],
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let shape_type =
        context.tensor_type(context.signless_integer_type(64), &[MlirSize::Static(extents.len())], None, location)?;
    if extents.is_empty() {
        let elements = context.dense_i64_elements_attribute(shape_type, &[])?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        return Ok(constant.result(0).unwrap().as_ref());
    }

    let dimensions = extents
        .iter()
        .map(|extent| {
            let reshape = block.append_operation(stable_hlo::reshape(*extent, &[1], location)?)?;
            Ok(reshape.result(0).unwrap().as_ref())
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if let [dimension] = dimensions.as_slice() {
        return Ok(*dimension);
    }
    let shape = block.append_operation(stable_hlo::concatenate(dimensions.as_slice(), 0, location)?)?;
    Ok(shape.result(0).unwrap().as_ref())
}

/// Packs scalar first-class dimension operands into the rank-1 `i32` shape tensor required by
/// `stablehlo.dynamic_reshape`.
pub(crate) fn lower_explicit_reshape_shape<'b, 'c: 'b, 't: 'c>(
    extents: &[ValueRef<'b, 'c, 't>],
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let scalar_type = context.tensor_type(context.signless_integer_type(32), &[], None, location)?;
    let shape_type =
        context.tensor_type(context.signless_integer_type(32), &[MlirSize::Static(extents.len())], None, location)?;
    if extents.is_empty() {
        let elements = context.dense_i32_elements_attribute(shape_type, &[])?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        return Ok(constant.result(0).unwrap().as_ref());
    }

    let dimensions = extents
        .iter()
        .map(|extent| {
            let extent = block.append_operation(stable_hlo::convert(*extent, scalar_type, location)?)?;
            let extent = extent.result(0).unwrap().as_ref();
            let reshape = block.append_operation(stable_hlo::reshape(extent, &[1], location)?)?;
            Ok(reshape.result(0).unwrap().as_ref())
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if let [dimension] = dimensions.as_slice() {
        return Ok(*dimension);
    }
    let shape = block.append_operation(stable_hlo::concatenate(dimensions.as_slice(), 0, location)?)?;
    Ok(shape.result(0).unwrap().as_ref())
}

/// Lowers an exact `u64` value as a splatted constant of `output_type`.
fn lower_u64_constant_splat<'b, 'c: 'b, 't: 'c, B, L>(
    value: u64,
    output_type: &ArrayType,
    output_tensor_type: TensorTypeRef<'c, 't>,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, Error>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let scalar_tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::U64), context, location)?;
    let attribute = context.dense_u64_elements_attribute(scalar_tensor_type, &[value])?;
    let constant = block.append_operation(stable_hlo::constant(attribute, location)?)?;
    let constant = constant.result(0).unwrap().as_ref();
    if output_type.shape().dimensions().is_empty() {
        return Ok(constant);
    }
    let broadcast = block.append_operation(stable_hlo::broadcast(constant, output_tensor_type, &[], location)?)?;
    Ok(broadcast.result(0).unwrap().as_ref())
}

/// Picks the right StableHLO comparison semantic based on the element type of an MLIR value.
///
/// Tensor values are unwrapped to their element type, while scalar types that are not tensors are inspected directly.
/// Floating-point types route to [`stable_hlo::ComparisonType::Float`], explicitly unsigned integers and one-bit
/// predicate carriers route to [`stable_hlo::ComparisonType::Unsigned`], and other integers route to
/// [`stable_hlo::ComparisonType::Signed`]. Every other element type (e.g., complex) routes to
/// [`stable_hlo::ComparisonType::Float`]. Logical signed one-bit inputs must be widened before calling this function,
/// because their MLIR carrier alone cannot distinguish them from Boolean inputs.
fn comparison_type_for_mlir_type(r#type: TypeRef) -> Result<stable_hlo::ComparisonType, Error> {
    let element_type = if let Some(tensor) = r#type.cast::<TensorTypeRef>() { tensor.element_type()? } else { r#type };
    if element_type.is::<FloatTypeRef>() {
        return Ok(stable_hlo::ComparisonType::Float);
    }
    if let Some(integer) = element_type.cast::<IntegerTypeRef>() {
        if integer.is_unsigned() || integer.bit_width() == 1 {
            return Ok(stable_hlo::ComparisonType::Unsigned);
        }
        return Ok(stable_hlo::ComparisonType::Signed);
    }
    // Other element types compare as floating-point values, which matches StableHLO's lenient handling of numeric
    // types that are neither integers nor floating-point types (e.g., complex types).
    Ok(stable_hlo::ComparisonType::Float)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ConvertElementTypeOperation, DimensionBounds, DimensionVariable, Error as CoreError, Placeholder,
        ReferenceType, TiledLayout, ZeroOperation,
    };

    use crate::lowering::tests::{
        ProgramBuilder, dynamic_dimension, test_vector_type, to_mlir_module_for_plain_program,
    };

    use super::*;

    #[test]
    fn test_comparison_direction_to_mlir() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        for (direction, expected) in [
            (ComparisonDirection::Equal, stable_hlo::ComparisonDirection::Equal),
            (ComparisonDirection::NotEqual, stable_hlo::ComparisonDirection::NotEqual),
            (ComparisonDirection::LessThan, stable_hlo::ComparisonDirection::LessThan),
            (ComparisonDirection::LessThanOrEqual, stable_hlo::ComparisonDirection::LessThanOrEqual),
            (ComparisonDirection::GreaterThan, stable_hlo::ComparisonDirection::GreaterThan),
            (ComparisonDirection::GreaterThanOrEqual, stable_hlo::ComparisonDirection::GreaterThanOrEqual),
        ] {
            assert_eq!(direction.to_mlir(location), Ok(expected));
        }
    }

    #[test]
    fn test_validate_boundary_memory() {
        // Only unpinned host memory at the boundary of a program compiled for CUDA is rejected.
        for memory in [Memory::Device, Memory::Host { pinned: true }, Memory::Host { pinned: false }] {
            assert!(validate_boundary_memory(memory, None).is_ok());
            assert!(validate_boundary_memory(memory, Some("cpu")).is_ok());
        }
        assert!(validate_boundary_memory(Memory::Device, Some("cuda")).is_ok());
        assert!(validate_boundary_memory(Memory::Host { pinned: true }, Some("cuda")).is_ok());
        let error = validate_boundary_memory(Memory::Host { pinned: false }, Some("cuda")).unwrap_err();
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::UnsupportedBoundaryMemory { memory, platform })
                if *memory == (Memory::Host { pinned: false }) && platform == "cuda",
        ));
        assert_eq!(
            error.to_string(),
            "programs compiled for platform `cuda` cannot take or return values in `Host[Unpinned]` memory",
        );
    }

    #[test]
    fn test_annotate_output_memory() {
        // Device outputs need no annotation, while host outputs gain one `annotate_device_placement` custom call in
        // addition to their `mhlo.memory_kind` signature attribute.
        let zero_program = |memory| {
            let mut builder = ProgramBuilder::new();
            let output_type = test_vector_type(4).with_memory(memory);
            let output =
                builder.add_instruction(ZeroOperation::new(output_type), Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], Vec::new(), vec![Placeholder])
                .unwrap()
        };
        assert_eq!(
            to_mlir_module_for_plain_program(&zero_program(Memory::Device), "main").unwrap(),
            indoc! {r#"
            module {
              func.func @main() -> tensor<4xf32> {
                %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
                return %0 : tensor<4xf32>
              }
            }
        "#},
        );
        for (memory, kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let expected = indoc! {r#"
                module {
                  func.func @main() -> (tensor<4xf32> {mhlo.memory_kind = "KIND"}) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
                    %1 = stablehlo.custom_call @annotate_device_placement(%0) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "KIND"}} : (tensor<4xf32>) -> tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#}
            .replace("KIND", kind);
            assert_eq!(to_mlir_module_for_plain_program(&zero_program(memory), "main").unwrap(), expected);
        }
    }

    #[test]
    fn test_lower_signed_one_bit_to_mlir() {
        let mut builder = ProgramBuilder::new();
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
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
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
    fn test_physical_bound_type() {
        // Every bounded dynamic dimension is replaced by the largest extent that its exclusive upper bound admits.
        let bounded =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(6)), Dimension::Static(3)]));
        assert_eq!(physical_bound_type(&bounded).unwrap(), ArrayType::new_static(DataType::F32, [5, 3]));

        let unbounded = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", None)]));
        assert!(matches!(
            physical_bound_type(&unbounded),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == "physical_bound_type"
                    && reason == "dynamic dimension `rows` needs a finite positive physical bound",
        ));
    }

    #[test]
    fn test_dimension_extent_i64() {
        if usize::MAX <= i64::MAX as usize {
            return;
        }

        let error = dimension_extent_i64(usize::MAX).unwrap_err();
        let Error::Core(core_error) = error else { unreachable!() };
        let CoreError::Type(error) = *core_error else { unreachable!() };
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::ExtentExceedsBackendWidth { value: usize::MAX, maximum: i64::MAX as usize }),
        );
    }

    #[test]
    fn test_dimension_extent_i32() {
        let value = i32::MAX as usize + 1;
        let error = dimension_extent_i32(value).unwrap_err();
        let Error::Core(core_error) = error else { unreachable!() };
        let CoreError::Type(error) = *core_error else { unreachable!() };
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::ExtentExceedsBackendWidth { value, maximum: i32::MAX as usize }),
        );
    }

    #[test]
    fn test_static_dimensions() {
        let static_type = ArrayType::new_static(DataType::F32, [2, 3]);
        assert_eq!(static_dimensions(&static_type).unwrap(), vec![2, 3]);
        let dimension = DimensionVariable::new("size", DimensionBounds::unbounded());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![dimension.into()]));
        assert!(matches!(
            static_dimensions(&dynamic_type),
            Err(Error::Array(error)) if matches!(error.as_ref(),
                ArrayError::DynamicShape { shape } if shape == dynamic_type.shape(),
            ),
        ));
    }

    #[test]
    fn test_lower_boundary_layout() {
        let context = MlirContext::new();
        let plain = ArrayType::new_static(DataType::F32, [2, 3]);
        assert!(lower_boundary_layout(&plain, "input", &context).unwrap().is_none());
        let column_major = plain.with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let attribute = lower_boundary_layout(&column_major, "output", &context).unwrap().unwrap();
        assert_eq!(attribute.to_string(), "\"{0,1}\"");
    }

    #[test]
    fn test_stable_hlo_dynamic_dimension_bound() {
        assert_eq!(stable_hlo_dynamic_dimension_bound(&Dimension::Static(4)), None);
        assert_eq!(stable_hlo_dynamic_dimension_bound(&dynamic_dimension("unbounded", None)), None);
        assert_eq!(stable_hlo_dynamic_dimension_bound(&dynamic_dimension("bounded", Some(6))), Some(5));
    }

    #[test]
    fn test_lower_array_ir_type() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let reference_type = Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        assert!(matches!(
            lower_array_ir_type(&reference_type, &context, location),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct }))
                if construct == reference_type.to_string(),
        ));
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
                    lower_f64_scalar_elements_attribute(data_type, tensor_type, integer_value as f64 + 0.75, &context)
                        .unwrap()
                        .to_string(),
                    expected,
                );
            }
        }
    }
}
