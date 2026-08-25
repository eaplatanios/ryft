//! StableHLO lowering for programs that mix arrays with first-class dimensions. The composite IR can also store
//! references, but ordinary XLA lowering rejects any unresolved reference type or operation before entering these
//! array/dimension lowering rules.

use ryft_core::{
    ArrayIrOperation, ArrayIrType, ArrayType, ComparisonDirection, DYNAMIC_SHAPE_SLICE_OPERATION_NAME, DataType,
    Dimension, DimensionOperation, DimensionRequirementOperation, DimensionType, Effect, Operation, ProgramError,
    Shape,
};
use ryft_mlir::dialects::{stable_hlo, tensor};
use ryft_mlir::{
    Block, Context as MlirContext, Location, Operation as MlirOperation, Size as MlirSize, Type as MlirType,
    Value as MlirValue, ValueRef,
};

use super::{
    CollectiveLoweringState, EffectTokens, LowerableXlaOperation, LoweringError, MlirLowerableValue, PlainMlirLowerer,
    PlainMlirLoweringMode, broadcast_changes_explicit_sharding, lower_all_gather_to_mlir, lower_all_to_all_to_mlir,
    lower_compare_to_mlir, lower_concatenate_extent_assertion, lower_constant_elements_attribute,
    lower_constant_output, lower_custom_call_to_mlir, lower_dimension_arithmetic_assertion, lower_dimension_extent,
    lower_dimension_requirement_to_assertion, lower_dynamic_shape_slice_assertion, lower_pad_to_mlir,
    lower_parallel_sum_scatter_to_mlir, lower_physical_bound_value, lower_ragged_all_to_all_to_mlir,
    lower_rng_bit_generator_to_mlir, lower_runtime_dimension_size_i64, lower_sharding_constraint,
    lower_static_index_constants, lower_tensor_type, physical_bound_type, reshape_dimension_i32, reshape_dimension_i64,
    stable_hlo_dynamic_dimension_bound, static_dimensions,
};

/// Physical lowering plan for one axis of a first-class dynamic shape slice.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct DynamicShapeSliceAxisPlan {
    /// Maximum logical output size admitted by the size operand's type.
    size: usize,

    /// Maximum physical input span needed to realize [`Self::size`] elements at the static stride.
    span: usize,

    /// Whether declared bounds leave some runtime input/start/size combinations that need a checked assertion.
    needs_runtime_assertion: bool,
}

/// Plans one axis of a first-class dynamic shape slice from its declared input, start, and size bounds.
fn plan_dynamic_shape_slice_axis(
    axis: usize,
    stride: usize,
    input_dimension: &Dimension,
    start_type: &DimensionType,
    size_type: &DimensionType,
) -> Result<DynamicShapeSliceAxisPlan, LoweringError> {
    let Some(size_upper) = size_type.bounds().upper() else {
        return Err(LoweringError::UnsupportedOp {
            op: format!("{DYNAMIC_SHAPE_SLICE_OPERATION_NAME} size on axis {axis} needs a finite upper bound",),
        });
    };
    let size = size_upper - 1;
    let span = if size == 0 {
        0
    } else {
        (size - 1).checked_mul(stride).and_then(|size| size.checked_add(1)).ok_or_else(|| {
            LoweringError::UnsupportedOp {
                op: format!("{DYNAMIC_SHAPE_SLICE_OPERATION_NAME} physical span overflows on axis {axis}"),
            }
        })?
    };
    let (input_minimum, input_physical_size) = match input_dimension {
        Dimension::Static(extent) => (*extent, *extent),
        Dimension::Dynamic(variable) => {
            let Some(upper) = variable.bounds().upper() else {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("{DYNAMIC_SHAPE_SLICE_OPERATION_NAME} input on axis {axis} needs a finite upper bound",),
                });
            };
            (variable.bounds().lower(), upper - 1)
        }
    };
    if span > input_physical_size {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "{DYNAMIC_SHAPE_SLICE_OPERATION_NAME} physical span {span} exceeds physical input axis {axis} size \
                 {input_physical_size}",
            ),
        });
    }
    let bounds_prove_runtime_in_bounds = start_type
        .bounds()
        .upper()
        .and_then(|upper| upper.checked_sub(1))
        .and_then(|start_maximum| start_maximum.checked_add(span))
        .is_some_and(|limit| limit <= input_minimum);
    Ok(DynamicShapeSliceAxisPlan { size, span, needs_runtime_assertion: !bounds_prove_runtime_in_bounds })
}

/// Lowers a composite array IR type to its physical StableHLO tensor type.
pub(super) fn lower_array_ir_type<'c, 't, L: Location<'c, 't>>(
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
fn lower_explicit_shape<'b, 'c: 'b, 't: 'c>(
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
fn lower_explicit_reshape_shape<'b, 'c: 'b, 't: 'c>(
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

/// Derives the declared and statically bounded physical types shared by dynamic array constructors.
fn dynamic_constructor_types(
    name: &str,
    input_count: usize,
    output_types: &[ArrayIrType],
) -> Result<(ArrayType, ArrayType), LoweringError> {
    let [output_type] = output_types else {
        return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
    };
    let output_type = <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
    let dynamic_count = output_type
        .shape()
        .dimensions()
        .iter()
        .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        .count();
    if input_count != dynamic_count {
        return Err(ProgramError::InvalidInputCount { expected: dynamic_count, actual: input_count }.into());
    }

    let physical_dimensions = output_type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
            Dimension::Dynamic(variable) => stable_hlo_dynamic_dimension_bound(dimension)
                .map(Dimension::Static)
                .ok_or_else(|| LoweringError::UnsupportedOp {
                    op: format!(
                        "{name} output dimension `{}` needs a finite upper bound for physical buffer allocation",
                        variable,
                    ),
                }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let physical_type = output_type.clone().with_shape(Shape::new(physical_dimensions));
    Ok((output_type.clone(), physical_type))
}

/// Refines one statically bounded constructor result from its compact first-class dimension operands.
fn refine_dynamic_constructor_result<'b, 'c: 'b, 't: 'c>(
    mut result: ValueRef<'b, 'c, 't>,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_type: &ArrayType,
    mut refined_type: ArrayType,
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let i32_scalar_type = context
        .tensor_type(context.signless_integer_type(32), &[], None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I32) })?;
    // The dimension operands are compact (one per dynamic axis, in axis order), so advance the operand iterator only
    // when a dynamic axis is encountered rather than indexing by the physical axis number.
    let mut operands = input_values.iter();
    for (axis, dimension) in output_type.shape().dimensions().iter().cloned().enumerate() {
        if !matches!(dimension, Dimension::Dynamic(_)) {
            continue;
        }
        let operand = *operands.next().unwrap();
        let extent = block.append_operation(stable_hlo::convert(operand, i32_scalar_type, location)?)?;
        let extent = extent.result(0).expect("stablehlo.convert should return one result").as_ref();
        let mut dimensions = refined_type.shape().dimensions().to_vec();
        dimensions[axis] = dimension;
        refined_type = refined_type.with_shape(Shape::new(dimensions));
        let refined_tensor_type = lower_tensor_type(&refined_type, context, location)?;
        let operation = block.append_operation(stable_hlo::set_dimension_size(
            result,
            extent,
            refined_tensor_type,
            axis,
            location,
        )?)?;
        result = operation.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
    }
    Ok(vec![result])
}

/// Applies one axis-ordered explicit dimension operand per result axis to one native collective result.
fn refine_collective_result_dimensions<'b, 'c: 'b, 't: 'c>(
    mut result: ValueRef<'b, 'c, 't>,
    extents: &[ValueRef<'b, 'c, 't>],
    output_type: &ArrayType,
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if output_type.rank() != extents.len() {
        return Err(ProgramError::InvalidInputCount { expected: output_type.rank(), actual: extents.len() }.into());
    }
    let i32_scalar_type = context
        .tensor_type(context.signless_integer_type(32), &[], None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I32) })?;
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    for (axis, &extent) in extents.iter().enumerate() {
        if !matches!(output_type.shape().dimensions()[axis], Dimension::Dynamic(_)) {
            continue;
        }
        let extent = block.append_operation(stable_hlo::convert(extent, i32_scalar_type, location)?)?;
        let extent = extent.result(0).expect("stablehlo.convert should return one result").as_ref();
        let refined = block.append_operation(stable_hlo::set_dimension_size(
            result,
            extent,
            output_tensor_type,
            axis,
            location,
        )?)?;
        result = refined.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
    }
    Ok(vec![result])
}

/// Lowers one bounded dynamic integer-valued array constructor from its compact first-class dimension operands.
fn lower_dynamic_constructor<'b, 'c: 'b, 't: 'c>(
    name: &str,
    integer_value: i64,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayIrType],
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let (output_type, physical_type) = dynamic_constructor_types(name, input_values.len(), output_types)?;
    let result =
        lower_constant_output(std::slice::from_ref(&physical_type), integer_value, block, context, location)?.remove(0);
    refine_dynamic_constructor_result(result, input_values, &output_type, physical_type, block, context, location)
}

/// Lowers one array IR instruction.
pub(super) fn lower_array_ir_operation<'b, 'c: 'b, 't: 'c, A>(
    operation: &ArrayIrOperation<A>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    A: MlirLowerableValue,
{
    match operation {
        ArrayIrOperation::Zero(operation) => {
            lower_dynamic_constructor(operation.name(), 0, input_values, output_types, block, context, location)
        }
        ArrayIrOperation::DynamicOne(operation) => {
            lower_dynamic_constructor(operation.name(), 1, input_values, output_types, block, context, location)
        }
        ArrayIrOperation::DynamicIota(operation) => {
            let (output_type, physical_type) =
                dynamic_constructor_types(operation.name(), input_values.len(), output_types)?;
            let tensor_type = lower_tensor_type(&physical_type, context, location)?;
            let iota = block.append_operation(stable_hlo::iota(tensor_type, operation.dimension(), location)?)?;
            let result = iota.result(0).expect("stablehlo.iota should return one result").as_ref();
            refine_dynamic_constructor_result(
                result,
                input_values,
                &output_type,
                physical_type,
                block,
                context,
                location,
            )
        }
        ArrayIrOperation::Array(operation) => {
            let input_types = input_types
                .iter()
                .map(|r#type| {
                    <&ArrayType>::try_from(r#type).cloned().map_err(|error| LoweringError::Tracing(error.into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output_types = output_types
                .iter()
                .map(|r#type| {
                    <&ArrayType>::try_from(r#type).cloned().map_err(|error| LoweringError::Tracing(error.into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut lowerer = PlainMlirLowerer::new(*block, context, location)
                .with_input_types(input_types)
                .with_effect_tokens(*effect_tokens)
                .with_collective_state(collective_state.clone());
            let outputs = operation.lower_to_mlir(
                input_values,
                output_types.as_slice(),
                PlainMlirLoweringMode::Unpacked,
                &mut lowerer,
            )?;
            *effect_tokens = lowerer.effect_tokens;
            Ok(outputs)
        }
        ArrayIrOperation::Dimension(operation) => {
            match operation {
                DimensionOperation::Constant(operation) => {
                    if !input_values.is_empty() {
                        return Err(ProgramError::InvalidInputCount { expected: 0, actual: input_values.len() }.into());
                    }
                    Ok(vec![lower_dimension_extent(operation.value(), block, context, location)?])
                }
                DimensionOperation::Add(_)
                | DimensionOperation::Sub(_)
                | DimensionOperation::SaturatingSub(_)
                | DimensionOperation::Mul(_)
                | DimensionOperation::Pow(_)
                | DimensionOperation::DivFloor(_)
                | DimensionOperation::Rem(_)
                | DimensionOperation::Min(_)
                | DimensionOperation::Max(_) => {
                    let [left, right] = input_values else {
                        return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into());
                    };
                    let [left_type, right_type] = input_types else {
                        return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_types.len() }.into());
                    };
                    let left_type =
                        <&DimensionType>::try_from(left_type).map_err(|error| LoweringError::Tracing(error.into()))?;
                    let right_type =
                        <&DimensionType>::try_from(right_type).map_err(|error| LoweringError::Tracing(error.into()))?;
                    // Bounds-proven arithmetic lowers without assertion overhead. Otherwise, preserve eager checked
                    // semantics with one diagnostic runtime check on the ordered-assertion chain.
                    let requires_runtime_assertion = operation.effects().contains(Effect::OrderedAssertion);
                    if requires_runtime_assertion {
                        lower_dimension_arithmetic_assertion(
                            operation,
                            left_type,
                            right_type,
                            *left,
                            *right,
                            effect_tokens,
                            block,
                            context,
                            location,
                        )?;
                    }
                    // A runtime assertion and the arithmetic operation do not have a StableHLO data dependency.
                    // Select a valid divisor for the data operation so speculative or reordered execution cannot
                    // evaluate division by zero before the host callback reports the original operands.
                    let safe_right = if requires_runtime_assertion
                        && matches!(operation, DimensionOperation::DivFloor(_) | DimensionOperation::Rem(_))
                    {
                        let constants = lower_static_index_constants(&[0, 1], block, context, location)?;
                        let positive = lower_compare_to_mlir(
                            ComparisonDirection::GreaterThan,
                            *right,
                            constants[0],
                            block,
                            location,
                        )?;
                        let selected =
                            block.append_operation(stable_hlo::select(positive, *right, constants[1], location)?)?;
                        selected.result(0).expect("stablehlo.select should return one result").as_ref()
                    } else {
                        *right
                    };
                    let result = match operation {
                        DimensionOperation::Add(_) => {
                            block.append_operation(stable_hlo::add(*left, *right, location)?)?
                        }
                        DimensionOperation::Sub(_) => {
                            block.append_operation(stable_hlo::subtract(*left, *right, location)?)?
                        }
                        DimensionOperation::SaturatingSub(_) => {
                            let difference = block.append_operation(stable_hlo::subtract(*left, *right, location)?)?;
                            let difference = difference.result(0).unwrap().as_ref();
                            let tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
                            let elements =
                                lower_constant_elements_attribute(DataType::I64, tensor_type, 0_i64, context)?;
                            let zero = block.append_operation(stable_hlo::constant(elements, location)?)?;
                            let zero = zero.result(0).unwrap().as_ref();
                            block.append_operation(stable_hlo::maximum(difference, zero, location)?)?
                        }
                        DimensionOperation::Mul(_) => {
                            block.append_operation(stable_hlo::multiply(*left, *right, location)?)?
                        }
                        DimensionOperation::Pow(_) => {
                            block.append_operation(stable_hlo::power(*left, *right, location)?)?
                        }
                        DimensionOperation::DivFloor(_) => {
                            block.append_operation(stable_hlo::divide(*left, safe_right, location)?)?
                        }
                        DimensionOperation::Rem(_) => {
                            block.append_operation(stable_hlo::remainder(*left, safe_right, location)?)?
                        }
                        DimensionOperation::Min(_) => {
                            block.append_operation(stable_hlo::minimum(*left, *right, location)?)?
                        }
                        DimensionOperation::Max(_) => {
                            block.append_operation(stable_hlo::maximum(*left, *right, location)?)?
                        }
                        _ => unreachable!(),
                    };
                    // For the same no-data-dependency reason, an unproven subtraction can hand a negative extent to
                    // consumers such as `set_dimension_size`, and unproven addition/multiplication/power can hand
                    // them a wrapped negative, failing inside XLA before the host callback reports the original
                    // operands. Clamping the unproven data path to zero keeps it well-defined so the assertion owns
                    // the diagnostic.
                    let result = if requires_runtime_assertion
                        && matches!(
                            operation,
                            DimensionOperation::Add(_)
                                | DimensionOperation::Sub(_)
                                | DimensionOperation::Mul(_)
                                | DimensionOperation::Pow(_)
                        ) {
                        let constants = lower_static_index_constants(&[0], block, context, location)?;
                        let raw = result.result(0).unwrap().as_ref();
                        block.append_operation(stable_hlo::maximum(raw, constants[0], location)?)?
                    } else {
                        result
                    };
                    Ok(vec![result.result(0).unwrap().as_ref()])
                }
                DimensionOperation::Requirement(operation) => {
                    if operation.effects().contains(ryft_core::Effect::OrderedAssertion) {
                        lower_dimension_requirement_to_assertion(
                            operation,
                            operation.name(),
                            input_values,
                            effect_tokens,
                            block,
                            context,
                            location,
                        )?;
                    }
                    Ok(Vec::new())
                }
            }
        }
        ArrayIrOperation::Compare(operation) => {
            let [left, right] = input_values else {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into());
            };
            // Dimension atoms already lower to scalar `i64` SSA values, so the ordinary comparison lowering can
            // consume them directly without a data gateway or host-side shape reconstruction.
            Ok(vec![lower_compare_to_mlir(operation.direction(), *left, *right, block, location)?])
        }
        ArrayIrOperation::DimensionSize(operation) => {
            let [input] = input_values else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
            };
            let [input_type] = input_types else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_types.len() }.into());
            };
            let input_type =
                <&ArrayType>::try_from(input_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            match input_type.shape().dimensions()[operation.axis()] {
                Dimension::Static(extent) => {
                    let output_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
                    let extent = i64::try_from(extent).map_err(|_| LoweringError::UnsupportedOp {
                        op: format!("dimension extent {extent} does not fit StableHLO i64"),
                    })?;
                    let elements = lower_constant_elements_attribute(DataType::I64, output_type, extent, context)?;
                    let operation = block.append_operation(stable_hlo::constant(elements, location)?)?;
                    Ok(vec![operation.result(0).expect("stablehlo.constant should return one result").as_ref()])
                }
                Dimension::Dynamic(_) => {
                    let size =
                        block.append_operation(stable_hlo::get_dimension_size(*input, operation.axis(), location)?)?;
                    let size = size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref();
                    let output_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
                    let converted = block.append_operation(stable_hlo::convert(size, output_type, location)?)?;
                    Ok(vec![converted.result(0).expect("stablehlo.convert should return one result").as_ref()])
                }
            }
        }
        operation @ (ArrayIrOperation::NewReference(_)
        | ArrayIrOperation::ReferenceIndex(_)
        | ArrayIrOperation::ReferenceSlice(_)
        | ArrayIrOperation::ReferenceRead(_)
        | ArrayIrOperation::ReferenceWrite(_)
        | ArrayIrOperation::ReferenceSwap(_)
        | ArrayIrOperation::ReferenceAddUpdate(_)
        | ArrayIrOperation::FreezeReference(_)) => {
            Err(LoweringError::UnresolvedReference { construct: operation.name().to_string() })
        }
        ArrayIrOperation::DimensionFromScalar(operation) => {
            let [input] = input_values else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
            };
            let i64_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
            let converted = block.append_operation(stable_hlo::convert(*input, i64_type, location)?)?;
            let converted = converted.result(0).expect("stablehlo.convert should return one result").as_ref();
            let requirement =
                DimensionRequirementOperation::bounds(operation.result_type(), operation.result_type().bounds());
            lower_dimension_requirement_to_assertion(
                &requirement,
                operation.name(),
                &[converted],
                effect_tokens,
                block,
                context,
                location,
            )?;
            Ok(vec![converted])
        }
        ArrayIrOperation::DimensionToScalar(_) => {
            let [input] = input_values else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
            };
            Ok(vec![*input])
        }
        ArrayIrOperation::Reshape(operation) => {
            let Some((input, output_extents)) = input_values.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let [output_type] = output_types else {
                return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
            };
            let output_type =
                <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            let input = if let Some(dimensions) = operation.dimensions() {
                let transpose =
                    block.append_operation(stable_hlo::transpose(*input, dimensions.as_slice(), location)?)?;
                transpose.result(0).expect("stablehlo.transpose should return one result").as_ref()
            } else {
                *input
            };
            let result = if output_type.static_shape().is_some() {
                let output_shape = static_dimensions(output_type)?;
                for dimension in &output_shape {
                    reshape_dimension_i64(*dimension)?;
                }
                let reshape = block.append_operation(stable_hlo::reshape(input, output_shape.as_slice(), location)?)?;
                reshape.result(0).expect("stablehlo.reshape should return one result").as_ref()
            } else {
                let shape = lower_explicit_reshape_shape(output_extents, block, context, location)?;
                let output_bounds = output_type
                    .shape()
                    .dimensions()
                    .iter()
                    .map(|dimension| match dimension {
                        Dimension::Static(extent) => Some(*extent),
                        Dimension::Dynamic(_) => stable_hlo_dynamic_dimension_bound(dimension),
                    })
                    .collect::<Vec<_>>();
                for bound in output_bounds.iter().flatten() {
                    reshape_dimension_i32(*bound)?;
                }
                let reshape =
                    block.append_operation(stable_hlo::dynamic_reshape(input, shape, &output_bounds, location)?)?;
                let result = reshape.result(0).expect("stablehlo.dynamic_reshape should return one result").as_ref();
                let expected_type = lower_tensor_type(output_type, context, location)?;
                if result.r#type()? == expected_type.as_ref() {
                    result
                } else {
                    let cast = block.append_operation(tensor::cast(result, expected_type, location)?)?;
                    cast.result(0).expect("tensor.cast should return one result").as_ref()
                }
            };
            if operation.output_sharding().is_some() {
                let output_sharding =
                    output_type.sharding().expect("reshape type inference should preserve requested output sharding");
                lower_sharding_constraint(&[result], output_sharding, block, location)
            } else {
                Ok(vec![result])
            }
        }
        ArrayIrOperation::Broadcast(operation) => {
            let Some((input, output_extents)) = input_values.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let Some(input_type) = input_types.first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let input_type =
                <&ArrayType>::try_from(input_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            let [output_type] = output_types else {
                return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
            };
            let output_type =
                <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            let result = if output_type.static_shape().is_some() {
                let output_tensor_type = lower_tensor_type(output_type, context, location)?;
                let broadcast = block.append_operation(stable_hlo::broadcast(
                    *input,
                    output_tensor_type,
                    operation.output_axes(),
                    location,
                )?)?;
                broadcast.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref()
            } else if physical_bound_type(input_type).is_ok() && physical_bound_type(output_type).is_ok() {
                // Finite bounded inputs can be materialized at their physical shape, broadcast statically, and then
                // refined from the explicit result extents. This avoids `dynamic_broadcast_in_dim`, which the XLA
                // importer cannot legalize when an intermediate shape includes a computed dynamic ratio.
                let dynamic_extents = output_type
                    .shape()
                    .dimensions()
                    .iter()
                    .zip(output_extents)
                    .filter_map(|(dimension, extent)| matches!(dimension, Dimension::Dynamic(_)).then_some(*extent))
                    .collect::<Vec<_>>();
                let (declared_type, physical_type) =
                    dynamic_constructor_types(operation.name(), dynamic_extents.len(), output_types)?;
                let input = lower_physical_bound_value(*input, input_type, 0.0, block, context, location)?;
                let output_tensor_type = lower_tensor_type(&physical_type, context, location)?;
                let broadcast = block.append_operation(stable_hlo::broadcast(
                    input,
                    output_tensor_type,
                    operation.output_axes(),
                    location,
                )?)?;
                let result = broadcast.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref();
                refine_dynamic_constructor_result(
                    result,
                    dynamic_extents.as_slice(),
                    &declared_type,
                    physical_type,
                    block,
                    context,
                    location,
                )?
                .remove(0)
            } else {
                let shape = lower_explicit_shape(output_extents, block, context, location)?;
                let broadcast = block.append_operation(stable_hlo::dynamic_broadcast(
                    *input,
                    shape,
                    operation.output_axes(),
                    None,
                    None,
                    location,
                )?)?;
                let result =
                    broadcast.result(0).expect("stablehlo.dynamic_broadcast_in_dim should return one result").as_ref();
                let expected_type = lower_tensor_type(output_type, context, location)?;
                if result.r#type()? == expected_type.as_ref() {
                    result
                } else {
                    let cast = block.append_operation(tensor::cast(result, expected_type, location)?)?;
                    cast.result(0).expect("tensor.cast should return one result").as_ref()
                }
            };
            if broadcast_changes_explicit_sharding(input_type, output_type, operation.output_axes()) {
                lower_sharding_constraint(&[result], output_type.sharding().unwrap(), block, location)
            } else {
                Ok(vec![result])
            }
        }
        ArrayIrOperation::Concatenate(operation) => {
            let Some((result_extent, array_inputs)) = input_values.split_last() else {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }.into());
            };
            let Some((result_extent_type, array_input_types)) = input_types.split_last() else {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_types.len() }.into());
            };
            if array_inputs.is_empty() {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into());
            }
            <&DimensionType>::try_from(result_extent_type).map_err(|error| LoweringError::Tracing(error.into()))?;

            if operation.effects().contains(Effect::OrderedAssertion) {
                // The callback receives every concrete logical input extent and computes their checked sum on the
                // host. This avoids both overflow in a speculative StableHLO sum and false rejection from conservative
                // declared maxima. A type-derived proof omits this entire assertion path for static signatures.
                let i64_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
                let mut input_extents = Vec::with_capacity(array_inputs.len());
                for (input, r#type) in array_inputs.iter().zip(array_input_types) {
                    let r#type =
                        <&ArrayType>::try_from(r#type).map_err(|error| LoweringError::Tracing(error.into()))?;
                    let extent = match r#type.shape().dimensions()[operation.axis()] {
                        Dimension::Static(extent) => {
                            lower_static_index_constants(&[extent], block, context, location)?[0]
                        }
                        Dimension::Dynamic(_) => {
                            let extent = block.append_operation(stable_hlo::get_dimension_size(
                                *input,
                                operation.axis(),
                                location,
                            )?)?;
                            let extent = extent
                                .result(0)
                                .expect("stablehlo.get_dimension_size should return one result")
                                .as_ref();
                            let extent = block.append_operation(stable_hlo::convert(extent, i64_type, location)?)?;
                            extent.result(0).expect("stablehlo.convert should return one result").as_ref()
                        }
                    };
                    input_extents.push(extent);
                }
                lower_concatenate_extent_assertion(
                    operation.axis(),
                    *result_extent,
                    input_extents.as_slice(),
                    effect_tokens,
                    block,
                    context,
                    location,
                )?;
            }

            // StableHLO receives only the physical arrays. The trailing scalar is consumed by the optional assertion
            // or is redundant with the type-level proof.
            let result = block.append_operation(stable_hlo::concatenate(array_inputs, operation.axis(), location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.concatenate should return one result").as_ref()])
        }
        ArrayIrOperation::CustomCall(operation) => {
            let dynamic_output_dimension_count = operation
                .output_types()
                .iter()
                .flat_map(|output_type| output_type.shape().dimensions())
                .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
                .count();
            let Some(array_input_count) = input_values.len().checked_sub(dynamic_output_dimension_count) else {
                return Err(ProgramError::InvalidInputCount {
                    expected: dynamic_output_dimension_count,
                    actual: input_values.len(),
                }
                .into());
            };
            let (array_inputs, output_extents) = input_values.split_at(array_input_count);
            let array_input_types = input_types[..array_input_count]
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| LoweringError::Tracing(error.into()))?;
            let mut physical_output_types = operation
                .output_types()
                .iter()
                .map(|output_type| {
                    let dimensions = output_type
                        .shape()
                        .dimensions()
                        .iter()
                        .map(|dimension| match dimension {
                            Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
                            Dimension::Dynamic(variable) => variable
                                .bounds()
                                .upper()
                                .and_then(|upper| upper.checked_sub(1))
                                .map(Dimension::Static)
                                .ok_or_else(|| LoweringError::UnsupportedOp {
                                    op: format!(
                                        "custom-call output dimension `{}` needs a finite upper bound for physical \
                                         buffer allocation",
                                        variable,
                                    ),
                                }),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(output_type.clone().with_shape(Shape::new(dimensions)))
                })
                .collect::<Result<Vec<_>, LoweringError>>()?;
            for alias in operation.input_output_aliases() {
                physical_output_types[alias.output_index()] = array_input_types[alias.input_index()].clone();
            }
            let mut results = lower_custom_call_to_mlir(
                operation,
                array_inputs,
                array_input_types.as_slice(),
                physical_output_types.as_slice(),
                effect_tokens,
                block,
                context,
                location,
            )?;
            let i32_scalar_type = context
                .tensor_type(context.signless_integer_type(32), &[], None, location)
                .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I32) })?;
            let mut output_extents = output_extents.iter();
            for (output_index, output_type) in operation.output_types().iter().enumerate() {
                let mut refined_type = physical_output_types[output_index].clone();
                for (axis, dimension) in output_type.shape().dimensions().iter().cloned().enumerate() {
                    if !matches!(dimension, Dimension::Dynamic(_)) {
                        continue;
                    }
                    let extent = *output_extents.next().unwrap();
                    let extent = block.append_operation(stable_hlo::convert(extent, i32_scalar_type, location)?)?;
                    let extent = extent.result(0).expect("stablehlo.convert should return one result").as_ref();
                    let mut dimensions = refined_type.shape().dimensions().to_vec();
                    dimensions[axis] = dimension;
                    refined_type = refined_type.with_shape(Shape::new(dimensions));
                    let refined_tensor_type = lower_tensor_type(&refined_type, context, location)?;
                    let result = block.append_operation(stable_hlo::set_dimension_size(
                        results[output_index],
                        extent,
                        refined_tensor_type,
                        axis,
                        location,
                    )?)?;
                    results[output_index] =
                        result.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
                }
            }
            Ok(results)
        }
        ArrayIrOperation::Pad(operation) => {
            let [output_type] = output_types else {
                return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
            };
            let output_type =
                <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            let expected_input_count = output_type.rank() + 2;
            if input_values.len() != expected_input_count {
                return Err(ProgramError::InvalidInputCount {
                    expected: expected_input_count,
                    actual: input_values.len(),
                }
                .into());
            }
            let mut results = lower_pad_to_mlir(
                operation,
                &input_values[..2],
                std::slice::from_ref(output_type),
                block,
                context,
                location,
            )?;
            let i32_scalar_type = context
                .tensor_type(context.signless_integer_type(32), &[], None, location)
                .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I32) })?;
            let mut refined_type = output_type.clone();
            for (axis, dimension) in output_type.shape().dimensions().iter().cloned().enumerate() {
                if !matches!(dimension, Dimension::Dynamic(_)) {
                    continue;
                }
                let extent =
                    block.append_operation(stable_hlo::convert(input_values[axis + 2], i32_scalar_type, location)?)?;
                let extent = extent.result(0).expect("stablehlo.convert should return one result").as_ref();
                let mut dimensions = refined_type.shape().dimensions().to_vec();
                dimensions[axis] = dimension;
                refined_type = refined_type.with_shape(Shape::new(dimensions));
                let refined_tensor_type = lower_tensor_type(&refined_type, context, location)?;
                let result = block.append_operation(stable_hlo::set_dimension_size(
                    results[0],
                    extent,
                    refined_tensor_type,
                    axis,
                    location,
                )?)?;
                results[0] = result.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
            }
            Ok(results)
        }
        ArrayIrOperation::DynamicShapeSlice(operation) => {
            let Some((input, bounds)) = input_values.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let input_type =
                <&ArrayType>::try_from(&input_types[0]).map_err(|error| LoweringError::Tracing(error.into()))?;
            let [output_type] = output_types else {
                return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
            };
            let output_type =
                <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            let rank = output_type.rank();
            if bounds.len() != 2 * rank {
                return Err(
                    ProgramError::InvalidInputCount { expected: 1 + 2 * rank, actual: input_values.len() }.into()
                );
            }
            let (starts, sizes) = bounds.split_at(rank);
            let start_types = &input_types[1..1 + rank];
            let size_types = &input_types[1 + rank..];
            let mut physical_sizes = Vec::with_capacity(rank);
            let mut physical_spans = Vec::with_capacity(rank);
            for axis in 0..rank {
                let start_type = <&DimensionType>::try_from(&start_types[axis])
                    .map_err(|error| LoweringError::Tracing(error.into()))?;
                let size_type = <&DimensionType>::try_from(&size_types[axis])
                    .map_err(|error| LoweringError::Tracing(error.into()))?;
                let plan = plan_dynamic_shape_slice_axis(
                    axis,
                    operation.strides()[axis],
                    &input_type.shape().dimensions()[axis],
                    start_type,
                    size_type,
                )?;
                if plan.needs_runtime_assertion {
                    let input_size = lower_runtime_dimension_size_i64(*input, axis, block, context, location)?;
                    lower_dynamic_shape_slice_assertion(
                        axis,
                        operation.strides()[axis],
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

            // `real_dynamic_slice` is not accepted by the pinned XLA translator. Extract each axis's maximum admitted
            // physical span with the ordinary dynamic-slice operation, apply static strides, and then restore the
            // logical runtime sizes. The checks above ensure the physical span fits the bound-shaped buffer and
            // preserve eager semantics with a runtime assertion whenever the declared bounds alone do not prove the
            // logical slice limit valid. Therefore, `dynamic_slice` clamps only executions that fail the assertion.
            let slice = block.append_operation(stable_hlo::dynamic_slice(
                *input,
                starts,
                physical_spans.as_slice(),
                location,
            )?)?;
            let mut result = slice.result(0).expect("stablehlo.dynamic_slice should return one result").as_ref();
            if operation.strides().iter().any(|stride| *stride != 1) {
                let start_indices = vec![0; rank];
                let slice = block.append_operation(stable_hlo::slice(
                    result,
                    start_indices.as_slice(),
                    physical_spans.as_slice(),
                    operation.strides(),
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
        ArrayIrOperation::RngBitGenerator(operation) => {
            let has_dynamic_output = operation
                .output_type()
                .shape()
                .dimensions()
                .iter()
                .any(|dimension| matches!(dimension, Dimension::Dynamic(_)));
            if has_dynamic_output {
                return Err(LoweringError::UnsupportedOp {
                    op: "rng-bit-generator with dynamic output extents cannot lower by generating the physical \
                         upper-bound shape because that would advance its functional state by the physical rather \
                         than logical element count"
                        .to_string(),
                });
            }
            let expected_input_count = 1;
            if input_values.len() != expected_input_count {
                return Err(ProgramError::InvalidInputCount {
                    expected: expected_input_count,
                    actual: input_values.len(),
                }
                .into());
            }
            lower_rng_bit_generator_to_mlir(operation, input_values, block, context, location)
        }
        ArrayIrOperation::AllGather(operation) => {
            let Some(input) = input_values.first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let [output_type] = output_types else {
                return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
            };
            let output_type =
                <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            let result =
                lower_all_gather_to_mlir(operation, collective_state, *input, output_type, block, context, location)?
                    .remove(0);
            refine_collective_result_dimensions(result, &input_values[1..], output_type, block, context, location)
        }
        ArrayIrOperation::ParallelSumScatter(operation) => {
            let Some(input) = input_values.first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let [output_type] = output_types else {
                return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
            };
            let output_type =
                <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            let result = lower_parallel_sum_scatter_to_mlir(
                operation,
                collective_state,
                *input,
                output_type,
                block,
                context,
                location,
            )?
            .remove(0);
            refine_collective_result_dimensions(result, &input_values[1..], output_type, block, context, location)
        }
        ArrayIrOperation::AllToAll(operation) => {
            let Some(input) = input_values.first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let Some(input_type) = input_types.first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let input_type =
                <&ArrayType>::try_from(input_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            let [output_type] = output_types else {
                return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
            };
            let output_type =
                <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            let result = lower_all_to_all_to_mlir(
                operation,
                collective_state,
                *input,
                input_type,
                output_type,
                block,
                context,
                location,
            )?
            .remove(0);
            refine_collective_result_dimensions(result, &input_values[1..], output_type, block, context, location)
        }
        ArrayIrOperation::RaggedAllToAll(operation) => {
            if input_values.len() != 6 {
                return Err(ProgramError::InvalidInputCount { expected: 6, actual: input_values.len() }.into());
            }
            if input_types.len() != 6 {
                return Err(ProgramError::InvalidInputCount { expected: 6, actual: input_types.len() }.into());
            }
            let array_input_types = input_types
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| LoweringError::Tracing(error.into()))?;
            let [output_type] = output_types else {
                return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
            };
            <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            lower_ragged_all_to_all_to_mlir(
                operation,
                collective_state,
                input_values,
                array_input_types.as_slice(),
                block,
                context,
                location,
            )
        }
        ArrayIrOperation::Condition(_)
        | ArrayIrOperation::While(_)
        | ArrayIrOperation::Scan(_)
        | ArrayIrOperation::CustomJvp(_)
        | ArrayIrOperation::CustomVjp(_)
        | ArrayIrOperation::LinearCall(_)
        | ArrayIrOperation::Rematerialize(_) => Err(LoweringError::UnsupportedOp {
            op: format!(
                "core composite higher-order operation `{}` must be promoted to the XLA operation family before \
                     lowering",
                operation.name(),
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{DimensionBounds, DimensionVariable};

    use super::*;

    /// Creates a first-class dimension type with the provided bounds.
    fn dimension_type(name: &str, bounds: DimensionBounds) -> DimensionType {
        DimensionType::new(DimensionVariable::new(name, bounds))
    }

    /// Extracts the unsupported-operation diagnostic produced by the physical slice planner.
    fn unsupported_operation(error: LoweringError) -> String {
        let LoweringError::UnsupportedOp { op } = error else {
            panic!("expected an unsupported-operation error but received {error}");
        };
        op
    }

    #[test]
    fn test_plan_dynamic_shape_slice_axis() {
        let unbounded = dimension_type("unbounded", DimensionBounds::unbounded());
        let bounded_start = dimension_type("start", DimensionBounds::non_negative(Some(5)).unwrap());
        let bounded_size = dimension_type("size", DimensionBounds::non_negative(Some(5)).unwrap());

        assert_eq!(
            unsupported_operation(
                plan_dynamic_shape_slice_axis(2, 1, &Dimension::Static(4), &bounded_start, &unbounded).unwrap_err(),
            ),
            "dynamic_shape_slice size on axis 2 needs a finite upper bound",
        );

        let unbounded_input = Dimension::Dynamic(DimensionVariable::new("input", DimensionBounds::unbounded()));
        assert_eq!(
            unsupported_operation(
                plan_dynamic_shape_slice_axis(1, 1, &unbounded_input, &bounded_start, &bounded_size).unwrap_err(),
            ),
            "dynamic_shape_slice input on axis 1 needs a finite upper bound",
        );

        let overflowing_size = dimension_type("size", DimensionBounds::non_negative(Some(usize::MAX)).unwrap());
        assert_eq!(
            unsupported_operation(
                plan_dynamic_shape_slice_axis(3, 2, &Dimension::Static(usize::MAX), &bounded_start, &overflowing_size,)
                    .unwrap_err(),
            ),
            "dynamic_shape_slice physical span overflows on axis 3",
        );

        assert_eq!(
            unsupported_operation(
                plan_dynamic_shape_slice_axis(0, 1, &Dimension::Static(3), &bounded_start, &bounded_size).unwrap_err(),
            ),
            "dynamic_shape_slice physical span 4 exceeds physical input axis 0 size 3",
        );

        let bounded_input =
            Dimension::Dynamic(DimensionVariable::new("input", DimensionBounds::positive(Some(5)).unwrap()));
        assert_eq!(
            plan_dynamic_shape_slice_axis(0, 1, &bounded_input, &bounded_start, &bounded_size).unwrap(),
            DynamicShapeSliceAxisPlan { size: 4, span: 4, needs_runtime_assertion: true },
        );

        let proven_start = dimension_type("start", DimensionBounds::non_negative(Some(1)).unwrap());
        let proven_size = dimension_type("size", DimensionBounds::non_negative(Some(4)).unwrap());
        assert_eq!(
            plan_dynamic_shape_slice_axis(0, 1, &Dimension::Static(4), &proven_start, &proven_size).unwrap(),
            DynamicShapeSliceAxisPlan { size: 3, span: 3, needs_runtime_assertion: false },
        );
    }
}
