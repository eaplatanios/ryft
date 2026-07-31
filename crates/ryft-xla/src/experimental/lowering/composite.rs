//! StableHLO lowering for programs that mix arrays with first-class dimensions.

use ryft_core::backends::array_programs::ArrayProgramOperation;
use ryft_core::backends::dimensions::DimensionOperation;
use ryft_core::operations::collectives::CollectiveMode;
use ryft_core::operations::manipulation::CONCATENATE_OPERATION_NAME;
use ryft_core::programs::{Operation as CoreOperation, ProgramError};
use ryft_core::types::{ArrayProgramType, ArrayType, DataType, Dimension, DimensionType, MAX_DIMENSION_EXTENT, Shape};
use ryft_mlir::dialects::{stable_hlo, tensor};
use ryft_mlir::{
    Block, Context as MlirContext, Location, Operation as MlirOperation, Size as MlirSize, Type as MlirType,
    Value as MlirValue, ValueRef,
};

use super::{
    CollectiveLoweringState, LowerableXlaOperation, LoweringError, MlirLowerableValue, PlainMlirLowerer,
    PlainMlirLoweringMode, broadcast_changes_explicit_sharding, lower_all_gather_to_mlir, lower_all_to_all_to_mlir,
    lower_compare_to_mlir, lower_constant_elements_attribute, lower_constant_output, lower_custom_call_to_mlir,
    lower_pad_to_mlir, lower_psum_scatter_to_mlir, lower_rng_bit_generator_to_mlir, lower_sharding_constraint,
    lower_tensor_type, reshape_dimension_i32, reshape_dimension_i64, stable_hlo_dynamic_dimension_bound,
    static_dimensions,
};

/// Lowers a composite array-program type to its physical StableHLO tensor type.
pub(super) fn lower_array_program_type<'c, 't, L: Location<'c, 't>>(
    r#type: &ArrayProgramType,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
    match r#type {
        ArrayProgramType::Array(r#type) => lower_tensor_type(r#type, context, location),
        ArrayProgramType::Dimension(_) => lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location),
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
    output_types: &[ArrayProgramType],
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
                        "{name} output dimension '{}' needs a finite upper bound for physical buffer allocation",
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

/// Applies explicit changed-axis dimension operands to one native collective result.
fn refine_collective_result_axes<'b, 'c: 'b, 't: 'c>(
    mut result: ValueRef<'b, 'c, 't>,
    axes: &[usize],
    extents: &[ValueRef<'b, 'c, 't>],
    output_type: &ArrayType,
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if axes.len() != extents.len() {
        return Err(ProgramError::InvalidInputCount { expected: axes.len(), actual: extents.len() }.into());
    }
    let i32_scalar_type = context
        .tensor_type(context.signless_integer_type(32), &[], None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I32) })?;
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    for (&axis, &extent) in axes.iter().zip(extents) {
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
    output_types: &[ArrayProgramType],
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let (output_type, physical_type) = dynamic_constructor_types(name, input_values.len(), output_types)?;
    let result =
        lower_constant_output(std::slice::from_ref(&physical_type), integer_value, block, context, location)?.remove(0);
    refine_dynamic_constructor_result(result, input_values, &output_type, physical_type, block, context, location)
}

/// Lowers one array-program instruction.
pub(super) fn lower_array_program_operation<'b, 'c: 'b, 't: 'c, A>(
    operation: &ArrayProgramOperation<A>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayProgramType],
    output_types: &[ArrayProgramType],
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    A: MlirLowerableValue,
{
    match operation {
        ArrayProgramOperation::Zero(_) => {
            let output_types = output_types
                .iter()
                .map(|r#type| {
                    <&ArrayType>::try_from(r#type).cloned().map_err(|error| LoweringError::Tracing(error.into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            lower_constant_output(output_types.as_slice(), 0, block, context, location)
        }
        ArrayProgramOperation::DynamicZero(operation) => {
            lower_dynamic_constructor(operation.name(), 0, input_values, output_types, block, context, location)
        }
        ArrayProgramOperation::DynamicOne(operation) => {
            lower_dynamic_constructor(operation.name(), 1, input_values, output_types, block, context, location)
        }
        ArrayProgramOperation::DynamicIota(operation) => {
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
        ArrayProgramOperation::Array(operation) => {
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
                .with_token(*token)
                .with_collective_state(collective_state.clone());
            let outputs = operation.lower_to_mlir(
                input_values,
                output_types.as_slice(),
                PlainMlirLoweringMode::Unpacked,
                &mut lowerer,
            )?;
            *token = lowerer.token;
            Ok(outputs)
        }
        ArrayProgramOperation::Dimension(operation) => {
            match operation {
                DimensionOperation::Constant(operation) => {
                    if !input_values.is_empty() {
                        return Err(ProgramError::InvalidInputCount { expected: 0, actual: input_values.len() }.into());
                    }
                    let tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
                    let extent = i64::try_from(operation.value().extent()).unwrap();
                    let elements = lower_constant_elements_attribute(DataType::I64, tensor_type, extent, context)?;
                    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
                    Ok(vec![constant.result(0).unwrap().as_ref()])
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
                    let maximum = |r#type: &DimensionType| r#type.bounds().upper()?.checked_sub(1);
                    let checked_power = |mut base: usize, mut exponent: usize| {
                        let mut result = 1usize;
                        while exponent > 0 {
                            if exponent & 1 == 1 {
                                result = result.checked_mul(base)?;
                            }
                            exponent >>= 1;
                            if exponent > 0 {
                                base = base.checked_mul(base)?;
                            }
                        }
                        Some(result)
                    };
                    let arithmetic_is_proven = match operation {
                        DimensionOperation::Add(_) => maximum(left_type)
                            .zip(maximum(right_type))
                            .and_then(|(left, right)| left.checked_add(right))
                            .is_some_and(|result| result <= MAX_DIMENSION_EXTENT),
                        DimensionOperation::Sub(_) => {
                            maximum(right_type).is_some_and(|right| left_type.bounds().lower() >= right)
                        }
                        DimensionOperation::SaturatingSub(_)
                        | DimensionOperation::Min(_)
                        | DimensionOperation::Max(_) => true,
                        DimensionOperation::Mul(_) => maximum(left_type)
                            .zip(maximum(right_type))
                            .and_then(|(left, right)| left.checked_mul(right))
                            .is_some_and(|result| result <= MAX_DIMENSION_EXTENT),
                        DimensionOperation::Pow(_) => maximum(left_type)
                            .zip(maximum(right_type))
                            .and_then(|(left, right)| checked_power(left, right))
                            .is_some_and(|result| result <= MAX_DIMENSION_EXTENT),
                        DimensionOperation::DivFloor(_) | DimensionOperation::Rem(_) => right_type.bounds().lower() > 0,
                        _ => unreachable!(),
                    };
                    // A bounds proof makes these physical scalar `i64` operations equivalent to checked dimension
                    // arithmetic. P7 owns runtime assertion lowering for the remaining overflow, underflow, and
                    // zero-divisor cases.
                    if !arithmetic_is_proven {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!(
                                "first-class dimension operation `{}` requires checked runtime assertion lowering",
                                operation.name(),
                            ),
                        });
                    }
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
                            block.append_operation(stable_hlo::divide(*left, *right, location)?)?
                        }
                        DimensionOperation::Rem(_) => {
                            block.append_operation(stable_hlo::remainder(*left, *right, location)?)?
                        }
                        DimensionOperation::Min(_) => {
                            block.append_operation(stable_hlo::minimum(*left, *right, location)?)?
                        }
                        DimensionOperation::Max(_) => {
                            block.append_operation(stable_hlo::maximum(*left, *right, location)?)?
                        }
                        _ => unreachable!(),
                    };
                    Ok(vec![result.result(0).unwrap().as_ref()])
                }
                _ => Err(LoweringError::UnsupportedOp {
                    op: format!("first-class dimension operation `{}` has not been lowered yet", operation.name()),
                }
                .into()),
            }
        }
        ArrayProgramOperation::Compare(operation) => {
            let [left, right] = input_values else {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into());
            };
            // Dimension atoms already lower to scalar `i64` SSA values, so the ordinary comparison lowering can
            // consume them directly without a data gateway or host-side shape reconstruction.
            Ok(vec![lower_compare_to_mlir(operation.direction(), *left, *right, block, location)?])
        }
        ArrayProgramOperation::DimensionSize(operation) => {
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
        ArrayProgramOperation::DimensionFromScalar(_) => Err(LoweringError::UnsupportedOp {
            op: "dimension_from_scalar requires checked runtime assertion lowering".to_string(),
        }),
        ArrayProgramOperation::DimensionToScalar(_) => {
            let [input] = input_values else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
            };
            Ok(vec![*input])
        }
        ArrayProgramOperation::Reshape(operation) => {
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
        ArrayProgramOperation::Broadcast(operation) => {
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
            } else if input_type.static_shape().is_some() {
                // A statically shaped input can be broadcast to the finite physical upper-bound shape and then
                // refined directly from the explicit extent operands. This avoids dynamic_broadcast_in_dim, which
                // some XLA importers cannot legalize, without recovering geometry from array data.
                let dynamic_extents = output_type
                    .shape()
                    .dimensions()
                    .iter()
                    .zip(output_extents)
                    .filter_map(|(dimension, extent)| matches!(dimension, Dimension::Dynamic(_)).then_some(*extent))
                    .collect::<Vec<_>>();
                let (declared_type, physical_type) =
                    dynamic_constructor_types(operation.name(), dynamic_extents.len(), output_types)?;
                let output_tensor_type = lower_tensor_type(&physical_type, context, location)?;
                let broadcast = block.append_operation(stable_hlo::broadcast(
                    *input,
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
        ArrayProgramOperation::Concatenate(operation) => {
            let Some((_result_extent, array_inputs)) = input_values.split_last() else {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }.into());
            };
            let Some((result_extent_type, array_input_types)) = input_types.split_last() else {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_types.len() }.into());
            };
            if array_inputs.is_empty() {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into());
            }
            let result_extent_type =
                <&DimensionType>::try_from(result_extent_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            let input_extents_are_static = array_input_types.iter().try_fold(true, |all_static, r#type| {
                let r#type = <&ArrayType>::try_from(r#type).map_err(|error| LoweringError::Tracing(error.into()))?;
                Ok::<_, LoweringError>(
                    all_static && matches!(r#type.shape().dimensions()[operation.axis()], Dimension::Static(_)),
                )
            })?;
            if !input_extents_are_static || matches!(result_extent_type.to_dimension(), Dimension::Dynamic(_)) {
                return Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "{} with first-class dimensions requires runtime equality assertion lowering when its \
                         explicit result extent is not statically proven equal to the input extent sum",
                        CONCATENATE_OPERATION_NAME,
                    ),
                });
            }

            // The mixed type-inference contract has already proven that the exact trailing extent equals the sum of
            // the exact concatenated input axes. The scalar extent is therefore consumed as compile-time shape
            // authority, while StableHLO receives only the physical array operands.
            let result = block.append_operation(stable_hlo::concatenate(array_inputs, operation.axis(), location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.concatenate should return one result").as_ref()])
        }
        ArrayProgramOperation::CustomCall(operation) => {
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
            let physical_output_types = operation
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
                                        "custom-call output dimension '{}' needs a finite upper bound for physical \
                                         buffer allocation",
                                        variable,
                                    ),
                                }),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(output_type.clone().with_shape(Shape::new(dimensions)))
                })
                .collect::<Result<Vec<_>, LoweringError>>()?;
            let mut results = lower_custom_call_to_mlir(
                operation,
                array_inputs,
                physical_output_types.as_slice(),
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
        ArrayProgramOperation::Pad(operation) => {
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
        ArrayProgramOperation::RngBitGenerator(operation) => {
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
        ArrayProgramOperation::AllGather(operation) => {
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
            let axes = (operation.options().mode() == CollectiveMode::Tiled)
                .then_some(operation.concat_axis())
                .into_iter()
                .collect::<Vec<_>>();
            refine_collective_result_axes(
                result,
                axes.as_slice(),
                &input_values[1..],
                output_type,
                block,
                context,
                location,
            )
        }
        ArrayProgramOperation::PSumScatter(operation) => {
            let Some(input) = input_values.first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let [output_type] = output_types else {
                return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
            };
            let output_type =
                <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            let result =
                lower_psum_scatter_to_mlir(operation, collective_state, *input, output_type, block, context, location)?
                    .remove(0);
            let axes = (operation.options().mode() == CollectiveMode::Tiled)
                .then_some(operation.scatter_axis())
                .into_iter()
                .collect::<Vec<_>>();
            refine_collective_result_axes(
                result,
                axes.as_slice(),
                &input_values[1..],
                output_type,
                block,
                context,
                location,
            )
        }
        ArrayProgramOperation::AllToAll(operation) => {
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
            let axes = if operation.options().mode() == CollectiveMode::Tiled
                && operation.split_axis() != operation.concat_axis()
            {
                vec![operation.split_axis(), operation.concat_axis()]
            } else {
                Vec::new()
            };
            refine_collective_result_axes(
                result,
                axes.as_slice(),
                &input_values[1..],
                output_type,
                block,
                context,
                location,
            )
        }
        ArrayProgramOperation::Condition(_) | ArrayProgramOperation::While(_) | ArrayProgramOperation::Scan(_) => {
            Err(LoweringError::UnsupportedOp {
                op: format!(
                    "core composite higher-order operation `{}` must be promoted to the XLA operation family before \
                     lowering",
                    operation.name(),
                ),
            })
        }
    }
}
