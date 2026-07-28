//! StableHLO lowering for programs that mix arrays with first-class dimensions.

use ryft_core::backends::array_programs::{ArrayProgramOperation, ArrayProgramValue};
use ryft_core::backends::dimensions::DimensionOperation;
use ryft_core::operations::manipulation::CONCATENATE_OPERATION_NAME;
use ryft_core::parameters::Parameterized;
use ryft_core::programs::{Operation as CoreOperation, Program, ProgramError, Typed};
use ryft_core::sharding::LogicalMesh;
use ryft_core::types::{ArrayProgramType, ArrayType, DataType, Dimension, DimensionType, MAX_DIMENSION_EXTENT, Shape};
use ryft_mlir::dialects::{func, stable_hlo, tensor};
use ryft_mlir::{
    Block, Context as MlirContext, Location, Operation as MlirOperation, Region, Size as MlirSize, Type as MlirType,
    TypeAndAttributes, Value as MlirValue, ValueRef,
};

use super::{
    LowerableXlaOperation, LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode,
    broadcast_changes_explicit_sharding, lower_compare_to_mlir, lower_constant_elements_attribute,
    lower_constant_output, lower_custom_call_to_mlir, lower_pad_to_mlir, lower_rng_bit_generator_to_mlir,
    lower_sharding_constraint, lower_tensor_type, merge_logical_meshes, normalize_function_name,
    replay_region_ref_into_block, reshape_dimension_i64, stable_hlo_dynamic_dimension_bound, static_dimensions,
};
use crate::experimental::ops::XlaConstant;
use crate::mlir::ToMlir;

/// Error returned while lowering a stored program that mixes arrays and first-class dimensions.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ArrayProgramLoweringError {
    /// Program validation or replay failure.
    #[error(transparent)]
    Program(#[from] ProgramError),

    /// StableHLO construction, validation, or backend-specific lowering failure.
    #[error("{message}")]
    Lowering { message: String },
}

impl From<LoweringError> for ArrayProgramLoweringError {
    fn from(error: LoweringError) -> Self {
        match error {
            LoweringError::Tracing(error) => Self::Program(error),
            error => Self::Lowering { message: error.to_string() },
        }
    }
}

/// Lowers a composite array-program type to its physical StableHLO tensor type.
fn lower_array_program_type<'c, 't, L: Location<'c, 't>>(
    r#type: &ArrayProgramType,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
    match r#type {
        ArrayProgramType::Array(r#type) => lower_tensor_type(r#type, context, location),
        ArrayProgramType::Dimension(_) => lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location),
    }
}

/// Lowers one composite constant without converting first-class dimensions into array-program authority.
fn lower_array_program_constant<'b, 'c: 'b, 't: 'c, A: MlirLowerableValue>(
    value: &ArrayProgramValue<A>,
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    match value {
        ArrayProgramValue::Array(value) => value.lower_constant_value(&[], block, context, location),
        ArrayProgramValue::Dimension(value) => {
            let tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
            let extent = i64::try_from(value.extent()).map_err(|_| LoweringError::UnsupportedOp {
                op: format!("dimension extent {} does not fit StableHLO i64", value.extent()),
            })?;
            let elements = lower_constant_elements_attribute(DataType::I64, tensor_type, extent, context)?;
            let operation = block.append_operation(stable_hlo::constant(elements, location)?)?;
            Ok(operation.result(0).expect("stablehlo.constant should return one result").as_ref())
        }
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

/// Lowers one array-program instruction.
fn lower_array_program_operation<'b, 'c: 'b, 't: 'c, A>(
    operation: &ArrayProgramOperation<A>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayProgramType],
    output_types: &[ArrayProgramType],
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
            let mut lowerer = PlainMlirLowerer::new(*block, context, location).with_input_types(input_types);
            operation.lower_to_mlir(
                input_values,
                &[],
                output_types.as_slice(),
                PlainMlirLoweringMode::Unpacked,
                &mut lowerer,
            )
        }
        ArrayProgramOperation::Dimension(operation) => {
            match operation {
                DimensionOperation::Add(_) | DimensionOperation::Mul(_) => {
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
                    let maximum_result =
                        maximum(left_type).zip(maximum(right_type)).and_then(|(left, right)| match operation {
                            DimensionOperation::Add(_) => left.checked_add(right),
                            DimensionOperation::Mul(_) => left.checked_mul(right),
                            _ => unreachable!(),
                        });
                    // A bounds proof makes these physical scalar `i64` operations equivalent to checked dimension
                    // arithmetic. P7 owns runtime overflow assertions for cases whose declared bounds cannot prove
                    // this.
                    if !maximum_result.is_some_and(|result| result <= MAX_DIMENSION_EXTENT) {
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
                        DimensionOperation::Mul(_) => {
                            block.append_operation(stable_hlo::multiply(*left, *right, location)?)?
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
                let shape = lower_explicit_shape(output_extents, block, context, location)?;
                let output_bounds = output_type
                    .shape()
                    .dimensions()
                    .iter()
                    .map(|dimension| match dimension {
                        Dimension::Static(extent) => Some(*extent),
                        Dimension::Dynamic(_) => stable_hlo_dynamic_dimension_bound(dimension),
                    })
                    .collect::<Vec<_>>();
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
    }
}

/// Lowers a stored array program containing ordinary arrays and first-class dimensions to verified StableHLO.
///
/// Array atoms lower to their tensor representation and dimension atoms lower to scalar `i64` tensors. Mixed
/// operations retain their explicit SSA data flow: in particular, a
/// [`DimensionToScalarOperation`](ryft_core::DimensionToScalarOperation) lowers to the identity because both sides of
/// that logical boundary use the same physical scalar representation.
pub fn lower_array_program_to_stable_hlo<
    Input: Parameterized<ArrayProgramValue<XlaConstant>>,
    Output: Parameterized<ArrayProgramValue<XlaConstant>>,
>(
    program: &Program<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>, Input, Output>,
    function_name: &str,
) -> Result<String, ArrayProgramLoweringError> {
    to_mlir_module_for_array_program(program, function_name).map_err(Into::into)
}

/// Generic implementation of [`lower_array_program_to_stable_hlo`] shared with the reference-backend tests.
fn to_mlir_module_for_array_program<
    A,
    Input: Parameterized<ArrayProgramValue<A>>,
    Output: Parameterized<ArrayProgramValue<A>>,
>(
    program: &Program<ArrayProgramValue<A>, ArrayProgramOperation<A>, Input, Output>,
    function_name: &str,
) -> Result<String, LoweringError>
where
    A: MlirLowerableValue,
{
    let function_name = normalize_function_name(function_name)?;
    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location)?;

    let mut mesh: Option<LogicalMesh> = None;
    for region in program.regions().iter() {
        for atom in region.atoms() {
            let atom_type = atom.r#type();
            let ArrayProgramType::Array(r#type) = atom_type.as_ref() else {
                continue;
            };
            let Some(sharding) = r#type.sharding() else {
                continue;
            };
            mesh = Some(match mesh.take() {
                Some(existing) => merge_logical_meshes(&existing, sharding.mesh())?,
                None => sharding.mesh().clone(),
            });
        }
    }
    if let Some(mesh) = mesh {
        module.body()?.append_operation(mesh.to_mlir(location)?)?;
    }

    let input_types = program
        .input_ids()
        .iter()
        .map(|id| lower_array_program_type(program.atoms()[id.index()].r#type().as_ref(), &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let output_types = program
        .output_ids()
        .iter()
        .map(|id| lower_array_program_type(program.atoms()[id.index()].r#type().as_ref(), &context, location))
        .collect::<Result<Vec<_>, _>>()?;

    module.body()?.append_operation({
        let function_block =
            context.block(input_types.iter().map(|r#type| (r#type.as_ref(), location)).collect::<Vec<_>>().as_slice());
        {
            let mut block = function_block.as_ref();
            let arguments = (0..input_types.len())
                .map(|index| block.argument(index).expect("function block arguments should exist").as_ref())
                .collect::<Vec<_>>();
            let outputs = replay_region_ref_into_block(
                program.entry_region_ref(),
                arguments,
                &mut block,
                &context,
                location.as_ref(),
                |_, value, block, context, location| lower_array_program_constant(value, block, context, location),
                |instruction, inputs, block, context, location| {
                    if !instruction.regions().is_empty() {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!(
                                "composite operation `{}` with attached regions is not supported",
                                instruction.operation().name(),
                            ),
                        });
                    }
                    let input_types = instruction
                        .inputs()
                        .iter()
                        .map(|id| program.atoms()[id.index()].r#type().into_owned())
                        .collect::<Vec<_>>();
                    let output_types = instruction
                        .outputs()
                        .iter()
                        .map(|id| program.atoms()[id.index()].r#type().into_owned())
                        .collect::<Vec<_>>();
                    lower_array_program_operation(
                        instruction.operation(),
                        inputs,
                        input_types.as_slice(),
                        output_types.as_slice(),
                        block,
                        context,
                        location,
                    )
                },
            )?;
            block.append_operation(func::r#return(outputs.as_slice(), location)?)?;
        }
        let mut region = context.region();
        region.append_block(function_block)?;
        func::func(
            function_name.as_str(),
            func::FuncAttributes {
                arguments: input_types
                    .iter()
                    .map(|r#type| TypeAndAttributes { r#type: r#type.as_ref(), attributes: None })
                    .collect(),
                results: output_types
                    .iter()
                    .map(|r#type| TypeAndAttributes { r#type: r#type.as_ref(), attributes: None })
                    .collect(),
                ..Default::default()
            },
            region,
            location,
        )?
    })?;

    if !module.verify()? {
        return Err(LoweringError::MlirVerificationFailure);
    }
    Ok(module.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use ryft_core::contexts::{Context, StagingContext};
    use ryft_core::operations::compare::{Compare, ComparisonDirection};
    use ryft_core::operations::custom_call::CustomCallOperation;
    use ryft_core::operations::dimensions::{
        DimensionAddOperation, DimensionFromScalar, DimensionMulOperation, DimensionSize, DimensionSizeOperation,
        DimensionToScalar,
    };
    use ryft_core::operations::manipulation::{
        BroadcastOperation, ConcatenateOperation, PadOperation, ReshapeOperation,
    };
    use ryft_core::operations::random::{RandomAlgorithm, RngBitGeneratorOperation};
    use ryft_core::parameters::Placeholder;
    use ryft_core::programs::ProgramBuilder;
    use ryft_core::programs::values::ValueProjection;
    use ryft_core::tracing::Tracer;
    use ryft_core::tracing::TracingContext;
    use ryft_core::types::dimensions::{DimensionBounds, DimensionVariable};
    use ryft_core::types::{DataType, DimensionType, Shape};
    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use ryft_pjrt::{
        BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
        load_cpu_plugin,
    };

    use super::*;
    use crate::tests::{values_from_bytes, values_to_bytes};

    /// Returns the ordinary single-device compilation options used by the composite CPU execution tests.
    fn cpu_compilation_options() -> CompilationOptions {
        CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 1,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: std::collections::HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        }
    }

    /// Builds the canonical stored concatenate program whose result extent is computed from its array inputs.
    fn explicit_concatenate_program(
        left_type: ArrayType,
        right_type: ArrayType,
    ) -> Program<
        ArrayProgramValue<XlaConstant>,
        ArrayProgramOperation<XlaConstant>,
        Vec<ArrayProgramValue<XlaConstant>>,
        Vec<ArrayProgramValue<XlaConstant>>,
    > {
        let left_size_operation = DimensionSizeOperation::new(&left_type, 0).unwrap();
        let right_size_operation = DimensionSizeOperation::new(&right_type, 0).unwrap();
        let add_operation =
            DimensionAddOperation::new(left_size_operation.result_type(), right_size_operation.result_type()).unwrap();
        let mut builder = ProgramBuilder::<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>::new();
        let left = builder.add_input(left_type.into());
        let right = builder.add_input(right_type.into());
        let left_extent = builder.add_instruction(left_size_operation, Vec::new(), vec![left]).unwrap()[0];
        let right_extent = builder.add_instruction(right_size_operation, Vec::new(), vec![right]).unwrap()[0];
        let result_extent = builder
            .add_instruction(DimensionOperation::Add(add_operation), Vec::new(), vec![left_extent, right_extent])
            .unwrap()[0];
        let output = builder
            .add_instruction(ConcatenateOperation::new(0, 1).unwrap(), Vec::new(), vec![left, right, result_extent])
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_array_program_lowering() {
        type TestContext = TracingContext<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>;

        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(17)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let context = TestContext::new();
        let input = context.input(input_type.into());
        let output = input.dimension_size(0).unwrap().to_scalar().unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let dynamic_module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(dynamic_module.matches("stablehlo.get_dimension_size").count(), 1);
        assert_eq!(dynamic_module.matches("stablehlo.convert").count(), 1);
        assert_eq!(dynamic_module.matches("dimension_to_scalar").count(), 0);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(7)]));
        let context = TestContext::new();
        let input = context.input(input_type.into());
        let output = input.dimension_size(0).unwrap().to_scalar().unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let static_module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(static_module.matches("stablehlo.constant dense<7> : tensor<i64>").count(), 1);
        assert_eq!(static_module.matches("stablehlo.get_dimension_size").count(), 0);
        assert_eq!(static_module.matches("stablehlo.convert").count(), 0);

        // Compile and execute the static path on the real CPU plugin. The bounded-dynamic module above is verified as
        // StableHLO, but the CPU XLA plugin currently lowers bounded arguments through its unavailable `PadToStatic`
        // custom call. Static execution still proves that the first-class dimension remains scalar SSA through
        // `dimension_to_scalar`, without a host readback or reconstruction step.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let options = cpu_compilation_options();
        let executable = client.compile(&PjrtProgram::Mlir { bytecode: static_module.into_bytes() }, &options).unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        let input_values = [0.0_f32; 7];
        let input_bytes = values_to_bytes(input_values.as_slice());
        let inputs = ExecutionDeviceInputs {
            inputs: &[ExecutionInput {
                buffer: Arc::new(
                    client.buffer(input_bytes.as_slice(), BufferType::F32, &[7], None, device, None).unwrap(),
                ),
                donatable: false,
            }],
            ..Default::default()
        };
        let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
        let mut outputs = execution.block_until_ready().unwrap().remove(0);
        assert_eq!(outputs.outputs.len(), 1);
        let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<i64>(output_bytes.as_slice()), vec![7]);
    }

    #[test]
    fn test_explicit_custom_call_lowering() {
        type TestContext = TracingContext<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>;

        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable.clone())]));
        let context = TestContext::new();
        let input = context.input(output_type.clone().into());
        let output_extent = input.dimension_size(0).unwrap();
        let output = context
            .bind(CustomCallOperation::new("ryft.test.dynamic", vec![output_type]), Vec::new(), &[input, output_extent])
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(module.matches("stablehlo.custom_call @ryft.test.dynamic").count(), 1);
        assert!(
            module.contains("stablehlo.custom_call @ryft.test.dynamic(%arg0)"),
            "the logical extent operand must not enter the foreign-kernel ABI:\n{module}",
        );
        assert_eq!(module.matches("stablehlo.get_dimension_size").count(), 1);
        assert_eq!(module.matches("stablehlo.set_dimension_size").count(), 1);
        assert!(module.contains("-> tensor<8xf32>"), "{module}");

        let variable = DimensionVariable::new("unbounded", DimensionBounds::unbounded());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable.clone())]));
        let context = TestContext::new();
        let input = context.input(output_type.clone().into());
        let output_extent = input.dimension_size(0).unwrap();
        let output = context
            .bind(CustomCallOperation::new("ryft.test.dynamic", vec![output_type]), Vec::new(), &[input, output_extent])
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            lower_array_program_to_stable_hlo(&program, "main"),
            Err(ArrayProgramLoweringError::Lowering {
                message: "unsupported staged op 'custom-call output dimension 'unbounded' needs a finite upper bound \
                          for physical buffer allocation' during XLA lowering"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_explicit_pad_lowering() {
        let input_variable = DimensionVariable::new("input", DimensionBounds::new(1, Some(5)).unwrap());
        let output_variable = DimensionVariable::new("output", DimensionBounds::new(3, Some(7)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(input_variable)]));
        let padding_value_type = ArrayType::scalar(DataType::F32);
        let output_extent_type = DimensionType::new(output_variable);
        let mut builder = ProgramBuilder::<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>::new();
        let input = builder.add_input(input_type.into());
        let padding_value = builder.add_input(padding_value_type.into());
        let output_extent = builder.add_input(output_extent_type.into());
        let output = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![1], vec![0]).unwrap(),
                Vec::new(),
                vec![input, padding_value, output_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(module.matches("stablehlo.pad").count(), 1);
        assert_eq!(module.matches("stablehlo.set_dimension_size").count(), 1);
        assert_eq!(module.matches("stablehlo.get_dimension_size").count(), 0);
    }

    #[test]
    fn test_explicit_rng_bit_generator_lowering() {
        let mut builder = ProgramBuilder::<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>::new();
        let state = builder.add_input(RandomAlgorithm::ThreeFry.state_type().into());
        let outputs = builder
            .add_instruction(
                RngBitGeneratorOperation::new(
                    RandomAlgorithm::ThreeFry,
                    ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(8)])),
                ),
                Vec::new(),
                vec![state],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                outputs,
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(module.matches("stablehlo.rng_bit_generator").count(), 1);
        assert_eq!(module.matches("stablehlo.set_dimension_size").count(), 0);

        let output_variable = DimensionVariable::new("count", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(output_variable.clone())]));
        let mut builder = ProgramBuilder::<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>::new();
        let state = builder.add_input(RandomAlgorithm::ThreeFry.state_type().into());
        let output_extent = builder.add_input(DimensionType::new(output_variable).into());
        let outputs = builder
            .add_instruction(
                RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, output_type),
                Vec::new(),
                vec![state, output_extent],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                outputs,
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(
            lower_array_program_to_stable_hlo(&program, "main"),
            Err(ArrayProgramLoweringError::Lowering {
                message: "unsupported staged op 'rng-bit-generator with dynamic output extents cannot lower by \
                          generating the physical upper-bound shape because that would advance its functional state \
                          by the physical rather than logical element count' during XLA lowering"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_explicit_concatenate_lowering() {
        // Compute the exact result extent through ordinary dimension SSA and prove that lowering retains only the
        // physical arrays as StableHLO concatenate operands.
        let left_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let right_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let program = explicit_concatenate_program(left_type, right_type);
        assert_eq!(program.instructions().len(), 4);

        let module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(module.matches("stablehlo.get_dimension_size").count(), 0);
        assert_eq!(module.matches("stablehlo.add").count(), 1);
        assert_eq!(module.matches("stablehlo.concatenate").count(), 1);
        assert!(
            module.contains(
                "stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<2xf32>, tensor<3xf32>) -> tensor<5xf32>",
            ),
            "{module}",
        );

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        let left_values = [1.0_f32, 2.0];
        let right_values = [3.0_f32, 4.0, 5.0];
        let left_bytes = values_to_bytes(left_values.as_slice());
        let right_bytes = values_to_bytes(right_values.as_slice());
        let inputs = ExecutionDeviceInputs {
            inputs: &[
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(left_bytes.as_slice(), BufferType::F32, &[2], None, device.clone(), None)
                            .unwrap(),
                    ),
                    donatable: false,
                },
                ExecutionInput {
                    buffer: Arc::new(
                        client.buffer(right_bytes.as_slice(), BufferType::F32, &[3], None, device, None).unwrap(),
                    ),
                    donatable: false,
                },
            ],
            ..Default::default()
        };
        let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
        let mut outputs = execution.block_until_ready().unwrap().remove(0);
        assert_eq!(outputs.outputs.len(), 1);
        let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]);

        // A dynamic input sum still has an explicit SSA extent, but lowering cannot trust it until the assertion
        // effect can verify the equality at runtime.
        let left_variable = DimensionVariable::new("left", DimensionBounds::new(1, Some(5)).unwrap());
        let right_variable = DimensionVariable::new("right", DimensionBounds::new(1, Some(6)).unwrap());
        let left_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(left_variable)]));
        let right_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(right_variable)]));
        let program = explicit_concatenate_program(left_type, right_type);
        assert_eq!(
            lower_array_program_to_stable_hlo(&program, "main"),
            Err(ArrayProgramLoweringError::Lowering {
                message: "unsupported staged op 'concatenate with first-class dimensions requires runtime equality \
                          assertion lowering when its explicit result extent is not statically proven equal to the \
                          input extent sum' during XLA lowering"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_explicit_reshape_lowering() {
        type TestContext = TracingContext<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>;

        let context = TestContext::new();
        let input = context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(6)])).into());
        let first_extent =
            context.constant(ArrayProgramValue::Dimension(ryft_core::DimensionValue::constant(2).unwrap()));
        let second_extent =
            context.constant(ArrayProgramValue::Dimension(ryft_core::DimensionValue::constant(3).unwrap()));
        let output = context
            .bind(ReshapeOperation::new(), Vec::new(), &[input, first_extent, second_extent])
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let static_module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(static_module.matches("stablehlo.reshape").count(), 1);
        assert_eq!(static_module.matches("stablehlo.dynamic_reshape").count(), 0);
        assert_eq!(static_module.matches("stablehlo.get_dimension_size").count(), 0);
        assert_eq!(static_module.matches("dimension_to_scalar").count(), 0);

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: static_module.into_bytes() }, &cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        let input_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input_bytes = values_to_bytes(input_values.as_slice());
        let inputs = ExecutionDeviceInputs {
            inputs: &[ExecutionInput {
                buffer: Arc::new(
                    client.buffer(input_bytes.as_slice(), BufferType::F32, &[6], None, device, None).unwrap(),
                ),
                donatable: false,
            }],
            ..Default::default()
        };
        let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
        let mut outputs = execution.block_until_ready().unwrap().remove(0);
        assert_eq!(outputs.outputs.len(), 1);
        let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), input_values);

        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let context = TestContext::new();
        let input = context.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable), Dimension::Static(4)])).into(),
        );
        let first_extent = input.dimension_size(0).unwrap();
        let second_extent =
            context.constant(ArrayProgramValue::Dimension(ryft_core::DimensionValue::constant(4).unwrap()));
        let output = context
            .bind(ReshapeOperation::new(), Vec::new(), &[input, first_extent, second_extent])
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let dynamic_module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(dynamic_module.matches("stablehlo.get_dimension_size").count(), 1);
        assert_eq!(dynamic_module.matches("stablehlo.dynamic_reshape").count(), 1);
        assert_eq!(dynamic_module.matches("stablehlo.concatenate").count(), 1);
        assert_eq!(dynamic_module.matches("dimension_to_scalar").count(), 0);
    }

    #[test]
    fn test_explicit_broadcast_lowering() {
        type TestContext = TracingContext<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>;

        let context = TestContext::new();
        let input = context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into());
        let first_extent =
            context.constant(ArrayProgramValue::Dimension(ryft_core::DimensionValue::constant(3).unwrap()));
        let second_extent =
            context.constant(ArrayProgramValue::Dimension(ryft_core::DimensionValue::constant(2).unwrap()));
        let output = context
            .bind(BroadcastOperation::new(vec![1]), Vec::new(), &[input, first_extent, second_extent])
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let static_module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(static_module.matches("stablehlo.broadcast_in_dim").count(), 1);
        assert_eq!(static_module.matches("stablehlo.dynamic_broadcast_in_dim").count(), 0);
        assert_eq!(static_module.matches("stablehlo.get_dimension_size").count(), 0);
        assert_eq!(static_module.matches("dimension_to_scalar").count(), 0);

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: static_module.into_bytes() }, &cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        let input_values = [1.0_f32, 2.0];
        let input_bytes = values_to_bytes(input_values.as_slice());
        let inputs = ExecutionDeviceInputs {
            inputs: &[ExecutionInput {
                buffer: Arc::new(
                    client.buffer(input_bytes.as_slice(), BufferType::F32, &[2], None, device, None).unwrap(),
                ),
                donatable: false,
            }],
            ..Default::default()
        };
        let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
        let mut outputs = execution.block_until_ready().unwrap().remove(0);
        let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), vec![1.0_f32, 2.0, 1.0, 2.0, 1.0, 2.0],);

        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let context = TestContext::new();
        let input = context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).into());
        let first_extent = context.input(DimensionType::new(variable).into());
        let second_extent =
            context.constant(ArrayProgramValue::Dimension(ryft_core::DimensionValue::constant(1).unwrap()));
        let output = context
            .bind(BroadcastOperation::new(vec![1]), Vec::new(), &[input, first_extent, second_extent])
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let dynamic_module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(dynamic_module.matches("stablehlo.dynamic_broadcast_in_dim").count(), 1);
        assert_eq!(dynamic_module.matches("stablehlo.concatenate").count(), 1);
        assert_eq!(dynamic_module.matches("stablehlo.get_dimension_size").count(), 0);
        assert_eq!(dynamic_module.matches("dimension_to_scalar").count(), 0);
    }

    #[test]
    fn test_explicit_shape_vertical_slice_lowering() {
        type TestContext = TracingContext<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>;

        // The static program uses the same computed dimension SSA value as a reshape and broadcast operand.
        let context = TestContext::new();
        let input = context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into());
        let one_value = ryft_core::DimensionValue::constant(1).unwrap();
        let one_type = one_value.r#type().clone();
        let one = context.constant(ArrayProgramValue::Dimension(one_value));
        let two = context
            .bind(
                DimensionOperation::Add(DimensionAddOperation::new(&one_type, &one_type).unwrap()),
                Vec::new(),
                &[one.clone(), one.clone()],
            )
            .unwrap()
            .remove(0);
        let reshaped = context.bind(ReshapeOperation::new(), Vec::new(), &[input, one, two.clone()]).unwrap().remove(0);
        let output = context
            .bind(BroadcastOperation::new(vec![0, 1]), Vec::new(), &[reshaped, two.clone(), two])
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(program.instructions().len(), 3);
        let static_module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(static_module.matches("stablehlo.add").count(), 1);
        assert_eq!(static_module.matches("stablehlo.reshape").count(), 1);
        assert_eq!(static_module.matches("stablehlo.broadcast_in_dim").count(), 1);
        assert_eq!(static_module.matches("stablehlo.dynamic_reshape").count(), 0);
        assert_eq!(static_module.matches("stablehlo.dynamic_broadcast_in_dim").count(), 0);
        assert_eq!(static_module.matches("stablehlo.get_dimension_size").count(), 0);
        assert_eq!(static_module.matches("dimension_to_scalar").count(), 0);

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: static_module.into_bytes() }, &cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        let input_values = [1.0_f32, 2.0];
        let input_bytes = values_to_bytes(input_values.as_slice());
        let inputs = ExecutionDeviceInputs {
            inputs: &[ExecutionInput {
                buffer: Arc::new(
                    client.buffer(input_bytes.as_slice(), BufferType::F32, &[2], None, device, None).unwrap(),
                ),
                donatable: false,
            }],
            ..Default::default()
        };
        let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
        let mut outputs = execution.block_until_ready().unwrap().remove(0);
        let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), vec![1.0_f32, 2.0, 1.0, 2.0]);

        // The dynamic program proves that arithmetic results remain scalar SSA edges through both dynamic shape
        // operations. The pinned XLA translator cannot compile dynamic_broadcast_in_dim, so this case is structural.
        let extent_variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let extent_type = DimensionType::new(extent_variable.clone());
        let context = TestContext::new();
        let input =
            context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent_variable)])).into());
        let extent = context.input(extent_type.clone().into());
        let one_value = ryft_core::DimensionValue::constant(1).unwrap();
        let one_type = one_value.r#type().clone();
        let one = context.constant(ArrayProgramValue::Dimension(one_value));
        let repeated_extent = context
            .bind(
                DimensionOperation::Mul(DimensionMulOperation::new(&extent_type, &one_type).unwrap()),
                Vec::new(),
                &[extent, one.clone()],
            )
            .unwrap()
            .remove(0);
        let two = context
            .bind(
                DimensionOperation::Add(DimensionAddOperation::new(&one_type, &one_type).unwrap()),
                Vec::new(),
                &[one.clone(), one.clone()],
            )
            .unwrap()
            .remove(0);
        let reshaped = context
            .bind(ReshapeOperation::new(), Vec::new(), &[input, one, repeated_extent.clone()])
            .unwrap()
            .remove(0);
        let output = context
            .bind(BroadcastOperation::new(vec![0, 1]), Vec::new(), &[reshaped, two, repeated_extent])
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(program.instructions().len(), 4);
        let dynamic_module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(dynamic_module.matches("stablehlo.multiply").count(), 1);
        assert_eq!(dynamic_module.matches("stablehlo.add").count(), 1);
        assert_eq!(dynamic_module.matches("stablehlo.dynamic_reshape").count(), 1);
        assert_eq!(dynamic_module.matches("stablehlo.dynamic_broadcast_in_dim").count(), 1);
        assert_eq!(dynamic_module.matches("stablehlo.get_dimension_size").count(), 0);
        assert_eq!(dynamic_module.matches("dimension_to_scalar").count(), 0);

        // Bounds that cannot prove representable arithmetic remain deferred to checked assertion lowering instead of
        // silently accepting StableHLO integer overflow.
        let left_type = DimensionType::new(DimensionVariable::new("left", DimensionBounds::unbounded()));
        let right_type = DimensionType::new(DimensionVariable::new("right", DimensionBounds::unbounded()));
        let context = TestContext::new();
        let left = context.input(left_type.clone().into());
        let right = context.input(right_type.clone().into());
        let output = context
            .bind(
                DimensionOperation::Add(DimensionAddOperation::new(&left_type, &right_type).unwrap()),
                Vec::new(),
                &[left, right],
            )
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            lower_array_program_to_stable_hlo(&program, "main"),
            Err(ArrayProgramLoweringError::Lowering {
                message: "unsupported staged op 'first-class dimension operation `dimension_add` requires checked \
                          runtime assertion lowering' during XLA lowering"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_dimension_comparison_lowering() {
        type TestContext = TracingContext<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        for (direction, stable_hlo_direction, left_extent, right_extent) in
            [(ComparisonDirection::Equal, "EQ", 3_i64, 3_i64), (ComparisonDirection::LessThan, "LT", 3_i64, 5_i64)]
        {
            let bounds = DimensionBounds::new(0, Some(17)).unwrap();
            let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
            let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
            let context = TestContext::new();
            let left = context.input(left_type.into());
            let right = context.input(right_type.into());
            let left = <Tracer<TestContext> as ValueProjection<DimensionType>>::into_projected(left).unwrap();
            let right = <Tracer<TestContext> as ValueProjection<DimensionType>>::into_projected(right).unwrap();
            let output = left.compare(&right, direction).unwrap();
            let program = context
                .builder()
                .borrow()
                .clone()
                .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                    vec![output.atom_id().unwrap()],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap();

            let module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
            assert_eq!(module.matches("stablehlo.compare").count(), 1);
            assert!(module.contains(&format!("stablehlo.compare {stable_hlo_direction}")), "{module}");
            assert!(module.contains("SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>"), "{module}");
            assert_eq!(module.matches("dimension_to_scalar").count(), 0);

            let executable = client
                .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &cpu_compilation_options())
                .unwrap();
            let device = executable.addressable_devices().unwrap()[0].clone();
            let left_bytes = values_to_bytes(&[left_extent]);
            let right_bytes = values_to_bytes(&[right_extent]);
            let inputs = ExecutionDeviceInputs {
                inputs: &[
                    ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(left_bytes.as_slice(), BufferType::I64, &[], None, device.clone(), None)
                                .unwrap(),
                        ),
                        donatable: false,
                    },
                    ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(right_bytes.as_slice(), BufferType::I64, &[], None, device.clone(), None)
                                .unwrap(),
                        ),
                        donatable: false,
                    },
                ],
                ..Default::default()
            };
            let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
            let mut outputs = execution.block_until_ready().unwrap().remove(0);
            assert_eq!(outputs.outputs.len(), 1);
            let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<u8>(output_bytes.as_slice()), vec![1]);
        }
    }

    #[test]
    fn test_dimension_from_scalar_lowering_is_deferred() {
        type TestContext = TracingContext<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>;

        let variable = DimensionVariable::new("extent", DimensionBounds::new(0, Some(17)).unwrap());
        let context = TestContext::new();
        let input = context.input(ArrayType::scalar(DataType::I32).into());
        let output = input.to_dimension(variable).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            lower_array_program_to_stable_hlo(&program, "main"),
            Err(ArrayProgramLoweringError::Lowering {
                message: "unsupported staged op 'dimension_from_scalar requires checked runtime assertion lowering' \
                          during XLA lowering"
                    .to_string(),
            }),
        );
    }
}
