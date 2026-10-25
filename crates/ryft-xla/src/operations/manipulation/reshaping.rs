use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, Dimension, DynamicReshapeOperation, EffectClass, Operation, ProgramError, ReshapeOperation,
};
use ryft_mlir::dialects::{stable_hlo, tensor};
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Type, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::EffectTokens;
use crate::experimental::lowering::types::{
    lower_explicit_reshape_shape, lower_runtime_dimension_size_i64, lower_static_index_constants, lower_tensor_type,
    physical_bound_type, reshape_dimension_i32, reshape_dimension_i64, stable_hlo_dynamic_dimension_bound,
    static_dimensions,
};
use crate::operations::assertions::{
    ASSERT_ACTOR_ATTRIBUTE, ASSERT_DETAIL_ATTRIBUTE, ASSERT_KIND_ATTRIBUTE, ASSERT_RESHAPE_KIND,
    lower_assertion_custom_call, lower_deliberately_false_assertion_predicate,
};
use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for ReshapeOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_types, 1, ProgramError);
        lower_reshape_to_mlir(
            self,
            &input_types[0],
            input_values,
            output_types,
            &lowerer.collective_state.bound_manual_axes,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

/// Lowers a [`ReshapeOperation`] after validating its unary input and single output contract. A static output lowers
/// to `stablehlo.reshape`. A dynamic output is either the input's own shape, which needs no operation, or that shape
/// with static singleton axes inserted or removed, which lowers to `stablehlo.dynamic_reshape` with a runtime shape
/// that reads each dynamic extent back from the input.
#[allow(clippy::too_many_arguments)]
fn lower_reshape_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ReshapeOperation,
    input_type: &ArrayType,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    bound_manual_axes: &[String],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let input = input_values[0];
    let result = if output_types[0].static_shape().is_none() {
        if input_type.shape() == output_types[0].shape() {
            // The dynamic target is exactly the input shape, so the input already has the required result type.
            input
        } else {
            // Core type inference admits a dynamic target only when it inserts or removes static singleton axes
            // around the input's non-singleton dimensions, so every dynamic output axis corresponds in order
            // to a dynamic input axis whose runtime size is read back, and every static output axis is a constant.
            let mut input_axes = input_type
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .filter(|(_, dimension)| **dimension != Dimension::Static(1))
                .map(|(axis, _)| axis);
            let mut extents = Vec::with_capacity(output_types[0].rank());
            for dimension in output_types[0].shape().dimensions() {
                extents.push(match dimension {
                    Dimension::Static(1) => lower_static_index_constants(&[1], block, context, location)?[0],
                    Dimension::Static(extent) => {
                        input_axes.next();
                        lower_static_index_constants(&[*extent], block, context, location)?[0]
                    }
                    Dimension::Dynamic(_) => {
                        let axis = input_axes.next().ok_or_else(|| LoweringError::UnsupportedOp {
                            op: format!(
                                "reshape from {} to {} does not preserve the non-singleton dimensions in order",
                                input_type.shape(),
                                output_types[0].shape(),
                            ),
                        })?;
                        lower_runtime_dimension_size_i64(input, axis, block, context, location)?
                    }
                });
            }
            let shape = lower_explicit_reshape_shape(&extents, block, context, location)?;
            let output_bounds = output_types[0]
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
            let expected_type = lower_tensor_type(&output_types[0], context, location)?;
            if result.r#type()? == expected_type.as_ref() {
                result
            } else {
                let cast = block.append_operation(tensor::cast(result, expected_type, location)?)?;
                cast.result(0).expect("tensor.cast should return one result").as_ref()
            }
        }
    } else {
        let output_shape = static_dimensions(&output_types[0])?;
        for dimension in &output_shape {
            reshape_dimension_i64(*dimension)?;
        }
        let reshape = block.append_operation(stable_hlo::reshape(input, output_shape.as_slice(), location)?)?;
        reshape.result(0).expect("stablehlo.reshape should return one result").as_ref()
    };
    if operation.output_sharding().is_some() {
        let output_sharding = output_types[0]
            .sharding()
            .expect("reshape type inference should preserve a requested output sharding");
        lower_sharding_constraint(&[result], output_sharding, bound_manual_axes, block, location)
    } else {
        Ok(vec![result])
    }
}

/// Checks equality of the input and output logical element counts without overflowing device-side arithmetic.
pub(crate) fn lower_reshape_element_count_assertion<'b, 'c: 'b, 't: 'c>(
    input_extents: &[ValueRef<'b, 'c, 't>],
    output_extents: &[ValueRef<'b, 'c, 't>],
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let observed = input_extents.iter().chain(output_extents).copied().collect::<Vec<_>>();
    let seed = match observed.first() {
        Some(value) => *value,
        None => lower_static_index_constants(&[1], block, context, location)?[0],
    };
    let predicate = lower_deliberately_false_assertion_predicate(seed, block, context, location)?;
    let input_rank = input_extents.len().to_string();
    let backend_config = context.dictionary_attribute(&[
        context.named_attribute(context.identifier(ASSERT_ACTOR_ATTRIBUTE), context.string_attribute("reshape")),
        context
            .named_attribute(context.identifier(ASSERT_KIND_ATTRIBUTE), context.string_attribute(ASSERT_RESHAPE_KIND)),
        context.named_attribute(
            context.identifier(ASSERT_DETAIL_ATTRIBUTE),
            context.string_attribute(input_rank.as_str()),
        ),
    ]);
    lower_assertion_custom_call(predicate, &observed, backend_config, effect_tokens, block, context, location)
}

impl LowerableOperation for DynamicReshapeOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let context = lowerer.context;
        let location = lowerer.location;
        let collective_state = &lowerer.collective_state;
        let effect_tokens = &mut lowerer.effect_tokens;
        let block = &mut lowerer.block;
        let Some((input, output_extents)) = input_values.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let [output_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
        };
        let output_type = <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
        let Some((input_type, _)) = input_types.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let input_type = <&ArrayType>::try_from(input_type).map_err(|error| LoweringError::Tracing(error.into()))?;
        if output_extents.len() != output_type.rank() {
            return Err(ProgramError::InvalidInputCount {
                expected: output_type.rank() + 1,
                actual: input_values.len(),
            }
            .into());
        }

        // Native reshape requires equal allocation capacities even when the runtime logical products agree.
        // Reject unsupported bound combinations here rather than letting HLO import fail with a static-count error.
        let physical_input = physical_bound_type(input_type).map_err(|_| LoweringError::UnsupportedOp {
            op: "reshape requires finite physical bounds for every dynamic input dimension".to_string(),
        })?;
        let physical_output = physical_bound_type(output_type).map_err(|_| LoweringError::UnsupportedOp {
            op: "reshape requires finite physical bounds for every dynamic output dimension".to_string(),
        })?;
        let input_capacity = physical_input.element_count().map_err(ProgramError::from)?.unwrap();
        let output_capacity = physical_output.element_count().map_err(ProgramError::from)?.unwrap();
        if input_capacity != output_capacity {
            return Err(LoweringError::UnsupportedOp {
                op: format!(
                    "reshape with unequal physical capacities (input {input_capacity}, output {output_capacity}); \
                     dynamic bounds must preserve the physical element count",
                ),
            });
        }

        if self.effects().classes().contains(EffectClass::OrderedAssertion) {
            // Keep products on the checked host assertion path. Multiplying extents in i64 on the device could
            // overflow before the mismatch is diagnosed, or reject a valid empty shape with large other axes.
            let mut input_extents = Vec::with_capacity(input_type.rank());
            for (axis, dimension) in input_type.shape().dimensions().iter().enumerate() {
                let extent = match dimension {
                    Dimension::Static(extent) => lower_static_index_constants(&[*extent], block, context, location)?[0],
                    Dimension::Dynamic(_) => lower_runtime_dimension_size_i64(*input, axis, block, context, location)?,
                };
                input_extents.push(extent);
            }
            lower_reshape_element_count_assertion(
                &input_extents,
                output_extents,
                effect_tokens,
                block,
                context,
                location,
            )?;
        }
        let input = *input;
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
        if self.output_sharding().is_some() {
            let output_sharding =
                output_type.sharding().expect("reshape type inference should preserve requested output sharding");
            lower_sharding_constraint(&[result], output_sharding, &collective_state.bound_manual_axes, block, location)
        } else {
            Ok(vec![result])
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayOperation, BatchAxis, BatchableOperation, BatchedProgram,
        BatchingContext, ConstantOperation, Context, DataType, DimensionBounds, DimensionFromScalarOperation,
        DimensionOperation, DimensionType, DimensionValue, DimensionVariable, DynamicReshapeOperation, EffectClass,
        EffectClasses, EmptyRegionDriver, LogicalMesh, MeshAxis, MeshAxisType, Operation, Placeholder,
        ProgramBatchingOutputAxesPolicy, ProgramError, Reshape, Shape, Sharding, ShardingDimension, StagingContext,
        TracedShardMap, TracingContext, TransposeOperation, Typed, trace_shard_map,
    };
    use ryft_mlir::Location;
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::experimental::XlaArrayTracer;
    use crate::experimental::lowering::tests::{
        MixedValue, PlainXlaProgram, XlaProgramBuilder, dynamic_dimension, execute_mixed_program,
        ragged_dot_cpu_compilation_options, test_matrix_type, test_vector_type, unproject_plain_program,
    };
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::{
        FlatXlaProgram, XlaArrayConstant, XlaConstant, XlaOperation, XlaProgramBuilder as ArrayIrXlaProgramBuilder,
    };
    use crate::operations::sharding::shard_map::to_mlir_module;
    use crate::tests::{execution_client, values_to_bytes};
    use crate::{XlaDomain, to_mlir_module_for_program};

    use super::*;

    /// Builds one homogeneous reshape program over an `f64` input for the dynamic singleton lowering tests.
    fn plain_reshape_program(input_shape: Shape, output_shape: Shape) -> PlainXlaProgram {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, input_shape));
        let output =
            builder.add_instruction(ReshapeOperation::new(output_shape), Vec::new(), vec![input], None).unwrap()[0];
        builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    /// Builds the mixed reshape of a static `f64[6]` input to a dynamic `f64[rows]` output whose bound keeps the
    /// physical capacity at six elements, so only a runtime extent of six passes the element-count assertion.
    fn static_input_dynamic_output_reshape_program() -> FlatXlaProgram {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(7)).unwrap());
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [6]).into());
        let extent = builder.add_input(DimensionType::from(rows).into());
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, extent], None)
            .unwrap()[0];
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
    }

    #[test]
    fn test_lower_reshape_element_count_assertion() {
        let client = execution_client();
        crate::operations::assertions::ensure_assertion_handler_registered(&client).unwrap();
        for static_output in [false, true] {
            let input_type = ArrayType::new(DataType::I64, Shape::new(vec![dynamic_dimension("input", Some(5))]));
            let size_type = ArrayType::scalar(DataType::I64);
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let input = builder.add_input(input_type.clone().into());
            let size = builder.add_input(size_type.clone().into());
            let dimension_type = if static_output {
                DimensionValue::constant(4).unwrap().r#type().into_owned()
            } else {
                DimensionType::new("output", DimensionBounds::new(0, Some(5)).unwrap())
            };
            let dimension = if static_output {
                builder
                    .add_instruction(
                        DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(4).unwrap())),
                        Vec::new(),
                        Vec::new(),
                        None,
                    )
                    .unwrap()[0]
            } else {
                builder
                    .add_instruction(
                        DimensionFromScalarOperation::new(dimension_type.clone()),
                        Vec::new(),
                        vec![size],
                        None,
                    )
                    .unwrap()[0]
            };
            let operation = DynamicReshapeOperation::new()
                .with_input_types(&[input_type.clone().into(), dimension_type.into()])
                .unwrap();
            let result = builder.add_instruction(operation, Vec::new(), vec![input, dimension], None).unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![result], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            let output_type = <&ArrayType>::try_from(&program.output_types()[0]).unwrap().clone();
            let module = to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type, size_type],
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
            let cases = if static_output {
                vec![(4_i64, 4_i64), (3, 4), (0, 4)]
            } else {
                vec![(4, 4), (2, 2), (0, 0), (4, 3), (4, 0)]
            };
            // Hidden i32 boundary extents describe the input's logical size separately from its four-element storage.
            for (input_size, output_size) in cases {
                let values = [i64::MAX, i64::MIN, 9_007_199_254_740_993, -9_007_199_254_740_993];
                let inputs = vec![
                    ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(
                                    values_to_bytes(&values).as_slice(),
                                    BufferType::I64,
                                    [4],
                                    None,
                                    device.clone(),
                                    None,
                                )
                                .unwrap(),
                        ),
                        donatable: false,
                    },
                    ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(
                                    values_to_bytes(&[output_size]).as_slice(),
                                    BufferType::I64,
                                    [],
                                    None,
                                    device.clone(),
                                    None,
                                )
                                .unwrap(),
                        ),
                        donatable: false,
                    },
                    ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(
                                    values_to_bytes(&[input_size as i32]).as_slice(),
                                    BufferType::I32,
                                    [],
                                    None,
                                    device.clone(),
                                    None,
                                )
                                .unwrap(),
                        ),
                        donatable: false,
                    },
                ];
                let execution = executable
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
                if input_size != output_size {
                    assert!(execution.err().unwrap().to_string().contains("`reshape`"));
                } else {
                    let outputs = execution.unwrap().remove(0).outputs;
                    let bytes = outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
                    assert_eq!(&bytes[..input_size as usize * 8], values_to_bytes(&values[..input_size as usize]));
                    if !static_output {
                        assert_eq!(
                            outputs[1].copy_to_host(None).unwrap().r#await().unwrap(),
                            values_to_bytes(&[output_size]),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_lower_reshape_physical_capacities() {
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let dimension_type = DimensionType::new("columns", DimensionBounds::new(0, Some(4)).unwrap());
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let rows = builder
            .add_instruction(
                DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(2).unwrap())),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap()[0];
        // Dimension constants have exact types, so the bounded dynamic extent is a program input.
        let columns = builder.add_input(dimension_type.into());
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, rows, columns], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let output_type = <&ArrayType>::try_from(&program.output_types()[0]).unwrap().clone();
        let input_types = vec![input_type, ArrayType::scalar(DataType::I64)];
        assert_eq!(
            to_mlir_module_for_program(&program, &[], &input_types, &vec![output_type], "main", None, None),
            Err(LoweringError::UnsupportedOp {
                op: "reshape with unequal physical capacities (input 4, output 6); dynamic bounds must preserve the \
                     physical element count"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_plain_transpose_then_reshape_lower_transpose_before_reshape() {
        let input_type = test_matrix_type(2, 3);
        let mut builder = ryft_core::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type);
        let input =
            builder.add_instruction(TransposeOperation::new([-1, -2]), Vec::new(), vec![input], None).unwrap()[0];
        let output = builder
            .add_instruction(
                ReshapeOperation::new(Shape::new(vec![Dimension::Static(6)])),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
                    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
                    %1 = stablehlo.reshape %0 : (tensor<3x2xf32>) -> tensor<6xf32>
                    return %1 : tensor<6xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_fixed_dynamic_identity_reshape_lowers_without_an_operation() {
        let shape = Shape::new(vec![dynamic_dimension("rows", None), Dimension::Static(3)]);
        let mut builder = ryft_core::ProgramBuilder::<Array, ReshapeOperation>::new();
        let input = builder.add_input(ArrayType::new(DataType::F32, shape.clone()));
        let output = builder.add_instruction(ReshapeOperation::new(shape), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?x3xf32>) -> tensor<?x3xf32> {
                    return %arg0 : tensor<?x3xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_fixed_dynamic_transpose_then_identity_reshape_lowers_to_transpose() {
        let rows = DimensionVariable::new("rows", DimensionBounds::unbounded());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(5)).unwrap());
        let input_shape = Shape::new(vec![rows.clone().into(), Dimension::Static(3), columns.clone().into()]);
        let output_shape = Shape::new(vec![columns.into(), rows.into(), Dimension::Static(3)]);
        let mut builder = ryft_core::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F32, input_shape));
        let input =
            builder.add_instruction(TransposeOperation::new([2, 0, 1]), Vec::new(), vec![input], None).unwrap()[0];
        let output =
            builder.add_instruction(ReshapeOperation::new(output_shape), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?x3x?xf32, #stablehlo.bounds<?, ?, 4>>) -> tensor<?x?x3xf32, #stablehlo.bounds<4, ?, ?>> {
                    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<?x3x?xf32, #stablehlo.bounds<?, ?, 4>>) -> tensor<?x?x3xf32, #stablehlo.bounds<4, ?, ?>>
                    return %0 : tensor<?x?x3xf32, #stablehlo.bounds<4, ?, ?>>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_dynamic_singleton_insertion_reshape_lowers_to_dynamic_reshape() {
        // Inserting a static singleton axis changes the rank, so neither the identity nor the transpose shortcut
        // applies. The runtime shape reads the dynamic extent back from the corresponding input axis, static axes
        // become `i64` constants, and `tensor.cast` restores the refined result type after `dynamic_reshape`.
        let rows = dynamic_dimension("rows", Some(5));
        let program = plain_reshape_program(Shape::new(vec![rows.clone()]), Shape::new(vec![1.into(), rows.clone()]));
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?xf64, #stablehlo.bounds<4>>) -> tensor<1x?xf64, #stablehlo.bounds<?, 4>> {
                    %c = stablehlo.constant dense<1> : tensor<i64>
                    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf64, #stablehlo.bounds<4>>) -> tensor<i32>
                    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
                    %2 = stablehlo.convert %c : (tensor<i64>) -> tensor<i32>
                    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
                    %4 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
                    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
                    %6 = stablehlo.concatenate %3, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
                    %7 = stablehlo.dynamic_reshape %arg0, %6 : (tensor<?xf64, #stablehlo.bounds<4>>, tensor<2xi32>) -> tensor<?x?xf64, #stablehlo.bounds<1, 4>>
                    %cast = tensor.cast %7 : tensor<?x?xf64, #stablehlo.bounds<1, 4>> to tensor<1x?xf64, #stablehlo.bounds<?, 4>>
                    return %cast : tensor<1x?xf64, #stablehlo.bounds<?, 4>>
                  }
                }
            "#},
        );
        // Execute below the allocation bound and with an empty runtime extent; the logical shape follows the input.
        let client = execution_client();
        let program = unproject_plain_program(program);
        let storage = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage.clone(), vec![4])], &[2]),
            Ok(vec![MixedValue::Array(vec![1.0, 2.0], vec![1, 2]), MixedValue::Dimension(2)]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage, vec![4])], &[0]),
            Ok(vec![MixedValue::Array(Vec::new(), vec![1, 0]), MixedValue::Dimension(0)]),
        );

        // A trailing static axis is preserved as a constant behind the inserted singleton.
        let program =
            plain_reshape_program(Shape::new(vec![rows.clone(), 3.into()]), Shape::new(vec![rows, 1.into(), 3.into()]));
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?x3xf64, #stablehlo.bounds<4, ?>>) -> tensor<?x1x3xf64, #stablehlo.bounds<4, ?, ?>> {
                    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x3xf64, #stablehlo.bounds<4, ?>>) -> tensor<i32>
                    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
                    %c = stablehlo.constant dense<1> : tensor<i64>
                    %c_0 = stablehlo.constant dense<3> : tensor<i64>
                    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
                    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
                    %4 = stablehlo.convert %c : (tensor<i64>) -> tensor<i32>
                    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
                    %6 = stablehlo.convert %c_0 : (tensor<i64>) -> tensor<i32>
                    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
                    %8 = stablehlo.concatenate %3, %5, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
                    %9 = stablehlo.dynamic_reshape %arg0, %8 : (tensor<?x3xf64, #stablehlo.bounds<4, ?>>, tensor<3xi32>) -> tensor<?x?x?xf64, #stablehlo.bounds<4, 1, 3>>
                    %cast = tensor.cast %9 : tensor<?x?x?xf64, #stablehlo.bounds<4, 1, 3>> to tensor<?x1x3xf64, #stablehlo.bounds<4, ?, ?>>
                    return %cast : tensor<?x1x3xf64, #stablehlo.bounds<4, ?, ?>>
                  }
                }
            "#},
        );
        let program = unproject_plain_program(program);
        let storage = (1..=12).map(f64::from).collect::<Vec<_>>();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage.clone(), vec![4, 3])], &[2]),
            Ok(vec![MixedValue::Array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 1, 3]), MixedValue::Dimension(2)]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage, vec![4, 3])], &[0]),
            Ok(vec![MixedValue::Array(Vec::new(), vec![0, 1, 3]), MixedValue::Dimension(0)]),
        );
    }

    #[test]
    fn test_plain_dynamic_singleton_removal_reshape_lowers_to_dynamic_reshape() {
        // Removing the only static axis leaves a rank-one dynamic target whose `dynamic_reshape` result type already
        // carries the refined bound, so no `tensor.cast` is needed.
        let rows = dynamic_dimension("rows", Some(5));
        let program = plain_reshape_program(Shape::new(vec![1.into(), rows.clone()]), Shape::new(vec![rows.clone()]));
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<1x?xf64, #stablehlo.bounds<?, 4>>) -> tensor<?xf64, #stablehlo.bounds<4>> {
                    %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<1x?xf64, #stablehlo.bounds<?, 4>>) -> tensor<i32>
                    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
                    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
                    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
                    %4 = stablehlo.dynamic_reshape %arg0, %3 : (tensor<1x?xf64, #stablehlo.bounds<?, 4>>, tensor<1xi32>) -> tensor<?xf64, #stablehlo.bounds<4>>
                    return %4 : tensor<?xf64, #stablehlo.bounds<4>>
                  }
                }
            "#},
        );
        let client = execution_client();
        let program = unproject_plain_program(program);
        let storage = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage.clone(), vec![1, 4])], &[2]),
            Ok(vec![MixedValue::Array(vec![1.0, 2.0], vec![2]), MixedValue::Dimension(2)]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage, vec![1, 4])], &[0]),
            Ok(vec![MixedValue::Array(Vec::new(), vec![0]), MixedValue::Dimension(0)]),
        );

        // Removing an interior singleton keeps the surrounding dynamic and static axes in order.
        let program =
            plain_reshape_program(Shape::new(vec![rows.clone(), 1.into(), 3.into()]), Shape::new(vec![rows, 3.into()]));
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?x1x3xf64, #stablehlo.bounds<4, ?, ?>>) -> tensor<?x3xf64, #stablehlo.bounds<4, ?>> {
                    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x1x3xf64, #stablehlo.bounds<4, ?, ?>>) -> tensor<i32>
                    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
                    %c = stablehlo.constant dense<3> : tensor<i64>
                    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
                    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
                    %4 = stablehlo.convert %c : (tensor<i64>) -> tensor<i32>
                    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
                    %6 = stablehlo.concatenate %3, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
                    %7 = stablehlo.dynamic_reshape %arg0, %6 : (tensor<?x1x3xf64, #stablehlo.bounds<4, ?, ?>>, tensor<2xi32>) -> tensor<?x?xf64, #stablehlo.bounds<4, 3>>
                    %cast = tensor.cast %7 : tensor<?x?xf64, #stablehlo.bounds<4, 3>> to tensor<?x3xf64, #stablehlo.bounds<4, ?>>
                    return %cast : tensor<?x3xf64, #stablehlo.bounds<4, ?>>
                  }
                }
            "#},
        );
        let program = unproject_plain_program(program);
        let storage = (1..=12).map(f64::from).collect::<Vec<_>>();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage.clone(), vec![4, 1, 3])], &[2]),
            Ok(vec![MixedValue::Array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]), MixedValue::Dimension(2)]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage, vec![4, 1, 3])], &[0]),
            Ok(vec![MixedValue::Array(Vec::new(), vec![0, 3]), MixedValue::Dimension(0)]),
        );
    }

    #[test]
    fn test_plain_dynamic_transpose_then_singleton_reshape_lowers_to_transpose_and_dynamic_reshape() {
        // An explicit transpose moves the dynamic extent to its permuted axis and reorders the elements
        // before the singleton is inserted.
        let rows = dynamic_dimension("rows", Some(5));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![rows.clone(), 3.into()])));
        let transposed =
            builder.add_instruction(TransposeOperation::new([1, 0]), Vec::new(), vec![input], None).unwrap()[0];
        let output = builder
            .add_instruction(
                ReshapeOperation::new(Shape::new(vec![3.into(), 1.into(), rows.clone()])),
                Vec::new(),
                vec![transposed],
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
                  func.func @main(%arg0: tensor<?x3xf64, #stablehlo.bounds<4, ?>>) -> tensor<3x1x?xf64, #stablehlo.bounds<?, ?, 4>> {
                    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<?x3xf64, #stablehlo.bounds<4, ?>>) -> tensor<3x?xf64, #stablehlo.bounds<?, 4>>
                    %c = stablehlo.constant dense<3> : tensor<i64>
                    %c_0 = stablehlo.constant dense<1> : tensor<i64>
                    %1 = stablehlo.get_dimension_size %0, dim = 1 : (tensor<3x?xf64, #stablehlo.bounds<?, 4>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %3 = stablehlo.convert %c : (tensor<i64>) -> tensor<i32>
                    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
                    %5 = stablehlo.convert %c_0 : (tensor<i64>) -> tensor<i32>
                    %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
                    %7 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
                    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
                    %9 = stablehlo.concatenate %4, %6, %8, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
                    %10 = stablehlo.dynamic_reshape %0, %9 : (tensor<3x?xf64, #stablehlo.bounds<?, 4>>, tensor<3xi32>) -> tensor<?x?x?xf64, #stablehlo.bounds<3, 1, 4>>
                    %cast = tensor.cast %10 : tensor<?x?x?xf64, #stablehlo.bounds<3, 1, 4>> to tensor<3x1x?xf64, #stablehlo.bounds<?, ?, 4>>
                    return %cast : tensor<3x1x?xf64, #stablehlo.bounds<?, ?, 4>>
                  }
                }
            "#},
        );
        let client = execution_client();
        let program = unproject_plain_program(program);
        let storage = (1..=12).map(f64::from).collect::<Vec<_>>();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage.clone(), vec![4, 3])], &[2]),
            Ok(vec![MixedValue::Array(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 1, 2]), MixedValue::Dimension(2)]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage, vec![4, 3])], &[0]),
            Ok(vec![MixedValue::Array(Vec::new(), vec![3, 1, 0]), MixedValue::Dimension(0)]),
        );

        // The permutation may also move a singleton that the reshape then removes.
        let mut builder = XlaProgramBuilder::new();
        let input =
            builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![1.into(), rows.clone(), 3.into()])));
        let transposed =
            builder.add_instruction(TransposeOperation::new([2, 1, 0]), Vec::new(), vec![input], None).unwrap()[0];
        let output = builder
            .add_instruction(
                ReshapeOperation::new(Shape::new(vec![3.into(), rows])),
                Vec::new(),
                vec![transposed],
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
                  func.func @main(%arg0: tensor<1x?x3xf64, #stablehlo.bounds<?, 4, ?>>) -> tensor<3x?xf64, #stablehlo.bounds<?, 4>> {
                    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<1x?x3xf64, #stablehlo.bounds<?, 4, ?>>) -> tensor<3x?x1xf64, #stablehlo.bounds<?, 4, ?>>
                    %c = stablehlo.constant dense<3> : tensor<i64>
                    %1 = stablehlo.get_dimension_size %0, dim = 1 : (tensor<3x?x1xf64, #stablehlo.bounds<?, 4, ?>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %3 = stablehlo.convert %c : (tensor<i64>) -> tensor<i32>
                    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
                    %5 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
                    %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
                    %7 = stablehlo.concatenate %4, %6, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
                    %8 = stablehlo.dynamic_reshape %0, %7 : (tensor<3x?x1xf64, #stablehlo.bounds<?, 4, ?>>, tensor<2xi32>) -> tensor<?x?xf64, #stablehlo.bounds<3, 4>>
                    %cast = tensor.cast %8 : tensor<?x?xf64, #stablehlo.bounds<3, 4>> to tensor<3x?xf64, #stablehlo.bounds<?, 4>>
                    return %cast : tensor<3x?xf64, #stablehlo.bounds<?, 4>>
                  }
                }
            "#},
        );
        let program = unproject_plain_program(program);
        let storage = (1..=12).map(f64::from).collect::<Vec<_>>();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage.clone(), vec![1, 4, 3])], &[2]),
            Ok(vec![MixedValue::Array(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]), MixedValue::Dimension(2)]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage, vec![1, 4, 3])], &[0]),
            Ok(vec![MixedValue::Array(Vec::new(), vec![3, 0]), MixedValue::Dimension(0)]),
        );
    }

    #[test]
    fn test_plain_dynamic_repeated_identity_singleton_reshape_lowers_to_dynamic_reshape() {
        // Two axes sharing one dimension identity are each read back from their own input axis, in order, so the
        // repeated identity never collapses into a single runtime read.
        let rows = dynamic_dimension("rows", Some(5));
        let program = plain_reshape_program(
            Shape::new(vec![rows.clone(), rows.clone()]),
            Shape::new(vec![rows.clone(), 1.into(), rows]),
        );
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?x?xf64, #stablehlo.bounds<4, 4>>) -> tensor<?x1x?xf64, #stablehlo.bounds<4, ?, 4>> {
                    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x?xf64, #stablehlo.bounds<4, 4>>) -> tensor<i32>
                    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
                    %c = stablehlo.constant dense<1> : tensor<i64>
                    %2 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<?x?xf64, #stablehlo.bounds<4, 4>>) -> tensor<i32>
                    %3 = stablehlo.convert %2 : (tensor<i32>) -> tensor<i64>
                    %4 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
                    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
                    %6 = stablehlo.convert %c : (tensor<i64>) -> tensor<i32>
                    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
                    %8 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
                    %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
                    %10 = stablehlo.concatenate %5, %7, %9, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
                    %11 = stablehlo.dynamic_reshape %arg0, %10 : (tensor<?x?xf64, #stablehlo.bounds<4, 4>>, tensor<3xi32>) -> tensor<?x?x?xf64, #stablehlo.bounds<4, 1, 4>>
                    %cast = tensor.cast %11 : tensor<?x?x?xf64, #stablehlo.bounds<4, 1, 4>> to tensor<?x1x?xf64, #stablehlo.bounds<4, ?, 4>>
                    return %cast : tensor<?x1x?xf64, #stablehlo.bounds<4, ?, 4>>
                  }
                }
            "#},
        );
        // Both input axes receive the shared runtime extent, and both dynamic output axes report it back.
        let client = execution_client();
        let program = unproject_plain_program(program);
        let storage = (1..=16).map(f64::from).collect::<Vec<_>>();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage.clone(), vec![4, 4])], &[2, 2]),
            Ok(vec![
                MixedValue::Array(vec![1.0, 2.0, 5.0, 6.0], vec![2, 1, 2]),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
            ]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(storage, vec![4, 4])], &[0, 0]),
            Ok(vec![MixedValue::Array(Vec::new(), vec![0, 1, 0]), MixedValue::Dimension(0), MixedValue::Dimension(0),]),
        );
    }

    #[test]
    fn test_lower_static_input_dynamic_output_reshape_pullback_executes() {
        let client = execution_client();
        crate::operations::assertions::ensure_assertion_handler_registered(&client).unwrap();
        let program = static_input_dynamic_output_reshape_program();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tangents = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        // The primal keeps the elements and reports the requested dynamic extent; a mismatching extent fails the
        // runtime element-count assertion because the output shape is unproven.
        assert_eq!(
            execute_mixed_program(
                &client,
                &program,
                &[MixedValue::Array(values.clone(), vec![6]), MixedValue::Dimension(6)],
                &[],
            ),
            Ok(vec![MixedValue::Array(values.clone(), vec![6]), MixedValue::Dimension(6)]),
        );
        let error = execute_mixed_program(
            &client,
            &program,
            &[MixedValue::Array(values, vec![6]), MixedValue::Dimension(3)],
            &[],
        )
        .unwrap_err();
        assert!(error.to_string().contains("`reshape`"), "{error}");

        // Linearization retains the dynamic extent as the residual, and the pullback reshapes the dynamic cotangent
        // back to the static input through `stablehlo.reshape` after asserting its runtime element count.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let pullback = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        let cotangent_type = <&ArrayType>::try_from(&pullback.input_types()[0]).unwrap().clone();
        assert_eq!(
            to_mlir_module_for_program(
                &pullback,
                &[],
                &vec![cotangent_type, ArrayType::scalar(DataType::I64)],
                &vec![ArrayType::new_static(DataType::F64, [6])],
                "main",
                None,
                None,
            )
            .unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<6xf64>, %arg1: tensor<i64>, %arg2: tensor<i32>) -> tensor<6xf64> {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<6xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<6>>
                    %c = stablehlo.constant dense<6> : tensor<i64>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf64, #stablehlo.bounds<6>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %c_0 = stablehlo.constant dense<0> : tensor<i64>
                    %3 = stablehlo.compare LT, %2, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %4 = stablehlo.after_all  : !stablehlo.token
                    %5 = stablehlo.custom_call @ryft.assert(%3, %2, %c, %4) {api_version = 4 : i32, backend_config = {actor = "reshape", detail = "1", kind = "reshape"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %6 = stablehlo.reshape %0 : (tensor<?xf64, #stablehlo.bounds<6>>) -> tensor<6xf64>
                    return %6 : tensor<6xf64>
                  }
                }
            "#},
        );
        assert_eq!(
            execute_mixed_program(
                &client,
                &pullback,
                &[MixedValue::Array(tangents.clone(), vec![6]), MixedValue::Dimension(6)],
                &[6],
            ),
            Ok(vec![MixedValue::Array(tangents.clone(), vec![6])]),
        );
        let error = execute_mixed_program(
            &client,
            &pullback,
            &[MixedValue::Array(tangents, vec![6]), MixedValue::Dimension(3)],
            &[3],
        )
        .unwrap_err();
        assert!(error.to_string().contains("`reshape`"), "{error}");
    }

    #[test]
    fn test_lower_static_input_dynamic_output_reshape_batched_jvp_executes() {
        // Batching threads the batch extent as a leading dimension input, and the forward derivative of the batched
        // program reshapes the tangent alongside the primal. Both outputs report the batch and row extents.
        let client = execution_client();
        crate::operations::assertions::ensure_assertion_handler_registered(&client).unwrap();
        let batch = DimensionType::new("batch", DimensionBounds::non_negative(Some(4)).unwrap());
        let (batched, output_axes) = static_input_dynamic_output_reshape_program()
            .batched_with_threaded_extent(
                batch,
                ShardingDimension::replicated(),
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        let jvp = batched.jvp().unwrap();
        let values = (1..=18).map(f64::from).collect::<Vec<_>>();
        let tangents = (101..=118).map(f64::from).collect::<Vec<_>>();
        assert_eq!(
            execute_mixed_program(
                &client,
                &jvp,
                &[
                    MixedValue::Dimension(2),
                    MixedValue::Array(values.clone(), vec![3, 6]),
                    MixedValue::Dimension(6),
                    MixedValue::Array(tangents.clone(), vec![3, 6]),
                ],
                &[2, 2],
            ),
            Ok(vec![
                MixedValue::Dimension(2),
                MixedValue::Array(values[..12].to_vec(), vec![2, 6]),
                MixedValue::Array(tangents[..12].to_vec(), vec![2, 6]),
                MixedValue::Dimension(2),
                MixedValue::Dimension(6),
                MixedValue::Dimension(2),
                MixedValue::Dimension(6),
            ]),
        );
        assert_eq!(
            execute_mixed_program(
                &client,
                &jvp,
                &[
                    MixedValue::Dimension(0),
                    MixedValue::Array(values, vec![3, 6]),
                    MixedValue::Dimension(6),
                    MixedValue::Array(tangents, vec![3, 6]),
                ],
                &[0, 0],
            ),
            Ok(vec![
                MixedValue::Dimension(0),
                MixedValue::Array(Vec::new(), vec![0, 6]),
                MixedValue::Array(Vec::new(), vec![0, 6]),
                MixedValue::Dimension(0),
                MixedValue::Dimension(6),
                MixedValue::Dimension(0),
                MixedValue::Dimension(6),
            ]),
        );
    }

    #[test]
    fn test_lower_batched_proven_reshape_omits_element_count_assertion() {
        // A proven, effect-free mixed reshape batched over a mapped input recomputes its proof on the lifted
        // signature, so the batched instruction lowers to a plain `stablehlo.reshape` without an assertion.
        let parent = TracingContext::<XlaConstant, XlaOperation>::new();
        let input = parent.input(ArrayType::new_static(DataType::F64, [2, 6]).into());
        let two = parent.lift(XlaConstant::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
        let three = parent.lift(XlaConstant::Dimension(DimensionValue::constant(3).unwrap())).unwrap();
        let operation = DynamicReshapeOperation::new()
            .with_input_types(&[
                ArrayType::new_static(DataType::F64, [6]).into(),
                two.r#type().into_owned(),
                three.r#type().into_owned(),
            ])
            .unwrap();
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), two.clone());
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap(),
                    ArrayIrBatch::replicated(two),
                    ArrayIrBatch::replicated(three),
                ],
            )
            .unwrap();
        let program = parent
            .builder()
            .borrow()
            .clone()
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![outputs[0].value().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![ArrayType::new_static(DataType::F64, [2, 6])],
                &vec![ArrayType::new_static(DataType::F64, [2, 2, 3])],
                "main",
                None,
                None,
            )
            .unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x6xf64>) -> tensor<2x2x3xf64> {
                    %0 = stablehlo.reshape %arg0 : (tensor<2x6xf64>) -> tensor<2x2x3xf64>
                    return %0 : tensor<2x2x3xf64>
                  }
                }
            "#},
        );
        let values = (1..=12).map(f64::from).collect::<Vec<_>>();
        assert_eq!(
            execute_mixed_program(&execution_client(), &program, &[MixedValue::Array(values.clone(), vec![2, 6])], &[]),
            Ok(vec![MixedValue::Array(values, vec![2, 2, 3])]),
        );
    }

    #[test]
    fn test_lower_batched_unproven_reshape_keeps_element_count_assertion_for_unused_output() {
        // An unproven mixed reshape stays effectful after batching, so its assertion survives even though the program
        // never uses the reshaped value and the executable still rejects mismatching runtime element counts.
        let client = execution_client();
        crate::operations::assertions::ensure_assertion_handler_registered(&client).unwrap();
        let parent = TracingContext::<XlaConstant, XlaOperation>::new();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), dynamic_dimension("input", Some(5))]));
        let input = parent.input(input_type.clone().into());
        let size = parent.input(ArrayType::scalar(DataType::I64).into());
        let output_type = DimensionType::new("output", DimensionBounds::new(0, Some(5)).unwrap());
        let dimension = parent
            .bind(DimensionFromScalarOperation::new(output_type.clone()), Vec::new(), &[size])
            .unwrap()
            .remove(0);
        let two = parent.lift(XlaConstant::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
        let operation = DynamicReshapeOperation::new()
            .with_input_types(&[input_type.unbatched(BatchAxis::new(0)).unwrap().into(), output_type.into()])
            .unwrap();
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), two);
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayIrBatch::new(input.clone(), BatchAxis::new(0)).unwrap(), ArrayIrBatch::replicated(dimension)],
            )
            .unwrap();
        assert_eq!(outputs[0].value().r#type().to_string(), "f64[2, output]");
        let program = parent
            .builder()
            .borrow()
            .clone()
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![input.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type.clone(), ArrayType::scalar(DataType::I64)],
            &vec![input_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(module.matches("stablehlo.dynamic_reshape").count(), 1, "{module}");
        assert_eq!(
            module.matches(r#"backend_config = {actor = "reshape", detail = "2", kind = "reshape"}"#).count(),
            1,
            "{module}",
        );
        let values = (1..=8).map(f64::from).collect::<Vec<_>>();
        assert_eq!(
            execute_mixed_program(
                &client,
                &program,
                &[MixedValue::Array(values.clone(), vec![2, 4]), MixedValue::Dimension(2)],
                &[2],
            ),
            Ok(vec![MixedValue::Array(vec![1.0, 2.0, 5.0, 6.0], vec![2, 2]), MixedValue::Dimension(2)]),
        );
        let error = execute_mixed_program(
            &client,
            &program,
            &[MixedValue::Array(values, vec![2, 4]), MixedValue::Dimension(3)],
            &[2],
        )
        .unwrap_err();
        assert!(error.to_string().contains("`reshape`"), "{error}");
    }

    #[test]
    fn test_reshape_lowering_validates_input_arity_before_indexing() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let mut block = module.body().unwrap();
        let error = lower_reshape_to_mlir(
            &ReshapeOperation::new(Shape::new(vec![Dimension::Static(4)])),
            &test_vector_type(4),
            &[],
            &[test_vector_type(4)],
            &[],
            &mut block,
            &context,
            location.as_ref(),
        )
        .unwrap_err();

        assert_eq!(error, LoweringError::Tracing(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),);
    }

    #[test]
    fn test_xla_operation_transpose_then_reshape_use_shared_lowering() {
        let input_type = test_matrix_type(2, 3);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let input = builder.add_instruction(TransposeOperation::new([1, 0]), Vec::new(), vec![input], None).unwrap()[0];
        let output = builder
            .add_instruction(
                ReshapeOperation::new(Shape::new(vec![Dimension::Static(6)])),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
                    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
                    %1 = stablehlo.reshape %0 : (tensor<3x2xf32>) -> tensor<6xf32>
                    return %1 : tensor<6xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_reshape_explicit_output_sharding_lowers_constraint() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let output_sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap();
        let mut builder = ryft_core::ProgramBuilder::<Array, ReshapeOperation>::new();
        let input = builder.add_input(test_vector_type(4));
        let output = builder
            .add_instruction(
                ReshapeOperation::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
                    .with_output_sharding(output_sharding),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<2x2xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}, {}]> : tensor<2x2xf32>
                    return %1 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_renders_reshape_singleton_axis_sharding_propagation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let output_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();

        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |x: XlaArrayTracer| {
                x.reshape(Shape::new(vec![Dimension::Static(1), Dimension::Static(2), Dimension::Static(1)]))
                    .unwrap()
            },
            global_input_type,
            mesh.clone(),
            input_sharding.clone(),
            output_sharding.clone(),
        )
        .unwrap();

        assert_eq!(
            to_mlir_module(&traced, "main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<1x8x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{}, {"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.reshape %arg1 : (tensor<2xf32>) -> tensor<1x2x1xf32>
                      sdy.return %1 : tensor<1x2x1xf32>
                    } : (tensor<8xf32>) -> tensor<1x8x1xf32>
                    return %0 : tensor<1x8x1xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_renders_reshape_replicated_merge_sharding_propagation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let global_input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(8), Dimension::Static(2), Dimension::Static(3)]),
        );
        let input_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        )
        .unwrap();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();

        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |x: XlaArrayTracer| x.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(6)])).unwrap(),
            global_input_type,
            mesh.clone(),
            input_sharding.clone(),
            output_sharding.clone(),
        )
        .unwrap();

        assert_eq!(
            to_mlir_module(&traced, "main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> (tensor<8x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x2x3xf32>) {
                      %1 = stablehlo.reshape %arg1 : (tensor<2x2x3xf32>) -> tensor<2x6xf32>
                      sdy.return %1 : tensor<2x6xf32>
                    } : (tensor<8x2x3xf32>) -> tensor<8x6xf32>
                    return %0 : tensor<8x6xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_renders_reshape_replicated_split_sharding_propagation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let global_input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(6)]));
        let input_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let output_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        )
        .unwrap();

        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |x: XlaArrayTracer| {
                x.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(3)]))
                    .unwrap()
            },
            global_input_type,
            mesh.clone(),
            input_sharding.clone(),
            output_sharding.clone(),
        )
        .unwrap();

        assert_eq!(
            to_mlir_module(&traced, "main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}, {}]>] manual_axes={"x"} (%arg1: tensor<2x6xf32>) {
                      %1 = stablehlo.reshape %arg1 : (tensor<2x6xf32>) -> tensor<2x2x3xf32>
                      sdy.return %1 : tensor<2x2x3xf32>
                    } : (tensor<8x6xf32>) -> tensor<8x2x3xf32>
                    return %0 : tensor<8x2x3xf32>
                  }
                }
            "#},
        );
    }
}
