use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, BroadcastLikeOperation, BroadcastOperation, Dimension, DynamicBroadcastOperation, Operation,
    ProgramError,
};
use ryft_mlir::dialects::{stable_hlo, tensor};
use ryft_mlir::{Block, Operation as MlirOperation, Type as MlirType, Value as MlirValue, ValueRef};

use crate::lowering::modules::broadcast_changes_explicit_sharding;
use crate::lowering::sharding::{OpenDimensions, ShardingExtension};
use crate::lowering::types::{
    ArrayTypeLoweringExtension, dimension_extent_i64, lower_explicit_shape, lower_runtime_dimension_size_i64,
    lower_static_index_constants, physical_bound_type,
};
use crate::operations::constants::{
    ArrayTypeLayoutConstraintLoweringExtension, dynamic_constructor_types, refine_dynamic_constructor_result,
};
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError, Type};

impl LowerableOperation for BroadcastOperation {
    /// Lowers a [`BroadcastOperation`] and explicitly constrains any placement transition over an explicit mesh axis.
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let bound_manual_axes = lowerer.lowering_state().bound_manual_axes();
        let block = &mut lowerer.block();
        let (context, location) = (lowerer.context(), lowerer.location());
        check_count!("input", input_values, 1, ProgramError);
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let output_type = &output_types[0];
        let input = input_values[0];
        // A static result can refine exact symbolic input axes. Expose their proved physical sizes so that XLA does not
        // retain a dynamic unit axis that would prevent a subsequent singleton expansion.
        let input = if output_type.static_shape().is_some() {
            input_types[0].lower_physical_storage(input, block, context, location)?
        } else {
            input
        };
        let result = if output_type.static_shape().is_none() && physical_bound_type(output_type).is_err() {
            if output_type.layout().is_some() {
                return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                    operation: self.name().to_string(),
                    reason: "dynamic broadcast with explicit output layout needs finite allocation bounds".to_owned(),
                }));
            }
            // Every dynamic result axis is a passthrough of a mapped input axis, so obtain its runtime extent there.
            let extents = output_type
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .map(|(axis, dimension)| {
                    if let Dimension::Static(extent) = dimension {
                        dimension_extent_i64(*extent)?;
                        Ok(lower_static_index_constants(&[*extent], block, context, location)?.remove(0))
                    } else {
                        let input_axis =
                            self.output_axes().iter().position(|output_axis| *output_axis == axis).unwrap();
                        lower_runtime_dimension_size_i64(input, input_axis, block, context, location)
                    }
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let shape = lower_explicit_shape(&extents, block, context, location)?;
            let broadcast = block.append_operation(stable_hlo::dynamic_broadcast(
                input,
                shape,
                self.output_axes(),
                None,
                None,
                location,
            )?)?;
            let result = broadcast.result(0).unwrap().as_ref();
            let expected_type = output_type.lower(context, location)?;
            if result.r#type()? == expected_type.as_ref() {
                result
            } else {
                block.append_operation(tensor::cast(result, expected_type, location)?)?.result(0).unwrap().as_ref()
            }
        } else if output_type.static_shape().is_none()
            && (output_type.layout().is_some()
                || output_type
                    .shape()
                    .dimensions()
                    .iter()
                    .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
                    .count()
                    > 1)
        {
            // `stablehlo.broadcast_in_dim` results admit at most one bounded dynamic axis. Broadcast the physical
            // allocation for multiple dynamic axes or an explicit layout, then restore every logical size from its
            // mapped input axis.
            let physical_type = physical_bound_type(output_type)?;
            let physical_input = input_types[0].lower_physical_bound_value(input, 0.0, block, context, location)?;
            let broadcast = block.append_operation(stable_hlo::broadcast(
                physical_input,
                physical_type.lower(context, location)?,
                self.output_axes(),
                location,
            )?)?;
            let result =
                physical_type.lower_layout_constraint(broadcast.result(0).unwrap().as_ref(), block, location)?;
            let mut sources = vec![(input, 0); output_type.rank()];
            for (input_axis, output_axis) in self.output_axes().iter().enumerate() {
                sources[*output_axis] = (input, input_axis);
            }
            output_type.lower_restore_dynamic_dimensions(result, &sources, block, context, location)?
        } else {
            let broadcast = block.append_operation(stable_hlo::broadcast(
                input,
                output_type.lower(context, location)?,
                self.output_axes(),
                location,
            )?)?;
            output_type.lower_layout_constraint(broadcast.result(0).unwrap().as_ref(), block, location)?
        };
        let result = if broadcast_changes_explicit_sharding(&input_types[0], &output_types[0], self.output_axes()) {
            let sharding = output_types[0].sharding().unwrap();
            sharding.lower_sharding_constraint(result, OpenDimensions::All, bound_manual_axes, block, location)?
        } else {
            result
        };
        Ok(vec![result])
    }
}

impl LowerableOperation for BroadcastLikeOperation {
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
        let context = lowerer.context();
        let location = lowerer.location();
        let block = &mut lowerer.block();
        let output_type = &output_types[0];
        // The exemplar only provides the logical sizes of the dynamic axes. The value is broadcast at the physical
        // allocation, where a stretched unit axis has a static extent even when its output axis is dynamic, and
        // every logical size is then restored from the exemplar.
        let physical_type = physical_bound_type(output_type)?;
        let value = input_types[0].lower_physical_bound_value(input_values[0], 0.0, block, context, location)?;
        let broadcast = block.append_operation(stable_hlo::broadcast(
            value,
            physical_type.lower(context, location)?,
            self.output_axes(),
            location,
        )?)?;
        let result = broadcast.result(0).unwrap().as_ref();
        let result = physical_type.lower_layout_constraint(result, block, location)?;
        let sources = (0..output_type.rank()).map(|axis| (input_values[1], axis)).collect::<Vec<_>>();
        let result = output_type.lower_restore_dynamic_dimensions(result, &sources, block, context, location)?;
        let result = if broadcast_changes_explicit_sharding(&input_types[0], output_type, self.output_axes()) {
            output_type.sharding().unwrap().lower_sharding_constraint(
                result,
                OpenDimensions::All,
                lowerer.lowering_state().bound_manual_axes(),
                block,
                location,
            )?
        } else {
            result
        };
        Ok(vec![result])
    }
}

impl LowerableOperation for DynamicBroadcastOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let context = lowerer.context();
        let location = lowerer.location();
        let lowering_state = lowerer.lowering_state();
        let block = &mut lowerer.block();
        let Some((input, output_extents)) = input_values.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let Some(input_type) = input_types.first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        let [output_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
        };
        let output_type = <&ArrayType>::try_from(output_type)?;
        let result = if output_type.static_shape().is_some() {
            // Exact symbolic axes have no inactive storage, and must lose their dynamic bit before a static result
            // is consumed as singleton broadcast storage by a later operation.
            let input = input_type.lower_physical_storage(*input, block, context, location)?;
            let output_tensor_type = output_type.lower(context, location)?;
            let broadcast = block.append_operation(stable_hlo::broadcast(
                input,
                output_tensor_type,
                self.output_axes(),
                location,
            )?)?;
            let result = broadcast.result(0).unwrap().as_ref();
            output_type.lower_layout_constraint(result, block, location)?
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
                dynamic_constructor_types(self.name(), dynamic_extents.len(), output_types)?;
            let input = input_type.lower_physical_bound_value(*input, 0.0, block, context, location)?;
            let output_tensor_type = physical_type.lower(context, location)?;
            let broadcast = block.append_operation(stable_hlo::broadcast(
                input,
                output_tensor_type,
                self.output_axes(),
                location,
            )?)?;
            let result = broadcast.result(0).unwrap().as_ref();
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
            if output_type.layout().is_some() {
                return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                    operation: self.name().to_string(),
                    reason: "dynamic broadcast with explicit output layout needs finite allocation bounds".to_owned(),
                }));
            }
            let shape = lower_explicit_shape(output_extents, block, context, location)?;
            let broadcast = block.append_operation(stable_hlo::dynamic_broadcast(
                *input,
                shape,
                self.output_axes(),
                None,
                None,
                location,
            )?)?;
            let result = broadcast.result(0).unwrap().as_ref();
            let expected_type = output_type.lower(context, location)?;
            if result.r#type()? == expected_type.as_ref() {
                result
            } else {
                let cast = block.append_operation(tensor::cast(result, expected_type, location)?)?;
                cast.result(0).unwrap().as_ref()
            }
        };
        let result = if broadcast_changes_explicit_sharding(input_type, output_type, self.output_axes()) {
            output_type.sharding().unwrap().lower_sharding_constraint(
                result,
                OpenDimensions::All,
                lowering_state.bound_manual_axes(),
                block,
                location,
            )?
        } else {
            result
        };
        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fmt::Display;
    use std::ops::Range;
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ADD_OPERATION_NAME, Add, Array as CoreArray, BatchAxis, Broadcast, ConvertElementType, DataType, Device,
        DeviceMesh, DifferentiableType, DimensionBounds, DimensionSizeOperation, DimensionType, DimensionValue,
        DimensionVariable, DynamicBroadcastOperation, Error as CoreError, Layout, LogicalMesh, Max, MeshAxis,
        MeshAxisType, Min, Placeholder, ProgramBuilder as CoreProgramBuilder, ProgramError, Shape, ShardMapOptions,
        Sharding, ShardingDimension, StridedLayout, Tile, TileDimension, TiledLayout, Typed, TypedProgram,
        UnreducedKind, batch, differentiate_at,
    };
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::lowering::tests::{
        MixedValue, PlainProgram, ProgramBuilder, dynamic_dimension, execute_mixed_program, shard_map_module,
        test_manual_mesh, test_vector_type, to_rendered_mlir_for_plain_program,
    };
    use crate::operations::Operation;
    use crate::sessions::tests::{domain_mesh, replicated_vector_type};
    use crate::tests::{cpu_client, execution_client, test_compilation_options, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayConstant, ArrayTracer, CompileTracer, CompiledFunction, Constant, ExecutionArguments, FromPjrt,
        ProgramLoweringExtension, ProgramLoweringOptions, Session, TracedProgram, Type, Value, compile,
        trace_with_context,
    };

    use super::*;

    #[test]
    fn test_broadcast_operation_lower_explicit_singleton_expansion() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(1)
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let output_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let mut builder = CoreProgramBuilder::<CoreArray, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![0]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<CoreArray>, Vec<CoreArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_rendered_mlir_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<1xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xf32>) -> tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_broadcast_operation_lower_dynamic_result_type() {
        // Homogeneous broadcast admits a dynamic output extent that is identity-equal to the mapped input extent,
        // which is exactly the payload that program batching stages for a dynamic per-item dimension. Lowering keeps
        // that extent dynamic in the result tensor type and supplies no extent operand, so `stablehlo.broadcast_in_dim`
        // verifies only when the dynamic axis carries an upper bound that lowers to a `#stablehlo.bounds` encoding.
        let bounded = dynamic_dimension("n", Some(5));
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![bounded.clone()]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), bounded]));
        let mut builder = CoreProgramBuilder::<CoreArray, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<CoreArray>, Vec<CoreArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_rendered_mlir_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?xf32, #stablehlo.bounds<4>>) -> tensor<2x?xf32, #stablehlo.bounds<?, 4>> {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<?xf32, #stablehlo.bounds<4>>) -> tensor<2x?xf32, #stablehlo.bounds<?, 4>>
                    return %0 : tensor<2x?xf32, #stablehlo.bounds<?, 4>>
                  }
                }
            "#},
        );

        // Unbounded passthrough dimensions are supplied at runtime to the dynamic StableHLO operation.
        let unbounded = dynamic_dimension("n", None);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![unbounded.clone()]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), unbounded]));
        let mut builder = CoreProgramBuilder::<CoreArray, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<CoreArray>, Vec<CoreArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_rendered_mlir_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?xf32>) -> tensor<2x?xf32> {
                    %c = stablehlo.constant dense<2> : tensor<i64>
                    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
                    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
                    %2 = stablehlo.reshape %c : (tensor<i64>) -> tensor<1xi64>
                    %3 = stablehlo.reshape %1 : (tensor<i64>) -> tensor<1xi64>
                    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
                    %5 = stablehlo.dynamic_broadcast_in_dim %arg0, %4, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
                    %cast = tensor.cast %5 : tensor<?x?xf32> to tensor<2x?xf32>
                    return %cast : tensor<2x?xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_broadcast_operation_lower_dynamic_pullbacks() {
        /// Executes one homogeneous broadcast or its adjoint through the mixed executable ABI.
        fn execute(program: PlainProgram, values: &[f32], extent: i32) -> (Vec<f32>, Vec<u64>) {
            let input_type = program.input_types().into_owned().remove(0);
            let dimensions = physical_bound_type(&input_type)
                .unwrap()
                .static_shape()
                .unwrap()
                .dimensions()
                .iter()
                .map(|dimension| *dimension as u64)
                .collect::<Vec<_>>();
            let program = program.into_unprojected::<Constant, Operation>().unwrap();
            let module = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();
            let client = execution_client();
            let executable = client
                .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, None))
                .unwrap();
            let device = executable.addressable_devices().unwrap().remove(0);
            let mut inputs = vec![ExecutionInput {
                buffer: Arc::new(
                    client
                        .buffer(
                            values_to_bytes(values).as_slice(),
                            BufferType::F32,
                            &dimensions,
                            None,
                            device.clone(),
                            None,
                        )
                        .unwrap(),
                ),
                donatable: false,
            }];
            for dimension in input_type.shape().dimensions() {
                if matches!(dimension, Dimension::Dynamic(_)) {
                    inputs.push(ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(
                                    values_to_bytes(&[extent]).as_slice(),
                                    BufferType::I32,
                                    [],
                                    None,
                                    device.clone(),
                                    None,
                                )
                                .unwrap(),
                        ),
                        donatable: false,
                    });
                }
            }
            let output = executable
                .execute(
                    vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
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
                .unwrap()
                .remove(0)
                .outputs
                .remove(0);
            let dimensions = output.unpadded_dimensions().unwrap();
            let bytes = output.copy_to_host(None).unwrap().r#await().unwrap();
            (values_from_bytes::<f32>(&bytes), dimensions.to_vec())
        }

        // The first axis keeps the identity in scope while the second changes between its exact dynamic and static
        // representations. The pullback must preserve both the values and the statically refined second axis.
        let size = DimensionVariable::new("size", DimensionBounds::new(2, Some(3)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![size.clone().into(), 2.into()]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![size.clone().into(), size.into()]));
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![0, 1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(execute(program, &[2., 5., 7., 9.], 2), (vec![2., 5., 7., 9.], vec![2, 2]));
        assert_eq!(execute(pullback, &[1., 2., 3., 4.], 2), (vec![1., 2., 3., 4.], vec![2, 2]));

        // Reduction removes the stretched singleton, and the pullback reinserts it beside a genuinely dynamic axis.
        // Execute below the allocation bound to ensure the restored axis does not specialize the runtime extent.
        let size = dynamic_dimension("size", Some(5));
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![size.clone(), 1.into()]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![size, 3.into()]));
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![0, 1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(execute(program, &[2., 5., 0., 0.], 2), (vec![2., 2., 2., 5., 5., 5.], vec![2, 3]));
        assert_eq!(
            execute(pullback, &[1., 2., 3., 4., 5., 6., 0., 0., 0., 0., 0., 0.], 2),
            (vec![6., 15.], vec![2, 1]),
        );
    }

    #[test]
    fn test_broadcast_operation_lower_exact_symbolic_extent_refinement() {
        // Both broadcast forms can normalize proved symbolic extents to static storage. The unit axis must then
        // expand as a static singleton, while the exact nonunit axis preserves its distinct element values.
        let client = execution_client();
        for explicit_extents in [false, true] {
            let unit = DimensionVariable::new("unit", DimensionBounds::new(1, Some(2)).unwrap());
            let fixed = DimensionVariable::new("fixed", DimensionBounds::new(2, Some(3)).unwrap());
            let input_type = ArrayType::new(DataType::F64, Shape::new(vec![unit.into(), fixed.into()]));
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(input_type.into());
            let normalized = if explicit_extents {
                let one = builder.add_constant(Constant::Dimension(DimensionValue::constant(1).unwrap()));
                let two = builder.add_constant(Constant::Dimension(DimensionValue::constant(2).unwrap()));
                builder
                    .add_instruction(
                        DynamicBroadcastOperation::new(vec![0, 1]),
                        Vec::new(),
                        vec![input, one, two],
                        None,
                    )
                    .unwrap()[0]
            } else {
                builder
                    .add_instruction(
                        BroadcastOperation::new(ArrayType::new_static(DataType::F64, [1, 2]), vec![0, 1]),
                        Vec::new(),
                        vec![input],
                        None,
                    )
                    .unwrap()[0]
            };
            let expanded = builder
                .add_instruction(
                    BroadcastOperation::new(ArrayType::new_static(DataType::F64, [3, 2]), vec![0, 1]),
                    Vec::new(),
                    vec![normalized],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![expanded], vec![Placeholder], vec![Placeholder])
                .unwrap();
            assert_eq!(
                program.to_string(),
                if explicit_extents {
                    indoc! {"
                        lambda %0:f64[unit, fixed] .
                        let %1:dimension<1> = const 1
                            %2:dimension<2> = const 2
                            %3:f64[1, 2] = broadcast [output_axes=[0, 1]] %0 %1 %2
                            %4:f64[3, 2] = broadcast [output_type=f64[3, 2], output_axes=[0, 1]] %3
                        in (%4)"
                    }
                } else {
                    indoc! {"
                        lambda %0:f64[unit, fixed] .
                        let %1:f64[1, 2] = broadcast [output_type=f64[1, 2], output_axes=[0, 1]] %0
                            %2:f64[3, 2] = broadcast [output_type=f64[3, 2], output_axes=[0, 1]] %1
                        in (%2)"
                    }
                },
            );
            assert_eq!(
                execute_mixed_program(client, &program, &[MixedValue::Array(vec![2., 3.], vec![1, 2])], &[1, 2],)
                    .unwrap(),
                vec![MixedValue::Array(vec![2., 3., 2., 3., 2., 3.], vec![3, 2])],
            );
        }
    }

    #[test]
    fn test_broadcast_operation_lower_output_layout() {
        // Both static and bounded dynamic outputs constrain the physical result allocation.
        for (dimension, expected) in [
            (
                Dimension::Static(4),
                indoc! {r#"
                    module {
                      func.func @main(%arg0: tensor<4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>> {
                        %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>
                        %1 = stablehlo.custom_call @LayoutConstraint(%0) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>
                        return %1 : tensor<2x4xcomplex<f64>>
                      }
                    }
                "#},
            ),
            (
                dynamic_dimension("size", Some(5)),
                indoc! {r#"
                    module {
                      func.func @main(%arg0: tensor<?xcomplex<f64>, #stablehlo.bounds<4>>) -> tensor<2x?xcomplex<f64>, #stablehlo.bounds<?, 4>> {
                        %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xcomplex<f64>, #stablehlo.bounds<4>>) -> tensor<i32>
                        %c = stablehlo.constant dense<4> : tensor<i32>
                        %1 = stablehlo.set_dimension_size %arg0, %c, dim = 0 : (tensor<?xcomplex<f64>, #stablehlo.bounds<4>>, tensor<i32>) -> tensor<4xcomplex<f64>>
                        %2 = stablehlo.iota dim = 0 : tensor<4xi32>
                        %3 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<4xi32>
                        %4 = stablehlo.compare LT, %2, %3, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
                        %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                        %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
                        %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                        %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<4xf64>
                        %7 = stablehlo.complex %5, %6 : tensor<4xcomplex<f64>>
                        %8 = stablehlo.select %4, %1, %7 : tensor<4xi1>, tensor<4xcomplex<f64>>
                        %9 = stablehlo.broadcast_in_dim %8, dims = [1] : (tensor<4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>
                        %10 = stablehlo.custom_call @LayoutConstraint(%9) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>
                        %11 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xcomplex<f64>, #stablehlo.bounds<4>>) -> tensor<i32>
                        %12 = stablehlo.set_dimension_size %10, %11, dim = 1 : (tensor<2x4xcomplex<f64>>, tensor<i32>) -> tensor<2x?xcomplex<f64>, #stablehlo.bounds<?, 4>>
                        return %12 : tensor<2x?xcomplex<f64>, #stablehlo.bounds<?, 4>>
                      }
                    }
                "#},
            ),
        ] {
            let input_type = ArrayType::new(DataType::C128, Shape::new(vec![dimension.clone()]));
            let output_type = ArrayType::new(DataType::C128, Shape::new(vec![Dimension::Static(2), dimension]))
                .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new())));
            let mut builder = CoreProgramBuilder::<CoreArray, BroadcastOperation>::new();
            let input = builder.add_input(input_type);
            let output = builder
                .add_instruction(BroadcastOperation::new(output_type, vec![1]), Vec::new(), vec![input], None)
                .unwrap()[0];
            let program = builder
                .build::<Vec<CoreArray>, Vec<CoreArray>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            assert_eq!(to_rendered_mlir_for_plain_program(&program, "main").unwrap(), expected);
        }
    }

    #[test]
    fn test_broadcast_operation_lower_output_layout_unsupported() {
        for (layout, kind) in [
            (Layout::Strided(StridedLayout::new(vec![4])), "strided"),
            (Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])), "tiled"),
        ] {
            let output_type = ArrayType::new_static(DataType::F32, [4]).with_layout(layout.clone());
            let mut builder = CoreProgramBuilder::<CoreArray, BroadcastOperation>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F32));
            let output = builder
                .add_instruction(BroadcastOperation::new(output_type, Vec::new()), Vec::new(), vec![input], None)
                .unwrap()[0];
            let program = builder
                .build::<Vec<CoreArray>, Vec<CoreArray>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            assert!(matches!(
                to_rendered_mlir_for_plain_program(&program, "main"),
                Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                    if reason == format!("{kind} output layout `{layout}`"),
            ));
        }
    }

    #[test]
    fn test_broadcast_operation_lower_dynamic_output_layout_requires_finite_bounds() {
        let size = dynamic_dimension("size", None);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![size.clone()]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), size]))
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new())));
        let mut builder = CoreProgramBuilder::<CoreArray, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<CoreArray>, Vec<CoreArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            to_rendered_mlir_for_plain_program(&program, "main"),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                if reason == "dynamic broadcast with explicit output layout needs finite allocation bounds",
        ));
    }

    #[test]
    fn test_broadcast_operation_lower_explicit_sharding_transition() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(4).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let output_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let mut builder = CoreProgramBuilder::<CoreArray, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![0]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<CoreArray>, Vec<CoreArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_rendered_mlir_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_broadcast_operation_lower_explicit_sharding_transition_with_boundary_shardings() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(4).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let output_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let input_sharding = input_type.sharding().unwrap().clone();
        let output_sharding = output_type.sharding().unwrap().clone();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type.clone(), vec![0]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .into_unprojected::<Constant, Operation>()
            .unwrap();

        let stablehlo = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&vec![input_type])
                    .with_output_types(&vec![output_type])
                    .without_boundary_shardings()
                    .with_exact_input_shardings(vec![Some(input_sharding)])
                    .with_exact_output_shardings(vec![Some(output_sharding)]),
            )
            .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_broadcast_operation_lower_explicit_sharding_transition_executes_on_cpu() {
        let client = cpu_client(2);
        let client_devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let input_type =
            test_vector_type(4).with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let output_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let input_sharding = input_type.sharding().unwrap().clone();
        let output_sharding = output_type.sharding().unwrap().clone();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type.clone(), vec![0]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .into_unprojected::<Constant, Operation>()
            .unwrap();
        let module = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&vec![input_type.clone()])
                    .with_output_types(&vec![output_type])
                    .without_boundary_shardings()
                    .with_exact_input_shardings(vec![Some(input_sharding)])
                    .with_exact_output_shardings(vec![Some(output_sharding)]),
            )
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, Some(2)))
            .unwrap();
        let input = Array::from_host_buffer(
            &Session::builder().client(client).build().unwrap(),
            input_type,
            mesh,
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let execution_devices = executable.addressable_devices().unwrap();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let arguments = ExecutionArguments::from_arrays(vec![input], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
            .unwrap()
            .block_until_ready()
            .unwrap();

        assert_eq!(outputs.len(), 2);
        for (output, expected) in outputs.into_iter().zip([[1.0f32, 2.0], [3.0, 4.0]]) {
            assert_eq!(output.outputs.len(), 1);
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(bytes.as_slice()), expected);
        }
    }

    #[test]
    fn test_broadcast_operation_lower_batches_values_with_pending_reductions() {
        // Every observable fact of an array, so that a batched result can be compared with the stacked results of the
        // unbatched function applied to every batch item.
        #[derive(Clone, Debug, PartialEq)]
        struct Observed {
            r#type: ArrayType,
            shards: Vec<Vec<f32>>,
            values: Vec<f32>,
        }

        // Applies the function under test to `x` and `u` in the reduction state `state`, through an operation that
        // commutes with the reduction that completes the values of that state.
        fn combine<V: Add + Max + Min>(state: &str, x: &V, u: &V) -> Result<V, ProgramError> {
            match state {
                "unreduced max" => Max::max(x, u),
                "unreduced min" => Min::min(x, u),
                _ => Add::add(x, u),
            }
        }

        // Returns the first line of the message of `error`. The rest of an MLIR verification diagnostic prints the
        // rejected operation, whose shapes differ between a batched function and the function of one batch item.
        fn diagnostic(error: impl Display) -> String {
            error.to_string().lines().next().unwrap_or_default().to_string()
        }

        // Batches `combine` over the leading axis of `x`, with `u` the same for every batch item.
        macro_rules! batched_combine {
            ($state:expr, $x:expr, $u:expr) => {
                batch(
                    |(x, u)| combine($state, &x, &u),
                    ($x, $u),
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    BatchAxis::new(0),
                    None,
                )
                .map_err(ProgramError::from)
            };
        }

        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let pjrt_devices = client.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            pjrt_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let r#type = |state: &str, shape: &[usize]| {
            let sharding = Sharding::replicated(logical_mesh.clone(), shape.len());
            let sharding = match state {
                "replicated" => sharding,
                "reduced" => sharding.with_reduced_axes(["x"]).unwrap(),
                "unreduced sum" => sharding.with_unreduced_axes(["x"]).unwrap(),
                "unreduced max" => {
                    sharding.with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(UnreducedKind::Max).unwrap()
                }
                "unreduced min" => {
                    sharding.with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(UnreducedKind::Min).unwrap()
                }
                _ => unreachable!("unknown reduction state `{state}`"),
            };
            ArrayType::new_static(DataType::F32, shape.to_vec()).with_sharding(sharding).unwrap()
        };
        // Uploads an array of type `type` whose devices hold the slice `range` of their own contribution when it is
        // unreduced, and of the first contribution otherwise.
        let upload = |r#type: ArrayType, contributions: &[Vec<f32>; 2], range: Range<usize>| {
            let is_unreduced = !r#type.unreduced_axes().is_empty();
            let shape = r#type.static_shape().unwrap();
            let dimensions = shape.dimensions().iter().map(|dimension| *dimension as u64).collect::<Vec<_>>();
            let buffers = pjrt_devices
                .iter()
                .enumerate()
                .map(|(index, device)| {
                    let values = &contributions[if is_unreduced { index } else { 0 }][range.clone()];
                    let bytes = values_to_bytes(values);
                    client.buffer(bytes.as_slice(), BufferType::F32, &dimensions, None, device.clone(), None).unwrap()
                })
                .collect();
            Array::from_addressable_buffers(&session, r#type, mesh.clone(), None, buffers).unwrap()
        };
        let observe = |array: &Array| Observed {
            r#type: array.r#type().into_owned(),
            shards: array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                .collect(),
            values: array.copy_to_host().unwrap().elements::<f32>().unwrap(),
        };
        // Stacks observed batch items along a new leading batch dimension.
        let stack = |items: &[Observed]| Observed {
            r#type: items[0].r#type.batched(0, Dimension::Static(items.len()), ShardingDimension::Replicated).unwrap(),
            shards: (0..items[0].shards.len())
                .map(|device| items.iter().flat_map(|item| item.shards[device].clone()).collect())
                .collect(),
            values: items.iter().flat_map(|item| item.values.clone()).collect(),
        };
        let x_contributions = [
            (1..=16).map(|value| value as f32).collect::<Vec<_>>(),
            (1..=16)
                .map(|value| if value % 2 == 0 { -10. * value as f32 } else { 10. * value as f32 })
                .collect(),
        ];
        let u_contributions = [vec![0.5, -1., 2., 8.], vec![4., 3., -2., -8.]];
        let seed_contributions =
            [vec![9., 10., 11., 12., 13., 14., 15., 16.], vec![-1., 2., -3., 4., -5., 6., -7., 8.]];

        let mut disagreements = Vec::new();
        let states = ["replicated", "reduced", "unreduced sum", "unreduced max", "unreduced min"];
        let pairs = states
            .iter()
            .map(|state| (*state, *state))
            .chain([("replicated", "unreduced sum"), ("unreduced sum", "replicated")]);
        for (x_state, u_state) in pairs {
            let cell = format!("mapped {x_state}, unmapped {u_state}");
            let state = if x_state == "replicated" { u_state } else { x_state };
            let x = upload(r#type(x_state, &[2, 4]), &x_contributions, 0..8);
            let u = upload(r#type(u_state, &[4]), &u_contributions, 0..4);
            let items = (0..4)
                .map(|item| upload(r#type(x_state, &[4]), &x_contributions, 4 * item..4 * item + 4))
                .collect::<Vec<_>>();
            let outputs = items
                .iter()
                .map(|x| combine(state, x, &u).map(|output| observe(&output)).map_err(diagnostic))
                .collect::<Result<Vec<_>, _>>();
            let expected = outputs.as_ref().map(|outputs| stack(&outputs[..2])).map_err(Clone::clone);

            // The function of one batch item rejects a mix of reduction states.
            let expected_rejection = match (x_state, u_state) {
                ("replicated", "unreduced sum") | ("unreduced sum", "replicated") => {
                    Some(format!("`{ADD_OPERATION_NAME}` inputs must be unreduced over the same axes"))
                }
                _ => None,
            };
            if expected.as_ref().err() != expected_rejection.as_ref() {
                disagreements.push(format!("{cell}, expected: {expected:?} does not report {expected_rejection:?}"));
            }

            // Eager batching, and batching nested in batching.
            let eager: Result<Array, _> = batched_combine!(state, x.clone(), u.clone());
            let eager = eager.map(|output| observe(&output)).map_err(diagnostic);
            if eager != expected {
                disagreements.push(format!("{cell}, eager: {eager:?} != {expected:?}"));
            }
            let nested_x = upload(r#type(x_state, &[2, 2, 4]), &x_contributions, 0..16);
            let nested: Result<Array, _> = batch(
                |(x, u)| batched_combine!(state, x, u),
                (nested_x, u.clone()),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            );
            let nested = nested.map(|output| observe(&output)).map_err(diagnostic);
            let expected_nested = outputs
                .as_ref()
                .map(|outputs| stack(&[stack(&outputs[..2]), stack(&outputs[2..])]))
                .map_err(Clone::clone);
            if nested != expected_nested {
                disagreements.push(format!("{cell}, nested: {nested:?} != {expected_nested:?}"));
            }

            // The replay of the staged batched function.
            let input_types = (x.r#type().into_owned(), u.r#type().into_owned());
            let traced: Result<TracedProgram<(ArrayType, ArrayType), ArrayType>, _> =
                trace_with_context(|_, (x, u): (ArrayTracer, ArrayTracer)| batched_combine!(state, x, u), input_types);
            let staged = traced.map_err(diagnostic).and_then(|traced| {
                let inputs = (Value::Array(x.clone()), Value::Array(u.clone()));
                match traced.interpret_in_context(&session, inputs) {
                    Ok(Value::Array(output)) => Ok(observe(&output)),
                    Ok(_) => Err("a batched function of arrays must produce an array".to_string()),
                    Err(error) => Err(diagnostic(error)),
                }
            });
            if staged != expected {
                disagreements.push(format!("{cell}, staged: {staged:?} != {expected:?}"));
            }

            // A compiled batched function.
            let error = RefCell::new(None::<String>);
            let compiled: Result<CompiledFunction<(ArrayType, ArrayType), ArrayType>, _> = compile(
                |(x, u): (CompileTracer, CompileTracer)| {
                    batched_combine!(state, x.clone(), u).unwrap_or_else(|message| {
                        error.replace(Some(diagnostic(message)));
                        x
                    })
                },
                (x.r#type().into_owned(), u.r#type().into_owned()),
                &session,
                mesh.clone(),
            );
            let compiled = match error.take() {
                Some(message) => Err(message),
                None => compiled.map_err(diagnostic),
            };
            let compiled = compiled.and_then(|compiled| {
                let output = compiled.executable().execute(&session, (x.clone(), u.clone()));
                output.map(|output| observe(&output)).map_err(diagnostic)
            });
            if compiled != expected {
                disagreements.push(format!("{cell}, compiled: {compiled:?} != {expected:?}"));
            }

            // Forward mode differentiation of the batched function, with the inputs as their own tangents.
            let jvp = differentiate_at((x.clone(), u.clone()))
                .jvp((x.clone(), u.clone()), |(x, u)| batched_combine!(state, x, u))
                .map(|(primal, tangent): (Array, Array)| (observe(&primal), observe(&tangent)))
                .map_err(diagnostic);
            let expected_jvp = items[..2]
                .iter()
                .map(|x| {
                    differentiate_at((x.clone(), u.clone()))
                        .jvp((x.clone(), u.clone()), |(x, u)| combine(state, &x, &u))
                        .map(|(primal, tangent): (Array, Array)| (observe(&primal), observe(&tangent)))
                        .map_err(diagnostic)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|items| {
                    let (primals, tangents) = items.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                    (stack(&primals), stack(&tangents))
                });
            if jvp != expected_jvp {
                disagreements.push(format!("{cell}, jvp: {jvp:?} != {expected_jvp:?}"));
            }

            // Reverse mode needs a seed of the cotangent type of an output, and a value that is unreduced with the
            // maximum or the minimum kind defines no cotangent.
            let Ok(outputs) = &outputs else { continue };
            if matches!(x_state, "unreduced max" | "unreduced min") {
                continue;
            }
            let seed_type = match outputs[0].r#type.cotangent() {
                Ok(seed_type) => seed_type,
                Err(error) => {
                    disagreements.push(format!("{cell}, seed: {error}"));
                    continue;
                }
            };
            let vjp = |x: Array, seed: Array| {
                let (output, pullback) =
                    differentiate_at((x, u.clone())).vjp(|(x, u)| combine(state, &x, &u)).map_err(diagnostic)?;
                let (x_cotangent, u_cotangent): (Array, Array) =
                    pullback.apply_materialized(seed).map_err(diagnostic)?;
                Ok::<_, String>((observe(&output), observe(&x_cotangent), observe(&u_cotangent)))
            };

            // Batched reverse mode differentiation, with a seed that is the same for every batch item.
            let seed = upload(seed_type.clone(), &seed_contributions, 0..4);
            let batched_vjp: Result<(Array, Array, Array), _> = batch(
                |(x, u, seed)| {
                    let (output, pullback) = differentiate_at((x, u)).vjp(|(x, u)| combine(state, &x, &u))?;
                    let (x_cotangent, u_cotangent) = pullback.apply_materialized(seed)?;
                    Ok((output, x_cotangent, u_cotangent))
                },
                (x.clone(), u.clone(), seed.clone()),
                (BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::replicated()),
                (BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
                None,
            );
            let batched_vjp = batched_vjp
                .map(|(output, x_cotangent, u_cotangent)| {
                    (observe(&output), observe(&x_cotangent), observe(&u_cotangent))
                })
                .map_err(diagnostic);
            let expected_batched_vjp =
                items[..2].iter().map(|x| vjp(x.clone(), seed.clone())).collect::<Result<Vec<_>, _>>().map(|items| {
                    let outputs = items.iter().map(|item| item.0.clone()).collect::<Vec<_>>();
                    let x_cotangents = items.iter().map(|item| item.1.clone()).collect::<Vec<_>>();
                    let u_cotangents = items.iter().map(|item| item.2.clone()).collect::<Vec<_>>();
                    (stack(&outputs), stack(&x_cotangents), stack(&u_cotangents))
                });
            if batched_vjp != expected_batched_vjp {
                disagreements.push(format!("{cell}, batched vjp: {batched_vjp:?} != {expected_batched_vjp:?}"));
            }

            // Reverse mode differentiation of the batched function, with a seed that differs per batch item. The
            // cotangent of the unmapped input is the sum of the cotangents of every batch item.
            let packed_seed_type = seed_type.batched(0, Dimension::Static(2), ShardingDimension::Replicated).unwrap();
            let packed_seed = upload(packed_seed_type, &seed_contributions, 0..8);
            let vjp_of_batched = differentiate_at((x.clone(), u.clone()))
                .vjp(|(x, u)| batched_combine!(state, x, u))
                .map_err(diagnostic)
                .and_then(|(output, pullback)| {
                    let (x_cotangent, u_cotangent): (Array, Array) =
                        pullback.apply_materialized(packed_seed.clone()).map_err(diagnostic)?;
                    let u_cotangent = observe(&u_cotangent);
                    Ok((observe(&output), observe(&x_cotangent), (u_cotangent.r#type, u_cotangent.values)))
                });
            let expected_vjp_of_batched = (0..2)
                .map(|item| {
                    vjp(items[item].clone(), upload(seed_type.clone(), &seed_contributions, 4 * item..4 * item + 4))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|items| {
                    let outputs = items.iter().map(|item| item.0.clone()).collect::<Vec<_>>();
                    let x_cotangents = items.iter().map(|item| item.1.clone()).collect::<Vec<_>>();
                    let u_cotangent =
                        items.iter().map(|item| item.2.values.as_slice()).fold(vec![0.; 4], |sum, values| {
                            sum.iter().zip(values).map(|(sum, value)| sum + value).collect()
                        });
                    (stack(&outputs), stack(&x_cotangents), (items[0].2.r#type.clone(), u_cotangent))
                });
            if vjp_of_batched != expected_vjp_of_batched {
                disagreements
                    .push(format!("{cell}, vjp of batched: {vjp_of_batched:?} != {expected_vjp_of_batched:?}"));
            }
        }
        assert_eq!(disagreements, Vec::<String>::new());
    }

    #[test]
    fn test_broadcast_operation_lower_rejects_bound_manual_output_placement() {
        let mesh = test_manual_mesh("x", 2);
        let boundary = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_output = test_vector_type(2)
            .with_sharding(boundary.clone().with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, input| input.broadcast(local_output.clone(), &[0]),
            test_vector_type(4),
            mesh,
            boundary.clone(),
            boundary,
            ShardMapOptions::new(),
        );
        assert!(matches!(
            module,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "`broadcast` cannot assign bound manual mesh axes to local output dimensions",
            ),
        ));
    }

    #[test]
    fn test_broadcast_operation_lower_converted_operand() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 3);
        let output_type = ArrayType::new_static(DataType::C64, [2, 3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |value| value.convert_element_type(DataType::C64).unwrap().broadcast(output_type.clone(), &[1]).unwrap(),
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[3][sharding={mesh<['x'=1:auto]>, [{}]}] .
                let %1:c64[3][sharding={mesh<['x'=1:auto]>, [{}]}] = convert_element_type [data_type=c64] %0
                    %2:c64[2, 3][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = broadcast [output_type=c64[2, 3][sharding={mesh<['x'=1:auto]>, [{}, {}]}], output_axes=[1]] %1
                in (%2)"},
        );
        assert_eq!(compiled.source_program().captures().len(), 0);
        let input = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3.]).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, input).unwrap();
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        // Complex storage interleaves real and imaginary components after scalar element conversion.
        assert_eq!(values_from_bytes::<f32>(&bytes), vec![1., 0., 2., 0., 3., 0., 1., 0., 2., 0., 3., 0.]);
    }

    #[test]
    fn test_broadcast_like_operation_lower_dynamic_shape() {
        // The exemplar provides the logical size of the dynamic output axis, both for a value that the axis
        // replicates and for a value whose own dynamic axis maps to it next to a stretched unit axis.
        let size = dynamic_dimension("size", Some(5));
        let exemplar_type = ArrayType::new(DataType::F32, Shape::new(vec![size.clone(), Dimension::Static(3)]));
        let column_type = ArrayType::new(DataType::F32, Shape::new(vec![size, Dimension::Static(1)]));
        let mut builder = ProgramBuilder::new();
        let row = builder.add_input(test_vector_type(3));
        let column = builder.add_input(column_type);
        let exemplar = builder.add_input(exemplar_type);
        let rows = builder
            .add_instruction(BroadcastLikeOperation::new(vec![1]), Vec::new(), vec![row, exemplar], None)
            .unwrap()[0];
        let columns = builder
            .add_instruction(BroadcastLikeOperation::new(vec![0, 1]), Vec::new(), vec![column, exemplar], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(
                vec![rows, columns],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = to_rendered_mlir_for_plain_program(&program, "main").unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<?x1xf32, #stablehlo.bounds<4, ?>>, %arg2: tensor<?x3xf32, #stablehlo.bounds<4, ?>>) -> (tensor<?x3xf32, #stablehlo.bounds<4, ?>>, tensor<?x3xf32, #stablehlo.bounds<4, ?>>) {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xf32>) -> tensor<4x3xf32>
                    %1 = stablehlo.get_dimension_size %arg2, dim = 0 : (tensor<?x3xf32, #stablehlo.bounds<4, ?>>) -> tensor<i32>
                    %2 = stablehlo.set_dimension_size %0, %1, dim = 0 : (tensor<4x3xf32>, tensor<i32>) -> tensor<?x3xf32, #stablehlo.bounds<4, ?>>
                    %3 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x1xf32, #stablehlo.bounds<4, ?>>) -> tensor<i32>
                    %c = stablehlo.constant dense<4> : tensor<i32>
                    %4 = stablehlo.set_dimension_size %arg1, %c, dim = 0 : (tensor<?x1xf32, #stablehlo.bounds<4, ?>>, tensor<i32>) -> tensor<4x1xf32>
                    %5 = stablehlo.iota dim = 0 : tensor<4x1xi32>
                    %6 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<4x1xi32>
                    %7 = stablehlo.compare LT, %5, %6, SIGNED : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x1xi1>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x1xf32>
                    %9 = stablehlo.select %7, %4, %8 : tensor<4x1xi1>, tensor<4x1xf32>
                    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x3xf32>
                    %11 = stablehlo.get_dimension_size %arg2, dim = 0 : (tensor<?x3xf32, #stablehlo.bounds<4, ?>>) -> tensor<i32>
                    %12 = stablehlo.set_dimension_size %10, %11, dim = 0 : (tensor<4x3xf32>, tensor<i32>) -> tensor<?x3xf32, #stablehlo.bounds<4, ?>>
                    return %2, %12 : tensor<?x3xf32, #stablehlo.bounds<4, ?>>, tensor<?x3xf32, #stablehlo.bounds<4, ?>>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_broadcast_like_operation_lower_dynamic_shape_requires_finite_bound() {
        let exemplar_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("size", None)]));
        let mut builder = ProgramBuilder::new();
        let value = builder.add_input(ArrayType::scalar(DataType::F32));
        let exemplar = builder.add_input(exemplar_type);
        let output = builder
            .add_instruction(BroadcastLikeOperation::new(Vec::new()), Vec::new(), vec![value, exemplar], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let error = to_rendered_mlir_for_plain_program(&program, "main").unwrap_err();
        assert!(matches!(
            error,
            Error::Lowering(LoweringError::UnsupportedOperation { reason, .. })
                if reason == "dynamic dimension `size` needs a finite positive physical bound",
        ));
    }

    #[test]
    fn test_broadcast_like_operation_lower_executes_dynamic_shape() {
        // One compiled program serves every runtime extent of the exemplar, including an empty one, and the
        // exemplar's elements never reach the results.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(0, Some(5)).unwrap());
        let exemplar_type = ArrayType::new(DataType::F64, Shape::new(vec![rows.clone().into(), Dimension::Static(2)]));
        let column_type = ArrayType::new(DataType::F64, Shape::new(vec![rows.into(), Dimension::Static(1)]));
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let row = builder.add_input(ArrayType::new_static(DataType::F64, [2]).into());
        let column = builder.add_input(column_type.into());
        let exemplar = builder.add_input(exemplar_type.into());
        let rows = builder
            .add_instruction(BroadcastLikeOperation::new(vec![1]), Vec::new(), vec![row, exemplar], None)
            .unwrap()[0];
        let columns = builder
            .add_instruction(BroadcastLikeOperation::new(vec![0, 1]), Vec::new(), vec![column, exemplar], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![rows, columns], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        for extent in [3, 0] {
            let length = extent as usize;
            let outputs = execute_mixed_program(
                execution_client(),
                &program,
                &[
                    MixedValue::Array(vec![1.0, 2.0], vec![2]),
                    MixedValue::Array(vec![10.0, 20.0, 30.0, 40.0], vec![4, 1]),
                    MixedValue::Array(vec![7.0; 8], vec![4, 2]),
                ],
                &[extent, extent],
            )
            .unwrap();
            assert_eq!(
                outputs,
                vec![
                    MixedValue::Array([1.0, 2.0].repeat(length), vec![length as u64, 2]),
                    MixedValue::Array(
                        [10.0, 20.0, 30.0][..length].iter().flat_map(|value| [*value, *value]).collect(),
                        vec![length as u64, 2],
                    ),
                    MixedValue::Dimension(extent as i64),
                    MixedValue::Dimension(extent as i64),
                ],
            );
        }
    }

    #[test]
    fn test_broadcast_like_operation_lower_explicit_sharding_transition() {
        // The output takes the exemplar's sharding, which differs from the value's over an explicit mesh axis.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let value_type = test_vector_type(4).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let exemplar_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let mut builder = ProgramBuilder::new();
        let value = builder.add_input(value_type);
        let exemplar = builder.add_input(exemplar_type);
        let output = builder
            .add_instruction(BroadcastLikeOperation::new(vec![0]), Vec::new(), vec![value, exemplar], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let module = to_rendered_mlir_for_plain_program(&program, "main").unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_dynamic_broadcast_operation_lower_executes_dynamic_shape() {
        let shape = Shape::new(vec![dynamic_dimension("size", Some(5))]);
        let integer_type = ArrayType::new(DataType::I64, shape.clone());
        let complex_type = ArrayType::new(DataType::C64, shape.clone());
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let integer_input = builder.add_input(integer_type.clone().into());
        let complex_input = builder.add_input(complex_type.clone().into());
        let size = builder
            .add_instruction(
                DimensionSizeOperation::new(&integer_type, 0).unwrap(),
                Vec::new(),
                vec![integer_input],
                None,
            )
            .unwrap()[0];
        let leading = builder.add_constant(Constant::Dimension(DimensionValue::constant(2).unwrap()));
        let integer_output = builder
            .add_instruction(
                DynamicBroadcastOperation::new(vec![1]),
                Vec::new(),
                vec![integer_input, leading, size],
                None,
            )
            .unwrap()[0];
        let complex_output = builder
            .add_instruction(
                DynamicBroadcastOperation::new(vec![1]),
                Vec::new(),
                vec![complex_input, leading, size],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![integer_output, complex_output],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();
        // The executable ABI exposes physical buffers followed by each input's runtime size, and returns
        // each dynamic output size separately. One executable must handle both populated and empty shapes.
        let client = execution_client();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, None))
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        // Integers beyond exact f64 precision detect accidental numeric conversion. Negative imaginary parts and
        // signed zeros detect unintended conjugation or loss of exact complex storage.
        let integers = [i64::MAX, -9_007_199_254_740_993, 5, 6];
        let complex_components = [1f32, -2.0, -0.0, 0.0, 3.0, -4.0, 5.0, -6.0];
        for extent in [2i32, 0] {
            let inputs = vec![
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(
                                values_to_bytes(&integers).as_slice(),
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
                                values_to_bytes(&complex_components).as_slice(),
                                BufferType::C64,
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
                                values_to_bytes(&[extent]).as_slice(),
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
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(
                                values_to_bytes(&[extent]).as_slice(),
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
                .unwrap()
                .remove(0);
            let outputs = execution
                .outputs
                .iter()
                .map(|output| output.copy_to_host(None).unwrap().r#await().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(values_from_bytes::<i64>(outputs[2].as_slice()), vec![i64::from(extent)]);
            assert_eq!(values_from_bytes::<i64>(outputs[3].as_slice()), vec![i64::from(extent)]);
            assert_eq!(execution.outputs[0].unpadded_dimensions().unwrap(), &[2, extent as u64]);
            assert_eq!(execution.outputs[1].unpadded_dimensions().unwrap(), &[2, extent as u64]);
            // Verify active payloads, because an empty logical result need not have an empty backing allocation.
            if extent == 2 {
                assert_eq!(
                    values_from_bytes::<i64>(outputs[0].as_slice()),
                    vec![integers[0], integers[1], integers[0], integers[1]],
                );
                assert_eq!(
                    values_from_bytes::<u32>(outputs[1].as_slice()),
                    vec![1f32, -2.0, -0.0, 0.0, 1.0, -2.0, -0.0, 0.0].into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                );
            }
        }
    }

    #[test]
    fn test_dynamic_broadcast_operation_lower_output_layout() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(4)).unwrap());
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::F32).into());
        // Dimension constants have exact types, so the bounded dynamic extents are program inputs.
        let rows_input = builder.add_input(DimensionType::from(rows.clone()).into());
        let columns_input = builder.add_input(DimensionType::from(columns.clone()).into());
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()));
        let output = builder
            .add_instruction(
                DynamicBroadcastOperation::new(Vec::new()).with_output_layout(layout.clone()),
                Vec::new(),
                vec![scalar, rows_input, columns_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![rows.into(), columns.into()])).with_layout(layout);
        assert_eq!(program.output_types(), vec![Type::Array(output_type.clone())]);
        assert!(matches!(
            program.to_rendered_mlir("main", &ProgramLoweringOptions::new().with_input_types(&vec![
                    ArrayType::scalar(DataType::F32),
                    ArrayType::scalar(DataType::I64),
                    ArrayType::scalar(DataType::I64)
                ]).with_output_types(&output_type)),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                if reason == "non-row-major bounded-dynamic output layout `tiled{0,1}`",
        ));
        // An unspecified executable result layout still permits the internal constraint.
        let module = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .with_input_types(&vec![
                        ArrayType::scalar(DataType::F32),
                        ArrayType::scalar(DataType::I64),
                        ArrayType::scalar(DataType::I64),
                    ])
                    .with_output_types(&output_type.with_layout(None)),
            )
            .unwrap();
        // Constrain the bounded physical allocation before attaching the two runtime extents.
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> (tensor<?x?xf32, #stablehlo.bounds<4, 3>>, tensor<i64>, tensor<i64>) {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<4x3xf32>
                    %1 = stablehlo.custom_call @LayoutConstraint(%0) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x3xf32>) -> tensor<4x3xf32>
                    %2 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
                    %3 = stablehlo.set_dimension_size %1, %2, dim = 0 : (tensor<4x3xf32>, tensor<i32>) -> tensor<?x3xf32, #stablehlo.bounds<4, ?>>
                    %4 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<i32>
                    %5 = stablehlo.set_dimension_size %3, %4, dim = 1 : (tensor<?x3xf32, #stablehlo.bounds<4, ?>>, tensor<i32>) -> tensor<?x?xf32, #stablehlo.bounds<4, 3>>
                    %6 = stablehlo.get_dimension_size %5, dim = 0 : (tensor<?x?xf32, #stablehlo.bounds<4, 3>>) -> tensor<i32>
                    %7 = stablehlo.convert %6 : (tensor<i32>) -> tensor<i64>
                    %8 = stablehlo.get_dimension_size %5, dim = 1 : (tensor<?x?xf32, #stablehlo.bounds<4, 3>>) -> tensor<i32>
                    %9 = stablehlo.convert %8 : (tensor<i32>) -> tensor<i64>
                    return %5, %7, %9 : tensor<?x?xf32, #stablehlo.bounds<4, 3>>, tensor<i64>, tensor<i64>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_dynamic_broadcast_operation_lower_output_layout_unsupported() {
        for (layout, kind) in [
            (Layout::Strided(StridedLayout::new(vec![4])), "strided"),
            (Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])), "tiled"),
        ] {
            let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(5)).unwrap());
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let scalar = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let dimension = builder
                .add_constant(Constant::Dimension(DimensionValue::new(DimensionType::from(size.clone()), 2).unwrap()));
            let output = builder
                .add_instruction(
                    DynamicBroadcastOperation::new(Vec::new()).with_output_layout(layout.clone()),
                    Vec::new(),
                    vec![scalar, dimension],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            assert!(matches!(
                program.to_rendered_mlir("main", &ProgramLoweringOptions::new()),
                Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                    if reason == format!("{kind} output layout `{layout}`"),
            ));
        }
    }

    #[test]
    fn test_dynamic_broadcast_operation_lower_output_layout_requires_finite_bounds() {
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(None).unwrap());
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let size_input = builder.add_input(DimensionType::from(size.clone()).into());
        let layout = Layout::Tiled(TiledLayout::new(vec![0], Vec::new()));
        let output = builder
            .add_instruction(
                DynamicBroadcastOperation::new(Vec::new()).with_output_layout(layout),
                Vec::new(),
                vec![scalar, size_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        // An unspecified executable result layout defers the layout check to the operation lowering.
        assert!(matches!(
            program.to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .with_input_types(&vec![ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::I64)])
                    .with_output_types(&ArrayType::new(DataType::F32, Shape::new(vec![size.into()]))),
            ),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                if reason == "dynamic broadcast with explicit output layout needs finite allocation bounds",
        ));
    }
}
