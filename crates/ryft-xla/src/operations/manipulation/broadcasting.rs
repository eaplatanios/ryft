use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, BroadcastOperation, Dimension, DynamicBroadcastOperation, MeshAxisType, Operation,
    ProgramError, Sharding, ShardingDimension,
};
use ryft_mlir::dialects::{stable_hlo, tensor};
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Type, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::types::{
    lower_explicit_shape, lower_physical_bound_value, lower_restore_dynamic_dimensions,
    lower_runtime_dimension_size_i64, lower_static_index_constants, lower_tensor_type, physical_bound_type,
    reshape_dimension_i64,
};
use crate::operations::constants::{
    dynamic_constructor_types, lower_constructor_layout, refine_dynamic_constructor_result,
};
use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for BroadcastOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_broadcast_to_mlir(
            self,
            input_values,
            input_types,
            output_types,
            &lowerer.collective_state.bound_manual_axes,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

/// Returns whether a broadcast requires an explicit output-placement constraint.
pub(crate) fn broadcast_changes_explicit_sharding(
    input_type: &ArrayType,
    output_type: &ArrayType,
    output_axes: &[usize],
) -> bool {
    let Some(output_sharding) = output_type.sharding() else {
        return false;
    };
    let input_sharding = input_type.sharding();
    let uses_explicit_axis = |dimension: &ShardingDimension, sharding: &Sharding| match dimension {
        ShardingDimension::Sharded(axis_names) => {
            axis_names.iter().any(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Explicit))
        }
        ShardingDimension::Replicated | ShardingDimension::Unconstrained => false,
    };
    if input_sharding.is_none() {
        return output_sharding.mesh().axes().iter().any(|axis| axis.r#type() == MeshAxisType::Explicit);
    }
    if let Some(input_sharding) = input_sharding
        && input_sharding.mesh() != output_sharding.mesh()
    {
        return input_sharding
            .mesh()
            .axes()
            .iter()
            .chain(output_sharding.mesh().axes())
            .any(|axis| axis.r#type() == MeshAxisType::Explicit);
    }

    let projected_input_sharding = input_sharding
        .map(|sharding| {
            sharding
                .with_broadcasted_dimensions(output_type.rank(), output_axes)
                .expect("broadcast output axes are validated before lowering")
        })
        .unwrap_or_else(|| Sharding::replicated(output_sharding.mesh().clone(), output_type.rank()));

    // Even when the ranked placement labels are unchanged, expanding a unit dimension that is partitioned over an
    // explicit mesh axis changes which partitions own the replicated values and must remain visible to Shardy.
    let expands_explicit_dimension = output_axes.iter().copied().enumerate().any(|(input_axis, output_axis)| {
        input_type.dimension(input_axis as isize) == Dimension::Static(1)
            && output_type.dimension(output_axis as isize) != Dimension::Static(1)
            && (uses_explicit_axis(&projected_input_sharding.dimensions()[output_axis], &projected_input_sharding)
                || uses_explicit_axis(&output_sharding.dimensions()[output_axis], output_sharding))
    });

    expands_explicit_dimension || projected_input_sharding.conflicts_on_explicit_axes_with(output_sharding)
}

/// Lowers a broadcast and explicitly constrains any placement transition over an explicit mesh axis.
#[allow(clippy::too_many_arguments)]
fn lower_broadcast_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &BroadcastOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    bound_manual_axes: &[String],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    check_count!("input", input_types, 1, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let output_type = &output_types[0];
    let input = input_values[0];
    let result = if output_type.static_shape().is_none() && physical_bound_type(output_type).is_err() {
        if output_type.layout().is_some() {
            return Err(LoweringError::UnsupportedOp {
                op: "dynamic broadcast with explicit output layout needs finite allocation bounds".to_owned(),
            });
        }
        // Every dynamic result axis is a passthrough of a mapped input axis, so obtain its runtime extent there.
        let extents = output_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .map(|(axis, dimension)| {
                if let Dimension::Static(extent) = dimension {
                    reshape_dimension_i64(*extent)?;
                    Ok(lower_static_index_constants(&[*extent], block, context, location)?.remove(0))
                } else {
                    let input_axis =
                        operation.output_axes().iter().position(|output_axis| *output_axis == axis).unwrap();
                    lower_runtime_dimension_size_i64(input, input_axis, block, context, location)
                }
            })
            .collect::<Result<Vec<_>, LoweringError>>()?;
        let shape = lower_explicit_shape(&extents, block, context, location)?;
        let broadcast = block.append_operation(stable_hlo::dynamic_broadcast(
            input,
            shape,
            operation.output_axes(),
            None,
            None,
            location,
        )?)?;
        let result = broadcast.result(0).unwrap().as_ref();
        let expected_type = lower_tensor_type(output_type, context, location)?;
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
        let physical_input = lower_physical_bound_value(input, &input_types[0], 0.0, block, context, location)?;
        let broadcast = block.append_operation(stable_hlo::broadcast(
            physical_input,
            lower_tensor_type(&physical_type, context, location)?,
            operation.output_axes(),
            location,
        )?)?;
        let result = lower_constructor_layout(broadcast.result(0).unwrap().as_ref(), &physical_type, block, location)?;
        let mut sources = vec![(input, 0); output_type.rank()];
        for (input_axis, output_axis) in operation.output_axes().iter().enumerate() {
            sources[*output_axis] = (input, input_axis);
        }
        lower_restore_dynamic_dimensions(result, output_type, &sources, block, context, location)?
    } else {
        let broadcast = block.append_operation(stable_hlo::broadcast(
            input,
            lower_tensor_type(output_type, context, location)?,
            operation.output_axes(),
            location,
        )?)?;
        lower_constructor_layout(broadcast.result(0).unwrap().as_ref(), output_type, block, location)?
    };
    if broadcast_changes_explicit_sharding(&input_types[0], &output_types[0], operation.output_axes()) {
        lower_sharding_constraint(&[result], output_types[0].sharding().unwrap(), bound_manual_axes, block, location)
    } else {
        Ok(vec![result])
    }
}

impl LowerableOperation for DynamicBroadcastOperation {
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
        let block = &mut lowerer.block;
        let Some((input, output_extents)) = input_values.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let Some(input_type) = input_types.first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let input_type = <&ArrayType>::try_from(input_type).map_err(|error| LoweringError::Tracing(error.into()))?;
        let [output_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
        };
        let output_type = <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
        let result = if output_type.static_shape().is_some() {
            let output_tensor_type = lower_tensor_type(output_type, context, location)?;
            let broadcast = block.append_operation(stable_hlo::broadcast(
                *input,
                output_tensor_type,
                self.output_axes(),
                location,
            )?)?;
            let result = broadcast.result(0).unwrap().as_ref();
            lower_constructor_layout(result, output_type, block, location)?
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
            let input = lower_physical_bound_value(*input, input_type, 0.0, block, context, location)?;
            let output_tensor_type = lower_tensor_type(&physical_type, context, location)?;
            let broadcast = block.append_operation(stable_hlo::broadcast(
                input,
                output_tensor_type,
                self.output_axes(),
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
            if output_type.layout().is_some() {
                return Err(LoweringError::UnsupportedOp {
                    op: "dynamic broadcast with explicit output layout needs finite allocation bounds".to_owned(),
                });
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
        if broadcast_changes_explicit_sharding(input_type, output_type, self.output_axes()) {
            lower_sharding_constraint(
                &[result],
                output_type.sharding().unwrap(),
                &collective_state.bound_manual_axes,
                block,
                location,
            )
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
        Array, ArrayIrType, DataType, DimensionBounds, DimensionSizeOperation, DimensionType, DimensionValue,
        DimensionVariable, DynamicBroadcastOperation, Layout, LogicalMesh, MeshAxis, Placeholder, ProgramBuilder,
        ProgramError, Shape, StridedLayout, Tile, TileDimension, TiledLayout, TracedShardMap, trace_shard_map,
    };
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::experimental::lowering::tests::{
        PlainXlaProgram, XlaProgramBuilder, dynamic_dimension, ragged_dot_cpu_compilation_options, test_manual_mesh,
        test_vector_type,
    };
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::{
        XlaArrayConstant, XlaConstant, XlaOperation, XlaProgramBuilder as CompositeXlaProgramBuilder,
    };
    use crate::operations::sharding::shard_map::to_mlir_module;
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{XlaDomain, XlaSession, to_mlir_module_for_program};

    use super::*;

    #[test]
    fn test_broadcast_sharding_transition_detection() {
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = test_vector_type(4).with_sharding(Sharding::replicated(explicit_mesh.clone(), 1)).unwrap();
        let explicitly_sharded = test_vector_type(4)
            .with_sharding(Sharding::new(explicit_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&test_vector_type(4), &replicated, &[0]));
        assert!(broadcast_changes_explicit_sharding(&replicated, &explicitly_sharded, &[0]));
        assert!(!broadcast_changes_explicit_sharding(&explicitly_sharded, &explicitly_sharded, &[0]));

        let first_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let second_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let first_mesh_replicated = test_vector_type(4).with_sharding(Sharding::replicated(first_mesh, 1)).unwrap();
        let second_mesh_replicated = test_vector_type(4).with_sharding(Sharding::replicated(second_mesh, 1)).unwrap();
        assert!(broadcast_changes_explicit_sharding(&first_mesh_replicated, &second_mesh_replicated, &[0]));

        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = test_vector_type(4).with_sharding(Sharding::replicated(manual_mesh.clone(), 1)).unwrap();
        let manually_sharded = test_vector_type(4)
            .with_sharding(
                Sharding::new(manual_mesh, vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert!(!broadcast_changes_explicit_sharding(&replicated, &manually_sharded, &[0]));

        // Expanding a partitioned singleton and changing explicit reduction state both require constraints even when
        // the ranked sharding dimensions themselves remain unchanged.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let singleton = test_vector_type(1)
            .with_sharding(Sharding::new(explicit_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let expanded = test_vector_type(4)
            .with_sharding(Sharding::new(explicit_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&singleton, &expanded, &[0]));

        let unreduced = test_vector_type(4)
            .with_sharding(Sharding::replicated(explicit_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let reduced = test_vector_type(4)
            .with_sharding(Sharding::replicated(explicit_mesh, 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&unreduced, &reduced, &[0]));
    }

    #[test]
    fn test_plain_broadcast_explicit_singleton_expansion_lowers_to_constraint() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(1)
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let output_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let mut builder = ryft_core::ProgramBuilder::<Array, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![0]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

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
    fn test_plain_broadcast_with_dynamic_result_type_lowers_to_bounded_broadcast_in_dim() {
        // Homogeneous broadcast admits a dynamic output extent that is identity-equal to the mapped input extent,
        // which is exactly the payload that program batching stages for a dynamic per-item dimension. Lowering keeps
        // that extent dynamic in the result tensor type and supplies no extent operand, so `stablehlo.broadcast_in_dim`
        // verifies only when the dynamic axis carries an upper bound that lowers to a `#stablehlo.bounds` encoding.
        let bounded = dynamic_dimension("n", Some(5));
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![bounded.clone()]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), bounded]));
        let mut builder = ryft_core::ProgramBuilder::<Array, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

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
        let mut builder = ryft_core::ProgramBuilder::<Array, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(module.matches("stablehlo.dynamic_broadcast_in_dim").count(), 1, "{module}");
        assert_eq!(module.matches("stablehlo.get_dimension_size").count(), 1, "{module}");
    }

    #[test]
    fn test_plain_broadcast_dynamic_pullbacks_execute_on_cpu() {
        /// Executes one homogeneous broadcast or its adjoint through the mixed executable ABI.
        fn execute(program: PlainXlaProgram, values: &[f32], extent: i32) -> (Vec<f32>, Vec<u64>) {
            let input_type = program.input_types().remove(0);
            let output_type = program.output_types().remove(0);
            let dimensions = physical_bound_type(&input_type)
                .unwrap()
                .static_shape()
                .unwrap()
                .dimensions()
                .iter()
                .map(|dimension| *dimension as u64)
                .collect::<Vec<_>>();
            let program = program.into_unprojected::<XlaConstant, XlaOperation>().unwrap();
            let module = to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type.clone()],
                &vec![output_type],
                "main",
                None,
                None,
            )
            .unwrap();
            let client = execution_client();
            let executable = client
                .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
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
                                    &[],
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
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![0, 1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(execute(program, &[2., 5., 7., 9.], 2), (vec![2., 5., 7., 9.], vec![2, 2]));
        assert_eq!(execute(pullback, &[1., 2., 3., 4.], 2), (vec![1., 2., 3., 4.], vec![2, 2]));

        // Reduction removes the stretched singleton, and the pullback reinserts it beside a genuinely dynamic axis.
        // Execute below the allocation bound to ensure the restored axis does not specialize the runtime extent.
        let size = dynamic_dimension("size", Some(5));
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![size.clone(), 1.into()]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![size, 3.into()]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![0, 1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(execute(program, &[2., 5., 0., 0.], 2), (vec![2., 2., 2., 5., 5., 5.], vec![2, 3]));
        assert_eq!(
            execute(pullback, &[1., 2., 3., 4., 5., 6., 0., 0., 0., 0., 0., 0.], 2),
            (vec![6., 15.], vec![2, 1]),
        );
    }

    #[test]
    fn test_plain_broadcast_output_layout() {
        // Both static and bounded dynamic outputs constrain the physical result allocation.
        for dimension in [Dimension::Static(4), dynamic_dimension("size", Some(5))] {
            let input_type = ArrayType::new(DataType::C128, Shape::new(vec![dimension.clone()]));
            let output_type = ArrayType::new(DataType::C128, Shape::new(vec![Dimension::Static(2), dimension]))
                .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new())));
            let mut builder = ProgramBuilder::<Array, BroadcastOperation>::new();
            let input = builder.add_input(input_type);
            let output = builder
                .add_instruction(BroadcastOperation::new(output_type, vec![1]), Vec::new(), vec![input], None)
                .unwrap()[0];
            let program =
                builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
            let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert_eq!(module.matches("stablehlo.custom_call @LayoutConstraint").count(), 1, "{module}");
            assert_eq!(module.matches("result_layouts = [dense<[0, 1]> : tensor<2xindex>]").count(), 1, "{module}");
        }
    }

    #[test]
    fn test_plain_broadcast_output_layout_unsupported() {
        for (layout, kind) in [
            (Layout::Strided(StridedLayout::new(vec![4])), "strided"),
            (Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])), "tiled"),
        ] {
            let output_type = ArrayType::new_static(DataType::F32, [4]).with_layout(layout.clone());
            let mut builder = ProgramBuilder::<Array, BroadcastOperation>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F32));
            let output = builder
                .add_instruction(BroadcastOperation::new(output_type, Vec::new()), Vec::new(), vec![input], None)
                .unwrap()[0];
            let program =
                builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
            assert_eq!(
                to_mlir_module_for_plain_program(&program, "main"),
                Err(LoweringError::UnsupportedOp {
                    op: format!("dynamic constructor with {kind} output layout `{layout}`"),
                }),
            );
        }
    }

    #[test]
    fn test_lower_dynamic_broadcast_executes_dynamic_geometry() {
        let shape = Shape::new(vec![dynamic_dimension("size", Some(5))]);
        let integer_type = ArrayType::new(DataType::I64, shape.clone());
        let complex_type = ArrayType::new(DataType::C64, shape.clone());
        let output_shape = Shape::new(vec![Dimension::Static(2), shape.dimensions()[0].clone()]);
        let mut builder = CompositeXlaProgramBuilder::new();
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
        let leading = builder.add_constant(XlaConstant::Dimension(DimensionValue::constant(2).unwrap()));
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
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![integer_output, complex_output],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![integer_type, complex_type],
            &vec![ArrayType::new(DataType::I64, output_shape.clone()), ArrayType::new(DataType::C64, output_shape)],
            "main",
            None,
            None,
        )
        .unwrap();
        // The executable ABI exposes physical buffers followed by each input's runtime size, and returns
        // each dynamic output size separately. One executable must handle both populated and empty shapes.
        let client = execution_client();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        // Integers beyond exact f64 precision detect accidental numeric conversion. Negative imaginary parts and
        // signed zeros detect unintended conjugation or loss of exact complex storage.
        let integers = [i64::MAX, -9_007_199_254_740_993, 5, 6];
        let complex_components = [1_f32, -2.0, -0.0, 0.0, 3.0, -4.0, 5.0, -6.0];
        for extent in [2_i32, 0] {
            let inputs = vec![
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(
                                values_to_bytes(&integers).as_slice(),
                                BufferType::I64,
                                &[4],
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
                                &[4],
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
                                &[],
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
                                &[],
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
            // Verify active payloads; an empty logical result need not have an empty backing allocation.
            if extent == 2 {
                assert_eq!(
                    values_from_bytes::<i64>(outputs[0].as_slice()),
                    vec![integers[0], integers[1], integers[0], integers[1]],
                );
                assert_eq!(
                    values_from_bytes::<u32>(outputs[1].as_slice()),
                    vec![1_f32, -2.0, -0.0, 0.0, 1.0, -2.0, -0.0, 0.0]
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                );
            }
        }
    }

    #[test]
    fn test_plain_broadcast_explicit_sharding_transition_lowers_to_constraint() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(4).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let output_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let mut builder = ryft_core::ProgramBuilder::<Array, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![0]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

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
    fn test_broadcast_rejects_bound_manual_output_placement() {
        use ryft_core::Broadcast;

        let mesh = test_manual_mesh("x", 2);
        let boundary = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_output = test_vector_type(2)
            .with_sharding(boundary.clone().with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |input| input.broadcast(local_output.clone(), &[0]).unwrap(),
            test_vector_type(4),
            mesh,
            boundary.clone(),
            boundary,
        )
        .unwrap();
        assert!(matches!(
            to_mlir_module(&traced, "main"),
            Err(LoweringError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == "`broadcast` cannot assign bound manual mesh axes to local output dimensions",
        ));
    }

    #[test]
    fn test_broadcast_explicit_sharding_transition_lowers_to_constraint() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(4).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let output_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let input_sharding = input_type.sharding().unwrap().clone();
        let output_sharding = output_type.sharding().unwrap().clone();
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type.clone(), vec![0]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .into_unprojected::<XlaConstant, XlaOperation>()
            .unwrap();

        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type],
            &vec![output_type],
            "main",
            Some(&[input_sharding]),
            Some(&[output_sharding]),
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
    fn test_broadcast_explicit_sharding_transition_executes_on_cpu() {
        use std::collections::HashMap;

        use ryft_core::{Device, DeviceMesh};
        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
        use ryft_pjrt::{ClientOptions, CpuClientOptions, Program as PjrtProgram, load_cpu_plugin};

        use crate::tests::{values_from_bytes, values_to_bytes};
        use crate::{FromPjrt, XlaArray};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
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
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type.clone(), vec![0]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .into_unprojected::<XlaConstant, XlaOperation>()
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type.clone()],
            &vec![output_type],
            "main",
            Some(&[input_sharding]),
            Some(&[output_sharding]),
        )
        .unwrap();
        let executable = client
            .compile(
                &PjrtProgram::Mlir { bytecode: module.into_bytes() },
                &CompilationOptions {
                    argument_layouts: Vec::new(),
                    parameter_is_tupled_arguments: false,
                    executable_build_options: Some(ExecutableCompilationOptions {
                        device_ordinal: -1,
                        replica_count: 1,
                        partition_count: 2,
                        use_spmd_partitioning: true,
                        use_shardy_partitioner: true,
                        ..Default::default()
                    }),
                    compile_portable_executable: false,
                    profile_version: 0,
                    individually_defined_output_indices: Vec::new(),
                    serialized_multi_slice_configuration: Vec::new(),
                    environment_option_overrides: HashMap::new(),
                    target_config: None,
                    allow_in_place_mlir_modification: false,
                    matrix_unit_operand_precision: Precision::Default as i32,
                },
            )
            .unwrap();
        let input = XlaArray::from_host_buffer(
            &XlaSession::new(&client).domain(),
            input_type,
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let execution_devices = executable.addressable_devices().unwrap();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(vec![input], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
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
    fn test_lower_dynamic_broadcast_output_layout() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(4)).unwrap());
        let mut builder = CompositeXlaProgramBuilder::new();
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
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![rows.into(), columns.into()])).with_layout(layout);
        assert_eq!(program.output_types(), vec![ArrayIrType::Array(output_type.clone())]);
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::I64), ArrayType::scalar(DataType::I64)],
            &output_type,
            "main",
            None,
            None,
        )
        .unwrap();
        // Constrain the bounded physical allocation before attaching the two runtime extents.
        assert_eq!(module.matches("stablehlo.custom_call @LayoutConstraint").count(), 1, "{module}");
        assert_eq!(module.matches("result_layouts = [dense<[0, 1]> : tensor<2xindex>]").count(), 1, "{module}");
        assert_eq!(module.matches("stablehlo.set_dimension_size").count(), 2, "{module}");
    }

    #[test]
    fn test_lower_dynamic_broadcast_output_layout_unsupported() {
        for (layout, kind) in [
            (Layout::Strided(StridedLayout::new(vec![4])), "strided"),
            (Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])), "tiled"),
        ] {
            let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(5)).unwrap());
            let mut builder = CompositeXlaProgramBuilder::new();
            let scalar = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let dimension = builder.add_constant(XlaConstant::Dimension(
                DimensionValue::new(DimensionType::from(size.clone()), 2).unwrap(),
            ));
            let output = builder
                .add_instruction(
                    DynamicBroadcastOperation::new(Vec::new()).with_output_layout(layout.clone()),
                    Vec::new(),
                    vec![scalar, dimension],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            let output_type = ArrayType::new(DataType::F32, Shape::new(vec![size.into()])).with_layout(layout.clone());
            assert_eq!(
                to_mlir_module_for_program(
                    &program,
                    &[],
                    &ArrayType::scalar(DataType::F32),
                    &output_type,
                    "main",
                    None,
                    None,
                ),
                Err(LoweringError::UnsupportedOp {
                    op: format!("dynamic constructor with {kind} output layout `{layout}`"),
                }),
            );
        }
    }
}
