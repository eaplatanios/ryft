use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, AxisIndexOperation, DataType, Dimension, MemberOperation, Operation,
    ParallelRaggedAllToAllOperation, ProgramError, ReductionKind, Shape,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::CustomCallApiVersion;
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Size as MlirSize,
    Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::CollectiveLoweringState;
use crate::experimental::lowering::types::{
    lower_constant_elements_attribute, lower_constant_output, lower_restore_dynamic_dimensions, lower_tensor_type,
    lower_u64_scalar_constant, physical_bound_type,
};
use crate::operations::collectives::axis_index::lower_axis_index_to_coordinate;
use crate::operations::collectives::collective_replica_groups;
use crate::operations::reductions::lower_reduce_to_mlir;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, MemberLowerableOperation, XlaLowerer};

/// Emits one overwrite-mode `parallel_ragged_all_to_all` typed-FFI custom call.
fn emit_parallel_ragged_all_to_all_custom_call<'b, 'c: 'b, 't: 'c>(
    collective_state: &CollectiveLoweringState,
    replica_groups: &[Vec<usize>],
    inputs: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let group_size = replica_groups.first().map(Vec::len).unwrap_or(0);
    if replica_groups.iter().any(|group| group.len() != group_size) {
        return Err(ProgramError::MalformedProgram(
            "`parallel_ragged_all_to_all` replica groups must all have the same size".to_string(),
        )
        .into());
    }
    let replica_group_type = context
        .tensor_type(
            context.signless_integer_type(64),
            &[MlirSize::Static(replica_groups.len()), MlirSize::Static(group_size)],
            None,
            location,
        )
        .map_err(|_| LoweringError::InvalidTensorType {
            array_type: ArrayType::new_static(DataType::I64, [replica_groups.len(), group_size]),
        })?;
    let replica_group_values = replica_groups
        .iter()
        .flatten()
        .map(|&device| {
            i64::try_from(device).map_err(|_| {
                ProgramError::MalformedProgram(format!(
                    "`parallel_ragged_all_to_all` replica id {device} cannot be represented as an i64 backend \
                     attribute",
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let replica_groups = context
        .dense_i64_elements_attribute(replica_group_type, replica_group_values.as_slice())
        .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::I64 })?;
    let channel_id = collective_state.next_channel_id();
    let channel_id = i64::try_from(channel_id).map_err(|_| {
        ProgramError::MalformedProgram(format!(
            "`parallel_ragged_all_to_all` channel id {channel_id} cannot be represented as an i64 backend attribute",
        ))
    })?;
    let backend_config = context.dictionary_attribute(&[
        context.named_attribute(context.identifier("replica_groups"), replica_groups.as_ref()),
        context.named_attribute(
            context.identifier("channel_id"),
            context.integer_attribute(context.signless_integer_type(64), channel_id),
        ),
    ]);
    let output_type = inputs[1].r#type()?;
    let custom_call = block.append_operation(stable_hlo::custom_call(
        inputs,
        "ragged_all_to_all",
        false,
        Some(backend_config.as_ref()),
        CustomCallApiVersion::TypedFfi,
        &[],
        None,
        &[],
        None,
        &[output_type],
        location,
    )?)?;
    Ok(custom_call
        .result(0)
        .expect("parallel_ragged_all_to_all custom call should return one result")
        .as_ref())
}

/// Rejects target platforms whose runtime does not register the accelerator ragged-collective custom call.
fn validate_parallel_ragged_all_to_all_target(collective_state: &CollectiveLoweringState) -> Result<(), LoweringError> {
    if collective_state.target_platform().is_some_and(|platform| platform.eq_ignore_ascii_case("cpu")) {
        return Err(ProgramError::UnsupportedOperation {
            message: "`parallel_ragged_all_to_all` is not supported by the XLA CPU backend".to_string(),
        }
        .into());
    }
    Ok(())
}

/// Lowers one named-axis ragged exchange to XLA's typed-FFI `ragged_all_to_all` custom call.
///
/// The public overwrite operation maps directly to one call. The transpose-only additive form rebases every transfer
/// into a private lane of an expanded output, then reduces the lanes and adds the supplied output seed. The custom
/// call therefore still writes disjoint regions while duplicate sends accumulate in the reduction.
pub(crate) fn lower_parallel_ragged_all_to_all_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelRaggedAllToAllOperation,
    collective_state: &CollectiveLoweringState,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if operation.is_physical() {
        return Err(ProgramError::MalformedProgram(
            "`parallel_ragged_all_to_all` lowering does not support the batching-internal physical operand \
             representation"
                .to_string(),
        )
        .into());
    }
    validate_parallel_ragged_all_to_all_target(collective_state)?;
    check_count!("input", input_values, 6, ProgramError);
    check_count!("input", input_types, 6, ProgramError);
    let (replica_groups, effective_axis_size) = collective_replica_groups(
        collective_state,
        operation.axis_name(),
        operation.axis_size(),
        operation.axis_index_groups(),
    )?;
    if !operation.accumulates_updates() {
        return Ok(vec![emit_parallel_ragged_all_to_all_custom_call(
            collective_state,
            replica_groups.as_slice(),
            input_values,
            block,
            context,
            location,
        )?]);
    }

    let metadata_length = input_types[2].shape().dimensions()[0].value().ok_or_else(|| {
        ProgramError::MalformedProgram(
            "`parallel_ragged_all_to_all` additive lowering requires a static metadata length".to_string(),
        )
    })?;
    if metadata_length % effective_axis_size != 0 {
        return Err(ProgramError::MalformedProgram(format!(
            "`parallel_ragged_all_to_all` metadata length {metadata_length} is not divisible by group size \
             {effective_axis_size}",
        ))
        .into());
    }
    let transfers_per_participant = metadata_length / effective_axis_size;
    let output_extent = input_types[1].shape().dimensions()[0].value().ok_or_else(|| {
        ProgramError::MalformedProgram(
            "`parallel_ragged_all_to_all` additive lowering requires a static output leading dimension".to_string(),
        )
    })?;
    let expanded_extent = metadata_length.checked_mul(output_extent).ok_or_else(|| {
        ProgramError::MalformedProgram(
            "`parallel_ragged_all_to_all` additive lowering overflowed its expanded output extent".to_string(),
        )
    })?;
    let mut expanded_dimensions = input_types[1].shape().dimensions().to_vec();
    expanded_dimensions[0] = Dimension::Static(expanded_extent);
    let expanded_output_type = input_types[1].clone().with_shape(Shape::new(expanded_dimensions));
    let expanded_output = if expanded_output_type.static_shape().is_some() {
        lower_constant_output(std::slice::from_ref(&expanded_output_type), 0, block, context, location)?.remove(0)
    } else {
        let physical_type = physical_bound_type(&expanded_output_type)?;
        let zero = lower_constant_output(std::slice::from_ref(&physical_type), 0, block, context, location)?.remove(0);
        let dimension_sources =
            (0..expanded_output_type.rank()).map(|axis| (input_values[1], axis)).collect::<Vec<_>>();
        lower_restore_dynamic_dimensions(
            zero,
            &expanded_output_type,
            dimension_sources.as_slice(),
            block,
            context,
            location,
        )?
    };

    let metadata_type = input_types[2].clone().with_data_type(DataType::U64);
    let metadata_tensor_type = lower_tensor_type(&metadata_type, context, location)?;
    let metadata = input_values[2..]
        .iter()
        .zip(&input_types[2..])
        .map(|(&value, r#type)| {
            if r#type.data_type() == DataType::U64 {
                return Ok(value);
            }
            let converted = block.append_operation(stable_hlo::convert(value, metadata_tensor_type, location)?)?;
            Ok(converted.result(0).expect("stablehlo.convert should return one result").as_ref())
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let metadata_indices = block.append_operation(stable_hlo::iota(metadata_tensor_type, 0, location)?)?;
    let metadata_indices = metadata_indices.result(0).expect("stablehlo.iota should return one result").as_ref();
    let transfers_per_participant_value = lower_u64_scalar_constant(
        u64::try_from(transfers_per_participant).map_err(|_| {
            ProgramError::MalformedProgram(
                "`parallel_ragged_all_to_all` transfers per participant cannot be represented as u64".to_string(),
            )
        })?,
        block,
        context,
        location,
    )?;
    let transfer_count = block.append_operation(stable_hlo::broadcast(
        transfers_per_participant_value,
        metadata_tensor_type,
        &[],
        location,
    )?)?;
    let transfer_count =
        transfer_count.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref();
    let transfer_slots = block.append_operation(stable_hlo::remainder(metadata_indices, transfer_count, location)?)?;
    let transfer_slots = transfer_slots.result(0).expect("stablehlo.remainder should return one result").as_ref();
    let source_axis_index = lower_axis_index_to_coordinate(
        &AxisIndexOperation::new(operation.axis_name().to_string()),
        collective_state,
        block,
        context,
        location,
    )?;
    let source_group_position = if let Some(groups) = operation.axis_index_groups() {
        let mut positions = vec![0_i64; operation.axis_size()];
        for group in groups {
            for (position, &axis_index) in group.iter().enumerate() {
                positions[axis_index] = i64::try_from(position).map_err(|_| {
                    ProgramError::MalformedProgram(format!(
                        "`parallel_ragged_all_to_all` group position {position} cannot be represented as an i64 \
                         backend attribute",
                    ))
                })?;
            }
        }
        let positions_type =
            lower_tensor_type(&ArrayType::new_static(DataType::I64, [operation.axis_size()]), context, location)?;
        let positions = context
            .dense_i64_elements_attribute(positions_type, positions.as_slice())
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::I64 })?;
        let positions = block.append_operation(stable_hlo::constant(positions, location)?)?;
        let positions = positions.result(0).expect("stablehlo.constant should return one result").as_ref();
        let position =
            block.append_operation(stable_hlo::dynamic_slice(positions, &[source_axis_index], &[1], location)?)?;
        let position = position.result(0).expect("stablehlo.dynamic_slice should return one result").as_ref();
        let position = block.append_operation(stable_hlo::convert(
            position,
            lower_tensor_type(&ArrayType::new_static(DataType::U64, [1]), context, location)?,
            location,
        )?)?;
        let position = position.result(0).expect("stablehlo.convert should return one result").as_ref();
        let position = block.append_operation(stable_hlo::reshape_with_output_type(
            position,
            lower_tensor_type(&ArrayType::scalar(DataType::U64), context, location)?,
            location,
        )?)?;
        position.result(0).expect("stablehlo.reshape should return one result").as_ref()
    } else {
        source_axis_index
    };
    let source_lane_base = block.append_operation(stable_hlo::multiply(
        source_group_position,
        transfers_per_participant_value,
        location,
    )?)?;
    let source_lane_base = source_lane_base.result(0).expect("stablehlo.multiply should return one result").as_ref();
    let source_lane_base =
        block.append_operation(stable_hlo::broadcast(source_lane_base, metadata_tensor_type, &[], location)?)?;
    let source_lane_base =
        source_lane_base.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref();
    let lane_indices = block.append_operation(stable_hlo::add(source_lane_base, transfer_slots, location)?)?;
    let lane_indices = lane_indices.result(0).expect("stablehlo.add should return one result").as_ref();
    let lane_stride = lower_constant_elements_attribute(
        DataType::U64,
        metadata_tensor_type,
        i64::try_from(output_extent).map_err(|_| {
            ProgramError::MalformedProgram(format!(
                "`parallel_ragged_all_to_all` output extent {output_extent} cannot be represented by its metadata",
            ))
        })?,
        context,
    )?;
    let lane_stride = block.append_operation(stable_hlo::constant(lane_stride, location)?)?;
    let lane_stride = lane_stride.result(0).expect("stablehlo.constant should return one result").as_ref();
    let lane_offsets = block.append_operation(stable_hlo::multiply(lane_indices, lane_stride, location)?)?;
    let lane_offsets = lane_offsets.result(0).expect("stablehlo.multiply should return one result").as_ref();
    let output_offsets = block.append_operation(stable_hlo::add(metadata[2], lane_offsets, location)?)?;
    let output_offsets = output_offsets.result(0).expect("stablehlo.add should return one result").as_ref();
    let transfer_inputs = [input_values[0], expanded_output, metadata[0], metadata[1], output_offsets, metadata[3]];
    let expanded = emit_parallel_ragged_all_to_all_custom_call(
        collective_state,
        replica_groups.as_slice(),
        transfer_inputs.as_slice(),
        block,
        context,
        location,
    )?;
    let mut lane_dimensions = input_types[1].shape().dimensions().to_vec();
    lane_dimensions[0] = Dimension::Static(output_extent);
    lane_dimensions.insert(0, Dimension::Static(metadata_length));
    let lane_type =
        lower_tensor_type(&input_types[1].clone().with_shape(Shape::new(lane_dimensions)), context, location)?;
    let lanes = block.append_operation(stable_hlo::reshape_with_output_type(expanded, lane_type, location)?)?;
    let lanes = lanes.result(0).expect("stablehlo.reshape should return one result").as_ref();
    let contribution =
        lower_reduce_to_mlir(ReductionKind::Sum, &[0], lanes, &input_types[1], block, context, location)?;
    let sum = block.append_operation(stable_hlo::add(input_values[1], contribution, location)?)?;
    Ok(vec![sum.result(0).expect("stablehlo.add should return one result").as_ref()])
}

impl LowerableOperation for ParallelRaggedAllToAllOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("output", output_types, 1, ProgramError);
        let collective_state = lowerer.collective_state.clone();
        collective_state.check_manual_collective(
            self.name(),
            self.axis_name(),
            self.axis_size(),
            self.mesh(),
            output_types,
            |mesh| self.clone().with_mesh(mesh.clone()).infer_output_types(input_types, &[]),
        )?;
        lower_parallel_ragged_all_to_all_to_mlir(
            self,
            &collective_state,
            input_values,
            input_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

impl MemberLowerableOperation<ArrayIrType> for ParallelRaggedAllToAllOperation {
    fn lower_in_parent<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
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
        collective_state.check_manual_collective(
            self.name(),
            self.axis_name(),
            self.axis_size(),
            self.mesh(),
            output_types,
            |mesh| self.clone().with_mesh(mesh.clone()).infer_parent_output_types(input_types, &[]),
        )?;
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
        lower_parallel_ragged_all_to_all_to_mlir(
            self,
            collective_state,
            input_values,
            array_input_types.as_slice(),
            block,
            context,
            location,
        )
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array, ArrayBatch, ArrayOperation, BatchAxis, BatchAxisSpecification, BatchableOperation, BatchingContext,
        Context, Differentiate, EmptyRegionDriver, LogicalMesh, MeshAxis, MeshAxisType, ParallelRaggedAllToAll, Reduce,
        ShardingDimension, StagingContext, TracedShardMap, TracingContext, Value, batch, trace_shard_map,
    };
    use ryft_mlir::Location;

    use crate::XlaDomain;
    use crate::experimental::XlaArrayTracer;
    use crate::experimental::lowering::tests::test_sharding;
    use crate::operations::sharding::shard_map::to_mlir_module;

    use super::*;

    #[cfg(feature = "cuda-13")]
    use std::sync::Arc;

    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{
        BufferType, ClientOptions, ExecutionDeviceInputs, ExecutionInput, GpuClientOptions, GpuMemoryAllocator,
        GpuPlatform, Program as PjrtProgram, load_cuda_13_plugin,
    };

    #[cfg(feature = "cuda-13")]
    use crate::experimental::lowering::tests::test_spmd_compilation_options;
    #[cfg(feature = "cuda-13")]
    use crate::tests::{values_from_bytes, values_to_bytes};

    #[test]
    fn test_to_mlir_module_renders_parallel_ragged_all_to_all() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let traced: TracedShardMap<XlaDomain<'static>, Vec<ArrayType>, ArrayType> = trace_shard_map(
            |inputs: Vec<XlaArrayTracer>| {
                inputs[0]
                    .parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
                    .unwrap()
            },
            vec![
                ArrayType::new_static(DataType::F32, [6]),
                ArrayType::new_static(DataType::F32, [8]),
                ArrayType::new_static(DataType::I32, [4]),
                ArrayType::new_static(DataType::I32, [4]),
                ArrayType::new_static(DataType::I32, [4]),
                ArrayType::new_static(DataType::I32, [4]),
            ],
            mesh.clone(),
            vec![sharding.clone(); 6],
            sharding,
        )
        .unwrap();
        assert_eq!(
            to_mlir_module(&traced, "main").unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<6xf32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}, %arg1: tensor<8xf32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}, %arg2: tensor<4xi32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}, %arg3: tensor<4xi32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}, %arg4: tensor<4xi32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}, %arg5: tensor<4xi32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                  (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) \
                    in_shardings=[<@mesh, [{\"x\"}]>, <@mesh, [{\"x\"}]>, <@mesh, [{\"x\"}]>, \
                    <@mesh, [{\"x\"}]>, <@mesh, [{\"x\"}]>, <@mesh, [{\"x\"}]>] \
                    out_shardings=[<@mesh, [{\"x\"}]>] manual_axes={\"x\"} \
                    (%arg6: tensor<3xf32>, %arg7: tensor<4xf32>, %arg8: tensor<2xi32>, \
                    %arg9: tensor<2xi32>, %arg10: tensor<2xi32>, %arg11: tensor<2xi32>) {
                      %1 = stablehlo.custom_call @ragged_all_to_all\
                      (%arg6, %arg7, %arg8, %arg9, %arg10, %arg11) \
                      {api_version = 4 : i32, backend_config = {channel_id = 1 : i64, \
                      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}} : \
                      (tensor<3xf32>, tensor<4xf32>, tensor<2xi32>, tensor<2xi32>, \
                      tensor<2xi32>, tensor<2xi32>) -> tensor<4xf32>
                      sdy.return %1 : tensor<4xf32>
                    } : (tensor<6xf32>, tensor<8xf32>, tensor<4xi32>, tensor<4xi32>, \
                    tensor<4xi32>, tensor<4xi32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_renders_parallel_ragged_all_to_all_batched_over_an_unrelated_axis() {
        // A named batch over `y` inside a manual region over `x` merges its batch items into the packed exchange and
        // rebases the varying metadata by offsets that share their variation, so the merged program lowers.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding =
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()], vec![]);
        let traced: TracedShardMap<XlaDomain<'static>, Vec<ArrayType>, ArrayType> = trace_shard_map(
            |inputs: Vec<XlaArrayTracer>| {
                batch(
                    |inputs: Vec<_>| {
                        inputs[0]
                            .parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
                    },
                    inputs,
                    vec![BatchAxis::new(0); 6],
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("y"),
                )
                .unwrap()
            },
            vec![
                ArrayType::new_static(DataType::F32, [4, 3]),
                ArrayType::new_static(DataType::F32, [4, 4]),
                ArrayType::new_static(DataType::I32, [4, 2]),
                ArrayType::new_static(DataType::I32, [4, 2]),
                ArrayType::new_static(DataType::I32, [4, 2]),
                ArrayType::new_static(DataType::I32, [4, 2]),
            ],
            mesh.clone(),
            vec![sharding.clone(); 6],
            sharding,
        )
        .unwrap();
        assert_eq!(
            to_mlir_module(&traced, "main").unwrap(),
            indoc! {"
                #loc = loc(unknown)
                module {
                  sdy.mesh @mesh = <[\"x\"=2]> loc(#loc)
                  func.func @main(%arg0: tensor<4x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>} \
                  loc(unknown), %arg1: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>} \
                  loc(unknown), %arg2: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>} \
                  loc(unknown), %arg3: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>} \
                  loc(unknown), %arg4: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>} \
                  loc(unknown), %arg5: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>} \
                  loc(unknown)) -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) in_shardings=[<@mesh, \
                    [{\"x\"}, {}]>, <@mesh, [{\"x\"}, {}]>, <@mesh, [{\"x\"}, {}]>, <@mesh, [{\"x\"}, {}]>, <@mesh, \
                    [{\"x\"}, {}]>, <@mesh, [{\"x\"}, {}]>] out_shardings=[<@mesh, [{\"x\"}, {}]>] \
                    manual_axes={\"x\"} (%arg6: tensor<2x3xf32> loc(unknown), %arg7: tensor<2x4xf32> loc(unknown), \
                    %arg8: tensor<2x2xi32> loc(unknown), %arg9: tensor<2x2xi32> loc(unknown), %arg10: tensor<2x2xi32> \
                    loc(unknown), %arg11: tensor<2x2xi32> loc(unknown)) {
                      %1 = stablehlo.reshape %arg6 : (tensor<2x3xf32>) -> tensor<6xf32> loc(#loc3)
                      %2 = stablehlo.reshape %arg7 : (tensor<2x4xf32>) -> tensor<8xf32> loc(#loc3)
                      %3 = stablehlo.transpose %arg8, dims = [1, 0] : (tensor<2x2xi32>) -> tensor<2x2xi32> loc(#loc3)
                      %4 = stablehlo.convert %3 : (tensor<2x2xi32>) -> tensor<2x2xui64> loc(#loc3)
                      %5 = stablehlo.iota dim = 1 : tensor<2x2xui64> loc(#loc3)
                      %c = stablehlo.constant dense<3> : tensor<ui64> loc(#loc3)
                      %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui64>) -> tensor<2x2xui64> loc(#loc3)
                      %7 = stablehlo.multiply %5, %6 : tensor<2x2xui64> loc(#loc3)
                      %8 = stablehlo.add %4, %7 : tensor<2x2xui64> loc(#loc3)
                      %9 = stablehlo.reshape %8 : (tensor<2x2xui64>) -> tensor<4xui64> loc(#loc3)
                      %10 = stablehlo.transpose %arg9, dims = [1, 0] : (tensor<2x2xi32>) -> tensor<2x2xi32> loc(#loc3)
                      %11 = stablehlo.convert %10 : (tensor<2x2xi32>) -> tensor<2x2xui64> loc(#loc3)
                      %12 = stablehlo.reshape %11 : (tensor<2x2xui64>) -> tensor<4xui64> loc(#loc3)
                      %13 = stablehlo.transpose %arg10, dims = [1, 0] : (tensor<2x2xi32>) -> tensor<2x2xi32> loc(#loc3)
                      %14 = stablehlo.convert %13 : (tensor<2x2xi32>) -> tensor<2x2xui64> loc(#loc3)
                      %15 = stablehlo.iota dim = 1 : tensor<2x2xui64> loc(#loc3)
                      %c_0 = stablehlo.constant dense<4> : tensor<ui64> loc(#loc3)
                      %16 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui64>) -> tensor<2x2xui64> loc(#loc3)
                      %17 = stablehlo.multiply %15, %16 : tensor<2x2xui64> loc(#loc3)
                      %18 = stablehlo.add %14, %17 : tensor<2x2xui64> loc(#loc3)
                      %19 = stablehlo.reshape %18 : (tensor<2x2xui64>) -> tensor<4xui64> loc(#loc3)
                      %20 = stablehlo.transpose %arg11, dims = [1, 0] : (tensor<2x2xi32>) -> tensor<2x2xi32> loc(#loc3)
                      %21 = stablehlo.convert %20 : (tensor<2x2xi32>) -> tensor<2x2xui64> loc(#loc3)
                      %22 = stablehlo.reshape %21 : (tensor<2x2xui64>) -> tensor<4xui64> loc(#loc3)
                      %23 = stablehlo.custom_call @ragged_all_to_all(%1, %2, %9, %12, %19, %22) {api_version = 4 : \
                      i32, backend_config = {channel_id = 1 : i64, replica_groups = dense<[[0, 1]]> : \
                      tensor<1x2xi64>}} : (tensor<6xf32>, tensor<8xf32>, tensor<4xui64>, tensor<4xui64>, \
                      tensor<4xui64>, tensor<4xui64>) -> tensor<8xf32> loc(#loc3)
                      %24 = stablehlo.reshape %23 : (tensor<8xf32>) -> tensor<2x4xf32> loc(#loc3)
                      %25 = sdy.sharding_constraint %24 <@mesh, [{}, {}]> : tensor<2x4xf32> loc(#loc3)
                      sdy.return %25 : tensor<2x4xf32> loc(#loc)
                    } : (tensor<4x3xf32>, tensor<4x4xf32>, tensor<4x2xi32>, tensor<4x2xi32>, tensor<4x2xi32>, \
                    tensor<4x2xi32>) -> tensor<4x4xf32> loc(#loc)
                    return %0 : tensor<4x4xf32> loc(#loc)
                  } loc(#loc)
                } loc(#loc)
                #loc1 = loc(\"parallel_ragged_all_to_all\")
                #loc2 = loc(\"batching\"(#loc1))
                #loc3 = loc(\"ryft\"(#loc2))
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_renders_grouped_parallel_ragged_all_to_all_transpose_with_group_local_sources() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let traced: TracedShardMap<XlaDomain<'static>, Vec<ArrayType>, ArrayType> = trace_shard_map(
            |inputs: Vec<XlaArrayTracer>| {
                let input_offsets = inputs[2].clone();
                let send_sizes = inputs[3].clone();
                let output_offsets = inputs[4].clone();
                let receive_sizes = inputs[5].clone();
                let (_, gradients) = inputs[0]
                    .domain()
                    .differentiate_at((inputs[0].clone(), inputs[1].clone()))
                    .with_captures((input_offsets, send_sizes, output_offsets, receive_sizes))
                    .value_and_gradient(
                        |(operand, output), (input_offsets, send_sizes, output_offsets, receive_sizes)| {
                            Ok(operand
                                .parallel_ragged_all_to_all_with_axis_index_groups(
                                    "x",
                                    &output,
                                    &input_offsets,
                                    &send_sizes,
                                    &output_offsets,
                                    &receive_sizes,
                                    vec![vec![0, 2], vec![3, 1]],
                                )?
                                .reduce(&[0], ReductionKind::Sum)?)
                        },
                    )
                    .unwrap();
                gradients.0
            },
            vec![
                ArrayType::new_static(DataType::F32, [12]),
                ArrayType::new_static(DataType::F32, [16]),
                ArrayType::new_static(DataType::I32, [16]),
                ArrayType::new_static(DataType::I32, [16]),
                ArrayType::new_static(DataType::I32, [16]),
                ArrayType::new_static(DataType::I32, [16]),
            ],
            mesh.clone(),
            vec![sharding.clone(); 6],
            sharding,
        )
        .unwrap();
        assert_eq!(
            to_mlir_module(&traced, "main").unwrap(),
            indoc! {"
                #loc = loc(unknown)
                module {
                  sdy.mesh @mesh = <[\"x\"=4]> loc(#loc)
                  func.func @main(%arg0: tensor<12xf32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>} loc(unknown), %arg1: tensor<16xf32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>} loc(unknown), %arg2: tensor<16xi32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>} loc(unknown), %arg3: tensor<16xi32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>} loc(unknown), %arg4: tensor<16xi32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>} loc(unknown), %arg5: tensor<16xi32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>} loc(unknown)) -> \
                  (tensor<12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) \
                    in_shardings=[<@mesh, [{\"x\"}]>, <@mesh, [{\"x\"}]>, <@mesh, [{\"x\"}]>, \
                    <@mesh, [{\"x\"}]>, <@mesh, [{\"x\"}]>, <@mesh, [{\"x\"}]>] \
                    out_shardings=[<@mesh, [{\"x\"}]>] manual_axes={\"x\"} \
                    (%arg6: tensor<3xf32> loc(unknown), %arg7: tensor<4xf32> loc(unknown), \
                    %arg8: tensor<4xi32> loc(unknown), %arg9: tensor<4xi32> loc(unknown), \
                    %arg10: tensor<4xi32> loc(unknown), %arg11: tensor<4xi32> loc(unknown)) {
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc(#loc)
                      %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32> loc(#loc)
                      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc3)
                      %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3xf32> loc(#loc3)
                      %3 = \"stablehlo.all_to_all\"(%arg10) \
                      <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, \
                      concat_dimension = 0 : i64, \
                      replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>, split_count = 2 : i64, \
                      split_dimension = 0 : i64}> {use_global_device_ids} : \
                      (tensor<4xi32>) -> tensor<4xi32> loc(#loc3)
                      %4 = \"stablehlo.all_to_all\"(%arg8) \
                      <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, \
                      concat_dimension = 0 : i64, \
                      replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>, split_count = 2 : i64, \
                      split_dimension = 0 : i64}> {use_global_device_ids} : \
                      (tensor<4xi32>) -> tensor<4xi32> loc(#loc3)
                      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc3)
                      %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<12xf32> loc(#loc3)
                      %6 = stablehlo.convert %3 : (tensor<4xi32>) -> tensor<4xui64> loc(#loc3)
                      %7 = stablehlo.convert %arg11 : (tensor<4xi32>) -> tensor<4xui64> loc(#loc3)
                      %8 = stablehlo.convert %4 : (tensor<4xi32>) -> tensor<4xui64> loc(#loc3)
                      %9 = stablehlo.convert %arg9 : (tensor<4xi32>) -> tensor<4xui64> loc(#loc3)
                      %10 = stablehlo.iota dim = 0 : tensor<4xui64> loc(#loc3)
                      %c = stablehlo.constant dense<2> : tensor<ui64> loc(#loc3)
                      %11 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui64>) -> tensor<4xui64> loc(#loc3)
                      %12 = stablehlo.remainder %10, %11 : tensor<4xui64> loc(#loc3)
                      %13 = stablehlo.partition_id : tensor<ui32> loc(#loc3)
                      %14 = stablehlo.convert %13 : (tensor<ui32>) -> tensor<ui64> loc(#loc3)
                      %c_2 = stablehlo.constant dense<[0, 1, 1, 0]> : tensor<4xi64> loc(#loc3)
                      %15 = stablehlo.dynamic_slice %c_2, %14, sizes = [1] : \
                      (tensor<4xi64>, tensor<ui64>) -> tensor<1xi64> loc(#loc3)
                      %16 = stablehlo.convert %15 : (tensor<1xi64>) -> tensor<1xui64> loc(#loc3)
                      %17 = stablehlo.reshape %16 : (tensor<1xui64>) -> tensor<ui64> loc(#loc3)
                      %18 = stablehlo.multiply %17, %c : tensor<ui64> loc(#loc3)
                      %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<ui64>) -> tensor<4xui64> loc(#loc3)
                      %20 = stablehlo.add %19, %12 : tensor<4xui64> loc(#loc3)
                      %c_3 = stablehlo.constant dense<3> : tensor<4xui64> loc(#loc3)
                      %21 = stablehlo.multiply %20, %c_3 : tensor<4xui64> loc(#loc3)
                      %22 = stablehlo.add %8, %21 : tensor<4xui64> loc(#loc3)
                      %23 = stablehlo.custom_call @ragged_all_to_all(%1, %5, %6, %7, %22, %9) \
                      {api_version = 4 : i32, backend_config = {channel_id = 3 : i64, \
                      replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>}} : \
                      (tensor<4xf32>, tensor<12xf32>, tensor<4xui64>, tensor<4xui64>, \
                      tensor<4xui64>, tensor<4xui64>) -> tensor<12xf32> loc(#loc3)
                      %24 = stablehlo.reshape %23 : (tensor<12xf32>) -> tensor<4x3xf32> loc(#loc3)
                      %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc3)
                      %25 = stablehlo.reduce(%24 init: %cst_4) applies stablehlo.add across dimensions = [0] : \
                      (tensor<4x3xf32>, tensor<f32>) -> tensor<3xf32> loc(#loc3)
                      %26 = stablehlo.add %2, %25 : tensor<3xf32> loc(#loc3)
                      sdy.return %26 : tensor<3xf32> loc(#loc)
                    } : (tensor<12xf32>, tensor<16xf32>, tensor<16xi32>, tensor<16xi32>, \
                    tensor<16xi32>, tensor<16xi32>) -> tensor<12xf32> loc(#loc)
                    return %0 : tensor<12xf32> loc(#loc)
                  } loc(#loc)
                } loc(#loc)
                #loc1 = loc(\"parallel_ragged_all_to_all_transpose\")
                #loc2 = loc(\"differentiation\"(#loc1))
                #loc3 = loc(\"ryft\"(#loc2))
            "},
        );
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_to_mlir_module_executes_parallel_ragged_all_to_all_u64_metadata_on_cuda() {
        // XLA's GPU `ragged_all_to_all` receives `u64` metadata from three sources: inputs that are already `u64`, the
        // widened metadata of an exchange merged with an unrelated batch axis, and the rebased metadata of the additive
        // transpose. A single-participant axis exercises all three on one device.
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let device = client.addressable_devices().unwrap()[0].clone();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Manual).unwrap()]).unwrap();
        let execute = |program: String, inputs: Vec<(Vec<u8>, BufferType, Vec<u64>)>| {
            let inputs = inputs
                .into_iter()
                .map(|(bytes, buffer_type, shape)| ExecutionInput {
                    buffer: Arc::new(
                        client.buffer(bytes.as_slice(), buffer_type, shape, None, device.clone(), None).unwrap(),
                    ),
                    donatable: false,
                })
                .collect::<Vec<_>>();
            let executable = client
                .compile(&PjrtProgram::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(1))
                .unwrap();
            let mut outputs = executable
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
                .unwrap();
            let bytes = outputs.remove(0).outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            values_from_bytes::<f32>(bytes.as_slice())
        };
        // Direct `u64` metadata: copy `operand` rows `1..3` into `output` rows `0..2`.
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let traced: TracedShardMap<XlaDomain<'static>, Vec<ArrayType>, ArrayType> = trace_shard_map(
            |inputs: Vec<XlaArrayTracer>| {
                inputs[0]
                    .parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
                    .unwrap()
            },
            vec![
                ArrayType::new_static(DataType::F32, [3]),
                ArrayType::new_static(DataType::F32, [4]),
                ArrayType::new_static(DataType::U64, [1]),
                ArrayType::new_static(DataType::U64, [1]),
                ArrayType::new_static(DataType::U64, [1]),
                ArrayType::new_static(DataType::U64, [1]),
            ],
            mesh.clone(),
            vec![sharding.clone(); 6],
            sharding.clone(),
        )
        .unwrap();
        let metadata = |value: u64| (values_to_bytes::<u64>(&[value]), BufferType::U64, vec![1]);
        assert_eq!(
            execute(
                to_mlir_module(&traced, "main").unwrap(),
                vec![
                    (values_to_bytes::<f32>(&[10.0, 11.0, 12.0]), BufferType::F32, vec![3]),
                    (values_to_bytes::<f32>(&[100.0, 101.0, 102.0, 103.0]), BufferType::F32, vec![4]),
                    metadata(1),
                    metadata(2),
                    metadata(0),
                    metadata(2),
                ],
            ),
            vec![11.0, 12.0, 102.0, 103.0],
        );

        // Merged batching over `y` widens the `i32` metadata to `u64` and rebases the offsets of the second item.
        let batched_sharding =
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()], vec![]);
        let traced: TracedShardMap<XlaDomain<'static>, Vec<ArrayType>, ArrayType> = trace_shard_map(
            |inputs: Vec<XlaArrayTracer>| {
                batch(
                    |inputs: Vec<_>| {
                        inputs[0]
                            .parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
                    },
                    inputs,
                    vec![BatchAxis::new(0); 6],
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("y"),
                )
                .unwrap()
            },
            vec![
                ArrayType::new_static(DataType::F32, [2, 3]),
                ArrayType::new_static(DataType::F32, [2, 4]),
                ArrayType::new_static(DataType::I32, [2, 1]),
                ArrayType::new_static(DataType::I32, [2, 1]),
                ArrayType::new_static(DataType::I32, [2, 1]),
                ArrayType::new_static(DataType::I32, [2, 1]),
            ],
            mesh.clone(),
            vec![batched_sharding.clone(); 6],
            batched_sharding,
        )
        .unwrap();
        let program = to_mlir_module(&traced, "main").unwrap();
        assert!(program.contains("tensor<2xui64>"));
        let metadata = |values: [i32; 2]| (values_to_bytes::<i32>(&values), BufferType::I32, vec![2, 1]);
        assert_eq!(
            execute(
                program,
                vec![
                    (values_to_bytes::<f32>(&[10.0, 11.0, 12.0, 20.0, 21.0, 22.0]), BufferType::F32, vec![2, 3]),
                    (
                        values_to_bytes::<f32>(&[100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0]),
                        BufferType::F32,
                        vec![2, 4],
                    ),
                    metadata([1, 0]),
                    metadata([2, 1]),
                    metadata([0, 3]),
                    metadata([2, 1]),
                ],
            ),
            vec![11.0, 12.0, 102.0, 103.0, 200.0, 201.0, 202.0, 20.0],
        );

        // The gradient of the summed exchange with respect to `operand` lowers through the additive transpose, whose
        // rebased metadata are `u64`. Only the sent rows `1..3` reach the sum.
        let traced: TracedShardMap<XlaDomain<'static>, Vec<ArrayType>, ArrayType> = trace_shard_map(
            |inputs: Vec<XlaArrayTracer>| {
                inputs[0]
                    .domain()
                    .differentiate_at(inputs[0].clone())
                    .with_captures((
                        inputs[1].clone(),
                        inputs[2].clone(),
                        inputs[3].clone(),
                        inputs[4].clone(),
                        inputs[5].clone(),
                    ))
                    .gradient(|operand, (output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                        operand
                            .parallel_ragged_all_to_all(
                                "x",
                                &output,
                                &input_offsets,
                                &send_sizes,
                                &output_offsets,
                                &receive_sizes,
                            )
                            .map(|result| result.reduce(&[0], ReductionKind::Sum).unwrap())
                    })
                    .unwrap()
            },
            vec![
                ArrayType::new_static(DataType::F32, [3]),
                ArrayType::new_static(DataType::F32, [4]),
                ArrayType::new_static(DataType::I32, [1]),
                ArrayType::new_static(DataType::I32, [1]),
                ArrayType::new_static(DataType::I32, [1]),
                ArrayType::new_static(DataType::I32, [1]),
            ],
            mesh,
            vec![sharding.clone(); 6],
            sharding,
        )
        .unwrap();
        let program = to_mlir_module(&traced, "main").unwrap();
        assert!(program.contains("tensor<1xui64>"));
        let metadata = |value: i32| (values_to_bytes::<i32>(&[value]), BufferType::I32, vec![1]);
        assert_eq!(
            execute(
                program,
                vec![
                    (values_to_bytes::<f32>(&[10.0, 11.0, 12.0]), BufferType::F32, vec![3]),
                    (values_to_bytes::<f32>(&[0.0; 4]), BufferType::F32, vec![4]),
                    metadata(1),
                    metadata(2),
                    metadata(0),
                    metadata(2),
                ],
            ),
            vec![0.0, 1.0, 1.0],
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_lowering_rejects_batching_internal_physical_representation() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let operand = context.input(ArrayType::new_static(DataType::F32, [2, 3]));
        let output = context.input(ArrayType::new_static(DataType::F32, [2, 4]));
        let metadata = context.lift(Array::matrix(2, 2, vec![0_i32; 4]).unwrap()).unwrap();
        let inputs = [operand, output, metadata.clone(), metadata.clone(), metadata.clone(), metadata]
            .into_iter()
            .map(|input| ArrayBatch::new(input, BatchAxis::new(0)).unwrap())
            .collect::<Vec<_>>();
        let batching_context = BatchingContext::new(context.clone(), 2).with_axis_name("x".to_string());
        let outputs = ParallelRaggedAllToAllOperation::new("x".to_string(), 2)
            .batch(&batching_context, &EmptyRegionDriver, inputs.as_slice())
            .unwrap();
        assert_eq!(outputs.len(), 1);

        let builder = context.builder();
        let builder = builder.borrow();
        let operation = builder
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                ArrayOperation::ParallelRaggedAllToAll(operation) => Some(operation.clone()),
                _ => None,
            })
            .unwrap();
        assert!(operation.is_physical());
        drop(builder);

        let mlir_context = MlirContext::new();
        let location = mlir_context.unknown_location();
        let module = mlir_context.module(location).unwrap();
        let mut block = module.body().unwrap();
        assert!(matches!(
            lower_parallel_ragged_all_to_all_to_mlir(
                &operation,
                &CollectiveLoweringState::new(),
                &[],
                &[],
                &mut block,
                &mlir_context,
                location.as_ref(),
            ),
            Err(LoweringError::Tracing(ProgramError::MalformedProgram(message)))
                if message
                    == "`parallel_ragged_all_to_all` lowering does not support the batching-internal physical operand \
                        representation",
        ));
    }
}
