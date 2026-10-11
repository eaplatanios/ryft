use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, CollectiveMode, Dimension, MemberOperation, Operation, ParallelAllToAllOperation,
    ProgramError,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::CollectiveLoweringState;
use crate::experimental::lowering::types::lower_tensor_type;
use crate::operations::collectives::{
    collapse_singleton_axis, collective_replica_groups, refine_collective_result_dimensions,
};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, MemberLowerableOperation, XlaLowerer};

/// Lowers one traced `parallel_all_to_all` to a channeled `stablehlo.all_to_all` over the named mesh axis's replica
/// groups.
pub(crate) fn lower_parallel_all_to_all_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelAllToAllOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    input_array_type: &ArrayType,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let (replica_groups, axis_size) = collective_replica_groups(
        collective_state,
        operation.axis_name(),
        operation.axis_size(),
        operation.options().axis_index_groups(),
    )?;
    let replica_groups: Vec<&[usize]> = replica_groups.iter().map(Vec::as_slice).collect();
    let (input_value, split_axis, concatenation_axis) = match operation.options().mode() {
        CollectiveMode::Tiled => (input_value, operation.split_axis(), operation.concatenation_axis()),
        CollectiveMode::Untiled if operation.split_axis() == operation.concatenation_axis() => {
            (input_value, operation.split_axis(), operation.concatenation_axis())
        }
        CollectiveMode::Untiled => {
            let mut split_axis = operation.split_axis();
            let mut concatenation_axis = operation.concatenation_axis();
            if split_axis < concatenation_axis {
                concatenation_axis += 1;
            } else {
                split_axis += 1;
            }
            let expanded_type = input_array_type
                .with_inserted_dimension(concatenation_axis, Dimension::Static(1))
                .map_err(ProgramError::from)?;
            let expanded_type = lower_tensor_type(&expanded_type, context, location)?;
            let broadcast_dimensions =
                (0..input_array_type.rank() + 1).filter(|&axis| axis != concatenation_axis).collect::<Vec<_>>();
            let expanded = block.append_operation(stable_hlo::broadcast(
                input_value,
                expanded_type,
                broadcast_dimensions.as_slice(),
                location,
            )?)?;
            (
                expanded.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref(),
                split_axis,
                concatenation_axis,
            )
        }
    };
    let result = block.append_operation(stable_hlo::all_to_all(
        &[input_value],
        split_axis,
        axis_size,
        concatenation_axis,
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        location,
    )?)?;
    let result = result.result(0).expect("stablehlo.all_to_all should return one result").as_ref();
    if operation.options().mode() == CollectiveMode::Tiled || split_axis == concatenation_axis {
        return Ok(vec![result]);
    }
    Ok(vec![collapse_singleton_axis(result, output_array_type, block, context, location)?])
}

impl LowerableOperation for ParallelAllToAllOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 1, ProgramError);
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
        lower_parallel_all_to_all_to_mlir(
            self,
            &collective_state,
            input_values[0],
            &input_types[0],
            &output_types[0],
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

impl MemberLowerableOperation<ArrayIrType> for ParallelAllToAllOperation {
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
        let Some(input) = input_values.first() else {
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
        let result = lower_parallel_all_to_all_to_mlir(
            self,
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
}
