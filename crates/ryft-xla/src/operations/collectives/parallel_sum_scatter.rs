use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, CollectiveMode, Dimension, MemberOperation, Operation, ParallelSumScatterOperation,
    ProgramError, ReductionKind,
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
use crate::operations::reductions::build_reduce_body_region;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, MemberLowerableOperation, XlaLowerer};

/// Lowers one traced `parallel_sum_scatter` to a channeled `stablehlo.reduce_scatter` with a sum reduction over the
/// named mesh axis's replica groups.
pub(crate) fn lower_parallel_sum_scatter_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelSumScatterOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let (replica_groups, _) = collective_replica_groups(
        collective_state,
        operation.axis_name(),
        operation.axis_size(),
        operation.options().axis_index_groups(),
    )?;
    let replica_groups: Vec<&[usize]> = replica_groups.iter().map(Vec::as_slice).collect();
    let computation = build_reduce_body_region(ReductionKind::Sum, output_array_type.data_type(), context, location)?;
    let native_output_type = match operation.options().mode() {
        CollectiveMode::Tiled => output_array_type.clone(),
        CollectiveMode::Untiled => output_array_type
            .with_inserted_dimension(operation.scatter_axis(), Dimension::Static(1))
            .map_err(ProgramError::from)?,
    };
    let output_type = lower_tensor_type(&native_output_type, context, location)?;
    let result = block.append_operation(stable_hlo::reduce_scatter(
        input_value,
        operation.scatter_axis(),
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        computation,
        output_type,
        location,
    )?)?;
    let result = result.result(0).expect("stablehlo.reduce_scatter should return one result").as_ref();
    if operation.options().mode() == CollectiveMode::Tiled {
        return Ok(vec![result]);
    }
    Ok(vec![collapse_singleton_axis(result, output_array_type, block, context, location)?])
}

impl LowerableOperation for ParallelSumScatterOperation {
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
        lower_parallel_sum_scatter_to_mlir(
            self,
            &collective_state,
            input_values[0],
            &output_types[0],
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

impl MemberLowerableOperation<ArrayIrType> for ParallelSumScatterOperation {
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
        let [output_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
        };
        let output_type = <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
        let result =
            lower_parallel_sum_scatter_to_mlir(self, collective_state, *input, output_type, block, context, location)?
                .remove(0);
        refine_collective_result_dimensions(result, &input_values[1..], output_type, block, context, location)
    }
}
