use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, CollectiveMode, Dimension, MemberOperation, Operation, ParallelAllGatherOperation,
    ProgramError, Shape,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::CollectiveLoweringState;
use crate::experimental::lowering::types::lower_tensor_type;
use crate::operations::collectives::{collective_replica_groups, refine_collective_result_dimensions};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, MemberLowerableOperation, XlaLowerer};

/// Lowers one traced `parallel_all_gather` to a channeled `stablehlo.all_gather` over the named mesh axis's replica
/// groups.
pub(crate) fn lower_parallel_all_gather_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelAllGatherOperation,
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
    let input_value = match operation.options().mode() {
        CollectiveMode::Tiled => input_value,
        CollectiveMode::Untiled => {
            let mut dimensions = output_array_type.shape().dimensions().to_vec();
            dimensions[operation.concatenation_axis()] = Dimension::Static(1);
            let input_type =
                lower_tensor_type(&output_array_type.clone().with_shape(Shape::new(dimensions)), context, location)?;
            let broadcast_dimensions = (0..output_array_type.rank())
                .filter(|&axis| axis != operation.concatenation_axis())
                .collect::<Vec<_>>();
            let broadcast = block.append_operation(stable_hlo::broadcast(
                input_value,
                input_type,
                broadcast_dimensions.as_slice(),
                location,
            )?)?;
            broadcast.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref()
        }
    };
    let output_type = lower_tensor_type(output_array_type, context, location)?;
    let result = block.append_operation(stable_hlo::all_gather(
        &[input_value],
        operation.concatenation_axis(),
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        &[output_type],
        location,
    )?)?;
    Ok(vec![result.result(0).expect("stablehlo.all_gather should return one result").as_ref()])
}

impl LowerableOperation for ParallelAllGatherOperation {
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
        lower_parallel_all_gather_to_mlir(
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

impl MemberLowerableOperation<ArrayIrType> for ParallelAllGatherOperation {
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
            lower_parallel_all_gather_to_mlir(self, collective_state, *input, output_type, block, context, location)?
                .remove(0);
        refine_collective_result_dimensions(result, &input_values[1..], output_type, block, context, location)
    }
}
