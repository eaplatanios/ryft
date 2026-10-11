use ryft_core::macros::check_count;
use ryft_core::{ArrayType, Operation, ParallelPermuteOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, BlockRef, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::experimental::lowering::CollectiveLoweringState;
use crate::operations::collectives::collective_replica_groups;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Lowers one traced `parallel_permute` to a channeled `stablehlo.collective_permute`, expanding the axis-local
/// `(source, target)` pairs to global device pairs across the named mesh axis's replica groups.
pub(crate) fn lower_parallel_permute_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelPermuteOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let (replica_groups, _) =
        collective_replica_groups(collective_state, operation.axis_name(), operation.axis_size(), None)?;
    let mut source_target_pairs = Vec::with_capacity(replica_groups.len() * operation.source_target_pairs().len());
    for group in &replica_groups {
        for (source, target) in operation.source_target_pairs() {
            source_target_pairs.push((group[*source], group[*target]));
        }
    }
    let result = block.append_operation(stable_hlo::collective_permute(
        input_value,
        source_target_pairs.as_slice(),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        location,
    )?)?;
    Ok(vec![result.result(0).expect("stablehlo.collective_permute should return one result").as_ref()])
}

impl LowerableOperation for ParallelPermuteOperation {
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
        lower_parallel_permute_to_mlir(self, &collective_state, input_values[0], &mut lowerer.block, lowerer.location)
    }
}
