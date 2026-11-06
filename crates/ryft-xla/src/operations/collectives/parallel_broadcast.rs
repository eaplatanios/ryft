use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, Operation as CoreOperation, PARALLEL_BROADCAST_OPERATION_NAME, ParallelBroadcastOperation, ProgramError,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, BlockRef, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::experimental::lowering::CollectiveLoweringState;
use crate::operations::collectives::collective_replica_groups;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

/// Lowers one traced `parallel_broadcast` to a channeled `stablehlo.collective_broadcast`. The operation broadcasts
/// from the first member of every replica group, so each group of the named mesh axes is reordered to start with its
/// member at the source position, followed by the remaining members in axis order, as JAX's GPU lowering of
/// `pbroadcast` does. The channel handle and global device IDs in the groups make XLA interpret the devices as
/// partitions of one replica, which is how `shard_map` regions execute.
///
/// A broadcast over a single participant is the identity and emits nothing. Otherwise, the XLA CPU backend cannot
/// compile `collective-broadcast` (its thunk emitter rejects the opcode), so a lowering whose target platform is known
/// to be the CPU fails here with a precise error, while a lowering without target information emits the operation and
/// leaves the rejection to XLA's compiler. JAX registers its lowering for GPUs only.
pub(crate) fn lower_parallel_broadcast_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelBroadcastOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    if operation.axis_size() == 1 {
        return Ok(vec![input_value]);
    }
    if collective_state.target_platform().is_some_and(|platform| platform.eq_ignore_ascii_case("cpu")) {
        return Err(ProgramError::UnsupportedOperation {
            message: format!("`{PARALLEL_BROADCAST_OPERATION_NAME}` is not supported by the XLA CPU backend"),
        }
        .into());
    }
    let (replica_groups, _) =
        collective_replica_groups(collective_state, operation.axes(), operation.axis_size(), None)?;
    let source = operation.source();
    let replica_groups = replica_groups
        .iter()
        .map(|group| {
            std::iter::once(group[source])
                .chain(group[..source].iter().copied())
                .chain(group[source + 1..].iter().copied())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let replica_groups = replica_groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let result = block.append_operation(stable_hlo::collective_broadcast(
        &[input_value],
        None,
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        location,
    )?)?;
    Ok(vec![result.result(0).expect("stablehlo.collective_broadcast should return one result").as_ref()])
}

impl LowerableOperation for ParallelBroadcastOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let collective_state = lowerer.collective_state.clone();
        collective_state.check_manual_collective(
            self.name(),
            self.axes(),
            self.axis_size(),
            self.mesh(),
            output_types,
            |mesh| {
                let operation = match mesh {
                    Some(mesh) => self.clone().with_mesh(mesh.clone()),
                    None => self.clone(),
                };
                operation.infer_output_types(input_types, &[])
            },
        )?;
        lower_parallel_broadcast_to_mlir(self, &collective_state, input_values[0], &mut lowerer.block, lowerer.location)
    }
}
