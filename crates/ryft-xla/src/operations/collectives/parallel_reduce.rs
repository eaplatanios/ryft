use ryft_core::macros::check_count;
use ryft_core::{ArrayType, Operation, ParallelReduceOperation, ProgramError, ReductionKind};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::CollectiveLoweringState;
use crate::experimental::lowering::types::{lower_f64_constant_splat, lower_tensor_type};
use crate::operations::collectives::{collective_replica_groups, mesh_axis_replica_groups};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Lowers one [`ParallelReduceOperation`] inside a `sdy.manual_computation` region to a `stablehlo.all_reduce` over
/// the device mesh axis the collective names.
///
/// The replica groups are the dense global device-id groups that [`mesh_axis_replica_groups`] derives from the mesh's
/// row-major device linearization, or, when the collective records explicit logical participant groups, those groups
/// expanded inside every fixed coordinate of the mesh's other axes without reordering either the groups or their
/// members. The emitted operation carries a module-unique channel id with a device-to-device channel type and
/// `use_global_device_ids`, the standard SPMD emission for cross-partition collectives.
///
/// Each kind combines with the single StableHLO operation of its reduction (i.e., `add` for `Sum` and `Mean`,
/// `multiply` for `Product`, `maximum` and `minimum` for `Max` and `Min`, and `or` and `and` for `Any` and `All`), and
/// a `Mean` then divides the all-reduced sum by the effective group size. Unlike the `stablehlo.reduce` lowering, the
/// combiner never needs a guard for an identity element because an all-reduce combines only its participants' values,
/// and it must stay a single operation because XLA backends implement all-reduce only for the combiners that they
/// recognize. Floating-point extrema therefore follow the backend's native `maximum` and `minimum`, as
/// `jax.lax.pmax` and `jax.lax.pmin` do, so their NaN and signed-zero behavior is backend-defined. The combiner also
/// accumulates in the input element type, without the `f32` accumulation that [`ReductionKind::Sum`] documents for
/// narrower floating-point `reduce` inputs, so that, e.g., `bf16` gradient all-reduces keep their communication
/// volume. A `LogSumExp` has no single combiner and is rejected, because the
/// [`ParallelReduce`](ryft_core::ParallelReduce) capability composes it from a mesh maximum and a mesh sum instead. A
/// collective lowered outside any manual region, or naming an axis the innermost region does not bind as a manual mesh
/// axis, is an error.
pub(crate) fn lower_collective_to_all_reduce<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelReduceOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if let Some(mesh) = operation.mesh()
        && collective_state
            .manual_axis_region(operation.axis_name())
            .is_some_and(|region| region.mesh() != mesh)
    {
        return Err(ProgramError::MalformedProgram(format!(
            "`{}` mesh does not match its active manual binding",
            operation.name(),
        ))
        .into());
    }
    let (replica_groups, effective_axis_size) = match operation.axis_index_groups() {
        Some(axis_index_groups) => collective_replica_groups(
            collective_state,
            operation.axis_name(),
            operation.axis_size().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "grouped `{}` over axis `{}` does not record the full axis size",
                    operation.name(),
                    operation.axis_name(),
                ))
            })?,
            Some(axis_index_groups),
        )?,
        None => mesh_axis_replica_groups(collective_state, operation.axis_name())?,
    };
    let replica_groups: Vec<&[usize]> = replica_groups.iter().map(Vec::as_slice).collect();

    let scalar_tensor_type = lower_tensor_type(&ArrayType::scalar(output_array_type.data_type()), context, location)?;
    let mut computation = context.region();
    let mut combiner_block =
        computation.append_block(context.block(&[(scalar_tensor_type, location), (scalar_tensor_type, location)]))?;
    let left = combiner_block.argument(0)?.as_ref();
    let right = combiner_block.argument(1)?.as_ref();
    let combined = match operation.kind() {
        ReductionKind::Sum | ReductionKind::Mean => combiner_block
            .append_operation(stable_hlo::add(left, right, location)?)?
            .result(0)
            .unwrap()
            .as_ref(),
        ReductionKind::Product => combiner_block
            .append_operation(stable_hlo::multiply(left, right, location)?)?
            .result(0)
            .unwrap()
            .as_ref(),
        ReductionKind::Max => combiner_block
            .append_operation(stable_hlo::maximum(left, right, location)?)?
            .result(0)
            .unwrap()
            .as_ref(),
        ReductionKind::Min => combiner_block
            .append_operation(stable_hlo::minimum(left, right, location)?)?
            .result(0)
            .unwrap()
            .as_ref(),
        ReductionKind::Any => {
            combiner_block.append_operation(stable_hlo::or(left, right, location)?)?.result(0).unwrap().as_ref()
        }
        ReductionKind::All => combiner_block
            .append_operation(stable_hlo::and(left, right, location)?)?
            .result(0)
            .unwrap()
            .as_ref(),
        ReductionKind::LogSumExp => {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{}` with kind `{}` has no all-reduce combiner; stage it through the `ParallelReduce` \
                     capability, which composes it from primitive mesh reductions",
                    operation.name(),
                    operation.kind(),
                ),
            }
            .into());
        }
    };
    combiner_block.append_operation(stable_hlo::r#return(&[combined], location)?)?;
    let result = block.append_operation(stable_hlo::all_reduce(
        &[input_value],
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        computation,
        location,
    )?)?;
    let reduced = result.result(0).expect("stablehlo.all_reduce should return one result").as_ref();
    if operation.kind() != ReductionKind::Mean {
        return Ok(reduced);
    }
    // `Mean` is the mean over the effective participant group: divide the all-reduced sum by the group size.
    let output_tensor_type = lower_tensor_type(output_array_type, context, location)?;
    let divisor = lower_f64_constant_splat(
        effective_axis_size as f64,
        output_array_type,
        output_tensor_type,
        block,
        context,
        location,
    )?;
    let mean = block.append_operation(stable_hlo::divide(reduced, divisor, location)?)?;
    Ok(mean.result(0).expect("stablehlo.divide should return one result").as_ref())
}

impl LowerableOperation for ParallelReduceOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        // This plain dispatch serves nested programs (control-flow bodies, inlined custom-function primals,
        // and rematerialized bodies), which can sit inside a shard_map manual region: the threaded
        // `CollectiveLoweringState` resolves the collective's mesh axis there and errors outside manual
        // regions (a batched axis would have been consumed into a `Reduce` at trace time).
        check_count!("input", input_values, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let collective_state = lowerer.collective_state.clone();
        let result = lower_collective_to_all_reduce(
            self,
            &collective_state,
            input_values[0],
            &output_types[0],
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        Ok(vec![result])
    }
}
