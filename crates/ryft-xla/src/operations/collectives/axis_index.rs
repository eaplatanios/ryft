use ryft_core::macros::check_count;
use ryft_core::{
    AXIS_INDEX_OPERATION_NAME, ArrayType, AxisIndexOperation, DataType, ProgramError, SHARD_MAP_OPERATION_NAME,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::CollectiveLoweringState;
use crate::experimental::lowering::types::{lower_tensor_type, lower_u64_scalar_constant};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Lowers one [`AxisIndexOperation`] inside a `sdy.manual_computation` region to the executing device's coordinate
/// along the named device mesh axis, as a scalar `u64`.
///
/// Devices are linearized row-major over the mesh axes, so the coordinate along the named axis is `(partition_id /
/// stride) % size`, where `stride` is the product of the mesh axis sizes after the named axis (the same linearization
/// [`lower_collective_to_all_reduce`](crate::operations::collectives::parallel_reduce::lower_collective_to_all_reduce)
/// uses for replica groups) and `size` is the named axis's size. `partition_id` yields the executing device's linear
/// id; it is converted to `u64` and the coordinate arithmetic is emitted as scalar StableHLO divide/remainder. A
/// single-device axis collapses to the constant `0`.
pub(crate) fn lower_axis_index_to_coordinate<'b, 'c: 'b, 't: 'c>(
    operation: &AxisIndexOperation,
    collective_state: &CollectiveLoweringState,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let axis_name = operation.axis_name();
    let Some(shard_map) = collective_state.manual_shard_map() else {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "{AXIS_INDEX_OPERATION_NAME} for axis `{axis_name}` can only be lowered inside a \
                 {SHARD_MAP_OPERATION_NAME} manual region",
            ),
        }
        .into());
    };
    let axis_region = collective_state.manual_axis_region(axis_name);
    let mesh = axis_region.unwrap_or(shard_map).mesh();
    if axis_region.is_none() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "{AXIS_INDEX_OPERATION_NAME} for axis `{axis_name}` cannot lower inside this \
                 {SHARD_MAP_OPERATION_NAME} manual region because the region does not bind that axis as a manual mesh \
                 axis",
            ),
        }
        .into());
    }
    if operation.mesh().is_some_and(|declared| declared != mesh) {
        return Err(ProgramError::MalformedProgram(
            "`axis_index` mesh does not match its active manual binding".to_string(),
        )
        .into());
    }
    let axis_index = mesh.axis_index(axis_name).unwrap();
    let axis_size = mesh.axis_size(axis_name).unwrap();
    let axis_sizes: Vec<usize> = mesh.axes().iter().map(|axis| axis.size()).collect();
    let stride: usize = axis_sizes[axis_index + 1..].iter().product();

    let scalar_u64_type = lower_tensor_type(&ArrayType::scalar(DataType::U64), context, location)?;
    let partition_id = block.append_operation(stable_hlo::partition_id(location)?)?;
    let partition_id = partition_id.result(0).expect("stablehlo.partition_id should return one result").as_ref();
    let partition_id = block.append_operation(stable_hlo::convert(partition_id, scalar_u64_type, location)?)?;
    let partition_id = partition_id.result(0).expect("stablehlo.convert should return one result").as_ref();
    // `partition_id / stride` selects the coordinate's digit group; `% size` isolates this axis's coordinate. A unit
    // stride and a full-size axis make the respective step an identity, so they are skipped.
    let strided = if stride == 1 {
        partition_id
    } else {
        let stride_constant = lower_u64_scalar_constant(stride as u64, block, context, location)?;
        let divided = block.append_operation(stable_hlo::divide(partition_id, stride_constant, location)?)?;
        divided.result(0).expect("stablehlo.divide should return one result").as_ref()
    };
    if axis_size == mesh.device_count() {
        return Ok(strided);
    }
    let size_constant = lower_u64_scalar_constant(axis_size as u64, block, context, location)?;
    let coordinate = block.append_operation(stable_hlo::remainder(strided, size_constant, location)?)?;
    Ok(coordinate.result(0).expect("stablehlo.remainder should return one result").as_ref())
}

impl LowerableOperation for AxisIndexOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 0, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let collective_state = lowerer.collective_state.clone();
        if self.mesh().is_none() && collective_state.manual_axis_region(self.axis_name()).is_some() {
            return Err(ProgramError::MalformedProgram(
                "`axis_index` requires mesh metadata inside a manual region".to_string(),
            )
            .into());
        }
        let result = lower_axis_index_to_coordinate(
            self,
            &collective_state,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use ryft_core::{Context, Sharding, TracedShardMap, Value, trace_shard_map};

    use crate::XlaDomain;
    use crate::experimental::lowering::tests::{test_manual_mesh, test_vector_type};
    use crate::operations::sharding::shard_map::to_mlir_module;

    use super::*;

    #[test]
    fn test_axis_index_rejects_missing_checked_mesh_metadata() {
        let mesh = test_manual_mesh("x", 2);
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |input| input.domain().bind(AxisIndexOperation::new("x".to_string()), Vec::new(), &[]).unwrap().remove(0),
            test_vector_type(4),
            mesh.clone(),
            Sharding::replicated(mesh.clone(), 1),
            Sharding::replicated(mesh, 0),
        )
        .unwrap();
        assert!(matches!(
            to_mlir_module(&traced, "main"),
            Err(LoweringError::Tracing(ProgramError::MalformedProgram(message)))
                if message == "`axis_index` requires mesh metadata inside a manual region",
        ));
    }
}
