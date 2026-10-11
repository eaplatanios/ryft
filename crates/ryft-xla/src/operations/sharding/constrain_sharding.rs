use ryft_core::macros::check_count;
use ryft_core::{ArrayType, ConstrainShardingOperation, LogicalMesh, ProgramError, Sharding, ShardingDimension};
use ryft_mlir::dialects::shardy;
use ryft_mlir::{Block, BlockRef, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, ToMlir, XlaLowerer};

/// Lowers a tensor placement constraint after excluding mesh axes bound by enclosing manual regions.
pub(crate) fn lower_sharding_constraint<'b, 'c: 'b, 't: 'c>(
    input_values: &[ValueRef<'b, 'c, 't>],
    sharding: &Sharding,
    bound_manual_axes: &[String],
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    // Bound manual axes describe ownership outside this local tensor and cannot appear on constraints inside the
    // manual region. Restrict only the attribute's mesh inventory; its symbol still names the module's full mesh.
    let local_sharding;
    let sharding = if bound_manual_axes.is_empty() {
        sharding
    } else {
        let retain = |axis: &String| !bound_manual_axes.contains(axis);
        let projected = || -> Result<Sharding, ryft_core::ShardingError> {
            let mesh = LogicalMesh::new(
                sharding
                    .mesh()
                    .axes()
                    .iter()
                    .filter(|axis| !bound_manual_axes.iter().any(|bound| bound == axis.name()))
                    .cloned()
                    .collect(),
            )?;
            let dimensions = sharding
                .dimensions()
                .iter()
                .map(|dimension| match dimension {
                    ShardingDimension::Sharded(axes) => {
                        let axes = axes.iter().filter(|axis| retain(axis)).cloned().collect::<Vec<_>>();
                        if axes.is_empty() { ShardingDimension::Replicated } else { ShardingDimension::Sharded(axes) }
                    }
                    dimension => dimension.clone(),
                })
                .collect();
            Sharding::new(mesh, dimensions)?
                .with_unreduced_axes(sharding.unreduced_axes().iter().filter(|axis| retain(axis)).cloned())?
                .with_reduced_axes(sharding.reduced_axes().iter().filter(|axis| retain(axis)).cloned())?
                .with_varying_manual_axes(sharding.varying_manual_axes().iter().filter(|axis| retain(axis)).cloned())
        };
        local_sharding = projected().map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        &local_sharding
    };
    let sharding_attribute = sharding.to_mlir(location)?;
    let operation =
        block.append_operation(shardy::sharding_constraint(input_values[0], sharding_attribute, location)?)?;
    Ok(vec![operation.result(0).expect("sdy.sharding_constraint should return one result").as_ref()])
}

impl LowerableOperation for ConstrainShardingOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        // The constraint is untracked, so the output type is the input type, and the emitted constraint must
        // carry the input's tracked placement alongside the constraint's own auto-axis placement.
        check_count!("output", output_types, 1, ProgramError);
        let sharding = self.lowered_sharding(&output_types[0]).map_err(ProgramError::from)?;
        lower_sharding_constraint(
            input_values,
            &sharding,
            &lowerer.collective_state.bound_manual_axes,
            &mut lowerer.block,
            lowerer.location,
        )
    }
}
