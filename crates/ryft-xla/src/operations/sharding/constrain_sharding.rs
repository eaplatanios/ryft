use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, ConstrainShardingOperation, LogicalMesh, ProgramError, Sharding, ShardingDimension, ShardingError,
    UnreducedKind,
};
use ryft_mlir::dialects::shardy;
use ryft_mlir::{Block, BlockRef, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::sharding::ShardyTensorSharding;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, ToMlir};

/// Lowers the tracked placement `sharding` of a value's type as a tensor placement constraint after excluding mesh axes
/// bound by enclosing manual regions. A type does not fully describe the placement along
/// [`Auto`](ryft_core::MeshAxisType::Auto) axes, so the constraint leaves the compiler free to place the value along
/// them further (refer to [`ShardyTensorSharding`]).
pub(crate) fn lower_sharding_constraint<'b, 'c: 'b, 't: 'c>(
    input_values: &[ValueRef<'b, 'c, 't>],
    sharding: &Sharding,
    bound_manual_axes: &[String],
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    lower_constraint(input_values, sharding, &[], bound_manual_axes, block, location, true)
}

/// Lowers a tensor placement constraint that replicates a value with the tracked placement `sharding` along `axes`
/// and returns the constrained value. The constraint is that of [`lower_sharding_constraint`] for `sharding` with
/// those dimensions replicated, except that the dimensions are closed, so that the compiler cannot shard them along
/// auto axes either. Operations whose lowering XLA cannot distribute along a dimension (e.g., a reduction whose
/// combiner is not a single operation) use it to have every device hold all of the elements along that dimension.
pub(crate) fn lower_replication_constraint<'b, 'c: 'b, 't: 'c>(
    input_value: ValueRef<'b, 'c, 't>,
    sharding: &Sharding,
    axes: &[usize],
    bound_manual_axes: &[String],
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let mut dimensions = sharding.dimensions().to_vec();
    for axis in axes {
        dimensions[*axis] = ShardingDimension::Replicated;
    }
    let replicated = sharding
        .with_dimensions(dimensions)
        .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
    Ok(lower_constraint(&[input_value], &replicated, axes, bound_manual_axes, block, location, true)?[0])
}

impl LowerableOperation for ConstrainShardingOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = input_types;
        // The constraint is untracked, so the output type is the input type, and the emitted constraint must
        // carry the input's tracked placement alongside the constraint's own auto-axis placement. Unlike a tracked
        // placement, the constraint does describe the placement along auto axes, so its dimensions stay closed.
        check_count!("output", output_types, 1, ProgramError);
        let sharding = self.lowered_sharding(&output_types[0]).map_err(ProgramError::from)?;
        lower_constraint(
            input_values,
            &sharding,
            &[],
            lowerer.collective_state().bound_manual_axes(),
            &mut lowerer.block(),
            lowerer.location(),
            false,
        )
    }
}

/// Lowers a tensor placement constraint after excluding mesh axes bound by enclosing manual regions, whose dimensions
/// are open over auto axes when `open_over_auto_axes` is set, except for `closed_dimensions` (refer to
/// [`ShardyTensorSharding`]).
fn lower_constraint<'b, 'c: 'b, 't: 'c>(
    input_values: &[ValueRef<'b, 'c, 't>],
    sharding: &Sharding,
    closed_dimensions: &[usize],
    bound_manual_axes: &[String],
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
    open_over_auto_axes: bool,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    check_count!("input", input_values, 1, ProgramError);
    // Bound manual axes describe ownership outside this local tensor and cannot appear on constraints inside the
    // manual region. Restrict only the attribute's mesh inventory, because its symbol still names the module's full
    // mesh.
    let local_sharding;
    let sharding = if bound_manual_axes.is_empty() {
        sharding
    } else {
        let retain = |axis: &String| !bound_manual_axes.contains(axis);
        let projected = || -> Result<Sharding, ShardingError> {
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
            let projected_sharding = Sharding::new(mesh, dimensions)?
                .with_unreduced_axes(sharding.unreduced_axes().iter().filter(|axis| retain(axis)).cloned())?;
            // The pending reduction keeps its kind unless the projection drops all of its axes.
            let unreduced_kind = if projected_sharding.unreduced_axes().is_empty() {
                UnreducedKind::Sum
            } else {
                sharding.unreduced_kind()
            };
            projected_sharding
                .with_unreduced_kind(unreduced_kind)?
                .with_reduced_axes(sharding.reduced_axes().iter().filter(|axis| retain(axis)).cloned())?
                .with_varying_manual_axes(sharding.varying_manual_axes().iter().filter(|axis| retain(axis)).cloned())
        };
        local_sharding = projected().map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        &local_sharding
    };
    let sharding_attribute = ShardyTensorSharding::new(sharding, open_over_auto_axes)
        .with_closed_dimensions(closed_dimensions)
        .to_mlir(location)?;
    let operation =
        block.append_operation(shardy::sharding_constraint(input_values[0], sharding_attribute, location)?)?;
    Ok(vec![operation.result(0).unwrap().as_ref()])
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ConstrainSharding, DataType, Dimension, MeshAxis, MeshAxisType, Operation as CoreOperation, Shape,
        ShardMapBoundary, Sin,
    };
    use ryft_mlir::{Context as MlirContext, Location};

    use crate::lowering::CollectiveLoweringState;
    use crate::lowering::types::lower_tensor_type;
    use crate::operations::Operation;
    use crate::sessions::tests::two_device_mesh;
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{Array, CompiledFunction, Options, Session, compile_with_options};

    use super::*;

    /// Returns the mesh with the manual axis `m`, the explicit axis `x`, and the auto axis `y`, each of size two.
    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    /// Returns the `f32[4, 2]` type placed along `x` on its first dimension.
    fn test_input_type() -> ArrayType {
        let sharding =
            Sharding::new(test_mesh(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        ArrayType::new_static(DataType::F32, [4, 2]).with_sharding(sharding).unwrap()
    }

    #[test]
    fn test_lower_sharding_constraint() {
        // The constraints are rendered outside any function, so their detached operands print as unknown SSA values.
        let input_type = test_input_type();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(&input_type, &context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let outputs = lower_sharding_constraint(
            &[input],
            input_type.sharding().unwrap(),
            &[],
            &mut block.as_ref(),
            location.as_ref(),
        )
        .unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(outputs, vec![operations[0].result(0).unwrap().as_ref()]);
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{"x", ?}, {?}], replicated={"m"}> : tensor<4x2xf32>"#
            ],
        );
    }

    #[test]
    fn test_lower_replication_constraint() {
        let input_type = test_input_type();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(&input_type, &context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let output = lower_replication_constraint(
            input,
            input_type.sharding().unwrap(),
            &[0],
            &[],
            &mut block.as_ref(),
            location.as_ref(),
        )
        .unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations[0].result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{}, {?}], replicated={"m", "x"}> : tensor<4x2xf32>"#
            ],
        );
    }

    #[test]
    fn test_constrain_sharding_operation_lower() {
        let input_type = test_input_type();
        let mesh = test_mesh().with_axis_types(["y"], MeshAxisType::Auto).unwrap();
        let constraint =
            Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])]).unwrap();
        let operation = ConstrainShardingOperation::new(constraint);
        let output_types = operation.infer_output_types(std::slice::from_ref(&input_type), &[]).unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(&input_type, &context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let mut lowerer: Lowerer<'_, '_, '_, Operation> = Lowerer::new(block.as_ref(), &context, location.as_ref());
        let outputs = operation.lower(&mut lowerer, &[input], &[input_type], &output_types).unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(outputs, vec![operations[0].result(0).unwrap().as_ref()]);
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{"x"}, {"y"}], replicated={"m"}> : tensor<4x2xf32>"#
            ],
        );
    }

    #[test]
    fn test_constrain_sharding_operation_lower_omits_bound_manual_axes() {
        // Inside a manual region over `m`, the constraint names neither `m` nor its unreduced state. Dropping every
        // unreduced axis also drops the kind of the pending reduction, which falls back to the default sum kind.
        let mesh = test_mesh();
        let input_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["m"])
                    .unwrap()
                    .with_unreduced_kind(UnreducedKind::Max)
                    .unwrap(),
            )
            .unwrap();
        let operation = ConstrainShardingOperation::new(
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap(),
        );
        let output_types = operation.infer_output_types(std::slice::from_ref(&input_type), &[]).unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(&input_type, &context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let state = CollectiveLoweringState::new()
            .enter_manual_region(ShardMapBoundary::new(mesh, Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap());
        let mut lowerer: Lowerer<'_, '_, '_, Operation> =
            Lowerer::new(block.as_ref(), &context, location.as_ref()).with_collective_state(state);
        operation.lower(&mut lowerer, &[input], &[input_type], &output_types).unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{"x"}, {}]> : tensor<4x2xf32>"#],
        );
    }

    #[test]
    fn test_constrain_sharding_operation_lower_executes_on_cpu() {
        // `ConstrainSharding::constrain_sharding` works inside a function compiled with `compile` over an auto mesh
        // axis. The propagation hint is staged into the trace and lowers to `sdy.sharding_constraint`, and the output
        // array carries the sharding derived from its input on each device.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();

        let shape = Shape::new(vec![Dimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::F32, shape);
        let target_sharding = sharded.clone();

        // The user invokes `constrain_sharding` directly inside the staged closure, so it is compiled into the same
        // MLIR program as the rest of the function body. Types record no placement over auto mesh axes,
        // so the input placement is an `input_shardings` override.
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            move |x| {
                let constrained = x.constrain_sharding(&target_sharding).unwrap();
                constrained.sin().unwrap()
            },
            input_type.clone(),
            &engine,
            Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]),
        )
        .unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            Some(sharded.clone()),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, source).unwrap();
        assert_eq!(output.sharding(), &sharded);

        let mut observed: Vec<f32> = Vec::with_capacity(values.len());
        for device in client.addressable_devices().unwrap().iter().take(2) {
            let device_id = device.id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            observed.extend(values_from_bytes::<f32>(shard_bytes.as_slice()));
        }
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }
}
