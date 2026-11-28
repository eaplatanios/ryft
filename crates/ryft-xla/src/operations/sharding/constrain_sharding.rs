use ryft_core::macros::check_count;
use ryft_core::{ArrayType, CONSTRAIN_SHARDING_OPERATION_NAME, ConstrainShardingOperation, ProgramError, Sharding};
use ryft_mlir::ValueRef;

use crate::lowering::sharding::validate_declared_pending_reduction;
use crate::{Error, LowerableOperation, Lowerer, OpenDimensions, ProgramLoweringOperation};

// TODO(eaplatanios): Review this module.

impl LowerableOperation for ConstrainShardingOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 1, ProgramError);
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        // Add the constraint's auto-axis placement to the tracked placement that its input already carries.
        let input_type = &input_types[0];
        let declared = self.lowered_sharding(input_type).map_err(ProgramError::from)?;
        let value = lower_declared_sharding_constraint(
            lowerer,
            input_values[0],
            input_type,
            &declared,
            CONSTRAIN_SHARDING_OPERATION_NAME,
        )?;
        Ok(vec![value])
    }
}

/// Constrains `value` to the placement `operation` declares, including its placement along auto axes.
/// The pending reduction carried by `value` comes from `value_type` even when a call overrides its placement.
///
/// # Errors
///
/// Returns an error if the declared transition cannot be lowered (refer to [`validate_declared_pending_reduction`])
/// or the sharding constraint cannot be emitted.
pub(in crate::operations) fn lower_declared_sharding_constraint<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
    lowerer: &Lowerer<'b, 'c, 't, O>,
    value: ValueRef<'b, 'c, 't>,
    value_type: &ArrayType,
    declared: &Sharding,
    operation: &str,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let context = lowerer.lowering_context();
    validate_declared_pending_reduction(
        declared,
        value_type.sharding(),
        value_type.data_type(),
        context.bound_manual_axes(),
        operation,
    )?;
    context.lower_sharding_constraint(declared, value, OpenDimensions::None, &mut lowerer.block(), lowerer.location())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ConstrainSharding, DataType, Dimension, MeshAxisType, Operation as CoreOperation, Shape, ShardMapBoundary,
        Sharding, ShardingDimension, Sin, UnreducedKind,
    };
    use ryft_mlir::{Block, Context as MlirContext, Location, Operation as MlirOperation, Value as MlirValue};

    use crate::lowering::contexts::ProgramLoweringContext;
    use crate::lowering::sharding::tests::{test_input_type, test_mesh};
    use crate::lowering::types::ArrayTypeLoweringExtension;
    use crate::operations::Operation;
    use crate::sessions::tests::two_device_mesh;
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{Array, CompiledFunction, Options, Session, compile_with_options};

    use super::*;

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
        let tensor_type = input_type.lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let mut lowerer: Lowerer<'_, '_, '_, Operation> =
            Lowerer::new(block.as_ref(), &context, location.as_ref(), ProgramLoweringContext::new());
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
        let tensor_type = input_type.lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let lowering_context = ProgramLoweringContext::new()
            .enter_manual_region(ShardMapBoundary::new(mesh, Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap());
        let mut lowerer: Lowerer<'_, '_, '_, Operation> =
            Lowerer::new(block.as_ref(), &context, location.as_ref(), lowering_context);
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
