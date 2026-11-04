use std::collections::HashSet;

use ryft_core::{
    ArrayIrType, ArrayType, PARALLEL_ALL_GATHER_OPERATION_NAME, ParallelAllGatherOperation,
    ParallelAllGatherOutputVariance, Parameterized, ProgramError, SHARD_MAP_OPERATION_NAME, ShardMapBoundary,
    ShardMapError, ShardMapOperation, Sharding, ShardingDimension, Typed, TypedProgram,
};
use ryft_mlir::dialects::shardy;
use ryft_mlir::dialects::shardy::{
    DimensionShardingAttributeRef, ManualAxesAttributeRef, TensorShardingAttributeRef,
    TensorShardingPerValueAttributeRef,
};
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Type, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::regions::lower_program_outputs_with_inputs;
use crate::experimental::lowering::types::lower_tensor_type;
use crate::experimental::lowering::{
    CollectiveLoweringState, EffectTokens, current_or_new_token, token_threaded_effects,
};
use crate::experimental::ops::{XlaConstant, XlaProgram};
use crate::operations::control_flow::lowered_effect_classes;
use crate::sharding::SHARDY_MESH_SYMBOL_NAME;
use crate::{Error, LowerableOperation, LowerableOperationFamily, XlaLowerer};

/// Lowers one `sdy.manual_computation` operation, threading per-device effect tokens across its boundary. Assertions
/// remain local to an executable, and the only ordered I/O that reaches this function is per-device
/// (`DeviceOrderedIo`), because core shard-map boundary validation rejects bodies with the `OrderedIo` effect (refer to
/// [`ShardMapError::OrderedIoNotSupported`]).
pub(crate) fn lower_manual_computation<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    block: &mut BlockRef<'b, 'c, 't>,
    outer_inputs: &[ValueRef<'b, 'c, 't>],
    boundary: &ShardMapBoundary,
    program: &XlaProgram<ProgramInput, ProgramOutput, O>,
    local_input_types: &[ArrayType],
    global_output_types: &[ArrayType],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error>
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
{
    // Tracked variation adjoints would duplicate the conservative unchecked boundary's gradient reductions. Low-level
    // programs can bypass the closure constructor, so enforce ordinary active-axis types before lowering their bodies.
    if !boundary.check_manual_variation() {
        let body = program.entry_region_ref();
        for region_id in body.region_ids_in_closure() {
            let region = &body.arena()[region_id.index()];
            for atom in region.atoms() {
                let r#type = atom.r#type();
                if let Ok(array_type) = <&ArrayType>::try_from(r#type.as_ref())
                    && let Some(axis) = boundary.manual_axes().iter().find(|axis| {
                        array_type.sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(*axis))
                    })
                {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{SHARD_MAP_OPERATION_NAME}` with unchecked manual variation requires ordinary body types, \
                         but its body tracks variation over active manual axis `{axis}`",
                    ))
                    .into());
                }
            }
            for instruction in region.instructions() {
                if let Some(gather) = instruction.operation().projected_payload::<ParallelAllGatherOperation>()
                    && gather.output_variance() == ParallelAllGatherOutputVariance::Invariant
                    && boundary.manual_axes().iter().any(|axis| axis == gather.axis_name())
                {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{SHARD_MAP_OPERATION_NAME}` with unchecked manual variation requires varying output \
                         variance for `{PARALLEL_ALL_GATHER_OPERATION_NAME}` over active manual axis `{}`",
                        gather.axis_name(),
                    ))
                    .into());
                }
            }
        }
    }
    // Shardy forbids a nested manual computation from listing an axis that an enclosing one already made manual. The
    // closure entry points never construct such a nesting, and boundary validation rejects a nested map whose inputs
    // vary along such an axis, but a checked or transformed program can still nest a map over invariant inputs this
    // way, so it is rejected here instead of emitting an invalid manual computation.
    if let Some(axis_name) =
        boundary.manual_axes().iter().find(|axis| collective_state.bound_manual_axes.contains(*axis))
    {
        return Err(ProgramError::from(ShardMapError::AxisAlreadyManual { axis_name: axis_name.clone() }).into());
    }
    // For the same reason, an input or output sharding must not name such an axis (the closure entry points reject
    // these shardings too), so it is rejected here rather than silently dropped from the Shardy shardings.
    for (value_kind, shardings) in [("input", boundary.input_shardings()), ("output", boundary.output_shardings())] {
        for (value_index, sharding) in shardings.iter().enumerate() {
            let axis_name = sharding
                .dimensions()
                .iter()
                .flat_map(|dimension| match dimension {
                    ShardingDimension::Sharded(axis_names) => axis_names.as_slice(),
                    ShardingDimension::Replicated | ShardingDimension::Unconstrained => &[],
                })
                .chain(sharding.unreduced_axes())
                .chain(sharding.reduced_axes())
                .find(|axis_name| collective_state.bound_manual_axes.contains(*axis_name));
            if let Some(axis_name) = axis_name {
                let error = ShardMapError::ShardingNamesEnclosingManualAxis {
                    value_kind,
                    value_index,
                    axis_name: axis_name.clone(),
                };
                return Err(ProgramError::from(error).into());
            }
        }
    }

    // `OrderedIo` bodies never reach this point: shard-map boundary validation rejects them when a body is traced or
    // attached (refer to `ShardMapError::OrderedIoNotSupported`), so only `DeviceOrderedIo` and unordered effects are
    // threaded through the manual computation.
    let threaded_effects = token_threaded_effects(lowered_effect_classes(program)?).collect::<Vec<_>>();
    let mut local_input_tensor_types = local_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location).map(|r#type| r#type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location).map(|r#type| r#type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut manual_inputs = outer_inputs.to_vec();
    let mut input_shardings =
        to_shardy_in_shardings(boundary, &collective_state.bound_manual_axes, context)?.shardings();
    let mut output_shardings =
        to_shardy_out_shardings(boundary, &collective_state.bound_manual_axes, context)?.shardings();
    if !threaded_effects.is_empty() {
        let token_type = context.stable_hlo_token_type()?.as_ref();
        // Tokens have rank zero and no replicated/unreduced axes. Each device carries its own token, without
        // imposing a total order between devices or attempting to shard a numeric array dimension.
        let token_sharding = context.shardy_tensor_sharding(
            context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME),
            &[],
            &[],
            &[],
            shardy::ReductionOperation::Sum,
        )?;
        for &effect in &threaded_effects {
            manual_inputs.push(current_or_new_token(effect, effect_tokens, block, location)?);
            local_input_tensor_types.push(token_type);
            global_output_tensor_types.push(token_type);
            input_shardings.push(token_sharding);
            output_shardings.push(token_sharding);
        }
    }
    let mut body_region = context.region();
    let body_block =
        context.block(local_input_tensor_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>().as_slice());
    {
        let mut body_block_ref = body_block.as_ref();
        // Shard-local callees always inline. The fresh-root body has no enclosing capture namespace.
        let body_collective_state = collective_state.enter_manual_region(boundary.clone());
        let mut body_effect_tokens = EffectTokens::default();
        for (index, &effect) in threaded_effects.iter().enumerate() {
            body_effect_tokens.set(effect, body_block.argument(local_input_types.len() + index).unwrap().as_ref());
        }
        let body_inputs = (0..local_input_types.len())
            .map(|index| body_block.argument(index).unwrap().as_ref())
            .collect::<Vec<_>>();
        let mut body_outputs = lower_program_outputs_with_inputs(
            program,
            &[],
            &body_inputs,
            &mut body_block_ref,
            context,
            location,
            None,
            &body_collective_state,
            &mut body_effect_tokens,
        )?;
        body_outputs.extend(threaded_effects.iter().map(|&effect| body_effect_tokens.get(effect).unwrap()));
        body_block_ref.append_operation(shardy::r#return(body_outputs.as_slice(), location)?)?;
    }
    body_region.append_block(body_block)?;
    let manual_computation = block.append_operation(shardy::manual_computation(
        manual_inputs.as_slice(),
        global_output_tensor_types.as_slice(),
        context.shardy_tensor_sharding_per_value(&input_shardings)?,
        context.shardy_tensor_sharding_per_value(&output_shardings)?,
        to_shardy_manual_axes(boundary, context)?,
        body_region,
        location,
    )?)?;
    for (index, &effect) in threaded_effects.iter().enumerate() {
        effect_tokens.set(effect, manual_computation.result(global_output_types.len() + index).unwrap().as_ref());
    }
    manual_computation
        .results()
        .take(global_output_types.len())
        .map(|result| result.map(|result| result.as_ref()).map_err(Error::from))
        .collect()
}

/// Builds the typed Shardy `in_shardings` attribute of `shard_map`, excluding axes already bound by enclosing manual
/// regions.
fn to_shardy_in_shardings<'c, 't>(
    boundary: &ShardMapBoundary,
    enclosing_manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<TensorShardingPerValueAttributeRef<'c, 't>, Error> {
    shardy_tensor_sharding_per_value(boundary.input_shardings(), boundary.manual_axes(), enclosing_manual_axes, context)
}

/// Builds the typed Shardy `out_shardings` attribute of `shard_map`, excluding axes already bound by enclosing manual
/// regions.
fn to_shardy_out_shardings<'c, 't>(
    boundary: &ShardMapBoundary,
    enclosing_manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<TensorShardingPerValueAttributeRef<'c, 't>, Error> {
    shardy_tensor_sharding_per_value(
        boundary.output_shardings(),
        boundary.manual_axes(),
        enclosing_manual_axes,
        context,
    )
}

/// Builds the typed Shardy `manual_axes` attribute that `shard_map` attaches to `sdy.manual_computation`.
fn to_shardy_manual_axes<'c, 't>(
    boundary: &ShardMapBoundary,
    context: &'c MlirContext<'t>,
) -> Result<ManualAxesAttributeRef<'c, 't>, Error> {
    Ok(context.shardy_manual_axes(boundary.manual_axes())?)
}

/// Builds the typed Shardy per-value tensor sharding attribute of a `sdy.manual_computation` boundary, with one tensor
/// sharding per entry of `shardings` (refer to [`manual_computation_tensor_sharding`]).
fn shardy_tensor_sharding_per_value<'c, 't>(
    shardings: &[Sharding],
    manual_axes: &[String],
    enclosing_manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<TensorShardingPerValueAttributeRef<'c, 't>, Error> {
    let shardings = shardings
        .iter()
        .map(|sharding| manual_computation_tensor_sharding(sharding, manual_axes, enclosing_manual_axes, context))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(context.shardy_tensor_sharding_per_value(shardings.as_slice())?)
}

/// Builds the typed Shardy tensor sharding of one `sdy.manual_computation` boundary value placed by `sharding`. The
/// axes of enclosing manual regions are omitted from the replicated axes, because they are already bound outside this
/// manual computation, and the dimension shardings are built by [`manual_computation_dimension_shardings`]. `sharding`
/// must not name such an axis in its dimension shardings or reduction state, which [`lower_manual_computation`] rejects
/// with [`ShardMapError::ShardingNamesEnclosingManualAxis`] before building these attributes.
fn manual_computation_tensor_sharding<'c, 't>(
    sharding: &Sharding,
    manual_axes: &[String],
    enclosing_manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<TensorShardingAttributeRef<'c, 't>, Error> {
    let mesh_symbol_ref = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
    let dimension_shardings =
        manual_computation_dimension_shardings(sharding, manual_axes, enclosing_manual_axes, context)?;
    let replicated_axis_names = sharding.replicated_axes();
    let replicated_axes = replicated_axis_names
        .iter()
        .filter(|axis_name| !enclosing_manual_axes.iter().any(|axis| axis.as_str() == **axis_name))
        .map(|axis_name| context.shardy_axis_ref(*axis_name, None))
        .collect::<Result<Vec<_>, _>>()?;
    let unreduced_axes = sharding
        .unreduced_axes()
        .iter()
        .map(|axis_name| context.shardy_axis_ref(axis_name.as_str(), None))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(context.shardy_tensor_sharding(
        mesh_symbol_ref,
        dimension_shardings.as_slice(),
        replicated_axes.as_slice(),
        unreduced_axes.as_slice(),
        shardy::ReductionOperation::Sum,
    )?)
}

/// Builds the typed Shardy dimension shardings of one `sdy.manual_computation` boundary value placed by `sharding`. A
/// free axis is a mesh axis that is neither an active manual axis nor bound by an enclosing manual region. A dimension
/// is open (i.e., the compiler may further shard it along free axes) when it is unconstrained, when it is sharded over
/// a free axis, or when `sharding` leaves some free axis unused, and it is closed otherwise. `sharding` must not shard
/// a dimension along an axis of an enclosing manual region (refer to [`manual_computation_tensor_sharding`]).
fn manual_computation_dimension_shardings<'c, 't>(
    sharding: &Sharding,
    manual_axes: &[String],
    enclosing_manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<Vec<DimensionShardingAttributeRef<'c, 't>>, Error> {
    let manual_axis_names = manual_axes.iter().map(String::as_str).collect::<HashSet<_>>();
    let free_axis_names = sharding
        .mesh()
        .axes()
        .iter()
        .filter_map(|axis| {
            (!manual_axis_names.contains(axis.name()) && !enclosing_manual_axes.iter().any(|name| name == axis.name()))
                .then_some(axis.name())
        })
        .collect::<HashSet<_>>();
    let mut used_axes = HashSet::new();
    for partition_dimension in sharding.dimensions() {
        if let ShardingDimension::Sharded(axis_names) = partition_dimension {
            used_axes.extend(axis_names.iter().map(String::as_str));
        }
    }
    used_axes.extend(sharding.unreduced_axes().iter().map(String::as_str));
    used_axes.extend(sharding.reduced_axes().iter().map(String::as_str));
    let has_unused_free_axes = free_axis_names.iter().any(|axis_name| !used_axes.contains(axis_name));

    sharding
        .dimensions()
        .iter()
        .map(|partition_dimension| match partition_dimension {
            ShardingDimension::Replicated => context.shardy_dimension_sharding([], !has_unused_free_axes, None),
            ShardingDimension::Sharded(axis_names) => {
                let axes = axis_names
                    .iter()
                    .map(|axis_name| context.shardy_axis_ref(axis_name.as_str(), None))
                    .collect::<Result<Vec<_>, _>>()?;
                let contains_free_axis =
                    axis_names.iter().any(|axis_name| free_axis_names.contains(axis_name.as_str()));
                context.shardy_dimension_sharding(axes, !(contains_free_axis || has_unused_free_axes), None)
            }
            ShardingDimension::Unconstrained => context.shardy_dimension_sharding([], false, None),
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Error::from)
}

impl LowerableOperation for ShardMapOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let [body] = lowerer.regions.as_slice() else {
            return Err(Error::from(ProgramError::MalformedProgram(format!(
                "{} expected 1 attached region but got {}",
                SHARD_MAP_OPERATION_NAME,
                lowerer.regions.len()
            ))));
        };
        let simplified_body = body.simplified()?;
        let local_input_types = simplified_body
            .input_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        // References are discharged before lowering, so every global boundary type is an array by the time the manual
        // computation is lowered.
        let global_output_types = self
            .global_output_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        lower_manual_computation(
            &mut lowerer.block,
            input_values,
            self.boundary(),
            &simplified_body,
            local_input_types.as_slice(),
            global_output_types.as_slice(),
            lowerer.context,
            lowerer.location,
            &lowerer.collective_state,
            &mut lowerer.effect_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AxisIndexOperation, Context, ConvertElementTypeOperation, CustomCallOperation, DataType, Device, DeviceMesh,
        Dimension, DomainTracingContext, Dot, DotDimensionNumbers, EffectClass, LogicalMesh, MeshAxis, MeshAxisType,
        Placeholder, Shape, ShardMapOptions, Typed, ValueProjection, shard_map,
    };
    use ryft_mlir::Block;
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program as PjrtProgram, load_cpu_plugin};

    use crate::experimental::XlaArrayTracer;
    use crate::experimental::domains::XlaTracer;
    use crate::experimental::lowering::tests::{
        lower_replicated_shard_map_body, shard_map_module, static_sharded_array_type, test_manual_mesh, test_sharding,
        test_spmd_compilation_options, test_vector_type,
    };
    use crate::experimental::ops::XlaProgramBuilder as ArrayIrXlaProgramBuilder;
    use crate::operations::debugging::PRINT_CUSTOM_CALL_TARGET;
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{FromPjrt, ToMlir, XlaArray, XlaDomain, XlaSession};

    use super::*;

    /// Four-device mesh with the manual axes `x` and `y`.
    fn test_logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    /// Binds a `shard_map` with `boundary` over `x` in the context of `x`, whose body doubles its local input. The map
    /// is built from its boundary through [`ShardMapOperation::from_program`], so it bypasses the checks that the
    /// frontend applies against the manual axes of enclosing manual regions (e.g., as a transformation could).
    fn bind_doubling_shard_map(x: XlaArrayTracer, boundary: ShardMapBoundary) -> Result<XlaArrayTracer, ProgramError> {
        let global_type = boundary.global_input_type(0, &x.r#type())?;
        let local_type = boundary.local_input_type(0, &global_type)?;
        let (_, body) = DomainTracingContext::<XlaDomain<'static>>::trace(
            |inputs: Vec<XlaTracer<'static>>| {
                let y = ValueProjection::<ArrayType>::into_projected(inputs[0].clone()).map_err(ProgramError::from)?;
                Ok(vec![(y.clone() + y).into_value()])
            },
            vec![ArrayIrType::Array(local_type)],
        )?;
        let operation = ShardMapOperation::from_program(&body, vec![global_type.into()], boundary)?;
        let output = x.value().context().bind(operation, vec![body], &[x.value().clone()])?.remove(0);
        Ok(ValueProjection::<ArrayType>::into_projected(output)?)
    }

    /// Eight-device mesh with a manual `data` axis and an auto `model` axis.
    fn test_logical_mesh_data_model() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("data", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("model", 4, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    /// Eight-device mesh with a manual `data` axis and an explicit `model` axis.
    fn test_logical_mesh_data_model_explicit() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("data", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("model", 4, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn test_device_ordered_effects_cross_manual_computation_boundary() {
        let vector_type = test_vector_type(4);
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(vector_type.into());
        for label in ["first", "second"] {
            builder
                .add_instruction(
                    CustomCallOperation::new(PRINT_CUSTOM_CALL_TARGET, Vec::new())
                        .with_attribute("label", label)
                        .with_effect_class(EffectClass::DeviceOrderedIo),
                    Vec::new(),
                    vec![input],
                    None,
                )
                .unwrap();
        }
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = lower_replicated_shard_map_body(program, 2).unwrap();
        assert!(module.contains("sdy.mesh @mesh = <[\"x\"=2]>"));
        assert_eq!(module.matches("stablehlo.after_all").count(), 1);
        assert_eq!(module.matches("stablehlo.custom_call @ryft.print").count(), 2);
        assert!(module.contains("in_shardings=[<@mesh, [{}], replicated={\"x\"}>, <@mesh, []>]"));
        assert!(module.contains("out_shardings=[<@mesh, [{}], replicated={\"x\"}>, <@mesh, []>]"));
        assert!(module.contains("sdy.return %arg1, %3 : tensor<4xf32>, !stablehlo.token"), "{module}");
    }

    #[test]
    fn test_device_ordered_effects_execute_on_two_cpu_devices() {
        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
        use ryft_pjrt::{
            BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
            load_cpu_plugin,
        };

        use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};
        use crate::tests::{values_from_bytes, values_to_bytes};

        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(test_vector_type(4).into());
        let device_index = builder
            .add_instruction(
                AxisIndexOperation::new("x".to_string()).with_mesh(test_manual_mesh("x", 2)),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap()[0];
        let device_index = builder
            .add_instruction(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::F64, false),
                Vec::new(),
                vec![device_index],
                None,
            )
            .unwrap()[0];
        for label in ["first", "second"] {
            builder
                .add_instruction(
                    CustomCallOperation::new(PRINT_CUSTOM_CALL_TARGET, Vec::new())
                        .with_attribute("label", label)
                        .with_effect_class(EffectClass::DeviceOrderedIo),
                    Vec::new(),
                    vec![device_index],
                    None,
                )
                .unwrap();
        }
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = lower_replicated_shard_map_body(program, 2).unwrap();
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        ensure_print_handler_registered(&client).unwrap();
        let options = CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 2,
                use_spmd_partitioning: true,
                use_shardy_partitioner: true,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            individually_defined_output_indices: Vec::new(),
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
            compiler_variant: None,
        };
        let executable = client.compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &options).unwrap();
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = values_to_bytes(&values);
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| {
                vec![ExecutionInput {
                    buffer: Arc::new(
                        client.buffer(bytes.as_slice(), BufferType::F32, &[4], None, device.clone(), None).unwrap(),
                    ),
                    donatable: false,
                }]
            })
            .collect::<Vec<_>>();
        let (outputs, lines) = with_captured_prints(|| {
            executable
                .execute(
                    inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                    Vec::new(),
                    0,
                    None,
                    Some(file!()),
                    None,
                    None,
                    &[],
                )
                .unwrap()
                .block_until_ready()
                .unwrap()
        });
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            assert_eq!(output.outputs.len(), 1);
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(&bytes), values);
        }
        assert_eq!(lines.len(), 4);
        for device in [0, 1] {
            let first = lines.iter().position(|line| line == &format!("first: {device}.0")).unwrap();
            let second = lines.iter().position(|line| line == &format!("second: {device}.0")).unwrap();
            assert!(first < second);
        }
    }

    #[test]
    #[allow(deprecated)] // Inspect physical result storage independently of the lowered IR.
    fn test_to_mlir_module_for_program_preserves_cast_layouts() {
        use ryft_core::{Cast, Layout, TiledLayout, Typed};
        use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput};

        use crate::ToPjrt;
        use crate::tests::execution_client;

        let mesh = test_manual_mesh("x", 1);
        let sharding = Sharding::replicated(mesh.clone(), 2);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let input_type = ArrayType::new_static(DataType::F32, [2, 3]).with_layout(layout.clone());
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, input| {
                let input = input.cast(input.r#type().into_owned().with_layout(None), &[])?;
                input.cast(input.r#type().into_owned().with_layout(layout.clone()), &[])
            },
            input_type,
            mesh,
            sharding.clone(),
            sharding,
            ShardMapOptions::new(),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=1]>
                  func.func @main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\", \
                      sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={\"x\"}>}) \
                      -> (tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\", \
                      sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={\"x\"}>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}], replicated={\"x\"}>] \
                        out_shardings=[<@mesh, [{}, {}], replicated={\"x\"}>] \
                        manual_axes={\"x\"} (%arg1: tensor<2x3xf32>) {
                      %1 = stablehlo.custom_call @LayoutConstraint(%arg1) {\
                          operand_layouts = [dense<[0, 1]> : tensor<2xindex>], \
                          result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<2x3xf32>) -> tensor<2x3xf32>
                      sdy.return %1 : tensor<2x3xf32>
                    } : (tensor<2x3xf32>) -> tensor<2x3xf32>
                    return %0 : tensor<2x3xf32>
                  }
                }
            "},
        );
        let client = execution_client();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(1))
            .unwrap();
        let device = client.addressable_devices().unwrap().remove(0);
        let values = vec![1f32, 2., 3., 4., 5., 6.];
        let buffer = client
            .buffer(
                values_to_bytes(&values).as_slice(),
                BufferType::F32,
                [2, 3],
                None,
                device,
                Some(layout.to_pjrt().unwrap()),
            )
            .unwrap();
        let inputs = vec![ExecutionInput { buffer: Arc::new(buffer), donatable: false }];
        let outputs = executable
            .execute(
                vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                vec![],
                0,
                None,
                None,
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        let output = &outputs[0].outputs[0];
        assert_eq!(output.layout().unwrap(), layout.to_pjrt().unwrap(),);
        let bytes = output.copy_to_host(Some(ryft_pjrt::Layout::dense_major_to_minor(2))).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(&bytes), values);
    }

    #[test]
    fn test_to_mlir_module_for_program_renders_auto_axes_as_open_dimension_shardings() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(16)]));
        let mesh = test_logical_mesh_data_model();
        let input_sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], vec![]);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| Ok(x.clone() + x),
            global_input_type.clone(),
            mesh.clone(),
            input_sharding.clone(),
            input_sharding,
            ShardMapOptions::new(),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["data"=2, "model"=4]>
                  func.func @main(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"data", ?}]>] out_shardings=[<@mesh, [{"data", ?}]>] manual_axes={"data"} (%arg1: tensor<8xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<8xf32>
                      sdy.return %1 : tensor<8xf32>
                    } : (tensor<16xf32>) -> tensor<16xf32>
                    return %0 : tensor<16xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_renders_nested_shard_maps() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let inner_mesh = mesh.clone();
        let outer_sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let inner_sharding = test_sharding(&inner_mesh, vec![ShardingDimension::sharded(["y"])], vec![]);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            {
                let inner_mesh = inner_mesh.clone();
                let inner_sharding = inner_sharding.clone();
                move |_, x: XlaArrayTracer| {
                    let nested: XlaArrayTracer = shard_map(
                        |y: XlaArrayTracer| Ok(y.clone() + y),
                        x.clone(),
                        inner_mesh.clone(),
                        inner_sharding.clone(),
                        inner_sharding.clone(),
                        ShardMapOptions::new().with_manual_axes(["y"]),
                    )?;
                    Ok(nested + x)
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)])),
            mesh,
            outer_sharding.clone(),
            outer_sharding,
            ShardMapOptions::new().with_manual_axes(["x"]),
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}], replicated={"y"}>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}], replicated={"y"}>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", ?}], replicated={"y"}>] out_shardings=[<@mesh, [{"x", ?}], replicated={"y"}>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
                      %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={"y"} (%arg2: tensor<2xf32>) {
                        %3 = stablehlo.add %arg2, %arg2 : tensor<2xf32>
                        sdy.return %3 : tensor<2xf32>
                      } : (tensor<4xf32>) -> tensor<4xf32>
                      %2 = stablehlo.add %1, %arg1 : tensor<4xf32>
                      sdy.return %2 : tensor<4xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_rejects_nested_manual_computations_over_already_manual_axes() {
        // A nested map built from its boundary (e.g., as a transformation could) bypasses the checks of the frontend
        // and makes the axis `x` of the enclosing manual region manual again. The nested input is invariant along `x`,
        // so boundary validation cannot detect the nesting, and lowering rejects the nested manual computation instead
        // of emitting an invalid module.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![]);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x: XlaArrayTracer| {
                let boundary = ShardMapBoundary::new(
                    mesh.clone(),
                    vec![replicated.clone()],
                    vec![replicated.clone()],
                    vec!["x".into()],
                )?;
                bind_doubling_shard_map(x, boundary)
            },
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            replicated.clone(),
            replicated.clone(),
            ShardMapOptions::new(),
        );
        let expected = ShardMapError::AxisAlreadyManual { axis_name: "x".to_string() };
        assert!(matches!(
            module,
            Err(Error::Core(ryft_core::Error::Program(error)))
                if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
    }

    #[test]
    fn test_to_mlir_module_for_program_rejects_shardings_naming_already_manual_axes() {
        // A nested map over `y` built from its boundary (e.g., as a transformation could) bypasses the checks of the
        // frontend, so its input and output shardings can shard along the axis `x` of the enclosing manual region.
        // Shardy cannot represent those shardings inside that region, so lowering rejects them instead of silently
        // dropping `x` from them.
        let mesh = test_logical_mesh_2x2();
        let replicated = test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![]);
        let sharded = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x: XlaArrayTracer| {
                let boundary = ShardMapBoundary::new(
                    mesh.clone(),
                    vec![sharded.clone()],
                    vec![sharded.clone()],
                    vec!["y".into()],
                )?;
                bind_doubling_shard_map(x, boundary)
            },
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            replicated.clone(),
            replicated.clone(),
            ShardMapOptions::new().with_manual_axes(["x"]),
        );
        let expected = ShardMapError::ShardingNamesEnclosingManualAxis {
            value_kind: "input",
            value_index: 0,
            axis_name: "x".to_string(),
        };
        assert!(matches!(
            module,
            Err(Error::Core(ryft_core::Error::Program(error)))
                if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
    }

    #[test]
    fn test_to_mlir_module_for_program_executes_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .expect("failed to create 4-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();

        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let mlir_program = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| Ok(x.clone() + x),
            global_input_type,
            device_mesh.logical_mesh().clone(),
            sharding.clone(),
            sharding.clone(),
            ShardMapOptions::new(),
        )
        .unwrap();

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = PjrtProgram::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
        let expected_values_by_device = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                (device.id().unwrap(), [device_index as f32 * 4.0 + 2.0, device_index as f32 * 4.0 + 4.0])
            })
            .collect::<HashMap<_, _>>();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();

        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(
                execute_arguments.as_execution_device_inputs(),
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, *expected_values_by_device.get(&device_id).unwrap());
        }
    }

    #[test]
    fn test_to_mlir_module_for_program_unchecked_output_replication_executes_on_cpu() {
        use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput};

        let mesh = test_logical_mesh_2x2();
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, input| Ok(input),
            ArrayType::new_static(DataType::F32, [8]),
            mesh,
            input_sharding,
            output_sharding,
            ShardMapOptions::new().with_check_output_replication(false),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\", \"y\"}]>}) \
                      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"y\"}], replicated={\"x\"}>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\", \"y\"}]>] \
                        out_shardings=[<@mesh, [{\"y\"}], replicated={\"x\"}>] manual_axes={\"x\", \"y\"} \
                        (%arg1: tensor<2xf32>) {
                      sdy.return %arg1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, device)| {
                // Coordinates along `x` contain identical copies, while coordinates along `y` contain distinct tiles.
                let values = if index % 2 == 0 { [1f32, 2.] } else { [3f32, 4.] };
                vec![ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(values_to_bytes(&values).as_slice(), BufferType::F32, [2], None, device, None)
                            .unwrap(),
                    ),
                    donatable: false,
                }]
            })
            .collect::<Vec<_>>();
        let outputs = executable
            .execute(
                inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                Vec::new(),
                0,
                None,
                None,
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 4);
        for (index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let expected = if index % 2 == 0 { vec![1f32, 2.] } else { vec![3f32, 4.] };
            assert_eq!(values_from_bytes::<f32>(&bytes), expected);
        }
    }

    #[test]
    fn test_to_mlir_module_for_program_transposes_unchecked_output_replication_on_cpu() {
        use ryft_core::CotangentDestinationKind;
        use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput};

        use crate::experimental::lowering::to_mlir_module_for_program;

        let mesh = test_logical_mesh_2x2();
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let global_input_type =
            ArrayType::new_static(DataType::F32, [8]).with_sharding(input_sharding.clone()).unwrap();
        let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
            |inputs: Vec<XlaTracer<'static>>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    |input| Ok(input),
                    input,
                    mesh,
                    input_sharding.clone(),
                    output_sharding.clone(),
                    ShardMapOptions::new().with_check_output_replication(false),
                )?;
                Ok(vec![output.into_value()])
            },
            vec![ArrayIrType::Array(global_input_type)],
        )
        .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        let input_types = pullback
            .input_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let output_types = pullback
            .output_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let module = to_mlir_module_for_program(
            &pullback,
            &[],
            &input_types,
            &output_types,
            "main",
            Some(&[output_sharding]),
            Some(&[input_sharding]),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = \
                      #sdy.sharding<@mesh, [{\"y\"}], replicated={\"x\"}>}) \
                      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\", \"y\"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"y\"}], replicated={\"x\"}>] \
                        out_shardings=[<@mesh, [{\"x\", \"y\"}]>] manual_axes={\"x\", \"y\"} \
                        (%arg1: tensor<2xf32>) {
                      %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
                      %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
                      %2 = stablehlo.divide %arg1, %1 : tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| {
                vec![ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(values_to_bytes(&[1f32, 1.]).as_slice(), BufferType::F32, [2], None, device, None)
                            .unwrap(),
                    ),
                    donatable: false,
                }]
            })
            .collect::<Vec<_>>();
        let outputs = executable
            .execute(
                inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                Vec::new(),
                0,
                None,
                None,
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 4);
        for output in outputs {
            assert_eq!(output.outputs.len(), 1);
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(&bytes), vec![0.5, 0.5]);
        }
    }

    #[test]
    fn test_to_mlir_module_for_program_without_manual_variation_transposes_replicated_collectives_on_cpu() {
        use ryft_core::{CotangentDestinationKind, ParallelReduce, ReductionKind};
        use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput};

        use crate::experimental::lowering::to_mlir_module_for_program;

        // The primal is x -> 4*x. Its reverse rule divides each replicated seed by four, transposes the physical
        // collective, and sums the replicated input contributions. Both all-reduces must survive lowering.
        let mesh = test_manual_mesh("x", 4);
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let global_type = ArrayType::scalar(DataType::F32).with_sharding(replicated.clone()).unwrap();
        let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
            |inputs: Vec<XlaTracer<'static>>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    |input: XlaArrayTracer| input.parallel_reduce(ReductionKind::Sum, "x"),
                    input,
                    mesh,
                    replicated.clone(),
                    replicated.clone(),
                    ShardMapOptions::new().with_check_manual_variation(false),
                )?;
                Ok(vec![output.into_value()])
            },
            vec![ArrayIrType::Array(global_type)],
        )
        .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        let input_types = pullback
            .input_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let output_types = pullback
            .output_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let module = to_mlir_module_for_program(
            &pullback,
            &[],
            &input_types,
            &output_types,
            "main",
            Some(&[replicated.clone()]),
            Some(&[replicated]),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=4]>
                  func.func @main(%arg0: tensor<f32> {sdy.sharding = \
                      #sdy.sharding<@mesh, [], replicated={\"x\"}>}) -> (tensor<f32> {sdy.sharding = \
                      #sdy.sharding<@mesh, [], replicated={\"x\"}>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [], replicated={\"x\"}>] \
                        out_shardings=[<@mesh, [], replicated={\"x\"}>] manual_axes={\"x\"} (%arg1: tensor<f32>) {
                      %cst = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                      %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<f32>
                      %2 = stablehlo.divide %arg1, %1 : tensor<f32>
                      %3 = \"stablehlo.all_reduce\"(%2) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = \
                          dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %5 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %5 : tensor<f32>
                      }) : (tensor<f32>) -> tensor<f32>
                      %4 = \"stablehlo.all_reduce\"(%3) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = \
                          dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %5 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %5 : tensor<f32>
                      }) : (tensor<f32>) -> tensor<f32>
                      sdy.return %4 : tensor<f32>
                    } : (tensor<f32>) -> tensor<f32>
                    return %0 : tensor<f32>
                  }
                }
            "},
        );

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| {
                vec![ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(values_to_bytes(&[3f32]).as_slice(), BufferType::F32, [], None, device, None)
                            .unwrap(),
                    ),
                    donatable: false,
                }]
            })
            .collect::<Vec<_>>();
        let outputs = executable
            .execute(
                inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                Vec::new(),
                0,
                None,
                None,
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 4);
        for output in outputs {
            assert_eq!(output.outputs.len(), 1);
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(&bytes), vec![12.0]);
        }
    }

    #[test]
    fn test_to_mlir_module_for_program_without_manual_variation_executes_unreduced_outputs_on_cpu() {
        use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput};

        // Each device multiplies its shard of `x = [1, 2, 3, 4]` by the whole reduced input `r = [5, 6]` and returns
        // the product as its summand of the unreduced output, so the devices hold `[5, 12]` and `[15, 24]`, which sum
        // to the logical output.
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let module = shard_map_module::<_, (ArrayType, ArrayType), ArrayType>(
            |_, (x, r)| Ok(x * r),
            (ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::F32, [2])),
            mesh,
            (sharded, replicated.clone().with_reduced_axes(["x"]).unwrap()),
            replicated.with_unreduced_axes(["x"]).unwrap(),
            ShardMapOptions::new().with_check_manual_variation(false),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}, \
                      %arg1: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<2xf32> \
                      {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={\"x\"}>}) {
                    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{\"x\"}]>, <@mesh, [{}]>] \
                        out_shardings=[<@mesh, [{}], unreduced={\"x\"}>] manual_axes={\"x\"} \
                        (%arg2: tensor<2xf32>, %arg3: tensor<2xf32>) {
                      %1 = stablehlo.multiply %arg2, %arg3 : tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<4xf32>, tensor<2xf32>) -> tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "},
        );

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .into_iter()
            .zip([[1f32, 2.], [3., 4.]])
            .map(|(device, x)| {
                [x.as_slice(), &[5f32, 6.]]
                    .into_iter()
                    .map(|values| ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(
                                    values_to_bytes(values).as_slice(),
                                    BufferType::F32,
                                    [2],
                                    None,
                                    device.clone(),
                                    None,
                                )
                                .unwrap(),
                        ),
                        donatable: false,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let outputs = executable
            .execute(
                inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                Vec::new(),
                0,
                None,
                None,
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        let summands = outputs
            .into_iter()
            .map(|output| values_from_bytes::<f32>(&output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(summands, vec![vec![5.0, 12.0], vec![15.0, 24.0]]);
    }

    #[test]
    fn test_to_mlir_module_for_program_without_manual_variation_passes_unreduced_summands_between_maps_on_cpu() {
        use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput};

        use crate::experimental::tracing::{TracedXlaProgram, trace_with_context};

        // The first map returns each device's shard of `x = [1, 2]` as its summand of an unreduced output, and the
        // second map squares each summand of its unreduced input. The summands survive between the maps, so the
        // devices return `[1]` and `[4]`, whose sum is `5`, while the canonical decomposition of their sum (`[3]` on
        // the first device and `[0]` on the second) would give `9`.
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let options = ShardMapOptions::new().with_check_manual_variation(false);
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace_with_context(
            |_, x: XlaArrayTracer| {
                let summands = shard_map(
                    |x: XlaArrayTracer| Ok(x),
                    x,
                    mesh.clone(),
                    sharded.clone(),
                    unreduced.clone(),
                    options.clone(),
                )?;
                Ok(shard_map(
                    |summand: XlaArrayTracer| Ok(summand.clone() * summand),
                    summands,
                    mesh.clone(),
                    unreduced.clone(),
                    unreduced.clone(),
                    options.clone(),
                )?)
            },
            ArrayType::new_static(DataType::F32, [2]).with_sharding(sharded.clone()).unwrap(),
        )
        .unwrap();
        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                      (tensor<1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={\"x\"}>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\"}]>] \
                        out_shardings=[<@mesh, [{}], unreduced={\"x\"}>] manual_axes={\"x\"} (%arg1: tensor<1xf32>) {
                      sdy.return %arg1 : tensor<1xf32>
                    } : (tensor<2xf32>) -> tensor<1xf32>
                    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{}], unreduced={\"x\"}>] \
                        out_shardings=[<@mesh, [{}], unreduced={\"x\"}>] manual_axes={\"x\"} (%arg1: tensor<1xf32>) {
                      %2 = stablehlo.multiply %arg1, %arg1 : tensor<1xf32>
                      sdy.return %2 : tensor<1xf32>
                    } : (tensor<1xf32>) -> tensor<1xf32>
                    return %1 : tensor<1xf32>
                  }
                }
            "},
        );

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .into_iter()
            .zip([1f32, 2.])
            .map(|(device, x)| {
                vec![ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(values_to_bytes(&[x]).as_slice(), BufferType::F32, [1], None, device, None)
                            .unwrap(),
                    ),
                    donatable: false,
                }]
            })
            .collect::<Vec<_>>();
        let outputs = executable
            .execute(
                inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                Vec::new(),
                0,
                None,
                None,
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        let summands = outputs
            .into_iter()
            .map(|output| values_from_bytes::<f32>(&output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(summands, vec![vec![1.0], vec![4.0]]);
    }

    #[test]
    fn test_to_mlir_module_for_program_without_manual_variation_transposes_reduced_boundaries_on_cpu() {
        use ryft_core::CotangentDestinationKind;
        use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput};

        use crate::experimental::lowering::to_mlir_module_for_program;

        // The primal is `x -> 2x` over a reduced input and output. Its pullback receives one summand of the unreduced
        // seed on each device, here `[1]` and `[2]` for the seed `3`, and returns one summand of the unreduced input
        // cotangent on each device without dividing the seed or summing the cotangent, so the devices return `[2]`
        // and `[4]`, whose sum is the cotangent `6`. Dividing the seed would give `[1]` and `[2]`, and summing the
        // cotangent along `x` would give `[6]` on both devices.
        let mesh = test_manual_mesh("x", 2);
        let reduced = Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x"]).unwrap();
        let global_type = ArrayType::new_static(DataType::F32, [1]).with_sharding(reduced.clone()).unwrap();
        let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
            |inputs: Vec<XlaTracer<'static>>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    |input: XlaArrayTracer| Ok(input.clone() + input),
                    input,
                    mesh,
                    reduced.clone(),
                    reduced.clone(),
                    ShardMapOptions::new().with_check_manual_variation(false),
                )?;
                Ok(vec![output.into_value()])
            },
            vec![ArrayIrType::Array(global_type)],
        )
        .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        let array_types = |types: Vec<ArrayIrType>| {
            types
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        let input_types = array_types(pullback.input_types().into_owned());
        let output_types = array_types(pullback.output_types().into_owned());
        let shardings = |types: &[ArrayType]| types.iter().map(|r#type| r#type.sharding().unwrap().clone()).collect();
        let input_shardings: Vec<Sharding> = shardings(&input_types);
        let output_shardings: Vec<Sharding> = shardings(&output_types);
        let module = to_mlir_module_for_program(
            &pullback,
            &[],
            &input_types,
            &output_types,
            "main",
            Some(&input_shardings),
            Some(&output_shardings),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<1xf32> {sdy.sharding = \
                      #sdy.sharding<@mesh, [{}], unreduced={\"x\"}>}) -> (tensor<1xf32> {sdy.sharding = \
                      #sdy.sharding<@mesh, [{}], unreduced={\"x\"}>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}], unreduced={\"x\"}>] \
                        out_shardings=[<@mesh, [{}], unreduced={\"x\"}>] manual_axes={\"x\"} (%arg1: tensor<1xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<1xf32>
                      sdy.return %1 : tensor<1xf32>
                    } : (tensor<1xf32>) -> tensor<1xf32>
                    return %0 : tensor<1xf32>
                  }
                }
            "},
        );

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .into_iter()
            .zip([1f32, 2.])
            .map(|(device, seed)| {
                vec![ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(values_to_bytes(&[seed]).as_slice(), BufferType::F32, [1], None, device, None)
                            .unwrap(),
                    ),
                    donatable: false,
                }]
            })
            .collect::<Vec<_>>();
        let outputs = executable
            .execute(
                inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                Vec::new(),
                0,
                None,
                None,
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        let summands = outputs
            .into_iter()
            .map(|output| values_from_bytes::<f32>(&output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(summands, vec![vec![2.0], vec![4.0]]);
    }

    #[test]
    fn test_to_mlir_module_for_program_without_manual_variation_executes_coordinate_readers_on_cpu() {
        use ryft_core::{AxisIndex, Reshape};
        use ryft_pjrt::ExecutionDeviceInputs;

        let mesh = test_manual_mesh("x", 4);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let module = shard_map_module::<_, (), ArrayType>(
            |context, ()| context.axis_index("x")?.reshape([1]),
            (),
            mesh,
            (),
            sharded,
            ShardMapOptions::new().with_check_manual_variation(false),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=4]>
                  func.func @main() -> (tensor<4xui64> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = sdy.manual_computation() in_shardings=[] out_shardings=[<@mesh, [{\"x\"}]>] \
                        manual_axes={\"x\"} () {
                      %1 = stablehlo.partition_id : tensor<ui32>
                      %2 = stablehlo.convert %1 : (tensor<ui32>) -> tensor<ui64>
                      %3 = stablehlo.reshape %2 : (tensor<ui64>) -> tensor<1xui64>
                      sdy.return %3 : tensor<1xui64>
                    } : () -> tensor<4xui64>
                    return %0 : tensor<4xui64>
                  }
                }
            "},
        );

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let outputs = executable
            .execute(
                executable
                    .addressable_devices()
                    .unwrap()
                    .iter()
                    .map(|_| ExecutionDeviceInputs { inputs: &[], ..Default::default() })
                    .collect(),
                Vec::new(),
                0,
                None,
                None,
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 4);
        for (index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<u64>(&bytes), vec![index as u64]);
        }
    }

    #[test]
    fn test_to_mlir_module_for_program_renders_and_executes_matmul_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(8), ..Default::default() }))
            .expect("failed to create 8-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 8);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 8, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();

        let lhs_sharding = Sharding::new(
            device_mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let rhs_sharding = Sharding::replicated(device_mesh.logical_mesh().clone(), 2);
        let output_sharding = Sharding::new(
            device_mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let global_input_types = (
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(4)])),
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)])),
        );
        let mlir_program = shard_map_module::<_, (ArrayType, ArrayType), ArrayType>(
            |_, (lhs, rhs)| lhs.dot(&rhs, &DotDimensionNumbers::matmul()),
            global_input_types,
            device_mesh.logical_mesh().clone(),
            (lhs_sharding.clone(), rhs_sharding.clone()),
            output_sharding.clone(),
            ShardMapOptions::new(),
        )
        .unwrap();

        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=8]>
                  func.func @main(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={"x"}>}) -> (tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {}], replicated={"x"}>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg2: tensor<1x4xf32>, %arg3: tensor<4x2xf32>) {
                      %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4xf32>, tensor<4x2xf32>) -> tensor<1x2xf32>
                      sdy.return %1 : tensor<1x2xf32>
                    } : (tensor<8x4xf32>, tensor<4x2xf32>) -> tensor<8x2xf32>
                    return %0 : tensor<8x2xf32>
                  }
                }
            "#},
        );

        let lhs_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(row_index, device)| {
                let row = row_index as f32;
                client
                    .buffer(
                        values_to_bytes::<f32>(&[row, row + 1.0, row + 2.0, row + 3.0]).as_slice(),
                        BufferType::F32,
                        [1u64, 4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let rhs_values = [1.0f32, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
        let rhs_buffers = client_devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        values_to_bytes::<f32>(rhs_values.as_slice()).as_slice(),
                        BufferType::F32,
                        [4u64, 2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let lhs_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[8, 4], lhs_sharding.clone()),
            device_mesh.clone(),
            lhs_buffers,
        )
        .unwrap();
        let rhs_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[4, 2], rhs_sharding),
            device_mesh,
            rhs_buffers,
        )
        .unwrap();
        let program = PjrtProgram::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(8)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 8);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let row_start_by_device = execution_device_ids
            .iter()
            .map(|device_id| {
                let row_start = lhs_array.device_shard(*device_id).unwrap().slice()[0].start;
                (*device_id, row_start)
            })
            .collect::<HashMap<_, _>>();

        let execute_arguments =
            XlaArray::into_execute_arguments(vec![lhs_array, rhs_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(
                execute_arguments.as_execution_device_inputs(),
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            let row = *row_start_by_device.get(&device_id).unwrap() as f32;
            assert_eq!(values[0], 4.0 * row + 8.0);
            assert_eq!(values[1], 4.0 * row + 4.0);
        }
    }

    #[test]
    fn test_to_shardy_in_shardings() {
        let context = MlirContext::new();
        let mesh = test_logical_mesh_2x2();
        let boundary = ShardMapBoundary::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], Vec::new())],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            to_shardy_in_shardings(&boundary, &[], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"x"}], replicated={"y"}>]>"#,
        );
    }

    #[test]
    fn test_to_shardy_in_shardings_leaves_inactive_manual_axes_open() {
        // Manual axes outside the selected subset are free axes of the manual computation, so they stay open, unless
        // an enclosing manual region already bound them, in which case they are omitted.
        let context = MlirContext::new();
        let mesh = test_logical_mesh_2x2();
        let boundary = ShardMapBoundary::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x", "y"])], Vec::new())],
            Vec::new(),
            vec!["x".into()],
        )
        .unwrap();
        assert_eq!(
            to_shardy_in_shardings(&boundary, &[], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"x", "y", ?}]>]>"#,
        );
        let boundary = ShardMapBoundary::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["y"])], Vec::new())],
            Vec::new(),
            vec!["y".into()],
        )
        .unwrap();
        assert_eq!(
            to_shardy_in_shardings(&boundary, &[], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"y", ?}], replicated={"x"}>]>"#,
        );
        assert_eq!(
            to_shardy_in_shardings(&boundary, &["x".to_string()], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"y"}]>]>"#,
        );
    }

    #[test]
    fn test_to_shardy_in_shardings_renders_free_axes_as_open_dimension_shardings() {
        let context = MlirContext::new();
        let mesh = test_logical_mesh_data_model();
        let boundary = ShardMapBoundary::new(
            mesh.clone(),
            vec![
                test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], Vec::new()),
                test_sharding(&mesh, vec![ShardingDimension::sharded(["data"])], Vec::new()),
            ],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            to_shardy_in_shardings(&boundary, &[], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"data", ?}]>, <@mesh, [{"data", ?}]>]>"#,
        );
    }

    #[test]
    fn test_to_shardy_in_shardings_keeps_explicit_axes() {
        let context = MlirContext::new();
        let mesh = test_logical_mesh_data_model_explicit();
        let boundary = ShardMapBoundary::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], Vec::new())],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            to_shardy_in_shardings(&boundary, &[], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"data", "model", ?}]>]>"#,
        );
    }

    #[test]
    fn test_to_shardy_out_shardings() {
        let context = MlirContext::new();
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let boundary = ShardMapBoundary::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(&mesh, vec![ShardingDimension::replicated()], Vec::new())],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            to_shardy_out_shardings(&boundary, &[], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{?}], replicated={"x"}>]>"#,
        );

        // Output shardings render exactly like input shardings.
        let mesh = test_logical_mesh_data_model();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["data"])], Vec::new());
        let boundary = ShardMapBoundary::new(mesh, vec![sharding.clone()], vec![sharding], Vec::new()).unwrap();
        assert_eq!(
            to_shardy_out_shardings(&boundary, &[], &context).unwrap().to_string(),
            to_shardy_in_shardings(&boundary, &[], &context).unwrap().to_string(),
        );
    }

    #[test]
    fn test_to_shardy_manual_axes() {
        let context = MlirContext::new();
        let boundary = ShardMapBoundary::new(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            ])
            .unwrap(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(to_shardy_manual_axes(&boundary, &context).unwrap().to_string(), r#"#sdy<manual_axes{"y"}>"#);
        let boundary =
            ShardMapBoundary::new(test_logical_mesh_2x2(), Vec::new(), Vec::new(), vec!["x".into()]).unwrap();
        assert_eq!(to_shardy_manual_axes(&boundary, &context).unwrap().to_string(), r#"#sdy<manual_axes{"x"}>"#);
    }

    #[test]
    fn test_shard_map_manual_computation_executes_end_to_end_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .expect("failed to create 4-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();

        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let boundary = ShardMapBoundary::new(
            device_mesh.logical_mesh().clone(),
            vec![sharding.clone()],
            vec![sharding.clone()],
            vec![],
        )
        .unwrap();
        assert_eq!(
            boundary.local_input_type(0, &ArrayType::new_static(DataType::F32, [8])).unwrap().shape(),
            &Shape::new(vec![Dimension::Static(2)]),
        );

        let context = MlirContext::new();
        let input_sharding = boundary.input_shardings()[0].to_mlir(context.unknown_location()).unwrap().to_string();
        let output_sharding = boundary.output_shardings()[0].to_mlir(context.unknown_location()).unwrap().to_string();
        let manual_computation_attributes =
            r#"in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"}"#;
        let mesh_module = context.module(context.unknown_location()).unwrap();
        let mesh_operation = mesh_module
            .body()
            .unwrap()
            .append_operation(boundary.mesh().to_mlir(context.unknown_location()).unwrap())
            .unwrap()
            .to_string();

        let mlir_program = format!(
            r#"
                module {{
                    {mesh_operation}
                    func.func @main(
                        %arg0: tensor<8xf32> {{sdy.sharding = {input_sharding}}}
                    ) -> (tensor<8xf32> {{sdy.sharding = {output_sharding}}}) {{
                        %0 = sdy.manual_computation(%arg0) {manual_computation_attributes} (%arg1: tensor<2xf32>) {{
                            %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                            sdy.return %1 : tensor<2xf32>
                        }} : (tensor<8xf32>) -> tensor<8xf32>
                        return %0 : tensor<8xf32>
                    }}
                }}
            "#,
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = PjrtProgram::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
        let expected_values_by_device = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                (device.id().unwrap(), [device_index as f32 * 4.0 + 2.0, device_index as f32 * 4.0 + 4.0])
            })
            .collect::<HashMap<_, _>>();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();

        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(
                execute_arguments.as_execution_device_inputs(),
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, *expected_values_by_device.get(&device_id).unwrap());
        }
    }
}
