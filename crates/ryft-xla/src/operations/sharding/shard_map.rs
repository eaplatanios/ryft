use std::collections::HashSet;

use ryft_core::{
    ArrayIrType, ArrayType, PARALLEL_ALL_GATHER_OPERATION_NAME, ParallelAllGatherOperation,
    ParallelAllGatherOutputVariance, Parameterized, ProgramError, SHARD_MAP_OPERATION_NAME, ShardMapBoundary,
    ShardMapError, ShardMapOperation, ShardMapValueKind, Sharding, ShardingDimension, Typed, TypedProgram,
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
use crate::experimental::ops::{Constant, Program};
use crate::operations::control_flow::lowered_effect_classes;
use crate::sharding::SHARDY_MESH_SYMBOL_NAME;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, ToMlir};

/// Lowers one `sdy.manual_computation` operation, threading per-device effect tokens across its boundary. Assertions
/// remain local to an executable, and the only ordered I/O that reaches this function is per-device
/// (`DeviceOrderedIo`), because core shard-map boundary validation rejects bodies with the `OrderedIo` effect (refer to
/// [`ShardMapError::OrderedIoNotSupported`]).
pub(crate) fn lower_manual_computation<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    block: &mut BlockRef<'b, 'c, 't>,
    outer_inputs: &[ValueRef<'b, 'c, 't>],
    boundary: &ShardMapBoundary,
    program: &Program<ProgramInput, ProgramOutput, O>,
    local_input_types: &[ArrayType],
    global_output_types: &[ArrayType],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    // Tracked variation adjoints would duplicate the conservative unchecked boundary's gradient reductions. Low-level
    // programs can bypass the closure constructor, so enforce ordinary active-axis types before lowering their bodies.
    if !boundary.checks_manual_variation() {
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
                    && boundary.manual_axes().iter().any(|axis| gather.axes().contains(axis))
                {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{SHARD_MAP_OPERATION_NAME}` with unchecked manual variation requires varying output \
                         variance for `{PARALLEL_ALL_GATHER_OPERATION_NAME}` over active manual axis `{}`",
                        gather.axes(),
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
    let shardings = [
        (ShardMapValueKind::Input, boundary.input_shardings()),
        (ShardMapValueKind::Output, boundary.output_shardings()),
    ];
    for (value_kind, shardings) in shardings {
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
    // Shardy rejects a manual computation whose result drops an unreduced axis of an input (e.g., a map that completes
    // the pending sum of its unreduced input) unless a blessed operation marks that input unreduced, so every input with
    // unreduced axes passes through a sharding constraint, as the output resharding of JAX's lowering emits.
    for (index, sharding) in boundary.input_shardings().iter().enumerate() {
        if !sharding.unreduced_axes().is_empty() {
            let operation = shardy::sharding_constraint(manual_inputs[index], input_shardings[index], location)?;
            let constraint = block.append_operation(operation)?;
            manual_inputs[index] =
                constraint.result(0).expect("sdy.sharding_constraint should return one result").as_ref();
        }
    }
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
        sharding.unreduced_kind().to_mlir(context.unknown_location())?,
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
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
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
        let simplified_body = body.clone().into_simplified()?;
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
        AxisIndexOperation, BatchAxis, BatchAxisSpecification, BroadcastOperation, Context, ConvertElementType,
        ConvertElementTypeOperation, CotangentDestinationKind, CustomCallOperation, DataType, Device, DeviceMesh,
        Dimension as CoreDimension, DomainTracer, DomainTracingContext, Dot, DotDimensionNumbers, EffectClass, Exp,
        LogicalMesh, MeshAxis, MeshAxisType, PARALLEL_BROADCAST_OPERATION_NAME, ParallelAllGather, ParallelAllToAll,
        ParallelArgExtrema, ParallelBroadcast, ParallelCast, ParallelCastTarget, ParallelPermute, ParallelReduce,
        ParallelSumScatter, Placeholder, ProjectedValue, ReductionKind, ReshapeOperation, Shape, ShardMapOptions,
        Typed, UnreducedKind, Value as CoreValue, ValueProjection, batch, differentiate_at, shard_map, shard_map_along,
        shard_map_with_inferred_input_shardings,
    };
    use ryft_mlir::Block;
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program as PjrtProgram, load_cpu_plugin};

    use crate::domains::Tracer;
    use crate::experimental::ArrayTracer;
    use crate::experimental::lowering::tests::{
        lower_replicated_shard_map_body, shard_map_module, static_sharded_array_type, test_manual_mesh, test_sharding,
        test_spmd_compilation_options, test_vector_type,
    };
    use crate::experimental::lowering::{lower_mlir_module_for_program, to_mlir_module_for_program};
    use crate::experimental::ops::ProgramBuilder as ArrayIrProgramBuilder;
    use crate::experimental::tracing::{TracedProgram, trace_with_context};
    use crate::operations::assertions::ensure_assertion_handler_registered;
    use crate::operations::debugging::PRINT_CUSTOM_CALL_TARGET;
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{Array, Domain, FromPjrt, Options, Session, ToMlir, Value};

    use super::*;

    /// Returns the elements of the buffer that every addressable device holds for `array`, in device order.
    fn read_device_f32s(array: &Array<'_>) -> Vec<Vec<f32>> {
        array
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect()
    }

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
    fn bind_doubling_shard_map(x: ArrayTracer, boundary: ShardMapBoundary) -> Result<ArrayTracer, ProgramError> {
        let global_type = boundary.global_input_type(0, &x.r#type())?;
        let local_type = boundary.local_input_type(0, &global_type)?;
        let (_, body) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
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
        let mut builder = ArrayIrProgramBuilder::new();
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
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
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

        let mut builder = ArrayIrProgramBuilder::new();
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
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
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
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(16)]));
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
                move |_, x: ArrayTracer| {
                    let nested: ArrayTracer = shard_map(
                        |y: ArrayTracer| Ok(y.clone() + y),
                        x.clone(),
                        inner_mesh.clone(),
                        inner_sharding.clone(),
                        inner_sharding.clone(),
                        ShardMapOptions::new().with_manual_axes(["y"]),
                    )?;
                    Ok(nested + x)
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)])),
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
    fn test_to_mlir_module_for_program_lowers_collectives_over_nested_meshes_that_retype_free_axes() {
        // An outer map over `x` contains an inner map over `y` whose mesh types the free axis `z` as explicit instead of
        // auto, as `explicit_axes` does for the maps inside it. A collective over both manual axes communicates across
        // both regions. Their meshes share their topology, which is all that its replica groups depend on, so the
        // groups are those of either mesh.
        let mesh = |free_axis_type| {
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
                MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
                MeshAxis::new("z", 1, free_axis_type).unwrap(),
            ])
            .unwrap()
        };
        let outer_mesh = mesh(MeshAxisType::Auto);
        let inner_mesh = mesh(MeshAxisType::Explicit);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            {
                let inner_mesh = inner_mesh.clone();
                move |_, x: ArrayTracer| {
                    let nested: ArrayTracer = shard_map(
                        |y: ArrayTracer| {
                            Ok(y.parallel_all_gather(["x", "y"], 0)?
                                .parallel_reduce(ReductionKind::Sum, "x")?
                                .parallel_reduce(ReductionKind::Sum, "y")?)
                        },
                        x,
                        inner_mesh.clone(),
                        Sharding::replicated(inner_mesh.clone(), 0),
                        Sharding::replicated(inner_mesh.clone(), 1),
                        ShardMapOptions::new().with_manual_axes(["y"]),
                    )?;
                    Ok(nested)
                }
            },
            ArrayType::scalar(DataType::F32),
            outer_mesh.clone(),
            Sharding::replicated(outer_mesh.clone(), 0),
            Sharding::replicated(outer_mesh, 1),
            ShardMapOptions::new().with_manual_axes(["x"]),
        )
        .unwrap();
        assert_eq!(module, "");
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
            |_, x: ArrayTracer| {
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
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(error)
                    if error.downcast_custom::<ShardMapError>() == Some(&expected),
            ),
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
            |_, x: ArrayTracer| {
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
            value_kind: ShardMapValueKind::Input,
            value_index: 0,
            axis_name: "x".to_string(),
        };
        assert!(matches!(
            module,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(error)
                    if error.downcast_custom::<ShardMapError>() == Some(&expected),
            ),
        ));
    }

    #[test]
    fn test_to_mlir_module_for_program_executes_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .expect("failed to create 4-device CPU client");
        let domain = Session::new(&client).domain();
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
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]));
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

        let input_array = Array::from_addressable_buffers(
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
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
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
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
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
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    |input: ArrayTracer| input.parallel_reduce(ReductionKind::Sum, "x"),
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

    /// Compiles `module` for two CPU devices, executes it with one `f32` input buffer of shape `local_shape` per device
    /// holding `device_inputs[device]`, and returns the elements of the first output of every device.
    fn execute_on_two_cpu_devices(module: String, local_shape: &[u64], device_inputs: [&[f32]; 2]) -> Vec<Vec<f32>> {
        execute_inputs_on_two_cpu_devices(module, &[(local_shape, device_inputs)])
    }

    /// Compiles `module` for two CPU devices, executes it with one `f32` buffer per device for every `(local_shape,
    /// device_inputs)` entry of `inputs`, holding `device_inputs[device]`, and returns the elements of the first output
    /// of every device.
    fn execute_inputs_on_two_cpu_devices(module: String, inputs: &[(&[u64], [&[f32]; 2])]) -> Vec<Vec<f32>> {
        execute_inputs_on_cpu_devices(module, inputs)
    }

    /// Compiles `module` for `N` CPU devices, executes it with one `f32` buffer per device for every `(local_shape,
    /// device_inputs)` entry of `inputs`, holding `device_inputs[device]`, and returns the elements of the first output
    /// of every device.
    fn execute_inputs_on_cpu_devices<const N: usize>(
        module: String,
        inputs: &[(&[u64], [&[f32]; N])],
    ) -> Vec<Vec<f32>> {
        use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(N), ..Default::default() }))
            .unwrap();
        ensure_assertion_handler_registered(&client).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(N))
            .unwrap();
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, device)| {
                inputs
                    .iter()
                    .map(|(local_shape, device_inputs)| {
                        let bytes = values_to_bytes(device_inputs[index]);
                        let buffer = client
                            .buffer(bytes.as_slice(), BufferType::F32, *local_shape, None, device.clone(), None)
                            .unwrap();
                        ExecutionInput { buffer: Arc::new(buffer), donatable: false }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        executable
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
            .unwrap()
            .into_iter()
            .map(|output| values_from_bytes::<f32>(&output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap()))
            .collect()
    }

    #[test]
    fn test_to_mlir_module_for_program_with_parallel_casts_executes_on_cpu() {
        // The body casts the shared `w` to reduced and then to varying, scales each device's tile of `v` by it, casts
        // the products to unreduced, and sums them. The casts lower to nothing and the sum to an all-reduce, so both
        // devices hold `w · (v₀ + v₁)`, as the same program does in JAX (`.tasks/shard_map_jax_parity_gaps/
        // m3_oracles.py`).
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace_with_context(
            |_, (w, v): (ArrayTracer, ArrayTracer)| {
                Ok(shard_map(
                    |(w, v): (ArrayTracer, ArrayTracer)| {
                        let w = w.parallel_cast("x", ParallelCastTarget::Reduced)?;
                        let w = w.parallel_cast("x", ParallelCastTarget::Varying)?;
                        let product = (w * v).parallel_cast("x", ParallelCastTarget::Unreduced)?;
                        Ok(product.parallel_reduce(ReductionKind::Sum, "x")?)
                    },
                    (w, v),
                    mesh.clone(),
                    (replicated.clone(), sharded.clone()),
                    replicated.clone(),
                    ShardMapOptions::new(),
                )?)
            },
            (
                ArrayType::new_static(DataType::F32, [2]).with_sharding(replicated.clone()).unwrap(),
                ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
            ),
        )
        .unwrap();
        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={\"x\"}>}, \
                      %arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                      (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={\"x\"}>}) {
                    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{}], replicated={\"x\"}>, \
                        <@mesh, [{\"x\"}]>] out_shardings=[<@mesh, [{}], replicated={\"x\"}>] manual_axes={\"x\"} \
                        (%arg2: tensor<2xf32>, %arg3: tensor<2xf32>) {
                      %1 = stablehlo.multiply %arg2, %arg3 : tensor<2xf32>
                      %2 = \"stablehlo.all_reduce\"(%1) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = \
                          dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
                        %3 = stablehlo.add %arg4, %arg5 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<2xf32>, tensor<4xf32>) -> tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "},
        );
        assert_eq!(
            execute_inputs_on_two_cpu_devices(
                module,
                &[(&[2], [&[2.0, 3.0], &[2.0, 3.0]]), (&[2], [&[1.0, 4.0], &[5.0, 7.0]])],
            ),
            vec![vec![12.0, 33.0], vec![12.0, 33.0]],
        );

        // An unreduced output of one checked map carries the per-device summands into another checked map, whose sum
        // completes them.
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace_with_context(
            |_, x: ArrayTracer| {
                let summands = shard_map(
                    |x: ArrayTracer| Ok(x.parallel_cast("x", ParallelCastTarget::Unreduced)?),
                    x,
                    mesh.clone(),
                    sharded.clone(),
                    unreduced.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(shard_map(
                    |summands: ArrayTracer| Ok(summands.parallel_reduce(ReductionKind::Sum, "x")?),
                    summands,
                    mesh.clone(),
                    unreduced.clone(),
                    replicated.clone(),
                    ShardMapOptions::new(),
                )?)
            },
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
        )
        .unwrap();
        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                      (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={\"x\"}>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\"}]>] \
                        out_shardings=[<@mesh, [{}], unreduced={\"x\"}>] manual_axes={\"x\"} (%arg1: tensor<2xf32>) {
                      sdy.return %arg1 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<2xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}], unreduced={\"x\"}> : tensor<2xf32>
                    %2 = sdy.manual_computation(%1) in_shardings=[<@mesh, [{}], unreduced={\"x\"}>] \
                        out_shardings=[<@mesh, [{}], replicated={\"x\"}>] manual_axes={\"x\"} (%arg1: tensor<2xf32>) {
                      %3 = \"stablehlo.all_reduce\"(%arg1) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = \
                          dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %4 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %3 : tensor<2xf32>
                    } : (tensor<2xf32>) -> tensor<2xf32>
                    return %2 : tensor<2xf32>
                  }
                }
            "},
        );
        assert_eq!(execute_on_two_cpu_devices(module, &[2], [&[1.0, 2.0], &[3.0, 4.0]]), vec![vec![4.0, 6.0]; 2]);
    }

    #[test]
    fn test_to_mlir_module_for_program_with_unreduced_kinds_executes_on_cpu() {
        // An unchecked map returns each device's tile as its candidate for an output that is unreduced along `x` with
        // kind `max` (respectively, `min`), which lowers to Shardy's reduction operation of the unreduced axes, and a
        // checked map completes it with an all-reduce of that kind. Device 0 holds `[1, 7]` and device 1 holds
        // `[5, 2]`, so both devices end with `[5, 7]` (respectively, `[1, 2]`), as JAX's `unreduced_pmax` and
        // `unreduced_pmin` do (`.tasks/shard_map_jax_parity_gaps/m4_oracles.py`).
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let module = |kind: UnreducedKind, reduction: ReductionKind| {
            let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap();
            let traced: TracedProgram<ArrayType, ArrayType> = trace_with_context(
                |_, x: ArrayTracer| {
                    let candidates = shard_map(
                        |x: ArrayTracer| Ok(x),
                        x,
                        mesh.clone(),
                        sharded.clone(),
                        unreduced.clone(),
                        ShardMapOptions::new().with_check_manual_variation(false),
                    )?;
                    Ok(shard_map(
                        |candidates: ArrayTracer| Ok(candidates.parallel_reduce(reduction, "x")?),
                        candidates,
                        mesh.clone(),
                        unreduced.clone(),
                        replicated.clone(),
                        ShardMapOptions::new(),
                    )?)
                },
                ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
            )
            .unwrap();
            traced.to_mlir_module("main").unwrap()
        };
        let maximum = module(UnreducedKind::Max, ReductionKind::Max);
        assert_eq!(
            maximum,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                      (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={\"x\"}>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\"}]>] \
                        out_shardings=[<@mesh, [{}], unreduced=max{\"x\"}>] manual_axes={\"x\"} \
                        (%arg1: tensor<2xf32>) {
                      sdy.return %arg1 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<2xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}], unreduced=max{\"x\"}> : tensor<2xf32>
                    %2 = sdy.manual_computation(%1) in_shardings=[<@mesh, [{}], unreduced=max{\"x\"}>] \
                        out_shardings=[<@mesh, [{}], replicated={\"x\"}>] manual_axes={\"x\"} (%arg1: tensor<2xf32>) {
                      %3 = \"stablehlo.all_reduce\"(%arg1) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = \
                          dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %4 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
                        stablehlo.return %4 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %3 : tensor<2xf32>
                    } : (tensor<2xf32>) -> tensor<2xf32>
                    return %2 : tensor<2xf32>
                  }
                }
            "},
        );
        assert_eq!(execute_on_two_cpu_devices(maximum, &[2], [&[1.0, 7.0], &[5.0, 2.0]]), vec![vec![5.0, 7.0]; 2]);
        let minimum = module(UnreducedKind::Min, ReductionKind::Min);
        assert_eq!(execute_on_two_cpu_devices(minimum, &[2], [&[1.0, 7.0], &[5.0, 2.0]]), vec![vec![1.0, 2.0]; 2]);
    }

    #[test]
    fn test_to_mlir_module_for_program_with_parallel_cast_gradient_executes_on_cpu() {
        // Each device computes exp(input), casts it to a pending sum, and sums across x. The reverse pass must
        // propagate through all three cast adjoints and sum the two device contributions, so the gradient at zero is 2.
        let mesh = test_manual_mesh("x", 2);
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let traced: TracedProgram<ArrayType, ArrayType> = trace_with_context(
            |_, input: ArrayTracer| {
                let gradient = differentiate_at(input.into_value()).gradient(|input| {
                    let input = ValueProjection::<ArrayType>::into_projected(input)?;
                    let output = shard_map(
                        |input| {
                            let reduced = input.parallel_cast("x", ParallelCastTarget::Reduced)?;
                            let varying = reduced.parallel_cast("x", ParallelCastTarget::Varying)?;
                            let summand = varying.exp()?.parallel_cast("x", ParallelCastTarget::Unreduced)?;
                            Ok(summand.parallel_reduce(ReductionKind::Sum, "x")?)
                        },
                        input,
                        mesh.clone(),
                        replicated.clone(),
                        replicated.clone(),
                        ShardMapOptions::new(),
                    )?;
                    Ok(output.into_value())
                })?;
                Ok(ValueProjection::<ArrayType>::into_projected(gradient)?)
            },
            ArrayType::scalar(DataType::F32).with_sharding(replicated.clone()).unwrap(),
        )
        .unwrap();
        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={\"x\"}>}) -> \
                      (tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={\"x\"}>}) {
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [], replicated={\"x\"}>] \
                        out_shardings=[<@mesh, [{\"x\"}]>] manual_axes={\"x\"} (%arg1: tensor<f32>) {
                      %2 = stablehlo.exponential %arg1 : tensor<f32>
                      %3 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
                      %4 = sdy.sharding_constraint %3 <@mesh, [{}]> : tensor<1xf32>
                      sdy.return %4 : tensor<1xf32>
                    } : (tensor<f32>) -> tensor<2xf32>
                    %1 = sdy.manual_computation(%cst, %0) in_shardings=[<@mesh, [], replicated={\"x\"}>, <@mesh, [{\"x\"}]>] \
                        out_shardings=[<@mesh, [], replicated={\"x\"}>] manual_axes={\"x\"} \
                        (%arg1: tensor<f32>, %arg2: tensor<1xf32>) {
                      %2 = stablehlo.reshape %arg2 : (tensor<1xf32>) -> tensor<f32>
                      %3 = sdy.sharding_constraint %2 <@mesh, []> : tensor<f32>
                      %4 = stablehlo.multiply %3, %arg1 : tensor<f32>
                      %5 = \"stablehlo.all_reduce\"(%4) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = \
                          dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
                        %6 = stablehlo.add %arg3, %arg4 : tensor<f32>
                        stablehlo.return %6 : tensor<f32>
                      }) : (tensor<f32>) -> tensor<f32>
                      sdy.return %5 : tensor<f32>
                    } : (tensor<f32>, tensor<2xf32>) -> tensor<f32>
                    return %1 : tensor<f32>
                  }
                }
            "},
        );
        assert_eq!(execute_on_two_cpu_devices(module, &[], [&[0.0], &[0.0]]), vec![vec![2.0]; 2]);
    }

    #[test]
    fn test_to_mlir_module_for_program_with_parallel_arg_extrema_executes_on_cpu() {
        // Device 0 holds `[3, 1, 2]` and device 1 holds `[3, 5, 0]`, so the first element ties and resolves to the lowest
        // position, as JAX's `pargmax` and `pargmin` resolve it (`.tasks/shard_map_jax_parity_gaps/m3_oracles.py`). The
        // positions are converted to `f32` outside the map only to read them back.
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        for (argmax, expected) in [(true, [0.0, 1.0, 0.0]), (false, [0.0, 0.0, 1.0])] {
            let traced: TracedProgram<ArrayType, ArrayType> = trace_with_context(
                |_, x: ArrayTracer| {
                    let positions = shard_map(
                        |x: ArrayTracer| Ok(if argmax { x.parallel_argmax("x")? } else { x.parallel_argmin("x")? }),
                        x,
                        mesh.clone(),
                        sharded.clone(),
                        replicated.clone(),
                        ShardMapOptions::new(),
                    )?;
                    Ok(positions.convert_element_type(DataType::F32)?)
                },
                ArrayType::new_static(DataType::F32, [6]).with_sharding(sharded.clone()).unwrap(),
            )
            .unwrap();
            let module = traced.to_mlir_module("main").unwrap();
            assert_eq!(
                execute_on_two_cpu_devices(module, &[3], [&[3.0, 1.0, 2.0], &[3.0, 5.0, 0.0]]),
                vec![expected.to_vec(); 2],
            );
        }
    }

    #[test]
    fn test_to_mlir_module_for_program_with_parallel_broadcast_rejected_on_cpu() {
        // Every device receives the shard of the device at the source position. The replica group is reordered so that
        // the source comes first, which yields the module that JAX's GPU lowering of `pbroadcast` emits
        // (`.tasks/shard_map_jax_parity_gaps/m8/pbroadcast.mlir`).
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    |input: ArrayTracer| Ok(input.parallel_broadcast("x", 1)?),
                    input,
                    mesh.clone(),
                    sharded.clone(),
                    sharded.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(vec![output.into_value()])
            },
            vec![ArrayIrType::Array(global_type.clone())],
        )
        .unwrap();
        let types = [global_type];
        let shardings = [sharded];
        let module =
            to_mlir_module_for_program(&program, &[], &types, &types, "main", Some(&shardings), Some(&shardings))
                .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                      (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\"}]>] \
                        out_shardings=[<@mesh, [{\"x\"}]>] manual_axes={\"x\"} (%arg1: tensor<2xf32>) {
                      %1 = \"stablehlo.collective_broadcast\"(%arg1) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = \
                          dense<[[1, 0]]> : tensor<1x2xi64>}> : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );

        // The XLA CPU backend cannot compile `collective-broadcast`, so compiling the module fails in XLA, and lowering
        // for a CPU target fails earlier with a precise error.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        assert!(matches!(
            client.compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2)),
            Err(ryft_pjrt::Error::Unimplemented { message, .. })
                if message.contains("HLO opcode `collective-broadcast` is not supported by XLA:CPU"),
        ));
        assert!(matches!(
            lower_mlir_module_for_program(
                &program,
                &[],
                &types,
                &types,
                "main",
                Some(&shardings),
                Some(&shardings),
                Some("cpu"),
            ),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message == format!(
                        "`{PARALLEL_BROADCAST_OPERATION_NAME}` is \
                         not supported by the XLA CPU backend",
                    ),
            ),
        ));
    }

    #[test]
    fn test_to_mlir_module_for_program_with_degenerate_parallel_broadcast_executes_on_cpu() {
        // A broadcast over the single device of axis `y` is the identity, so it lowers to nothing and also executes on
        // the CPU backend, which cannot compile `collective-broadcast`.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let global_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    |input: ArrayTracer| Ok(input.parallel_broadcast("y", 0)?),
                    input,
                    mesh.clone(),
                    sharded.clone(),
                    sharded.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(vec![output.into_value()])
            },
            vec![ArrayIrType::Array(global_type.clone())],
        )
        .unwrap();
        let types = [global_type];
        let shardings = [sharded];
        let module = lower_mlir_module_for_program(
            &program,
            &[],
            &types,
            &types,
            "main",
            Some(&shardings),
            Some(&shardings),
            Some("cpu"),
        )
        .unwrap()
        .stable_hlo;
        assert_eq!(
            execute_on_two_cpu_devices(module, &[2], [&[1.0, 2.0], &[3.0, 4.0]]),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_with_dynamic_parallel_broadcast_executes_on_cpu() {
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let input_types = [
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
            ArrayType::scalar(DataType::F32).with_sharding(replicated.clone()).unwrap(),
        ];
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let value = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let source = ValueProjection::<ArrayType>::into_projected(inputs[1].clone())?;
                let output = shard_map(
                    |(value, source): (ArrayTracer, ArrayTracer)| {
                        value.parallel_broadcast("x", &source.convert_element_type(DataType::U64)?)
                    },
                    (value, source),
                    mesh.clone(),
                    (sharded.clone(), replicated.clone()),
                    sharded.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(vec![output.into_value()])
            },
            input_types.iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>(),
        )
        .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &input_types,
            &[input_types[0].clone()],
            "main",
            Some(&[sharded.clone(), replicated]),
            Some(&[sharded]),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}, %arg1: \
                      tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={\"x\"}>}) -> \
                      (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2:3 = sdy.manual_computation(%arg0, %arg1, %0, %1) in_shardings=[<@mesh, [{\"x\"}]>, <@mesh, \
                        [], replicated={\"x\"}>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{\"x\"}]>, \
                        <@mesh, []>, <@mesh, []>] manual_axes={\"x\"} (%arg2: tensor<2xf32>, %arg3: tensor<f32>, \
                        %arg4: !stablehlo.token, %arg5: !stablehlo.token) {
                      %3 = stablehlo.convert %arg3 : (tensor<f32>) -> tensor<ui64>
                      %4 = \"stablehlo.all_reduce\"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 1, \
                          type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg6: tensor<ui64>, %arg7: tensor<ui64>):
                        %19 = stablehlo.minimum %arg6, %arg7 : tensor<ui64>
                        stablehlo.return %19 : tensor<ui64>
                      }) : (tensor<ui64>) -> tensor<ui64>
                      %5 = \"stablehlo.all_reduce\"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 2, \
                          type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg6: tensor<ui64>, %arg7: tensor<ui64>):
                        %19 = stablehlo.maximum %arg6, %arg7 : tensor<ui64>
                        stablehlo.return %19 : tensor<ui64>
                      }) : (tensor<ui64>) -> tensor<ui64>
                      %6 = stablehlo.compare EQ, %4, %5, UNSIGNED : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
                      %7 = stablehlo.custom_call @ryft.assert(%6, %4, %5, %arg4) {api_version = 4 : i32, \
                          backend_config = {actor = \"assert\", kind = \"generic\", label_0 = \"minimum\", label_1 = \
                          \"maximum\", label_count = \"2\", message = \"`parallel_broadcast` runtime sources must \
                          agree across participants\"}, has_side_effect = true} : (tensor<i1>, tensor<ui64>, \
                          tensor<ui64>, !stablehlo.token) -> !stablehlo.token
                      %c = stablehlo.constant dense<2> : tensor<ui64>
                      %8 = stablehlo.compare LT, %5, %c, UNSIGNED : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
                      %9 = stablehlo.custom_call @ryft.assert(%8, %3, %7) {api_version = 4 : i32, \
                          backend_config = {actor = \"assert\", kind = \"generic\", label_0 = \"source\", label_count \
                          = \"1\", message = \"`parallel_broadcast` runtime source must be smaller than axis size \
                          2\"}, has_side_effect = true} : (tensor<i1>, tensor<ui64>, !stablehlo.token) -> \
                          !stablehlo.token
                      %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
                      %11 = \"stablehlo.all_gather\"(%10) <{all_gather_dim = 0 : i64, channel_handle = \
                          #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = dense<[[0, 1]]> : \
                          tensor<1x2xi64>, use_global_device_ids}> : (tensor<1x2xf32>) -> tensor<2x2xf32>
                      %12 = stablehlo.slice %11 [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
                      %13 = stablehlo.reshape %12 : (tensor<1x2xf32>) -> tensor<2xf32>
                      %14 = stablehlo.slice %11 [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
                      %15 = stablehlo.reshape %14 : (tensor<1x2xf32>) -> tensor<2xf32>
                      %c_0 = stablehlo.constant dense<1> : tensor<ui64>
                      %16 = stablehlo.compare EQ, %3, %c_0, UNSIGNED : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
                      %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<2xi1>
                      %18 = stablehlo.select %17, %15, %13 : tensor<2xi1>, tensor<2xf32>
                      sdy.return %18, %9, %arg5 : tensor<2xf32>, !stablehlo.token, !stablehlo.token
                    } : (tensor<4xf32>, tensor<f32>, !stablehlo.token, !stablehlo.token) -> (tensor<4xf32>, \
                        !stablehlo.token, !stablehlo.token)
                    return %2#0 : tensor<4xf32>
                  }
                }
            "
            },
        );
        assert_eq!(
            execute_inputs_on_cpu_devices(
                module.clone(),
                &[(&[2], [&[1f32, 2.0], &[3.0, 4.0]]), (&[], [&[0f32], &[0f32]])],
            ),
            vec![vec![1f32, 2.0], vec![1.0, 2.0]],
        );
        assert_eq!(
            execute_inputs_on_cpu_devices(
                module.clone(),
                &[(&[2], [&[1f32, 2.0], &[3.0, 4.0]]), (&[], [&[1f32], &[1f32]])],
            ),
            vec![vec![3f32, 4.0], vec![3.0, 4.0]],
        );
        let nan = f32::from_bits(0x7fc01234);
        let output =
            execute_inputs_on_cpu_devices(module, &[(&[2], [&[1f32, 2.0], &[-0.0, nan]]), (&[], [&[1f32], &[1f32]])]);
        assert_eq!(
            output.into_iter().flatten().map(f32::to_bits).collect::<Vec<_>>(),
            vec![(-0.0f32).to_bits(), nan.to_bits(), (-0.0f32).to_bits(), nan.to_bits()],
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_with_multi_axis_parallel_broadcast() {
        // On a 2x2 mesh, device `(x, y)` has ID `2x + y`. Source 2 along `("x", "y")` is device `(1, 0)`, whose ID is 2,
        // and along `("y", "x")` it is device `(0, 1)`, whose ID is 1, so each single replica group starts with that
        // device. Along `y` alone, every row of the mesh forms its own group, which starts with its device at `y = 1`.
        let mesh = test_logical_mesh_2x2();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace_with_context(
            |_, x: ArrayTracer| {
                Ok(shard_map(
                    |x: ArrayTracer| {
                        let x = x.parallel_broadcast(["x", "y"], 2)?;
                        let x = x.parallel_broadcast(["y", "x"], 2)?;
                        Ok(x.parallel_broadcast("y", 1)?)
                    },
                    x,
                    mesh.clone(),
                    sharded.clone(),
                    sharded.clone(),
                    ShardMapOptions::new(),
                )?)
            },
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\", \"y\"}]>}) -> \
                      (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\", \"y\"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\", \"y\"}]>] \
                        out_shardings=[<@mesh, [{\"x\", \"y\"}]>] manual_axes={\"x\", \"y\"} (%arg1: tensor<1xf32>) {
                      %1 = \"stablehlo.collective_broadcast\"(%arg1) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = \
                          dense<[[2, 0, 1, 3]]> : tensor<1x4xi64>}> : (tensor<1xf32>) -> tensor<1xf32>
                      %2 = \"stablehlo.collective_broadcast\"(%1) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = \
                          dense<[[1, 0, 2, 3]]> : tensor<1x4xi64>}> : (tensor<1xf32>) -> tensor<1xf32>
                      %3 = \"stablehlo.collective_broadcast\"(%2) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = \
                          dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}> : (tensor<1xf32>) -> tensor<1xf32>
                      sdy.return %3 : tensor<1xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_transposes_parallel_broadcast_on_cpu() {
        // The transpose hands the sum of every device's cotangent to the source device and zeros to the other device,
        // through an all-reduce and a collective permute whose only pair sends the source's sum to itself, so it
        // executes on the CPU backend, unlike the broadcast itself. With the cotangent `[1, 2, 3, 4]` and source 1, the
        // gradient is `[0, 0, 4, 6]`, as JAX computes it (`.tasks/shard_map_jax_parity_gaps/m8_oracles.py`).
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    |input: ArrayTracer| Ok(input.parallel_broadcast("x", 1)?),
                    input,
                    mesh.clone(),
                    sharded.clone(),
                    sharded.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(vec![output.into_value()])
            },
            vec![ArrayIrType::Array(global_type.clone())],
        )
        .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        let module = to_mlir_module_for_program(
            &pullback,
            &[],
            &[global_type.clone()],
            &[global_type],
            "main",
            Some(&[sharded.clone()]),
            Some(&[sharded]),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                      (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\"}]>] \
                        out_shardings=[<@mesh, [{\"x\"}]>] manual_axes={\"x\"} (%arg1: tensor<2xf32>) {
                      %1 = \"stablehlo.all_reduce\"(%arg1) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = \
                          dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      %2 = \"stablehlo.collective_permute\"(%1) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 2, type = 1>, source_target_pairs = \
                          dense<1> : tensor<1x2xi64>}> : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );
        assert_eq!(
            execute_on_two_cpu_devices(module, &[2], [&[1.0, 2.0], &[3.0, 4.0]]),
            vec![vec![0.0, 0.0], vec![4.0, 6.0]],
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_with_multi_axis_collectives_executes_on_cpu() {
        // Over several manual axes, the participants are positioned along the combined axis, which linearizes the axes
        // in the given order with the first axis major. Every device holds its tile of `0..n`, sharded along `("x", "y")`
        // on a 2x2 mesh, so the expected values are those of JAX's collectives with tuple axis names
        // (`.tasks/shard_map_jax_parity_gaps/m3_oracles.py`).
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let run = |n: usize, body: &dyn Fn(ArrayTracer) -> Result<ArrayTracer, ProgramError>| {
            let traced: TracedProgram<ArrayType, ArrayType> = trace_with_context(
                |_, x: ArrayTracer| {
                    Ok(shard_map(
                        |x: ArrayTracer| body(x),
                        x,
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                        ShardMapOptions::new(),
                    )?)
                },
                ArrayType::new_static(DataType::F32, [n]).with_sharding(sharded.clone()).unwrap(),
            )
            .unwrap();
            let tile = n / 4;
            let values = (0..n).map(|value| value as f32).collect::<Vec<_>>();
            let tiles: [&[f32]; 4] = std::array::from_fn(|device| &values[device * tile..(device + 1) * tile]);
            execute_inputs_on_cpu_devices(traced.to_mlir_module("main").unwrap(), &[(&[tile as u64], tiles)]).concat()
        };
        for (axes, scatter, all_to_all, gather) in [
            (
                ["x", "y"],
                vec![24.0, 28.0, 32.0, 36.0],
                vec![0.0, 4.0, 8.0, 12.0, 1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0],
                [0.0, 1.0, 2.0, 3.0],
            ),
            (
                ["y", "x"],
                vec![24.0, 32.0, 28.0, 36.0],
                vec![0.0, 8.0, 4.0, 12.0, 2.0, 10.0, 6.0, 14.0, 1.0, 9.0, 5.0, 13.0, 3.0, 11.0, 7.0, 15.0],
                [0.0, 2.0, 1.0, 3.0],
            ),
        ] {
            assert_eq!(run(16, &|x| Ok(x.parallel_sum_scatter_tiled(axes, 0)?)), scatter);
            assert_eq!(run(16, &|x| Ok(x.parallel_all_to_all_tiled(axes, 0, 0)?)), all_to_all);
            assert_eq!(run(4, &|x| Ok(x.parallel_all_gather_tiled(axes, 0)?)), gather.repeat(4));
        }
        let pairs = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        assert_eq!(run(4, &|x| Ok(x.parallel_permute(["x", "y"], pairs.clone())?)), vec![3.0, 0.0, 1.0, 2.0]);
        assert_eq!(run(4, &|x| Ok(x.parallel_permute(["y", "x"], pairs.clone())?)), vec![3.0, 2.0, 0.0, 1.0]);
    }

    #[test]
    fn test_to_mlir_module_for_program_with_inferred_input_shardings_executes_on_cpu() {
        use crate::experimental::tracing::{TracedProgram, trace_with_context};

        // The input sharding comes from the input type, which is sharded along `x`, and each device doubles its shard.
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace_with_context(
            |_, x: ArrayTracer| {
                Ok(shard_map_with_inferred_input_shardings(
                    |x: ArrayTracer| Ok(x.clone() + x),
                    x,
                    mesh.clone(),
                    sharded.clone(),
                    ShardMapOptions::new(),
                )?)
            },
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
        )
        .unwrap();
        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                      (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\"}]>] \
                        out_shardings=[<@mesh, [{\"x\"}]>] manual_axes={\"x\"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );
        assert_eq!(
            execute_on_two_cpu_devices(module, &[2], [&[1.0, 2.0], &[3.0, 4.0]]),
            vec![vec![2.0, 4.0], vec![6.0, 8.0]]
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_with_shard_map_along_executes_on_cpu() {
        use crate::experimental::tracing::{TracedProgram, trace_with_context};

        // Each device receives one row of the input mapped along `x` at dimension 0 and returns it doubled, placed along
        // `x` at dimension 1 of the output, so the rows of the input become the column blocks of the output.
        let mesh = test_manual_mesh("x", 2);
        let traced: TracedProgram<ArrayType, ArrayType> = trace_with_context(
            |_, x: ArrayTracer| {
                Ok(shard_map_along(|x: ArrayTracer| Ok(x.clone() + x), x, mesh.clone(), "x", Some(0), Some(1))?)
            },
            ArrayType::new_static(DataType::F32, [2, 3])
                .with_sharding(
                    Sharding::new(
                        mesh.clone(),
                        vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                    )
                    .unwrap(),
                )
                .unwrap(),
        )
        .unwrap();
        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) -> \
                      (tensor<1x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {\"x\"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\"}, {}]>] \
                        out_shardings=[<@mesh, [{}, {\"x\"}]>] manual_axes={\"x\"} (%arg1: tensor<1x3xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<1x3xf32>
                      sdy.return %1 : tensor<1x3xf32>
                    } : (tensor<2x3xf32>) -> tensor<1x6xf32>
                    return %0 : tensor<1x6xf32>
                  }
                }
            "},
        );
        assert_eq!(
            execute_on_two_cpu_devices(module, &[1, 3], [&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]),
            vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]]
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_without_manual_variation_passes_unreduced_summands_between_maps_on_cpu() {
        use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput};

        use crate::experimental::tracing::{TracedProgram, trace_with_context};

        // The first map returns each device's shard of `x = [1, 2]` as its summand of an unreduced output, and the
        // second map squares each summand of its unreduced input. The summands survive between the maps, so the
        // devices return `[1]` and `[4]`, whose sum is `5`, while the canonical decomposition of their sum (`[3]` on
        // the first device and `[0]` on the second) would give `9`.
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let options = ShardMapOptions::new().with_check_manual_variation(false);
        let traced: TracedProgram<ArrayType, ArrayType> = trace_with_context(
            |_, x: ArrayTracer| {
                let summands = shard_map(
                    |x: ArrayTracer| Ok(x),
                    x,
                    mesh.clone(),
                    sharded.clone(),
                    unreduced.clone(),
                    options.clone(),
                )?;
                Ok(shard_map(
                    |summand: ArrayTracer| Ok(summand.clone() * summand),
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
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}], unreduced={\"x\"}> : tensor<1xf32>
                    %2 = sdy.manual_computation(%1) in_shardings=[<@mesh, [{}], unreduced={\"x\"}>] \
                        out_shardings=[<@mesh, [{}], unreduced={\"x\"}>] manual_axes={\"x\"} (%arg1: tensor<1xf32>) {
                      %3 = stablehlo.multiply %arg1, %arg1 : tensor<1xf32>
                      sdy.return %3 : tensor<1xf32>
                    } : (tensor<1xf32>) -> tensor<1xf32>
                    return %2 : tensor<1xf32>
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
    fn test_eager_shard_map_without_manual_variation_passes_unreduced_summands_between_maps_on_cpu() {
        // The pipeline of the previous test, with each map applied eagerly to concrete arrays through the frontend:
        // the first map returns each device's shard of `x = [1, 2]` as its summand of an unreduced output, and the
        // second map squares each summand of that unreduced input. The devices return `[1]` and `[4]`, whose sum (`5`)
        // host materialization recovers.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let logical_mesh = test_manual_mesh("x", 2);
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let unreduced = Sharding::replicated(logical_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let options = ShardMapOptions::new().with_check_manual_variation(false);
        let input_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharded.clone()).unwrap();
        let input = Array::from_host_buffer(&domain, input_type, mesh, values_to_bytes(&[1f32, 2.])).unwrap();
        let summands =
            shard_map(|x| Ok(x), input, logical_mesh.clone(), sharded, unreduced.clone(), options.clone()).unwrap();
        let squares = shard_map(
            |summand: ProjectedValue<ArrayType, DomainTracer<Domain<'_>>>| Ok(summand.clone() * summand),
            summands,
            logical_mesh,
            unreduced.clone(),
            unreduced.clone(),
            options,
        )
        .unwrap();
        assert_eq!(squares.r#type().sharding(), Some(&unreduced));
        assert_eq!(read_device_f32s(&squares), vec![vec![1.0], vec![4.0]]);
        let materialized = crate::arrays_v0::host::materialize_dense_array_bytes(&squares).unwrap();
        assert_eq!(values_from_bytes::<f32>(materialized.as_slice()), vec![5.0]);
    }

    #[test]
    fn test_eager_shard_map_functions_on_cpu() {
        // The frontend functions apply to concrete arrays of the XLA backend, which executes each map eagerly. A sharded
        // input meets a replicated one, the input shardings can be inferred from the input types, and a map along `x`
        // places the dimension that its axes select. Repeating a map at the same signature hits the dispatch cache.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let logical_mesh = test_manual_mesh("x", 2);
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let upload = |sharding: &Sharding, values: &[f32]| {
            let r#type = ArrayType::new_static(DataType::F32, [values.len()]).with_sharding(sharding.clone()).unwrap();
            Array::from_host_buffer(&domain, r#type, mesh.clone(), values_to_bytes(values)).unwrap()
        };
        let lhs = upload(&sharded, &[1.0, 2.0, 3.0, 4.0]);
        let rhs = upload(&replicated, &[10.0, 20.0]);
        let add = |lhs, rhs| {
            shard_map(
                |(lhs, rhs): (
                    ProjectedValue<ArrayType, DomainTracer<Domain<'_>>>,
                    ProjectedValue<ArrayType, DomainTracer<Domain<'_>>>,
                )| Ok(lhs + rhs),
                (lhs, rhs),
                logical_mesh.clone(),
                (sharded.clone(), replicated.clone()),
                sharded.clone(),
                ShardMapOptions::new(),
            )
            .unwrap()
        };
        let sum = add(lhs.clone(), rhs.clone());
        assert_eq!(sum.r#type().sharding(), Some(&sharded));
        assert_eq!(read_device_f32s(&sum), vec![vec![11.0, 22.0], vec![13.0, 24.0]]);
        let program_count = domain.program_count();
        assert_eq!(read_device_f32s(&add(lhs.clone(), rhs.clone())), vec![vec![11.0, 22.0], vec![13.0, 24.0]]);
        assert_eq!(domain.program_count(), program_count);

        let inferred = shard_map_with_inferred_input_shardings(
            |(lhs, rhs): (
                ProjectedValue<ArrayType, DomainTracer<Domain<'_>>>,
                ProjectedValue<ArrayType, DomainTracer<Domain<'_>>>,
            )| Ok(lhs * rhs),
            (lhs.clone(), rhs),
            logical_mesh.clone(),
            sharded.clone(),
            ShardMapOptions::new(),
        )
        .unwrap();
        assert_eq!(read_device_f32s(&inferred), vec![vec![10.0, 40.0], vec![30.0, 80.0]]);

        let doubled = shard_map_along(
            |x: ProjectedValue<ArrayType, DomainTracer<Domain<'_>>>| Ok(x.clone() + x),
            lhs,
            logical_mesh,
            "x",
            Some(0),
            Some(0),
        )
        .unwrap();
        assert_eq!(doubled.r#type().sharding(), Some(&sharded));
        assert_eq!(read_device_f32s(&doubled), vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    }

    #[test]
    fn test_eager_shard_map_uses_the_default_mesh_of_its_context_on_cpu() {
        // An eager map over `x` of a `2x2` mesh runs a nested map without a mesh, which is defined over the mesh of the
        // enclosing map (its context's default mesh) and makes the remaining axis `y` manual, as JAX's
        // `shard_map(mesh=None)` does inside a `shard_map` body. Outside of every manual region, a map needs a mesh.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let input =
            Array::from_host_buffer(&domain, r#type.clone(), mesh.clone(), values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        let doubled = shard_map(
            |x: ProjectedValue<ArrayType, DomainTracer<Domain<'_>>>| {
                Ok(shard_map(
                    |x: ProjectedValue<ArrayType, DomainTracer<Domain<'_>>>| Ok(x.clone() + x),
                    x,
                    None,
                    replicated.clone(),
                    replicated.clone(),
                    ShardMapOptions::new(),
                )?)
            },
            input.clone(),
            logical_mesh,
            sharded.clone(),
            sharded.clone(),
            ShardMapOptions::new().with_manual_axes(["x"]),
        )
        .unwrap();
        assert_eq!(read_device_f32s(&doubled), vec![vec![2.0, 4.0], vec![2.0, 4.0], vec![6.0, 8.0], vec![6.0, 8.0]]);
        let missing = shard_map(
            |x: ProjectedValue<ArrayType, DomainTracer<Domain<'_>>>| Ok(x),
            input,
            None,
            sharded.clone(),
            sharded.clone(),
            ShardMapOptions::new(),
        );
        assert!(matches!(missing, Err(ShardMapError::MissingMesh)));

        // A domain that was given a device mesh owns its logical view as the default mesh, so a top-level map in it
        // can omit the mesh, as under JAX's `jax.set_mesh`.
        let meshed_domain = domain.with_mesh(mesh.clone());
        let input =
            Array::from_host_buffer(&meshed_domain, r#type, mesh, values_to_bytes(&[1f32, 2., 3., 4.])).unwrap();
        let doubled = shard_map(
            |x: ProjectedValue<ArrayType, DomainTracer<Domain<'_>>>| Ok(x.clone() + x),
            input,
            None,
            sharded.clone(),
            sharded,
            ShardMapOptions::new().with_manual_axes(["x"]),
        )
        .unwrap();
        assert_eq!(read_device_f32s(&doubled), vec![vec![2.0, 4.0], vec![2.0, 4.0], vec![6.0, 8.0], vec![6.0, 8.0]]);
    }

    #[test]
    fn test_shard_map_batched_along_every_manual_axis_executes_for_callers_over_explicit_meshes_on_cpu() {
        // The caller types its input over an `Explicit` copy of the mesh of the map, which differs from it only in axis
        // types, and maps its leading dimension along `x`, the only active manual axis of the map. Batching releases `x`,
        // so the map has no active manual axes, and its outputs are typed over the `Manual` mesh of the map, which the
        // caller's device mesh accepts because both meshes have the same topology. Each device holds one doubled item.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let map_mesh = test_manual_mesh("x", 2);
        let caller_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            caller_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let domain = Session::new(&client).domain();
        let replicated = Sharding::replicated(map_mesh.clone(), 1);
        let input_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(
                Sharding::new(caller_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |input| {
                let output = batch(
                    |item| {
                        let item = ValueProjection::<ArrayType>::into_projected(item)?;
                        let output = shard_map(
                            |x| Ok(x.clone() + x),
                            item,
                            map_mesh.clone(),
                            replicated.clone(),
                            replicated.clone(),
                            ShardMapOptions::new(),
                        )?;
                        Ok(output.into_value())
                    },
                    input.into_value(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap();
                ValueProjection::<ArrayType>::into_projected(output).unwrap()
            },
            input_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled = staged.lower(&domain).unwrap().compile(&domain).unwrap();
        let input =
            Array::from_host_buffer(&domain, input_type, mesh, values_to_bytes(&[1f32, 2., 3., 4., 5., 6.])).unwrap();
        let Value::Array(output) = compiled.executable().execute(&domain, Value::Array(input)).unwrap() else {
            panic!("the compiled function returned a dimension");
        };
        let items = output
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(items, vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]]);
    }

    #[test]
    fn test_shard_map_batched_along_a_used_manual_axis_executes_with_gathered_batches_on_cpu() {
        // The caller maps the items `[1, 2, 3, 4]` along `b`, which its input places on `x`, through a map over `x` whose
        // body reduces over both `b` and `x`. The body uses `x`, so the batch is gathered along `x` on entry, every
        // device computes `sum_b(a) + sum_x(a) = 10 + 2 * a` for every item, and each device keeps the block of items
        // that the output sharding assigns to it along `x`. Splitting the batch along `x` instead (as JAX's
        // `spmd_axis_name` does) would make the reduction over `x` add different items.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let map_mesh = test_manual_mesh("x", 2);
        let caller_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            caller_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let domain = Session::new(&client).domain();
        let replicated = Sharding::replicated(map_mesh.clone(), 0);
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::new(caller_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |input| {
                let output = batch(
                    |item| {
                        let item = ValueProjection::<ArrayType>::into_projected(item)?;
                        let output = shard_map(
                            |a| {
                                Ok(a.parallel_reduce(ReductionKind::Sum, "b")?
                                    + a.parallel_reduce(ReductionKind::Sum, "x")?)
                            },
                            item,
                            map_mesh.clone(),
                            replicated.clone(),
                            replicated.clone(),
                            ShardMapOptions::new(),
                        )?;
                        Ok(output.into_value())
                    },
                    input.into_value(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("b"),
                )
                .unwrap();
                ValueProjection::<ArrayType>::into_projected(output).unwrap()
            },
            input_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled = staged.lower(&domain).unwrap().compile(&domain).unwrap();
        let input = Array::from_host_buffer(&domain, input_type, mesh, values_to_bytes(&[1f32, 2., 3., 4.])).unwrap();
        let Value::Array(output) = compiled.executable().execute(&domain, Value::Array(input)).unwrap() else {
            panic!("the compiled function returned a dimension");
        };
        let blocks = output
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(blocks, vec![vec![12.0, 14.0], vec![16.0, 18.0]]);
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
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    |input: ArrayTracer| Ok(input.clone() + input),
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
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], unreduced={\"x\"}> : tensor<1xf32>
                    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{}], unreduced={\"x\"}>] \
                        out_shardings=[<@mesh, [{}], unreduced={\"x\"}>] manual_axes={\"x\"} (%arg1: tensor<1xf32>) {
                      %2 = stablehlo.add %arg1, %arg1 : tensor<1xf32>
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
    fn test_shard_map_compilation_requires_one_mesh_topology_on_cpu() {
        // Two inputless maps each read the coordinate of every device along `x` and tile it along `x`. The module
        // declares one mesh, which assigns device ids to coordinates, while `axis_index` computes coordinates from the
        // axis order of the mesh of its own map, so maps over meshes with other axis orders must not share a module.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let mesh = test_logical_mesh_2x2();
        let reordered_mesh = LogicalMesh::new(vec![
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let device_mesh = |logical_mesh: &LogicalMesh| {
            let devices = client.addressable_devices().unwrap();
            DeviceMesh::new(
                logical_mesh.clone(),
                devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
            )
            .unwrap()
        };
        let coordinate_map = |mesh: &LogicalMesh| {
            let mut builder = ArrayIrProgramBuilder::new();
            let coordinate = builder
                .add_instruction(AxisIndexOperation::new("x".into()).with_mesh(mesh.clone()), vec![], vec![], None)
                .unwrap()[0];
            let coordinate = builder
                .add_instruction(
                    ReshapeOperation::new(Shape::new(vec![]), Shape::new(vec![CoreDimension::Static(1)])),
                    vec![],
                    vec![coordinate],
                    None,
                )
                .unwrap()[0];
            let body =
                builder.build::<Vec<Constant>, Vec<Constant>>(vec![coordinate], vec![], vec![Placeholder]).unwrap();
            let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let boundary = ShardMapBoundary::new(mesh.clone(), vec![], vec![output_sharding], vec![]).unwrap();
            (ShardMapOperation::from_program(&body, Vec::<ArrayIrType>::new(), boundary).unwrap(), body)
        };
        let input_type = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let output_type = ArrayType::new_static(DataType::U64, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let compile_maps = |second_mesh: &LogicalMesh, device_mesh: DeviceMesh| {
            let (first, first_body) = coordinate_map(&mesh);
            let (second, second_body) = coordinate_map(second_mesh);
            let output_type = output_type.clone();
            crate::compile::<_, ArrayType, (ArrayType, ArrayType)>(
                move |input| {
                    let context = input.into_value().domain();
                    let replicate = |output| {
                        let operation = BroadcastOperation::new(output_type.clone(), vec![0]);
                        let output = context.bind(operation, vec![], &[output]).unwrap().remove(0);
                        ValueProjection::<ArrayType>::into_projected(output).unwrap()
                    };
                    let first = context.bind(first, vec![first_body], &[]).unwrap().remove(0);
                    let second = context.bind(second, vec![second_body], &[]).unwrap().remove(0);
                    (replicate(first), replicate(second))
                },
                input_type.clone(),
                &domain,
                device_mesh,
            )
        };

        // Maps over meshes with the same axis order agree on the coordinates of every device.
        let compiled = compile_maps(&mesh, device_mesh(&mesh)).unwrap();
        let input =
            Array::from_host_buffer(&domain, input_type.clone(), device_mesh(&mesh), 0f32.to_ne_bytes()).unwrap();
        let (first, second) = compiled.executable().execute(&domain, input).unwrap();
        assert_eq!(values_from_bytes::<u64>(&first.copy_to_host().unwrap()), vec![0, 1]);
        assert_eq!(values_from_bytes::<u64>(&second.copy_to_host().unwrap()), vec![0, 1]);

        // A map over the reordered mesh is rejected instead of computing coordinates that do not match the device ids.
        assert!(matches!(
            compile_maps(&reordered_mesh, device_mesh(&mesh)),
            Err(Error::Lowering(crate::LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == mesh && actual == reordered_mesh,
        ));

        // A program whose module mesh is not the topology of the device mesh that it compiles for is rejected too.
        assert!(matches!(
            compile_maps(&mesh, device_mesh(&reordered_mesh)),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Sharding(ryft_core::ShardingError::MeshMismatch { expected, actual })
                    if expected == &reordered_mesh && actual == &mesh,
            ),
        ));
    }

    #[test]
    fn test_shard_map_compilation_replicates_unplaced_outputs_on_cpu() {
        // An inputless map reads the coordinate of every device along `x` and tiles it along `x`. Broadcasting its
        // output to a type without a sharding leaves the program output without a placement, which execution treats as
        // replicated, so lowering must annotate the result as replicated as well. Otherwise, the compiler keeps the
        // result tiled like the output of the map, and every device returns one element instead of both. The first
        // output has no placement and the second one is replicated explicitly.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh = || {
            let devices = client.addressable_devices().unwrap();
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let coordinate = builder
            .add_instruction(AxisIndexOperation::new("x".into()).with_mesh(mesh.clone()), vec![], vec![], None)
            .unwrap()[0];
        let coordinate = builder
            .add_instruction(
                ReshapeOperation::new(Shape::new(vec![]), Shape::new(vec![CoreDimension::Static(1)])),
                vec![],
                vec![coordinate],
                None,
            )
            .unwrap()[0];
        let body = builder.build::<Vec<Constant>, Vec<Constant>>(vec![coordinate], vec![], vec![Placeholder]).unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let boundary = ShardMapBoundary::new(mesh.clone(), vec![], vec![output_sharding], vec![]).unwrap();
        let map = ShardMapOperation::from_program(&body, Vec::<ArrayIrType>::new(), boundary).unwrap();
        let input_type = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let unplaced_type = ArrayType::new_static(DataType::U64, [2]);
        let replicated_type = unplaced_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let compiled = crate::compile::<_, ArrayType, (ArrayType, ArrayType)>(
            move |input| {
                let context = input.into_value().domain();
                let coordinates = context.bind(map, vec![body], &[]).unwrap().remove(0);
                let broadcast = |output_type| {
                    let operation = BroadcastOperation::new(output_type, vec![0]);
                    let output = context.bind(operation, vec![], std::slice::from_ref(&coordinates)).unwrap().remove(0);
                    ValueProjection::<ArrayType>::into_projected(output).unwrap()
                };
                (broadcast(unplaced_type), broadcast(replicated_type))
            },
            input_type.clone(),
            &domain,
            device_mesh(),
        )
        .unwrap();
        let input = Array::from_host_buffer(&domain, input_type, device_mesh(), 0f32.to_ne_bytes()).unwrap();
        let (unplaced, replicated) = compiled.executable().execute(&domain, input).unwrap();
        assert_eq!(values_from_bytes::<u64>(&unplaced.copy_to_host().unwrap()), vec![0, 1]);
        assert_eq!(values_from_bytes::<u64>(&replicated.copy_to_host().unwrap()), vec![0, 1]);
    }

    #[test]
    fn test_to_mlir_module_for_program_renders_and_executes_matmul_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(8), ..Default::default() }))
            .expect("failed to create 8-device CPU client");
        let domain = Session::new(&client).domain();
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
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8), CoreDimension::Static(4)])),
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4), CoreDimension::Static(2)])),
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

        let lhs_array = Array::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[8, 4], lhs_sharding.clone()),
            device_mesh.clone(),
            lhs_buffers,
        )
        .unwrap();
        let rhs_array = Array::from_addressable_buffers(
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
            Array::into_execute_arguments(vec![lhs_array, rhs_array], execution_device_ids.as_slice()).unwrap();
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
        let domain = Session::new(&client).domain();
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
            &Shape::new(vec![CoreDimension::Static(2)]),
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

        let input_array = Array::from_addressable_buffers(
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
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
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
