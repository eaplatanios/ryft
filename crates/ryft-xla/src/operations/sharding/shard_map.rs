use std::collections::HashSet;

use ryft_core::{
    ArrayType, PARALLEL_ALL_GATHER_OPERATION_NAME, ParallelAllGatherOperation, ParallelAllGatherOutputVariance,
    Parameterized, ProgramError, SHARD_MAP_OPERATION_NAME, ShardMapBoundary, ShardMapError, ShardMapOperation,
    ShardMapValueKind, Sharding, ShardingDimension, Typed, TypedProgram,
};
use ryft_mlir::dialects::shardy;
use ryft_mlir::dialects::shardy::{
    DimensionShardingAttributeRef, TensorShardingAttributeRef, TensorShardingPerValueAttributeRef,
};
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Type as MlirType,
    Value as MlirValue, ValueRef,
};

use crate::lowering::regions::lower_program_outputs_with_inputs;
use crate::lowering::types::lower_tensor_type;
use crate::lowering::{CollectiveLoweringState, EffectTokens, current_or_new_token, token_threaded_effects};
use crate::operations::Program;
use crate::operations::control_flow::lowered_effect_classes;
use crate::{
    Constant, Error, LowerableOperation, LowerableOperationFamily, Lowerer, SHARDY_MESH_SYMBOL_NAME, ToMlir, Type,
};

impl LowerableOperation for ShardMapOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let [body] = lowerer.regions() else {
            return Err(Error::from(ProgramError::MalformedProgram(format!(
                "`{SHARD_MAP_OPERATION_NAME}` expected 1 attached region but got {}",
                lowerer.regions().len(),
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
        let (mut block, context, location) = (lowerer.block(), lowerer.context(), lowerer.location());
        let (effect_tokens, _, _, _, collective_state) = lowerer.split_effect_tokens();
        lower_manual_computation(
            &mut block,
            input_values,
            self.boundary(),
            &simplified_body,
            local_input_types.as_slice(),
            global_output_types.as_slice(),
            context,
            location,
            collective_state,
            effect_tokens,
        )
    }
}

/// Lowers one `sdy.manual_computation` operation, threading per-device effect tokens across its boundary. Assertions
/// remain local to an executable, and the only ordered I/O that reaches this function is per-device (i.e.,
/// `DeviceOrderedIo`), because core `shard_map` boundary validation rejects bodies with the `OrderedIo` effect (refer
/// to [`ShardMapError::OrderedIoNotSupported`]).
#[allow(clippy::too_many_arguments)]
fn lower_manual_computation<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
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
        boundary.manual_axes().iter().find(|axis| collective_state.bound_manual_axes().contains(*axis))
    {
        return Err(ProgramError::from(ShardMapError::AxisAlreadyManual { axis_name: axis_name.clone() }).into());
    }
    // For the same reason, an input or output sharding must not name such an axis, so it is rejected here rather than
    // silently dropped from the Shardy shardings. The closure entry points reject these shardings too.
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
                .find(|axis_name| collective_state.bound_manual_axes().contains(*axis_name));
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

    // `OrderedIo` bodies never reach this point, because `shard_map` boundary validation rejects them when a body is
    // traced or attached (refer to `ShardMapError::OrderedIoNotSupported`), so only `DeviceOrderedIo` and unordered
    // effects are threaded through the manual computation.
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
    let enclosing_manual_axes = collective_state.bound_manual_axes();
    let mut input_shardings = shardy_tensor_sharding_per_value(
        boundary.input_shardings(),
        boundary.manual_axes(),
        enclosing_manual_axes,
        context,
    )?
    .shardings();
    let mut output_shardings = shardy_tensor_sharding_per_value(
        boundary.output_shardings(),
        boundary.manual_axes(),
        enclosing_manual_axes,
        context,
    )?
    .shardings();
    // Shardy rejects a manual computation whose result drops an unreduced axis of an input (e.g., a map that completes
    // the pending sum of its unreduced input) unless a blessed operation marks that input unreduced, so every input
    // with unreduced axes passes through a sharding constraint, as the output resharding of JAX's lowering emits.
    for (index, sharding) in boundary.input_shardings().iter().enumerate() {
        if !sharding.unreduced_axes().is_empty() {
            let operation = shardy::sharding_constraint(manual_inputs[index], input_shardings[index], location)?;
            let constraint = block.append_operation(operation)?;
            manual_inputs[index] = constraint.result(0).unwrap().as_ref();
        }
    }
    if !threaded_effects.is_empty() {
        let token_type = context.stable_hlo_token_type()?.as_ref();
        // Tokens are rank-0 and have no replicated or unreduced axes. Each device carries its own token, without
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
        // Callees inside the shard-local body always inline. The fresh-root body has no enclosing capture namespace.
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
        context.shardy_manual_axes(boundary.manual_axes())?,
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

/// Builds the typed Shardy per-value tensor sharding attribute of a `sdy.manual_computation` boundary (i.e., its
/// `in_shardings` or `out_shardings`), with one tensor sharding per entry of `shardings` (refer to
/// [`manual_computation_tensor_sharding`]). Axes in `enclosing_manual_axes` are already bound by enclosing manual
/// regions and are excluded.
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
    let mesh_symbol_reference = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
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
        mesh_symbol_reference,
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

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, Array as CoreArray, ArrayIrBatchingPolicy, ArrayIrOperation as CoreArrayIrOperation,
        ArrayIrValue, ArrayOperation, ArrayReference, ArrayReferenceTransform, ArrayReferenceTransformIndex, AtomId,
        AttentionConfiguration, AttentionImplementation, AttentionInputs, AxisIndex, AxisIndexOperation, BatchAxis,
        BatchAxisSpecification, BatchedProgram, BatchingContext, BroadcastOperation, CollectiveOptions,
        CompareOperation, ComparisonDirection, CompositeOperation, ConstantOperation, Context, ConvertElementType,
        ConvertElementTypeOperation, Cos, CotangentDestinationKind, CustomCallOperation, CustomRuleDefinition,
        CustomRuleRegistration, DataType, Device, DeviceMesh, Differentiate, Dimension, DimensionValue, DomainTracer,
        DomainTracingContext, Dot, DotDimensionNumbers, EagerContext, EffectClass, Error as CoreError, Exp, Fill,
        Gather, GatherMode, LogicalMesh, MeshAxis, MeshAxisType, MulOperation, NamedAxis, Operation as CoreOperation,
        OperationPayloadProjection, ParallelAllGather, ParallelAllToAllOperation, ParallelArgExtrema,
        ParallelBroadcast, ParallelCast, ParallelCastTarget, ParallelPermuteOperation, ParallelReduce,
        ParallelReduceOperation, ParallelVary, ParallelVaryOperation, Placeholder, Program as CoreProgram,
        ProgramBatchingOutputAxesPolicy, ProgramBuilder as CoreProgramBuilder, ProjectedContext, ProjectedValue,
        Reduce, ReduceOperation, ReductionKind, ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNew,
        ReferenceNewOperation, ReferenceRead, ReferenceReadOperation, ReferenceType, ReferenceWrite, Reshape,
        ReshapeOperation, Shape, ShardMap, ShardMapOptions, ShardingError, Sin, Sort, SortDirection, StagingContext,
        TracingScope, Type as CoreType, Typed, TypedProgram, UnreducedKind, Value as CoreValue, ValueProjection,
        WhileOperation, batch, composite, differentiable_dot_product_attention, differentiate_at, explicit_axes,
        shard_map, shard_map_along, shard_map_with_inferred_input_shardings,
    };
    use ryft_pjrt::{
        BufferType, Client, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput,
        Program as PjrtProgram,
    };

    use crate::lowering::tests::{
        lower_replicated_shard_map_body, shard_map_module, static_sharded_array_type, test_manual_mesh, test_sharding,
        test_vector_type,
    };
    use crate::operations::assertions::ensure_assertion_handler_registered;
    use crate::operations::collectives::parallel_reduce::tests::execute_parallel_reduce_on_cpu;
    use crate::operations::debugging::{
        PRINT_CUSTOM_CALL_TARGET, ensure_print_handler_registered, with_captured_prints,
    };
    use crate::operations::{
        FlatProgram, Operation, Operation as ArrayIrOperation, ProgramBuilder, ProgramBuilder as ArrayIrProgramBuilder,
    };
    use crate::sessions::tests::{read_sharded_f32_array, two_device_mesh};
    use crate::tests::{cpu_client, test_compilation_options, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayConstant, ArrayTracer, ExecutionArguments, FromPjrt, Options, Platform, Reference, Session,
        TracedProgram, Tracer, Value, compile_statefully, to_mlir_module_for_program, trace, trace_with_context,
    };

    use super::*;

    /// Returns the elements of the buffer that every addressable device holds for `array`, in device order.
    pub(crate) fn read_device_f32s(array: &Array) -> Vec<Vec<f32>> {
        array
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect()
    }

    /// Four-device mesh with the manual axes `x` and `y`.
    pub(crate) fn test_logical_mesh_2x2() -> LogicalMesh {
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
        let (_, body) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
                let y = ValueProjection::<ArrayType>::into_projected(inputs[0].clone()).map_err(ProgramError::from)?;
                Ok(vec![(y.clone() + y).into_value()])
            },
            vec![Type::Array(local_type)],
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

    /// Compiles `module` for `N` CPU devices, executes it with one `f32` buffer per device for every `(local_shape,
    /// device_inputs)` entry of `inputs`, holding `device_inputs[device]`, and returns the elements of the first output
    /// of every device.
    pub(crate) fn execute_inputs_on_cpu_devices<const N: usize>(
        module: String,
        inputs: &[(&[u64], [&[f32]; N])],
    ) -> Vec<Vec<f32>> {
        let client = cpu_client(N);
        ensure_assertion_handler_registered(client).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, Some(N)))
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

    /// Builds a nonlinear reference body with reads and products on each device. Its derivatives require saved values
    /// from both operands, so different shard values must survive the boundary between the primal and the tangent.
    fn nonlinear_reference_shard_map_program(
        mesh: &LogicalMesh,
        replicated_reference: bool,
    ) -> CoreProgram<Constant, Operation, Vec<Constant>, Vec<Constant>> {
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let reference_sharding =
            if replicated_reference { Sharding::replicated(mesh.clone(), 1) } else { sharded.clone() };
        let reference_type = ArrayType::new_static(DataType::F32, [if replicated_reference { 2 } else { 4 }])
            .with_sharding(reference_sharding.clone())
            .unwrap();
        let value_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let boundary = ShardMapBoundary::new(
            mesh.clone(),
            vec![reference_sharding.clone(), sharded.clone()],
            vec![sharded.clone()],
            vec!["x".to_string()],
        )
        .unwrap();
        let mut body = CoreProgramBuilder::<Constant, Operation>::new();
        let reference =
            body.add_input(ReferenceType::new(boundary.local_input_type(0, &reference_type).unwrap()).into());
        let value = body.add_input(boundary.local_input_type(1, &value_type).unwrap().into());
        let state = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let state = if replicated_reference {
            body.add_instruction(ParallelVaryOperation::new("x".to_string()), Vec::new(), vec![state], None)
                .unwrap()[0]
        } else {
            state
        };
        let output = body.add_instruction(MulOperation::new(), Vec::new(), vec![state, value], None).unwrap()[0];
        let body = body
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let inputs = vec![Type::Reference(ReferenceType::new(reference_type)), Type::Array(value_type)];
        let operation = ShardMapOperation::from_program(&body, inputs.clone(), boundary).unwrap();
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let inputs = inputs.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let body = builder.import_program(body);
        let outputs =
            builder.add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], inputs, None).unwrap();
        let outputs = outputs.to_vec();
        builder.build(outputs, vec![Placeholder; 2], vec![Placeholder]).unwrap()
    }

    /// Builds the program `r -> shard_map(|r| r += read(r))(r)` for an unchecked map over a mesh of two devices whose
    /// reference input is reduced along the manual axis `x`, together with its referent type.
    pub(crate) fn doubling_reduced_reference_program(
        logical_mesh: &LogicalMesh,
    ) -> (CoreProgram<Constant, Operation, Vec<Constant>, Vec<Constant>>, ArrayType) {
        let reduced = Sharding::replicated(logical_mesh.clone(), 1).with_reduced_axes(["x"]).unwrap();
        let boundary = ShardMapBoundary::new(logical_mesh.clone(), vec![reduced.clone()], Vec::new(), Vec::new())
            .unwrap()
            .with_check_manual_variation(false);
        let referent_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(reduced).unwrap();
        let reference_type = Type::Reference(ReferenceType::new(referent_type.clone()));
        let mut body = CoreProgramBuilder::<Constant, Operation>::new();
        let local_type = boundary.local_input_type(0, &referent_type).unwrap();
        let reference = body.add_input(Type::Reference(ReferenceType::new(local_type)));
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        body.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let body = body.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let operation = ShardMapOperation::from_program(&body, vec![reference_type.clone()], boundary).unwrap();
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let reference = builder.add_input(reference_type);
        let body = builder.import_program(body);
        builder
            .add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], vec![reference], None)
            .unwrap();
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        (program, referent_type)
    }

    /// Kind of `shard_map` program that [`shard_map_differential_program`] builds to compare the reference backend's
    /// emulation of `shard_map` with its execution through XLA.
    #[derive(Copy, Clone, Debug)]
    enum ShardMapDifferentialCase {
        /// Tiled identity of an input sharded along the manual axis `x`, together with its sum across `x`.
        IdentityAndSum,

        /// Coordinate of every device along the manual axis `x`.
        AxisIndex,

        /// Invariant tiled all-gather across the manual axis `x`.
        AllGather,

        /// Tiled all-to-all across the manual axis `x` that moves a row-sharded matrix to a column-sharded one.
        AllToAll,

        /// Permutation across the manual axis `x` that sends the shard of device 0 to device 1 and zeros to device 0.
        Permute,

        /// Map over the manual axis `x` whose body maps over the manual axis `y` and sums across the enclosing axis
        /// `x`.
        Nested,

        /// Map that adds its shard of an update into its shard of a caller-allocated reference and reads it back.
        Reference,

        /// Map that reads a reference replicated along the manual axis `x` at the coordinate of every device along `x`.
        VaryingIndexReference,

        /// Map that doubles a reference replicated along the manual axis `x` on every device and reads it back.
        ReplicatedReference,
    }

    /// Builds a program over `input_types` whose outputs are the atoms that `build` adds to it.
    fn build_differential_program<V: CoreValue<Type = Type>, O: CoreOperation<Type = Type>>(
        input_types: Vec<Type>,
        build: impl FnOnce(&mut CoreProgramBuilder<V, O>, &[AtomId]) -> Vec<AtomId>,
    ) -> CoreProgram<V, O, Vec<V>, Vec<V>> {
        let mut builder = CoreProgramBuilder::<V, O>::new();
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = build(&mut builder, &inputs);
        let output_count = outputs.len();
        builder.build(outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_count]).unwrap()
    }

    /// Builds the program of `case` over any operation family `O` that lifts the array IR family over `A`, so that the
    /// reference backend's emulation over [`CoreArrayIrOperation<Array>`] and XLA execution over [`Operation`] run
    /// the same program. Every program input and output is an array, because the reference of
    /// [`ShardMapDifferentialCase::Reference`] is allocated from a program input and its final value is frozen into a
    /// trailing program output.
    fn shard_map_differential_program<A, V, O>(case: ShardMapDifferentialCase) -> CoreProgram<V, O, Vec<V>, Vec<V>>
    where
        A: CoreValue<Type = ArrayType>,
        V: CoreValue<Type = Type>,
        O: CoreOperation<Type = Type> + OperationPayloadProjection + From<CoreArrayIrOperation<A>>,
    {
        let array_operation = CoreArrayIrOperation::<A>::from;
        let mesh = match case {
            ShardMapDifferentialCase::Nested => LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
                MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            ]),
            _ => LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]),
        }
        .unwrap();
        let sharded = |axis: &str| Sharding::new(mesh.clone(), vec![ShardingDimension::sharded([axis])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let f32_type = |shape: &[usize], sharding: &Sharding| {
            ArrayType::new_static(DataType::F32, shape.to_vec()).with_sharding(sharding.clone()).unwrap()
        };
        let sum_over_x = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
        let (boundary, global_input_types, body) = match case {
            ShardMapDifferentialCase::IdentityAndSum => {
                let boundary = ShardMapBoundary::new(
                    mesh.clone(),
                    vec![sharded("x")],
                    vec![sharded("x"), replicated],
                    vec!["x".into()],
                )
                .unwrap();
                let global_type = f32_type(&[4], &sharded("x"));
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                    let sum = array_operation(ArrayOperation::ParallelReduce(sum_over_x));
                    vec![inputs[0], builder.add_instruction(sum, Vec::new(), vec![inputs[0]], None).unwrap()[0]]
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::AxisIndex => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![sharded("x")], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[2], &sharded("x"));
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, _| {
                    let index = AxisIndexOperation::new("x".to_string()).with_mesh(mesh.clone());
                    let index = array_operation(ArrayOperation::AxisIndex(index));
                    let index = builder.add_instruction(index, Vec::new(), Vec::new(), None).unwrap()[0];
                    let reshape = ReshapeOperation::new(Shape::new(Vec::new()), vec![1]);
                    let reshape = array_operation(ArrayOperation::Reshape(reshape));
                    builder.add_instruction(reshape, Vec::new(), vec![index], None).unwrap().to_vec()
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::AllGather => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![replicated], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[4], &sharded("x"));
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                    let gather = ParallelAllGatherOperation::new(
                        "x".to_string(),
                        2,
                        0,
                        CollectiveOptions::tiled(),
                        ParallelAllGatherOutputVariance::Invariant,
                    );
                    let gather = array_operation(ArrayOperation::ParallelAllGather(gather.with_mesh(mesh.clone())));
                    builder.add_instruction(gather, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::AllToAll => {
                let rows =
                    Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]);
                let columns =
                    Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])]);
                let (rows, columns) = (rows.unwrap(), columns.unwrap());
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![rows.clone()], vec![columns], vec!["x".into()]).unwrap();
                let global_type = f32_type(&[2, 4], &rows);
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                    let exchange = ParallelAllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::tiled());
                    let exchange = array_operation(ArrayOperation::ParallelAllToAll(exchange.with_mesh(mesh.clone())));
                    builder.add_instruction(exchange, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::Permute => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![sharded("x")], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[4], &sharded("x"));
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                    let permute = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]);
                    let permute = array_operation(ArrayOperation::ParallelPermute(permute.with_mesh(mesh.clone())));
                    builder.add_instruction(permute, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::Nested => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![replicated], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[4], &sharded("x"));
                let outer_local_type = boundary.local_input_type(0, &global_type).unwrap();
                let inner_boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("y")], vec![sharded("y")], vec!["y".into()])
                        .unwrap();
                let inner_local_type = inner_boundary.local_input_type(0, &outer_local_type).unwrap();
                let inner_body =
                    build_differential_program::<V, O>(vec![inner_local_type.into()], |builder, inputs| {
                        let sum = array_operation(ArrayOperation::ParallelReduce(sum_over_x));
                        builder.add_instruction(sum, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
                    });
                let inner_operation =
                    ShardMapOperation::from_program(&inner_body, vec![outer_local_type.clone().into()], inner_boundary)
                        .unwrap();
                let body = build_differential_program::<V, O>(vec![outer_local_type.into()], |builder, inputs| {
                    let inner_body = builder.import_program(inner_body);
                    let inner_operation = CoreArrayIrOperation::<A>::from(inner_operation);
                    builder.add_instruction(inner_operation, vec![inner_body], inputs.to_vec(), None).unwrap().to_vec()
                });
                (boundary, vec![global_type.into()], body)
            }
            ShardMapDifferentialCase::Reference => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![sharded("x"); 2], vec![sharded("x")], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[4], &sharded("x"));
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let input_types = vec![ReferenceType::new(local_type.clone()).into(), local_type.into()];
                let body = build_differential_program::<V, O>(input_types, |builder, inputs| {
                    let add_update = ReferenceAddUpdateOperation::<ArrayType, Type, ArrayReferenceTransform>::new();
                    let add_update = CoreArrayIrOperation::<A>::from(add_update);
                    builder.add_instruction(add_update, Vec::new(), inputs.to_vec(), None).unwrap();
                    let read = ReferenceReadOperation::<ArrayType, Type, ArrayReferenceTransform>::new();
                    let read = CoreArrayIrOperation::<A>::from(read);
                    builder.add_instruction(read, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
                });
                (boundary, vec![ReferenceType::new(global_type.clone()).into(), global_type.into()], body)
            }
            ShardMapDifferentialCase::VaryingIndexReference => {
                let table_sharding = Sharding::replicated(mesh.clone(), 2);
                let boundary = ShardMapBoundary::new(
                    mesh.clone(),
                    vec![table_sharding.clone()],
                    vec![sharded("x")],
                    vec!["x".into()],
                )
                .unwrap();
                let global_type = f32_type(&[2, 3], &table_sharding);
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let input_types = vec![ReferenceType::new(local_type).into()];
                let body = build_differential_program::<V, O>(input_types, |builder, inputs| {
                    let index = AxisIndexOperation::new("x".to_string()).with_mesh(mesh.clone());
                    let index = array_operation(ArrayOperation::AxisIndex(index));
                    let index = builder.add_instruction(index, Vec::new(), Vec::new(), None).unwrap()[0];
                    let read = ReferenceReadOperation::<ArrayType, Type, ArrayReferenceTransform>::new()
                        .with_transforms(vec![ArrayReferenceTransform::Index {
                            axis: 0,
                            index: ArrayReferenceTransformIndex::Dynamic,
                        }]);
                    let read = CoreArrayIrOperation::<A>::from(read);
                    builder.add_instruction(read, Vec::new(), vec![inputs[0], index], None).unwrap().to_vec()
                });
                (boundary, vec![ReferenceType::new(global_type).into()], body)
            }
            ShardMapDifferentialCase::ReplicatedReference => {
                let boundary =
                    ShardMapBoundary::new(mesh.clone(), vec![replicated.clone()], vec![sharded("x")], vec!["x".into()])
                        .unwrap();
                let global_type = f32_type(&[2], &replicated);
                let local_type = boundary.local_input_type(0, &global_type).unwrap();
                let input_types = vec![ReferenceType::new(local_type).into()];
                let body = build_differential_program::<V, O>(input_types, |builder, inputs| {
                    let read = ReferenceReadOperation::<ArrayType, Type, ArrayReferenceTransform>::new();
                    let read = CoreArrayIrOperation::<A>::from(read);
                    let state = builder.add_instruction(read.clone(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
                    let add_update = ReferenceAddUpdateOperation::<ArrayType, Type, ArrayReferenceTransform>::new();
                    let add_update = CoreArrayIrOperation::<A>::from(add_update);
                    builder.add_instruction(add_update, Vec::new(), vec![inputs[0], state], None).unwrap();
                    let state = builder.add_instruction(read, Vec::new(), vec![inputs[0]], None).unwrap()[0];
                    let vary = array_operation(ArrayOperation::ParallelVary(ParallelVaryOperation::new("x".into())));
                    builder.add_instruction(vary, Vec::new(), vec![state], None).unwrap().to_vec()
                });
                (boundary, vec![ReferenceType::new(global_type).into()], body)
            }
        };
        let operation = ShardMapOperation::from_program(&body, global_input_types, boundary).unwrap();

        // Reference inputs of the map are allocated from the corresponding program inputs and frozen after the map.
        let program_input_types = operation
            .global_input_types()
            .iter()
            .map(|r#type| match r#type {
                Type::Reference(reference) => Type::Array(reference.referent().clone()),
                r#type => r#type.clone(),
            })
            .collect();
        build_differential_program::<V, O>(program_input_types, |builder, inputs| {
            let mut map_inputs = inputs.to_vec();
            let mut references = Vec::new();
            for (input, r#type) in map_inputs.iter_mut().zip(operation.global_input_types()) {
                if r#type.is_reference() {
                    let new = CoreArrayIrOperation::<A>::from(ReferenceNewOperation::<ArrayType, Type>::new());
                    *input = builder.add_instruction(new, Vec::new(), vec![*input], None).unwrap()[0];
                    references.push(*input);
                }
            }
            let body = builder.import_program(body);
            let operation = CoreArrayIrOperation::<A>::from(operation);
            let mut outputs = builder.add_instruction(operation, vec![body], map_inputs, None).unwrap().to_vec();
            for reference in references {
                let freeze = CoreArrayIrOperation::<A>::from(ReferenceFreezeOperation::<ArrayType, Type>::new());
                outputs.push(builder.add_instruction(freeze, Vec::new(), vec![reference], None).unwrap()[0]);
            }
            outputs
        })
    }

    /// Runs the program that [`shard_map_differential_program`] builds for `case` through the reference backend's
    /// emulation and through XLA execution on all devices of `client`, which must match the devices of the mesh of
    /// `case`, over the same inputs with integer values, and checks that both produce exactly the same outputs,
    /// including their types.
    fn assert_shard_map_emulation_matches_execution(client: &'static Client<'static>, case: ShardMapDifferentialCase) {
        type ArrayMember = <Constant as ValueProjection<ArrayType>>::Projected;
        let emulated_program =
            shard_map_differential_program::<CoreArray, ArrayIrValue<CoreArray>, CoreArrayIrOperation<CoreArray>>(case);
        let xla_program = shard_map_differential_program::<ArrayMember, Constant, Operation>(case);
        let input_types = xla_program
            .input_types()
            .into_owned()
            .into_iter()
            .map(|r#type| match r#type {
                Type::Array(r#type) => r#type,
                r#type => panic!("differential programs have array inputs only but got `{type}`"),
            })
            .collect::<Vec<_>>();
        let input_bytes = input_types
            .iter()
            .enumerate()
            .map(|(index, r#type)| {
                let element_count = r#type.static_shape().unwrap().dimensions().iter().product::<usize>();
                let elements = (0..element_count).map(|element| (100 * index + element + 1) as f32).collect::<Vec<_>>();
                values_to_bytes::<f32>(&elements)
            })
            .collect::<Vec<_>>();

        // The reference backend emulates every device of the map on the host.
        let emulated_inputs = input_types
            .iter()
            .zip(&input_bytes)
            .map(|(r#type, bytes)| ArrayIrValue::Array(CoreArray::from_logical_bytes(r#type.clone(), bytes).unwrap()))
            .collect();
        let emulated_outputs = emulated_program.interpret(emulated_inputs).unwrap();

        // XLA compiles the map into one manual computation and executes it on the devices of `client`.
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(input_types[0].sharding().unwrap().mesh().clone(), devices).unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let inputs = input_types
            .into_iter()
            .zip(input_bytes)
            .map(|(r#type, bytes)| Array::from_host_buffer(&session, r#type, mesh.clone(), None, bytes).unwrap())
            .collect();
        let lowered = session.lower_program(&xla_program, 0, &Options::new(mesh)).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session.execute_program(&compiled, inputs).unwrap();
        let outputs = outputs
            .iter()
            .map(|output| {
                let bytes = output.copy_to_host_bytes().unwrap();
                ArrayIrValue::Array(CoreArray::from_logical_bytes(output.r#type().into_owned(), &bytes).unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs, emulated_outputs, "`{case:?}` outputs differ between XLA and the emulation");
    }

    /// Linearizes the program that [`shard_map_differential_program`] builds for `case` with respect to its first
    /// input and runs its primal program and then its pullback through the reference backend's emulation and through
    /// XLA execution on all devices of `client`, which must match the devices of the mesh of `case`, over the same
    /// inputs and output seeds with integer values, and checks that both produce exactly the same input cotangent,
    /// including its type. The pullback consumes the residuals of the primal program of the same backend.
    fn assert_shard_map_pullback_emulation_matches_execution(
        client: &'static Client<'static>,
        case: ShardMapDifferentialCase,
    ) {
        type ArrayMember = <Constant as ValueProjection<ArrayType>>::Projected;
        let emulated_program =
            shard_map_differential_program::<CoreArray, ArrayIrValue<CoreArray>, CoreArrayIrOperation<CoreArray>>(case);
        let xla_program = shard_map_differential_program::<ArrayMember, Constant, Operation>(case);
        let emulated = emulated_program.entry_region_ref().linearize(&[0]).unwrap();
        let xla = xla_program.entry_region_ref().linearize(&[0]).unwrap();
        let array_types = |types: Vec<Type>| {
            types
                .into_iter()
                .map(|r#type| match r#type {
                    Type::Array(r#type) => r#type,
                    r#type => panic!("differential programs have array inputs and outputs only but got `{type}`"),
                })
                .collect::<Vec<_>>()
        };
        let input_types = array_types(xla_program.input_types().into_owned());
        let output_types = array_types(xla_program.output_types().into_owned());
        let bytes = |offset: usize, r#type: &ArrayType| {
            let element_count = r#type.static_shape().unwrap().dimensions().iter().product::<usize>();
            let elements = (0..element_count).map(|element| (offset + element + 1) as f32).collect::<Vec<_>>();
            values_to_bytes::<f32>(&elements)
        };
        let input_bytes =
            input_types.iter().enumerate().map(|(index, r#type)| bytes(100 * index, r#type)).collect::<Vec<_>>();
        let seed_bytes = output_types
            .iter()
            .enumerate()
            .map(|(index, r#type)| bytes(1000 + 100 * index, r#type))
            .collect::<Vec<_>>();

        // The reference backend emulates every device of both maps on the host.
        let emulated_values = |types: &[ArrayType], bytes: &[Vec<u8>]| {
            types
                .iter()
                .zip(bytes)
                .map(|(r#type, bytes)| {
                    ArrayIrValue::Array(CoreArray::from_logical_bytes(r#type.clone(), bytes).unwrap())
                })
                .collect::<Vec<_>>()
        };
        let mut primal_outputs = emulated.primal().interpret(emulated_values(&input_types, &input_bytes)).unwrap();
        let residuals = primal_outputs.split_off(primal_outputs.len() - emulated.residual_count());
        let mut pullback_inputs = emulated_values(&output_types, &seed_bytes);
        pullback_inputs.extend(residuals);
        let emulated_cotangents = emulated.pullback().unwrap().program().interpret(pullback_inputs).unwrap();

        // XLA compiles each map into one manual computation and executes it on the devices of `client`.
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(input_types[0].sharding().unwrap().mesh().clone(), devices).unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let xla_values = |types: &[ArrayType], bytes: &[Vec<u8>]| {
            types
                .iter()
                .zip(bytes)
                .map(|(r#type, bytes)| {
                    Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, bytes).unwrap()
                })
                .collect::<Vec<_>>()
        };
        let execute = |program: &FlatProgram, inputs: Vec<_>| {
            let lowered = session.lower_program(program, 0, &Options::new(mesh.clone())).unwrap();
            let compiled = session.compile_program(&lowered).unwrap();
            session.execute_program(&compiled, inputs).unwrap()
        };
        let mut primal_outputs = execute(xla.primal(), xla_values(&input_types, &input_bytes));
        let residuals = primal_outputs.split_off(primal_outputs.len() - xla.residual_count());
        let mut pullback_inputs = xla_values(&output_types, &seed_bytes);
        pullback_inputs.extend(residuals);
        let cotangents = execute(&xla.pullback().unwrap().into_program(), pullback_inputs)
            .iter()
            .map(|cotangent| {
                let bytes = cotangent.copy_to_host_bytes().unwrap();
                ArrayIrValue::Array(CoreArray::from_logical_bytes(cotangent.r#type().into_owned(), &bytes).unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(cotangents, emulated_cotangents, "`{case:?}` cotangents differ between XLA and the emulation");
    }

    #[test]
    fn test_shard_map_operation_lower_renders_auto_axes_as_open_dimension_shardings() {
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
                  func.func @main(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data", ?}]>}) {
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
    fn test_shard_map_operation_lower_renders_nested_shard_maps() {
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
    fn test_shard_map_operation_lower_lowers_collectives_over_nested_meshes_that_retype_free_axes() {
        // An outer map over `x` contains an inner map over `y` inside `explicit_axes`, which types the free axis `z` as
        // explicit instead of auto, so the meshes of the two maps differ in the type of `z`. A collective over both
        // manual axes communicates across both regions. Their meshes share their topology, which is all that its
        // replica groups depend on, so the groups are those of either mesh.
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
                let outer_mesh = outer_mesh.clone();
                let inner_mesh = inner_mesh.clone();
                move |_, x: ArrayTracer| {
                    let nested: ArrayTracer = explicit_axes(
                        |x: ArrayTracer| {
                            Ok(shard_map(
                                |y: ArrayTracer| {
                                    y.parallel_all_gather(["x", "y"], 0)?
                                        .parallel_reduce(ReductionKind::Sum, "x")?
                                        .parallel_reduce(ReductionKind::Sum, "y")
                                },
                                x,
                                inner_mesh.clone(),
                                Sharding::replicated(inner_mesh.clone(), 0),
                                Sharding::replicated(inner_mesh.clone(), 1),
                                ShardMapOptions::new().with_manual_axes(["y"]),
                            )?)
                        },
                        x,
                        Some(&["z"]),
                        Sharding::replicated(outer_mesh.clone(), 0),
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
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2, "z"=1]>
                  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={"x", "y"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={"x", "y"}>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [], replicated={"x", "y"}>] out_shardings=[<@mesh, [{?}], replicated={"x", "y"}>] manual_axes={"x"} (%arg1: tensor<f32>) {
                      %1 = sdy.sharding_constraint %arg1 <@mesh, [], replicated={"y", "z"}> : tensor<f32>
                      %2 = sdy.manual_computation(%1) in_shardings=[<@mesh, [], replicated={"y", "z"}>] out_shardings=[<@mesh, [{?}], replicated={"y", "z"}>] manual_axes={"y"} (%arg2: tensor<f32>) {
                        %4 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1xf32>
                        %5 = "stablehlo.all_gather"(%4) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<1xf32>) -> tensor<4xf32>
                        %6 = "stablehlo.all_reduce"(%5) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, use_global_device_ids}> ({
                        ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
                          %8 = stablehlo.add %arg3, %arg4 : tensor<f32>
                          stablehlo.return %8 : tensor<f32>
                        }) : (tensor<4xf32>) -> tensor<4xf32>
                        %7 = "stablehlo.all_reduce"(%6) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>, use_global_device_ids}> ({
                        ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
                          %8 = stablehlo.add %arg3, %arg4 : tensor<f32>
                          stablehlo.return %8 : tensor<f32>
                        }) : (tensor<4xf32>) -> tensor<4xf32>
                        sdy.return %7 : tensor<4xf32>
                      } : (tensor<f32>) -> tensor<4xf32>
                      %3 = sdy.sharding_constraint %2 <@mesh, [{?}], replicated={"y"}> : tensor<4xf32>
                      sdy.return %3 : tensor<4xf32>
                    } : (tensor<f32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_shard_map_operation_lower_lowers_derivatives_of_batched_maps_whose_retained_rules_use_manual_axes() {
        // The body of a map over `x` calls an identity composite of this operation family, whose retained forward mode
        // rule reduces its tangent over `x`. No instruction of the body names `x`, so only the hidden axes that the
        // composite reports show that the rule uses it. Batching the map with the batch placed on `x` must therefore
        // keep `x` manual and gather the batch along it. Otherwise, the map would have no manual axes, and
        // differentiating it afterwards would trace the rule into a map that does not bind `x`, which cannot be
        // lowered.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let global_type = ArrayType::scalar(DataType::F32);
        let boundary = ShardMapBoundary::new(mesh.clone(), vec![replicated.clone()], vec![replicated], vec![]).unwrap();
        let local_type = boundary.local_input_type(0, &global_type).unwrap();
        let binding = NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true };
        let scope = TracingScope::new(vec![("x".to_string(), binding.clone())], Some(mesh))
            .with_manual_axes(vec![("x".to_string(), binding)]);
        let rules = CustomRuleRegistration::new(
            CustomRuleDefinition::<Constant, ArrayIrOperation>::new("jvp_over_x")
                .with_jvp(|primals, tangents| {
                    let tangent = ValueProjection::<ArrayType>::into_projected(tangents[0].clone())?;
                    let tangent = tangent.parallel_reduce(ReductionKind::Sum, "x")?;
                    Ok((primals.to_vec(), vec![tangent.into_value()]))
                })
                .with_scope(scope)
                .with_batching_support(),
        );
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(local_type.clone().into());
        let primal = builder.build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder]);
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(local_type.into());
        let primal = builder.import_program(primal.unwrap());
        let call = CompositeOperation::<Constant, ArrayIrOperation>::new(rules.reference());
        let outputs = builder.add_instruction(call, vec![primal], vec![input], None).unwrap().to_vec();
        let body = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        let map = ShardMapOperation::from_program(&body, vec![Type::Array(global_type.clone())], boundary).unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(global_type.into());
        let body = builder.import_program(body);
        let outputs = builder.add_instruction(map, vec![body], vec![input], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        let batched = program
            .batched_with_threaded_extent(
                DimensionValue::constant(2).unwrap().r#type().into_owned(),
                ShardingDimension::sharded(["x"]),
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let replicated = "{mesh<['x'=2:manual]>, [{}]}";
        let tiled = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let local = format!("f32[2][sharding={replicated}]");
        let varying = "{mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}";
        let matrix = "{mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}";
        let scalar = "{mesh<['x'=2:manual]>, [], varying_manual={'x'}}";
        assert_eq!(
            batched.to_string(),
            formatdoc! {"
                lambda %0:dimension<2>, %1:f32[2] .
                let %2:f32[2][sharding={tiled}] = shard_map [
                    mesh=['x'=2:manual],
                    input_shardings=[{replicated}],
                    output_shardings=[{tiled}],
                    manual_axes=['x'],
                    global_input_types=[{local}],
                    global_output_types=[f32[2][sharding={tiled}]],
                ] %1 [
                    body={{
                        lambda %0:{local} .
                        let %1:dimension<2> = constant [value=2]
                            %2:{local} = composite [
                                name=\"jvp_over_x\",
                                non_differentiated_count=1,
                                batching=[(extent=dimension<2>, input_axes=[axis 0], output_axes=[axis 0])],
                            ] %1 %0 [
                                primal={{
                                    lambda %0:dimension<2>, %1:{local} .
                                    in (%1)
                                }},
                            ]
                            %3:f32[2][sharding={varying}] = parallel_vary [axis_name=\"x\"] %2
                            %4:f32[2, 1][sharding={matrix}] = reshape [
                                shape=[2, 1],
                                output_sharding={matrix},
                            ] %3
                            %5:u64[][sharding={scalar}] = axis_index [axis_name=\"x\", mesh=['x'=2:manual]]
                            %6:u64[][sharding={scalar}] = zero [type=u64[][sharding={scalar}]]
                            %7:f32[1, 1][sharding={matrix}] = dynamic_slice [sizes=[1, 1]] %4 %5 %6
                            %8:f32[1][sharding={varying}] = reshape [shape=[1], output_sharding={varying}] %7
                        in (%8)
                    }},
                ]
                in (%0, %2)"},
        );

        // First-class dimensions cross the boundary of a lowered function as `i64` scalars.
        let derivative = batched.jvp().unwrap();
        let boundary_type = |r#type: &Type| match r#type {
            Type::Array(r#type) => r#type.clone(),
            _ => ArrayType::scalar(DataType::I64),
        };
        let input_types = derivative.input_types().iter().map(boundary_type).collect::<Vec<_>>();
        let output_types = derivative.output_types().iter().map(boundary_type).collect::<Vec<_>>();
        assert_eq!(
            to_mlir_module_for_program(&derivative, &[], &input_types, &output_types, "main", None, None).unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<i64>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> (tensor<i64>, tensor<2xf32>, tensor<2xf32>) {
                    %0:2 = sdy.manual_computation(%arg1, %arg2) in_shardings=[<@mesh, [{}], replicated={"x"}>, <@mesh, [{}], replicated={"x"}>] out_shardings=[<@mesh, [{"x"}]>, <@mesh, [{"x"}]>] manual_axes={"x"} (%arg3: tensor<2xf32>, %arg4: tensor<2xf32>) {
                      %1 = stablehlo.reshape %arg3 : (tensor<2xf32>) -> tensor<2x1xf32>
                      %2 = sdy.sharding_constraint %1 <@mesh, [{}, {}]> : tensor<2x1xf32>
                      %3 = stablehlo.partition_id : tensor<ui32>
                      %4 = stablehlo.convert %3 : (tensor<ui32>) -> tensor<ui64>
                      %c = stablehlo.constant dense<0> : tensor<ui64>
                      %5 = sdy.sharding_constraint %c <@mesh, []> : tensor<ui64>
                      %6 = stablehlo.convert %4 : tensor<ui64>
                      %7 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<2x1xf32>) -> tensor<i32>
                      %8 = stablehlo.convert %7 : (tensor<i32>) -> tensor<i64>
                      %c_0 = stablehlo.constant dense<0> : tensor<ui64>
                      %c_1 = stablehlo.constant dense<1> : tensor<i64>
                      %9 = stablehlo.subtract %8, %c_1 : tensor<i64>
                      %10 = stablehlo.convert %9 : (tensor<i64>) -> tensor<ui64>
                      %11 = stablehlo.maximum %6, %c_0 : tensor<ui64>
                      %12 = stablehlo.minimum %11, %10 : tensor<ui64>
                      %13 = stablehlo.convert %12 : (tensor<ui64>) -> tensor<i64>
                      %14 = stablehlo.convert %5 : tensor<ui64>
                      %15 = stablehlo.get_dimension_size %2, dim = 1 : (tensor<2x1xf32>) -> tensor<i32>
                      %16 = stablehlo.convert %15 : (tensor<i32>) -> tensor<i64>
                      %c_2 = stablehlo.constant dense<0> : tensor<ui64>
                      %c_3 = stablehlo.constant dense<1> : tensor<i64>
                      %17 = stablehlo.subtract %16, %c_3 : tensor<i64>
                      %18 = stablehlo.convert %17 : (tensor<i64>) -> tensor<ui64>
                      %19 = stablehlo.maximum %14, %c_2 : tensor<ui64>
                      %20 = stablehlo.minimum %19, %18 : tensor<ui64>
                      %21 = stablehlo.convert %20 : (tensor<ui64>) -> tensor<i64>
                      %22 = stablehlo.dynamic_slice %2, %13, %21, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i64>, tensor<i64>) -> tensor<1x1xf32>
                      %23 = stablehlo.reshape %22 : (tensor<1x1xf32>) -> tensor<1xf32>
                      %24 = sdy.sharding_constraint %23 <@mesh, [{}]> : tensor<1xf32>
                      %25 = "stablehlo.all_reduce"(%arg4) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
                        %47 = stablehlo.add %arg5, %arg6 : tensor<f32>
                        stablehlo.return %47 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      %26 = stablehlo.reshape %25 : (tensor<2xf32>) -> tensor<2x1xf32>
                      %27 = sdy.sharding_constraint %26 <@mesh, [{}, {}]> : tensor<2x1xf32>
                      %28 = stablehlo.convert %4 : tensor<ui64>
                      %29 = stablehlo.get_dimension_size %27, dim = 0 : (tensor<2x1xf32>) -> tensor<i32>
                      %30 = stablehlo.convert %29 : (tensor<i32>) -> tensor<i64>
                      %c_4 = stablehlo.constant dense<0> : tensor<ui64>
                      %c_5 = stablehlo.constant dense<1> : tensor<i64>
                      %31 = stablehlo.subtract %30, %c_5 : tensor<i64>
                      %32 = stablehlo.convert %31 : (tensor<i64>) -> tensor<ui64>
                      %33 = stablehlo.maximum %28, %c_4 : tensor<ui64>
                      %34 = stablehlo.minimum %33, %32 : tensor<ui64>
                      %35 = stablehlo.convert %34 : (tensor<ui64>) -> tensor<i64>
                      %36 = stablehlo.convert %5 : tensor<ui64>
                      %37 = stablehlo.get_dimension_size %27, dim = 1 : (tensor<2x1xf32>) -> tensor<i32>
                      %38 = stablehlo.convert %37 : (tensor<i32>) -> tensor<i64>
                      %c_6 = stablehlo.constant dense<0> : tensor<ui64>
                      %c_7 = stablehlo.constant dense<1> : tensor<i64>
                      %39 = stablehlo.subtract %38, %c_7 : tensor<i64>
                      %40 = stablehlo.convert %39 : (tensor<i64>) -> tensor<ui64>
                      %41 = stablehlo.maximum %36, %c_6 : tensor<ui64>
                      %42 = stablehlo.minimum %41, %40 : tensor<ui64>
                      %43 = stablehlo.convert %42 : (tensor<ui64>) -> tensor<i64>
                      %44 = stablehlo.dynamic_slice %27, %35, %43, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i64>, tensor<i64>) -> tensor<1x1xf32>
                      %45 = stablehlo.reshape %44 : (tensor<1x1xf32>) -> tensor<1xf32>
                      %46 = sdy.sharding_constraint %45 <@mesh, [{}]> : tensor<1xf32>
                      sdy.return %24, %46 : tensor<1xf32>, tensor<1xf32>
                    } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
                    return %arg0, %0#0, %0#1 : tensor<i64>, tensor<2xf32>, tensor<2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_shard_map_operation_lower_rejects_nested_manual_computations_over_already_manual_axes() {
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
                CoreError::Program(error)
                    if error.downcast_custom::<ShardMapError>() == Some(&expected),
            ),
        ));
    }

    #[test]
    fn test_shard_map_operation_lower_rejects_shardings_naming_already_manual_axes() {
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
                CoreError::Program(error)
                    if error.downcast_custom::<ShardMapError>() == Some(&expected),
            ),
        ));
    }

    #[test]
    fn test_shard_map_operation_lower_executes_on_cpu() {
        let client = Platform::Cpu
            .client(Some(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() })))
            .expect("failed to create 4-device CPU client");
        let session = Session::builder().client(client).build().unwrap();
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

        let input_array = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            None,
            input_buffers,
        )
        .unwrap();
        let program = PjrtProgram::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_compilation_options(client, Some(4))).unwrap();

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
            ExecutionArguments::from_arrays(vec![input_array], execution_device_ids.as_slice()).unwrap();
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
    fn test_shard_map_operation_lower_unchecked_output_replication_executes_on_cpu() {
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

        let client = cpu_client(4);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, Some(4)))
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
    fn test_shard_map_operation_lower_transposes_unchecked_output_replication_on_cpu() {
        let mesh = test_logical_mesh_2x2();
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let global_input_type =
            ArrayType::new_static(DataType::F32, [8]).with_sharding(input_sharding.clone()).unwrap();
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    Ok,
                    input,
                    mesh,
                    input_sharding.clone(),
                    output_sharding.clone(),
                    ShardMapOptions::new().with_check_output_replication(false),
                )?;
                Ok(vec![output.into_value()])
            },
            vec![Type::Array(global_input_type)],
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

        let client = cpu_client(4);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, Some(4)))
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
    fn test_shard_map_operation_lower_without_manual_variation_transposes_replicated_collectives_on_cpu() {
        // The primal is x -> 4*x. Its reverse rule divides each replicated seed by four, transposes the physical
        // collective, and sums the replicated input contributions. Both all-reduces must survive lowering.
        let mesh = test_manual_mesh("x", 4);
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let global_type = ArrayType::scalar(DataType::F32).with_sharding(replicated.clone()).unwrap();
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
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
            vec![Type::Array(global_type)],
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
            Some(std::slice::from_ref(&replicated)),
            Some(std::slice::from_ref(&replicated)),
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

        let client = cpu_client(4);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, Some(4)))
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
    fn test_shard_map_operation_lower_without_manual_variation_executes_unreduced_outputs_on_cpu() {
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

        let client = cpu_client(2);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, Some(2)))
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
    fn test_shard_map_operation_lower_with_parallel_casts_executes_on_cpu() {
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
                        product.parallel_reduce(ReductionKind::Sum, "x")
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
            execute_inputs_on_cpu_devices(
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
                    |x: ArrayTracer| x.parallel_cast("x", ParallelCastTarget::Unreduced),
                    x,
                    mesh.clone(),
                    sharded.clone(),
                    unreduced.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(shard_map(
                    |summands: ArrayTracer| summands.parallel_reduce(ReductionKind::Sum, "x"),
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
        assert_eq!(
            execute_inputs_on_cpu_devices(module, &[(&[2], [&[1.0, 2.0], &[3.0, 4.0]])]),
            vec![vec![4.0, 6.0]; 2]
        );
    }

    #[test]
    fn test_shard_map_operation_lower_with_unreduced_kinds_executes_on_cpu() {
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
                        |candidates: ArrayTracer| candidates.parallel_reduce(reduction, "x"),
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
        assert_eq!(
            execute_inputs_on_cpu_devices(maximum, &[(&[2], [&[1.0, 7.0], &[5.0, 2.0]])]),
            vec![vec![5.0, 7.0]; 2]
        );
        let minimum = module(UnreducedKind::Min, ReductionKind::Min);
        assert_eq!(
            execute_inputs_on_cpu_devices(minimum, &[(&[2], [&[1.0, 7.0], &[5.0, 2.0]])]),
            vec![vec![1.0, 2.0]; 2]
        );
    }

    #[test]
    fn test_shard_map_operation_lower_with_parallel_cast_gradient_executes_on_cpu() {
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
                            summand.parallel_reduce(ReductionKind::Sum, "x")
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
                    %0 = sdy.sharding_constraint %cst <@mesh, [], replicated={\"x\"}> : tensor<f32>
                    %1 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [], replicated={\"x\"}>] \
                        out_shardings=[<@mesh, [{\"x\"}]>] manual_axes={\"x\"} (%arg1: tensor<f32>) {
                      %3 = stablehlo.exponential %arg1 : tensor<f32>
                      %4 = stablehlo.reshape %3 : (tensor<f32>) -> tensor<1xf32>
                      %5 = sdy.sharding_constraint %4 <@mesh, [{}]> : tensor<1xf32>
                      sdy.return %5 : tensor<1xf32>
                    } : (tensor<f32>) -> tensor<2xf32>
                    %2 = sdy.manual_computation(%0, %1) in_shardings=[<@mesh, [], replicated={\"x\"}>, <@mesh, [{\"x\"}]>] \
                        out_shardings=[<@mesh, [], replicated={\"x\"}>] manual_axes={\"x\"} \
                        (%arg1: tensor<f32>, %arg2: tensor<1xf32>) {
                      %3 = stablehlo.reshape %arg2 : (tensor<1xf32>) -> tensor<f32>
                      %4 = sdy.sharding_constraint %3 <@mesh, []> : tensor<f32>
                      %5 = stablehlo.multiply %4, %arg1 : tensor<f32>
                      %6 = \"stablehlo.all_reduce\"(%5) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = \
                          dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
                        %7 = stablehlo.add %arg3, %arg4 : tensor<f32>
                        stablehlo.return %7 : tensor<f32>
                      }) : (tensor<f32>) -> tensor<f32>
                      sdy.return %6 : tensor<f32>
                    } : (tensor<f32>, tensor<2xf32>) -> tensor<f32>
                    return %2 : tensor<f32>
                  }
                }
            "},
        );
        assert_eq!(execute_inputs_on_cpu_devices(module, &[(&[], [&[0.0], &[0.0]])]), vec![vec![2.0]; 2]);
    }

    #[test]
    fn test_shard_map_operation_lower_with_parallel_arg_extrema_executes_on_cpu() {
        // Device 0 holds `[3, 1, 2]` and device 1 holds `[3, 5, 0]`, so the first element ties and resolves to the
        // lowest position, as JAX's `pargmax` and `pargmin` resolve it
        // (`.tasks/shard_map_jax_parity_gaps/m3_oracles.py`). The positions are converted to `f32` outside the map only
        // to read them back.
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
                    positions.convert_element_type(DataType::F32)
                },
                ArrayType::new_static(DataType::F32, [6]).with_sharding(sharded.clone()).unwrap(),
            )
            .unwrap();
            let module = traced.to_mlir_module("main").unwrap();
            assert_eq!(
                execute_inputs_on_cpu_devices(module, &[(&[3], [&[3.0, 1.0, 2.0], &[3.0, 5.0, 0.0]])]),
                vec![expected.to_vec(); 2],
            );
        }
    }

    #[test]
    fn test_shard_map_operation_lower_transposes_parallel_broadcast_on_cpu() {
        // The transpose hands the sum of every device's cotangent to the source device and zeros to the other device,
        // through an all-reduce and a collective permute whose only pair sends the source's sum to itself, so it
        // executes on the CPU backend, unlike the broadcast itself. With the cotangent `[1, 2, 3, 4]` and source 1, the
        // gradient is `[0, 0, 4, 6]`, as JAX computes it (`.tasks/shard_map_jax_parity_gaps/m8_oracles.py`).
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    |input: ArrayTracer| input.parallel_broadcast("x", 1),
                    input,
                    mesh.clone(),
                    sharded.clone(),
                    sharded.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(vec![output.into_value()])
            },
            vec![Type::Array(global_type.clone())],
        )
        .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        let global_types = [global_type];
        let module = to_mlir_module_for_program(
            &pullback,
            &[],
            &global_types,
            &global_types,
            "main",
            Some(std::slice::from_ref(&sharded)),
            Some(std::slice::from_ref(&sharded)),
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
            execute_inputs_on_cpu_devices(module, &[(&[2], [&[1.0, 2.0], &[3.0, 4.0]])]),
            vec![vec![0.0, 0.0], vec![4.0, 6.0]],
        );
    }

    #[test]
    fn test_shard_map_operation_lower_with_inferred_input_shardings_executes_on_cpu() {
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
            execute_inputs_on_cpu_devices(module, &[(&[2], [&[1.0, 2.0], &[3.0, 4.0]])]),
            vec![vec![2.0, 4.0], vec![6.0, 8.0]]
        );
    }

    #[test]
    fn test_shard_map_operation_lower_with_shard_map_along_executes_on_cpu() {
        // Each device receives one row of the input mapped along `x` at dimension 0 and returns it doubled, placed
        // along `x` at dimension 1 of the output, so the rows of the input become the column blocks of the output.
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
            execute_inputs_on_cpu_devices(module, &[(&[1, 3], [&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]])]),
            vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]]
        );
    }

    #[test]
    fn test_shard_map_operation_lower_without_manual_variation_passes_unreduced_summands_between_maps_on_cpu() {
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

        let client = cpu_client(2);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, Some(2)))
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
    fn test_shard_map_operation_lower_without_manual_variation_transposes_reduced_boundaries_on_cpu() {
        // The primal is `x -> 2x` over a reduced input and output. Its pullback receives one summand of the unreduced
        // seed on each device, here `[1]` and `[2]` for the seed `3`, and returns one summand of the unreduced input
        // cotangent on each device without dividing the seed or summing the cotangent, so the devices return `[2]`
        // and `[4]`, whose sum is the cotangent `6`. Dividing the seed would give `[1]` and `[2]`, and summing the
        // cotangent along `x` would give `[6]` on both devices.
        let mesh = test_manual_mesh("x", 2);
        let reduced = Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x"]).unwrap();
        let global_type = ArrayType::new_static(DataType::F32, [1]).with_sharding(reduced.clone()).unwrap();
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
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
            vec![Type::Array(global_type)],
        )
        .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        let array_types = |types: Vec<Type>| {
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

        let client = cpu_client(2);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, Some(2)))
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
    fn test_shard_map_operation_lower_without_manual_variation_executes_coordinate_readers_on_cpu() {
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

        let client = cpu_client(4);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, Some(4)))
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
    fn test_shard_map_operation_lower_renders_and_executes_matmul_on_cpu() {
        let client = Platform::Cpu
            .client(Some(ClientOptions::CPU(CpuClientOptions { device_count: Some(8), ..Default::default() })))
            .expect("failed to create 8-device CPU client");
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 8);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 8, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();

        let left_sharding = Sharding::new(
            device_mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let right_sharding = Sharding::replicated(device_mesh.logical_mesh().clone(), 2);
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
            |_, (left, right)| left.dot(&right, &DotDimensionNumbers::matmul()),
            global_input_types,
            device_mesh.logical_mesh().clone(),
            (left_sharding.clone(), right_sharding.clone()),
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

        let left_buffers = client_devices
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
        let right_values = [1.0f32, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
        let right_buffers = client_devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        values_to_bytes::<f32>(right_values.as_slice()).as_slice(),
                        BufferType::F32,
                        [4u64, 2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let left_array = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[8, 4], left_sharding.clone()),
            device_mesh.clone(),
            None,
            left_buffers,
        )
        .unwrap();
        let right_array = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[4, 2], right_sharding),
            device_mesh,
            None,
            right_buffers,
        )
        .unwrap();
        let program = PjrtProgram::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_compilation_options(client, Some(8))).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 8);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let row_start_by_device = execution_device_ids
            .iter()
            .map(|device_id| {
                let row_start = left_array.device_shard(*device_id).unwrap().slice()[0].start;
                (*device_id, row_start)
            })
            .collect::<HashMap<_, _>>();

        let execute_arguments =
            ExecutionArguments::from_arrays(vec![left_array, right_array], execution_device_ids.as_slice()).unwrap();
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
    fn test_shard_map_operation_lower_gradients_through_body_allocated_reference_residuals() {
        // The body allocates `r = reference_new(x)`, calls a custom `sin` whose reverse mode rule saves `r` as a
        // residual and whose backward rule scales the cotangent by the state that it reads from `r`, and then writes
        // `x * x` into `r`. The reference crosses from the primal `shard_map` to the tangent `shard_map` as a snapshot
        // of its final state, from which the tangent body allocates its own reference, so the gradient of
        // `sum(sin(x))` is `cos(x) * x * x`, as it is outside `shard_map`. Both bodies discharge their allocations.
        let client = cpu_client(2);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 0);
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        type BodyTracer = Tracer;
        let function = composite(|(_, x): (BodyTracer, BodyTracer)| {
            Ok(ValueProjection::<ArrayType>::into_projected(x)?.sin()?.into_value())
        })
        .with_non_differentiated_count(1)
        .with_vjp(
            |(r, x): (BodyTracer, BodyTracer)| {
                let x = ValueProjection::<ArrayType>::into_projected(x)?;
                Ok((x.sin()?.into_value(), (r, x.cos()?.into_value())))
            },
            |(r, cosine): (BodyTracer, BodyTracer), cotangent: BodyTracer| {
                let state = ValueProjection::<ArrayType>::into_projected(r.read()?)?;
                let cosine = ValueProjection::<ArrayType>::into_projected(cosine)?;
                let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
                Ok((r, (cosine * cotangent * state).into_value()))
            },
        );
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
                let (value, gradient) = inputs[0]
                    .domain()
                    .differentiate_at(inputs[0].clone())
                    .value_and_gradient(|x| {
                        let x = ValueProjection::<ArrayType>::into_projected(x)?;
                        Ok(shard_map(
                            |local_x: ArrayTracer| {
                                let value = local_x.clone().into_value();
                                let reference = value.reference_new()?;
                                let output = function.call((reference.clone(), value))?;
                                reference.write(&(local_x.clone() * local_x).into_value())?;
                                ValueProjection::<ArrayType>::into_projected(output)?
                                    .reduce(&[0], ReductionKind::Sum)?
                                    .parallel_reduce(ReductionKind::Sum, "x")
                            },
                            x,
                            logical_mesh.clone(),
                            sharded.clone(),
                            replicated.clone(),
                            ShardMapOptions::new(),
                        )
                        .unwrap()
                        .into_value())
                    })
                    .unwrap();
                Ok(vec![value, gradient])
            },
            vec![Type::Array(input_type.clone())],
        )
        .unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let inputs = [0.25f32, 0.5, 1.0, 2.0];
        let input =
            Array::from_host_buffer(&session, input_type, mesh, None, values_to_bytes::<f32>(&inputs).as_slice())
                .unwrap();
        let outputs = session.execute_program(&compiled, vec![input]).unwrap();
        let shards = |array: &Array| {
            array
                .addressable_shards()
                .map(|shard| {
                    let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                    values_from_bytes::<f32>(bytes.as_slice())
                })
                .collect::<Vec<_>>()
        };
        let value = inputs.iter().map(|&input| f32::sin(input)).sum::<f32>();
        for shard in shards(&outputs[0]) {
            assert!((shard[0] - value).abs() <= 1e-5, "{shard:?}");
        }
        let gradient = shards(&outputs[1]).concat();
        assert_eq!(gradient.len(), inputs.len());
        for (gradient, input) in gradient.into_iter().zip(inputs) {
            assert!((gradient - f32::cos(input) * input * input).abs() <= 1e-5, "{gradient} for {input}");
        }
    }

    #[test]
    fn test_shard_map_operation_lower_transpositions_for_callers_placed_along_manual_axes() {
        // The caller places its input along the manual axis `x` and the explicit axis `y`, while the map places it
        // along `x` only, so the transposed map assembles the cotangent along `x`. One `broadcast` that only changes
        // placement restores the caller's placement along both axes, which compiles to local slicing without any
        // collective, and every device receives its shard of the identity map's cotangent.
        let client = cpu_client(4);
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let along_x = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let along_x_and_y = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let caller_type = ArrayType::new_static(DataType::F32, [8]).with_sharding(along_x_and_y).unwrap();
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
                let input =
                    ValueProjection::<ArrayType>::into_projected(inputs[0].clone()).map_err(ProgramError::from)?;
                let output = shard_map(
                    |x: ArrayTracer| Ok(x),
                    input,
                    logical_mesh.clone(),
                    along_x.clone(),
                    along_x.clone(),
                    ShardMapOptions::new(),
                )
                .unwrap();
                Ok(vec![output.into_value()])
            },
            vec![Type::Array(caller_type.clone())],
        )
        .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        let lowered = session.lower_program(&pullback, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();

        // The serialized optimized HLO module names the opcode of every instruction (e.g., of its `parameter`), so it
        // names no collective opcode.
        let optimized = compiled.optimized_program().unwrap();
        let names_opcode =
            |opcode: &str| optimized.bytes().windows(opcode.len()).any(|window| window == opcode.as_bytes());
        assert!(names_opcode("parameter"));
        for opcode in ["all-gather", "all-reduce", "all-to-all", "collective-permute", "reduce-scatter"] {
            assert!(!names_opcode(opcode), "the optimized program contains a `{opcode}`");
        }

        let cotangent_type = ArrayType::new_static(DataType::F32, [8]).with_sharding(along_x).unwrap();
        let cotangent = Array::from_host_buffer(
            &session,
            cotangent_type,
            mesh,
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).as_slice(),
        )
        .unwrap();
        let outputs = session.execute_program(&compiled, vec![cotangent]).unwrap();
        assert_eq!(outputs[0].r#type().as_ref(), &caller_type);
        let shards = outputs[0]
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(shards, vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]]);
    }

    #[test]
    fn test_shard_map_operation_lower_batched_along_every_manual_axis() {
        // The caller maps the leading dimension of its input, which it places along `x` over the mesh of the map
        // itself, over a map whose only active manual axis is `x`. The placement adds no manual variation to the batch
        // items, and batching releases `x`, which leaves a manual computation without manual axes that runs the batched
        // body once on the global values, while the boundary keeps the batch dimension placed along `x`, so each of the
        // two devices holds one batch item.
        let client = cpu_client(2);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let input_sharding = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(input_sharding).unwrap();
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
                let output = batch(
                    |item| {
                        let item = ValueProjection::<ArrayType>::into_projected(item)?;
                        let output = shard_map(
                            |x| Ok(x.clone() + x),
                            item,
                            logical_mesh.clone(),
                            replicated.clone(),
                            replicated.clone(),
                            ShardMapOptions::new(),
                        )?;
                        Ok(output.into_value())
                    },
                    inputs[0].clone(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?;
                Ok(vec![output])
            },
            vec![Type::Array(input_type.clone())],
        )
        .unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) -> \
                      (tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\", ?}, {}]>] \
                        out_shardings=[<@mesh, [{\"x\", ?}, {}]>] manual_axes={} (%arg1: tensor<2x3xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2x3xf32>
                      sdy.return %1 : tensor<2x3xf32>
                    } : (tensor<2x3xf32>) -> tensor<2x3xf32>
                    return %0 : tensor<2x3xf32>
                  }
                }
            "},
        );
        let compiled = session.compile_program(&lowered).unwrap();
        let input = Array::from_host_buffer(
            &session,
            input_type,
            mesh,
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).as_slice(),
        )
        .unwrap();
        let outputs = session.execute_program(&compiled, vec![input]).unwrap();
        let items = outputs[0]
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(items, vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]]);
    }

    #[test]
    fn test_shard_map_operation_lower_matches_reference_backend() {
        // The reference backend's lockstep emulation of `shard_map` and XLA's manual computation agree exactly on
        // boundary splitting and assembly, on every collective over the manual axis, and on reference accesses and
        // write-back (e.g., reads of a replicated reference at indices that vary across devices and invariant mutations
        // of a replicated reference).
        let client = cpu_client(2);
        assert_shard_map_emulation_matches_execution(client, ShardMapDifferentialCase::IdentityAndSum);
        assert_shard_map_emulation_matches_execution(client, ShardMapDifferentialCase::AxisIndex);
        assert_shard_map_emulation_matches_execution(client, ShardMapDifferentialCase::AllGather);
        assert_shard_map_emulation_matches_execution(client, ShardMapDifferentialCase::AllToAll);
        assert_shard_map_emulation_matches_execution(client, ShardMapDifferentialCase::Permute);
        assert_shard_map_emulation_matches_execution(client, ShardMapDifferentialCase::Reference);
        assert_shard_map_emulation_matches_execution(client, ShardMapDifferentialCase::VaryingIndexReference);
        assert_shard_map_emulation_matches_execution(client, ShardMapDifferentialCase::ReplicatedReference);
    }

    #[test]
    fn test_shard_map_operation_lower_pullback_matches_reference_backend() {
        // The pullback of a read of a replicated reference at indices that vary across devices scatters each device's
        // cotangent into a local buffer that varies along `x`, sums the buffers across `x`, and adds the sum into the
        // replicated cotangent reference, both in the emulation and in XLA's manual computation. The pullbacks of an
        // update of a sharded reference and of an invariant mutation of a replicated reference agree as well.
        let client = cpu_client(2);
        assert_shard_map_pullback_emulation_matches_execution(client, ShardMapDifferentialCase::Reference);
        let case = ShardMapDifferentialCase::VaryingIndexReference;
        assert_shard_map_pullback_emulation_matches_execution(client, case);
        let case = ShardMapDifferentialCase::ReplicatedReference;
        assert_shard_map_pullback_emulation_matches_execution(client, case);
    }

    #[test]
    fn test_shard_map_operation_lower_nested_matches_reference_backend() {
        // A nested map whose inner body sums across the enclosing manual axis runs on all four devices of a 2×2 mesh,
        // both in the emulation and in XLA's nested manual computations.
        let client = cpu_client(4);
        assert_shard_map_emulation_matches_execution(client, ShardMapDifferentialCase::Nested);
    }

    #[test]
    fn test_shard_map_operation_lower_with_references_jvp_preserves_nonlinear_residuals_on_each_device() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        let logical_mesh = mesh.logical_mesh().clone();
        let session = Session::builder().client(client).build().unwrap();
        let program = nonlinear_reference_shard_map_program(&logical_mesh, false).jvp().unwrap();
        let inputs = program.input_types().into_owned();
        let array_type = <&ReferenceType<ArrayType>>::try_from(&inputs[0]).unwrap().referent().clone();
        let compiled = compile_statefully::<_, Vec<Type>, Vec<Type>>(
            |inputs| program.interpret_in_context(inputs[0].context(), inputs.clone()).map_err(Into::into),
            inputs,
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let array = |values: &[f32]| {
            Array::from_host_buffer(
                &session,
                array_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes(values).as_slice(),
            )
            .unwrap()
        };
        let reference = ArrayReference::new(array(&[2.0, 3.0, 5.0, 7.0]));
        let tangent_reference = ArrayReference::new(array(&[23.0, 29.0, 31.0, 37.0]));
        let outputs = compiled
            .execute_statefully_and_wait(
                &session,
                vec![
                    Value::Reference(Reference::new(reference.clone(), session.clone())),
                    Value::Array(array(&[11.0, 13.0, 17.0, 19.0])),
                    Value::Reference(Reference::new(tangent_reference.clone(), session.clone())),
                    Value::Array(array(&[41.0, 43.0, 47.0, 53.0])),
                ],
            )
            .unwrap();
        let outputs = outputs
            .iter()
            .map(|output| match output {
                Value::Array(array) => read_sharded_f32_array(array),
                _ => panic!("nonlinear outputs are arrays"),
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs, vec![vec![22.0, 39.0, 85.0, 133.0], vec![335.0, 506.0, 762.0, 1074.0]]);
        assert_eq!(read_sharded_f32_array(&reference.read().unwrap()), vec![2.0, 3.0, 5.0, 7.0]);
        assert_eq!(read_sharded_f32_array(&tangent_reference.read().unwrap()), vec![23.0, 29.0, 31.0, 37.0]);
    }

    #[test]
    fn test_shard_map_operation_lower_with_references_pullback_sums_replicated_reference_contributions() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        let logical_mesh = mesh.logical_mesh().clone();
        let session = Session::builder().client(client).build().unwrap();
        let program = nonlinear_reference_shard_map_program(&logical_mesh, true);
        let mut input_types = program.input_types().into_owned();
        let reference_type = <&ReferenceType<ArrayType>>::try_from(&input_types[0]).unwrap().referent().clone();
        let array_type = <&ArrayType>::try_from(&input_types[1]).unwrap().clone();
        input_types.extend(input_types.clone());
        let (primal, tangent, ..) = program.linearize().unwrap().into_parts();
        let pullback = tangent
            .transpose_with_respect_to(
                &[0, 1],
                &[CotangentDestinationKind::Reference, CotangentDestinationKind::Return],
            )
            .unwrap();
        let compiled = compile_statefully::<_, Vec<Type>, Vec<Type>>(
            |inputs| {
                let context = inputs[0].context().clone();
                let mut primals = primal.interpret_in_context(&context, inputs[..2].to_vec())?;
                let output = primals.remove(0);
                let mut operands = vec![inputs[3].clone(), inputs[2].clone()];
                operands.extend(primals);
                let mut cotangents = pullback.interpret_in_context(&context, operands)?;
                Ok(vec![output, cotangents.remove(1)])
            },
            input_types,
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(values).as_slice())
                .unwrap()
        };
        let reference = ArrayReference::new(array(&reference_type, &[2.0, 3.0]));
        let destination = ArrayReference::new(array(&reference_type, &[5.0, 7.0]));
        let outputs = compiled
            .execute_statefully_and_wait(
                &session,
                vec![
                    Value::Reference(Reference::new(reference.clone(), session.clone())),
                    Value::Array(array(&array_type, &[11.0, 13.0, 17.0, 19.0])),
                    Value::Reference(Reference::new(destination.clone(), session.clone())),
                    Value::Array(array(&array_type, &[1.0, 2.0, 3.0, 4.0])),
                ],
            )
            .unwrap();
        let outputs = outputs
            .iter()
            .map(|output| match output {
                Value::Array(array) => read_sharded_f32_array(array),
                _ => panic!("nonlinear outputs are arrays"),
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs, vec![vec![22.0, 39.0, 34.0, 57.0], vec![2.0, 6.0, 6.0, 12.0]]);
        assert_eq!(read_sharded_f32_array(&destination.read().unwrap()), vec![67.0, 109.0, 67.0, 109.0]);
        assert_eq!(read_sharded_f32_array(&reference.read().unwrap()), vec![2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn test_shard_map_operation_lower_with_references_pullback_adds_replicated_reference_cotangents_once() {
        // The output seed `[1, 2]` reaches the replicated reference's cotangent destination `[5, 7]` exactly once,
        // without any normalization by the size of the manual axis or duplication of the destination per device.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        let logical_mesh = mesh.logical_mesh().clone();
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::replicated(logical_mesh.clone(), 1);
        let array_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap();
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));
        let mut body = CoreProgramBuilder::<Constant, Operation>::new();
        let reference = body.add_input(reference_type.clone());
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let body = body
            .build::<Vec<Constant>, Vec<Constant>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let operation = ShardMapOperation::from_program(
            &body,
            vec![reference_type.clone()],
            ShardMapBoundary::new(
                logical_mesh.clone(),
                vec![sharding.clone()],
                vec![sharding.clone()],
                vec!["x".to_string()],
            )
            .unwrap(),
        )
        .unwrap();
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let reference = builder.add_input(reference_type.clone());
        let body = builder.import_program(body);
        let output = builder
            .add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], vec![reference], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        let compiled = compile_statefully::<_, Vec<Type>, Vec<Type>>(
            |inputs| {
                let context = inputs[0].context().clone();
                let outputs = pullback.interpret_in_context(&context, inputs)?;
                Ok(vec![context.bind(ReferenceReadOperation::new(), Vec::new(), &outputs)?.remove(0)])
            },
            vec![Type::Array(array_type.clone()), reference_type.clone()],
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let array = |values: &[f32]| {
            Array::from_host_buffer(
                &session,
                array_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes(values).as_slice(),
            )
            .unwrap()
        };
        let destination = ArrayReference::new(array(&[5.0, 7.0]));
        compiled
            .execute_statefully_and_wait(
                &session,
                vec![
                    Value::Array(array(&[1.0, 2.0])),
                    Value::Reference(Reference::new(destination.clone(), session.clone())),
                ],
            )
            .unwrap();
        assert_eq!(read_sharded_f32_array(&destination.read().unwrap()), vec![6.0, 9.0, 6.0, 9.0]);
    }

    #[test]
    fn test_shard_map_operation_lower_with_references_unchecked_writes_back_reduced_references() {
        // Every device receives the whole reduced referent `[1, 2]`, doubles it, and writes back its copy, so both
        // devices hold `[2, 4]` afterwards.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        let session = Session::builder().client(client).build().unwrap();
        let (program, referent_type) = doubling_reduced_reference_program(mesh.logical_mesh());
        let reference_type = Type::Reference(ReferenceType::new(referent_type.clone()));
        let compiled = compile_statefully::<_, Vec<Type>, Vec<Type>>(
            |inputs| {
                let context = inputs[0].context().clone();
                program.interpret_in_context(&context, inputs)?;
                Ok(Vec::new())
            },
            vec![reference_type],
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, referent_type, mesh, None, values_to_bytes(&[1f32, 2.0]).as_slice())
                .unwrap(),
        );
        compiled
            .execute_statefully_and_wait(
                &session,
                vec![Value::Reference(Reference::new(reference.clone(), session.clone()))],
            )
            .unwrap();
        assert_eq!(read_sharded_f32_array(&reference.read().unwrap()), vec![2.0, 4.0, 2.0, 4.0]);
    }

    #[test]
    fn test_shard_map_operation_lower_with_references_unchecked_pullback_accumulates_mutated_reduced_references() {
        // The map doubles a reduced reference in place, so the cotangent of its initial state is twice that of its
        // final state. The gradient reference is unreduced, so each device doubles its own summand in place. The
        // logical destination `[5, 7]`, uploaded as the summands `[5, 7]` and `[0, 0]`, becomes `[10, 14]`.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        let session = Session::builder().client(client).build().unwrap();
        let (program, referent_type) = doubling_reduced_reference_program(mesh.logical_mesh());
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        let cotangent_type =
            referent_type.clone().with_sharding(referent_type.sharding().unwrap().cotangent().unwrap()).unwrap();
        let cotangent_reference_type = Type::Reference(ReferenceType::new(cotangent_type.clone()));
        let compiled = compile_statefully::<_, Vec<Type>, Vec<Type>>(
            |inputs| {
                let context = inputs[0].context().clone();
                pullback.interpret_in_context(&context, inputs)?;
                Ok(Vec::new())
            },
            vec![cotangent_reference_type],
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let destination = ArrayReference::new(
            Array::from_host_buffer(&session, cotangent_type, mesh, None, values_to_bytes(&[5f32, 7.0]).as_slice())
                .unwrap(),
        );
        compiled
            .execute_statefully_and_wait(
                &session,
                vec![Value::Reference(Reference::new(destination.clone(), session.clone()))],
            )
            .unwrap();
        let destination = destination.read().unwrap();
        assert_eq!(read_sharded_f32_array(&destination), vec![10.0, 14.0, 0.0, 0.0]);
        assert_eq!(destination.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![10.0, 14.0],);
    }

    #[test]
    fn test_shard_map_operation_lower_with_references_updates_sharded_state() {
        // A manual `shard_map` over a reference sharded along its manual axis runs through the stateful ABI. Each
        // device adds its shard of the update into the shard of the referent it owns and reads that shard back, so the
        // output and the committed global referent both carry every device's write in mesh order.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        let logical_mesh = mesh.logical_mesh().clone();
        let session = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let sharded_type = global_type.clone().with_sharding(sharded.clone()).unwrap();

        // The boundary is `[r: ref<f32[4]>, x: f32[4]] -> f32[4]`, everything sharded along `x`, and the local body
        // performs `add_update(r_local, x_local); read(r_local)` over the `f32[2]` shards the device owns.
        let boundary = ShardMapBoundary::new(
            logical_mesh,
            vec![sharded.clone(), sharded.clone()],
            vec![sharded.clone()],
            vec!["x".to_string()],
        )
        .unwrap();
        let body = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let local_referent = boundary.local_input_type(0, &global_type).unwrap();
            let reference = builder.add_input(Type::Reference(ReferenceType::new(local_referent)));
            let update = builder.add_input(Type::Array(boundary.local_input_type(1, &global_type).unwrap()));
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let operation = ShardMapOperation::from_program(
            &body,
            vec![Type::Reference(ReferenceType::new(sharded_type.clone())), Type::Array(sharded_type.clone())],
            boundary,
        )
        .unwrap();
        assert_eq!(operation.global_output_types(), &[Type::Array(sharded_type.clone())]);
        let operation = Operation::ShardMap(Box::new(operation));
        let compiled = compile_statefully::<_, (Type, Type), Type>(
            |(reference, update)| {
                let context = reference.domain();
                let mut outputs = context.bind(operation, vec![body], &[reference, update])?;
                Ok(outputs.pop().unwrap())
            },
            (Type::Reference(ReferenceType::new(sharded_type.clone())), Type::Array(sharded_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();

        let reference = ArrayReference::new(
            Array::from_host_buffer(
                &session,
                sharded_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
            )
            .unwrap(),
        );
        let update = Array::from_host_buffer(
            &session,
            sharded_type,
            mesh,
            None,
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0, 40.0]).as_slice(),
        )
        .unwrap();
        let Value::Array(output) = compiled
            .execute_statefully_and_wait(
                &session,
                (Value::Reference(Reference::new(reference.clone(), session.clone())), Value::Array(update)),
            )
            .unwrap()
        else {
            panic!("stateful shard-map output must be an array")
        };

        // Device 0 owns `[1, 2]` and device 1 owns `[3, 4]`, so the concatenation of the device shards in mesh order
        // shows that each device mutated exactly its own shard, both in the read output and in the committed referent.
        assert_eq!(output.sharding(), &sharded);
        assert_eq!(read_sharded_f32_array(&output), vec![11.0, 22.0, 33.0, 44.0]);
        let committed = reference.read().unwrap();
        assert_eq!(committed.sharding(), &sharded);
        assert_eq!(read_sharded_f32_array(&committed), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_shard_map_operation_lower_with_references_reads_replicated_reference_on_every_shard() {
        // A manual `shard_map` that reads a reference replicated along its manual axis runs through the stateful ABI.
        // Every device sees the whole referent and adds it to the shard of the value input it owns, while the referent
        // itself is left untouched on every device.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        let logical_mesh = mesh.logical_mesh().clone();
        let session = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let referent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let replicated_type = referent_type.clone().with_sharding(replicated.clone()).unwrap();
        let global_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let sharded_type = global_type.clone().with_sharding(sharded.clone()).unwrap();

        // The boundary is `[r: ref<f32[2]>, x: f32[4]] -> f32[4]` with `r` replicated and `x` and the output sharded
        // along `x`, and the local body performs `read(r_local) + x_local` over the whole `f32[2]` referent and the
        // `f32[2]` shard of `x` the device owns.
        let boundary = ShardMapBoundary::new(
            logical_mesh,
            vec![replicated, sharded.clone()],
            vec![sharded.clone()],
            vec!["x".to_string()],
        )
        .unwrap();
        let body = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let local_referent = boundary.local_input_type(0, &referent_type).unwrap();
            let reference = builder.add_input(Type::Reference(ReferenceType::new(local_referent)));
            let update = builder.add_input(Type::Array(boundary.local_input_type(1, &global_type).unwrap()));
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let state = builder
                .add_instruction(ParallelVaryOperation::new("x".to_string()), Vec::new(), vec![state], None)
                .unwrap()[0];
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![state, update], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let operation = ShardMapOperation::from_program(
            &body,
            vec![Type::Reference(ReferenceType::new(replicated_type.clone())), Type::Array(sharded_type.clone())],
            boundary,
        )
        .unwrap();
        assert_eq!(operation.global_output_types(), &[Type::Array(sharded_type.clone())]);
        let operation = Operation::ShardMap(Box::new(operation));
        let compiled = compile_statefully::<_, (Type, Type), Type>(
            |(reference, update)| {
                let context = reference.domain();
                let mut outputs = context.bind(operation, vec![body], &[reference, update])?;
                Ok(outputs.pop().unwrap())
            },
            (Type::Reference(ReferenceType::new(replicated_type.clone())), Type::Array(sharded_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();

        let reference = ArrayReference::new(
            Array::from_host_buffer(
                &session,
                replicated_type,
                mesh.clone(),
                None,
                values_to_bytes::<f32>(&[100.0, 200.0]).as_slice(),
            )
            .unwrap(),
        );
        let update = Array::from_host_buffer(
            &session,
            sharded_type,
            mesh,
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let Value::Array(output) = compiled
            .execute_statefully_and_wait(
                &session,
                (Value::Reference(Reference::new(reference.clone(), session.clone())), Value::Array(update)),
            )
            .unwrap()
        else {
            panic!("stateful shard-map output must be an array")
        };

        // Both devices add the whole `[100, 200]` referent to their own shard of `x`, and the replicated referent,
        // read here as the copy of each device in mesh order, is unchanged on both devices.
        assert_eq!(output.sharding(), &sharded);
        assert_eq!(read_sharded_f32_array(&output), vec![101.0, 202.0, 103.0, 204.0]);
        assert_eq!(read_sharded_f32_array(&reference.read().unwrap()), vec![100.0, 200.0, 100.0, 200.0]);
    }

    #[test]
    fn test_shard_map_operation_lower_without_inputs_executes_on_cpu() {
        // This mirrors JAX's `test_axis_index`. A body without inputs reads the coordinate of each device along `x`
        // through its context, and the output sharding tiles the coordinates along `x`, so the global output is
        // `[0, 1, 2, 3]`. The `shard_map` lowers to an `sdy.manual_computation` without operands, and the traced
        // program takes no inputs because `trace_with_context` passes the tracing context to its function.
        let client = cpu_client(4);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedProgram<(), ArrayType> = trace_with_context(
            {
                let sharded = sharded.clone();
                move |context, ()| {
                    Ok(context
                        .shard_map(
                            |context: &ProjectedContext<
                                DomainTracingContext<DomainTracingContext<Session>>,
                                ArrayType,
                            >,
                             ()| { context.axis_index("x")?.reshape([1]) },
                            (),
                            mesh,
                            (),
                            sharded,
                            ShardMapOptions::new(),
                        )
                        .unwrap())
                }
            },
            (),
        )
        .unwrap();
        let program = traced.to_mlir_module_with_signature_shardings("main", Some(&[]), Some(&[sharded])).unwrap();
        assert_eq!(
            program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main() -> (tensor<4xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation() in_shardings=[] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} () {
                      %1 = stablehlo.partition_id : tensor<ui32>
                      %2 = stablehlo.convert %1 : (tensor<ui32>) -> tensor<ui64>
                      %3 = stablehlo.reshape %2 : (tensor<ui64>) -> tensor<1xui64>
                      sdy.return %3 : tensor<1xui64>
                    } : () -> tensor<4xui64>
                    return %0 : tensor<4xui64>
                  }
                }
            "#},
        );

        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: program.into_bytes() }, &test_compilation_options(client, Some(4)))
            .unwrap();
        let outputs = executable
            .execute(
                (0..4).map(|_| ExecutionDeviceInputs::default()).collect(),
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

        // Device `d` holds shard `d` of the output, which is its own coordinate along `x`.
        let values = outputs
            .into_iter()
            .flat_map(|output| {
                assert_eq!(output.outputs.len(), 1);
                values_from_bytes::<u64>(&output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(values, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_shard_map_operation_lower_executes_checked_variation_gradients_on_cpu() {
        for device_count in [1, 2] {
            let client = cpu_client(device_count);
            let session = Session::builder().client(client).build().unwrap();
            let client_devices = client.addressable_devices().unwrap();
            assert_eq!(client_devices.len(), device_count);
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", device_count, MeshAxisType::Manual).unwrap()]).unwrap();
            let device_mesh = DeviceMesh::new(
                mesh.clone(),
                client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
            )
            .unwrap();
            let replicated = Sharding::replicated(mesh.clone(), 0);
            let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let traced: TracedProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
                {
                    let replicated = replicated.clone();
                    let sharded = sharded.clone();
                    move |inputs: Vec<ArrayTracer>| {
                        let (value, gradient) = inputs[0]
                            .clone()
                            .into_value()
                            .domain()
                            .differentiate_at((inputs[0].clone().into_value(), inputs[1].clone().into_value()))
                            .value_and_gradient(|(shared, weights)| {
                                let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                                let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                                Ok(shard_map(
                                    |(shared, weights): (ArrayTracer, ArrayTracer)| {
                                        (shared.clone() * shared * weights)
                                            .reduce(&[0], ReductionKind::Sum)?
                                            .parallel_reduce(ReductionKind::Sum, "x")
                                    },
                                    (shared, weights),
                                    mesh.clone(),
                                    (replicated.clone(), sharded.clone()),
                                    replicated.clone(),
                                    ShardMapOptions::new(),
                                )
                                .unwrap()
                                .into_value())
                            })
                            .unwrap();
                        let varying_gradient = inputs[1]
                            .clone()
                            .into_value()
                            .domain()
                            .differentiate_at(inputs[1].clone().into_value())
                            .gradient(|varying| {
                                let varying = ValueProjection::<ArrayType>::into_projected(varying)?;
                                Ok(shard_map(
                                    |varying: ArrayTracer| {
                                        (varying.clone() * varying)
                                            .reduce(&[0], ReductionKind::Sum)?
                                            .parallel_reduce(ReductionKind::Sum, "x")
                                    },
                                    varying,
                                    mesh.clone(),
                                    sharded.clone(),
                                    replicated.clone(),
                                    ShardMapOptions::new(),
                                )
                                .unwrap()
                                .into_value())
                            })
                            .unwrap();
                        let second_derivative = inputs[0]
                            .clone()
                            .into_value()
                            .domain()
                            .differentiate_at(inputs[0].clone().into_value())
                            .with_captures(inputs[1].clone().into_value())
                            .gradient(|shared, weights| {
                                Ok(shared
                                    .domain()
                                    .differentiate_at(shared.clone())
                                    .with_captures(weights)
                                    .gradient(|shared, weights| {
                                        let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                                        let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                                        Ok(shard_map(
                                            |(shared, weights): (ArrayTracer, ArrayTracer)| {
                                                (shared.clone() * shared * weights)
                                                    .reduce(&[0], ReductionKind::Sum)?
                                                    .parallel_reduce(ReductionKind::Sum, "x")
                                            },
                                            (shared, weights),
                                            mesh.clone(),
                                            (replicated.clone(), sharded.clone()),
                                            replicated.clone(),
                                            ShardMapOptions::new(),
                                        )
                                        .unwrap()
                                        .into_value())
                                    })
                                    .unwrap())
                            })
                            .unwrap();
                        let (_, pullback) = inputs[0]
                            .clone()
                            .into_value()
                            .domain()
                            .differentiate_at((inputs[0].clone().into_value(), inputs[1].clone().into_value()))
                            .vjp(|(shared, weights)| {
                                let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                                let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                                Ok(shard_map(
                                    |(shared, weights): (ArrayTracer, ArrayTracer)| {
                                        Ok(shared.clone() * shared * weights)
                                    },
                                    (shared, weights),
                                    mesh.clone(),
                                    (replicated.clone(), sharded.clone()),
                                    sharded.clone(),
                                    ShardMapOptions::new(),
                                )
                                .unwrap()
                                .into_value())
                            })
                            .unwrap();
                        let seeded_gradients = pullback.apply_materialized(inputs[1].clone().into_value()).unwrap();
                        let (_, directional_derivative) = inputs[0]
                            .clone()
                            .into_value()
                            .domain()
                            .differentiate_at((inputs[0].clone().into_value(), inputs[1].clone().into_value()))
                            .jvp(
                                (inputs[0].clone().into_value(), inputs[1].clone().into_value()),
                                |(shared, weights)| {
                                    let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                                    let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                                    Ok(shard_map(
                                        |(shared, weights): (ArrayTracer, ArrayTracer)| {
                                            Ok(shared.clone() * shared * weights)
                                        },
                                        (shared, weights),
                                        mesh.clone(),
                                        (replicated.clone(), sharded.clone()),
                                        sharded.clone(),
                                        ShardMapOptions::new(),
                                    )
                                    .unwrap()
                                    .into_value())
                                },
                            )
                            .unwrap();
                        let inner_gradient = shard_map(
                            |(shared, weights): (ArrayTracer, ArrayTracer)| {
                                let derivative = shared
                                    .clone()
                                    .into_value()
                                    .domain()
                                    .differentiate_at(shared.into_value())
                                    .with_captures(weights.into_value())
                                    .gradient(|shared, weights| {
                                        let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                                        let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                                        Ok((shared.clone() * shared * weights)
                                            .reduce(&[0], ReductionKind::Sum)
                                            .unwrap()
                                            .into_value())
                                    })?;
                                Ok(ValueProjection::<ArrayType>::into_projected(derivative)?)
                            },
                            (inputs[0].clone(), inputs[1].clone()),
                            mesh.clone(),
                            (replicated.clone(), sharded.clone()),
                            replicated.clone(),
                            ShardMapOptions::new(),
                        )
                        .unwrap();
                        let constant_sum = shard_map(
                            |input: ArrayTracer| {
                                let constant: ArrayTracer = input.domain().fill(&input.r#type(), 3.0f32)?;
                                constant.parallel_reduce(ReductionKind::Sum, "x")
                            },
                            inputs[0].clone(),
                            mesh.clone(),
                            replicated.clone(),
                            replicated.clone(),
                            ShardMapOptions::new(),
                        )
                        .unwrap();
                        vec![
                            ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(gradient.0).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(gradient.1).unwrap(),
                            constant_sum,
                            ValueProjection::<ArrayType>::into_projected(varying_gradient).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(second_derivative).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(seeded_gradients.0).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(seeded_gradients.1).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(directional_derivative).unwrap(),
                            inner_gradient,
                        ]
                    }
                },
                vec![
                    ArrayType::scalar(DataType::F32),
                    ArrayType::new_static(DataType::F32, [device_count]).with_sharding(sharded.clone()).unwrap(),
                ],
            )
            .unwrap();
            let program = traced
                .to_mlir_module_with_signature_shardings(
                    "main",
                    Some(&[replicated.clone(), sharded.clone()]),
                    Some(&[
                        replicated.clone(),
                        replicated.clone(),
                        sharded.clone(),
                        replicated.clone(),
                        sharded.clone(),
                        replicated.clone(),
                        replicated.clone(),
                        sharded.clone(),
                        sharded.clone(),
                        replicated.clone(),
                    ]),
                )
                .unwrap();
            let executable = client
                .compile(
                    &PjrtProgram::Mlir { bytecode: program.into_bytes() },
                    &test_compilation_options(client, Some(device_count)),
                )
                .unwrap();
            let shared_buffers = client_devices
                .iter()
                .map(|device| {
                    client.buffer(&3.0f32.to_ne_bytes(), BufferType::F32, [], None, device.clone(), None).unwrap()
                })
                .collect();
            let weight_buffers = client_devices
                .iter()
                .zip([2.0f32, 5.0])
                .map(|(device, weight)| {
                    client.buffer(&weight.to_ne_bytes(), BufferType::F32, [1], None, device.clone(), None).unwrap()
                })
                .collect();
            let inputs = vec![
                Array::from_addressable_buffers(
                    &session,
                    static_sharded_array_type(DataType::F32, &[], replicated),
                    device_mesh.clone(),
                    None,
                    shared_buffers,
                )
                .unwrap(),
                Array::from_addressable_buffers(
                    &session,
                    static_sharded_array_type(DataType::F32, &[device_count], sharded),
                    device_mesh,
                    None,
                    weight_buffers,
                )
                .unwrap(),
            ];
            let device_ids = executable
                .addressable_devices()
                .unwrap()
                .iter()
                .map(|device| device.id().unwrap())
                .collect::<Vec<_>>();
            let arguments = ExecutionArguments::from_arrays(inputs, &device_ids).unwrap();
            let outputs = executable
                .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
                .unwrap()
                .block_until_ready()
                .unwrap();
            assert_eq!(outputs.len(), device_count);
            let expected_varying_gradients = client_devices
                .iter()
                .zip([4.0, 10.0])
                .map(|(device, gradient)| (device.id().unwrap(), gradient))
                .collect::<HashMap<_, _>>();
            let total_weight = if device_count == 1 { 2.0 } else { 7.0 };
            let squared_weight_sum = if device_count == 1 { 4.0 } else { 29.0 };
            let mut output_pairing = 0.0;
            let mut input_pairing = 0.0;
            for (device_index, (output, device_id)) in outputs.into_iter().zip(device_ids).enumerate() {
                let expected_varying_gradient = expected_varying_gradients[&device_id];
                assert_eq!(output.outputs.len(), 10);
                let values = output
                    .outputs
                    .iter()
                    .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                    .collect::<Vec<_>>();
                let weight = expected_varying_gradient / 2.0;
                output_pairing += weight * values[8][0];
                input_pairing += weight * values[7][0];
                if device_index == 0 {
                    input_pairing += 3.0 * values[6][0];
                }
                // The shared derivative sums distinct contributions, and each varying weight receives only its local
                // one.
                assert_eq!(
                    values,
                    vec![
                        vec![9.0 * total_weight],
                        vec![6.0 * total_weight],
                        vec![9.0],
                        vec![3.0 * device_count as f32],
                        vec![expected_varying_gradient],
                        vec![2.0 * total_weight],
                        vec![6.0 * squared_weight_sum],
                        vec![9.0 * weight],
                        vec![27.0 * weight],
                        vec![6.0 * total_weight],
                    ],
                );
            }
            // Count the invariant shared input once, and pair varying inputs and outputs over their logical shards.
            assert_eq!(input_pairing, output_pairing);
            assert_eq!(output_pairing, 27.0 * squared_weight_sum);
        }
    }

    #[test]
    fn test_shard_map_operation_lower_executes_nested_variation_gradients_on_cpu() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let device_mesh = DeviceMesh::new(
            mesh.clone(),
            client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let outer = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let inner = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let global = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let traced: TracedProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
            {
                let replicated = replicated.clone();
                move |inputs: Vec<ArrayTracer>| {
                    let (value, gradient) = inputs[0]
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at((inputs[0].clone().into_value(), inputs[1].clone().into_value()))
                        .value_and_gradient(|(shared, weights)| {
                            let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                            let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                            Ok(shard_map(
                                |inputs: (ArrayTracer, ArrayTracer)| {
                                    shard_map(
                                        |(shared, weights): (ArrayTracer, ArrayTracer)| {
                                            (shared.clone() * shared * weights.clone() * weights)
                                                .reduce(&[0], ReductionKind::Sum)?
                                                .parallel_reduce(ReductionKind::Sum, "y")
                                        },
                                        inputs,
                                        mesh.clone(),
                                        (replicated.clone(), inner.clone()),
                                        replicated.clone(),
                                        ShardMapOptions::new().with_manual_axes(["y"]),
                                    )?
                                    .parallel_reduce(ReductionKind::Sum, "x")
                                },
                                (shared, weights),
                                mesh.clone(),
                                (replicated.clone(), outer.clone()),
                                replicated.clone(),
                                ShardMapOptions::new().with_manual_axes(["x"]),
                            )
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient.0).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient.1).unwrap(),
                    ]
                }
            },
            vec![ArrayType::scalar(DataType::F32), ArrayType::new_static(DataType::F32, [4])],
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(&[replicated.clone(), global.clone()]),
                Some(&[replicated.clone(), replicated.clone(), global.clone()]),
            )
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: program.into_bytes() }, &test_compilation_options(client, Some(4)))
            .unwrap();
        let shared_buffers = client_devices
            .iter()
            .map(|device| {
                client.buffer(&3.0f32.to_ne_bytes(), BufferType::F32, [], None, device.clone(), None).unwrap()
            })
            .collect();
        let weight_buffers = client_devices
            .iter()
            .zip([1.0f32, 2.0, 3.0, 4.0])
            .map(|(device, weight)| {
                client.buffer(&weight.to_ne_bytes(), BufferType::F32, [1], None, device.clone(), None).unwrap()
            })
            .collect();
        let inputs = vec![
            Array::from_addressable_buffers(
                &session,
                static_sharded_array_type(DataType::F32, &[], replicated),
                device_mesh.clone(),
                None,
                shared_buffers,
            )
            .unwrap(),
            Array::from_addressable_buffers(
                &session,
                static_sharded_array_type(DataType::F32, &[4], global),
                device_mesh,
                None,
                weight_buffers,
            )
            .unwrap(),
        ];
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = ExecutionArguments::from_arrays(inputs, &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 4);
        let expected_weight_gradients = client_devices
            .iter()
            .zip([18.0, 36.0, 54.0, 72.0])
            .map(|(device, gradient)| (device.id().unwrap(), gradient))
            .collect::<HashMap<_, _>>();
        for (output, device_id) in outputs.into_iter().zip(device_ids) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(values, vec![vec![270.0], vec![180.0], vec![expected_weight_gradients[&device_id]]]);
        }
    }

    #[test]
    fn test_shard_map_operation_lower_executes_mixed_variation_gradients_on_cpu() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let device_mesh = DeviceMesh::new(
            mesh.clone(),
            client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let shared_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();
        let global =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["z"])])
                .unwrap();
        let traced: TracedProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
            {
                let replicated = replicated.clone();
                let shared_sharding = shared_sharding.clone();
                let global = global.clone();
                move |inputs: Vec<ArrayTracer>| {
                    let (value, gradient) = inputs[0]
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at((inputs[0].clone().into_value(), inputs[1].clone().into_value()))
                        .value_and_gradient(|(shared, weights)| {
                            let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                            let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                            Ok(shard_map(
                                |(shared, weights): (ArrayTracer, ArrayTracer)| {
                                    (shared.clone() * shared * weights)
                                        .reduce(&[0, 1], ReductionKind::Sum)?
                                        .parallel_reduce(ReductionKind::Sum, "x")
                                },
                                (shared, weights),
                                mesh.clone(),
                                (shared_sharding.clone(), global.clone()),
                                replicated.clone(),
                                ShardMapOptions::new().with_manual_axes(["x"]),
                            )
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient.0).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient.1).unwrap(),
                    ]
                }
            },
            vec![
                ArrayType::new_static(DataType::F32, [2]).with_sharding(shared_sharding.clone()).unwrap(),
                ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(global.clone()).unwrap(),
            ],
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(&[shared_sharding.clone(), global.clone()]),
                Some(&[replicated.clone(), shared_sharding.clone(), global.clone()]),
            )
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: program.into_bytes() }, &test_compilation_options(client, Some(4)))
            .unwrap();
        let shared_buffers = client_devices
            .iter()
            .zip([3.0f32, 4.0, 3.0, 4.0])
            .map(|(device, shared)| {
                client.buffer(&shared.to_ne_bytes(), BufferType::F32, [1], None, device.clone(), None).unwrap()
            })
            .collect();
        let weight_buffers = client_devices
            .iter()
            .zip([2.0f32, 3.0, 5.0, 7.0])
            .map(|(device, weight)| {
                client.buffer(&weight.to_ne_bytes(), BufferType::F32, [1, 1], None, device.clone(), None).unwrap()
            })
            .collect();
        let inputs = vec![
            Array::from_addressable_buffers(
                &session,
                static_sharded_array_type(DataType::F32, &[2], shared_sharding),
                device_mesh.clone(),
                None,
                shared_buffers,
            )
            .unwrap(),
            Array::from_addressable_buffers(
                &session,
                static_sharded_array_type(DataType::F32, &[2, 2], global),
                device_mesh,
                None,
                weight_buffers,
            )
            .unwrap(),
        ];
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = ExecutionArguments::from_arrays(inputs, &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 4);
        let expected_gradients = client_devices
            .iter()
            .zip([(42.0, 9.0), (80.0, 16.0), (42.0, 9.0), (80.0, 16.0)])
            .map(|(device, gradient)| (device.id().unwrap(), gradient))
            .collect::<HashMap<_, _>>();
        for (output, device_id) in outputs.into_iter().zip(device_ids) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                values,
                vec![vec![223.0], vec![expected_gradients[&device_id].0], vec![expected_gradients[&device_id].1]],
            );
        }
    }

    #[test]
    fn test_shard_map_operation_lower_executes_gather_gradient_with_varying_indices_on_cpu() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_types = vec![
            ArrayType::new_static(DataType::F32, [3]),
            ArrayType::new_static(DataType::I32, [4]).with_sharding(sharded.clone()).unwrap(),
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
        ];
        let traced: TracedProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
            {
                let mesh = mesh.clone();
                let replicated = replicated.clone();
                let sharded = sharded.clone();
                move |inputs: Vec<ArrayTracer>| {
                    let (output, pullback) = inputs[0]
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at(inputs[0].clone().into_value())
                        .with_captures(inputs[1].clone().into_value())
                        .vjp(|data, indices| {
                            let data = ValueProjection::<ArrayType>::into_projected(data)?;
                            let indices = ValueProjection::<ArrayType>::into_projected(indices)?;
                            Ok(shard_map(
                                |(data, indices): (ArrayTracer, ArrayTracer)| {
                                    data.gather_axis(&indices, 0, GatherMode::Clip)
                                },
                                (data, indices),
                                mesh.clone(),
                                (replicated.clone(), sharded.clone()),
                                sharded.clone(),
                                ShardMapOptions::new(),
                            )
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    let gradient = pullback.apply_materialized(inputs[2].clone().into_value()).unwrap();
                    let maxima = shard_map(
                        |input: ArrayTracer| {
                            let maximum = input.parallel_reduce(ReductionKind::Max, "x")?;
                            let varying = maximum.parallel_vary("x")?;
                            Ok((maximum, varying))
                        },
                        inputs[2].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        (replicated.clone(), sharded.clone()),
                        ShardMapOptions::new(),
                    )
                    .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(output).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient).unwrap(),
                        maxima.0,
                        maxima.1,
                    ]
                }
            },
            input_types,
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(&[replicated.clone(), sharded.clone(), sharded.clone()]),
                Some(&[sharded.clone(), replicated.clone(), replicated.clone(), sharded.clone()]),
            )
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: program.into_bytes() }, &test_compilation_options(client, Some(2)))
            .unwrap();
        let data_buffers = devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        values_to_bytes(&[11.0f32, 13.0, 17.0]).as_slice(),
                        BufferType::F32,
                        [3],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect();
        let index_buffers = devices
            .iter()
            .zip([[0i32, 1], [0, 2]])
            .map(|(device, indices)| {
                client
                    .buffer(values_to_bytes(&indices).as_slice(), BufferType::I32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let seed_buffers = devices
            .iter()
            .zip([[2.0f32, 3.0], [5.0, 7.0]])
            .map(|(device, seeds)| {
                client
                    .buffer(values_to_bytes(&seeds).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let inputs = vec![
            Array::from_addressable_buffers(
                &session,
                static_sharded_array_type(DataType::F32, &[3], replicated),
                device_mesh.clone(),
                None,
                data_buffers,
            )
            .unwrap(),
            Array::from_addressable_buffers(
                &session,
                static_sharded_array_type(DataType::I32, &[4], sharded.clone()),
                device_mesh.clone(),
                None,
                index_buffers,
            )
            .unwrap(),
            Array::from_addressable_buffers(
                &session,
                static_sharded_array_type(DataType::F32, &[4], sharded),
                device_mesh,
                None,
                seed_buffers,
            )
            .unwrap(),
        ];
        let expected = devices
            .iter()
            .zip([[11.0f32, 13.0], [11.0, 17.0]])
            .map(|(device, values)| (device.id().unwrap(), values))
            .collect::<HashMap<_, _>>();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = ExecutionArguments::from_arrays(inputs, &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 2);
        for (output, device_id) in outputs.into_iter().zip(device_ids) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            // Index zero occurs on both devices, so its unequal seed contributions must be added exactly once.
            assert_eq!(
                values,
                vec![expected[&device_id].to_vec(), vec![7.0, 3.0, 7.0], vec![5.0, 7.0], vec![5.0, 7.0]],
            );
        }
    }

    #[test]
    fn test_shard_map_operation_lower_executes_sort_jvp_with_varying_keys_on_cpu() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_types = vec![
            ArrayType::new_static(DataType::F32, [2]),
            ArrayType::new_static(DataType::I32, [4]).with_sharding(sharded.clone()).unwrap(),
            ArrayType::new_static(DataType::F32, [2]),
        ];
        let traced: TracedProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
            {
                let mesh = mesh.clone();
                let replicated = replicated.clone();
                let sharded = sharded.clone();
                move |inputs: Vec<ArrayTracer>| {
                    let (output, tangent) = inputs[0]
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at(inputs[0].clone().into_value())
                        .with_captures(inputs[1].clone().into_value())
                        .jvp(inputs[2].clone().into_value(), |data, indices| {
                            let data = ValueProjection::<ArrayType>::into_projected(data)?;
                            let indices = ValueProjection::<ArrayType>::into_projected(indices)?;
                            Ok(shard_map(
                                |(data, indices): (ArrayTracer, ArrayTracer)| {
                                    Ok(Sort::sort(&[indices, data], 0, SortDirection::Ascending)?.remove(1))
                                },
                                (data, indices),
                                mesh.clone(),
                                (replicated.clone(), sharded.clone()),
                                sharded.clone(),
                                ShardMapOptions::new(),
                            )
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(output).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(tangent).unwrap(),
                    ]
                }
            },
            input_types,
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(&[replicated.clone(), sharded.clone(), replicated.clone()]),
                Some(&[sharded.clone(), sharded.clone()]),
            )
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: program.into_bytes() }, &test_compilation_options(client, Some(2)))
            .unwrap();
        let data_buffers = devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        values_to_bytes(&[11.0f32, 13.0]).as_slice(),
                        BufferType::F32,
                        [2],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect();
        let index_buffers = devices
            .iter()
            .zip([[2i32, 1], [1, 2]])
            .map(|(device, indices)| {
                client
                    .buffer(values_to_bytes(&indices).as_slice(), BufferType::I32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let seed_buffers = devices
            .iter()
            .zip([[2.0f32, 3.0], [2.0, 3.0]])
            .map(|(device, seeds)| {
                client
                    .buffer(values_to_bytes(&seeds).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let inputs = vec![
            Array::from_addressable_buffers(
                &session,
                static_sharded_array_type(DataType::F32, &[2], replicated.clone()),
                device_mesh.clone(),
                None,
                data_buffers,
            )
            .unwrap(),
            Array::from_addressable_buffers(
                &session,
                static_sharded_array_type(DataType::I32, &[4], sharded.clone()),
                device_mesh.clone(),
                None,
                index_buffers,
            )
            .unwrap(),
            Array::from_addressable_buffers(
                &session,
                static_sharded_array_type(DataType::F32, &[2], replicated),
                device_mesh,
                None,
                seed_buffers,
            )
            .unwrap(),
        ];
        let expected = devices
            .iter()
            .zip([[13.0f32, 11.0], [11.0, 13.0]])
            .map(|(device, values)| (device.id().unwrap(), values))
            .collect::<HashMap<_, _>>();
        let expected_tangents = devices
            .iter()
            .zip([[3.0f32, 2.0], [2.0, 3.0]])
            .map(|(device, values)| (device.id().unwrap(), values))
            .collect::<HashMap<_, _>>();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = ExecutionArguments::from_arrays(inputs, &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 2);
        for (output, device_id) in outputs.into_iter().zip(device_ids) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            // The invariant passenger and its tangent follow each device's distinct key permutation.
            assert_eq!(values, vec![expected[&device_id].to_vec(), expected_tangents[&device_id].to_vec()]);
        }
    }

    #[test]
    fn test_shard_map_operation_lower_executes_attention_gradient_with_varying_keys_on_cpu() {
        type AttentionDomain = EagerContext<ArrayConstant, ArrayOperation<ArrayConstant>>;

        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 4);
        let sharded = Sharding::new(
            mesh.clone(),
            vec![
                ShardingDimension::sharded(["x"]),
                ShardingDimension::Replicated,
                ShardingDimension::Replicated,
                ShardingDimension::Replicated,
            ],
        )
        .unwrap();
        let query_type = ArrayType::new_static(DataType::F32, [1, 1, 1, 1]).with_sharding(replicated.clone()).unwrap();
        let key_type = ArrayType::new_static(DataType::F32, [2, 2, 1, 1]).with_sharding(sharded.clone()).unwrap();
        let value_type = ArrayType::new_static(DataType::F32, [1, 2, 1, 1]).with_sharding(replicated.clone()).unwrap();
        let seed_type = ArrayType::new_static(DataType::F32, [2, 1, 1, 1]).with_sharding(sharded.clone()).unwrap();
        let input_types = vec![query_type.clone(), key_type.clone(), value_type.clone(), seed_type.clone()];
        let input_shardings = vec![replicated.clone(), sharded.clone(), replicated.clone(), sharded.clone()];
        let output_shardings = vec![sharded.clone(), replicated.clone()];
        let boundary = ShardMapBoundary::new(mesh.clone(), input_shardings, output_shardings, vec![]).unwrap();
        let local_input_types = input_types
            .iter()
            .enumerate()
            .map(|(index, input)| boundary.local_input_type(index, input).unwrap())
            .collect::<Vec<_>>();
        // Trace the composite in its homogeneous universe, then lift the complete program. Promoting the whole program
        // preserves the primal region of the call and converts its retained rules into the lifted family.
        let (_, body) = DomainTracingContext::<AttentionDomain>::trace_with_named_axes(
            |inputs: Vec<DomainTracer<AttentionDomain>>| {
                let function = differentiable_dot_product_attention::<AttentionDomain>(
                    AttentionConfiguration::new().with_implementation(AttentionImplementation::Portable),
                );
                let (output, pullback) = inputs[0]
                    .domain()
                    .differentiate_at(AttentionInputs::new(inputs[0].clone(), inputs[1].clone(), inputs[2].clone()))
                    .vjp(|inputs| function.call(inputs))
                    .unwrap();
                let gradient = pullback.apply_materialized(inputs[3].clone()).unwrap();
                // The query component of the full VJP is the partial derivative with respect to the query, and
                // making key and value cotangents available does not change this component.
                Ok(vec![output, gradient.query])
            },
            local_input_types,
            vec![(
                "x".to_string(),
                NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true },
            )],
        )
        .unwrap();
        let body = body.into_flat_program().into_unprojected::<Constant, Operation>().unwrap();
        let operation =
            ShardMapOperation::from_program(&body, input_types.iter().cloned().map(Type::Array).collect(), boundary)
                .unwrap();
        let traced: TracedProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
            move |inputs: Vec<ArrayTracer>| {
                let inputs = inputs.into_iter().map(ProjectedValue::into_value).collect::<Vec<_>>();
                inputs[0]
                    .domain()
                    .bind(Operation::ShardMap(Box::new(operation.clone())), vec![body.clone()], &inputs)
                    .unwrap()
                    .into_iter()
                    .map(|output| ValueProjection::<ArrayType>::into_projected(output).unwrap())
                    .collect()
            },
            input_types,
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(&[replicated.clone(), sharded.clone(), replicated.clone(), sharded.clone()]),
                Some(&[sharded, replicated]),
            )
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: program.into_bytes() }, &test_compilation_options(client, Some(2)))
            .unwrap();
        let inputs = vec![
            Array::from_host_buffer(&session, query_type, device_mesh.clone(), None, values_to_bytes(&[0.0f32]))
                .unwrap(),
            Array::from_host_buffer(
                &session,
                key_type,
                device_mesh.clone(),
                None,
                values_to_bytes(&[0.0f32, 1.0, 0.0, 3.0]),
            )
            .unwrap(),
            Array::from_host_buffer(&session, value_type, device_mesh.clone(), None, values_to_bytes(&[2.0f32, 6.0]))
                .unwrap(),
            Array::from_host_buffer(&session, seed_type, device_mesh, None, values_to_bytes(&[2.0f32, 5.0])).unwrap(),
        ];
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = ExecutionArguments::from_arrays(inputs, &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            // At query zero the attention weights are equal. The query derivatives are 1 and 3, so the distinct
            // output seeds give one shared cotangent of 2*1 + 5*3 = 17, rather than either local contribution.
            assert_eq!(values, vec![vec![4.0], vec![17.0]]);
        }
    }

    #[test]
    fn test_shard_map_operation_lower_executes_varying_while_predicate_gradient_on_cpu() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // Each device squares its shard while the shard is below `10`, so the loop predicate varies over `x` and the
        // devices run different numbers of iterations. Inputs `1.5`, `2`, `3`, and `20` take three, two, two, and zero
        // iterations. Reverse mode stores residual stacks and validity masks per device, with the same variation.
        let traced: TracedProgram<ArrayType, Vec<ArrayType>> = trace(
            {
                let sharded = sharded.clone();
                move |x: ArrayTracer| {
                    let (value, gradient) = x
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at(x.into_value())
                        .value_and_gradient(|x| {
                            let x = ValueProjection::<ArrayType>::into_projected(x)?;
                            Ok(shard_map(
                                |local_x: ArrayTracer| {
                                    let local_type = local_x.r#type().into_owned();
                                    // The limit is a non-differentiable constant, so it is created with the
                                    // variation of the state that it is compared against.
                                    let limit_sharding =
                                        Sharding::replicated(local_type.sharding().unwrap().mesh().clone(), 0)
                                            .with_varying_manual_axes(["x"])
                                            .unwrap();
                                    let limit_type =
                                        ArrayType::scalar(DataType::F32).with_sharding(limit_sharding).unwrap();
                                    let condition = {
                                        let mut builder = ProgramBuilder::new();
                                        let state = builder.add_input(Type::Array(local_type.clone()));
                                        let state = builder
                                            .add_instruction(
                                                ReduceOperation::new(vec![0], ReductionKind::Sum),
                                                Vec::new(),
                                                vec![state],
                                                None,
                                            )
                                            .unwrap()[0];
                                        let limit = CoreArray::from_elements(limit_type, &[10f32]).unwrap();
                                        let limit = builder
                                            .add_instruction(
                                                ConstantOperation::new(limit),
                                                Vec::new(),
                                                Vec::new(),
                                                None,
                                            )
                                            .unwrap()[0];
                                        let predicate = builder
                                            .add_instruction(
                                                Operation::Array(ArrayOperation::Compare(CompareOperation::new(
                                                    ComparisonDirection::LessThan,
                                                ))),
                                                Vec::new(),
                                                vec![state, limit],
                                                None,
                                            )
                                            .unwrap()[0];
                                        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
                                    };
                                    let body = {
                                        let mut builder = ProgramBuilder::new();
                                        let state = builder.add_input(Type::Array(local_type));
                                        let next_state = builder
                                            .add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None)
                                            .unwrap()[0];
                                        builder.build(vec![next_state], vec![Placeholder], vec![Placeholder]).unwrap()
                                    };
                                    let context = local_x.value().context().clone();
                                    let mut outputs = context.stage_operation(
                                        Operation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                                        vec![condition, body],
                                        &[local_x.into_value()],
                                    )?;
                                    Ok(ValueProjection::<ArrayType>::into_projected(outputs.remove(0))?)
                                },
                                x,
                                mesh.clone(),
                                sharded.clone(),
                                sharded.clone(),
                                ShardMapOptions::new(),
                            )
                            .unwrap()
                            .reduce(&[0], ReductionKind::Sum)
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient).unwrap(),
                    ]
                }
            },
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(std::slice::from_ref(&sharded)),
                Some(&[replicated, sharded.clone()]),
            )
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: program.into_bytes() }, &test_compilation_options(client, Some(4)))
            .unwrap();
        let inputs = [1.5f32, 2.0, 3.0, 20.0];
        let buffers = devices
            .iter()
            .zip(inputs)
            .map(|(device, input)| {
                client
                    .buffer(values_to_bytes(&[input]).as_slice(), BufferType::F32, [1], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let input = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[4], sharded),
            device_mesh,
            None,
            buffers,
        )
        .unwrap();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = ExecutionArguments::from_arrays(vec![input], &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Each shard computes `x^(2^k)` after `k` iterations, whose derivative is `2^k x^(2^k - 1)`.
        let expected_gradients = devices
            .iter()
            .map(|device| device.id().unwrap())
            .zip([8.0 * 1.5f32.powi(7), 4.0 * 2f32.powi(3), 4.0 * 3f32.powi(3), 1.0])
            .collect::<HashMap<_, _>>();
        assert_eq!(outputs.len(), 4);
        for (output, device_id) in outputs.into_iter().zip(device_ids) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(values, vec![vec![1.5f32.powi(8) + 16.0 + 81.0 + 20.0], vec![expected_gradients[&device_id]]]);
        }
    }

    #[test]
    fn test_shard_map_operation_lower_batched_matches_per_item_results_on_cpu() {
        // Batching a `shard_map` that sums its local shards across the four devices along `x` over a leading axis of
        // three items yields, for every item, what the unbatched `shard_map` yields for that item alone. The batch
        // dimension is an unpartitioned leading dimension of the boundary, so every device holds all three items of its
        // shard.
        type BatchedShardMapTracer = ProjectedValue<
            ArrayType,
            DomainTracer<BatchingContext<DomainTracingContext<Session>, ArrayIrBatchingPolicy>>,
        >;
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let item_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let batched_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let item_sharding = item_sharding.clone();
                move |x: ArrayTracer| {
                    let output = batch(
                        |item| {
                            let item = ValueProjection::<ArrayType>::into_projected(item)?;
                            let output = shard_map(
                                |local: BatchedShardMapTracer| local.parallel_reduce(ReductionKind::Sum, "x"),
                                item,
                                mesh.clone(),
                                item_sharding.clone(),
                                item_sharding.clone(),
                                ShardMapOptions::new(),
                            )?;
                            Ok(output.into_value())
                        },
                        x.into_value(),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxisSpecification::default(),
                    )
                    .unwrap();
                    ValueProjection::<ArrayType>::into_projected(output).unwrap()
                }
            },
            ArrayType::new_static(DataType::F32, [3, 8]),
        )
        .unwrap();
        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<3x8xf32>) -> (tensor<3x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"x"}]>] out_shardings=[<@mesh, [{}, {"x"}]>] manual_axes={"x"} (%arg1: tensor<3x2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %2 : tensor<f32>
                      }) : (tensor<3x2xf32>) -> tensor<3x2xf32>
                      sdy.return %1 : tensor<3x2xf32>
                    } : (tensor<3x8xf32>) -> tensor<3x8xf32>
                    return %0 : tensor<3x8xf32>
                  }
                }
            "#},
        );

        // Item `b` of the global `f32[3, 8]` input holds `8 · b + j` at column `j`, so device `d` holds columns
        // `2 · d` and `2 · d + 1` of every item.
        let shard = |item: usize, device: usize| [(8 * item + 2 * device) as f32, (8 * item + 2 * device + 1) as f32];
        let buffers = devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let values = (0..3).flat_map(|item| shard(item, device_index)).collect::<Vec<_>>();
                client
                    .buffer(
                        values_to_bytes::<f32>(&values).as_slice(),
                        BufferType::F32,
                        [3u64, 2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect();
        let input = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[3, 8], batched_sharding),
            device_mesh,
            None,
            buffers,
        )
        .unwrap();
        let executable = client
            .compile(
                &PjrtProgram::Mlir { bytecode: mlir_program.into_bytes() },
                &test_compilation_options(client, Some(4)),
            )
            .unwrap();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = ExecutionArguments::from_arrays(vec![input], &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
            .unwrap()
            .block_until_ready()
            .unwrap()
            .into_iter()
            .map(|output| {
                values_from_bytes::<f32>(output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap().as_slice())
            })
            .collect::<Vec<_>>();
        for item in 0..3 {
            let item_outputs = execute_parallel_reduce_on_cpu(
                ReductionKind::Sum,
                DataType::F32,
                BufferType::F32,
                2,
                std::array::from_fn(|device| values_to_bytes::<f32>(&shard(item, device))),
            );
            for (device_outputs, item_output) in outputs.iter().zip(item_outputs) {
                assert_eq!(
                    &device_outputs[2 * item..2 * item + 2],
                    values_from_bytes::<f32>(item_output.as_slice()).as_slice(),
                );
            }
        }
    }

    #[test]
    fn test_shard_map_operation_lower_batched_along_a_free_manual_axis_matches_per_item_results_on_cpu() {
        // Batching a `shard_map` over the manual axes `x` and `y` of four CPU devices, whose body sums its local shards
        // along `x` only, over a leading axis of four items that the input places on `y` (i.e., the analogue of JAX's
        // `spmd_axis_name`) makes `y` free in the batched `shard_map`. Every device holds the two items of its
        // coordinate along `y`, and the result for every item is what the unbatched `shard_map` yields for that item.
        type BatchedShardMapTracer = ProjectedValue<
            ArrayType,
            DomainTracer<BatchingContext<DomainTracingContext<Session>, ArrayIrBatchingPolicy>>,
        >;
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let caller_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let device_mesh = DeviceMesh::new(
            caller_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let item_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let batched_sharding =
            Sharding::new(caller_mesh, vec![ShardingDimension::sharded(["y"]), ShardingDimension::sharded(["x"])])
                .unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let item_sharding = item_sharding.clone();
                move |x: ArrayTracer| {
                    let output = batch(
                        |item| {
                            let item = ValueProjection::<ArrayType>::into_projected(item)?;
                            let output = shard_map(
                                |local: BatchedShardMapTracer| local.parallel_reduce(ReductionKind::Sum, "x"),
                                item,
                                mesh.clone(),
                                item_sharding.clone(),
                                item_sharding.clone(),
                                ShardMapOptions::new(),
                            )?;
                            Ok(output.into_value())
                        },
                        x.into_value(),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxisSpecification::default(),
                    )
                    .unwrap();
                    ValueProjection::<ArrayType>::into_projected(output).unwrap()
                }
            },
            static_sharded_array_type(DataType::F32, &[4, 8], batched_sharding.clone()),
        )
        .unwrap();
        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y", ?}, {"x"}]>] out_shardings=[<@mesh, [{"y", ?}, {"x"}]>] manual_axes={"x"} (%arg1: tensor<4x4xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %2 : tensor<f32>
                      }) : (tensor<4x4xf32>) -> tensor<4x4xf32>
                      sdy.return %1 : tensor<4x4xf32>
                    } : (tensor<4x8xf32>) -> tensor<4x8xf32>
                    return %0 : tensor<4x8xf32>
                  }
                }
            "#},
        );

        // Item `b` of the global `f32[4, 8]` input holds `8 · b + j` at column `j`, so the device at mesh coordinates
        // `(x, y)` (i.e., device `2 · x + y`) holds columns `4 · x` to `4 · x + 3` of items `2 · y` and `2 · y + 1`.
        let shard = |device: usize| {
            let (x, y) = (device / 2, device % 2);
            (2 * y..2 * y + 2)
                .flat_map(|item| (0..4).map(move |column| (8 * item + 4 * x + column) as f32))
                .collect::<Vec<_>>()
        };
        let buffers = devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard(device_index)).as_slice(),
                        BufferType::F32,
                        [2u64, 4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect();
        let input = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[4, 8], batched_sharding),
            device_mesh,
            None,
            buffers,
        )
        .unwrap();
        let executable = client
            .compile(
                &PjrtProgram::Mlir { bytecode: mlir_program.into_bytes() },
                &test_compilation_options(client, Some(4)),
            )
            .unwrap();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = ExecutionArguments::from_arrays(vec![input], &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
            .unwrap()
            .block_until_ready()
            .unwrap()
            .into_iter()
            .map(|output| {
                values_from_bytes::<f32>(output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap().as_slice())
            })
            .collect::<Vec<_>>();

        // The unbatched `shard_map` for item `b` sums the two column blocks of that item along `x`, so its output holds
        // `(8 · b + c) + (8 · b + 4 + c) = 16 · b + 2 · c + 4` at column `c` of every block. The device at mesh
        // coordinates `(x, y)` holds the output columns `4 · x` to `4 · x + 3` of items `2 · y` and `2 · y + 1`.
        for (device_index, device_outputs) in outputs.iter().enumerate() {
            let y = device_index % 2;
            let expected = (2 * y..2 * y + 2)
                .flat_map(|item| (0..4).map(move |column| (16 * item + 2 * column + 4) as f32))
                .collect::<Vec<_>>();
            assert_eq!(device_outputs, &expected);
        }
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
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:2 = sdy.manual_computation(%arg0, %0) in_shardings=[<@mesh, [{}], replicated={"x"}>, <@mesh, []>] out_shardings=[<@mesh, [{}], replicated={"x"}>, <@mesh, []>] manual_axes={"x"} (%arg1: tensor<4xf32>, %arg2: !stablehlo.token) {
                      %2 = stablehlo.custom_call @ryft.print(%arg1, %arg2) {api_version = 4 : i32, backend_config = {label = "first"}, has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<4xf32>, !stablehlo.token) -> !stablehlo.token
                      %3 = stablehlo.custom_call @ryft.print(%arg1, %2) {api_version = 4 : i32, backend_config = {label = "second"}, has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<4xf32>, !stablehlo.token) -> !stablehlo.token
                      sdy.return %arg1, %3 : tensor<4xf32>, !stablehlo.token
                    } : (tensor<4xf32>, !stablehlo.token) -> (tensor<4xf32>, !stablehlo.token)
                    return %1#0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_device_ordered_effects_execute_on_two_cpu_devices() {
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
        let client = cpu_client(2);
        ensure_print_handler_registered(client).unwrap();
        let options = test_compilation_options(client, Some(2));
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
                        client.buffer(bytes.as_slice(), BufferType::F32, [4], None, device.clone(), None).unwrap(),
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
    fn test_shardy_tensor_sharding_per_value() {
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
            shardy_tensor_sharding_per_value(boundary.input_shardings(), boundary.manual_axes(), &[], &context)
                .unwrap()
                .to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"x"}], replicated={"y"}>]>"#,
        );
    }

    #[test]
    fn test_shardy_tensor_sharding_per_value_leaves_inactive_manual_axes_open() {
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
            shardy_tensor_sharding_per_value(boundary.input_shardings(), boundary.manual_axes(), &[], &context)
                .unwrap()
                .to_string(),
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
            shardy_tensor_sharding_per_value(boundary.input_shardings(), boundary.manual_axes(), &[], &context)
                .unwrap()
                .to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"y", ?}], replicated={"x"}>]>"#,
        );
        assert_eq!(
            shardy_tensor_sharding_per_value(
                boundary.input_shardings(),
                boundary.manual_axes(),
                &["x".to_string()],
                &context
            )
            .unwrap()
            .to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"y"}]>]>"#,
        );
    }

    #[test]
    fn test_shardy_tensor_sharding_per_value_renders_free_axes_as_open_dimension_shardings() {
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
            shardy_tensor_sharding_per_value(boundary.input_shardings(), boundary.manual_axes(), &[], &context)
                .unwrap()
                .to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"data", ?}]>, <@mesh, [{"data", ?}]>]>"#,
        );
    }

    #[test]
    fn test_shardy_tensor_sharding_per_value_keeps_explicit_axes() {
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
            shardy_tensor_sharding_per_value(boundary.input_shardings(), boundary.manual_axes(), &[], &context)
                .unwrap()
                .to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"data", "model", ?}]>]>"#,
        );
    }

    #[test]
    fn test_shardy_tensor_sharding_per_value_renders_replicated_manual_axes() {
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
            shardy_tensor_sharding_per_value(boundary.output_shardings(), boundary.manual_axes(), &[], &context)
                .unwrap()
                .to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{?}], replicated={"x"}>]>"#,
        );
    }

    #[test]
    fn test_eager_shard_map_without_manual_variation_passes_unreduced_summands_between_maps_on_cpu() {
        // This is the pipeline of the previous test, with each map applied eagerly to concrete arrays through the
        // frontend. The first map returns each device's shard of `x = [1, 2]` as its summand of an unreduced output,
        // and the second map squares each summand of that unreduced input. The devices return `[1]` and `[4]`, whose
        // sum `5` host materialization recovers.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
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
        let input = Array::from_host_buffer(&session, input_type, mesh, None, values_to_bytes(&[1f32, 2.])).unwrap();
        let summands = shard_map(Ok, input, logical_mesh.clone(), sharded, unreduced.clone(), options.clone()).unwrap();
        let squares = shard_map(
            |summand: ProjectedValue<ArrayType, DomainTracer<Session>>| Ok(summand.clone() * summand),
            summands,
            logical_mesh,
            unreduced.clone(),
            unreduced.clone(),
            options,
        )
        .unwrap();
        assert_eq!(squares.r#type().sharding(), Some(&unreduced));
        assert_eq!(read_device_f32s(&squares), vec![vec![1.0], vec![4.0]]);
        let materialized = squares.copy_to_host_bytes().unwrap();
        assert_eq!(values_from_bytes::<f32>(materialized.as_slice()), vec![5.0]);
    }

    #[test]
    fn test_shard_map_batched_along_every_manual_axis_executes_for_callers_over_explicit_meshes_on_cpu() {
        // The caller types its input over an `Explicit` copy of the mesh of the map, which differs from it only in axis
        // types, and maps its leading dimension along `x`, the only active manual axis of the map. Batching releases
        // `x`, so the map has no active manual axes, and its outputs are typed over the `Manual` mesh of the map, which
        // the caller's device mesh accepts because both meshes have the same topology. Each device holds one doubled
        // item.
        let client = cpu_client(2);
        let map_mesh = test_manual_mesh("x", 2);
        let caller_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            caller_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
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
        let compiled = staged.lower(&session).unwrap().compile(&session).unwrap();
        let input =
            Array::from_host_buffer(&session, input_type, mesh, None, values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]))
                .unwrap();
        let Value::Array(output) = compiled.executable().execute(&session, Value::Array(input)).unwrap() else {
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
        // The caller maps the items `[1, 2, 3, 4]` along `b`, which its input places on `x`, through a map over `x`
        // whose body reduces over both `b` and `x`. The body uses `x`, so the batch is gathered along `x` on entry,
        // every device computes `sum_b(a) + sum_x(a) = 10 + 2 * a` for every item, and each device keeps the block of
        // items that the output sharding assigns to it along `x`. Splitting the batch along `x` instead, as JAX's
        // `spmd_axis_name` does, would make the reduction over `x` add different items.
        let client = cpu_client(2);
        let map_mesh = test_manual_mesh("x", 2);
        let caller_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            caller_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
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
        let compiled = staged.lower(&session).unwrap().compile(&session).unwrap();
        let input =
            Array::from_host_buffer(&session, input_type, mesh, None, values_to_bytes(&[1f32, 2., 3., 4.])).unwrap();
        let Value::Array(output) = compiled.executable().execute(&session, Value::Array(input)).unwrap() else {
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
    fn test_shard_map_compilation_requires_one_mesh_topology_on_cpu() {
        // Two inputless maps each read the coordinate of every device along `x` and tile it along `x`. The module
        // declares one mesh, which assigns device ids to coordinates, while `axis_index` computes coordinates from the
        // axis order of the mesh of its own map, so maps over meshes with other axis orders must not share a module.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
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
                    ReshapeOperation::new(Shape::new(vec![]), Shape::new(vec![Dimension::Static(1)])),
                    vec![],
                    vec![coordinate],
                    None,
                )
                .unwrap()[0];
            let body =
                builder.build::<Vec<Constant>, Vec<Constant>>(vec![coordinate], vec![], vec![Placeholder]).unwrap();
            let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let boundary = ShardMapBoundary::new(mesh.clone(), vec![], vec![output_sharding], vec![]).unwrap();
            (ShardMapOperation::from_program(&body, Vec::<Type>::new(), boundary).unwrap(), body)
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
                &session,
                device_mesh,
            )
        };

        // Maps over meshes with the same axis order agree on the coordinates of every device.
        let compiled = compile_maps(&mesh, device_mesh(&mesh)).unwrap();
        let input = Array::from_host_buffer(&session, input_type.clone(), device_mesh(&mesh), None, 0f32.to_ne_bytes())
            .unwrap();
        let (first, second) = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(first.copy_to_host().unwrap().elements::<u64>().unwrap(), vec![0, 1]);
        assert_eq!(second.copy_to_host().unwrap().elements::<u64>().unwrap(), vec![0, 1]);

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
                CoreError::Sharding(ShardingError::MeshMismatch { expected, actual })
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
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
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
                ReshapeOperation::new(Shape::new(vec![]), Shape::new(vec![Dimension::Static(1)])),
                vec![],
                vec![coordinate],
                None,
            )
            .unwrap()[0];
        let body = builder.build::<Vec<Constant>, Vec<Constant>>(vec![coordinate], vec![], vec![Placeholder]).unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let boundary = ShardMapBoundary::new(mesh.clone(), vec![], vec![output_sharding], vec![]).unwrap();
        let map = ShardMapOperation::from_program(&body, Vec::<Type>::new(), boundary).unwrap();
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
            &session,
            device_mesh(),
        )
        .unwrap();
        let input = Array::from_host_buffer(&session, input_type, device_mesh(), None, 0f32.to_ne_bytes()).unwrap();
        let (unplaced, replicated) = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(unplaced.copy_to_host().unwrap().elements::<u64>().unwrap(), vec![0, 1]);
        assert_eq!(replicated.copy_to_host().unwrap().elements::<u64>().unwrap(), vec![0, 1]);
    }
}
