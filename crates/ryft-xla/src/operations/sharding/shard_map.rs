use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use ryft_core::{
    ArrayIrType, ArrayType, AxisIndexOperation, LogicalMesh, MeshAxisType, Operation, OperationPayloadProjection,
    ParallelAllGatherOperation, ParallelAllToAllOperation, ParallelPermuteOperation, ParallelRaggedAllToAllOperation,
    ParallelReduceOperation, ParallelSumScatterOperation, ParallelVaryOperation, Parameterized, ProgramError, RegionId,
    SHARD_MAP_OPERATION_NAME, ShardMap, ShardMapError, ShardMapOperation, Sharding, ShardingDimension, TracedShardMap,
};
use ryft_mlir::dialects::shardy::{
    DimensionShardingAttributeRef, ManualAxesAttributeRef, TensorShardingAttributeRef,
    TensorShardingPerValueAttributeRef,
};
use ryft_mlir::dialects::{func, shardy};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, Location, LocationRef, Operation as MlirOperation, Region,
    Type, TypeAndAttributes, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::regions::lower_program_outputs_with_inputs;
use crate::experimental::lowering::types::{lower_boundary_layout, lower_tensor_type};
use crate::experimental::lowering::{
    CollectiveLoweringState, EffectTokens, contains_unresolved_references, contains_unresolved_state,
    current_or_new_token, normalize_function_name, remove_unused_constants, serialize_lowered_module,
    token_threaded_effects,
};
use crate::experimental::ops::{FlatXlaProgram, XlaConstant, XlaOperation, XlaProgram};
use crate::operations::control_flow::lowered_effect_classes;
use crate::sharding::SHARDY_MESH_SYMBOL_NAME;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, ToMlir, XlaDomain, XlaLowerer};

/// Lowers a [`TracedShardMap`] traced for [`XlaDomain`] (e.g., by [`ryft_core::trace_shard_map`]) to a textual
/// StableHLO/Shardy MLIR module whose single function runs the traced body as one `sdy.manual_computation` over the
/// boundary of the traced shard map. The function arguments and results carry the input and output shardings of the
/// boundary as `sdy.sharding` attributes and preserve requested physical layouts through `mhlo.layout_mode`. The
/// traced body is lowered as is, since tracing already simplified it.
///
/// # Parameters
///
///   - `traced_shard_map`: Traced shard map to lower.
///   - `function_name`: Symbol name to use for the outer `func.func`.
///
/// # Errors
///
/// Returns [`LoweringError::UnresolvedReference`] or [`LoweringError::UnresolvedState`] when the traced body contains
/// undischarged references or state, and [`LoweringError::InvalidFunctionName`] for an invalid `function_name`. Nested
/// manual computations that Shardy cannot represent are rejected with [`LoweringError::Tracing`] wrapping (as a
/// [`ProgramError::Custom`] that [`ProgramError::downcast_custom`] recovers) a [`ShardMapError::AxisAlreadyManual`]
/// when a nested shard map makes an axis manual that an enclosing manual computation already made manual, and a
/// [`ShardMapError::SpecificationNamesEnclosingManualAxis`] when an input or output sharding of a nested shard map
/// names such an axis. A traced shard map whose global input or output type varies along a mesh axis that is not one
/// of its own manual axes (e.g., one traced with [`ryft_core::trace_shard_map_with_named_axes`] inside an enclosing
/// manual region) is rejected with [`LoweringError::StandaloneShardMapVariesAlongEnclosingManualAxis`], because the
/// standalone module has no enclosing manual region that makes such an axis manual. For the same reason, a body that
/// applies a collective (including `axis_index`) along such an axis is rejected with
/// [`LoweringError::StandaloneShardMapCommunicatesAlongEnclosingManualAxis`], even when the boundary does not vary
/// along it. Other [`LoweringError`]s report failures to lower the traced body (e.g., for operations that the backend
/// cannot lower or invalid tensor types).
pub fn to_mlir_module<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>, S: AsRef<str>>(
    traced_shard_map: &TracedShardMap<XlaDomain<'static>, Input, Output>,
    function_name: S,
) -> Result<String, LoweringError> {
    let program = traced_shard_map.body();
    let shard_map = traced_shard_map.operation().shard_map();
    // The boundary of a standalone module is global, so it cannot carry the variation of enclosing manual regions.
    for (value_kind, types) in [
        ("input", traced_shard_map.global_input_types().parameters().collect::<Vec<_>>()),
        ("output", traced_shard_map.global_output_types().parameters().collect::<Vec<_>>()),
    ] {
        for (value_index, r#type) in types.into_iter().enumerate() {
            let axis_name = r#type.sharding().and_then(|sharding| {
                sharding.varying_manual_axes().iter().find(|axis| !shard_map.manual_axes().contains(*axis))
            });
            if let Some(axis_name) = axis_name {
                return Err(LoweringError::StandaloneShardMapVariesAlongEnclosingManualAxis {
                    value_kind,
                    value_index,
                    axis_name: axis_name.clone(),
                });
            }
        }
    }
    // A collective in the body can communicate along an enclosing manual axis even when the boundary does not vary
    // along it (e.g., a reduction of an invariant input), and the standalone module has no manual region for it either.
    if let Some((operation_name, axis_name)) = find_collective_along_enclosing_manual_axis(
        program,
        program.entry_region_ref().id(),
        shard_map.mesh(),
        shard_map.manual_axes(),
    ) {
        return Err(LoweringError::StandaloneShardMapCommunicatesAlongEnclosingManualAxis {
            operation_name,
            axis_name,
        });
    }
    // This module entry must enforce the same discharge preconditions as `lower_mlir_module_for_program`: these are
    // the only guards keeping unresolved state and references out of the shard-map token-threading machinery.
    if contains_unresolved_references(program) {
        return Err(LoweringError::UnresolvedReference { construct: "program".to_string() });
    }
    if contains_unresolved_state(program) {
        return Err(LoweringError::UnresolvedState { construct: "program".to_string() });
    }
    lower_shard_map_module(
        shard_map,
        program,
        traced_shard_map.global_input_types().parameters().cloned().collect(),
        traced_shard_map.local_input_types().parameters().cloned().collect(),
        traced_shard_map.global_output_types().parameters().cloned().collect(),
        function_name.as_ref(),
    )
}

/// Returns the name of the first collective (including `axis_index`) of `region` in `program`, or of one of its nested
/// regions, that names a manual axis of `mesh` outside of `manual_axes`, together with that axis's name. Every nested
/// shard map adds its own manual axes to `manual_axes` for its body.
fn find_collective_along_enclosing_manual_axis(
    program: &FlatXlaProgram,
    region: RegionId,
    mesh: &LogicalMesh,
    manual_axes: &[String],
) -> Option<(&'static str, String)> {
    for instruction in program.regions()[region.index()].instructions() {
        let operation = instruction.operation();
        let axis_name = operation
            .projected_payload::<AxisIndexOperation>()
            .map(AxisIndexOperation::axis_name)
            .or_else(|| {
                operation.projected_payload::<ParallelReduceOperation>().map(ParallelReduceOperation::axis_name)
            })
            .or_else(|| operation.projected_payload::<ParallelVaryOperation>().map(ParallelVaryOperation::axis_name))
            .or_else(|| {
                operation
                    .projected_payload::<ParallelAllGatherOperation>()
                    .map(ParallelAllGatherOperation::axis_name)
            })
            .or_else(|| {
                operation
                    .projected_payload::<ParallelSumScatterOperation>()
                    .map(ParallelSumScatterOperation::axis_name)
            })
            .or_else(|| {
                operation.projected_payload::<ParallelAllToAllOperation>().map(ParallelAllToAllOperation::axis_name)
            })
            .or_else(|| {
                operation.projected_payload::<ParallelPermuteOperation>().map(ParallelPermuteOperation::axis_name)
            })
            .or_else(|| {
                operation
                    .projected_payload::<ParallelRaggedAllToAllOperation>()
                    .map(ParallelRaggedAllToAllOperation::axis_name)
            });
        if let Some(axis_name) = axis_name
            && mesh.axis_type(axis_name) == Some(MeshAxisType::Manual)
            && !manual_axes.iter().any(|manual_axis| manual_axis == axis_name)
        {
            return Some((operation.name(), axis_name.to_string()));
        }
        let nested_manual_axes = match operation {
            XlaOperation::ShardMap(nested) => {
                Cow::Owned(manual_axes.iter().chain(nested.shard_map().manual_axes()).cloned().collect::<Vec<_>>())
            }
            _ => Cow::Borrowed(manual_axes),
        };
        for &nested_region in instruction.regions() {
            let found = find_collective_along_enclosing_manual_axis(program, nested_region, mesh, &nested_manual_axes);
            if found.is_some() {
                return found;
            }
        }
    }
    None
}

/// Lowers the shard-map `program` over its flat boundary types to a textual StableHLO/Shardy MLIR module (refer to
/// [`to_mlir_module`]).
fn lower_shard_map_module(
    shard_map: &ShardMap,
    program: &FlatXlaProgram,
    global_input_types: Vec<ArrayType>,
    local_input_types: Vec<ArrayType>,
    global_output_types: Vec<ArrayType>,
    function_name: &str,
) -> Result<String, LoweringError> {
    let function_name = normalize_function_name(function_name)?;

    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location)?;

    let global_input_tensor_types = global_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let mesh_operation = shard_map.mesh().to_mlir(location)?;
    module.body()?.append_operation(mesh_operation)?;

    // The global boundary types of a traced shard map carry no memory kind (refer to `ShardMap::global_input_type` and
    // to the global output type derivation of `ShardMapOperation::from_program`), so the function signature carries
    // the boundary shardings and any requested physical layouts.
    let function_arguments = global_input_tensor_types
        .iter()
        .zip(&global_input_types)
        .zip(shard_map.in_shardings().iter())
        .map(|((tensor_type, array_type), sharding)| {
            let sharding = sharding.to_mlir(location)?;
            let mut attributes = HashMap::from([("sdy.sharding".into(), sharding.as_ref())]);
            if let Some(layout) = lower_boundary_layout(array_type, "input", &context)? {
                attributes.insert("mhlo.layout_mode".into(), layout);
            }
            Ok(TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: Some(attributes) })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let function_results = global_output_tensor_types
        .iter()
        .zip(&global_output_types)
        .zip(shard_map.out_shardings().iter())
        .map(|((tensor_type, array_type), sharding)| {
            let sharding = sharding.to_mlir(location)?;
            let mut attributes = HashMap::from([("sdy.sharding".into(), sharding.as_ref())]);
            if let Some(layout) = lower_boundary_layout(array_type, "output", &context)? {
                attributes.insert("mhlo.layout_mode".into(), layout);
            }
            Ok(TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: Some(attributes) })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;

    // Module-scoped collective lowering state: this entry point lowers one whole module.
    let collective_state = CollectiveLoweringState::new();
    module.body()?.append_operation({
        let function_block = context.block(
            global_input_tensor_types
                .iter()
                .map(|tensor_type| (*tensor_type, location))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let outer_inputs = (0..global_input_tensor_types.len())
            .map(|index| function_block.argument(index).expect("function block arguments should exist").as_ref())
            .collect::<Vec<_>>();
        let mut function_block_ref = function_block.as_ref();
        let manual_results = lower_manual_computation(
            &mut function_block_ref,
            outer_inputs.as_slice(),
            shard_map,
            program,
            local_input_types.as_slice(),
            global_output_types.as_slice(),
            &context,
            location.as_ref(),
            &collective_state,
            &mut EffectTokens::default(),
        )?;
        function_block_ref.append_operation(func::r#return(manual_results.as_slice(), location)?)?;

        let mut function_region = context.region();
        function_region.append_block(function_block)?;
        func::func(
            function_name.as_str(),
            func::FuncAttributes { arguments: function_arguments, results: function_results, ..Default::default() },
            function_region,
            location,
        )?
    })?;

    remove_unused_constants(&module)?;
    if !module.verify()? {
        return Err(LoweringError::MlirVerificationFailure);
    }

    serialize_lowered_module(&module, &collective_state)
}

/// Lowers one `sdy.manual_computation` operation, threading per-device effect tokens across its boundary. Assertions
/// remain local to an executable, and the only ordered I/O that reaches this function is per-device
/// (`DeviceOrderedIo`), because core shard-map boundary validation rejects bodies with the `OrderedIo` effect (refer to
/// [`ShardMapError::OrderedIoNotSupported`]).
pub(crate) fn lower_manual_computation<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    block: &mut BlockRef<'b, 'c, 't>,
    outer_inputs: &[ValueRef<'b, 'c, 't>],
    shard_map: &ShardMap,
    program: &XlaProgram<ProgramInput, ProgramOutput, O>,
    local_input_types: &[ArrayType],
    global_output_types: &[ArrayType],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
{
    // Shardy forbids a nested manual computation from listing an axis that an enclosing one already made manual. The
    // closure entry points never construct such a nesting, and boundary validation rejects a nested map whose inputs
    // vary along such an axis, but a checked or transformed program can still nest a map over invariant inputs this
    // way, so it is rejected here instead of emitting an invalid manual computation.
    if let Some(axis_name) =
        shard_map.manual_axes().iter().find(|axis| collective_state.bound_manual_axes.contains(*axis))
    {
        return Err(ProgramError::from(ShardMapError::AxisAlreadyManual { axis_name: axis_name.clone() }).into());
    }
    // For the same reason, an input or output sharding must not name such an axis (the closure entry points reject
    // these specifications too), so it is rejected here rather than silently dropped from the Shardy shardings.
    for (value_kind, shardings) in [("input", shard_map.in_shardings()), ("output", shard_map.out_shardings())] {
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
                let error = ShardMapError::SpecificationNamesEnclosingManualAxis {
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
        to_shardy_in_shardings(shard_map, &collective_state.bound_manual_axes, context)?.shardings();
    let mut output_shardings =
        to_shardy_out_shardings(shard_map, &collective_state.bound_manual_axes, context)?.shardings();
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
        let body_collective_state = collective_state.enter_manual_region(shard_map.clone());
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
        to_shardy_manual_axes(shard_map, context)?,
        body_region,
        location,
    )?)?;
    for (index, &effect) in threaded_effects.iter().enumerate() {
        effect_tokens.set(effect, manual_computation.result(global_output_types.len() + index).unwrap().as_ref());
    }
    manual_computation
        .results()
        .take(global_output_types.len())
        .map(|result| result.map(|result| result.as_ref()).map_err(LoweringError::from))
        .collect()
}

/// Builds the typed Shardy `in_shardings` attribute of `shard_map`, excluding axes already bound by enclosing manual
/// regions.
fn to_shardy_in_shardings<'c, 't>(
    shard_map: &ShardMap,
    enclosing_manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<TensorShardingPerValueAttributeRef<'c, 't>, ryft_mlir::Error> {
    shardy_tensor_sharding_per_value(shard_map.in_shardings(), shard_map.manual_axes(), enclosing_manual_axes, context)
}

/// Builds the typed Shardy `out_shardings` attribute of `shard_map`, excluding axes already bound by enclosing manual
/// regions.
fn to_shardy_out_shardings<'c, 't>(
    shard_map: &ShardMap,
    enclosing_manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<TensorShardingPerValueAttributeRef<'c, 't>, ryft_mlir::Error> {
    shardy_tensor_sharding_per_value(shard_map.out_shardings(), shard_map.manual_axes(), enclosing_manual_axes, context)
}

/// Builds the typed Shardy `manual_axes` attribute that `shard_map` attaches to `sdy.manual_computation`.
fn to_shardy_manual_axes<'c, 't>(
    shard_map: &ShardMap,
    context: &'c MlirContext<'t>,
) -> Result<ManualAxesAttributeRef<'c, 't>, ryft_mlir::Error> {
    context.shardy_manual_axes(shard_map.manual_axes())
}

/// Builds the typed Shardy per-value tensor sharding attribute of a `sdy.manual_computation` boundary, with one tensor
/// sharding per entry of `shardings` (refer to [`manual_computation_tensor_sharding`]).
fn shardy_tensor_sharding_per_value<'c, 't>(
    shardings: &[Sharding],
    manual_axes: &[String],
    enclosing_manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<TensorShardingPerValueAttributeRef<'c, 't>, ryft_mlir::Error> {
    let shardings = shardings
        .iter()
        .map(|sharding| manual_computation_tensor_sharding(sharding, manual_axes, enclosing_manual_axes, context))
        .collect::<Result<Vec<_>, _>>()?;
    context.shardy_tensor_sharding_per_value(shardings.as_slice())
}

/// Builds the typed Shardy tensor sharding of one `sdy.manual_computation` boundary value placed by `sharding`. The
/// axes of enclosing manual regions are omitted from the replicated axes, because they are already bound outside this
/// manual computation, and the dimension shardings are built by [`manual_computation_dimension_shardings`]. `sharding`
/// must not name such an axis in its dimension shardings or reduction state, which [`lower_manual_computation`] rejects
/// with [`ShardMapError::SpecificationNamesEnclosingManualAxis`] before building these attributes.
fn manual_computation_tensor_sharding<'c, 't>(
    sharding: &Sharding,
    manual_axes: &[String],
    enclosing_manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<TensorShardingAttributeRef<'c, 't>, ryft_mlir::Error> {
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
    context.shardy_tensor_sharding(
        mesh_symbol_ref,
        dimension_shardings.as_slice(),
        replicated_axes.as_slice(),
        unreduced_axes.as_slice(),
        shardy::ReductionOperation::Sum,
    )
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
) -> Result<Vec<DimensionShardingAttributeRef<'c, 't>>, ryft_mlir::Error> {
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
        .collect()
}

impl LowerableOperation for ShardMapOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [body] = lowerer.regions.as_slice() else {
            return Err(LoweringError::UnsupportedOp {
                op: format!(
                    "{} expected 1 attached region but got {}",
                    SHARD_MAP_OPERATION_NAME,
                    lowerer.regions.len()
                ),
            });
        };
        let simplified_body = body
            .simplified()
            .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
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
            self.shard_map(),
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
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Context, ConvertElementTypeOperation, CustomCallOperation, DataType, Device, DeviceMesh, Dimension, Dot,
        DotDimensionNumbers, EffectClass, MeshAxis, NamedAxis, Placeholder, ReductionKind, Shape, ValueProjection,
        shard_map_with_options, trace_shard_map, trace_shard_map_with_named_axes, trace_shard_map_with_options,
    };
    use ryft_mlir::Block;
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program as PjrtProgram, load_cpu_plugin};

    use crate::experimental::XlaArrayTracer;
    use crate::experimental::lowering::tests::{
        lower_replicated_shard_map_body, static_sharded_array_type, test_manual_mesh, test_sharding,
        test_spmd_compilation_options, test_vector_type,
    };
    use crate::experimental::ops::XlaProgramBuilder as ArrayIrXlaProgramBuilder;
    use crate::operations::debugging::PRINT_CUSTOM_CALL_TARGET;
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{FromPjrt, ToMlir, XlaArray, XlaSession};

    use super::*;

    /// Four-device mesh with the manual axes `x` and `y`.
    fn test_logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
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
    fn test_to_mlir_module_preserves_cast_layouts() {
        use ryft_core::{Cast, Layout, TiledLayout, Typed};
        use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput};

        use crate::ToPjrt;
        use crate::tests::execution_client;

        let mesh = test_manual_mesh("x", 1);
        let sharding = Sharding::replicated(mesh.clone(), 2);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let input_type = ArrayType::new_static(DataType::F32, [2, 3]).with_layout(layout.clone());
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |input| {
                let input = input.cast(input.r#type().into_owned().with_layout(None), &[]).unwrap();
                input.cast(input.r#type().into_owned().with_layout(layout.clone()), &[]).unwrap()
            },
            input_type,
            mesh,
            sharding.clone(),
            sharding,
        )
        .unwrap();
        let module = to_mlir_module(&traced, "main").unwrap();
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
            )
            .unwrap()
            .block_until_ready()
            .unwrap();
        let output = &outputs[0].outputs[0];
        assert_eq!(output.layout(), Ok(layout.to_pjrt().unwrap()));
        let bytes = output.copy_to_host(Some(ryft_pjrt::Layout::dense_major_to_minor(2))).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(&bytes), values);
    }

    #[test]
    fn test_to_mlir_module_renders_auto_axes_as_open_dimension_shardings() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(16)]));
        let mesh = test_logical_mesh_data_model();
        let input_sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], vec![]);
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |x| x.clone() + x,
            global_input_type.clone(),
            mesh.clone(),
            input_sharding.clone(),
            input_sharding,
        )
        .unwrap();
        assert_eq!(
            to_mlir_module(&traced, "main").unwrap(),
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
    fn test_to_mlir_module_renders_nested_shard_maps() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let inner_mesh = mesh.clone();
        let outer_sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let inner_sharding = test_sharding(&inner_mesh, vec![ShardingDimension::sharded(["y"])], vec![]);
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map_with_options(
            {
                let inner_mesh = inner_mesh.clone();
                let inner_sharding = inner_sharding.clone();
                move |x: XlaArrayTracer| {
                    let nested: XlaArrayTracer = shard_map_with_options(
                        |y: XlaArrayTracer| y.clone() + y,
                        x.clone(),
                        inner_mesh.clone(),
                        inner_sharding.clone(),
                        inner_sharding.clone(),
                        vec!["y".to_string()],
                    )
                    .expect("nested shard_map should trace");
                    nested + x
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)])),
            mesh,
            outer_sharding.clone(),
            outer_sharding,
            vec!["x".to_string()],
        )
        .unwrap();

        assert_eq!(
            to_mlir_module(&traced, "main").unwrap(),
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
    fn test_to_mlir_module_rejects_nested_manual_computations_over_already_manual_axes() {
        // Descriptor-only tracing has no enclosing context, so it can trace a nested body that makes the axis `x` of
        // the enclosing manual region manual again. The nested input is invariant along `x`, so boundary validation
        // cannot detect the nesting, and lowering rejects the nested manual computation instead of emitting an invalid
        // module.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![]);
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |x: XlaArrayTracer| {
                let inner: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
                    |y: XlaArrayTracer| y.clone() + y,
                    ArrayType::new_static(DataType::F32, [4]),
                    mesh.clone(),
                    replicated.clone(),
                    replicated.clone(),
                )
                .unwrap();
                let output = x
                    .value()
                    .context()
                    .bind(inner.operation().clone(), vec![inner.body().clone()], &[x.value().clone()])
                    .unwrap()
                    .remove(0);
                ValueProjection::<ArrayType>::into_projected(output).unwrap()
            },
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            replicated.clone(),
            replicated.clone(),
        )
        .unwrap();
        let expected = ShardMapError::AxisAlreadyManual { axis_name: "x".to_string() };
        assert!(matches!(
            to_mlir_module(&traced, "main"),
            Err(LoweringError::Tracing(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
    }

    #[test]
    fn test_to_mlir_module_rejects_specifications_naming_already_manual_axes() {
        // Descriptor-only tracing has no enclosing context, so it can trace a nested body over `y` whose input and
        // output shardings shard along the axis `x` of the enclosing manual region. Shardy cannot represent those
        // shardings inside that region, so lowering rejects them instead of silently dropping `x` from them.
        let mesh = test_logical_mesh_2x2();
        let replicated = test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![]);
        let sharded = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map_with_options(
            |x: XlaArrayTracer| {
                let inner: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map_with_options(
                    |y: XlaArrayTracer| y.clone() + y,
                    ArrayType::new_static(DataType::F32, [4]),
                    mesh.clone(),
                    sharded.clone(),
                    sharded.clone(),
                    vec!["y".to_string()],
                )
                .unwrap();
                let output = x
                    .value()
                    .context()
                    .bind(inner.operation().clone(), vec![inner.body().clone()], &[x.value().clone()])
                    .unwrap()
                    .remove(0);
                ValueProjection::<ArrayType>::into_projected(output).unwrap()
            },
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            replicated.clone(),
            replicated.clone(),
            vec!["x".to_string()],
        )
        .unwrap();
        let expected = ShardMapError::SpecificationNamesEnclosingManualAxis {
            value_kind: "input",
            value_index: 0,
            axis_name: "x".to_string(),
        };
        assert!(matches!(
            to_mlir_module(&traced, "main"),
            Err(LoweringError::Tracing(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
    }

    #[test]
    fn test_to_mlir_module_rejects_shard_maps_varying_along_enclosing_manual_axes() {
        // A body traced for the named-axis scope of an enclosing manual region over `x` describes a computation nested
        // in that region, so its global input varies along `x`. A standalone module has no manual region that makes
        // `x` manual, so lowering the traced shard map on its own is rejected.
        let mesh = test_logical_mesh_2x2();
        let sharded = test_sharding(&mesh, vec![ShardingDimension::sharded(["y"])], vec![]);
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map_with_named_axes(
            |_, y: XlaArrayTracer| y.clone() + y,
            input_type,
            mesh.clone(),
            sharded.clone(),
            sharded,
            Vec::new(),
            vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2 })],
        )
        .unwrap();
        let error = to_mlir_module(&traced, "main").unwrap_err();
        assert_eq!(
            error,
            LoweringError::StandaloneShardMapVariesAlongEnclosingManualAxis {
                value_kind: "input",
                value_index: 0,
                axis_name: "x".to_string(),
            },
        );
        assert_eq!(
            error.to_string(),
            "input type #0 of a standalone `shard_map` varies along mesh axis `x`, which is not one of its manual axes; \
             lower it as part of the program whose manual region makes that axis manual",
        );
    }

    #[test]
    fn test_to_mlir_module_rejects_shard_maps_communicating_along_enclosing_manual_axes() {
        use ryft_core::ParallelReduce;

        // A body traced for the named-axis scope of an enclosing manual region over `x` can reduce an input that does
        // not vary along `x` over `x`, so its boundary does not vary along `x` although the body communicates along it.
        // A standalone module has no manual region that makes `x` manual, so lowering the traced shard map on its own
        // is rejected. Collectives along the shard map's own manual axis `y` remain accepted.
        let mesh = test_logical_mesh_2x2();
        let sharded = test_sharding(&mesh, vec![ShardingDimension::sharded(["y"])], vec![]);
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let replicated = test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![]);
        let trace = |axis_name: &'static str, out_sharding: Sharding| {
            let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map_with_named_axes(
                |_, y: XlaArrayTracer| y.parallel_reduce(ReductionKind::Sum, axis_name).unwrap(),
                input_type.clone(),
                mesh.clone(),
                sharded.clone(),
                out_sharding,
                Vec::new(),
                vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
            )
            .unwrap();
            traced
        };
        assert!(to_mlir_module(&trace("y", replicated), "main").is_ok());
        let error = to_mlir_module(&trace("x", sharded.clone()), "main").unwrap_err();
        assert!(
            matches!(
                &error,
                LoweringError::StandaloneShardMapCommunicatesAlongEnclosingManualAxis { axis_name, .. }
                    if axis_name == "x",
            ),
            "{error:?}",
        );
        assert!(
            error.to_string().ends_with(
                "in the body of a standalone `shard_map` communicates along mesh axis `x`, which is not one of its \
                 manual axes; lower it as part of the program whose manual region makes that axis manual",
            ),
            "{error}",
        );
    }

    #[test]
    fn test_to_mlir_module_executes_on_cpu() {
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
        let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
            |x| x.clone() + x,
            global_input_type,
            device_mesh.logical_mesh().clone(),
            sharding.clone(),
            sharding.clone(),
        )
        .unwrap();
        let mlir_program = to_mlir_module(&traced, "main").unwrap();

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
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
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
    fn test_to_mlir_module_renders_and_executes_matmul_on_cpu() {
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
        let traced: TracedShardMap<XlaDomain<'static>, (ArrayType, ArrayType), ArrayType> = trace_shard_map(
            |(lhs, rhs)| lhs.dot(&rhs, &DotDimensionNumbers::matmul()).unwrap(),
            global_input_types,
            device_mesh.logical_mesh().clone(),
            (lhs_sharding.clone(), rhs_sharding.clone()),
            output_sharding.clone(),
        )
        .unwrap();
        let mlir_program = to_mlir_module(&traced, "main").unwrap();

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
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
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
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], Vec::new())],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            to_shardy_in_shardings(&shard_map, &[], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"x"}], replicated={"y"}>]>"#,
        );
    }

    #[test]
    fn test_to_shardy_in_shardings_leaves_inactive_manual_axes_open() {
        // Manual axes outside the selected subset are free axes of the manual computation, so they stay open, unless
        // an enclosing manual region already bound them, in which case they are omitted.
        let context = MlirContext::new();
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x", "y"])], Vec::new())],
            Vec::new(),
            vec!["x".into()],
        )
        .unwrap();
        assert_eq!(
            to_shardy_in_shardings(&shard_map, &[], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"x", "y", ?}]>]>"#,
        );
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["y"])], Vec::new())],
            Vec::new(),
            vec!["y".into()],
        )
        .unwrap();
        assert_eq!(
            to_shardy_in_shardings(&shard_map, &[], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"y", ?}], replicated={"x"}>]>"#,
        );
        assert_eq!(
            to_shardy_in_shardings(&shard_map, &["x".to_string()], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"y"}]>]>"#,
        );
    }

    #[test]
    fn test_to_shardy_in_shardings_renders_free_axes_as_open_dimension_shardings() {
        let context = MlirContext::new();
        let mesh = test_logical_mesh_data_model();
        let shard_map = ShardMap::new(
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
            to_shardy_in_shardings(&shard_map, &[], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{"data", ?}]>, <@mesh, [{"data", ?}]>]>"#,
        );
    }

    #[test]
    fn test_to_shardy_in_shardings_keeps_explicit_axes() {
        let context = MlirContext::new();
        let mesh = test_logical_mesh_data_model_explicit();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], Vec::new())],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            to_shardy_in_shardings(&shard_map, &[], &context).unwrap().to_string(),
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
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(&mesh, vec![ShardingDimension::replicated()], Vec::new())],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            to_shardy_out_shardings(&shard_map, &[], &context).unwrap().to_string(),
            r#"#sdy.sharding_per_value<[<@mesh, [{?}], replicated={"x"}>]>"#,
        );

        // Output shardings render exactly like input shardings.
        let mesh = test_logical_mesh_data_model();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["data"])], Vec::new());
        let shard_map = ShardMap::new(mesh, vec![sharding.clone()], vec![sharding], Vec::new()).unwrap();
        assert_eq!(
            to_shardy_out_shardings(&shard_map, &[], &context).unwrap().to_string(),
            to_shardy_in_shardings(&shard_map, &[], &context).unwrap().to_string(),
        );
    }

    #[test]
    fn test_to_shardy_manual_axes() {
        let context = MlirContext::new();
        let shard_map = ShardMap::new(
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
        assert_eq!(to_shardy_manual_axes(&shard_map, &context).unwrap().to_string(), r#"#sdy<manual_axes{"y"}>"#);
        let shard_map = ShardMap::new(test_logical_mesh_2x2(), Vec::new(), Vec::new(), vec!["x".into()]).unwrap();
        assert_eq!(to_shardy_manual_axes(&shard_map, &context).unwrap().to_string(), r#"#sdy<manual_axes{"x"}>"#);
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
        let shard_map =
            ShardMap::new(device_mesh.logical_mesh().clone(), vec![sharding.clone()], vec![sharding.clone()], vec![])
                .unwrap();
        assert_eq!(
            shard_map.local_input_type(0, &ArrayType::new_static(DataType::F32, [8])).unwrap().shape(),
            &Shape::new(vec![Dimension::Static(2)]),
        );

        let context = MlirContext::new();
        let input_sharding = shard_map.in_shardings()[0].to_mlir(context.unknown_location()).unwrap().to_string();
        let output_sharding = shard_map.out_shardings()[0].to_mlir(context.unknown_location()).unwrap().to_string();
        let manual_computation_attributes =
            r#"in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"}"#;
        let mesh_module = context.module(context.unknown_location()).unwrap();
        let mesh_operation = mesh_module
            .body()
            .unwrap()
            .append_operation(shard_map.mesh().to_mlir(context.unknown_location()).unwrap())
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
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
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
