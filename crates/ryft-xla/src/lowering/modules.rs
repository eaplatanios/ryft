//! Lowering of whole programs to StableHLO modules, including the entry function, its boundary attributes, the
//! private functions that calls share, and the passes that run over the emitted module.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use ryft_core::{
    ArrayType, BroadcastLikeOperation, BroadcastOperation, CallOperation, CastOperation, ConstrainShardingOperation,
    Device, Dimension, DynamicBroadcastOperation, DynamicReshapeOperation, EffectClass, ExternalReferenceBinding,
    FillLikeOperation, GatherOperation, IotaLikeOperation, IotaOperation, LogicalMesh, Memory, MeshAxisType,
    OneLikeOperation, OneOperation, Parameterized, PlaceOperation, PlacementRequest, Program, ProgramError,
    ReshapeOperation, ReshardOperation, ScatterOperation, Shape, ShardMapOperation, Sharding, ShardingDimension,
    Type as CoreType, Typed, TypedProgram, ZeroLikeOperation, ZeroOperation,
};
use ryft_mlir::dialects::{func, shardy, stable_hlo};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, DetachedBlock, Location, LocationRef, Module,
    Operation as MlirOperation, OperationPrintingFlags, Region, Type as MlirType, TypeAndAttributes,
    Value as MlirValue, WalkOrder, WalkResult,
};

use crate::arrays::host::ReductionChange;
use crate::devices::MemoryExtension;
use crate::lowering::emissions::{emitted_regions, partition_session_constants, unread_entry_arguments};
use crate::lowering::extensions::ShardingLoweringExtension;
use crate::lowering::reductions::{
    lower_pending_reduction_constraint, validate_declared_pending_reduction, validate_partial_completion,
    validate_pending_reduction_change,
};
use crate::lowering::regions::lower_program_outputs_with_inputs;
use crate::lowering::signatures::ExecutableSignature;
use crate::lowering::states::CollectiveLoweringState;
use crate::lowering::types::{
    lower_boundary_layout, lower_runtime_dimension_size_i64, lower_tensor_type, lower_unplaced_constant_output,
    validate_boundary_memory,
};
use crate::lowering::{EffectTokens, LowerableOperationFamily, LoweringError, RaggedDotLoweringStrategy};
use crate::operations::call::{CallFunctionMap, collect_call_functions, emit_call_function};
use crate::operations::composites::{collect_named_composition_functions, emit_named_composition_function};
use crate::sharding::{OpenDimensions, ShardySharding};
use crate::{
    CompilationError, Constant, Error, ExecutionError, RetainedArray, RuntimeRequirement, SHARDY_MESH_SYMBOL_NAME,
    ToMlir, Type, normalize_runtime_requirements,
};

// TODO(eaplatanios): Review this module.

/// Textual StableHLO module paired with the exact executable signature used to emit its entry function.
pub(crate) struct LoweredModule {
    /// Textual StableHLO/Shardy module.
    stable_hlo: String,

    /// Signature of the entry function, which maps its logical inputs and outputs to its physical arguments and
    /// results.
    signature: ExecutableSignature,

    /// Whether execution requires the host runtime assertion handler.
    requires_assertion_handler: bool,

    /// Canonical runtime contracts required by emitted calls and synthesized checks.
    runtime_requirements: Arc<[RuntimeRequirement]>,

    /// Module mesh that every sharding and `shard_map` boundary of the module refers to, or `None` when the module
    /// declares no mesh. Compilation checks it against the topology of the device mesh that it targets.
    mesh: Option<LogicalMesh>,
}

impl LoweredModule {
    /// Returns the rendered StableHLO module of this lowering, consuming it.
    #[inline]
    pub(crate) fn into_stable_hlo(self) -> String {
        self.stable_hlo
    }

    /// Returns the signature of the entry function, which maps its logical inputs and outputs to its physical arguments
    /// and results.
    #[inline]
    pub(crate) fn signature(&self) -> &ExecutableSignature {
        &self.signature
    }

    /// Returns the module mesh that every sharding and `shard_map` boundary of the module refers to, or `None` when the
    /// module declares no mesh. Compilation checks it against the topology of the device mesh that it targets.
    #[inline]
    pub(crate) fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    /// Consumes the textual module, executable signature, assertion classification, and runtime requirements.
    #[inline]
    pub(crate) fn into_parts(self) -> (String, ExecutableSignature, bool, Arc<[RuntimeRequirement]>) {
        (self.stable_hlo, self.signature, self.requires_assertion_handler, self.runtime_requirements)
    }
}

/// Returns `true` if any operation in a region of `program` that lowering emits (i.e., its entry region and the
/// executed nested regions of that region) carries unresolved ordered state.
///
/// This scans those regions rather than consulting the effect summary of the program, because ordinary XLA cannot
/// preserve state in any lowered region, including one whose effects the summary of its owner does not expose.
/// Regions that lowering never reaches (e.g., the rule regions of composites, the transpose region of a `linear_call`,
/// or the body of a scan that admits no iteration, refer to [`emitted_regions`], here for no target platform) are not
/// part of the lowered module, so state in them is never lowered and cannot reject the program.
pub(crate) fn contains_unresolved_state<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
) -> bool
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    emitted_regions(program.entry_region_ref(), None).any(|region| {
        region
            .instructions()
            .iter()
            .any(|instruction| instruction.operation().effects().classes().contains(EffectClass::OrderedState))
    })
}

/// Returns `true` if `program` contains a reference-typed atom or a [`CoreOperation`](ryft_core::Operation) whose
/// [effects](ryft_core::Operation::effects) declare a [`ReferenceEffect`](ryft_core::ReferenceEffect) in a region that
/// lowering emits (i.e., its entry region and the executed nested regions of that region, refer to [`emitted_regions`],
/// here for no target platform). Regions that lowering never reaches (e.g., the rule regions of composites, the
/// transpose region of a `linear_call`, or the body of a scan that admits no iteration) are not part of the lowered
/// module, so references in them neither need discharge nor reject the program.
///
/// This check is independent from [`contains_unresolved_state`], because a pure reference pass-through or a forwarded
/// reference capture can carry reference values without executing a stateful instruction. Both predicates back the
/// state checks that precede compilation and the two direct module lowering entry points. Eager binding, dispatch, and
/// staging enforce their corresponding boundary invariants separately.
pub(crate) fn contains_unresolved_references<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
) -> bool
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    // The atom scan dominates for every well-formed program, because reference operations always touch an atom of a
    // reference type. The declaration scan is an independent check that still catches a core discharge bug which
    // retyped the boundary atoms but left a reference operation behind.
    emitted_regions(program.entry_region_ref(), None).any(|region| {
        region.atoms().iter().any(|atom| atom.r#type().is_reference())
            || region
                .instructions()
                .iter()
                .any(|instruction| instruction.operation().effects().has_reference_effects())
    })
}

/// Serializes one lowered module to StableHLO text, printing MLIR debug information (i.e., locations) exactly when
/// the lowering constructed provenance locations specific to instructions. Modules without provenance keep the plain
/// rendering, so their StableHLO text, snapshots, and compilation cache keys remain byte for byte what they were before
/// provenance existed, while modules that carry provenance embed their locations in the text and therefore in the
/// persistent compilation key derived from it. The pretty debug form is documented as unparsable and never used, and
/// the elision thresholds are disabled because the plain rendering performs no elision either.
///
/// Exact backend provenance can reduce the reuse of cached compiled artifacts, because two semantically equal modules
/// with different provenance serialize differently and therefore have different keys. If that cost ever matters, the
/// correct extension is an explicit compilation option that strips provenance before lowering, which makes this helper
/// fall back to the plain rendering. Silently returning an artifact labeled with the provenance of another program is
/// never acceptable.
pub(crate) fn serialize_lowered_module(
    module: &Module<'_, '_>,
    collective_state: &CollectiveLoweringState,
) -> Result<String, Error> {
    if !collective_state.has_provenance() {
        return Ok(module.to_string());
    }
    module
        .as_operation()?
        .to_string_with_flags(OperationPrintingFlags {
            elements_attribute_size_threshold: None,
            resource_string_size_threshold: None,
            enable_debug_information: true,
            pretty_print_debug_information: false,
            ..OperationPrintingFlags::default()
        })
        .map_err(|error| ryft_mlir::Error::internal(format!("serialized module is not valid UTF-8: {error}")).into())
}

/// Lowers a traced XLA program of any [`LowerableOperationFamily`] to a textual StableHLO/Shardy MLIR module whose
/// entry function is named `function_name`.
///
/// When `argument_shardings` and/or `output_shardings` are provided, the corresponding `sdy.sharding` attribute is
/// attached to each function argument or result, which drives per-device boundary slicing (e.g., for uneven splits) in
/// the XLA SPMD partitioner. When they are `None`, the corresponding values carry no sharding attributes, and Shardy
/// propagates their placement from the program.
///
/// # Parameters
///
///   - `program`: Traced program to lower. References and mutable state must have been discharged.
///   - `capture_types`: Types of the hidden capture arguments that precede the program inputs, in capture-table order.
///   - `global_input_types`: Global types of the program inputs.
///   - `global_output_types`: Global types of the program outputs.
///   - `function_name`: Symbol name of the entry function.
///   - `argument_shardings`: Optional shardings of the entry function arguments.
///   - `output_shardings`: Optional shardings of the entry function results.
///
/// # Errors
///
/// Returns [`LoweringError::UnresolvedReference`] or [`LoweringError::UnresolvedState`] for programs with undischarged
/// references or state, [`LoweringError::InvalidFunctionName`] for an invalid `function_name`, and other
/// [`Error`]s for operations or types that cannot be lowered.
pub fn to_mlir_module_for_program<Input, Output, ProgramInput, ProgramOutput, S, O: LowerableOperationFamily>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    argument_shardings: Option<&[Sharding]>,
    output_shardings: Option<&[Sharding]>,
) -> Result<String, Error>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
    S: AsRef<str>,
{
    Ok(lower_mlir_module_for_program(
        program,
        capture_types,
        global_input_types,
        global_output_types,
        function_name,
        argument_shardings,
        output_shardings,
        None,
    )?
    .stable_hlo)
}

/// Lowers an arbitrary traced XLA program and returns both its textual module and exact physical entry signature.
pub(crate) fn lower_mlir_module_for_program<
    Input,
    Output,
    ProgramInput,
    ProgramOutput,
    S,
    O: LowerableOperationFamily,
>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    argument_shardings: Option<&[Sharding]>,
    output_shardings: Option<&[Sharding]>,
    target_platform: Option<&str>,
) -> Result<LoweredModule, Error>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
    S: AsRef<str>,
{
    let argument_count = capture_types.len() + global_input_types.parameter_count();
    lower_mlir_module_for_program_with_effect_boundary(
        program,
        capture_types,
        global_input_types,
        global_output_types,
        function_name,
        &value_shardings(argument_shardings, argument_count),
        &value_shardings(output_shardings, global_output_types.parameter_count()),
        target_platform,
        true,
        &[],
        &[],
        &[],
        RaggedDotLoweringStrategy::default(),
        &[],
        false,
    )
}

/// Lowers an arbitrary traced XLA program like [`lower_mlir_module_for_program`], but with one optional sharding per
/// entry function argument (i.e., captures first) and per result, so that only the values whose sharding is known carry
/// an `sdy.sharding` attribute, while Shardy propagates the placement of the others from the program. The results
/// flagged in `open_outputs` (i.e., one flag per result, or none at all) are lowered open over auto mesh axes, so that
/// the compiler chooses their placement over those axes (refer to `entry_outputs_with_compiler_chosen_placement`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_mlir_module_for_program_with_value_shardings<
    Input,
    Output,
    ProgramInput,
    ProgramOutput,
    S,
    O: LowerableOperationFamily,
>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    argument_shardings: &[Option<Sharding>],
    output_shardings: &[Option<Sharding>],
    open_outputs: &[bool],
) -> Result<LoweredModule, Error>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
    S: AsRef<str>,
{
    lower_mlir_module_for_program_with_effect_boundary(
        program,
        capture_types,
        global_input_types,
        global_output_types,
        function_name,
        argument_shardings,
        output_shardings,
        None,
        true,
        &[],
        &[],
        &[],
        RaggedDotLoweringStrategy::default(),
        open_outputs,
        false,
    )
}

/// Lowers a discharged program with logical external reference state aliases on its entry boundary. The results
/// flagged in `open_outputs` (i.e., one flag per logical output, or none at all) are lowered open over auto mesh axes,
/// so that the compiler chooses their placement over those axes.
///
/// `reference_states` are the external state recipes of the discharge artifact, and each must name a distinct logical
/// state input. Each state input must survive the executable boundary. A mutated state must be a static device array
/// and name a hidden output of the identical type and sharding. Mutated pairs are recorded as `tf.aliasing_output`
/// aliases between inputs and outputs, which are non-semantic hints that the buffers may alias and that merely permit
/// backend buffer reuse. The runtime never donates reference state inputs, so an alias never authorizes in-place
/// mutation of the state buffer of the caller. Read-only state may be finite and bounded dynamic because it has no
/// hidden alias. The validation here owns the physical alias contract, and the XLA session additionally enforces the
/// supported memory, sharding, and dynamic shape classes before calling it.
///
/// `donated_inputs` flags the public inputs whose buffers the caller gives up to every execution (i.e., one flag per
/// public input, or none at all). Each donated input is paired with an output that it can share its buffers with and
/// the pair is recorded as the same kind of alias, as JAX does for the donated arguments of a `jit` function. An
/// output can share the buffers of an input when both are static device arrays of the identical type and sharding,
/// or, for an input without such an output, when both have shardings under which every device holds shards of the
/// same type (e.g., a donated input with a pending sum and an output that completes it, as in the compiled reshard of
/// a donated array), and in both cases only when the output is not open on a mesh of more than one device, because
/// the compiler could then place it differently than the input. Without such an alias the outputs of every execution
/// are written to new buffers, which is correct but forces backends that cache launch state per buffer address (e.g.,
/// the command buffers of the XLA GPU backend) to rebuild that state for every execution of a program that updates
/// its inputs in a loop.
///
/// `unread_inputs` flags the public inputs that the program never reads (i.e., one flag per public input, or none at
/// all), which the caller states because it cannot be read from `program` (e.g., an input of the program of an eager
/// call that the call passes as a capture which its callee never reads, refer to `unread_lifted_captures`). Such an
/// input with a static shape has no executable argument, like a capture that no emitted region reads (refer to
/// `emitted_capture_reads`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_mlir_module_for_program_with_reference_state<
    Input,
    Output,
    ProgramInput,
    ProgramOutput,
    S,
    O: LowerableOperationFamily,
>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    argument_shardings: Option<&[Sharding]>,
    output_shardings: Option<&[Sharding]>,
    target_platform: Option<&str>,
    supports_shardy: bool,
    reference_states: &[ExternalReferenceBinding],
    donated_inputs: &[bool],
    unread_inputs: &[bool],
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,
    open_outputs: &[bool],
) -> Result<LoweredModule, Error>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
    S: AsRef<str>,
{
    let argument_count = capture_types.len() + global_input_types.parameter_count();
    lower_mlir_module_for_program_with_effect_boundary(
        program,
        capture_types,
        global_input_types,
        global_output_types,
        function_name,
        &value_shardings(argument_shardings, argument_count),
        &value_shardings(output_shardings, global_output_types.parameter_count()),
        target_platform,
        supports_shardy,
        reference_states,
        donated_inputs,
        unread_inputs,
        ragged_dot_lowering_strategy,
        open_outputs,
        true,
    )
}

/// Renders `devices` as a list in diagnostics (e.g., `[0, 1]`), naming the process of each device that is not on
/// process zero (e.g., `[0, 5@1]`, refer to the [`Display`](std::fmt::Display) implementation of [`Device`]), so that
/// devices of different processes with the same identifier are told apart.
pub(crate) fn render_devices(devices: &[Device]) -> String {
    format!("[{}]", devices.iter().map(Device::to_string).collect::<Vec<_>>().join(", "))
}

/// Collects the module mesh from the meshes referenced by emitted sharding constraints and nested-computation
/// boundaries, starting from `existing`, and rejects every mesh whose topology differs from it with
/// [`LoweringError::MismatchedMeshTopology`]. The constraints include those of the boundary overrides of `call`s,
/// whose shardings no type records. A constraint that this function does not recognize but that lowering emits over
/// the sharding of a type still finds its mesh, because the module declares the mesh of those types whenever the
/// lowered module refers to a mesh that this function did not find (refer to `declare_constrained_mesh`).
pub(crate) fn collect_nested_sharding_mesh<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    existing: Option<LogicalMesh>,
    target_platform: Option<&str>,
) -> Result<Option<LogicalMesh>, Error>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    // Only the computation regions that lowering emits for `target_platform` contribute, so dormant rule regions (e.g.,
    // the transpose region of a `linear_call`) and the regions that lowering omits (refer to `omits_regions`, which the
    // validation of placements reads as well) cannot bring other meshes into the module.
    // Type metadata alone does not require a mesh, because bounded dynamic replica lowering retains that metadata
    // without emitting Shardy operations.
    let mut mesh = existing;
    for region in emitted_regions(program.entry_region_ref(), target_platform) {
        for instruction in region.instructions() {
            // Payloads are recognized through their projections, so any family that embeds them declares their meshes.
            let operation = instruction.operation();
            let declared_mesh = operation
                .projected_payload::<ShardMapOperation>()
                .map(|operation| operation.boundary().mesh())
                .or_else(|| {
                    operation.projected_payload::<ReshardOperation>().map(|operation| operation.sharding().mesh())
                })
                .or_else(|| {
                    operation
                        .projected_payload::<ConstrainShardingOperation>()
                        .map(|operation| operation.sharding().mesh())
                })
                .or_else(|| {
                    // A device target names no mesh. It lowers to a sharding constraint only when it completes a
                    // reduction, over the mesh of its result type, which `declare_constrained_mesh` declares.
                    operation
                        .projected_payload::<PlaceOperation>()
                        .and_then(PlaceOperation::target)
                        .and_then(PlacementRequest::placement)
                        .map(|target| target.sharding().mesh())
                })
                .or_else(|| {
                    // A call constrains its operands and results that the boundary of its callee overrides.
                    operation
                        .projected_payload::<CallOperation<Type>>()
                        .and_then(|call| call.boundary_shardings().next())
                        .map(Sharding::mesh)
                });
            let incoming = if let Some(mesh) = declared_mesh {
                Some(mesh.clone())
            } else {
                let output_axes = operation
                    .projected_payload::<DynamicBroadcastOperation>()
                    .map(DynamicBroadcastOperation::output_axes)
                    .or_else(|| {
                        operation.projected_payload::<BroadcastOperation>().map(BroadcastOperation::output_axes)
                    })
                    .or_else(|| {
                        operation.projected_payload::<BroadcastLikeOperation>().map(BroadcastLikeOperation::output_axes)
                    });
                let emits_constraint = if let Some(output_axes) = output_axes {
                    let input_type = region.atoms()[instruction.inputs()[0].index()].r#type();
                    let input_type = <&ArrayType>::try_from(input_type.as_ref()).map_err(ProgramError::from)?;
                    let output_type = region.atoms()[instruction.outputs()[0].index()].r#type();
                    let output_type = <&ArrayType>::try_from(output_type.as_ref()).map_err(ProgramError::from)?;
                    broadcast_changes_explicit_sharding(input_type, output_type, output_axes)
                } else if operation.projected_payload::<CastOperation<Type>>().is_some()
                    || operation.projected_payload::<CastOperation<ArrayType>>().is_some()
                {
                    // Casts of first-class dimensions carry no placement, and array casts constrain placement only
                    // when the explicit placement changes.
                    let input_type = region.atoms()[instruction.inputs()[0].index()].r#type();
                    let output_type = region.atoms()[instruction.outputs()[0].index()].r#type();
                    match (<&ArrayType>::try_from(input_type.as_ref()), <&ArrayType>::try_from(output_type.as_ref())) {
                        (Ok(input_type), Ok(output_type)) => cast_changes_explicit_placement(input_type, output_type),
                        _ => false,
                    }
                } else {
                    operation
                        .projected_payload::<DynamicReshapeOperation>()
                        .is_some_and(|operation| operation.output_sharding().is_some())
                        || operation
                            .projected_payload::<ReshapeOperation>()
                            .is_some_and(|operation| operation.output_sharding().is_some())
                        || operation
                            .projected_payload::<GatherOperation>()
                            .is_some_and(|operation| operation.output_sharding().is_some())
                        || operation
                            .projected_payload::<ScatterOperation>()
                            .is_some_and(|operation| operation.output_sharding().is_some())
                        || operation.projected_payload::<ZeroOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<OneOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<IotaOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<ZeroLikeOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<OneLikeOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<FillLikeOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<IotaLikeOperation<ArrayType>>().is_some()
                };
                if !emits_constraint {
                    continue;
                }
                let output_type = region.atoms()[instruction.outputs()[0].index()].r#type();
                let output_type = <&ArrayType>::try_from(output_type.as_ref()).map_err(ProgramError::from)?;
                output_type.sharding().map(|sharding| sharding.mesh().clone())
            };
            if let Some(incoming) = incoming {
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => merge_logical_meshes(existing_mesh, &incoming)?,
                    None => incoming,
                });
            }
        }
    }
    Ok(mesh)
}

/// Returns whether a broadcast from `input_type` to `output_type`, which maps each input axis to the corresponding
/// entry of `output_axes`, changes the placement over explicit mesh axes and therefore requires an explicit output
/// placement constraint.
pub(crate) fn broadcast_changes_explicit_sharding(
    input_type: &ArrayType,
    output_type: &ArrayType,
    output_axes: &[usize],
) -> bool {
    let Some(output_sharding) = output_type.sharding() else {
        return false;
    };
    let input_sharding = input_type.sharding();
    let uses_explicit_axis = |dimension: &ShardingDimension, sharding: &Sharding| match dimension {
        ShardingDimension::Sharded(axis_names) => {
            axis_names.iter().any(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Explicit))
        }
        ShardingDimension::Replicated | ShardingDimension::Unconstrained => false,
    };
    if input_sharding.is_none() {
        return output_sharding.mesh().axes().iter().any(|axis| axis.r#type() == MeshAxisType::Explicit);
    }
    if let Some(input_sharding) = input_sharding
        && input_sharding.mesh() != output_sharding.mesh()
    {
        return input_sharding
            .mesh()
            .axes()
            .iter()
            .chain(output_sharding.mesh().axes())
            .any(|axis| axis.r#type() == MeshAxisType::Explicit);
    }

    let projected_input_sharding = input_sharding
        .map(|sharding| sharding.with_broadcasted_dimensions(output_type.rank(), output_axes).unwrap())
        .unwrap_or_else(|| Sharding::replicated(output_sharding.mesh().clone(), output_type.rank()));

    // Even when the ranked placement labels are unchanged, expanding a unit dimension that is partitioned over an
    // explicit mesh axis changes which partitions own the replicated values and must remain visible to Shardy.
    let expands_explicit_dimension = output_axes.iter().copied().enumerate().any(|(input_axis, output_axis)| {
        input_type.dimension(input_axis as isize) == Dimension::Static(1)
            && output_type.dimension(output_axis as isize) != Dimension::Static(1)
            && (uses_explicit_axis(&projected_input_sharding.dimensions()[output_axis], &projected_input_sharding)
                || uses_explicit_axis(&output_sharding.dimensions()[output_axis], output_sharding))
    });

    expands_explicit_dimension || projected_input_sharding.conflicts_on_explicit_axes_with(output_sharding)
}

/// Returns whether a cast from `input_type` to `output_type` changes the placement that the type system tracks over
/// explicit mesh axes and therefore emits a sharding constraint carrying its complete output sharding, like
/// [`ReshardOperation`]. Carrying an existing sharding forward requests no
/// redistribution, and an output without a sharding is untracked rather than replicated. Type inference guarantees
/// that the placement over auto and manual axes is unchanged, as documented on [`CastOperation`], so the emitted
/// constraint never contradicts the type the program was checked against.
pub(crate) fn cast_changes_explicit_placement(input_type: &ArrayType, output_type: &ArrayType) -> bool {
    let Some(output_sharding) = output_type.sharding() else {
        return false;
    };
    let has_explicit_axis =
        |sharding: &Sharding| sharding.mesh().axes().iter().any(|axis| axis.r#type() == MeshAxisType::Explicit);
    match input_type.sharding() {
        None => has_explicit_axis(output_sharding),
        Some(input_sharding) if input_sharding.mesh() != output_sharding.mesh() => {
            has_explicit_axis(input_sharding) || has_explicit_axis(output_sharding)
        }
        Some(input_sharding) => input_sharding.conflicts_on_explicit_axes_with(output_sharding),
    }
}

/// Removes every `stablehlo.constant` operation whose result has no uses from `module`, including those in nested
/// regions. Lowering materializes first-class dimension extents eagerly (refer to
/// [`lower_dimension_extent`](crate::lowering::types::lower_dimension_extent)), while their consumers
/// often read them only for dynamic axes (e.g., the explicit-extent collectives only refine dynamic result axes), so a
/// static extent would otherwise remain in the lowered module as a dead constant. Constants are pure, so removing an
/// unused one never changes the semantics of the module.
pub(crate) fn remove_unused_constants(module: &Module<'_, '_>) -> Result<(), Error> {
    remove_unused_constants_in_block(module.body()?)
}

/// Removes every Shardy operation from `module`, including those in nested regions, for a plugin that cannot compile
/// with Shardy. Each `sdy.sharding_constraint` is replaced by its operand and the `sdy.mesh` declaration is erased.
/// Ryft runs programs on such targets on a single device. Every sharding then holds the whole array, so removing
/// these constraints preserves the result.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] for any other Shardy operation (e.g., the `sdy.manual_computation`
/// of a `shard_map`), which has no equivalent without the dialect.
pub(crate) fn remove_shardy_operations(module: &Module<'_, '_>) -> Result<(), Error> {
    remove_shardy_operations_in_block(module.body()?)
}

/// Normalizes an MLIR symbol name provided by the user by trimming surrounding whitespace and one leading `@`.
///
/// # Errors
///
/// Returns [`LoweringError::InvalidFunctionName`] when the normalized name is empty or contains whitespace.
pub(crate) fn normalize_function_name(function_name: &str) -> Result<String, Error> {
    let function_name = function_name.trim();
    let function_name = function_name.strip_prefix('@').unwrap_or(function_name);
    if function_name.is_empty() || function_name.chars().any(char::is_whitespace) {
        return Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name: function_name.to_string() }));
    }
    Ok(function_name.to_string())
}

/// Returns whether every device holds shards of the same type of a static input of `input_type` placed with
/// `input_sharding` and of an output of `output_type` placed with `output_sharding`, although the two shardings may
/// differ (e.g., when the output completes a pending sum of the input along a mesh axis, or splits a dimension along
/// another axis of the same size). The output can then take the buffers of the donated input on every device, as JAX
/// pairs a donated argument of a `jit` function with a result of the same shard shape. That is the case when both
/// shardings are on the same mesh, the types are identical apart from their shardings (i.e., in data type, shape,
/// layout, and memory), and both shardings split each dimension into shards of the same static extent.
fn has_same_shard_types(
    input_type: &ArrayType,
    input_sharding: Option<&Sharding>,
    output_type: &ArrayType,
    output_sharding: Option<&Sharding>,
) -> bool {
    let (Some(input_sharding), Some(output_sharding), Some(shape)) =
        (input_sharding, output_sharding, input_type.static_shape())
    else {
        return false;
    };
    // The extent of the shards of each dimension, or `None` when a dimension is unconstrained, names an axis that the
    // mesh does not have, or is not split evenly.
    let shard_extents = |sharding: &Sharding| {
        if sharding.rank() != shape.rank() {
            return None;
        }
        shape
            .dimensions()
            .iter()
            .zip(sharding.dimensions())
            .map(|(&extent, dimension)| match dimension {
                ShardingDimension::Replicated => Some(extent),
                ShardingDimension::Sharded(axes) => {
                    let count = axes.iter().map(|axis| sharding.mesh().axis_size(axis)).product::<Option<usize>>()?;
                    (count != 0 && extent % count == 0).then(|| extent / count)
                }
                ShardingDimension::Unconstrained => None,
            })
            .collect::<Option<Vec<_>>>()
    };
    let without_sharding = |r#type: &ArrayType| r#type.clone().with_sharding(None).ok();
    input_sharding.mesh() == output_sharding.mesh()
        && without_sharding(input_type) == without_sharding(output_type)
        && shard_extents(input_sharding).is_some_and(|extents| Some(extents) == shard_extents(output_sharding))
}

/// Expands optional shardings of every value of a signature with `count` values into one optional sharding per value.
/// Provided shardings are kept as they are, so that a count mismatch is still reported when the module is lowered.
fn value_shardings(shardings: Option<&[Sharding]>, count: usize) -> Vec<Option<Sharding>> {
    match shardings {
        Some(shardings) => shardings.iter().cloned().map(Some).collect(),
        None => vec![None; count],
    }
}

/// Boundary of the entry function of a lowered module, which [`plan_entry_boundary`] derives from a program before
/// anything is emitted.
struct EntryBoundary {
    /// Name of the entry function.
    function_name: String,

    /// Signature that maps the logical boundary of the program to the physical boundary of its executable.
    signature: ExecutableSignature,

    /// Session constants of the program, which the entry function receives as its trailing capture arguments.
    session_constants: Vec<RetainedArray>,

    /// Arrays that the program only passes on to calls that never read them, which the entry function binds to values
    /// that require no buffer.
    passed_on_constants: Vec<RetainedArray>,

    /// Types of the logical arguments (i.e., of the captures followed by the public inputs).
    logical_argument_types: Vec<ArrayType>,

    /// Types of the arguments of the executable.
    physical_argument_types: Vec<ArrayType>,

    /// Types of the results of the executable.
    physical_output_types: Vec<ArrayType>,

    /// Physical result that each physical argument may share its buffers with, by the index of the argument.
    aliases: HashMap<usize, usize>,
}

/// Validates the entry boundary of `program` and returns its [`EntryBoundary`], without emitting anything. The
/// arguments are those of [`lower_mlir_module_for_program_with_effect_boundary`].
#[allow(clippy::too_many_arguments)]
fn plan_entry_boundary<Input, Output, ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: &str,
    argument_shardings: &[Option<Sharding>],
    output_shardings: &[Option<Sharding>],
    target_platform: Option<&str>,
    reference_states: &[ExternalReferenceBinding],
    donated_inputs: &[bool],
    unread_inputs: &[bool],
    open_outputs: &[bool],
    export_effects: bool,
) -> Result<EntryBoundary, Error>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    if contains_unresolved_references(program) {
        return Err(Error::Lowering(LoweringError::UnresolvedReference { construct: "program".to_string() }));
    }
    if contains_unresolved_state(program) {
        return Err(Error::Lowering(LoweringError::UnresolvedState { construct: "program".to_string() }));
    }
    let function_name = normalize_function_name(function_name)?;
    // Session constants are carried by hidden arguments that form the suffix of the capture arguments. The caller
    // declares them through the trailing `capture_types`, in the order of `collect_session_constants`.
    let (session_constants, passed_on_constants) =
        partition_session_constants(program.entry_region_ref(), target_platform);
    let session_count = session_constants.len();
    if session_count > 0
        && (capture_types.len() < session_count
            || !capture_types[capture_types.len() - session_count..].iter().eq(session_constants
                .iter()
                .map(|constant| constant.r#type().into_owned())
                .collect::<Vec<_>>()
                .iter()))
    {
        return Err(ProgramError::MalformedProgram(format!(
            "the trailing capture types of the entry function must be the types of its {session_count} session \
             constant(s)",
        ))
        .into());
    }
    let global_input_types = global_input_types.parameters().cloned().collect::<Vec<_>>();
    let global_output_types = global_output_types.parameters().cloned().collect::<Vec<_>>();
    let logical_argument_types =
        capture_types.iter().cloned().chain(global_input_types.iter().cloned()).collect::<Vec<_>>();
    let effects = program.effects().classes();
    let ordered_io = effects.contains(EffectClass::OrderedIo) || effects.contains(EffectClass::DeviceOrderedIo);
    // An argument that the executable never reads (refer to `unread_entry_arguments`) has no executable argument, so
    // that executions neither admit, require, nor wait for its buffers, and the signature records it as unread.
    let reference_state_inputs = reference_states
        .iter()
        .filter_map(|state| state.source().flat_input_index(capture_types.len()).ok())
        .collect::<Vec<_>>();
    let unread_arguments = unread_entry_arguments(
        program,
        &logical_argument_types,
        capture_types.len(),
        session_count,
        unread_inputs,
        &reference_state_inputs,
        target_platform,
    );
    let signature = ExecutableSignature::new(logical_argument_types.as_slice(), global_output_types.as_slice())
        .with_unread_inputs(&unread_arguments)
        .with_effects(
            export_effects && ordered_io,
            effects.contains(EffectClass::UnorderedIo),
            effects.contains(EffectClass::OrderedAssertion),
            !effects.contains(EffectClass::OrderedIo),
        );
    if argument_shardings.len() != logical_argument_types.len() {
        return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
            kind: "argument",
            expected: logical_argument_types.len(),
            actual: argument_shardings.len(),
        }));
    }
    if output_shardings.len() != global_output_types.len() {
        return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
            kind: "output",
            expected: global_output_types.len(),
            actual: output_shardings.len(),
        }));
    }
    for (sharding, output_type) in output_shardings.iter().zip(&global_output_types) {
        if let Some(sharding) = sharding {
            let data_type = output_type.data_type();
            validate_declared_pending_reduction(sharding, output_type.sharding(), data_type, &[], "output_shardings")?;
        }
    }
    let aliases = entry_aliases(
        reference_states,
        donated_inputs,
        open_outputs,
        capture_types.len(),
        &logical_argument_types,
        &global_output_types,
        &signature,
        argument_shardings,
        output_shardings,
    )?;
    let physical_argument_types = signature.physical_input_types(logical_argument_types.as_slice());
    let physical_output_types = signature.physical_output_types(global_output_types.as_slice());
    Ok(EntryBoundary {
        function_name,
        signature,
        session_constants,
        passed_on_constants,
        logical_argument_types,
        physical_argument_types,
        physical_output_types,
        aliases,
    })
}

/// Returns the physical result that each physical argument of an entry function may share its buffers with, by the
/// index of the argument. The aliases of `reference_states` are validated and paired first, and each donated input is
/// then paired with a result that no earlier argument was paired with. The first `capture_count` logical arguments
/// are captures.
///
/// # Errors
///
/// Returns [`LoweringError::InvalidReferenceStateAbi`] for a reference state whose input and output cannot alias, and
/// [`ExecutionError::DonationFlagCountMismatch`] when `donated_inputs` does not have one flag for each public input.
#[allow(clippy::too_many_arguments)]
fn entry_aliases(
    reference_states: &[ExternalReferenceBinding],
    donated_inputs: &[bool],
    open_outputs: &[bool],
    capture_count: usize,
    logical_argument_types: &[ArrayType],
    global_output_types: &[ArrayType],
    signature: &ExecutableSignature,
    argument_shardings: &[Option<Sharding>],
    output_shardings: &[Option<Sharding>],
) -> Result<HashMap<usize, usize>, Error> {
    let mut aliases = HashMap::with_capacity(reference_states.len());
    let mut state_inputs = HashSet::with_capacity(reference_states.len());
    let mut aliased_outputs = HashSet::with_capacity(reference_states.len());
    for state in reference_states {
        let logical_input_index = state.source().flat_input_index(capture_count)?;
        if !state_inputs.insert(logical_input_index) {
            return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                message: format!("logical state input {logical_input_index} appears more than once"),
            }));
        }
        let input_type = logical_argument_types.get(logical_input_index).ok_or_else(|| {
            Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                message: format!("logical input index {logical_input_index} is out of range"),
            })
        })?;
        let physical_input_index = signature.input_mapping()[logical_input_index].ok_or_else(|| {
            Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                message: format!("logical state input {logical_input_index} is erased from the executable boundary"),
            })
        })?;
        if let Some(logical_output_index) = state.output_index() {
            let output_type = global_output_types.get(logical_output_index).ok_or_else(|| {
                Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!("logical output index {logical_output_index} is out of range"),
                })
            })?;
            let physical_output_index = signature.output_mapping()[logical_output_index].ok_or_else(|| {
                Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "logical state output {logical_output_index} is erased from the executable boundary",
                    ),
                })
            })?;
            if input_type.static_shape().is_none() || output_type.static_shape().is_none() {
                return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "state input {logical_input_index} and output {logical_output_index} must be static because \
                         bounded-dynamic mutation alias compatibility is unsupported",
                    ),
                }));
            }
            if input_type.memory() != Memory::Device || output_type.memory() != Memory::Device {
                return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "state input {logical_input_index} and output {logical_output_index} must use device \
                         memory",
                    ),
                }));
            }
            if input_type != output_type {
                return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "state input {logical_input_index} type `{input_type}` is incompatible with output \
                         {logical_output_index} type `{output_type}`",
                    ),
                }));
            }
            match (&argument_shardings[logical_input_index], &output_shardings[logical_output_index]) {
                (Some(argument_sharding), Some(output_sharding)) if argument_sharding != output_sharding => {
                    return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                        message: format!(
                            "state input {logical_input_index} and output {logical_output_index} must use the same \
                             sharding",
                        ),
                    }));
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                        message:
                            "reference-state aliases require both argument and result sharding metadata or neither"
                                .to_string(),
                    }));
                }
                _ => {}
            }
            if !aliased_outputs.insert(physical_output_index) {
                return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: "reference-state aliases must be injective".to_string(),
                }));
            }
            // Distinct logical inputs, which `state_inputs` enforces above, map to distinct physical inputs, so this
            // insertion can never displace an earlier alias.
            aliases.insert(physical_input_index, physical_output_index);
        }
    }
    if !donated_inputs.is_empty() && donated_inputs.len() != logical_argument_types.len() - capture_count {
        return Err(Error::Execution(ExecutionError::DonationFlagCountMismatch {
            expected_count: logical_argument_types.len() - capture_count,
            actual_count: donated_inputs.len(),
        }));
    }
    // Each donated input is paired with the first output that can share its buffers and that no earlier input was
    // paired with, as JAX pairs the donated arguments of a `jit` function with its results. The pairs are the same
    // hints as the reference state aliases above, so an input that finds no output simply stays unpaired. Every
    // donated input is first paired with an output of its own type and sharding, and only an input that finds none is
    // then paired with an output whose shards have the types of its shards (refer to `has_same_shard_types`), so that
    // the second pass only adds pairs to inputs that the first one leaves unpaired.
    for same_sharding in [true, false] {
        for (input_index, _) in donated_inputs.iter().enumerate().filter(|(_, donated)| **donated) {
            let logical_input_index = capture_count + input_index;
            let input_type = &logical_argument_types[logical_input_index];
            let Some(physical_input_index) = signature.input_mapping()[logical_input_index] else { continue };
            if state_inputs.contains(&logical_input_index)
                || aliases.contains_key(&physical_input_index)
                || input_type.static_shape().is_none()
                || input_type.memory() != Memory::Device
            {
                continue;
            }
            let input_sharding = &argument_shardings[logical_input_index];
            // The compiler places an open output itself, and only on a single device is that placement certain to be
            // the placement of the input.
            let has_single_device = input_sharding
                .as_ref()
                .is_none_or(|sharding| sharding.mesh().axes().iter().map(|axis| axis.size()).product::<usize>() == 1);
            let physical_output_index =
                global_output_types.iter().enumerate().find_map(|(logical_output_index, output_type)| {
                    let physical_output_index = signature.output_mapping()[logical_output_index]?;
                    let is_open = open_outputs.get(logical_output_index).copied().unwrap_or(false);
                    let output_sharding = &output_shardings[logical_output_index];
                    let shares_buffers = match same_sharding {
                        true => output_type == input_type && output_sharding == input_sharding,
                        false => has_same_shard_types(
                            input_type,
                            input_sharding.as_ref(),
                            output_type,
                            output_sharding.as_ref(),
                        ),
                    };
                    (shares_buffers
                        && (has_single_device || !is_open)
                        && !aliased_outputs.contains(&physical_output_index))
                    .then_some(physical_output_index)
                });
            if let Some(physical_output_index) = physical_output_index {
                aliased_outputs.insert(physical_output_index);
                aliases.insert(physical_input_index, physical_output_index);
            }
        }
    }
    Ok(aliases)
}

/// Returns the types and the attributes of the arguments and of the results of the entry function of `boundary`, in
/// the order of the physical boundary. The attributes declare the sharding, the memory kind, the layout, and the
/// aliased result of each value, and `has_mesh` states whether the module declares a mesh.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedBoundaryMemory`] for a boundary memory that `target_platform` cannot use.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn entry_function_attributes<'c, 't>(
    boundary: &EntryBoundary,
    argument_shardings: &[Option<Sharding>],
    output_shardings: &[Option<Sharding>],
    open_outputs: &[bool],
    has_mesh: bool,
    supports_shardy: bool,
    target_platform: Option<&str>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(Vec<TypeAndAttributes<'c, 't, 'static>>, Vec<TypeAndAttributes<'c, 't, 'static>>), Error> {
    let EntryBoundary { signature, physical_argument_types, physical_output_types, aliases, .. } = boundary;
    let mut physical_argument_tensor_types = physical_argument_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location).map(|r#type| r#type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    if signature.has_ordered_io() {
        physical_argument_tensor_types.push(context.stable_hlo_token_type()?.as_ref());
    }
    let mut physical_output_tensor_types = physical_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location).map(|r#type| r#type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    if signature.has_ordered_io() {
        physical_output_tensor_types.push(context.stable_hlo_token_type()?.as_ref());
    }
    let mut argument_sharding_attributes = signature
        .physical_optional_input_shardings(argument_shardings)
        .iter()
        .map(|sharding| sharding.as_ref().map(|sharding| sharding.to_mlir(location)).transpose())
        .collect::<Result<Vec<_>, _>>()?;
    // Results whose placement over auto mesh axes the compiler chooses are open over those axes, like JAX lowers the
    // shardings of output types without `out_shardings` (refer to `ShardySharding`).
    let physical_open_outputs =
        if open_outputs.is_empty() { Vec::new() } else { signature.project_outputs(open_outputs) };
    let mut result_sharding_attributes = signature
        .physical_optional_output_shardings(output_shardings)
        .iter()
        .enumerate()
        .map(|(index, sharding)| {
            let open = physical_open_outputs.get(index).copied().unwrap_or(false);
            let open_dimensions = if open { OpenDimensions::All } else { OpenDimensions::None };
            sharding
                .as_ref()
                .map(|sharding| ShardySharding::new(sharding, open_dimensions).to_mlir(location))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    // The effect token of a signature side is annotated when any value of that side is, as each device threads its
    // own token.
    if signature.has_ordered_io() && has_mesh {
        let token_sharding = context.shardy_tensor_sharding(
            context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME),
            &[],
            &[],
            &[],
            shardy::ReductionOperation::Sum,
        )?;
        for shardings in [&mut argument_sharding_attributes, &mut result_sharding_attributes] {
            if shardings.iter().any(Option::is_some) {
                shardings.push(Some(token_sharding));
            }
        }
    }
    // Internal layout constraints do not constrain the executable ABI. Declare layouts on the physical boundary, whose
    // types already omit static zeros and include hidden extent scalars, while tokens have no array type. Placement
    // belongs to physical array slots, including captures, but never to the appended effect token. Default device
    // placement needs no attribute, but host identities still need attributes even without transfers.
    let function_arguments = physical_argument_tensor_types
        .iter()
        .enumerate()
        .map(|(index, tensor_type)| {
            let mut attributes = HashMap::new();
            if supports_shardy && let Some(sharding) = argument_sharding_attributes.get(index).and_then(Option::as_ref)
            {
                attributes.insert("sdy.sharding".into(), sharding.as_ref());
            }
            if let Some(r#type) = physical_argument_types.get(index).filter(|r#type| r#type.memory() != Memory::Device)
            {
                validate_boundary_memory(r#type.memory(), target_platform)?;
                attributes.insert(
                    "mhlo.memory_kind".into(),
                    context.string_attribute(r#type.memory().placement_kind()).as_ref(),
                );
            }
            if let Some(output_index) = aliases.get(&index) {
                attributes.insert(
                    "tf.aliasing_output".into(),
                    context.integer_attribute(context.signless_integer_type(64), *output_index as i64).as_ref(),
                );
            }
            if let Some(array_type) = physical_argument_types.get(index)
                && let Some(layout) = lower_boundary_layout(array_type, "input", context)?
            {
                attributes.insert("mhlo.layout_mode".into(), layout);
            }
            let attributes = (!attributes.is_empty()).then_some(attributes);
            Ok::<_, Error>(TypeAndAttributes { r#type: tensor_type.as_ref(), attributes })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let function_results = physical_output_tensor_types
        .iter()
        .enumerate()
        .map(|(index, tensor_type)| {
            let mut attributes = HashMap::new();
            if supports_shardy && let Some(sharding) = result_sharding_attributes.get(index).and_then(Option::as_ref) {
                attributes.insert("sdy.sharding".into(), sharding.as_ref());
            }
            if let Some(r#type) = physical_output_types.get(index).filter(|r#type| r#type.memory() != Memory::Device) {
                validate_boundary_memory(r#type.memory(), target_platform)?;
                attributes.insert(
                    "mhlo.memory_kind".into(),
                    context.string_attribute(r#type.memory().placement_kind()).as_ref(),
                );
            }
            if let Some(array_type) = physical_output_types.get(index)
                && let Some(layout) = lower_boundary_layout(array_type, "output", context)?
            {
                attributes.insert("mhlo.layout_mode".into(), layout);
            }
            let attributes = (!attributes.is_empty()).then_some(attributes);
            Ok::<_, Error>(TypeAndAttributes { r#type: tensor_type.as_ref(), attributes })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((function_arguments, function_results))
}

/// Lowers the body of the entry function of `boundary` into `function_block`, whose arguments are the arguments of
/// the executable, ending with the `func.return` of its results. It rebuilds the logical arguments from the physical
/// ones, completes the pending reductions that arguments arrive with, lowers `program`, and projects its outputs onto
/// the physical results. The first `capture_count` logical arguments are captures.
#[allow(clippy::too_many_arguments)]
fn lower_entry_function_body<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    boundary: &EntryBoundary,
    capture_count: usize,
    argument_shardings: &[Option<Sharding>],
    supports_shardy: bool,
    function_block: &'b DetachedBlock<'c, 't>,
    nested_functions: &Rc<CallFunctionMap<O>>,
    collective_state: &CollectiveLoweringState,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), Error>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    let EntryBoundary {
        signature,
        session_constants,
        passed_on_constants,
        logical_argument_types,
        physical_argument_types,
        ..
    } = boundary;
    let session_count = session_constants.len();
    let mut function_block_ref = function_block.as_ref();
    let mut logical_argument_values = Vec::with_capacity(logical_argument_types.len());
    for (logical_index, array_type) in logical_argument_types.iter().enumerate() {
        let value = match signature.input_mapping()[logical_index] {
            Some(physical_index) => {
                let mut value = function_block.argument(physical_index).unwrap().as_ref();
                let mut refined_type = physical_argument_types[physical_index].clone();
                for input_dimension in signature
                    .input_dimensions()
                    .iter()
                    .filter(|dimension| dimension.logical_input_index() == logical_index)
                {
                    let mut dimensions = refined_type.shape().dimensions().to_vec();
                    dimensions[input_dimension.axis()] =
                        array_type.shape().dimensions()[input_dimension.axis()].clone();
                    refined_type = refined_type.with_shape(Shape::new(dimensions));
                    let size = function_block.argument(input_dimension.physical_input_index()).unwrap().as_ref();
                    let operation = function_block_ref.append_operation(stable_hlo::set_dimension_size(
                        value,
                        size,
                        lower_tensor_type(&refined_type, context, location)?,
                        input_dimension.axis(),
                        location,
                    )?)?;
                    value = operation.result(0).unwrap().as_ref();
                }
                value
            }
            None => {
                let value = lower_unplaced_constant_output(
                    std::slice::from_ref(array_type),
                    0,
                    &mut function_block_ref,
                    context,
                    location.as_ref(),
                )?
                .into_iter()
                .next()
                .unwrap();
                // An unread argument (refer to `unread_arguments` above) is a placeholder that no emitted region
                // reads and that XLA removes, so its sharding is not applied either.
                if signature.is_unread_input(logical_index) {
                    logical_argument_values.push(value);
                    continue;
                }
                value
            }
        };
        debug_assert_eq!(value.r#type()?, lower_tensor_type(array_type, context, location)?.as_ref());
        // An argument that arrives with a pending reduction that its type does not record (i.e., over auto mesh
        // axes, which an input sharding override states and types never record, refer to
        // `receivable_argument_sharding` in `crate::compilation::domains`) is reduced where it enters, because
        // no value of the program can carry that reduction. The sharding constraint is the transition that
        // Shardy accepts, and it completes the reduction (e.g., in the reshard that `Array::to_placement`
        // compiles from such a source onto a sharding without it). An argument whose type records a pending
        // reduction and that arrives split along auto mesh axes, which its type cannot record, is constrained
        // to its sharding where it enters as well, so that the constraints that carry its pending reduction
        // through the program keep that split (refer to `constrained_dimensions`).
        let states_auto_split = |sharding: &Sharding| {
            !array_type.unreduced_axes().is_empty()
                && array_type.sharding().is_some_and(|stated| {
                    stated.dimensions_with_unconstrained_replicated()
                        != sharding.dimensions_with_unconstrained_replicated()
                })
        };
        let value = match &argument_shardings[logical_index] {
            Some(sharding)
                if supports_shardy
                    && (sharding.unreduced_axes().iter().any(|axis| !array_type.unreduced_axes().contains(axis))
                        || states_auto_split(sharding)) =>
            {
                // The constraint completes the pending reduction along the axes that the type does not record,
                // with a collective that must make that completion for the data type of the argument.
                // The whole transition is checked as well, because the axes that the type records stay pending.
                let unrecorded = |axis: &&String| !array_type.unreduced_axes().contains(*axis);
                let (completed, kept): (Vec<&String>, Vec<&String>) =
                    sharding.unreduced_axes().iter().partition(unrecorded);
                let kind = sharding.unreduced_kind();
                let change = ReductionChange::Complete(kind);
                let data_type = array_type.data_type();
                let axes = completed.iter().copied();
                validate_pending_reduction_change(data_type, change, sharding, axes, "input_shardings")?;
                validate_partial_completion(data_type, kind, sharding, completed, kept, "input_shardings")?;
                let reduced = sharding.clone().with_unreduced_axes(array_type.unreduced_axes().iter().cloned())?;
                let block = &mut function_block_ref;
                lower_pending_reduction_constraint(value, &reduced, &[], block, location.as_ref())?
            }
            _ => value,
        };
        logical_argument_values.push(value);
    }
    let (capture_values, public_input_values) = logical_argument_values.split_at(capture_count);
    // Ordinary module callers lower an unlifted program whose entry inputs are only the public arguments.
    // The compilation domain instead supplies the program with lifted captures, so that core reference
    // discharge and lowering observe the same arena. In that form the complete logical argument list is also
    // the entry input list, while the leading prefix still serves attached `CaptureReference` constants. No
    // third boundary form exists, so the entry input count must match one of those two shapes exactly.
    // Session constants are not program inputs, so the lifted form omits the trailing session arguments of the
    // capture prefix from the entry inputs, while `capture_values` keeps them as its suffix for
    // `CollectiveLoweringState::lower_session_constant`.
    debug_assert!(
        program.input_count() + session_count == logical_argument_values.len()
            || program.input_count() == public_input_values.len(),
    );
    let lifted_input_values: Vec<_>;
    let input_values = if program.input_count() + session_count == logical_argument_values.len() {
        lifted_input_values = capture_values[..capture_values.len() - session_count]
            .iter()
            .chain(public_input_values)
            .copied()
            .collect();
        lifted_input_values.as_slice()
    } else {
        public_input_values
    };
    // The arrays that the program only passes on to calls that never read them are bound like unread arguments,
    // after the session constants (refer to `collective_state` above).
    let mut captured_values = Cow::Borrowed(capture_values);
    for constant in passed_on_constants {
        let r#type = constant.r#type();
        let values = lower_unplaced_constant_output(
            std::slice::from_ref(r#type.as_ref()),
            0,
            &mut function_block_ref,
            context,
            location.as_ref(),
        )?;
        captured_values.to_mut().extend(values);
    }
    let mut effect_tokens = EffectTokens::default();
    if let Some(index) = signature.ordered_io_input_index() {
        effect_tokens.set(EffectClass::OrderedIo, function_block.argument(index).unwrap().as_ref());
    }
    let logical_outputs = lower_program_outputs_with_inputs(
        program,
        &captured_values,
        input_values,
        &mut function_block_ref,
        context,
        location.as_ref(),
        Some(nested_functions),
        collective_state,
        &mut effect_tokens,
    )?;
    let mut physical_outputs = signature.project_outputs(logical_outputs.as_slice());
    // XLA writes the runtime-size trailer of a bounded dynamic result only in the `SliceToDynamic` that
    // `DynamicPadder` inserts before each output leaf. When several such leaves carry values that XLA proves
    // equal (e.g., the same SSA value or equal computations merged by CSE), one `SliceToDynamic` survives and
    // copy insertion fills the other result with a plain copy of the bounded payload, which leaves its trailer
    // unset (i.e., the buffer reports zero-sized dynamic dimensions). Routing these results through one
    // optimization barrier makes every leaf a distinct barrier result. XLA expands barriers only after copy
    // insertion, so each leaf keeps its own `SliceToDynamic`. A single dynamic result cannot alias a static
    // one, so the barrier is only needed with two or more of them.
    let mut dynamic_physical_indices = signature
        .output_dimensions()
        .iter()
        .map(|output_dimension| signature.output_mapping()[output_dimension.logical_output_index()].unwrap())
        .collect::<Vec<_>>();
    dynamic_physical_indices.dedup();
    if dynamic_physical_indices.len() > 1 {
        let barrier_inputs = dynamic_physical_indices.iter().map(|index| physical_outputs[*index]).collect::<Vec<_>>();
        let barrier = function_block_ref
            .append_operation(stable_hlo::optimization_barrier(barrier_inputs.as_slice(), location)?)?;
        for (result_index, physical_index) in dynamic_physical_indices.into_iter().enumerate() {
            physical_outputs[physical_index] = barrier.result(result_index).unwrap().as_ref();
        }
    }
    for output_dimension in signature.output_dimensions() {
        physical_outputs.push(lower_runtime_dimension_size_i64(
            logical_outputs[output_dimension.logical_output_index()],
            output_dimension.axis(),
            &mut function_block_ref,
            context,
            location.as_ref(),
        )?);
    }
    if signature.has_ordered_io() {
        physical_outputs.push(effect_tokens.get(EffectClass::OrderedIo).unwrap());
    }
    function_block_ref.append_operation(func::r#return(physical_outputs.as_slice(), location)?)?;
    Ok(())
}

/// Lowers `program` to a [`LoweredModule`] for the lowering entry points above, validating its reference state
/// aliases and entry boundary. When `export_effects` is set and the program has ordered I/O, the entry function also
/// exports the runtime ordered I/O token chain as an extra argument and result.
#[allow(clippy::too_many_arguments)]
fn lower_mlir_module_for_program_with_effect_boundary<
    Input,
    Output,
    ProgramInput,
    ProgramOutput,
    S,
    O: LowerableOperationFamily,
>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    argument_shardings: &[Option<Sharding>],
    output_shardings: &[Option<Sharding>],
    target_platform: Option<&str>,
    supports_shardy: bool,
    reference_states: &[ExternalReferenceBinding],
    donated_inputs: &[bool],
    unread_inputs: &[bool],
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,
    open_outputs: &[bool],
    export_effects: bool,
) -> Result<LoweredModule, Error>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
    S: AsRef<str>,
{
    let boundary = plan_entry_boundary(
        program,
        capture_types,
        global_input_types,
        global_output_types,
        function_name.as_ref(),
        argument_shardings,
        output_shardings,
        target_platform,
        reference_states,
        donated_inputs,
        unread_inputs,
        open_outputs,
        export_effects,
    )?;

    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location)?;

    // Emit `sdy.mesh` declarations for any sharding referenced either by inner operations or by the optional signature
    // shardings, so the `func.func` attributes can refer to `@mesh`.
    let mut signature_mesh = None;
    for sharding in argument_shardings.iter().chain(output_shardings).flatten() {
        signature_mesh = Some(match signature_mesh.take() {
            Some(existing_mesh) => merge_logical_meshes(existing_mesh, sharding.mesh())?,
            None => sharding.mesh().clone(),
        });
    }
    let mesh = collect_nested_sharding_mesh(program, signature_mesh, target_platform)?;
    if let Some(mesh) = mesh.as_ref() {
        let mesh_operation = mesh.to_mlir(location)?;
        module.body()?.append_operation(mesh_operation)?;
    }

    // Trace each distinct named semantic decomposition once and retain it in the lowering state of the module, so that
    // the entry and nested function bodies resolve the same private symbol.
    let named_compositions = Rc::new(collect_named_composition_functions(program, target_platform)?);

    // Module-scoped lowering state, shared between the entry function body and private functions below so channel ids
    // stay unique and target, representation policy, named compositions, and provenance state reach nested bodies. The
    // arrays that the program only passes on to calls that never read them follow the session constants in the table,
    // and the entry function binds them to values that require no buffer (refer to `partition_session_constants`).
    let collective_state = CollectiveLoweringState::new()
        .with_target_platform(target_platform)
        .with_ragged_dot_lowering_strategy(ragged_dot_lowering_strategy)
        .with_named_compositions(named_compositions.clone())
        .with_session_constants(
            &[boundary.session_constants.as_slice(), boundary.passed_on_constants.as_slice()].concat(),
        );

    // Deduplicate `call` callees that occur more than once into shared private `func.func`s, so repeated nested
    // programs (e.g., identical transformer blocks or the per-block primal and pullback programs of `grad`) lower
    // to one function plus N `func.call`s instead of N inlined copies. The map is empty for modules without repeated
    // calls, in which case every `call` inlines exactly as before.
    let nested_functions = Rc::new(collect_call_functions(program, target_platform)?);
    {
        let mut module_block = module.body()?;
        for key in named_compositions.order() {
            let function = named_compositions.functions().get(key).unwrap();
            emit_named_composition_function(
                &mut module_block,
                function,
                &collective_state,
                &context,
                location.as_ref(),
            )?;
        }
        for key in nested_functions.order() {
            let function = nested_functions.functions().get(key).unwrap();
            emit_call_function(
                &mut module_block,
                function,
                &nested_functions,
                &collective_state,
                &context,
                location.as_ref(),
            )?;
        }
    }

    let (function_arguments, function_results) = entry_function_attributes(
        &boundary,
        argument_shardings,
        output_shardings,
        open_outputs,
        mesh.is_some(),
        supports_shardy,
        target_platform,
        &context,
        location.as_ref(),
    )?;
    module.body()?.append_operation({
        let argument_types = function_arguments.iter().map(|argument| (argument.r#type, location)).collect::<Vec<_>>();
        let function_block = context.block(argument_types.as_slice());
        lower_entry_function_body(
            program,
            &boundary,
            capture_types.len(),
            argument_shardings,
            supports_shardy,
            &function_block,
            &nested_functions,
            &collective_state,
            &context,
            location.as_ref(),
        )?;
        let mut function_region = context.region();
        function_region.append_block(function_block)?;
        func::func(
            boundary.function_name.as_str(),
            func::FuncAttributes { arguments: function_arguments, results: function_results, ..Default::default() },
            function_region,
            location,
        )?
    })?;
    let mut signature = boundary.signature;
    let mesh = declare_constrained_mesh(&module, program, mesh, target_platform, location.as_ref())?;
    if !supports_shardy {
        remove_shardy_operations(&module)?;
    }

    remove_unused_constants(&module)?;
    if !module.verify()? {
        return Err(Error::Lowering(LoweringError::VerificationFailure));
    }
    if collective_state.has_assertions() {
        signature.require_assertion_handler();
    }
    let requires_assertion_handler = signature.requires_assertion_handler();
    let mut runtime_requirements = collective_state.runtime_requirements();
    if requires_assertion_handler {
        runtime_requirements.push(RuntimeRequirement::builtin("ryft.assert"));
    }
    if signature.has_ordered_io() {
        runtime_requirements.push(RuntimeRequirement::builtin("ryft.ordered-io"));
    }
    let runtime_requirements = normalize_runtime_requirements(runtime_requirements)?.into();
    Ok(LoweredModule {
        stable_hlo: serialize_lowered_module(&module, &collective_state)?,
        signature,
        requires_assertion_handler,
        runtime_requirements,
        mesh,
    })
}

/// Returns the module mesh of `module`, which is the lowering of `program`, after declaring it when `mesh`, the mesh
/// that the signature and `collect_nested_sharding_mesh` found before the program was lowered, is [`None`] but the
/// lowered operations refer to a mesh anyway. This makes the declaration follow what lowering emitted rather than a
/// list of the operations that emit sharding constraints, which a new constraint (e.g., one that a lowering rule emits
/// for the sharding of a result type) would otherwise have to join. Every Shardy operation other than the declaration
/// itself refers to the module mesh, and the declared mesh is the one shared by the shardings of the array types of
/// the regions that lowering emits for `target_platform` (refer to [`emitted_regions`]), which every constraint over
/// the sharding of a type refers to. A constraint over a sharding that no type carries must be found by
/// `collect_nested_sharding_mesh` instead, and verification of the module reports it otherwise.
///
/// # Errors
///
/// Returns [`LoweringError::MismatchedMeshTopology`] when the shardings of those types disagree on the topology.
fn declare_constrained_mesh<'c, 't, ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    module: &Module<'c, 't>,
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    mesh: Option<LogicalMesh>,
    target_platform: Option<&str>,
    location: LocationRef<'c, 't>,
) -> Result<Option<LogicalMesh>, Error>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    if mesh.is_some() {
        return Ok(mesh);
    }
    let mut refers_to_mesh = false;
    module.as_operation()?.walk(WalkOrder::PreOrder, |operation| match operation.name().as_str() {
        Ok(name) if name.starts_with("sdy.") && name != "sdy.mesh" => {
            refers_to_mesh = true;
            WalkResult::Interrupt
        }
        _ => WalkResult::Advance,
    });
    if !refers_to_mesh {
        return Ok(None);
    }
    let mut mesh = None;
    for region in emitted_regions(program.entry_region_ref(), target_platform) {
        for atom in region.atoms() {
            if let Type::Array(array_type) = atom.r#type().as_ref()
                && let Some(sharding) = array_type.sharding()
            {
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => merge_logical_meshes(existing_mesh, sharding.mesh())?,
                    None => sharding.mesh().clone(),
                });
            }
        }
    }
    if let Some(mesh) = &mesh {
        module.body()?.insert_operation(mesh.to_mlir(location)?, 0)?;
    }
    Ok(mesh)
}

/// Validates that `incoming` has the topology of the module mesh `existing`, which it returns. The module mesh keeps
/// the axis types of the first mesh that declared it, because the emitted `sdy.mesh` records only axis names and sizes.
///
/// # Errors
///
/// Returns [`LoweringError::MismatchedMeshTopology`] when the axis names or sizes of the two meshes differ, including
/// in order.
pub(super) fn merge_logical_meshes(existing: LogicalMesh, incoming: &LogicalMesh) -> Result<LogicalMesh, Error> {
    if !existing.has_same_topology(incoming) {
        return Err(Error::Lowering(LoweringError::MismatchedMeshTopology {
            expected: existing,
            actual: incoming.clone(),
        }));
    }
    Ok(existing)
}

/// Removes every Shardy operation from `block` and from the regions nested in its operations, as
/// [`remove_shardy_operations`] describes.
fn remove_shardy_operations_in_block<'b, 'c: 'b, 't: 'c>(block: BlockRef<'b, 'c, 't>) -> Result<(), Error> {
    let mut shardy_operations = Vec::new();
    for operation in block.operations()? {
        let operation = operation?;
        // An operation is classified before its regions are visited, so that an unsupported operation is reported
        // rather than the Shardy terminator of its body.
        let name = operation.name();
        match name.as_str() {
            Ok("sdy.sharding_constraint") => {
                operation.result(0)?.replace_uses(operation.operand_value(0)?);
                shardy_operations.push(operation);
            }
            Ok("sdy.mesh") => shardy_operations.push(operation),
            Ok(name) if name.starts_with("sdy.") => {
                return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                    operation: name.to_string(),
                    reason: "the target plugin cannot compile with Shardy".to_string(),
                }));
            }
            _ => {}
        }
        for region in operation.regions() {
            for nested_block in region?.blocks()? {
                remove_shardy_operations_in_block(nested_block?)?;
            }
        }
    }
    for operation in shardy_operations {
        // SAFETY: the uses of a removed constraint were replaced by its operand and a mesh declaration has no results,
        // so no remaining operation references a result of the removed operation, and dropping the detached operation
        // that `remove_operation` returns destroys it.
        unsafe { block.remove_operation(operation)? };
    }
    Ok(())
}

/// Removes every unused `stablehlo.constant` operation from `block` and from the regions nested in its operations.
fn remove_unused_constants_in_block<'b, 'c: 'b, 't: 'c>(block: BlockRef<'b, 'c, 't>) -> Result<(), Error> {
    let mut unused_constants = Vec::new();
    for operation in block.operations()? {
        let operation = operation?;
        for region in operation.regions() {
            for nested_block in region?.blocks()? {
                remove_unused_constants_in_block(nested_block?)?;
            }
        }
        if operation.name().as_str() == Ok("stablehlo.constant") && operation.result(0)?.uses()?.next().is_none() {
            unused_constants.push(operation);
        }
    }
    for constant in unused_constants {
        // SAFETY: the removed constant has no uses, so no remaining operation references its result, and dropping the
        // detached operation that `remove_operation` returns destroys it.
        unsafe { block.remove_operation(constant)? };
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, Array, ArrayOperation, ArrayType, CallOperation, CaptureReference, CastOperation, CompositeKey,
        CompositeOperation, ConditionOperation, Context, Cos, CustomRuleDefinition, CustomRuleRegistration, DataType,
        Device, DeviceMesh, Differentiate, Dimension, DimensionBounds, DimensionToScalarOperation, DimensionType,
        DimensionValue, DimensionVariable, Dot, DotDimensionNumbers, EagerContext, EffectClass, EffectClasses,
        ExternalReferenceBinding, FlatProgram, Layout, LogicalMesh, MeshAxis, MeshAxisType, NamedComposite, OneLike,
        PlaceOperation, Placeholder, Placement, PlacementRequest, PrintOperation, Program,
        ProgramBuilder as ArrayIrProgramBuilder, Provenance, ProvenanceScope, ReferenceFreezeOperation,
        ReferenceNewOperation, ReferenceReadOperation, ReferenceSource, ReferenceType, ReferenceWriteOperation,
        ReshapeOperation, ReverseModeDifferentiate, Shape, ShardMapOptions, Sharding, ShardingDimension, Sin,
        TiledLayout, Transpose, TypedProgram, ZeroLike,
    };
    use ryft_kernels::{KernelBoundaryContract, KernelParameterAccess, validate_kernel_body};
    use ryft_mlir::dialects::{func, stable_hlo};
    use ryft_mlir::{
        Attribute, Block, Context as MlirContext, DialectHandle, Operation as MlirOperation, Region, Size as MlirSize,
        Type as MlirType, TypeAndAttributes,
    };

    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::{ArrayConstant, Constant, Error, ExecutionError, Operation, Session, Type};

    use crate::lowering::{LoweringError, RaggedDotLoweringStrategy, token_threaded_effects};

    use crate::lowering::tests::{
        ProgramBuilder, TEST_ARRAY_DOMAIN, add_reference_state, dynamic_dimension, linear_call_program,
        lower_replicated_shard_map_body, neg_branch, scalar_bilinear_sin, scalar_linear_region,
        scalar_quartic_plus_sin, shard_map_module, stateful_scan_program, test_manual_mesh, test_matrix_type,
        test_vector_type, to_mlir_module_for_plain_program, unproject_plain_program,
    };

    use crate::tests::{cpu_client, test_compilation_options, values_from_bytes, values_to_bytes};

    use super::*;

    #[test]
    fn test_contains_unresolved_state_ignores_dormant_regions() {
        let stateful =
            |extended| scalar_linear_region(&ArrayType::scalar(DataType::F32), extended, add_reference_state);

        // State in the dormant transpose region of a `linear_call` is never lowered, unlike state in its forward region,
        // and neither is state in the body of a scan that admits no iteration, unlike state in one that iterates.
        assert!(!contains_unresolved_state(&linear_call_program(&stateful(false), &stateful(true))));
        assert!(contains_unresolved_state(&linear_call_program(&stateful(true), &stateful(false))));
        assert!(!contains_unresolved_state(&stateful_scan_program(0)));
        assert!(contains_unresolved_state(&stateful_scan_program(1)));
    }

    #[test]
    fn test_contains_unresolved_references_ignores_dormant_regions() {
        let stateful =
            |extended| scalar_linear_region(&ArrayType::scalar(DataType::F32), extended, add_reference_state);

        // References in the dormant transpose region of a `linear_call` are never lowered, unlike references in its
        // forward region, and neither are references in the body of a scan that admits no iteration.
        assert!(!contains_unresolved_references(&linear_call_program(&stateful(false), &stateful(true))));
        assert!(contains_unresolved_references(&linear_call_program(&stateful(true), &stateful(false))));
        assert!(!contains_unresolved_references(&stateful_scan_program(0)));
        assert!(contains_unresolved_references(&stateful_scan_program(1)));
    }

    #[test]
    fn test_entry_aliases() {
        // The entry function takes one capture and two public inputs of the same type, and returns one result of
        // another type followed by two results of that type.
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let other_type = ArrayType::new_static(DataType::F32, [2]);
        let argument_types = vec![r#type.clone(); 3];
        let output_types = [other_type, r#type.clone(), r#type];
        let signature = ExecutableSignature::new(&argument_types, &output_types);
        let shardings = [None, None, None];
        let aliases = |donated_inputs: &[bool]| {
            entry_aliases(
                &[],
                donated_inputs,
                &[],
                1,
                &argument_types,
                &output_types,
                &signature,
                &shardings,
                &shardings,
            )
        };

        // Each donated input is paired with the first result of its type that no earlier input was paired with, and
        // the capture is never paired.
        assert_eq!(aliases(&[]), Ok(HashMap::new()));
        assert_eq!(aliases(&[true, false]), Ok(HashMap::from([(1, 1)])));
        assert_eq!(aliases(&[false, true]), Ok(HashMap::from([(2, 1)])));
        assert_eq!(aliases(&[true, true]), Ok(HashMap::from([(1, 1), (2, 2)])));

        // The donation flags must cover the public inputs exactly.
        assert_eq!(
            aliases(&[true]),
            Err(Error::Execution(ExecutionError::DonationFlagCountMismatch { expected_count: 2, actual_count: 1 })),
        );
    }

    #[test]
    fn test_to_mlir_module_for_program() {
        let global_input_type = test_vector_type(8);
        let mesh = test_manual_mesh("x", 4);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| Ok(x.clone() + x),
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_renders_provenance_locations() {
        let global_input_type = test_vector_type(8);
        let mesh = test_manual_mesh("x", 4);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| {
                let context = x.value().context().clone();
                // One instruction staged under nested scopes, one under a fused origin, and one with unknown
                // provenance, covering every provenance shape in one lowered module.
                let scoped = context.invoke_with_provenance_scope(ProvenanceScope::new("outer"), || {
                    context.invoke_with_provenance_scope(ProvenanceScope::new("inner"), || x.clone() + x.clone())
                });
                let fused = Provenance::fused([
                    Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown()),
                    Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown()),
                ]);
                let product = context.invoke_with_provenance_origin(fused, || scoped.clone() * scoped);
                Ok(product + x)
            },
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        // The scoped addition carries the nested chain of name locations, the fused multiplication carries a fused location
        // without metadata, the addition with unknown provenance keeps the unknown base location, and the module and
        // function scaffolding keeps the base location. Locations render because modules that carry provenance serialize
        // with debug information enabled, in the parsable form that is not pretty printed.
        assert_eq!(
            module,
            indoc! {r#"
                #loc = loc(unknown)
                module {
                  sdy.mesh @mesh = <["x"=4]> loc(#loc)
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>} loc(unknown)) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32> loc(unknown)) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32> loc(#loc4)
                      %2 = stablehlo.multiply %1, %1 : tensor<2xf32> loc(#loc5)
                      %3 = stablehlo.add %2, %arg1 : tensor<2xf32> loc(#loc)
                      sdy.return %3 : tensor<2xf32> loc(#loc)
                    } : (tensor<8xf32>) -> tensor<8xf32> loc(#loc)
                    return %0 : tensor<8xf32> loc(#loc)
                  } loc(#loc)
                } loc(#loc)
                #loc1 = loc("inner")
                #loc2 = loc("a")
                #loc3 = loc("b")
                #loc4 = loc("outer"(#loc1))
                #loc5 = loc(fused[#loc2, #loc3])
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_renders_constants_and_supported_ops() {
        let global_input_type = test_matrix_type(4, 4);
        let mesh = test_manual_mesh("x", 2);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| {
                let product = x.transpose(vec![1, 0])?.dot(&x, &DotDimensionNumbers::matmul())?;
                let waveform = (-product).cos()?.sin()?;
                Ok((waveform.clone() * waveform.one_like()?) + waveform.zero_like()?)
            },
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x4xf32>) {
                      %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
                      %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
                      %3 = stablehlo.negate %2 : tensor<4x4xf32>
                      %4 = stablehlo.cosine %3 : tensor<4x4xf32>
                      %5 = stablehlo.sine %4 : tensor<4x4xf32>
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %7 = sdy.sharding_constraint %6 <@mesh, [{}, {}]> : tensor<4x4xf32>
                      %8 = stablehlo.multiply %5, %7 : tensor<4x4xf32>
                      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %10 = sdy.sharding_constraint %9 <@mesh, [{}, {}]> : tensor<4x4xf32>
                      %11 = stablehlo.add %8, %10 : tensor<4x4xf32>
                      sdy.return %11 : tensor<4x4xf32>
                    } : (tensor<4x4xf32>) -> tensor<8x4xf32>
                    return %0 : tensor<8x4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_preserves_boundary_layouts() {
        let array_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let zero_type = ArrayType::new_static(DataType::Zero, [2, 3]);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(array_type.clone().into());
        let zero = builder.add_input(zero_type.clone().into());
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![zero, input], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![array_type.clone(), zero_type.clone()],
                &vec![zero_type, array_type],
                "main",
                None,
                None,
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\"}) \
                      -> (tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\"}) {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<2x3xi1>
                    return %arg0 : tensor<2x3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_captures_as_hidden_arguments() {
        let array_type = test_vector_type(4);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(array_type.clone().into());
        let capture = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let capture_types = vec![array_type.clone()];
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        let stablehlo = to_mlir_module_for_program(
            &program,
            capture_types.as_slice(),
            &input_types,
            &output_types,
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg1, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_immediate_dimension_constants_in_place() {
        // An immediate dimension constant carries its own extent from the host, so unlike a capture reference, it
        // materializes as a scalar `i64` `stablehlo.constant` in the function body instead of consuming a hidden capture
        // argument. That is exactly what makes first-class extents usable where no capture table is reachable, such as
        // inside a `shard_map` manual region.
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let extent = builder.add_constant(Constant::Dimension(DimensionValue::new(extent_type, 4).unwrap()));
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![extent], None).unwrap()[0];
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
        let input_types: [ArrayType; 0] = [];
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &input_types,
            &vec![ArrayType::scalar(DataType::I64)],
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main() -> tensor<i64> {
                    %c = stablehlo.constant dense<4> : tensor<i64>
                    return %c : tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_capture_output_as_a_hidden_argument() {
        let array_type = test_vector_type(4);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let output = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
        let input_types: [ArrayType; 0] = [];
        let output_types = vec![array_type.clone()];
        let stablehlo = to_mlir_module_for_program(
            &program,
            std::slice::from_ref(&array_type),
            &input_types,
            &output_types,
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    return %arg0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_forwards_capture_into_nested_regions() {
        let array_type = test_vector_type(4);
        let branch = || {
            let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
            let output = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let true_region = builder.import_program(branch());
        let false_region = builder.import_program(branch());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let output = builder
            .add_instruction(
                Operation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = vec![array_type.clone()];
        let stablehlo = to_mlir_module_for_program(
            &program,
            std::slice::from_ref(&array_type),
            &[ArrayType::scalar(DataType::Boolean)],
            &output_types,
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i1>) -> tensor<4xf32> {
                    %0 = "stablehlo.if"(%arg1) ({
                      stablehlo.return %arg0 : tensor<4xf32>
                    }, {
                      stablehlo.return %arg0 : tensor<4xf32>
                    }) : (tensor<i1>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_erases_static_zero_space_boundary() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(zero_type.clone().into());
        let program: FlatProgram<Session> = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main() {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_preserves_effects_that_produce_an_erased_output() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(zero_type.clone().into());
        let output = builder
            .add_instruction(PrintOperation::<ArrayType>::new("zero"), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main() {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.print(%0, %1) {api_version = 4 : i32, \
                        backend_config = {label = \"zero\", logical_type = \"zero\"}, has_side_effect = true, \
                        operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], \
                        result_layouts = [dense<> : tensor<0xindex>]} : (tensor<3xi1>, !stablehlo.token) \
                        -> !stablehlo.token
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_preserves_dynamic_zero_space_shape_carrier() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("zero", Some(3))]));
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(zero_type.clone().into());
        let program: FlatProgram<Session> = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xi1>, %arg1: tensor<i32>) -> (tensor<?xi1, #stablehlo.bounds<2>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<2xi1>, tensor<i32>) -> tensor<?xi1, #stablehlo.bounds<2>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xi1, #stablehlo.bounds<2>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    return %0, %2 : tensor<?xi1, #stablehlo.bounds<2>>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_isolates_bounded_dynamic_results_behind_one_barrier() {
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("count", Some(4))]));
        let static_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let dynamic = builder.add_input(dynamic_type.clone().into());
        let r#static = builder.add_input(static_type.clone().into());
        let program: FlatProgram<Session> =
            builder.build(vec![dynamic, r#static, dynamic], vec![Placeholder; 2], vec![Placeholder; 3]).unwrap();

        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_type.clone(), static_type.clone()],
            &vec![dynamic_type.clone(), static_type, dynamic_type],
            "main",
            None,
            None,
        )
        .unwrap();

        // Both dynamic results pass through the barrier, while the static result and the hidden extents bypass it.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<3>>, tensor<2xf32>, tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<3xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<3>>
                    %1:2 = stablehlo.optimization_barrier %0, %0 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<?xf32, #stablehlo.bounds<3>>
                    %2 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                    %3 = stablehlo.convert %2 : (tensor<i32>) -> tensor<i64>
                    %4 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                    return %1#0, %arg1, %1#1, %3, %5 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<2xf32>, tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_declares_the_mesh_of_completed_device_reductions() {
        // A device target that completes a pending reduction over a mesh axis of size one constrains its result over
        // the mesh of its type, which neither the signature nor a payload names, so the module declares the mesh
        // because the lowered constraint refers to it.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let input_type = test_vector_type(4).with_sharding(unreduced).unwrap();
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(PlaceOperation::new(Device::new(0, 0), None), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = program
            .output_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            to_mlir_module_for_program(&program, &[], &vec![input_type], &output_types, "main", None, None).unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_omits_shardy_for_targets_without_the_dialect() {
        // A device target that completes a pending reduction over a mesh axis of size one constrains its result, and
        // the signature shardings annotate the entry function. A target whose plugin lacks the Shardy dialect gets
        // neither, nor the mesh that both refer to.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let input_type = test_vector_type(4).with_sharding(unreduced.clone()).unwrap();
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(PlaceOperation::new(Device::new(0, 0), None), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = program
            .output_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned().unwrap())
            .collect::<Vec<_>>();
        let lower = |supports_shardy| {
            lower_mlir_module_for_program_with_reference_state(
                &program,
                &[],
                &vec![input_type.clone()],
                &output_types,
                "main",
                Some(std::slice::from_ref(&unreduced)),
                Some(std::slice::from_ref(&replicated)),
                Some("cpu"),
                supports_shardy,
                &[],
                &[],
                &[],
                RaggedDotLoweringStrategy::default(),
                &[],
            )
            .unwrap()
        };
        let lowered = lower(true);
        assert_eq!(lowered.mesh(), Some(&mesh));
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={"x"}>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
        let lowered = lower(false);
        assert_eq!(lowered.mesh(), Some(&mesh));
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    return %arg0 : tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_remove_shardy_operations() {
        // Constraints are replaced by their operands in nested regions as well, and the mesh declaration is erased.
        let context = MlirContext::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        context.load_dialect(DialectHandle::stable_hlo().unwrap()).unwrap();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let module = context
            .parse_module(indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i32>) -> tensor<4xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    %1 = "stablehlo.case"(%arg1) ({
                      %2 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
                      %3 = stablehlo.negate %2 : tensor<4xf32>
                      stablehlo.return %3 : tensor<4xf32>
                    }, {
                      stablehlo.return %0 : tensor<4xf32>
                    }) : (tensor<i32>) -> tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#})
            .unwrap();
        remove_shardy_operations(&module).unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i32>) -> tensor<4xf32> {
                    %0 = "stablehlo.case"(%arg1) ({
                      %1 = stablehlo.negate %arg0 : tensor<4xf32>
                      stablehlo.return %1 : tensor<4xf32>
                    }, {
                      stablehlo.return %arg0 : tensor<4xf32>
                    }) : (tensor<i32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        // A manual computation has no equivalent without the dialect.
        let body = unproject_plain_program(neg_branch(test_vector_type(4)));
        let module = context.parse_module(lower_replicated_shard_map_body(body, 1).unwrap()).unwrap();
        assert!(matches!(
            remove_shardy_operations(&module),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == "sdy.manual_computation"
                    && reason == "the target plugin cannot compile with Shardy",
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_rejects_unresolved_reference_state() {
        assert_eq!(token_threaded_effects(EffectClasses::single(EffectClass::OrderedState)).next(), None);

        // The reference scan over the whole artifact catches the reference atoms and intrinsic semantics before token
        // threading, so two representative reference operations are enough to pin the diagnostic.
        let array_type = ArrayType::scalar(DataType::F32);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(Type::Array(array_type.clone()));
        let reference = builder
            .add_instruction(Operation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        builder
            .add_instruction(
                Operation::ReferenceWrite(ReferenceWriteOperation::new()),
                Vec::new(),
                vec![reference, input],
                None,
            )
            .unwrap();
        let output = builder
            .add_instruction(
                Operation::ReferenceFreeze(ReferenceFreezeOperation::new()),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The reference scan runs before the generic state scan at the module entry, so a program whose state comes
        // from reference operations receives the dedicated reference diagnostic. The state error remains behind it for
        // future state that does not come from references, and `token_threaded_effects` above pins that `OrderedState`
        // has no token slot either way.
        assert!(matches!(
            lower_mlir_module_for_program(
                &program,
                &[],
                &vec![array_type.clone()],
                &vec![array_type],
                "main",
                None,
                None,
                None,
            ),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "program",
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_rejects_preserved_reference_entries() {
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference = builder.add_input(Type::Reference(ReferenceType::new(scalar_type.clone())));
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly)]);
        validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
        assert!(matches!(
            lower_mlir_module_for_program(
                &program,
                &[],
                &vec![scalar_type.clone()],
                &vec![scalar_type],
                "main",
                None,
                None,
                None,
            ),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "program",
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_with_reference_state() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let state_type = ArrayType::new_static(DataType::F32, [4]);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let _zero = builder.add_input(Type::Array(zero_type.clone()));
        let mutated = builder.add_input(Type::Array(state_type.clone()));
        let _read_only = builder.add_input(Type::Array(state_type.clone()));
        let program: FlatProgram<Session> =
            builder.build(vec![mutated], vec![Placeholder; 3], vec![Placeholder]).unwrap();
        let input_types = vec![zero_type.clone(), state_type.clone(), state_type.clone()];
        let output_types = vec![state_type.clone()];
        let argument_shardings = vec![
            Sharding::replicated(mesh.clone(), 1),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        ];
        let output_shardings = vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()];
        let states = [
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, Some(0)),
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 2 }, None),
        ];

        let lowered = lower_mlir_module_for_program_with_reference_state(
            &program,
            &[],
            &input_types,
            &output_types,
            "main",
            Some(argument_shardings.as_slice()),
            Some(output_shardings.as_slice()),
            None,
            true,
            &states,
            &[],
            &[],
            RaggedDotLoweringStrategy::default(),
            &[],
        )
        .unwrap();
        assert_eq!(lowered.signature.input_mapping(), &[None, Some(0), Some(1)]);
        assert_eq!(lowered.signature.output_mapping(), &[Some(0)]);
        let expected_signature = concat!(
            "  func.func @main(",
            "%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>, ",
            "tf.aliasing_output = 0 : i64}, ",
            "%arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> ",
            "(tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {",
        );
        assert_eq!(
            lowered.stable_hlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                @SIGNATURE@
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    return %arg0 : tensor<4xf32>
                  }
                }
            "#}
            .replace("@SIGNATURE@", expected_signature),
        );

        let invalid_state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None);
        assert!(matches!(
            lower_mlir_module_for_program_with_reference_state(
                &program,
                &[],
                &input_types,
                &output_types,
                "main",
                Some(argument_shardings.as_slice()),
                Some(output_shardings.as_slice()),
                None,
                true,
                std::slice::from_ref(&invalid_state),
                &[],
                &[],
                RaggedDotLoweringStrategy::default(),
                &[],
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == "logical state input 0 is erased from the executable boundary",
        ));

        let asymmetric_sharding_message =
            "reference-state aliases require both argument and result sharding metadata or neither";
        for (argument_shardings, output_shardings) in
            [(Some(argument_shardings.as_slice()), None), (None, Some(output_shardings.as_slice()))]
        {
            assert!(matches!(
                lower_mlir_module_for_program_with_reference_state(
                    &program,
                    &[],
                    &input_types,
                    &output_types,
                    "main",
                    argument_shardings,
                    output_shardings,
                    None,
                    true,
                    &states,
                    &[],
                    &[],
                    RaggedDotLoweringStrategy::default(),
                    &[],
                ),
                Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                    if message == asymmetric_sharding_message,
            ));
        }

        let mismatched_output_shardings = vec![Sharding::replicated(mesh, 1)];
        assert!(matches!(
            lower_mlir_module_for_program_with_reference_state(
                &program,
                &[],
                &input_types,
                &output_types,
                "main",
                Some(argument_shardings.as_slice()),
                Some(mismatched_output_shardings.as_slice()),
                None,
                true,
                &states,
                &[],
                &[],
                RaggedDotLoweringStrategy::default(),
                &[],
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == "state input 1 and output 0 must use the same sharding",
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_with_reference_state_rejects_bounded_dynamic_aliases() {
        let extent = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let state_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let state = builder.add_input(Type::Array(state_type.clone()));
        let program: FlatProgram<Session> = builder.build(vec![state], vec![Placeholder], vec![Placeholder]).unwrap();
        let reference_state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(0));

        assert!(matches!(
            lower_mlir_module_for_program_with_reference_state(
                &program,
                &[],
                &vec![state_type.clone()],
                &vec![state_type],
                "main",
                Some(std::slice::from_ref(&sharding)),
                Some(std::slice::from_ref(&sharding)),
                None,
                true,
                std::slice::from_ref(&reference_state),
                &[],
                &[],
                RaggedDotLoweringStrategy::default(),
                &[],
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == "state input 0 and output 0 must be static because bounded-dynamic mutation alias \
                               compatibility is unsupported",
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_with_reference_state_aliases_donated_inputs() {
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let pair_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let first = builder.add_input(Type::Array(vector_type.clone()));
        let second = builder.add_input(Type::Array(vector_type.clone()));
        let third = builder.add_input(Type::Array(pair_type.clone()));
        let program: FlatProgram<Session> =
            builder.build(vec![second, first, third], vec![Placeholder; 3], vec![Placeholder; 3]).unwrap();
        let input_types = vec![vector_type.clone(), vector_type.clone(), pair_type.clone()];
        let output_types = vec![vector_type.clone(), vector_type.clone(), pair_type.clone()];
        let lower_with = |argument_shardings: Option<&[Sharding]>,
                          output_shardings: Option<&[Sharding]>,
                          donated_inputs: &[bool],
                          open_outputs: &[bool]| {
            lower_mlir_module_for_program_with_reference_state(
                &program,
                &[],
                &input_types,
                &output_types,
                "main",
                argument_shardings,
                output_shardings,
                None,
                true,
                &[],
                donated_inputs,
                &[],
                RaggedDotLoweringStrategy::default(),
                open_outputs,
            )
        };
        let lower = |shardings: Option<&[Sharding]>, donated_inputs: &[bool], open_outputs: &[bool]| {
            lower_with(shardings, shardings, donated_inputs, open_outputs)
        };
        let signature = |lowered: LoweredModule| {
            let line = lowered.stable_hlo.lines().find(|line| line.contains("func.func @main")).unwrap();
            line.trim().to_string()
        };
        // Pairs of the index of each aliased argument and the index of its output.
        let aliases = |lowered: LoweredModule| {
            signature(lowered)
                .split("%arg")
                .skip(1)
                .enumerate()
                .filter_map(|(index, argument)| {
                    let (_, alias) = argument.split_once("tf.aliasing_output = ")?;
                    Some((index, alias.split(' ').next().unwrap().parse::<usize>().unwrap()))
                })
                .collect::<Vec<_>>()
        };

        // Each donated input is paired with the first unpaired output of its type, and other inputs with none.
        assert_eq!(
            signature(lower(None, &[true, true, false], &[]).unwrap()),
            concat!(
                "func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i64}, ",
                "%arg1: tensor<4xf32> {tf.aliasing_output = 1 : i64}, %arg2: tensor<2xf32>) -> ",
                "(tensor<4xf32>, tensor<4xf32>, tensor<2xf32>) {",
            ),
        );
        assert_eq!(
            signature(lower(None, &[false, true, true], &[]).unwrap()),
            concat!(
                "func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32> {tf.aliasing_output = 0 : i64}, ",
                "%arg2: tensor<2xf32> {tf.aliasing_output = 2 : i64}) -> ",
                "(tensor<4xf32>, tensor<4xf32>, tensor<2xf32>) {",
            ),
        );
        assert!(!signature(lower(None, &[], &[]).unwrap()).contains("tf.aliasing_output"));

        // An open output is only paired on a single device, where the compiler cannot place it differently.
        let shardings = |device_count: usize| {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", device_count, MeshAxisType::Auto).unwrap()]).unwrap();
            vec![Sharding::replicated(mesh, 1); 3]
        };
        let open_outputs = [true, false, false];
        let single_device = signature(lower(Some(&shardings(1)), &[true, false, false], &open_outputs).unwrap());
        assert!(single_device.contains("tf.aliasing_output = 0 : i64"));
        let two_devices = signature(lower(Some(&shardings(2)), &[true, false, false], &open_outputs).unwrap());
        assert!(two_devices.contains("tf.aliasing_output = 1 : i64"));

        // An input that finds no output of its own sharding is paired with an output under which every device holds
        // shards of the same type, such as one that completes its pending sum or splits it along another axis of the
        // same size, and only after every donated input was paired with an output of its own sharding where one is
        // free. Before, only an output of the identical sharding was paired, so the first two cases had no alias.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let split = |axes: &[&str]| {
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(axes.iter().copied())]).unwrap()
        };
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let pending = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let cases = [
            ("completed sum", [pending, replicated.clone()], [replicated.clone(), replicated.clone()], [true, false]),
            ("other axis", [split(&["x"]), replicated.clone()], [replicated.clone(), split(&["y"])], [true, false]),
            (
                "other extents",
                [split(&["x"]), replicated.clone()],
                [replicated.clone(), split(&["x", "y"])],
                [true, false],
            ),
            ("own sharding first", [split(&["x"]), split(&["x"])], [split(&["y"]), split(&["x"])], [true, true]),
        ];
        let outcomes = cases
            .into_iter()
            .map(|(name, [first_argument, second_argument], [first_result, second_result], [first, second])| {
                let argument_shardings = [first_argument, second_argument, replicated.clone()];
                let output_shardings = [first_result, second_result, replicated.clone()];
                let donated_inputs = [first, second, false];
                let lowered = lower_with(Some(&argument_shardings), Some(&output_shardings), &donated_inputs, &[]);
                (name, aliases(lowered.unwrap()))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            outcomes,
            [
                ("completed sum", vec![(0, 0)]),
                ("other axis", vec![(0, 1)]),
                ("other extents", vec![]),
                ("own sharding first", vec![(0, 1), (1, 0)]),
            ],
        );

        assert!(matches!(
            lower(None, &[true], &[]),
            Err(Error::Execution(ExecutionError::DonationFlagCountMismatch { expected_count: 3, actual_count: 1 })),
        ));
    }

    #[test]
    fn test_entry_alias_attribute_executes_in_place_on_cpu() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[MlirSize::Static(4)], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let function_block = context.block(&[(tensor_type, location)]);
                {
                    let input = function_block.argument(0).unwrap();
                    let mut function_block_ref = function_block.as_ref();
                    let doubled =
                        function_block_ref.append_operation(stable_hlo::add(input, input, location).unwrap()).unwrap();
                    function_block_ref
                        .append_operation(func::r#return(&[doubled.result(0).unwrap()], location).unwrap())
                        .unwrap();
                }
                let mut function_region = context.region();
                function_region.append_block(function_block).unwrap();
                func::func(
                    "main",
                    func::FuncAttributes {
                        arguments: vec![TypeAndAttributes {
                            r#type: tensor_type.as_ref(),
                            attributes: Some(HashMap::from([
                                ("mhlo.sharding".into(), context.string_attribute("{replicated}").as_ref()),
                                (
                                    "tf.aliasing_output".into(),
                                    context.integer_attribute(context.signless_integer_type(64), 0).as_ref(),
                                ),
                            ])),
                        }],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    function_region,
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        let module = module.to_string();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32> {mhlo.sharding = "{replicated}", tf.aliasing_output = 0 : i64}) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        let client = cpu_client(1);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, None))
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        let input = client
            .buffer(values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(), BufferType::F32, [4], None, device, None)
            .unwrap();
        input.ready().unwrap().r#await().unwrap();
        // The input is ready and remains alive in `inputs` until execution completes, so comparing its opaque address
        // with the synchronized output address does not dereference either device pointer.
        let input_pointer = unsafe { input.unsafe_pointer().unwrap() };
        let inputs = [ExecutionInput { buffer: Arc::new(input), donatable: true }];
        let execution = executable
            .execute(
                vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
                &[],
            )
            .unwrap();
        let mut device_outputs = execution.block_until_ready().unwrap().remove(0);
        let output = device_outputs.outputs.remove(0);
        let output_pointer = unsafe { output.unsafe_pointer().unwrap() };
        // `tf.aliasing_output` is a hint rather than a guarantee, because the copy insertion of XLA may materialize a fresh
        // output. This pointer equality is therefore a deliberate canary pinned to the CPU plugin version that the
        // repository pins. If a plugin upgrade breaks it while the numeric assertion below still passes, re-evaluate
        // whether the alias hint is still honored rather than assuming a correctness bug.
        assert_eq!(output_pointer, input_pointer);
        let output = output.copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(output.as_slice()), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_to_mlir_module_for_plain_program() {
        let (_, compiled): (Array, Program<Array, ArrayOperation<Array>, (Array, Array), Array>) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |inputs| Ok(scalar_bilinear_sin(inputs)),
                (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
            )
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
                    %1 = stablehlo.sine %arg0 : tensor<f64>
                    %2 = stablehlo.add %0, %1 : tensor<f64>
                    return %2 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_gradients() {
        let (_, compiled): (Array, Program<Array, ArrayOperation<Array>, Array, Array>) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |x| {
                    let context = x.context().clone();
                    Ok(context.differentiate_at(x).gradient(scalar_quartic_plus_sin).unwrap())
                },
                Array::scalar(2.0).unwrap(),
            )
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();

        // The gradient computes one cosine for the derivative of `sin(x)`, expands the `x^4` chain rule into
        // multiplications, accumulates the cotangent contributions with additions, and contains no sine.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.cosine %arg0 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %1 = stablehlo.multiply %0, %cst : tensor<f64>
                    %2 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %3 = stablehlo.multiply %2, %arg0 : tensor<f64>
                    %4 = stablehlo.multiply %3, %cst : tensor<f64>
                    %5 = stablehlo.add %1, %4 : tensor<f64>
                    %6 = stablehlo.multiply %arg0, %cst : tensor<f64>
                    %7 = stablehlo.multiply %2, %6 : tensor<f64>
                    %8 = stablehlo.add %5, %7 : tensor<f64>
                    %9 = stablehlo.multiply %arg0, %6 : tensor<f64>
                    %10 = stablehlo.multiply %arg0, %9 : tensor<f64>
                    %11 = stablehlo.add %8, %10 : tensor<f64>
                    %12 = stablehlo.multiply %arg0, %9 : tensor<f64>
                    %13 = stablehlo.add %11, %12 : tensor<f64>
                    return %13 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_standalone_pullbacks() {
        // Standalone pullback over the primal operation family, produced by the reverse path that is aware of partitions.
        // The reverse program of `f(x, y) = x * y + sin(x)` consumes `[output_cotangent ++ residuals]`, where the residuals
        // are the factors `cos(x)`, `x`, and `y` at the linearization point. It lowers the backward pass, weighted by those
        // residuals, as `stablehlo.multiply`s of the cotangent against the residual inputs rather than baking the primal
        // point in as constants. This is the analogue of the standalone `vjp_fn` of JAX, with the residuals threaded as
        // explicit arguments.
        let (_, pullback): (Array, _) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |inputs, ()| Ok(scalar_bilinear_sin(inputs)),
                (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
                (),
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();

        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        // The pullback takes the output cotangent plus three residual inputs and returns one cotangent output for each of
        // `x` and `y`. It multiplies the cotangent by the residual inputs rather than baking the primal point in as
        // constants, so the lowering consists of multiplications and additions with no `stablehlo.constant`.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
                    %0 = stablehlo.multiply %arg3, %arg0 : tensor<f64>
                    %1 = stablehlo.multiply %arg2, %arg0 : tensor<f64>
                    %2 = stablehlo.multiply %arg1, %arg0 : tensor<f64>
                    %3 = stablehlo.add %0, %2 : tensor<f64>
                    return %3, %1 : tensor<f64>, tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_staged_gradients() {
        // The gradient of `f` staged for JIT compilation stays symbolic, like `jit(grad(f))` in JAX. It uses the traced
        // gradient path, which traces through the VJP and its pullback.
        let (_, compiled): ((Array, Array), Program<Array, ArrayOperation<Array>, (Array, Array), (Array, Array)>) =
            TEST_ARRAY_DOMAIN
                .interpret_and_trace(
                    |inputs| {
                        let context = inputs.0.context().clone();
                        Ok(context.differentiate_at(inputs).gradient(scalar_bilinear_sin).unwrap())
                    },
                    (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
                )
                .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();
        // `cos(x)` is computed symbolically from `%arg0` rather than baked in as a constant, both inputs are referenced, and
        // the gradient contains no sine, because the derivative of sine is cosine.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
                    %0 = stablehlo.cosine %arg0 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %1 = stablehlo.multiply %0, %cst : tensor<f64>
                    %2 = stablehlo.multiply %arg1, %cst : tensor<f64>
                    %3 = stablehlo.add %1, %2 : tensor<f64>
                    %4 = stablehlo.multiply %arg0, %cst : tensor<f64>
                    return %3, %4 : tensor<f64>, tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_collect_nested_sharding_mesh() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 2);
        let input_type = test_vector_type(4);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        builder
            .add_instruction(
                ReshapeOperation::new(vec![4], vec![2, 2]).with_output_sharding(sharding),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
            .unwrap()
            .into_unprojected::<Constant, Operation>()
            .unwrap();
        assert_eq!(collect_nested_sharding_mesh(&program, None, None).unwrap(), Some(mesh.clone()));
        // The mesh exists only on an internal value. Lowering must still declare the symbol referenced by its
        // reshape constraint, even though neither the signature nor an explicit sharding operation supplies it.
        to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type.clone()],
            &Vec::<ArrayType>::new(),
            "main",
            None,
            None,
        )
        .unwrap();
        let mut builder = ProgramBuilder::new();
        builder.add_input(input_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap());
        let metadata_only = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
            .unwrap()
            .into_unprojected::<Constant, Operation>()
            .unwrap();
        assert_eq!(collect_nested_sharding_mesh(&metadata_only, None, None).unwrap(), None);
        let incompatible = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Explicit).unwrap()]).unwrap();
        assert!(matches!(
            collect_nested_sharding_mesh(&program, Some(incompatible.clone()), None),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == incompatible && actual == mesh,
        ));
        assert!(matches!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type],
                &Vec::<ArrayType>::new(),
                "main",
                Some(&[Sharding::replicated(incompatible.clone(), 1)]),
                None,
            ),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == incompatible && actual == mesh,
        ));

        // Cast placement can be the module's only mesh use, including when its result is internal.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(4);
        let output_type = input_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        for operation in [
            CastOperation::<Type>::checked(output_type.clone().into(), vec![input_type.clone().into()])
                .unwrap()
                .into(),
            Operation::Array(ArrayOperation::Cast(
                CastOperation::checked(output_type, vec![input_type.clone()]).unwrap(),
            )),
        ] {
            let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(input_type.clone().into());
            builder.add_instruction(operation, vec![], vec![input], None).unwrap();
            let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![], vec![Placeholder], vec![]).unwrap();
            assert_eq!(collect_nested_sharding_mesh(&program, None, None).unwrap(), Some(mesh.clone()));
            assert_eq!(
                to_mlir_module_for_program(
                    &program,
                    &[],
                    &vec![input_type.clone()],
                    &Vec::<ArrayType>::new(),
                    "main",
                    None,
                    None,
                )
                .unwrap(),
                indoc! {r#"
                    module {
                      sdy.mesh @mesh = <["x"=2]>
                      func.func @main(%arg0: tensor<4xf32>) {
                        %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                        return
                      }
                    }
                "#},
            );
        }
    }

    #[test]
    fn test_collect_nested_sharding_mesh_requires_one_topology() {
        let mesh_2x2 = |names: [&str; 2], r#type: MeshAxisType| {
            LogicalMesh::new(names.map(|name| MeshAxis::new(name, 2, r#type).unwrap()).to_vec()).unwrap()
        };
        let mesh = mesh_2x2(["x", "y"], MeshAxisType::Explicit);
        let input_type = test_vector_type(4);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        builder
            .add_instruction(
                ReshapeOperation::new(vec![4], vec![2, 2]).with_output_sharding(Sharding::replicated(mesh.clone(), 2)),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
            .unwrap()
            .into_unprojected::<Constant, Operation>()
            .unwrap();

        // Device ids follow the order of the axes of the module mesh, so a reordered mesh is another topology.
        let reordered = mesh_2x2(["y", "x"], MeshAxisType::Explicit);
        assert!(matches!(
            collect_nested_sharding_mesh(&program, Some(reordered.clone()), None),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == reordered && actual == mesh,
        ));
        assert!(matches!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type.clone()],
                &Vec::<ArrayType>::new(),
                "main",
                Some(&[Sharding::replicated(reordered.clone(), 1)]),
                None,
            ),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == reordered && actual == mesh,
        ));
        assert_eq!(
            LoweringError::MismatchedMeshTopology { expected: reordered, actual: mesh.clone() }.to_string(),
            "mesh `['x'=2:explicit, 'y'=2:explicit]` does not have the topology of the module mesh \
             `['y'=2:explicit, 'x'=2:explicit]`",
        );

        // Meshes with other axes are not merged into a union.
        let other_axes = mesh_2x2(["x", "z"], MeshAxisType::Explicit);
        assert!(matches!(
            collect_nested_sharding_mesh(&program, Some(other_axes.clone()), None),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == other_axes && actual == mesh,
        ));

        // Meshes that differ only in axis types share the module mesh, which keeps the axis types of its first mesh.
        let manual = mesh_2x2(["x", "y"], MeshAxisType::Manual);
        assert_eq!(collect_nested_sharding_mesh(&program, Some(manual.clone()), None).unwrap(), Some(manual.clone()));
        assert_eq!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type],
                &Vec::<ArrayType>::new(),
                "main",
                Some(&[Sharding::replicated(manual, 1)]),
                None,
            )
            .unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>
                  func.func @main(%arg0: tensor<4xf32> \
                      {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={\"x\", \"y\"}>}) {
                    %0 = stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<2x2xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}, {}], replicated={\"x\", \"y\"}> : tensor<2x2xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_collect_nested_sharding_mesh_skips_bodies_that_the_target_replaces() {
        // A named composite call whose body places its input on another mesh contributes that mesh where lowering
        // emits the body, and nothing on a target whose backend implementation replaces the whole call
        // (`ryft.tests.xla_named_square` version 1 on CUDA, refer to `omits_regions`), which never lowers the body,
        // as the validation of placements skips it.
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let other = LogicalMesh::new(vec![MeshAxis::new("other", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_devices = DeviceMesh::new(other.clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let placement = Placement::new(other_devices, Sharding::replicated(other.clone(), 1)).unwrap();
        let body = {
            let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let operation = PlaceOperation::new(PlacementRequest::from(placement), None);
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let key = CompositeKey::new("ryft.tests.xla_named_square".to_string(), 1);
        let named = NamedComposite::new(key, |types: &[Type]| Ok(types.to_vec()), |inputs| Ok(vec![inputs[0].clone()]));
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::new("placing").with_named_composite(named));
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector_type.into());
        let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
        let regions = vec![builder.intern_callee(&body, None).unwrap()];
        let output = builder.add_instruction(operation, regions, vec![input], None).unwrap()[0];
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder]);
        let program = program.unwrap();
        let mesh = |target_platform| collect_nested_sharding_mesh(&program, None, target_platform).unwrap();
        assert_eq!(mesh(Some("cuda")), None);
        assert_eq!(mesh(Some("cpu")), Some(other));
    }

    #[test]
    fn test_broadcast_changes_explicit_sharding() {
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = test_vector_type(4).with_sharding(Sharding::replicated(explicit_mesh.clone(), 1)).unwrap();
        let explicitly_sharded = test_vector_type(4)
            .with_sharding(Sharding::new(explicit_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&test_vector_type(4), &replicated, &[0]));
        assert!(broadcast_changes_explicit_sharding(&replicated, &explicitly_sharded, &[0]));
        assert!(!broadcast_changes_explicit_sharding(&explicitly_sharded, &explicitly_sharded, &[0]));

        let first_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let second_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let first_mesh_replicated = test_vector_type(4).with_sharding(Sharding::replicated(first_mesh, 1)).unwrap();
        let second_mesh_replicated = test_vector_type(4).with_sharding(Sharding::replicated(second_mesh, 1)).unwrap();
        assert!(broadcast_changes_explicit_sharding(&first_mesh_replicated, &second_mesh_replicated, &[0]));

        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = test_vector_type(4).with_sharding(Sharding::replicated(manual_mesh.clone(), 1)).unwrap();
        let manually_sharded = test_vector_type(4)
            .with_sharding(
                Sharding::new(manual_mesh, vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert!(!broadcast_changes_explicit_sharding(&replicated, &manually_sharded, &[0]));

        // Expanding a partitioned singleton and changing explicit reduction state both require constraints even when
        // the ranked sharding dimensions themselves remain unchanged.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let singleton = test_vector_type(1)
            .with_sharding(Sharding::new(explicit_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let expanded = test_vector_type(4)
            .with_sharding(Sharding::new(explicit_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&singleton, &expanded, &[0]));

        let unreduced = test_vector_type(4)
            .with_sharding(Sharding::replicated(explicit_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let reduced = test_vector_type(4)
            .with_sharding(Sharding::replicated(explicit_mesh, 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&unreduced, &reduced, &[0]));
    }

    #[test]
    fn test_cast_changes_explicit_placement() {
        let explicit_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let resized_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let extended_auto_mesh = LogicalMesh::new(vec![
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("b", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let unsharded = ArrayType::new_static(DataType::F64, [4, 4]);
        let placed = |mesh: &LogicalMesh, first: ShardingDimension| {
            unsharded
                .clone()
                .with_sharding(Sharding::new(mesh.clone(), vec![first, ShardingDimension::replicated()]).unwrap())
                .unwrap()
        };
        let sharded = |axes: &[&str]| ShardingDimension::sharded(axes.iter().copied());
        let replicated = ShardingDimension::replicated;

        // An output without a sharding stays untracked, whatever the input placement.
        assert!(!cast_changes_explicit_placement(&unsharded, &unsharded));
        assert!(!cast_changes_explicit_placement(&placed(&explicit_mesh, sharded(&["x"])), &unsharded));

        // An unsharded input acquires a tracked placement only when the output mesh has explicit axes. Types record no
        // placement over auto mesh axes, so the placements below are replicated over them.
        assert!(cast_changes_explicit_placement(&unsharded, &placed(&explicit_mesh, replicated())));
        assert!(!cast_changes_explicit_placement(&unsharded, &placed(&auto_mesh, replicated())));

        // Carrying a mesh forward requests a redistribution only when the placement over explicit axes conflicts.
        assert!(!cast_changes_explicit_placement(
            &placed(&explicit_mesh, sharded(&["x"])),
            &placed(&explicit_mesh, sharded(&["x"])),
        ));
        assert!(cast_changes_explicit_placement(
            &placed(&explicit_mesh, sharded(&["x"])),
            &placed(&explicit_mesh, sharded(&["y"])),
        ));
        assert!(cast_changes_explicit_placement(
            &placed(&explicit_mesh, replicated()),
            &placed(&explicit_mesh, sharded(&["x"])),
        ));

        // A mesh change requests one whenever either mesh has explicit axes, but never between auto-only meshes.
        assert!(cast_changes_explicit_placement(
            &placed(&explicit_mesh, replicated()),
            &placed(&resized_mesh, replicated()),
        ));
        assert!(cast_changes_explicit_placement(
            &placed(&explicit_mesh, replicated()),
            &placed(&auto_mesh, replicated()),
        ));
        assert!(!cast_changes_explicit_placement(
            &placed(&auto_mesh, replicated()),
            &placed(&extended_auto_mesh, replicated()),
        ));
    }

    #[test]
    fn test_collect_nested_sharding_mesh_includes_call_boundaries() {
        // A call constrains its result to the override of the boundary of its callee, which no type records because the
        // override is over an auto mesh axis, so the module declares the mesh of the override even when it is lowered
        // without signature shardings, both for a static program and for one with a bounded dynamic value. The call
        // also constrains its operand replicated over that mesh, as an execution of its callee, whose input type has no
        // sharding, places its argument (refer to `CallOperation::boundary_input_sharding`).
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let lower = |array_type: ArrayType| {
            let callee = {
                let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
                let input = builder.add_input(array_type.clone().into());
                Arc::new(
                    builder
                        .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                        .unwrap(),
                )
            };
            let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(array_type.clone().into());
            let callee_region = builder.intern_callee(&callee, None).unwrap();
            let call =
                CallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), vec![Some(replicated.clone())]);
            let output =
                builder.add_instruction(Operation::Call(call), vec![callee_region], vec![input], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            assert_eq!(collect_nested_sharding_mesh(&program, None, None).unwrap(), Some(mesh.clone()));
            let types = vec![array_type];
            to_mlir_module_for_program(&program, &[], &types, &types, "main", None, None).unwrap()
        };
        assert_eq!(
            lower(test_vector_type(4)),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}]> : tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}]> : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
        let length = DimensionVariable::new("length", DimensionBounds::new(1, Some(4)).unwrap());
        assert_eq!(
            lower(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length)]))),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<3xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<3>>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}]> : tensor<?xf32, #stablehlo.bounds<3>>
                    %2 = sdy.sharding_constraint %1 <@mesh, [{}]> : tensor<?xf32, #stablehlo.bounds<3>>
                    %3 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                    %4 = stablehlo.convert %3 : (tensor<i32>) -> tensor<i64>
                    return %2, %4 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_normalize_function_name() {
        assert_eq!(normalize_function_name(" main ").unwrap(), "main");
        assert_eq!(normalize_function_name(" @main ").unwrap(), "main");
        assert!(matches!(
            normalize_function_name(" @ "),
            Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name })) if function_name.is_empty(),
        ));
        assert!(matches!(
            normalize_function_name("main function"),
            Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name })) if function_name == "main function",
        ));
    }
}
