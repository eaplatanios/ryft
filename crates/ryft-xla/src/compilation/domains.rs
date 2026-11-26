//! Compilation lifecycle of the XLA [`Session`], which implements [`CompilationDomain`] and [`CompilationCacheDomain`].
//!
//! `Session::lower_staged_program` validates the boundary of a staged program against the sharding overrides of its
//! [`Options`]. `Session::lower_program` then selects kernels, discharges references, places every executable argument
//! and result, and emits the StableHLO module together with its invocation contract. `Session::compile_program`
//! compiles that module through the PJRT client of the session and derives the reusable execution plan.
//! [`with_input_shardings`] applies input sharding overrides to input types before staging.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ryft_core::{
    ArrayOperation, ArrayType, CALL_OPERATION_NAME, CallOperation, CapturePlacement, CompilationCacheDomain,
    CompilationDomain, Device, DeviceMesh, Dimension, DispatchInputs, ExternalReferenceBinding, FlatProgram,
    LogicalMesh, Memory, Operation as CoreOperation, OperationPayloadProjection, PLACE_OPERATION_NAME, Parameterized,
    PlaceOperation, PlacementRequest, ProgramError, ReferenceDischargeResult, RegionRole, SHARD_MAP_OPERATION_NAME,
    Sharding, ShardingDimension, ShardingError, StagedProgram, Type as CoreType, TypeError, Typed, TypedProgram,
    declared_input_sharding, states_boundary_placement,
};
use ryft_pjrt::protos::{
    CompilationOptions, ComputationDeviceAssignment, DeviceAssignment, ExecutableCompilationOptions, OpShardingType,
};
use ryft_pjrt::{LoadedExecutable, Program as PjrtProgram};

use crate::arrays::host::validate_mesh_devices;
use crate::compilation::CompilationError;
use crate::compilation::keys::{CompilationKey, DispatchKey, dispatch_signature};
use crate::compilation::options::{FeedbackDirectedProfile, Options, resolve_shardy_debug_options, supports_shardy};
use crate::compilation::programs::{
    CompiledProgram, EagerDispatchProgram, LoweredProgram, OutputDeviceAssignment, ProgramLowering, RemoteEagerProgram,
    SessionConstantBinding,
};
use crate::execution::{ExecutionPlan, validate_capture_reduction};
use crate::lowering::admission::validate_data_dependent_compilation;
use crate::lowering::emissions::{
    collect_session_constants, emitted_capture_reads, emitted_regions, omits_regions, unread_entry_arguments,
};
use crate::lowering::extensions::{
    OutputShardingSource, ProgramLoweringExtension, ShardingLoweringExtension, entry_open_outputs,
    entry_output_shardings,
};
use crate::lowering::modules::{contains_unresolved_references, contains_unresolved_state};
use crate::lowering::signatures::ExecutableSignature;
use crate::{
    Array, Constant, Error, ExecutionError, LoweringError, OpShardingExtension, Operation, RetainedArray, Session,
    Type, Value,
};

// TODO(eaplatanios): Review this module.

impl CompilationDomain for Session {
    type DispatchKey = DispatchKey;
    type LoweredProgram = LoweredProgram;
    type CompiledProgram = CompiledProgram;
    type Options = Options;
    type Error = Error;

    fn dispatch_signature(
        inputs: DispatchInputs<'_, Self::Value>,
        options: &Self::Options,
    ) -> Result<(Self::DispatchKey, Arc<[Type]>, Option<Self::Options>), Self::Error> {
        dispatch_signature(inputs, options)
    }

    // Every compiled function places its captures where its call runs (refer to `captured_type`). The mesh on which it
    // receives them comes from `options` when it is lowered.
    fn capture_placement(_options: &Options) -> CapturePlacement {
        CapturePlacement::FollowCall
    }

    // A capture follows the call of the function that captures it, so the function receives every capture whose type
    // records nothing beyond replication with a type that names no mesh (refer to `captured_type`), as a session
    // constant of that type, whose array lowering places only if the function uses it. Every runtime array that enters
    // the trace of a compiled function takes this route, an explicit capture and an import alike. A capture whose type
    // records more (e.g., a split over explicit mesh axes) stays an ordinary capture with its own type, and one that the
    // function cannot receive on its mesh is rejected by lowering only if the function uses it.
    fn capture_constant(capture: &Value) -> Result<Option<Constant>, ProgramError> {
        let Value::Array(array) = capture else { return Ok(None) };
        let Some(captured_type) = captured_type(&array.r#type()) else { return Ok(None) };
        Ok(Some(Constant::Session(RetainedArray::new(array).with_declared_type(captured_type))))
    }

    // A `call` of a compiled function carries the boundary of that function (i.e., its sharding overrides and the
    // device mesh that the function names, refer to `CallOperation::with_boundary`), because the callee cannot show
    // it (e.g., a replicated output override over auto mesh axes stages nothing, and a callee that a caller on other
    // devices inlines no longer shows where it was compiled to run). The function names its devices exactly when it
    // requests a placement that names devices (refer to `requests_placement`), which a request that only moves a value
    // to another memory does not. Otherwise public arguments or the enclosing caller
    // select the devices, even for a function with committed captures. A dispatcher without input overrides also
    // keeps each argument's sharding over auto mesh axes, because it specializes on the placement of runtime inputs
    // (refer to `dispatch_signature`). Every call also records the mesh of its function, named or not, which only an
    // application of the call by a session reads, to run there (refer to `CallOperation::with_function_mesh`), and
    // which a call nested in a compiled caller never reads. Core constructs the call operation once for each staged
    // function.
    fn call_operation(
        call: CallOperation<Type>,
        callee: &FlatProgram<Session>,
        options: &Options,
        dispatched: bool,
    ) -> CallOperation<Type> {
        // The lifted capture prefix of the callee precedes its inputs and has no overrides.
        let input_shardings = std::iter::repeat_n(None, call.capture_count())
            .chain(options.input_shardings().into_iter().flatten().cloned().map(Some))
            .collect();
        let output_shardings = options.output_shardings().into_iter().flatten().cloned().map(Some).collect();
        // No session is at hand here, so the callee is read for no target, and the pruning of the program that
        // lowering emits for its target drops the mesh of a call whose callee names devices only inside a body that
        // the target replaces (refer to `ProgramLoweringExtension::pruned_for_lowering`).
        let mesh = requests_placement(callee, options, None, false).then(|| options.mesh());
        let call = call.with_boundary(mesh, input_shardings, output_shardings).with_function_mesh(options.mesh());
        if dispatched && options.input_shardings().is_none() { call.with_kept_input_placements() } else { call }
    }

    fn lower(&self, staged: &StagedProgram<Self>) -> Result<LoweredProgram, Self::Error> {
        self.lower_staged_program(staged)
    }

    // A cached executable contains no capture buffers. Materialize this lowering's hidden arguments on every
    // compile request, including cache hits.
    fn compile(&self, program: &LoweredProgram) -> Result<(Arc<CompiledProgram>, Vec<Value>), Self::Error> {
        let session_arrays = program
            .session_constants()
            .iter()
            .map(|binding| Ok(Value::Array(binding.argument().to_array(self)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        let compiled = self.compilation_cache().compile(self, program)?;

        Ok((compiled, session_arrays))
    }

    fn execute(&self, program: &CompiledProgram, inputs: Vec<Value>) -> Result<Vec<Value>, Self::Error> {
        self.execute_stateless(program, inputs).map(|(outputs, _)| outputs)
    }
}

impl Session {
    /// Lowers a staged program after validating that its boundary types carry the sharding overrides of its options and
    /// that no reference crosses its output boundary. This function implements
    /// [`CompilationDomain::lower`].
    fn lower_staged_program(&self, staged: &StagedProgram<Self>) -> Result<LoweredProgram, Error> {
        let program = staged.lifted_program()?;
        let program = program.as_ref();
        let capture_count = staged.source_program().captures().len();
        let input_types = staged.input_types();
        let output_types = staged.output_types();
        let options = staged.options();
        let mut declared_input_types = input_types.to_vec();
        apply_input_shardings(&mut declared_input_types, options, |r#type| r#type)?;
        if declared_input_types != input_types {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: "staged input types do not carry the shardings requested by `input_shardings`; apply them to \
                         the input types before staging (e.g., with `with_input_shardings`)"
                    .to_string(),
            }));
        }
        validate_reference_outputs(program, capture_count, output_types)?;
        validate_output_shardings(output_types, options)?;

        // The captures that survive pruning are placed as `capture_sharding` resolves them, which is as they are at
        // runtime, including over the auto mesh axes that their types do not record, so that executions need not
        // reshard them. A capture whose placement the function cannot keep fails here, and only if the function uses
        // it. So does a capture of another client than that of this session, with the error with which a session
        // constant of another client fails (refer to `placed_constant`), because every execution would reject it.
        // Staging already pruned the captures that the function does not use, which are therefore never checked. A
        // capture that only regions which lowering omits read (e.g., the body of a scan that admits no iteration) is
        // not checked either, because the executable never reads it, and lowering erases its argument through the same
        // decision (refer to `emitted_capture_reads`).
        let target_platform = Some(self.target()?.platform_name());
        let entry = staged.source_program().program().entry_region_ref();
        let is_read = emitted_capture_reads(entry, capture_count, false, target_platform);
        let capture_placements = staged
            .source_program()
            .captures()
            .iter()
            .enumerate()
            .map(|(index, capture)| match capture {
                Value::Array(_) if !is_read[index] => Ok(None),
                Value::Array(array) => {
                    if !std::ptr::eq(array.client(), self.client()) {
                        RetainedArray::new(array).validate_client(self.client())?;
                    }
                    admit_capture(array.sharding(), index, capture_count)?;
                    let sharding = capture_sharding(
                        &format!("capture #{index}"),
                        &array.r#type(),
                        array.sharding(),
                        options.mesh().logical_mesh(),
                    )?;
                    Ok((&sharding != array.sharding() || sharding.references_auto_axis()).then_some(sharding))
                }
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let placed_options;
        let options = if capture_placements.iter().any(Option::is_some) {
            placed_options = options.clone().with_capture_placements(capture_placements.into());
            &placed_options
        } else {
            options
        };
        let lowered_program = self.lower_function_program(program, capture_count, options, true)?;
        // A compiled function accepts exactly the arguments that its declared input types describe, so the public input
        // types of its executable are those types, which lowering never changes.
        let lowered_input_types = lowered_program.input_types();
        let lowered_public_input_types =
            &lowered_input_types[lowered_input_types.len().saturating_sub(input_types.len())..];
        if lowered_public_input_types.len() != input_types.len()
            || input_types
                .iter()
                .zip(lowered_public_input_types)
                .any(|(declared, lowered)| <&ArrayType>::try_from(declared).is_ok_and(|declared| declared != lowered))
        {
            return Err(ProgramError::MalformedProgram(format!(
                "the public input types `{}` of the executable are not the declared input types `{}` of the function",
                lowered_public_input_types.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                input_types.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
            ))
            .into());
        }
        // Lowering gives the program outputs the types of the output sharding overrides without their auto mesh axes,
        // so the staged output types are checked against the lowered ones with those same overrides applied.
        let mut declared_output_types = output_types.to_vec();
        apply_signature_shardings(declared_output_types.iter_mut(), options.output_shardings(), "output")?;
        validate_output_types(&declared_output_types, lowered_program.output_types())?;
        Ok(lowered_program)
    }

    /// Lowers a staged program to StableHLO and derives its complete logical and physical invocation contract for this
    /// session. A mesh that assigns a device to another process than the one that owns it, or that assigns to this
    /// process a device that it cannot address, is rejected first, as a direct placement and a placement of a capture
    /// reject it (refer to `validate_mesh_devices`), before anything is lowered, compiled, or looked up in a
    /// compilation cache, so that every process rejects it alike. The program is lowered as the program of an eager
    /// application, whose mesh is where its operands are, so the mesh does not commit its outputs for being named
    /// (refer to [`commits_public_outputs`] and `lower_function_program`).
    pub(crate) fn lower_program(
        &self,
        program: &FlatProgram<Session>,
        capture_count: usize,
        options: &Options,
    ) -> Result<LoweredProgram, Error> {
        self.lower_function_program(program, capture_count, options, false)
    }

    /// Creates a lowering with native compilation metadata. When `function` is true, a named mesh commits the public
    /// outputs (refer to [`commits_public_outputs`]).
    fn lower_function_program(
        &self,
        program: &FlatProgram<Session>,
        capture_count: usize,
        options: &Options,
        function: bool,
    ) -> Result<LoweredProgram, Error> {
        let lowering = self.lower_program_invocation(program, capture_count, options, function)?;
        LoweredProgram::new(lowering, self.target()?)
    }

    /// Prepares the executable or remote result metadata for an eager application, deriving both from the same lowering.
    pub(crate) fn prepare_eager_program(
        &self,
        program: &FlatProgram<Session>,
        options: &Options,
        session_constants: &[RetainedArray],
    ) -> Result<(EagerDispatchProgram, Vec<usize>), ProgramError> {
        let lowering = self.lower_program_invocation(program, 0, options, false)?;
        let target = self.target()?;

        // Record the first application slot of each hidden argument's source constant.
        let mut slots = HashMap::with_capacity(session_constants.len());
        for (slot, constant) in session_constants.iter().enumerate() {
            slots.entry(constant).or_insert(slot);
        }
        let constants = lowering
            .session_constants()
            .iter()
            .map(|binding| {
                let source = binding.source();
                slots.get(source).copied().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "lowered session constant `{source}` comes from no session constant of the eager application",
                    ))
                })
            })
            .collect::<Result<_, _>>()?;

        let program = if matches!(
            options.mesh().devices(),
            [device] if device.process_index() != target.process_index()
        ) {
            RemoteEagerProgram::new(&lowering).map(|program| EagerDispatchProgram::Remote(Arc::new(program)))
        } else {
            let lowered = LoweredProgram::new(lowering, target)?;
            self.compilation_cache().compile(self, &lowered).map(EagerDispatchProgram::Local)
        }?;
        Ok((program, constants))
    }

    /// Emits the shared module and invocation contract, including for a declared remote singleton whose device kind
    /// is unavailable to the local client.
    fn lower_program_invocation(
        &self,
        program: &FlatProgram<Session>,
        capture_count: usize,
        options: &Options,
        function: bool,
    ) -> Result<ProgramLowering, Error> {
        validate_mesh_devices(self, options.mesh())?;
        // Every step below sees the program that lowering emits, without its dead instructions (refer to
        // `ProgramLoweringExtension::pruned_for_lowering`), as the analysis of its placement requests does.
        let target_platform = Some(self.target()?.platform_name());
        self.lower_pruned_program(&program.pruned_for_lowering(target_platform)?, capture_count, options, function)
    }

    /// Lowers `program` after dead instruction pruning.
    fn lower_pruned_program(
        &self,
        program: &FlatProgram<Session>,
        capture_count: usize,
        options: &Options,
        function: bool,
    ) -> Result<ProgramLowering, Error> {
        // Kernel selection validates the actual attached body and discharges only its owned local state before the
        // ordinary reference discharge and effect gates. Existing lowering, caching, and runtime lifecycle then apply.
        let selected = crate::kernels::select_kernels(program, options.kernel_compiler(), || {
            let target = self
                .target()
                .map_err(|error| crate::kernels::KernelEmbeddingError::Invalid { message: error.to_string() })?;
            crate::kernels::KernelExecutionFacts::from_target(target, options.mesh())
        })?;
        let program = selected.as_ref().unwrap_or(program);
        if capture_count > program.input_count() {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "`capture_count` is {capture_count}, but the program has only {} flat input(s)",
                    program.input_count(),
                ),
            }
            .into());
        }
        // Avoid cloning the dominant reference-free program path. Discharge returns such a program untouched with an
        // empty external state list, so the same metadata can cross the lowering boundary by borrow instead.
        if !contains_unresolved_references(program) {
            return self.lower_verified_reference_free_program(
                program,
                capture_count,
                program.output_count(),
                &[],
                options,
                function,
            );
        }
        // Discharge the exact lifted program so capture references of attached regions resolve against the same lexical
        // capture prefixes used during staging. Programs that actually contain references are cloned because
        // discharge consumes and rewrites them; the reference-free identity path above remains borrowed.
        let discharged = program.clone().discharge_references_in_capture_lifted_program(capture_count)?;
        self.lower_discharged_program(&discharged, options, function)
    }

    /// Lowers one validated reference discharge artifact through the ordinary array-only XLA path (refer to
    /// `lower_function_program` for `function`).
    fn lower_discharged_program(
        &self,
        discharged: &ReferenceDischargeResult<Constant, Operation>,
        options: &Options,
        function: bool,
    ) -> Result<ProgramLowering, Error> {
        // The backend verifier remains independent from core discharge and runs before boundary projection,
        // partitioner selection, or StableHLO construction. The reference scan runs first so a malformed artifact
        // containing both a reference and ordered state receives the more precise reference diagnostic.
        if contains_unresolved_references(discharged.program()) {
            return Err(ProgramError::UnsupportedOperation {
                message: "discharged program still contains references, which XLA lowering cannot represent"
                    .to_string(),
            }
            .into());
        }
        self.lower_verified_reference_free_program(
            discharged.program(),
            discharged.capture_count(),
            discharged.output_count(),
            discharged.external_reference_bindings(),
            options,
            function,
        )
    }

    /// Lowers one program after an independent verifier has established that it contains no references (refer to
    /// `lower_function_program` for `function`).
    fn lower_verified_reference_free_program(
        &self,
        program: &FlatProgram<Session>,
        capture_count: usize,
        output_count: usize,
        reference_states: &[ExternalReferenceBinding],
        options: &Options,
        function: bool,
    ) -> Result<ProgramLowering, Error> {
        if contains_unresolved_state(program) {
            return Err(ProgramError::UnsupportedOperation {
                message: "discharged program still contains ordered state, which XLA lowering cannot represent"
                    .to_string(),
            }
            .into());
        }
        // The target platform gates platform-specific lowerings such as fused attention, and so also the regions that
        // lowering emits. A client whose platform cannot be queried cannot compile the program either, so a failed
        // query fails the lowering.
        let target = self.target()?;
        let target_platform = Some(target.platform_name());
        validate_data_dependent_compilation(program, target_platform)?;
        if output_count > program.output_count() {
            return Err(ProgramError::MalformedProgram(format!(
                "output count {output_count} exceeds the complete discharged output count {}",
                program.output_count(),
            ))
            .into());
        }
        // Each process binds, executes, and commits only its addressable shards of a reference, so a mesh that spans
        // processes is admitted as long as this process participates in it, as `Target::is_participating_mesh` checks.
        if !reference_states.is_empty() && !self.target()?.is_participating_mesh(options.mesh()) {
            return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                reason:
                    "external reference state requires a mesh of devices that this client sees under the processes \
                         that own them, at least one of which this process addresses"
                        .to_string(),
            }));
        }
        if !reference_states.is_empty() && options.input_bound_bucketing().is_some() {
            return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                reason: "external reference inputs do not support input-bound bucketing".to_string(),
            }));
        }
        let input_types = program.input_types();
        let (capture_types, public_input_types) = input_types.split_at(capture_count);
        if let Some(donation_flags) = &options.donation_flags()
            && donation_flags.len() != public_input_types.len()
        {
            return Err(Error::Execution(ExecutionError::DonationFlagCountMismatch {
                expected_count: public_input_types.len(),
                actual_count: donation_flags.len(),
            }));
        }
        let mut donation_flags =
            options.donation_flags().map_or_else(|| vec![false; public_input_types.len()], <[bool]>::to_vec);

        let mut effective_capture_types = capture_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        // Shardy rejects bounded dynamic tensors anywhere in the module, not only at the computation boundary, so a
        // program whose boundary is static but whose interior derives a dynamic extent through a gateway operation
        // still has to take the replica path. This is why every lowered region is inspected instead of just the
        // boundary types, which it already subsumes. A target whose plugin cannot compile with Shardy takes that path for
        // every program, and only on a single device, where lowering can drop every sharding constraint.
        let supports_shardy = supports_shardy(self.client())?;
        if !supports_shardy && options.mesh().devices().len() != 1 {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: format!(
                    "Ryft requires a single-device mesh because the plugin for platform `{}` cannot compile with Shardy",
                    target.platform_name(),
                ),
            }));
        }
        let use_shardy_partitioner = supports_shardy && has_only_static_array_types(program, target_platform);
        // Validate the emitted placement requests before placing any retained arrays on the compilation mesh.
        validate_placements(program, options.mesh(), use_shardy_partitioner, target_platform)?;
        // A `shard_map` body only ever has static boundary types, so it can coexist with dynamic tensors staged
        // elsewhere in the same program. Its `sdy.manual_computation` lowering is meaningless without Shardy on any
        // device count, and bounded dynamic tensors force Shardy off for the whole module.
        if !use_shardy_partitioner && contains_shard_map(program, target_platform) {
            return Err(Error::Lowering(crate::LoweringError::UnsupportedOperation {
                operation: SHARD_MAP_OPERATION_NAME.to_string(),
                reason: if supports_shardy {
                    "bounded-dynamic programs cannot contain `shard_map` regions because `sdy.manual_computation` \
                     requires the Shardy partitioner"
                        .to_string()
                } else {
                    format!(
                        "platform `{}` cannot run `shard_map` regions because `sdy.manual_computation` requires the \
                         Shardy partitioner",
                        target.platform_name(),
                    )
                },
            }));
        }
        // Assign hidden argument slots from the original constants in the program that lowering emits for the target.
        // Each constant supplies its declared type, and its placed array supplies buffers and runtime sharding.
        let session_constants = collect_session_constants(program, target_platform)
            .into_iter()
            .map(|source| {
                let argument = placed_constant(self, &source, options.mesh())?.unwrap_or_else(|| source.clone());
                Ok::<_, Error>(SessionConstantBinding::new(source, argument))
            })
            .collect::<Result<Vec<_>, _>>()?;
        effective_capture_types.extend(session_constants.iter().map(|binding| binding.source().r#type().into_owned()));
        let lowered_capture_count = effective_capture_types.len();
        let effective_public_input_types = public_input_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let effective_input_types = effective_capture_types
            .iter()
            .cloned()
            .chain(effective_public_input_types.iter().cloned())
            .collect::<Vec<_>>();
        let reference_state_input_indices = reference_states
            .iter()
            .map(|state| state.source().flat_input_index(lowered_capture_count))
            .collect::<Result<Vec<_>, _>>()?;
        if reference_state_input_indices.windows(2).any(|indices| indices[0] >= indices[1]) {
            return Err(ProgramError::MalformedProgram(
                "external reference states must follow strictly increasing logical input order".to_string(),
            )
            .into());
        }
        if !reference_states
            .iter()
            .filter_map(ExternalReferenceBinding::output_index)
            .eq(output_count..program.output_count())
        {
            return Err(ProgramError::MalformedProgram(
                "hidden discharged outputs must be covered exactly once by mutated external state".to_string(),
            )
            .into());
        }
        for &logical_input_index in &reference_state_input_indices {
            let input_type = effective_input_types.get(logical_input_index).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "external state logical input {logical_input_index} is out of range",
                ))
            })?;
            if let Some(sharding) = input_type.sharding()
                && !sharding.mesh().has_same_topology(options.mesh().logical_mesh())
            {
                return Err(ShardingError::MeshMismatch {
                    expected: options.mesh().logical_mesh().clone(),
                    actual: sharding.mesh().clone(),
                }
                .into());
            }
            if input_type.data_type().is_zero() {
                return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                    reason: format!(
                        "external reference state at logical input {logical_input_index} cannot use a zero-space type \
                         because it has no executable buffer",
                    ),
                }));
            }
            if input_type.memory() != Memory::Device {
                return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                    reason: format!(
                        "external reference state at logical input {logical_input_index} must use device memory"
                    ),
                }));
            }
            if input_type
                .shape()
                .dimensions()
                .iter()
                .any(|dimension| dimension.value().is_none() && dimension.upper_bound().is_none())
            {
                return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                    reason: format!(
                        "external reference state at logical input {logical_input_index} has an unbounded dynamic \
                         dimension",
                    ),
                }));
            }
        }
        let program_output_types = program.output_types();
        let mut public_output_types = program_output_types[..output_count].to_vec();
        apply_signature_shardings(public_output_types.iter_mut(), options.output_shardings(), "output")?;
        // Executable arguments are placed as their runtime arrays are, or as an input override requests, including
        // over the auto mesh axes that their types never record. A runtime placement drops a pending reduction over
        // those axes, which no value of the program can carry, whereas an input override, which states the placement
        // with which the argument arrives, keeps it (refer to `receivable_argument_sharding`). An argument that the
        // executable never reads has no executable argument (refer to `unread_entry_arguments`), so its placement is
        // neither resolved nor validated, whatever its runtime array is (e.g., split unevenly, or of another client), and
        // it is annotated replicated, which no physical argument carries.
        let source_capture_count = lowered_capture_count - session_constants.len();
        let unread_arguments = unread_entry_arguments(
            program,
            &effective_input_types,
            lowered_capture_count,
            session_constants.len(),
            options.unread_inputs().unwrap_or(&[]),
            &reference_state_input_indices,
            target_platform,
        );
        let logical_argument_shardings = effective_input_types
            .iter()
            .enumerate()
            .map(|(index, array_type)| {
                if unread_arguments[index] {
                    return Ok(Sharding::replicated(options.mesh().logical_mesh().clone(), array_type.rank()));
                }
                let placement = if index < source_capture_count {
                    options.capture_placements().and_then(|placements| placements.get(index).cloned().flatten())
                } else if index < lowered_capture_count {
                    let constant_index = index - source_capture_count;
                    let binding = &session_constants[constant_index];
                    let argument = binding.argument();
                    argument.validate_client(self.client())?;
                    admit_capture(argument.sharding(), index, lowered_capture_count)?;
                    let description = format!("retained array #{constant_index}");
                    Some(capture_sharding(
                        &description,
                        &binding.source().r#type(),
                        argument.sharding(),
                        options.mesh().logical_mesh(),
                    )?)
                } else {
                    let index = index - lowered_capture_count;
                    if let Some(sharding) = options.input_shardings().and_then(|shardings| shardings.get(index)) {
                        return Ok(sharding.clone());
                    }
                    options.input_placements().and_then(|placements| placements.get(index).cloned().flatten())
                };
                match placement {
                    Some(placement) => receivable_argument_sharding(placement),
                    None => Ok(declared_input_sharding(array_type, options.mesh().logical_mesh())),
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        // The executable expects the buffers of every argument to be split evenly (refer to `validate_even_sharding`),
        // so an argument that a placement splits unevenly is rejected here rather than when execution launches.
        for (index, (array_type, sharding)) in effective_input_types.iter().zip(&logical_argument_shardings).enumerate()
        {
            let (kind, index) = if index < source_capture_count {
                ("capture", index)
            } else if index < lowered_capture_count {
                ("retained array", index - source_capture_count)
            } else {
                ("input", index - lowered_capture_count)
            };
            validate_even_sharding(kind, index, array_type, sharding)?;
        }
        // An argument is annotated as its runtime array stores it, which replicates the dimensions that its placement
        // leaves unconstrained, as the StableHLO export of traced programs annotates it (refer to
        // `ShardingLoweringExtension::with_unconstrained_dimensions_replicated`). Execution still expects the placement
        // itself, which describes the same buffers.
        let argument_annotations = logical_argument_shardings
            .iter()
            .map(ShardingLoweringExtension::with_unconstrained_dimensions_replicated)
            .collect::<Result<Vec<_>, _>>()?;
        for (state, &logical_input_index) in reference_states.iter().zip(&reference_state_input_indices) {
            if effective_input_types[logical_input_index].static_shape().is_none() {
                if state.output_index().is_some() {
                    return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                        reason: format!(
                            "mutated bounded-dynamic external state input {logical_input_index} is unsupported because \
                             backend alias compatibility has not been verified",
                        ),
                    }));
                }
                if !logical_argument_shardings[logical_input_index].is_replicated() {
                    return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                        reason: format!(
                            "bounded-dynamic external state input {logical_input_index} must use replicated sharding",
                        ),
                    }));
                }
                // Bounded dynamic inputs are padded through host copies, which have not been verified across processes.
                if !self.target()?.is_fully_addressable_single_process_mesh(options.mesh()) {
                    return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                        reason: format!(
                            "bounded-dynamic external state input {logical_input_index} requires a fully addressable \
                             single-process mesh",
                        ),
                    }));
                }
            }
        }
        let mut output_types = public_output_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        // A public output that a `place` in the entry region or in the callee of a `call` produces, directly or
        // through operations that return its value and placement unchanged, has the placement of its target, which
        // states the placement along every mesh axis, and one that returns an argument in the same way is placed where
        // that argument enters the executable (refer to `ProgramLoweringExtension::output_sharding_sources`). Its
        // type drops the auto mesh axes of that placement, and its result has no physical buffer whose placement the
        // compiler could report when it has the zero data type, so the placement is carried to the boundary instead.
        // Outputs follow the output override first, then the placement from the program, both including over auto mesh
        // axes, and otherwise their types. The StableHLO export of traced programs follows the same rule
        // (refer to `entry_output_shardings`). The arguments of the program are its captures and its public inputs,
        // between which the executable receives the arrays of session constants.
        let constant_shardings = session_constants
            .iter()
            .map(|binding| (binding.source(), binding.argument().sharding()))
            .collect::<HashMap<_, _>>();
        let sharding_sources =
            program.output_sharding_sources(output_count, |constant| constant_shardings[constant].clone());
        let program_argument_shardings = logical_argument_shardings[..source_capture_count]
            .iter()
            .chain(&logical_argument_shardings[lowered_capture_count..])
            .cloned()
            .map(Some)
            .collect::<Vec<_>>();
        // An output without a placement is replicated. These are the result annotations that lowering emits, and a
        // result is never left without one, which would let the compiler place it differently (e.g., as the output of a
        // `shard_map` that it is derived from) than execution expects.
        let mut output_annotations = entry_output_shardings(
            &output_types,
            options.output_shardings(),
            &sharding_sources,
            &program_argument_shardings,
        )?
        .into_iter()
        .zip(&output_types)
        .map(|(sharding, array_type)| {
            sharding.unwrap_or_else(|| {
                Sharding::replicated(options.mesh().logical_mesh().clone(), array_type.shape().rank())
            })
        })
        .collect::<Vec<_>>();
        // The executable returns its outputs split evenly as well, so an output that its annotation splits unevenly
        // could not be reconstructed with its runtime placement. Hidden reference states are placed as their inputs.
        for (index, (array_type, sharding)) in output_types.iter().zip(&output_annotations).enumerate() {
            validate_even_sharding("output", index, array_type, sharding)?;
        }
        // Reconstruct the outputs with the placements of their executable annotations, preserving authored
        // unconstrained dimensions in runtime metadata (refer to `runtime_output_sharding`).
        let mut output_shardings = output_annotations
            .iter()
            .zip(&output_types)
            .enumerate()
            .map(|(index, (annotation, array_type))| {
                // An anonymous singleton can omit a requested unconstrained sharding from its array type.
                // Keep the unclosed placement of an explicit request or of an argument that the output forwards.
                let requested = options.output_shardings().map(|shardings| &shardings[index]).or_else(|| {
                    match &sharding_sources[index] {
                        Some(OutputShardingSource::Determined(sharding)) => Some(sharding),
                        Some(OutputShardingSource::Argument(index)) => {
                            program_argument_shardings.get(*index).and_then(Option::as_ref)
                        }
                        None => None,
                    }
                });
                runtime_output_sharding(annotation, array_type.sharding().or(requested))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        // Mutated bindings cover the hidden output suffix in order, as validated above. Derive both the hidden state
        // type and its placement from the same input binding in one pass.
        for (state, &logical_input_index) in reference_states.iter().zip(&reference_state_input_indices) {
            if state.is_mutated() {
                output_types.push(effective_input_types[logical_input_index].clone());
                output_annotations.push(argument_annotations[logical_input_index].clone());
                output_shardings.push(logical_argument_shardings[logical_input_index].clone());
            }
        }
        // Every public output of a function that requests a placement explicitly anywhere is committed to its devices
        // by every execution, as JAX commits every result of a `jit` function with any explicit sharding, which makes
        // this a property of the whole function rather than of the outputs that the request reaches. The function
        // requests a placement through an input or output sharding override, or through a `place` with a target or a
        // `constrain_sharding` in any region that lowering emits (refer to
        // `ProgramLoweringExtension::has_placement_requests`). Every output of a function compiled for more than one
        // device is committed as well, including its hidden reference states, as JAX commits every result of a
        // computation on more than one device (refer to
        // `lower_sharding_computation` in `jax._src.interpreters.pxla`), whereas a function that is compiled for a
        // single device leaves its outputs to the other rules. Every public output of a function compiled for a named
        // mesh (`function`) is committed too, as JAX commits every result of a `jit` function under `set_mesh` (refer
        // to `commits_public_outputs`). Other outputs are committed when a public argument of an execution is. The
        // arrays that a function captures never commit its outputs, committed or not, as JAX returns
        // `jit(lambda x: x + c)(x)` uncommitted for an uncommitted `x` whatever the commitment of `c`. Eager
        // operations and compiled reshards set the commitment of their results themselves, so these flags do not
        // reach them.
        let commits_public_outputs = commits_public_outputs(
            options.mesh(),
            function,
            requests_placement(program, options, target_platform, true),
        );
        let runs_on_multiple_devices = options.mesh().devices().len() > 1;
        let committed_outputs = (0..output_types.len())
            .map(|index| runs_on_multiple_devices || index < output_count && commits_public_outputs)
            .collect::<Vec<_>>();
        let output_device_assignment = if options.mesh().logical_mesh().is_named()
            || requests_placement(program, options, target_platform, false)
        {
            OutputDeviceAssignment::Fixed
        } else {
            OutputDeviceAssignment::FromCaller
        };
        // The effective placements of the arguments and results (i.e., their annotations) are checked rather than their
        // types, which do not record placements along auto mesh axes (e.g., an output override that splits an output
        // along an auto axis), so that no replica returns a full buffer where the runtime placement describes a part.
        if !use_shardy_partitioner
            && options.mesh().devices().len() > 1
            && argument_annotations.iter().chain(&output_annotations).any(|sharding| !sharding.is_replicated())
        {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: "bounded-dynamic multi-device programs currently require fully replicated shardings \
                             because Shardy rejects dynamic tensors"
                    .to_string(),
            }));
        }
        let mut compilation_options = jit_compilation_options(
            options.compilation_options().unwrap_or_else(|| self.compilation_options()),
            options.mesh().devices(),
            use_shardy_partitioner,
            options.feedback_directed_profile(),
        );
        resolve_shardy_debug_options(self.client(), &mut compilation_options);
        // Shardy currently rejects bounded dynamic tensors. Fully replicated dynamic programs need no partitioning,
        // so lower them without Shardy metadata and execute one replica per mesh device instead.
        let lowered_argument_shardings = use_shardy_partitioner.then_some(argument_annotations.as_slice());
        let lowered_output_shardings = use_shardy_partitioner.then_some(output_annotations.as_slice());
        // Public outputs without an override leave their placement over auto mesh axes to the compiler, like JAX
        // outputs without `out_shardings`, and compilation reads that placement back from the executable. The StableHLO
        // export of traced programs makes the same decision (refer to `entry_open_outputs`). Hidden reference states
        // are always placed as declared, and so is every output of a program that runs as replicas without Shardy.
        let public_open_outputs = entry_open_outputs(
            &output_types[..output_count],
            options.output_shardings(),
            &sharding_sources,
            &output_annotations[..output_count].iter().cloned().map(Some).collect::<Vec<_>>(),
        );
        let open_outputs = (0..output_shardings.len())
            .map(|index| use_shardy_partitioner && index < output_count && public_open_outputs[index])
            .collect::<Vec<_>>();
        let lowered_module = crate::lowering::modules::lower_mlir_module_for_program_with_reference_state(
            program,
            effective_capture_types.as_slice(),
            &effective_public_input_types,
            &output_types,
            "main",
            lowered_argument_shardings,
            lowered_output_shardings,
            target_platform,
            supports_shardy,
            reference_states,
            &donation_flags,
            options.unread_inputs().unwrap_or(&[]),
            options.ragged_dot_lowering_strategy(),
            &open_outputs,
        )?;
        // The module mesh assigns the devices of the target mesh to mesh coordinates, so a program over another
        // topology cannot be placed on it.
        if let Some(mesh) = lowered_module.mesh()
            && !mesh.has_same_topology(options.mesh().logical_mesh())
        {
            return Err(ShardingError::MeshMismatch {
                expected: options.mesh().logical_mesh().clone(),
                actual: mesh.clone(),
            }
            .into());
        }
        let signature = lowered_module.signature();
        for (mapping, donation) in
            signature.input_mapping()[lowered_capture_count..].iter().zip(donation_flags.iter_mut())
        {
            if mapping.is_none() {
                *donation = false;
            }
        }
        let expected_argument_shardings = signature.physical_input_shardings(logical_argument_shardings.as_slice());
        let embeds_kernels = emitted_regions(program.entry_region_ref(), target_platform).any(|region| {
            region.instructions().iter().any(|instruction| match instruction.operation() {
                Operation::CustomCall(operation) | Operation::Array(ArrayOperation::CustomCall(operation)) => {
                    operation.attributes().iter().any(|(name, _)| name == "ryft.kernel.schema")
                }
                _ => false,
            })
        });
        let kernel_execution_facts = if embeds_kernels {
            Some(crate::kernels::KernelExecutionFacts::from_target(target, options.mesh())?.configuration_key()?)
        } else {
            None
        };
        Ok(ProgramLowering::new(
            lowered_module,
            compilation_options,
            effective_input_types.into(),
            output_types.into(),
            output_count,
            reference_states.into(),
            donation_flags.into(),
            lowered_capture_count,
            session_constants.into(),
            expected_argument_shardings.into(),
            output_shardings.into(),
            open_outputs.into(),
            committed_outputs.into(),
            output_device_assignment,
            options.mesh().clone(),
            kernel_execution_facts,
        ))
    }

    /// Compiles a lowering through this session's client after resolving its runtime requirements, then derives the
    /// reusable execution plan.
    pub(crate) fn compile_program(&self, program: &LoweredProgram) -> Result<CompiledProgram, Error> {
        self.ensure_runtime_requirements(program.runtime_requirements())?;
        let pjrt_program = PjrtProgram::Mlir { bytecode: program.stable_hlo().as_bytes().to_vec() };
        let compilation_start = Instant::now();
        let executable = self.client().compile(&pjrt_program, program.compilation_options())?;
        let compilation_duration = compilation_start.elapsed();
        let output_shardings = resolve_output_shardings(&executable, program)?;
        let plan = ExecutionPlan::new(
            &executable,
            program.input_types(),
            program.all_output_types(),
            &output_shardings,
            program.signature(),
            program.donation_flags(),
            program.capture_count(),
            program.mesh(),
        )?;
        Ok(CompiledProgram::new(program, executable, output_shardings, plan, compilation_duration))
    }

    /// Validates ownership and the complete invocation contract before replacing a compiled executable.
    pub(crate) fn validate_replacement(
        &self,
        current: &CompiledProgram,
        replacement: &CompiledProgram,
    ) -> Result<(), Error> {
        self.validate_program_owner(current)?;
        self.validate_program_owner(replacement)?;
        current.invocation().validate_replacement(replacement.invocation())
    }
}

impl CompilationCacheDomain for Session {
    type CacheKey = CompilationKey;

    #[inline]
    fn compilation_key(&self, program: &Self::LoweredProgram) -> Result<Self::CacheKey, Self::Error> {
        program.compilation_key()
    }

    #[inline]
    fn compile_uncached(&self, program: &Self::LoweredProgram) -> Result<Self::CompiledProgram, Self::Error> {
        self.compile_program(program)
    }

    fn reuse_cached_program(
        &self,
        program: &Self::LoweredProgram,
        compiled: Arc<Self::CompiledProgram>,
    ) -> Result<Arc<Self::CompiledProgram>, Self::Error> {
        // Fresh compilations copy this metadata verbatim, so this rejects only reused programs whose invocation
        // contract does not match the requesting lowering (e.g., a corrupted or colliding persistent or exchanged
        // artifact). The stateful transaction trusts this metadata, so a mismatch must be caught before the first
        // call.
        match program.invocation().incompatible_field(compiled.invocation()) {
            Some(field) => Err(Error::Compilation(CompilationError::IncompatibleExecutable {
                reason: format!("cached executable has incompatible {field}"),
            })),
            None => Ok(CompiledProgram::rebound_to(compiled, program)),
        }
    }

    #[inline]
    fn persistent_cache_key(&self, key: &Self::CacheKey) -> Option<Vec<u8>> {
        Some(key.canonical_bytes().to_vec())
    }

    /// Returns the lowest index of a process that owns a device of the mesh of `program`, which is the exchange leader
    /// of its compilation, because exactly the processes that own those devices compile it. Every one of them computes
    /// the same index from the mesh, which they all share, and a program whose mesh includes a device of process zero
    /// is led by process zero, as every program was before.
    #[inline]
    fn artifact_exchange_leader(&self, program: &Self::LoweredProgram) -> usize {
        program.mesh().devices().iter().map(Device::process_index).min().unwrap_or(0)
    }

    #[inline]
    fn serialize_program(&self, program: &Self::CompiledProgram) -> Result<Option<Vec<u8>>, Self::Error> {
        program.serialize()
    }

    #[inline]
    fn deserialize_program(&self, bytes: &[u8]) -> Result<Option<Self::CompiledProgram>, Self::Error> {
        self.deserialize_compiled_program(bytes)
    }
}

/// Returns `sharding`, the placement of an argument of an executable that comes from a runtime array or a capture
/// rather than from an input sharding override, without its unreduced axes that are auto mesh axes. Types do not record
/// a pending reduction over auto mesh axes (refer to [`ArrayType::with_sharding`]), so neither the argument nor a value
/// that the program computes from it can carry one, and Shardy rejects a program that drops it without a reshard (e.g.,
/// an eager `sin` of such an argument, which eager dispatch places as its runtime array is). The executable therefore
/// receives such an argument reduced, and argument preparation rejects an argument whose runtime sharding still has the
/// reduction pending (refer to `validate_input_reduction`), for compiled functions, `jit` dispatchers, and eager
/// operations alike. An input sharding override states the placement with which its argument arrives, which the
/// boundary of the executable keeps, so that a program can complete the reduction (e.g., the reshard that
/// [`Array::to_placement`](crate::Array::to_placement) compiles from an override with a pending reduction to one
/// without it), and it is never passed here. A sharding without unreduced axes, which is the common case, is returned
/// as it is.
fn receivable_argument_sharding(sharding: Sharding) -> Result<Sharding, Error> {
    if sharding.unreduced_axes().is_empty() {
        return Ok(sharding);
    }
    Ok(sharding.without_auto_reduction_axes())
}

/// Admits `sharding`, the placement of the runtime array of an array that a function captures (i.e., an explicit
/// capture or the array of a session constant), as the input at `index` of its executable, whose first `capture_count`
/// inputs are captures. This is the rule by which an execution admits an argument (refer to
/// `validate_capture_reduction`), applied to every capture that a function uses before it is placed on the mesh of the
/// function, whatever the mesh of the function, so that a capture with a pending reduction over auto mesh axes is
/// rejected with the error with which an execution rejects such an argument, rather than reduced by the placement that
/// moves it onto another topology.
///
/// # Errors
///
/// Returns [`ExecutionError::UnreducedAutoAxisCapture`] for such a capture.
pub(crate) fn admit_capture(sharding: &Sharding, index: usize, capture_count: usize) -> Result<(), Error> {
    let expected = receivable_argument_sharding(sharding.clone())?;
    validate_capture_reduction(sharding, &expected, index, capture_count).map_err(Error::Execution)
}

/// Returns the sharding with which the executable of a function compiled for `mesh` receives an array that the function
/// captures, an explicit capture or an array that its program imports (i.e., a session constant) alike, whose type is
/// `array_type` and whose runtime array is placed as `sharding`. A capture never decides where the function runs, and
/// the executable receives it on `mesh` as it receives a public argument. A capture over a logical mesh of the topology
/// of `mesh` is received as it is placed, including over the auto mesh axes that its type does not record, so that an
/// execution only moves it onto the devices of `mesh` and never copies a capture that is already placed there. A
/// capture over a logical mesh of another topology is received replicated over `mesh` when its runtime array stores
/// what its type records as replicated, which includes a capture that is split along auto mesh axes only, which types
/// never record, and the dimensions that its type leaves unconstrained, which a runtime array stores replicated (refer
/// to `with_unconstrained_dimensions_replicated`). This is how JAX runs a `jit` function on the devices of its
/// arguments whatever the devices of the arrays that it closes over. `description` names the capture in the error
/// (e.g., "capture #0").
///
/// Only the captures that survive the pruning of the staged program are resolved, so that a capture that the
/// function does not use places nothing and fails nothing. A capture over a logical mesh of another topology that the
/// function receives replicated is traced with the type that [`captured_type`] gives it. For a session constant,
/// lowering binds its hidden argument to the placed array before resolving that argument's sharding.
///
/// # Errors
///
/// Returns [`ProgramError::InvalidArgument`] for a capture over a logical mesh of another topology whose type records a
/// placement other than replication (i.e., a split, a pending reduction, or manual variation over that mesh), which
/// the function computes with and which no placement over `mesh` can keep.
pub(crate) fn capture_sharding(
    description: &str,
    array_type: &ArrayType,
    sharding: &Sharding,
    mesh: &LogicalMesh,
) -> Result<Sharding, Error> {
    if sharding.mesh().has_same_topology(mesh) {
        return Ok(sharding.clone());
    }
    let stored = array_type
        .sharding()
        .map(ShardingLoweringExtension::with_unconstrained_dimensions_replicated)
        .transpose()?;
    if !stored.as_ref().is_none_or(Sharding::is_replicated) {
        return Err(Error::from(ProgramError::InvalidArgument {
            message: format!(
                "{description} has the type `{array_type}` over a mesh of another topology than the mesh `{}` that \
                 the function is compiled for, whose placement it cannot keep",
                mesh,
            ),
        }));
    }
    Ok(Sharding::replicated(mesh.clone(), array_type.rank()))
}

/// Returns the type with which a function traced under [`CapturePlacement::FollowCall`] receives, and is traced with,
/// an array whose type is `array_type`, or [`None`] when the function must keep the placement that the type records. A
/// capture follows the call of the function. A type without sharding, including that of a capture on the anonymous mesh
/// of one device, stays unsharded. When its type records only replication (refer to
/// `ShardingLoweringExtension::with_unconstrained_dimensions_replicated`), the function receives it without sharding,
/// as it receives a public argument whose type declares none. The types of the values that it computes from the capture
/// then name no mesh of the capture. A call of a function that names no devices therefore runs where its caller runs
/// whatever the devices of its captures, as JAX computes `jit(lambda x: x + c)` on the devices of `x` whether `c` is
/// uncommitted or committed elsewhere. A capture whose type records more (i.e., a split, a pending reduction, or manual
/// variation), or whose stored placement cannot be computed, keeps its own type, and lowering checks that the function
/// can receive it on the mesh of its options (refer to [`capture_sharding`]). This is the single rule for the type of a
/// captured array inside a compiled function, which every route by which an array enters the trace of a compiled
/// function shares (refer to `Session::capture_constant` and [`imported_constant`]). Lowering binds a session constant
/// to its placed argument (refer to [`placed_constant`]). An uncommitted public argument follows the call as a capture
/// does, so a `jit` dispatcher specializes for such an argument with the type that this function returns as well (refer
/// to [`argument_type`]).
pub(crate) fn captured_type(array_type: &ArrayType) -> Option<ArrayType> {
    let stored = array_type
        .sharding()
        .map(ShardingLoweringExtension::with_unconstrained_dimensions_replicated)
        .transpose()
        .ok()?;
    if !stored.as_ref().is_none_or(Sharding::is_replicated) {
        return None;
    }
    array_type.clone().with_sharding(None).ok()
}

/// Returns the type for which a `jit` dispatcher specializes its function for `array`, an argument of one of its
/// executions or eager calls whose effective type is `array_type` (i.e., the type of `array` or the staged type that
/// the dispatcher declares for it), when `array` is relocatable, and [`None`] when it is not, in which case the
/// dispatcher specializes for `array_type` and the placement of `array`. An uncommitted argument whose type records
/// nothing beyond replication (e.g., an upload without a destination, which lives on the canonical mesh of its device)
/// is relocatable: an execution moves it onto the devices of the function, as it moves an
/// uncommitted argument of a function compiled for a type without a sharding. The function is therefore specialized
/// for its type without a sharding, by the rule that types a capture (refer to [`captured_type`]), rather than for the
/// logical mesh where it happens to live, which another mesh of the same devices would reject. A committed argument and
/// one whose type records more keep their types, so the mesh of a function still rejects them when their logical
/// meshes differ. The dispatch signature uses this rule for both executions and eager calls, so they trace the function
/// alike. The rule also determines which arguments an eager call of a dispatcher
/// leaves only an argument that is not relocatable where it is (refer to `CallEagerExtension::eager_input_placement`).
/// A relocatable argument has no placement over auto mesh axes for the function to specialize on (e.g., an uncommitted
/// split along an auto axis of one device), because an execution moves it.
pub(crate) fn argument_type(array: &Array, array_type: &ArrayType) -> Option<ArrayType> {
    if array.is_committed() { None } else { captured_type(array_type) }
}

/// Returns the constant that a trace whose capture placement is `capture_placement` (refer to
/// [`CapturePlacementContext::capture_placement`](ryft_core::CapturePlacementContext::capture_placement)) stages for
/// `value`, a runtime value that it imports. An array becomes a [`Constant::Session`] declared with the type the function
/// receives it under [`CapturePlacement::FollowCall`] (refer to [`captured_type`]), or with its own type under
/// [`CapturePlacement::Preserve`] (e.g., in a plain trace or an eager region) or when its type records more than
/// replication, in which case lowering checks it if the function uses it. Every other value becomes its ordinary
/// constant.
///
/// # Errors
///
/// Returns the error of turning `value` into a constant.
pub(crate) fn imported_constant(value: Value, capture_placement: CapturePlacement) -> Result<Constant, ProgramError> {
    let constant = value.into_constant()?;
    let (Constant::Session(retained), CapturePlacement::FollowCall) = (&constant, capture_placement) else {
        return Ok(constant);
    };
    match captured_type(&retained.r#type()) {
        Some(captured_type) => Ok(Constant::Session(retained.with_declared_type(captured_type))),
        None => Ok(constant),
    }
}

/// Returns `constant`, an array that a function captures (i.e., a session constant of its program), placed where the
/// executable of the function that is compiled for `mesh` receives it, or [`None`] when the executable cannot receive
/// it on `mesh` (e.g., a capture whose type records a split over a mesh of another topology, or one with a pending
/// reduction over auto mesh axes, which lowering and admission reject). Lowering and eager dispatch use this placement
/// rule. A capture with the same logical mesh topology as `mesh` keeps its runtime sharding on the
/// devices of `mesh`, so that it adds no gather, and any other capture is placed as [`capture_sharding`] receives it
/// (i.e., replicated over `mesh`). Each placement is computed once for each captured array and destination, which every
/// retention of the array and every program that captures it share, and lives until the array and every retention of it
/// are gone (refer to [`RetainedArray::placed`]), so no execution, eager call, or other program that captures the array
/// moves the capture again. A placement that cannot communicate (refer to [`RetainedArray::placement_communicates`])
/// starts from a copy on the same devices whose work has already completed when there is one (e.g., for meshes that
/// differ only in their axis names), so that such destinations share its buffers, but only from a copy with the memory
/// and the declared layout of the capture, so that the result is what a placement of the capture itself gives, whatever
/// placements were made before (refer to [`RetainedArray::completed_placement_on`]). The placement moves the capture
/// with device work, except that a capture whose devices this process chose on its own is read back whole and uploaded,
/// and a shard whose storage the PJRT plugin cannot change on its device is copied through host memory on its own
/// (refer to [`Array::reshard_on_devices`]). It never takes the other host round trip to which [`Session::place`] can
/// fall back, which only a process that owns every destination device can take, so a placement that only that round
/// trip can make (e.g., of uneven column shards that a compiled reshard cannot produce) is an error for a capture. A
/// capture that already lives on the devices of `mesh` with that placement is not copied, but it is still handed out as
/// its placement onto the devices that the function names (refer to [`RetainedArray::on_named_devices`]), which shares
/// its buffers and keeps nothing alive beyond the capture itself. Every process therefore holds the same facts for the
/// capture of a function (i.e., committed to its devices and not locally assigned), whether a process placed it or
/// found it there (e.g., a capture without a target on a function on the first device of the first process, which the
/// first process finds on that device and every other process places there). The exception is such a capture that lacks
/// the buffer of a shard of this process that has no elements or that another local replica of the same contribution
/// holds (e.g., a partial import, refer to [`RetainedArray::can_fill_missing_local_buffers`]), which is placed onto its
/// own mesh and sharding once, as any other destination is, so that the copy of that replica is made once rather than
/// by every execution. Each process decides this from its own buffers, but that placement moves every shard to its own
/// device and copies only between devices of one process, so a process that places it submits no transfer that another
/// process pairs with, whichever replicas each process holds, and a contribution without any local replica is left as
/// it is.
///
/// Only session constants take this route, which are the captures whose types record nothing beyond replication. A
/// capture whose type records more (e.g., a split over explicit mesh axes) stays an ordinary capture of the
/// executable, which receives it as it is, so an execution on other devices than those of that capture moves it again
/// every time, and one on its devices moves nothing.
///
/// # Errors
///
/// Returns an error when `constant` retains an array of another client than that of `session`, when the session does
/// not recognize the devices or process assignments of the source or destination mesh, or when moving `constant` fails.
pub(crate) fn placed_constant(
    session: &Session,
    constant: &RetainedArray,
    mesh: &DeviceMesh,
) -> Result<Option<RetainedArray>, Error> {
    // A request of another client submits nothing, so it is rejected before it can consume the one placement attempt
    // of this array (refer to `RetainedArray::placed`) that its own client still needs.
    constant.validate_client(session.client())?;
    let follows_mesh = constant
        .r#type()
        .sharding()
        .is_none_or(|sharding| sharding.mesh().has_same_topology(mesh.logical_mesh()));
    // A capture that the function cannot admit (refer to `admit_capture`) is left as it is, so that its admission
    // rejects it rather than a placement that completes its pending reduction.
    let pending_auto_reduction = receivable_argument_sharding(constant.sharding().clone())? != *constant.sharding();
    if !follows_mesh || pending_auto_reduction {
        return Ok(None);
    }
    // The requesting session must recognize the source before it can reuse or record a shared placement.
    validate_mesh_devices(session, constant.mesh())?;
    let (target, sharding) = if constant.sharding().mesh().has_same_topology(mesh.logical_mesh()) {
        // A capture on the devices of `mesh` is placed onto its own mesh and sharding only to fill its missing local
        // replicas.
        if constant.mesh().devices() == mesh.devices() && !constant.can_fill_missing_local_buffers()? {
            return Ok(Some(constant.on_named_devices()));
        }
        (DeviceMesh::new(constant.sharding().mesh().clone(), mesh.devices().to_vec())?, constant.sharding().clone())
    } else {
        let declared_type = constant.r#type();
        let Ok(sharding) = capture_sharding("a capture", &declared_type, constant.sharding(), mesh.logical_mesh())
        else {
            return Ok(None);
        };
        (mesh.clone(), sharding)
    };
    // A destination that assigns a device to another process than the one that owns it is rejected on every process
    // before it can consume a placement attempt, as a direct placement rejects it (refer to `place_within_client`).
    validate_mesh_devices(session, &target)?;
    // The placed copy lives on devices that the mesh of the function names for every process, so it is committed to
    // them and not locally assigned, as an array that a placement onto named devices gives is, whatever the commitment
    // and the local assignment of the captured array are. The source constant stays in the IR and keeps its own
    // commitment for eager dispatch and for the interpretation of a program that holds it.
    //
    // When the placement cannot communicate, a copy on the same devices (e.g., for a mesh that differs only in its axis
    // names) whose work has already completed without an error is placed instead of the capture, which shares its
    // buffers rather than copying the capture again. It is chosen only once the exact key missed (i.e., inside the
    // attempt), and never waited for, so a request that finds its own placement never waits for other work. A copy is a
    // source only when it keeps the memory and the declared layout of the capture (refer to
    // `RetainedArray::completed_placement_on`), so that the result has the type that a placement of the capture gives,
    // whichever placements the index already holds. If the copy spans processes, it is reused only when it and the
    // destination are both replicated. Every destination shard then comes from the shard of the copy on the same
    // device. A process whose copy is missing, pending, or failed places from the capture instead. The shared planner
    // must classify the placement from the capture as local before processes can choose their own sources, because a
    // copy can exist on one process and have failed on another.
    let first_process = target.devices()[0].process_index();
    let one_process = target.devices().iter().all(|device| device.process_index() == first_process);
    let placed = constant.placed(&target, &sharding, |source| {
        let replicated = |copy: &RetainedArray| copy.sharding().is_replicated() && sharding.is_replicated();
        let reusable = |copy: &RetainedArray| one_process || replicated(copy);
        let copy = if source.placement_communicates(&target, &sharding) {
            None
        } else {
            source.completed_placement_on(target.devices(), reusable)
        };
        let source = copy.as_ref().unwrap_or(&source);
        let placed = source.to_array(session)?.reshard_on_devices(session, &target, &sharding, false)?;
        Ok::<_, Error>(placed.with_committed(true).with_locally_assigned(false))
    })?;
    Ok(Some(placed))
}

/// Applies the input sharding overrides of `options` (i.e., [`Options::input_shardings`]) through the `input_type`
/// projection of each input. The overrides must be applied before tracing, because tracers carry their input types
/// and sharding-aware operations read them. Retained dispatch applies them through
/// [`CompilationDomain::dispatch_signature`], the staging entry
/// points of this crate apply them through [`with_input_shardings`], and lowering rejects staged functions whose input
/// types do not carry them. External reference inputs are rejected when overrides are present, because holder state
/// derives its effective sharding from the compiled reference ABI.
pub(super) fn apply_input_shardings<T>(
    inputs: &mut [T],
    options: &Options,
    input_type: impl Fn(&mut T) -> &mut Type,
) -> Result<(), Error> {
    let Some(input_shardings) = options.input_shardings() else {
        return Ok(());
    };
    if inputs.iter_mut().any(|input| input_type(input).is_reference()) {
        return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
            reason: "external reference inputs do not support explicit input shardings".to_string(),
        }));
    }
    apply_signature_shardings(inputs.iter_mut().map(input_type), Some(input_shardings), "input")
}

/// Applies [`Options::input_shardings`] to `input_types`, preserving their parameter structure. Returns the input
/// types unchanged when no overrides are configured.
///
/// Call this before [`ryft_core::StagedFunction::stage`] when using the generic core staging API with XLA sharding
/// overrides. Tracing must see the effective input shardings, and lowering rejects staged input types that disagree
/// with the overrides. The staging entry points in [`crate::compilation`] already apply this normalization.
///
/// # Errors
///
/// Returns an error if the number of overrides differs from the number of flat inputs, an override cannot be applied
/// to an input type, or overrides are requested for external reference inputs. The reference ABI determines the
/// effective sharding of external state, so it does not accept input sharding overrides.
///
/// # Examples
///
/// ```
/// use ryft_core::{
///     ArrayType, DataType, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension,
/// };
/// use ryft_xla::compilation::with_input_shardings;
/// use ryft_xla::{Options, Type};
///
/// let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit)?])?;
/// let mesh = DeviceMesh::new(logical_mesh, vec![Device::new(0, 0), Device::new(1, 0)])?;
/// let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])])?;
/// let options = Options::new(mesh).with_input_shardings(vec![sharding.clone()]);
/// let input_type = ArrayType::new_static(DataType::F32, [4]);
/// let input_types = (Type::Array(input_type.clone()),);
/// let input_types = with_input_shardings(input_types, &options)?;
/// assert_eq!(input_types, (Type::Array(input_type.with_sharding(sharding)?),));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn with_input_shardings<In: Parameterized<Type>>(input_types: In, options: &Options) -> Result<In, Error> {
    if options.input_shardings().is_none() {
        return Ok(input_types);
    }
    let input_structure = input_types.parameter_structure();
    let mut input_types = input_types.into_parameters().collect::<Vec<_>>();
    apply_input_shardings(&mut input_types, options, |r#type| r#type)?;
    Ok(In::from_parameters(input_structure, input_types).map_err(ProgramError::from)?)
}

/// Overlays the selected multi-device execution mode on the base [`CompilationOptions`] template. Static programs
/// use Shardy SPMD partitioning, while bounded dynamic programs use one fully replicated executable replica per
/// device because Shardy does not yet accept dynamic tensors.
///
/// The executable is pinned to `devices`, which are the devices of the mesh that the program is compiled for, in
/// mesh order, as JAX pins its executables to the devices of their shardings. Without an explicit assignment, PJRT
/// assigns the first devices of the client, irrespective of the devices that the mesh names. Portable executables
/// are the exception, because they are compiled for no particular device and PJRT rejects an assignment for them.
pub(crate) fn jit_compilation_options(
    base: &CompilationOptions,
    devices: &[Device],
    use_shardy_partitioner: bool,
    feedback_directed_profile: Option<&FeedbackDirectedProfile>,
) -> CompilationOptions {
    let mut options = base.clone();
    let executable_options = options.executable_build_options.get_or_insert_with(ExecutableCompilationOptions::default);
    if executable_options.device_ordinal == 0 {
        executable_options.device_ordinal = -1;
    }
    let device_count = devices.len();
    executable_options.replica_count = if use_shardy_partitioner { 1 } else { device_count as i64 };
    executable_options.partition_count = if use_shardy_partitioner { device_count as i64 } else { 1 };
    executable_options.use_spmd_partitioning = use_shardy_partitioner;
    executable_options.use_shardy_partitioner = use_shardy_partitioner;
    if !options.compile_portable_executable {
        // Each computation lists the device of every one of its replicas. Shardy runs one replica of one computation
        // per device, while the replica path runs one computation with one replica per device.
        let device_ids = devices.iter().map(|device| device.id() as i64);
        let computation_devices = if use_shardy_partitioner {
            device_ids.map(|id| ComputationDeviceAssignment { replica_device_ids: vec![id] }).collect()
        } else {
            vec![ComputationDeviceAssignment { replica_device_ids: device_ids.collect() }]
        };
        executable_options.device_assignment = Some(DeviceAssignment {
            replica_count: executable_options.replica_count as i32,
            computation_count: executable_options.partition_count as i32,
            computation_devices,
        });
    }
    if let Some(profile) = feedback_directed_profile {
        executable_options.fdo_profile = profile.bytes().to_vec();
        options.profile_version = profile.version();
    }
    options
}

/// Returns whether every execution of a program on `mesh` commits its public outputs whatever the commitment of its
/// arguments, where `requests_placement` says whether the program requests a placement explicitly anywhere, including
/// one that only moves a value to another memory (refer to `requests_placement`), and `function_mesh` says whether
/// `mesh` is the mesh that the function of the program was compiled for and runs on, rather than the devices of the
/// arguments of an eager application or of an eager operation. This is the case when the program requests a placement,
/// as JAX commits every result of a `jit` function with any explicit sharding, when it runs on more than one device, as
/// JAX commits every result of a computation on more than one device, and when it runs on the named mesh of its
/// function (refer to [`LogicalMesh::is_named`]), as JAX commits every result of a `jit` function under
/// `set_mesh`, on one device too. A result of a function on a named mesh of one device is therefore committed to that
/// mesh, so combining it with a value committed to other devices is an error, as under `set_mesh`. On an anonymous
/// mesh, results are committed only by the other rules or by a committed public argument.
///
/// This is the one owner of the rule. Compiled functions record it (refer to [`CompiledProgram::committed_outputs`]),
/// so that `.executable().execute` commits by it, and the placement decision of an eager application of a call reads it
/// for the mesh on which the application runs (refer to `CallEagerExtension::eager_decision`), so that `.call`, the
/// interpretation of a call, and the calls that eager JVP, eager VJP, and `vmap` derive from a call commit as an
/// execution does. A call nested in a compiled caller does not read it, because the executable of its caller decides.
pub(crate) fn commits_public_outputs(mesh: &DeviceMesh, function_mesh: bool, requests_placement: bool) -> bool {
    requests_placement || mesh.devices().len() > 1 || function_mesh && mesh.logical_mesh().is_named()
}

/// Returns whether a function whose program is `program` and whose options are `options` requests the placement of a
/// value explicitly, through an input or output sharding override or anywhere in its program (refer to
/// [`ProgramLoweringExtension::has_placement_requests`]), with or without the requests that only move a value to
/// another memory, as `includes_memory_requests` selects. Every request commits every public output of the function,
/// so commitment counts them all. A request that names devices (i.e., every request except one that only moves a
/// value to another memory) also makes the function name the devices of the mesh of `options`, which every execution
/// and every call of it runs on whatever its inputs. Any other function has a program that is agnostic of devices, and
/// the inputs of a call decide where the call runs, as JAX runs a `jit` function whose body only moves a value to
/// another memory where its arguments are. The arrays that the function captures are not among them, whether they are
/// committed or not, as JAX ignores where the arrays that a `jit` function closes over live (refer to
/// `CallOperation`).
fn requests_placement(
    program: &FlatProgram<Session>,
    options: &Options,
    target_platform: Option<&str>,
    includes_memory_requests: bool,
) -> bool {
    let overrides = options
        .input_shardings()
        .into_iter()
        .flatten()
        .chain(options.output_shardings().into_iter().flatten());
    states_boundary_placement(overrides) || program.has_placement_requests(target_platform, includes_memory_requests)
}

/// Validates that the staged `output_types` carry the output sharding overrides of `options` (i.e.,
/// [`Options::output_shardings`]). The staging entry points of this crate stage the overrides after tracing (i.e., a
/// `reshard` over explicit mesh axes and a `constrain_sharding` over auto mesh axes), so that calls of a staged
/// function return the output types that its executions return. Types never record placement over auto mesh axes, as
/// [`ArrayType::with_sharding`] documents, so only the placement over the other mesh axes is compared, and lowering
/// then places the outputs over auto mesh axes as the overrides request. Outputs that are not arrays are left for the
/// later array-only conversions to reject.
fn validate_output_shardings(output_types: &[Type], options: &Options) -> Result<(), Error> {
    let Some(output_shardings) = options.output_shardings() else {
        return Ok(());
    };
    if output_shardings.len() != output_types.len() {
        return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
            kind: "output",
            expected: output_types.len(),
            actual: output_shardings.len(),
        }));
    }
    for (index, (r#type, sharding)) in output_types.iter().zip(output_shardings).enumerate() {
        let Type::Array(array_type) = r#type else {
            continue;
        };
        if !array_type
            .clone()
            .with_sharding(sharding.without_auto_axes())
            .is_ok_and(|overridden| overridden == *array_type)
        {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: format!(
                    "staged output {index} has type `{array_type}`, which does not carry the sharding `{sharding}` \
                     requested by `output_shardings`; reshard the outputs to their overrides before staging them",
                ),
            }));
        }
    }
    Ok(())
}

/// Rejects every output of reference type of a capture-lifted program before lowering it into an executable. The
/// stateful XLA ABI opens and commits external input references and has no protocol for turning an array that the
/// backend created into a host reference with persistent identity, nor for handing an entering reference back through
/// an output. The retained reference analysis of the capture-lifted program distinguishes the escaping local
/// allocation from the forwarded external reference. References returned by nested calls inside a program never cross
/// that ABI, so only the outputs of the lowered program itself are checked.
fn validate_reference_outputs(
    program: &FlatProgram<Session>,
    capture_count: usize,
    output_types: &[Type],
) -> Result<(), Error> {
    let Some(output_index) = output_types.iter().position(|r#type| r#type.is_reference()) else {
        return Ok(());
    };
    let analysis = program.reference_analysis(capture_count).map_err(ProgramError::from)?;
    let root = analysis.output_roots().get(output_index).copied().flatten();
    let reason = match root.map(|root| analysis.external_source(root)) {
        Some(Some(source)) => format!(
            "output {output_index} forwards the external reference bound at {source}, and the stateful XLA ABI does \
             not return forwarded references; read the reference after the call instead",
        ),
        Some(None) => format!(
            "output {output_index} is a reference allocated inside the program and cannot be returned through the \
             stateful XLA ABI, which has no protocol to turn a backend array into a host reference",
        ),
        None => format!(
            "output {output_index} is a reference whose root the reference analysis attributes to neither an input \
             nor an allocation, and the stateful XLA ABI cannot return it",
        ),
    };
    Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
}

/// Validates backend output arity and refinement of the declared logical array types.
fn validate_output_types(declared: &[Type], actual: &[ArrayType]) -> Result<(), Error> {
    if declared.len() != actual.len() {
        return Err(ProgramError::InvalidOutputCount { expected: declared.len(), actual: actual.len() }.into());
    }
    for (declared, actual) in declared.iter().zip(actual) {
        let declared = <&ArrayType>::try_from(declared).map_err(ProgramError::from)?;
        if !declared.is_refined_by(actual) {
            return Err(ProgramError::InvalidArgument {
                message: format!("backend output type `{actual}` does not refine declared type `{declared}`"),
            }
            .into());
        }
    }
    Ok(())
}

/// Applies an optional per-leaf sharding override to a public array IR signature, giving each overridden type the
/// override without its auto mesh axes, which types never record, as [`ArrayType::with_sharding`] documents. The
/// runtime placement that an override requests over auto mesh axes is applied by lowering instead. When overrides are
/// provided, every overridden leaf must be an array, because first-class dimensions are internal SSA values and
/// references must be discharged into explicit array state before either kind can cross the PJRT ABI. Note that without
/// overrides the signature passes through unchecked here. Boundary leaves that are not arrays are then rejected by the
/// later array-only conversions on the compile path.
fn apply_signature_shardings<'t>(
    types: impl ExactSizeIterator<Item = &'t mut Type>,
    shardings: Option<&[Sharding]>,
    kind: &'static str,
) -> Result<(), Error> {
    let Some(shardings) = shardings else {
        return Ok(());
    };
    if shardings.len() != types.len() {
        return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
            kind,
            expected: types.len(),
            actual: shardings.len(),
        }));
    }
    for (r#type, sharding) in types.zip(shardings) {
        *r#type = match r#type {
            Type::Array(array_type) => array_type.clone().with_sharding(sharding.without_auto_axes())?.into(),
            Type::Dimension(_) | Type::Reference(_) => {
                return Err(ProgramError::from(TypeError::invalid(format!(
                    "cannot apply an array sharding to type `{type}`",
                )))
                .into());
            }
        };
    }
    Ok(())
}

/// Returns the runtime placement of an output by restoring unconstrained dimensions of its declared or requested
/// sharding. The executable annotation closes those dimensions when it must fix their placement. The runtime sharding
/// keeps the original metadata and represents an unconstrained dimension with replicated buffers.
fn runtime_output_sharding(annotation: &Sharding, declaration: Option<&Sharding>) -> Result<Sharding, Error> {
    let Some(declaration) = declaration else {
        return Ok(annotation.clone());
    };
    if !declaration.dimensions().contains(&ShardingDimension::Unconstrained) {
        return Ok(annotation.clone());
    }
    let dimensions = annotation
        .dimensions()
        .iter()
        .zip(declaration.dimensions())
        .map(|(dimension, declared)| match declared {
            ShardingDimension::Unconstrained => ShardingDimension::Unconstrained,
            _ => dimension.clone(),
        })
        .collect();
    Ok(annotation.with_dimensions(dimensions)?)
}

/// Validates that `sharding`, the placement of the argument or result `index` of kind `kind` (e.g., `"input"`) of an
/// executable, splits every static extent of `array_type` evenly. The SPMD partitioner splits a dimension into shards
/// of one size and pads the last ones, so an uneven split (e.g., five elements over two devices) makes the executable
/// expect or return buffers other than the exact shards (i.e., `[3]` and `[2]`) that the runtime placement describes,
/// which launching the executable or reconstructing its outputs would reject. A dimension is split into as many shards
/// as the product of the sizes of the mesh axes that it is sharded along. Dynamic extents are not checked here, and
/// neither are values that the executable does not materialize (refer to [`ExecutableSignature::materializes`]),
/// because they have no buffers that the executable could expect or return (e.g., a `Zero[5]` split into shards of
/// lengths 3 and 2).
///
/// # Errors
///
/// Returns [`LoweringError::UnevenSharding`] for the first dimension whose static extent that number does not divide.
fn validate_even_sharding(
    kind: &'static str,
    index: usize,
    array_type: &ArrayType,
    sharding: &Sharding,
) -> Result<(), Error> {
    if !ExecutableSignature::materializes(array_type) {
        return Ok(());
    }
    for (dimension, (extent, sharding_dimension)) in
        array_type.shape().dimensions().iter().zip(sharding.dimensions()).enumerate()
    {
        let (Dimension::Static(extent), ShardingDimension::Sharded(axes)) = (extent, sharding_dimension) else {
            continue;
        };
        let shard_count = axes.iter().filter_map(|axis| sharding.mesh().axis_size(axis)).product::<usize>();
        if extent % shard_count != 0 {
            return Err(Error::Lowering(LoweringError::UnevenSharding {
                kind,
                index,
                dimension,
                extent: *extent,
                shard_count,
            }));
        }
    }
    Ok(())
}

/// Returns whether every array type that occurs in a region that lowering emits for `target_platform` (refer to
/// [`emitted_regions`]) of `program` has a static shape. The walk covers the entry region and its executed nested
/// regions, and therefore every region boundary, constant, and instruction result that lowering emits. Regions that
/// lowering never reaches (e.g., the rule regions of composites, the transpose region of a `linear_call`, or a body that
/// a backend implementation of a named composite replaces for the target) are skipped, so that they cannot change how
/// the rest of the program is compiled. First-class atoms of dimension type are ignored because they lower to ordinary
/// scalars. Only dynamically shaped *array* types make the lowered module unacceptable to the Shardy partitioner.
fn has_only_static_array_types(program: &FlatProgram<Session>, target_platform: Option<&str>) -> bool {
    emitted_regions(program.entry_region_ref(), target_platform).all(|region| {
        region.atoms().iter().all(|atom| match atom.r#type().as_ref() {
            Type::Array(array_type) => array_type.static_shape().is_some(),
            Type::Dimension(_) => true,
            // The predicate answers its own question rather than relying on the reference rejection during lowering
            // firing first. A dynamically shaped referent counts as dynamic so that partitioner selection and its
            // guards stay correct even for programs whose only dynamic type is a reference referent.
            Type::Reference(reference_type) => reference_type.referent().static_shape().is_some(),
        })
    })
}

/// Returns whether `program` lowers a `shard_map` for `target_platform`, including inside executed nested regions but
/// not inside regions that lowering never reaches (refer to [`emitted_regions`]). The documentation of
/// [`ShardMapOperation`](ryft_core::ShardMapOperation) requires `shard_map` boundaries to be static, so a `shard_map`
/// can still appear in a program that is bounded dynamic elsewhere, which this predicate detects.
fn contains_shard_map(program: &FlatProgram<Session>, target_platform: Option<&str>) -> bool {
    emitted_regions(program.entry_region_ref(), target_platform).any(|region| {
        region
            .instructions()
            .iter()
            .any(|instruction| matches!(instruction.operation(), Operation::ShardMap(_)))
    })
}

/// Validates that every `place` and every `call` that is lowered as part of `program`, including inside executed
/// nested regions, can be compiled for `mesh`, which is the device mesh that `program` is compiled for, with or
/// without the Shardy partitioner, depending on `use_shardy_partitioner`.
///
/// A `call` whose callee boundary has sharding overrides places values on the devices of the mesh that its callee
/// was compiled for (refer to [`CallOperation`]), so it is accepted exactly when that mesh has the devices of `mesh`
/// in the same order and its topology, as JAX rejects a nested `jit` whose shardings name devices other than those of
/// the computation that calls it. Without the Shardy partitioner and on more than one device, every override must also
/// be fully replicated once its unconstrained dimensions are (refer to
/// [`ShardingLoweringExtension::with_unconstrained_dimensions_replicated`]), which is the constraint that lowering
/// emits for it, because replicas cannot enforce a split.
///
/// A compiled program runs on the devices of one mesh, so a `place` inside it can only redistribute a value over those
/// devices and change its storage. Lowering only knows the logical mesh of the module, so the devices are checked
/// before lowering, which also reports a target over another mesh topology as such rather than as a less specific
/// mismatch of the module mesh. JAX treats the devices of a `device_put` target inside `jit` according to the kind of
/// target. A bare device leaves an uncommitted input on the devices that it already occupies, whereas a concrete
/// `NamedSharding` selects its devices for the computation and commits the output, and JAX raises an incompatible
/// devices error when an input is committed to other devices. The devices of a Ryft program are fixed by its mesh
/// before lowering, so a target on other devices is rejected instead. A device target (i.e., a
/// [`PlacementRequest::Device`]) keeps the sharding of its input, so it is accepted exactly when `mesh` consists of
/// that device alone, in which case it lowers to its layout and memory only. Unlike JAX, which leaves a value that a
/// bare device places inside `jit` uncommitted and on the devices that it occupies, a function with such a target
/// commits its outputs, as an explicit device commits an array eagerly. An in-place target (i.e., a
/// [`PlacementRequest::InPlace`]) leaves its input on the devices where it lives, which are those of `mesh`, so it has
/// nothing to validate and lowers to its layout and memory only. Without the Shardy partitioner (i.e., for bounded
/// dynamic programs), the compiler runs one fully replicated replica of the program on each device and ignores sharding
/// constraints, so it would silently ignore a `place` whose target is not replicated.
///
/// A source is not validated here, because the program moves nothing to it. It states where the caller expects the
/// input to live, which the type rule of `place` checks against the input type, and JAX likewise elides the devices of
/// a `device_put` source inside `jit`. So a `place` without a target, which leaves its input where it is and lowers to
/// nothing, compiles whatever devices its source names. Transposition turns the return placement of a `place` into
/// the target of the transposed placement, which is validated as a target when the transposed program is compiled,
/// and the lowering of `place` still rejects a source that names devices inside a manual region where transposition
/// would move the cotangent to it, as an eager `shard_map` does.
///
/// Regions that lowering never reaches (e.g., the rule regions of composites, the transpose region of a `linear_call`,
/// a body that a backend implementation replaces, or the body of a scan whose length admits no iteration) are not
/// checked, because they are not part of the compiled program. Entry points that lower a program without compiling it
/// for a device mesh (e.g., [`ProgramExtension::to_mlir_module`](crate::ProgramExtension::to_mlir_module)) know no devices
/// and do not run it.
///
/// # Errors
///
/// Returns [`LoweringError::PlacementOutsideProgramMesh`] for the first `place` whose target mesh has other devices,
/// the same devices in another order, or another topology than `mesh`, [`LoweringError::DeviceOutsideProgramMesh`] for
/// the first `place` whose device target is not the only device of `mesh`, [`LoweringError::CallOutsideProgramMesh`]
/// for the first `call` whose callee mesh differs from `mesh` in the same ways, and
/// [`LoweringError::UnsupportedOperation`] for the first `place` whose target sharding or `call` whose effective
/// boundary sharding (i.e., an override or where the executable of the callee places an input, refer to
/// [`CallOperation::boundary_input_sharding`]) is not fully replicated (refer to [`Sharding::is_replicated`]), once
/// its unconstrained dimensions are read as replicated (refer to
/// [`ShardingLoweringExtension::with_unconstrained_dimensions_replicated`]), when `use_shardy_partitioner` is not set
/// and `mesh` has more than one device.
fn validate_placements(
    program: &FlatProgram<Session>,
    mesh: &DeviceMesh,
    use_shardy_partitioner: bool,
    target_platform: Option<&str>,
) -> Result<(), Error> {
    // The walk follows the regions that lowering emits for `target_platform`, so it skips the regions of an instruction
    // that lowering omits (refer to `omits_regions`), as it skips dormant regions.
    let entry = program.entry_region_ref();
    let mut visited = vec![false; entry.arena().len()];
    let mut pending = vec![entry.id()];
    while let Some(id) = pending.pop() {
        if std::mem::replace(&mut visited[id.index()], true) {
            continue;
        }
        let region = entry.with_id(id)?;
        for instruction in region.instructions() {
            if !omits_regions(region, instruction, target_platform) {
                pending.extend(instruction.regions().iter().copied().enumerate().filter_map(|(index, id)| {
                    (instruction.operation().region_role(index) == Some(RegionRole::Computation)).then_some(id)
                }));
            }
        }
        for instruction in region.instructions() {
            // A call places values on the devices that its callee was compiled for, so it is valid exactly when those
            // are the devices of `mesh`, and the replicas of a bounded dynamic program cannot enforce an override that
            // splits a value, whose lowered constraint is the annotation of the corresponding callee boundary.
            if let Some(call) = instruction.operation().projected_payload::<CallOperation<Type>>() {
                if let Some(call_mesh) = call.mesh()
                    && (call_mesh.devices() != mesh.devices()
                        || !call_mesh.logical_mesh().has_same_topology(mesh.logical_mesh()))
                {
                    return Err(Error::Lowering(LoweringError::CallOutsideProgramMesh {
                        callee: call_mesh.clone(),
                        mesh: mesh.clone(),
                    }));
                }
                if !use_shardy_partitioner && mesh.devices().len() > 1 {
                    // The effective boundary of an input is its override or otherwise where the executable of the
                    // callee places it (refer to `CallOperation::boundary_input_sharding`), and the lowering of the
                    // call constrains the operand to it as the callee holds it once it has entered, without reduction
                    // state over auto mesh axes (refer to `Sharding::without_auto_reduction_axes`), which is what the
                    // replicas must enforce.
                    let callee = region.with_id(instruction.regions()[0])?;
                    let mut shardings = Vec::new();
                    for (index, (input, declared)) in
                        instruction.inputs().iter().zip(callee.input_types().iter()).enumerate()
                    {
                        if let Type::Array(actual) = region.atoms()[input.index()].r#type().as_ref()
                            && let Some(sharding) = call.boundary_input_sharding(index, actual, declared)
                        {
                            shardings.push(sharding.without_auto_reduction_axes());
                        }
                    }
                    shardings.extend(
                        (0..instruction.outputs().len()).filter_map(|index| call.output_sharding(index).cloned()),
                    );
                    for sharding in &shardings {
                        if !sharding.with_unconstrained_dimensions_replicated()?.is_replicated() {
                            return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                                operation: CALL_OPERATION_NAME.to_string(),
                                reason: "bounded-dynamic multi-device programs currently require fully replicated \
                                         boundary shardings of called functions because Shardy rejects dynamic \
                                         tensors"
                                    .to_string(),
                            }));
                        }
                    }
                }
                continue;
            }
            let Some(place) = instruction.operation().projected_payload::<PlaceOperation>() else {
                continue;
            };
            // Only the target is validated, because it is the only request that the program executes. A source states
            // where the caller expects the input to live, which the type rule checks against the input type, and the
            // forward program moves nothing to it, so its devices and its sharding are not checked here, as JAX elides
            // the source of a `device_put` inside `jit`. Transposition turns the return placement into the target of
            // the transposed placement, which is validated as a target when the transposed program is compiled.
            match place.target() {
                // A device keeps the sharding of the value, which is over the program mesh, so it is valid exactly when
                // that mesh consists of the device alone.
                Some(PlacementRequest::Device { device, .. }) if mesh.devices() != [*device] => {
                    return Err(Error::Lowering(LoweringError::DeviceOutsideProgramMesh {
                        device: *device,
                        mesh: mesh.clone(),
                    }));
                }
                Some(PlacementRequest::Resolved(placement)) => {
                    let placement_mesh = placement.mesh();
                    if placement_mesh.devices() != mesh.devices()
                        || !placement_mesh.logical_mesh().has_same_topology(mesh.logical_mesh())
                    {
                        return Err(Error::Lowering(LoweringError::PlacementOutsideProgramMesh {
                            target: placement_mesh.clone(),
                            mesh: mesh.clone(),
                        }));
                    }
                    // An unconstrained dimension is stored replicated, as at the boundary of a call.
                    if !use_shardy_partitioner
                        && mesh.devices().len() > 1
                        && !placement.sharding().with_unconstrained_dimensions_replicated()?.is_replicated()
                    {
                        return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                            operation: PLACE_OPERATION_NAME.to_string(),
                            reason: "bounded-dynamic multi-device programs currently require fully replicated \
                                     placement targets because Shardy rejects dynamic tensors"
                                .to_string(),
                        }));
                    }
                }
                // An in-place request leaves the value on the devices where it lives, which are those of the program
                // mesh, and an absent one states nothing, so neither has anything to validate.
                Some(PlacementRequest::Device { .. } | PlacementRequest::InPlace { .. }) | None => {}
            }
        }
    }
    Ok(())
}

/// Returns the runtime placement of every output of `program` compiled as `executable`. The placement of an output
/// that lowering left open over auto mesh axes, as recorded by [`LoweredProgram::open_outputs`], is the placement that
/// the compiler chose for it, which must agree with the placement that the output type records over the other mesh
/// axes. The compiler reports placements in a physical format that cannot name mesh axes of size one, because they do
/// not change how a value is stored, so the declared axes of size one of each dimension are kept in front of the auto
/// axes that the compiler added to it (refer to `with_declared_singleton_axes`).
fn resolve_output_shardings(
    executable: &LoadedExecutable<'_>,
    program: &LoweredProgram,
) -> Result<Arc<[Sharding]>, Error> {
    if !program.open_outputs().iter().any(|open| *open) {
        return Ok(program.output_shardings().into());
    }
    let incompatible = |reason: String| Error::Compilation(CompilationError::IncompatibleExecutable { reason });
    let op_shardings = executable.executable()?.output_shardings()?.ok_or_else(|| {
        incompatible("the PJRT plugin does not report the output shardings that the compiler chose".to_string())
    })?;
    let op_shardings = match op_shardings.as_slice() {
        [sharding] if sharding.r#type == OpShardingType::Tuple as i32 => sharding.tuple_shardings.clone(),
        _ => op_shardings,
    };
    program
        .output_shardings()
        .iter()
        .zip(program.open_outputs().iter())
        .enumerate()
        .map(|(index, (declared, &open))| {
            let Some(physical_index) = program.signature().output_mapping()[index].filter(|_| open) else {
                return Ok(declared.clone());
            };
            let op_sharding = op_shardings.get(physical_index).ok_or_else(|| {
                incompatible(format!("the executable reports no sharding for output {physical_index}"))
            })?;
            op_sharding
                .to_sharding(declared.mesh(), declared.rank())
                .and_then(|sharding| with_declared_singleton_axes(&sharding, declared))
                .filter(|sharding| sharding.without_auto_axes() == declared.without_auto_axes())
                .ok_or_else(|| {
                    incompatible(format!(
                        "the compiler placed output {index} as {op_sharding:?}, which is not a placement over the mesh \
                         axes that agrees with its declared sharding `{declared}`",
                    ))
                })
        })
        .collect()
}

/// Returns `physical`, which is a placement that the compiler reports for an output that lowering left open over auto
/// mesh axes and which therefore names no mesh axis of size one, with the axes of each dimension of `declared` put back
/// in front of the axes that the compiler added to that dimension, or [`None`] when `physical` does not extend
/// `declared` that way. A dimension of `declared` that is sharded only along axes of size one is stored like a
/// replicated dimension, so the compiler reports it as replicated or as sharded along auto axes alone, and the result
/// keeps the axes that `declared` (i.e., the output type) records for it (e.g., `[{"e", "a"}]` for a declared `[{"e"}]`
/// with `e` of size one that the compiler reports as `[{"a"}]`).
fn with_declared_singleton_axes(physical: &Sharding, declared: &Sharding) -> Option<Sharding> {
    let mesh = declared.mesh();
    let axes = |dimension: &ShardingDimension| match dimension {
        ShardingDimension::Sharded(axes) => axes.clone(),
        ShardingDimension::Replicated | ShardingDimension::Unconstrained => Vec::new(),
    };
    let dimensions = physical
        .dimensions()
        .iter()
        .zip(declared.dimensions())
        .map(|(physical, declared)| {
            let (physical, mut declared) = (axes(physical), axes(declared));
            let stored = declared
                .iter()
                .filter(|axis| mesh.axis_index(axis).is_some_and(|index| mesh.axes()[index].size() > 1))
                .collect::<Vec<_>>();
            if physical.len() < stored.len()
                || !physical.iter().zip(&stored).all(|(physical, stored)| physical == *stored)
            {
                return None;
            }
            declared.extend(physical[stored.len()..].iter().cloned());
            Some(if declared.is_empty() { ShardingDimension::Replicated } else { ShardingDimension::Sharded(declared) })
        })
        .collect::<Option<Vec<_>>>()?;
    physical.with_dimensions(dimensions).ok()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, Array as CoreArray, ArrayIrValue, ArrayReferenceTransform, ArrayReferenceTransformIndex,
        ArraySliceAxis, ArrayType, AttachedRegions, CallOperation, CaptureReference, CompilationArtifactExchange,
        CompilationArtifactExchangeError, CompilationArtifactExchangePolicy, CompilationCache, CompilationCacheDomain,
        CompilationTracer, CompiledFunction, CompositeKey, CompositeOperation, ConditionOperation, ConstantOperation,
        Context, CotangentDestinationKind, CumulativeKind, CumulativeOperation, CustomRuleDefinition,
        CustomRuleRegistration, DataType, Device, DeviceMesh, Differentiate, Dimension, DimensionAddOperation,
        DimensionBounds, DimensionFromScalarOperation, DimensionOperation, DimensionSizeOperation, DimensionType,
        DimensionValue, DimensionVariable, DynamicBroadcastOperation, DynamicReshapeOperation,
        EagerInterpretationDriver, Error as CoreError, Exp, ImportingContext, InterpretableOperation, Layout,
        LinearCallOperation, LogicalMesh, Memory, MeshAxis, MeshAxisType, MulOperation, NamedComposite, NegOperation,
        OneOperation, Operation as CoreOperation, Place, Placeholder, Placement, ProgramBuilder, ProgramError,
        Provenance, ProvenanceScope, ReduceOperation, ReductionKind, ReferenceAddUpdate, ReferenceAddUpdateOperation,
        ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation, ReferenceRead,
        ReferenceReadOperation, ReferenceSwapOperation, ReferenceType, ReferenceWrite, ReferenceWriteOperation,
        Reshard, ScanOperation, ScanOptions, Shape, ShardMapBoundary, ShardMapOperation, Sharding, ShardingDimension,
        Sin, SliceOperation, StagedFunction, StridedLayout, ThroughCapture, ValueProjection, WhileOperation,
        differentiate_at, scan,
    };
    use ryft_pjrt::protos::Precision;
    use ryft_pjrt::{BufferSpecification, BufferType, Error as PjrtError, ExecutionFence};

    #[cfg(feature = "mps")]
    use ryft_core::ConstrainShardingOperation;

    use crate::compilation::functions::StatefulCompileTracer;
    use crate::compilation::tests::external_reference_read_program;
    use crate::sessions::tests::{
        boolean_scalar, domain_mesh, f32_scalar, f32_vector, read_first_shard, replicated_scalar_type,
        replicated_vector_type, two_device_mesh,
    };
    use crate::tests::{
        cpu_client, execution_client, other_cpu_client, session_with_remote_devices, values_from_bytes, values_to_bytes,
    };
    use crate::{
        Array, ArrayError, CompileTracer, CompiledFunctionDispatcher, Constant, Error, FromPjrt, Operation, Options,
        PlacementOptions, RetainedArray, Session, StatefulCompiledFunction, Type, Value, compile,
        compile_statefully_with_captures, jit_with_options,
    };

    #[cfg(feature = "mps")]
    use crate::tests::mps_client;

    use super::*;

    #[test]
    fn test_states_boundary_placement() {
        // A nullary function that returns an uncommitted capture on a mesh of one device leaves its result uncommitted
        // when it states no placement at its boundary, also when its options hold an empty list of input overrides,
        // which states nothing, and commits it when an output override states a placement.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let capture = session.place(3f32, PlacementOptions::default()).unwrap();
        let committed = |options: Options| {
            let function: StatefulCompiledFunction<(), Type> = compile_statefully_with_captures(
                |mut captures, ()| Ok(captures.remove(0)),
                vec![Value::Array(capture.clone())],
                (),
                &session,
                options,
            )
            .unwrap();
            let Value::Array(result) = function.execute_statefully_and_wait(&session, ()).unwrap() else {
                unreachable!("the function returns an array")
            };
            assert_eq!(result.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![3.0]);
            result.is_committed()
        };
        assert_eq!(
            [
                committed(Options::new(mesh.clone())),
                committed(Options::new(mesh.clone()).with_input_shardings(vec![])),
                committed(Options::new(mesh.clone()).with_output_shardings(vec![replicated.clone()])),
            ],
            [false, false, true],
        );
        assert!(!states_boundary_placement([]));
        assert!(states_boundary_placement([&replicated]));
    }

    #[test]
    fn test_session_compilation_domain_pipeline() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let options = Options::new(mesh.clone());
        let staged = crate::stage::<_, ArrayType, ArrayType>(|x| x.sin().unwrap(), input_type.clone(), options)
            .unwrap()
            .into_inner();
        let compiled: CompiledFunction<Session, Type, Type> = staged.lower(&engine).unwrap().compile(&engine).unwrap();

        // Round-trip a small input through the `CompilationDomain` pipeline.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let array = compiled.executable().execute(&engine, Value::Array(source)).unwrap();
        let Value::Array(array) = array else {
            panic!("array-only compiled function returned a first-class dimension");
        };
        array.block_until_ready().unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }

        // Equivalent lowerings share the entry populated by the first compilation, independent of source location.
        let program_count_before = engine.cached_program_count();
        for _ in 0..3 {
            let staged = crate::stage::<_, ArrayType, ArrayType>(
                |x| x.sin().unwrap(),
                input_type.clone(),
                Options::new(mesh.clone()),
            )
            .unwrap()
            .into_inner();
            let _: CompiledFunction<Session, Type, Type> = staged.lower(&engine).unwrap().compile(&engine).unwrap();
        }
        assert_eq!(
            engine.cached_program_count(),
            program_count_before,
            "equivalent repeat compilations should reuse the existing cache entry",
        );
    }

    #[test]
    fn test_session_lower_staged_program_rejects_reference_boundary_shardings() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_sharding = Sharding::new(mesh.logical_mesh().clone(), Vec::new()).unwrap();
        let reference_type = Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // Reference inputs are structurally valid for stateful staging, but input sharding overrides that users provide
        // are not, because holder state derives its effective sharding from the compiled reference ABI. Lowering
        // rejects the overrides for staged functions whose input types did not receive them.
        fn forward(
            _: Vec<Constant>,
            _: Vec<CompilationTracer<Session>>,
            input: CompilationTracer<Session>,
        ) -> Result<CompilationTracer<Session>, Error> {
            Ok(input)
        }
        assert!(matches!(
            StagedFunction::<Session, Type, Type>::stage(forward,
                Vec::new(),
                reference_type.clone(),
                Options::new(mesh.clone()).with_input_shardings(vec![scalar_sharding.clone()]),
            )
            .and_then(|staged| staged.lower(&session))
            .map(|_| ()),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference inputs do not support explicit input shardings",
        ));

        // A reference output with `output_shardings` stages successfully and is rejected when lowered, by the contract
        // of the ABI for reference outputs rather than by the output override, which never runs on a reference type.
        fn allocate(
            _: Vec<Constant>,
            _: Vec<CompilationTracer<Session>>,
            input: CompilationTracer<Session>,
        ) -> Result<CompilationTracer<Session>, Error> {
            Ok(input.reference_new()?)
        }
        assert!(matches!(
            StagedFunction::<Session, Type, Type>::stage(allocate,
                Vec::new(),
                Type::Array(ArrayType::scalar(DataType::F32)),
                Options::new(mesh.clone()).with_output_shardings(vec![scalar_sharding]),
            )
            .and_then(|staged| staged.lower(&session))
            .map(|_| ()),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "output 0 is a reference allocated inside the program and cannot be returned through \
                              the stateful XLA ABI, which has no protocol to turn a backend array into a host \
                              reference",
        ));

        // Forwarding an external reference through an output is rejected by the same contract when lowered, because
        // the ABI has no protocol for handing an entering reference back through an output either.
        assert!(matches!(
            StagedFunction::<Session, Type, Type>::stage(forward,
                Vec::new(),
                reference_type,
                Options::new(mesh.clone()),
            )
            .and_then(|staged| staged.lower(&session))
            .map(|_| ()),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "output 0 forwards the external reference bound at input 0, and the stateful XLA ABI \
                              does not return forwarded references; read the reference after the call instead",
        ));
    }

    #[test]
    fn test_session_lower_staged_program_rejects_reference_outputs() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);

        fn allocate_reference(
            _: Vec<Constant>,
            _: Vec<StatefulCompileTracer>,
            input: StatefulCompileTracer,
        ) -> Result<StatefulCompileTracer, Error> {
            input.reference_new().map_err(Into::into)
        }

        // The stateful XLA ABI only restricts the outputs of a lowered program, so a function that returns a reference
        // stages successfully and is rejected when it is lowered on its own.
        let staged = StagedFunction::<Session, Type, Type>::stage(
            allocate_reference,
            Vec::new(),
            Type::Array(array_type),
            Options::new(mesh),
        )
        .unwrap();
        let error = staged.lower(&session).err().unwrap();
        assert!(matches!(
            error,
            Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason })
                if reason == "output 0 is a reference allocated inside the program and cannot be returned through the \
                              stateful XLA ABI, which has no protocol to turn a backend array into a host reference",
        ));
    }

    #[test]
    fn test_session_lower_staged_program_requires_input_shardings() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F32));
        let options = Options::new(mesh.clone())
            .with_input_shardings(vec![Sharding::new(mesh.logical_mesh().clone(), Vec::new()).unwrap()]);

        fn forward(
            _: Vec<Constant>,
            _: Vec<CompilationTracer<Session>>,
            input: CompilationTracer<Session>,
        ) -> Result<CompilationTracer<Session>, Error> {
            Ok(input)
        }

        // Staging traces exactly the input types that it receives, so lowering rejects a function whose input types
        // did not receive the requested input shardings before staging.
        let staged =
            StagedFunction::<Session, Type, Type>::stage(forward, Vec::new(), scalar_type.clone(), options.clone())
                .unwrap();
        assert!(matches!(
            staged.lower(&session).map(|_| ()),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "staged input types do not carry the shardings requested by `input_shardings`; apply them \
                              to the input types before staging (e.g., with `with_input_shardings`)",
        ));

        let input_types = with_input_shardings(scalar_type, &options).unwrap();
        let staged = StagedFunction::<Session, Type, Type>::stage(forward, Vec::new(), input_types, options).unwrap();
        staged.lower(&session).unwrap();
    }

    #[test]
    fn test_session_lower_staged_program_requires_output_shardings() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = |axis_type| {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let devices = client
                .addressable_devices()
                .unwrap()
                .into_iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect::<Vec<_>>();
            DeviceMesh::new(logical_mesh, devices).unwrap()
        };
        let input_type = |mesh: &DeviceMesh| {
            let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
            Type::Array(ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated).unwrap())
        };
        let options = |mesh: &DeviceMesh| {
            let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            Options::new(mesh.clone()).with_output_shardings(vec![sharded])
        };

        fn forward(
            _: Vec<Constant>,
            _: Vec<CompilationTracer<Session>>,
            input: CompilationTracer<Session>,
        ) -> Result<CompilationTracer<Session>, Error> {
            Ok(input)
        }

        fn reshard(
            _: Vec<Constant>,
            _: Vec<CompilationTracer<Session>>,
            input: CompilationTracer<Session>,
        ) -> Result<CompilationTracer<Session>, Error> {
            let mesh = <&ArrayType>::try_from(input.r#type().as_ref()).unwrap().sharding().unwrap().mesh().clone();
            Ok(input.reshard(&Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())?)
        }

        // Lowering checks rather than overwrites the placement that staged types track, so that calls of a staged
        // function return the output types of its executions.
        let mesh_explicit = mesh(MeshAxisType::Explicit);
        let staged = StagedFunction::<Session, Type, Type>::stage(
            forward,
            Vec::new(),
            input_type(&mesh_explicit),
            options(&mesh_explicit),
        )
        .unwrap();
        assert!(matches!(
            staged.lower(&session).map(|_| ()),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "staged output 0 has type `f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}]`, which does \
                              not carry the sharding `{mesh<['x'=2:explicit]>, [{'x'}]}` requested by \
                              `output_shardings`; reshard the outputs to their overrides before staging them",
        ));
        let staged = StagedFunction::<Session, Type, Type>::stage(
            reshard,
            Vec::new(),
            input_type(&mesh_explicit),
            options(&mesh_explicit),
        )
        .unwrap();
        staged.lower(&session).unwrap();

        // Staged types cannot record placement over auto mesh axes, so lowering applies it to the executable outputs,
        // whose types keep the staged placement while their runtime placements carry the override.
        let mesh_auto = mesh(MeshAxisType::Auto);
        let staged = StagedFunction::<Session, Type, Type>::stage(
            forward,
            Vec::new(),
            input_type(&mesh_auto),
            options(&mesh_auto),
        )
        .unwrap();
        let lowered = staged.lower(&session).unwrap();
        assert_eq!(
            lowered.lowered_program().output_types()[0].sharding().unwrap().to_string(),
            "{mesh<['x'=2:auto]>, [{}]}",
        );
        assert_eq!(lowered.lowered_program().output_shardings()[0].to_string(), "{mesh<['x'=2:auto]>, [{'x'}]}");
    }

    #[test]
    fn test_session_lower_program_compilation_options() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let compilation_options =
            CompilationOptions { matrix_unit_operand_precision: Precision::Highest as i32, ..Default::default() };
        let session = Session::builder().client(client).compilation_options(compilation_options).build().unwrap();
        let program = external_reference_read_program(replicated_vector_type(&mesh, 2));

        // Lowering starts from the template of the session and overlays the settings that it derives for the program.
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.compilation_options().matrix_unit_operand_precision, Precision::Highest as i32);
        assert_eq!(lowered.compilation_options().executable_build_options.as_ref().unwrap().partition_count, 1);

        // Compilation options that the options of one compilation supply replace the template for that compilation.
        let options = Options::new(mesh)
            .with_compilation_options(CompilationOptions { compile_portable_executable: true, ..Default::default() });
        let lowered = session.lower_program(&program, 0, &options).unwrap();
        assert!(lowered.compilation_options().compile_portable_executable);
        assert_eq!(lowered.compilation_options().matrix_unit_operand_precision, Precision::Default as i32);
        assert_eq!(lowered.compilation_options().executable_build_options.as_ref().unwrap().partition_count, 1);
    }

    #[test]
    fn test_session_lower_program_pins_the_executable_to_the_mesh_devices() {
        // The mesh lists the devices of the client in reverse order, so its first shard lives on the second device.
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .rev()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh, devices.clone()).unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let operation = CumulativeOperation::new(0, CumulativeKind::Sum);
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let options = Options::new(mesh.clone()).with_input_shardings(vec![sharding.clone()]);
        let lowered = session.lower_program(&program, 0, &options).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();

        // The executable runs its partitions on the devices of the mesh, in mesh order.
        let assignment = compiled.executable().device_assignment().unwrap();
        assert_eq!((assignment.replica_count(), assignment.computation_count()), (1, 2));
        assert_eq!(
            vec![assignment.device_id(0, 0).unwrap(), assignment.device_id(0, 1).unwrap()],
            devices.iter().map(Device::id).collect::<Vec<_>>(),
        );

        // A prefix sum depends on the order of the shards, so it is only correct when each partition runs on the
        // device that holds its shard.
        let input = Array::from_host_buffer(
            &session,
            input_type,
            mesh,
            Some(sharding),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let outputs = session.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(outputs[0].copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_session_lower_program_closes_unconstrained_output_shardings() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let input_type = ArrayType::new_static(DataType::I32, [4, 4]);
        let input_sharding = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["r"])],
        )
        .unwrap();
        let open_sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()])
                .unwrap();
        let target = Placement::new(mesh.clone(), open_sharding.clone()).unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let output =
            builder.add_instruction(PlaceOperation::new(target, None), Vec::new(), vec![input], None).unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // An output override with an unconstrained dimension is closed at the function boundary like the target of the
        // `place` that produces the output, because the runtime placement of the output reads that dimension as
        // replicated. Without either, the compiler would keep the split of the input along `r` on that dimension.
        let options = Options::new(mesh.clone())
            .with_input_shardings(vec![input_sharding.clone()])
            .with_output_shardings(vec![open_sharding.clone()]);
        let lowered = session.lower_program(&program, 0, &options).unwrap();
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "r"=2]>
                  func.func @main(%arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"r"}]>}) -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}, {?}]> : tensor<4x4xi32>
                    return %0 : tensor<4x4xi32>
                  }
                }
            "#},
        );
        let compiled = session.compile_program(&lowered).unwrap();
        let values = (0..16).collect::<Vec<i32>>();
        let input = Array::from_host_buffer(&session, input_type, mesh, Some(input_sharding), values_to_bytes(&values))
            .unwrap();
        let outputs = session.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(outputs[0].sharding(), &open_sharding);
        assert_eq!(
            outputs[0].shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
            vec!["[2, 4]", "[2, 4]", "[2, 4]", "[2, 4]"],
        );
        assert_eq!(
            outputs[0]
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<i32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>(),
            vec![
                (0..8).collect::<Vec<_>>(),
                (0..8).collect::<Vec<_>>(),
                (8..16).collect::<Vec<_>>(),
                (8..16).collect::<Vec<_>>(),
            ],
        );
    }

    #[test]
    fn test_session_lower_program_closes_unconstrained_output_type_shardings() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let input_type = ArrayType::new_static(DataType::I32, [4, 4]);
        let input_sharding = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let open_sharding = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()],
        )
        .unwrap();
        let output_sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::replicated(), ShardingDimension::unconstrained()])
                .unwrap();
        let target = Placement::new(mesh.clone(), open_sharding).unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let placed =
            builder.add_instruction(PlaceOperation::new(target, None), Vec::new(), vec![input], None).unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![placed, placed], None).unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // The output type inherits the unconstrained dimension of the `place` target, but the output is not the result
        // of that `place`. Its annotation closes that dimension like the target of a `place` that produces an output,
        // so that the compiler returns the buffers that the runtime placement of the output describes.
        let options = Options::new(mesh.clone()).with_input_shardings(vec![input_sharding.clone()]);
        let lowered = session.lower_program(&program, 0, &options).unwrap();
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "r"=2]>
                  func.func @main(%arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}, {?}]> : tensor<4x4xi32>
                    %1 = stablehlo.add %0, %0 : tensor<4x4xi32>
                    return %1 : tensor<4x4xi32>
                  }
                }
            "#},
        );
        let compiled = session.compile_program(&lowered).unwrap();
        let values = (0..16).collect::<Vec<i32>>();
        let input = Array::from_host_buffer(&session, input_type, mesh, Some(input_sharding), values_to_bytes(&values))
            .unwrap();
        let outputs = session.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(outputs[0].sharding(), &output_sharding);
        assert_eq!(
            outputs[0].shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
            vec!["[4, 4]", "[4, 4]", "[4, 4]", "[4, 4]"],
        );
        assert_eq!(
            outputs[0]
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<i32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>(),
            vec![(0..16).map(|value| 2 * value).collect::<Vec<_>>(); 4],
        );
    }

    #[test]
    fn test_session_lower_program_closes_unconstrained_argument_shardings() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let input_type = ArrayType::new_static(DataType::I32, [4, 4]);
        let input_sharding = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()],
        )
        .unwrap();
        let target_sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["r"])])
                .unwrap();
        let target = Placement::new(mesh.clone(), target_sharding).unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let placed =
            builder.add_instruction(PlaceOperation::new(target, None), Vec::new(), vec![input], None).unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![placed, placed], None).unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // An array placed with an unconstrained dimension stores that dimension replicated, so the argument that it
        // enters as is annotated with that dimension closed, whether its placement is an input override or the observed
        // placement of the runtime array. Otherwise, the compiler could shard that dimension and expect smaller buffers
        // than the array supplies.
        let values = (0..16).collect::<Vec<i32>>();
        for options in [
            Options::new(mesh.clone()).with_input_shardings(vec![input_sharding.clone()]),
            Options::new(mesh.clone()).with_input_placements(vec![Some(input_sharding.clone())].into()),
        ] {
            let lowered = session.lower_program(&program, 0, &options).unwrap();
            assert_eq!(
                lowered.stable_hlo(),
                indoc! {r#"
                    module {
                      sdy.mesh @mesh = <["x"=2, "r"=2]>
                      func.func @main(%arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}]>}) {
                        %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}, {"r"}]> : tensor<4x4xi32>
                        %1 = stablehlo.add %0, %0 : tensor<4x4xi32>
                        return %1 : tensor<4x4xi32>
                      }
                    }
                "#},
            );
            let compiled = session.compile_program(&lowered).unwrap();
            let input = Array::from_host_buffer(
                &session,
                input_type.clone(),
                mesh.clone(),
                Some(input_sharding.clone()),
                values_to_bytes(&values),
            )
            .unwrap();
            assert_eq!(
                input.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
                vec!["[2, 4]", "[2, 4]", "[2, 4]", "[2, 4]"],
            );
            let outputs = session.execute_program(&compiled, vec![input]).unwrap();
            // The output is open over the auto axes, and the compiler keeps the placement of the `place` target.
            assert_eq!(outputs[0].sharding().to_string(), "{mesh<['x'=2:auto, 'r'=2:auto]>, [{'x'}, {'r'}]}");
            assert_eq!(
                outputs[0].shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
                vec!["[2, 2]", "[2, 2]", "[2, 2]", "[2, 2]"],
            );
            assert_eq!(
                outputs[0].copy_to_host().unwrap().elements::<i32>().unwrap(),
                values.iter().map(|value| 2 * value).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn test_compiled_outputs_keep_requested_unconstrained_sharding_on_the_anonymous_mesh() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::I32, [4]);
        let values = [0i32, 1, 2, 3];
        let input = Array::from_host_buffer(&session, r#type.clone(), None, None, values_to_bytes(&values)).unwrap();
        let mesh = input.mesh();
        let unconstrained =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::unconstrained()]).unwrap();
        let target = Placement::new(mesh.clone(), unconstrained.clone()).unwrap();
        let options = || Options::new(mesh.clone());

        // Both an output that forwards an argument and one produced by an explicit placement retain the requested
        // metadata. The anonymous type omits its sharding, and every executable buffer holds the full extent.
        let identity = crate::compile_with_options::<_, ArrayType, ArrayType>(
            |x: CompileTracer| x,
            r#type.clone(),
            &session,
            options().with_input_shardings(vec![unconstrained.clone()]),
        )
        .unwrap();
        let placed = crate::compile_with_options::<_, ArrayType, ArrayType>(
            |x: CompileTracer| x.place(&target, None).unwrap(),
            r#type.clone(),
            &session,
            options(),
        )
        .unwrap();
        for function in [identity, placed] {
            let executed = function.executable().execute(&session, input.clone()).unwrap();
            let called = function.call(input.clone()).unwrap();
            for output in [executed, called] {
                assert_eq!(output.r#type().as_ref(), &r#type);
                assert_eq!(output.mesh(), mesh);
                assert_eq!(output.sharding(), &unconstrained);
                assert!(output.is_committed());
                let shard = output.addressable_shards().next().unwrap();
                assert_eq!(shard.shape().as_slice(), &[4]);
                assert_eq!(values_from_bytes::<i32>(&shard.copy_to_host_bytes().unwrap()), values);
            }
        }
    }

    #[test]
    fn test_session_lower_program_places_outputs_that_return_inputs_as_their_inputs() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let input_type = ArrayType::new_static(DataType::Zero, [4]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let output =
            builder.add_instruction(PlaceOperation::new(None, None), Vec::new(), vec![input], None).unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // A `place` without a target returns its input where it is. A result of the zero data type has no buffer whose
        // placement the compiler could report, so it takes the placement that the input enters with.
        let options = Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]);
        let lowered = session.lower_program(&program, 0, &options).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let input = Array::from_host_buffer(&session, input_type, mesh, Some(sharded.clone()), []).unwrap();
        let outputs = session.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(outputs[0].sharding(), &sharded);
        assert_eq!(
            outputs[0].shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
            vec!["[2]", "[2]"],
        );
    }

    #[test]
    fn test_session_lower_program_places_outputs_of_calls_as_their_callees() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let input_type = ArrayType::new_static(DataType::Zero, [4]);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let placing_callee = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(input_type.clone().into());
            let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
            let output =
                builder.add_instruction(PlaceOperation::new(target, None), Vec::new(), vec![input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let forwarding_callee = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(input_type.clone().into());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let placing_callee = builder.import_region(placing_callee.entry_region_ref());
        let forwarding_callee = builder.import_region(forwarding_callee.entry_region_ref());
        let replicated_input = builder.add_input(input_type.clone().into());
        let sharded_input = builder.add_input(input_type.clone().into());
        let call = |builder: &mut ProgramBuilder<Constant, Operation>, callee, input| {
            builder
                .add_instruction(Operation::Call(CallOperation::new(0)), vec![callee], vec![input], None)
                .unwrap()[0]
        };
        let placed = call(&mut builder, placing_callee, replicated_input);
        let forwarded = call(&mut builder, forwarding_callee, sharded_input);
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![placed, forwarded], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // A `call` whose callee places its result or returns one of its inputs places its own result as the callee
        // does. A result of the zero data type has no buffer whose placement the compiler could report, so it takes
        // its placement from the `place` in the callee or from the argument that the call forwards.
        let options = Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone(), sharded.clone()]);
        let lowered = session.lower_program(&program, 0, &options).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let replicated_input =
            Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), Some(replicated), []).unwrap();
        let sharded_input = Array::from_host_buffer(&session, input_type, mesh, Some(sharded.clone()), []).unwrap();
        let outputs = session.execute_program(&compiled, vec![replicated_input, sharded_input]).unwrap();
        assert_eq!(outputs[0].sharding(), &sharded);
        assert_eq!(outputs[1].sharding(), &sharded);
        assert_eq!(
            outputs[0].shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
            vec!["[2]", "[2]"],
        );
        assert_eq!(
            outputs[1].shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
            vec!["[2]", "[2]"],
        );
    }

    #[test]
    fn test_session_lower_program_places_completed_device_reductions_as_their_types() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(unreduced).unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(PlaceOperation::new(PlacementRequest::from(device), None), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // A device target completes the pending reduction of its input along `x`, which has one device, so its result
        // is placed and annotated as its type states rather than as the argument that it is computed from, and `exp`
        // applies to that result.
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x"}>}) -> (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={"x"}>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "#},
        );
        let compiled = session.compile_program(&lowered).unwrap();
        let input =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[1f32, 2.])).unwrap();
        let outputs = session.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(outputs[0].mesh(), &mesh);
        assert_eq!(outputs[0].sharding(), &replicated);
        assert_eq!(outputs[0].copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1f32, 2.]);
        let exponential = outputs[0].exp().unwrap();
        assert_eq!(exponential.sharding(), &replicated);
        assert_eq!(exponential.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1f32.exp(), 2f32.exp()]);
    }

    #[test]
    fn test_session_lower_program_places_retained_array_outputs_as_their_arrays() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let zero_type = ArrayType::new_static(DataType::Zero, [4]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array =
            Array::from_host_buffer(&session, zero_type.clone(), mesh.clone(), Some(sharded.clone()), []).unwrap();
        let retained = RetainedArray::new(&array).with_declared_type(zero_type.clone());
        let callee = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let output = builder.add_constant(Constant::Session(retained.clone()));
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap()
        };
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let returned = builder.add_constant(Constant::Session(retained));
        let called = builder
            .add_instruction(Operation::Call(CallOperation::new(0)), vec![callee], Vec::new(), None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![returned, called], Vec::new(), vec![Placeholder; 2])
            .unwrap();

        // A retained array of the zero data type has no buffer whose placement the compiler could report, so a result
        // that returns it, directly or from the callee of a `call`, takes the placement of the retained array.
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        // The executable receives the retained array as its hidden session constant argument.
        let outputs = session.execute_program(&compiled, vec![array]).unwrap();
        let output_type = zero_type.with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        assert_eq!(outputs[0].r#type().as_ref(), &output_type);
        assert_eq!(outputs[0].sharding(), &sharded);
        assert_eq!(
            outputs[0].shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
            vec!["[2]", "[2]"],
        );
        assert_eq!(outputs[1].r#type().as_ref(), &output_type);
        assert_eq!(outputs[1].sharding(), &sharded);
        assert_eq!(
            outputs[1].shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
            vec!["[2]", "[2]"],
        );
        outputs[0].block_until_ready().unwrap();
        outputs[1].block_until_ready().unwrap();

        // Retargeting the same program to a singleton mesh places both results as the bound hidden argument.
        let singleton = DeviceMesh::new(LogicalMesh::single_device(), vec![mesh.devices()[0]]).unwrap();
        let replicated = Sharding::replicated(singleton.logical_mesh().clone(), 1);
        let lowered = session.lower_program(&program, 0, &Options::new(singleton)).unwrap();
        let [binding] = lowered.session_constants() else { panic!("expected one hidden argument") };
        assert_eq!(binding.source().sharding(), &sharded);
        assert_eq!(binding.argument().sharding(), &replicated);
        assert_eq!(lowered.output_shardings(), &[replicated.clone(), replicated]);
    }

    #[test]
    fn test_session_lower_program_binds_shared_captures_by_original_commitment() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::scalar(DataType::F32);
        let array = Array::from_host_buffer(&session, r#type.clone(), None, None, 3f32.to_ne_bytes()).unwrap();
        let uncommitted = RetainedArray::new(&array).with_declared_type(r#type.clone());
        let committed = RetainedArray::new(&array.clone().with_committed(true)).with_declared_type(r#type);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let first = builder.add_constant(Constant::Session(uncommitted.clone()));
        let second = builder.add_constant(Constant::Session(committed.clone()));
        let program: FlatProgram<Session> =
            builder.build(vec![first, second], Vec::new(), vec![Placeholder; 2]).unwrap();

        let lowered = session.lower_program(&program, 0, &Options::new(array.mesh().clone())).unwrap();
        let [first, second] = lowered.session_constants() else { panic!("expected two hidden arguments") };
        assert_eq!(first.source(), &uncommitted);
        assert_eq!(second.source(), &committed);
        // Both arguments share the same committed placement, but their source identities have separate slots.
        assert_eq!(first.argument(), second.argument());
        let (_, captures) = CompilationDomain::compile(&session, &lowered).unwrap();
        let [Value::Array(first), Value::Array(second)] = captures.as_slice() else {
            panic!("expected two executable captures")
        };
        for capture in [first, second] {
            assert!(capture.is_committed());
            assert!(!capture.is_locally_assigned());
            assert!(Arc::ptr_eq(capture.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));
        }

        // Lifting the captures on a session with the mesh of the array, as the interpretation of a program that holds
        // them does, uses the same placed buffers and gives each capture its original commitment.
        let meshed = session.with_device_mesh(array.mesh().clone());
        for (source, committed) in [(uncommitted, false), (committed, true)] {
            let Value::Array(capture) = meshed.lift(Constant::Session(source)).unwrap() else {
                panic!("expected a lifted array")
            };
            assert_eq!(capture.is_committed(), committed);
            assert!(!capture.is_locally_assigned());
            assert!(Arc::ptr_eq(capture.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));
        }
    }

    #[test]
    fn test_session_lower_program_dynamic_dimension_arithmetic_broadcast_and_reshape() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let build_program = |vector_type: &ArrayType, increment: usize| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let vector = builder.add_input(vector_type.clone().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let size_operation = DimensionSizeOperation::new(vector_type, 0).unwrap();
            let size = builder.add_instruction(size_operation.clone(), Vec::new(), vec![vector], None).unwrap()[0];
            let increment_value = DimensionValue::constant(increment).unwrap();
            let increment = builder
                .add_instruction(
                    Operation::Dimension(DimensionOperation::Constant(ConstantOperation::new(increment_value.clone()))),
                    Vec::new(),
                    Vec::new(),
                    None,
                )
                .unwrap()[0];
            let add =
                DimensionAddOperation::new(size_operation.output_type(), increment_value.r#type().as_ref()).unwrap();
            let output_extent = builder
                .add_instruction(
                    Operation::Dimension(DimensionOperation::Add(add)),
                    Vec::new(),
                    vec![size, increment],
                    None,
                )
                .unwrap()[0];
            let broadcast = builder
                .add_instruction(
                    DynamicBroadcastOperation::new(Vec::new()),
                    Vec::new(),
                    vec![scalar, output_extent],
                    None,
                )
                .unwrap()[0];
            let reshaped = builder
                .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![broadcast, output_extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![reshaped],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        let dynamic_program = build_program(&vector_type, 1);
        let dynamic_lowering = session.lower_program(&dynamic_program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(
            dynamic_lowering.stable_hlo(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<7xf32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<8>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<7xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<7>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<7>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %c = stablehlo.constant dense<1> : tensor<i64>
                    %3 = stablehlo.add %2, %c : tensor<i64>
                    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<8xf32>
                    %5 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
                    %6 = stablehlo.set_dimension_size %4, %5, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %7 = stablehlo.get_dimension_size %6, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %8 = stablehlo.convert %7 : (tensor<i32>) -> tensor<i64>
                    %c_0 = stablehlo.constant dense<0> : tensor<i64>
                    %9 = stablehlo.compare LT, %8, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %10 = stablehlo.after_all  : !stablehlo.token
                    %11 = stablehlo.custom_call @ryft.assert(%9, %8, %3, %10) {api_version = 4 : i32, backend_config = {actor = "reshape", detail = "1", kind = "reshape"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %12 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
                    %13 = stablehlo.reshape %12 : (tensor<i32>) -> tensor<1xi32>
                    %14 = stablehlo.dynamic_reshape %6, %13 : (tensor<?xf32, #stablehlo.bounds<8>>, tensor<1xi32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %15 = stablehlo.get_dimension_size %14, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %16 = stablehlo.convert %15 : (tensor<i32>) -> tensor<i64>
                    return %14, %16 : tensor<?xf32, #stablehlo.bounds<8>>, tensor<i64>
                  }
                }
            "#},
        );

        // Persistent compilation identity canonicalizes variable names, which only serve diagnostics, while retaining
        // dimension SSA semantics. Rebuilding the same graph over a fresh nominal identity therefore shares its
        // compilation key, while changing the dimension arithmetic does not.
        let renamed_extent = DimensionVariable::new("renamed_extent", DimensionBounds::new(1, Some(8)).unwrap());
        let renamed_type = ArrayType::new(DataType::F32, Shape::new(vec![renamed_extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let renamed_program = build_program(&renamed_type, 1);
        let renamed_lowering = session.lower_program(&renamed_program, 0, &Options::new(mesh.clone())).unwrap();
        let different_program = build_program(&vector_type, 2);
        let different_lowering = session.lower_program(&different_program, 0, &Options::new(mesh.clone())).unwrap();
        let dynamic_key = session.compilation_key(&dynamic_lowering).unwrap();
        assert_eq!(dynamic_key, session.compilation_key(&renamed_lowering).unwrap());
        assert_ne!(dynamic_key, session.compilation_key(&different_lowering).unwrap());

        // CPU PJRT does not provide the `PadToStatic` custom call for bounded inputs. Execute the same first-class
        // dimension graph with a static input axis. This changes only the dimension size lowering from a runtime read
        // to its exact scalar constant and still exercises dimension SSA arithmetic and both mixed shape operations.
        let static_vector_type = replicated_vector_type(&mesh, 3);
        let static_program = build_program(&static_vector_type, 1);
        let static_lowering = session.lower_program(&static_program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&static_lowering).unwrap();
        let vector = Array::from_host_buffer(
            &session,
            static_vector_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0]),
        )
        .unwrap();
        let scalar = f32_scalar(&session, &mesh, 7.0);
        let outputs = session.execute_program(&compiled, vec![vector, scalar]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape().as_slice(), &[4]);
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![7.0; 4]);
    }

    #[test]
    fn test_session_lower_program_rejects_shard_maps_in_bounded_dynamic_programs() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Manual).unwrap()]).unwrap();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let value_type = replicated_scalar_type(&mesh, DataType::F32);
        let global_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let manual_sharding = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let boundary =
            ShardMapBoundary::new(manual_mesh, vec![manual_sharding.clone()], vec![manual_sharding], Vec::new())
                .unwrap();
        let local_type = boundary.local_input_type(0, &global_type).unwrap();

        // A `shard_map` body always has static boundary types, so it can coexist with dynamic tensors that are staged
        // beside it. Here the gateway makes the program bounded dynamic while the `shard_map` stays fully static.
        let mut body_builder = ProgramBuilder::<Constant, Operation>::new();
        let local = body_builder.add_input(local_type.into());
        let doubled =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![local, local], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let size = builder.add_input(size_type.into());
        let value = builder.add_input(value_type.into());
        let global = builder.add_input(global_type.clone().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let dynamic = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, dimension], None)
            .unwrap()[0];
        let operation = ShardMapOperation::from_program(&body, vec![global_type.into()], boundary).unwrap();
        let mapped = builder
            .add_instruction(Operation::ShardMap(Box::new(operation)), vec![body_region], vec![global], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![mapped, dynamic],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();

        let Err(error) = session.lower_program(&program, 0, &Options::new(mesh)) else {
            panic!("bounded-dynamic program containing a shard_map unexpectedly lowered");
        };
        assert!(matches!(&error,
            Error::Lowering(crate::LoweringError::UnsupportedOperation { operation, .. }) if operation == "shard_map",
        ));
        assert_eq!(
            error.to_string(),
            "cannot lower `shard_map`: bounded-dynamic programs cannot contain `shard_map` regions because \
             `sdy.manual_computation` requires the Shardy partitioner",
        );
    }

    #[test]
    fn test_session_lower_program_rejects_split_output_shardings_in_bounded_dynamic_programs() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let extent = DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]));
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.into());
        let program: FlatProgram<Session> = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let lower = |output_sharding: &Sharding| {
            let options = Options::new(mesh.clone())
                .with_input_shardings(vec![replicated.clone()])
                .with_output_shardings(vec![output_sharding.clone()]);
            session.lower_program(&program, 0, &options)
        };

        // Bounded dynamic programs run one fully replicated replica per device, because the Shardy partitioner rejects
        // dynamic tensors, so an output override that splits a dimension along an auto axis, which the output type does
        // not record, would return full buffers where the runtime placement of the output describes halves.
        assert!(matches!(
            lower(&sharded),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "bounded-dynamic multi-device programs currently require fully replicated shardings \
                              because Shardy rejects dynamic tensors",
        ));

        // A replicated override is what every replica computes anyway.
        let lowered = lower(&replicated).unwrap();
        assert_eq!(lowered.output_shardings(), &[replicated]);
        assert_eq!(lowered.open_outputs(), &[false]);
    }

    #[test]
    fn test_session_lower_program_validates_call_boundaries() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let devices = mesh.devices().to_vec();
        let single_device_mesh = |device: Device| {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            DeviceMesh::new(logical_mesh, vec![device]).unwrap()
        };
        let (first_mesh, second_mesh) = (single_device_mesh(devices[0]), single_device_mesh(devices[1]));
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let extent = DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]));

        // Program `(x, y) -> (call(x), y)`, whose callee is an identity with an output override over the devices of
        // `callee_mesh`.
        let program = |callee_mesh: &DeviceMesh, sharding: &Sharding, extra_type: &ArrayType| {
            let callee = {
                let mut builder = ProgramBuilder::<Constant, Operation>::new();
                let input = builder.add_input(vector_type.clone().into());
                let callee =
                    builder.build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder]);
                Arc::new(callee.unwrap())
            };
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let extra = builder.add_input(extra_type.clone().into());
            let callee_region = builder.intern_callee(&callee, None).unwrap();
            let call = CallOperation::new(0).with_boundary(Some(callee_mesh), Vec::new(), vec![Some(sharding.clone())]);
            let output =
                builder.add_instruction(Operation::Call(call), vec![callee_region], vec![input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output, extra], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let single_replicated = Sharding::replicated(first_mesh.logical_mesh().clone(), 1);

        // A call of a function compiled for other devices cannot run inside a program compiled for these devices, as
        // JAX rejects a nested `jit` whose shardings name other devices, whereas a call of a function compiled for the
        // same devices can.
        assert!(matches!(
            session.lower_program(&program(&second_mesh, &single_replicated, &vector_type), 0, &Options::new(first_mesh.clone())),
            Err(Error::Lowering(LoweringError::CallOutsideProgramMesh { callee, mesh }))
                if callee == second_mesh && mesh == first_mesh,
        ));
        assert_eq!(
            Error::Lowering(LoweringError::CallOutsideProgramMesh {
                callee: second_mesh.clone(),
                mesh: first_mesh.clone()
            })
            .to_string(),
            "`call` cannot call a function compiled for the devices [1] of mesh `['x'=1:explicit]` inside a program \
             compiled for the devices [0] of mesh `['x'=1:explicit]`",
        );
        assert!(
            session
                .lower_program(
                    &program(&first_mesh, &single_replicated, &vector_type),
                    0,
                    &Options::new(first_mesh.clone())
                )
                .is_ok()
        );

        // The replicas of a bounded dynamic program cannot enforce a call override that splits a value, as they cannot
        // enforce such a `place`, while a replicated override is what every replica computes anyway.
        assert!(matches!(
            session.lower_program(&program(&mesh, &sharded, &dynamic_type), 0, &Options::new(mesh.clone())),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == CALL_OPERATION_NAME
                    && reason == "bounded-dynamic multi-device programs currently require fully replicated boundary \
                                  shardings of called functions because Shardy rejects dynamic tensors",
        ));
        assert!(
            session
                .lower_program(&program(&mesh, &replicated, &dynamic_type), 0, &Options::new(mesh.clone()))
                .is_ok()
        );
        assert!(
            session
                .lower_program(&program(&mesh, &sharded, &vector_type), 0, &Options::new(mesh.clone()))
                .is_ok()
        );
    }

    #[test]
    fn test_session_lower_program_rejects_uneven_shardings() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let compile = |extent: usize, options: Options| {
            let target = Placement::new(mesh.clone(), replicated.clone()).unwrap();
            crate::compile_with_options::<_, ArrayType, ArrayType>(
                move |input| input.place(&target, None).unwrap(),
                ArrayType::new_static(DataType::F32, [extent]),
                &session,
                options,
            )
        };
        let uneven = |kind: &str| {
            format!(
                "the sharding of {kind} 0 splits dimension 0 of extent 5 into 2 shards, but XLA requires the number of \
                 shards to divide the extent",
            )
        };

        // The SPMD partitioner has no uneven splits, so an executable whose argument splits five elements over two
        // devices would expect buffers other than the `[3]` and `[2]` shards of its runtime array. Compilation rejects
        // such a placement, whether an input override requests it or a resident array has it.
        let input_options = Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]);
        assert!(matches!(
            compile(5, input_options.clone()),
            Err(Error::Lowering(LoweringError::UnevenSharding { .. })),
        ));
        assert_eq!(compile(5, input_options).err().unwrap().to_string(), uneven("input"));
        let placement_options = Options::new(mesh.clone()).with_input_placements(vec![Some(sharded.clone())].into());
        assert_eq!(compile(5, placement_options).err().unwrap().to_string(), uneven("input"));

        // An output that the executable returns split unevenly would not match its runtime placement either.
        let output_options = Options::new(mesh.clone()).with_output_shardings(vec![sharded.clone()]);
        assert_eq!(compile(5, output_options).err().unwrap().to_string(), uneven("output"));

        // Even splits compile, and the executable returns the exact shards of its runtime placement.
        let options = Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]);
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            Some(sharded.clone()),
            values_to_bytes(&[0f32, 1., 2., 3.]),
        )
        .unwrap();
        let output = compile(4, options).unwrap().executable().execute(&session, input).unwrap();
        output.block_until_ready().unwrap();
        assert_eq!(output.sharding(), &replicated);
        assert!(output.is_committed());
        assert_eq!(output.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(), ["[4]", "[4]"]);
        assert_eq!(
            output.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>(),
            [values_to_bytes(&[0f32, 1., 2., 3.]), values_to_bytes(&[0f32, 1., 2., 3.])],
        );
    }

    #[test]
    fn test_session_lower_program_accepts_uneven_shardings_of_values_without_buffers() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let shapes = |array: &Array| array.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>();
        let identity = |extent: usize, options: Options| {
            crate::compile_with_options::<_, ArrayType, ArrayType>(
                |input| input,
                ArrayType::new_static(DataType::Zero, [extent]),
                &session,
                options,
            )
        };

        // Statically shaped zero values have no buffers in the executable, so an uneven split of them describes no
        // buffer that the executable could expect or return. An identity passes such a value through, whether an input
        // override, an output override, or the resident array that a specialization is compiled for splits it.
        for extent in [5, 4] {
            let zero_type = ArrayType::new_static(DataType::Zero, [extent]);
            let zero =
                Array::from_host_buffer(&session, zero_type.clone(), mesh.clone(), Some(sharded.clone()), []).unwrap();
            let expected_shapes = if extent == 5 { ["[3]", "[2]"] } else { ["[2]", "[2]"] };
            assert_eq!(shapes(&zero), expected_shapes);
            let dispatcher: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> =
                jit_with_options(|(), x| Ok::<_, Error>(x), &session, Options::new(mesh.clone()));
            for output in [
                identity(extent, Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]))
                    .unwrap()
                    .executable()
                    .execute(&session, zero.clone())
                    .unwrap(),
                identity(extent, Options::new(mesh.clone()).with_output_shardings(vec![sharded.clone()]))
                    .unwrap()
                    .executable()
                    .execute(&session, zero.clone())
                    .unwrap(),
                dispatcher.execute((), zero.clone()).unwrap(),
            ] {
                output.block_until_ready().unwrap();
                assert_eq!(output.r#type().as_ref(), &zero_type.clone().with_sharding(replicated.clone()).unwrap());
                assert_eq!(output.sharding(), &sharded);
                assert!(output.is_committed());
                assert_eq!(shapes(&output), expected_shapes);
                assert!(output.shards().iter().all(|shard| shard.buffer().is_none()));
            }
        }
    }

    #[test]
    fn test_session_lower_program_ignores_placements_in_dormant_regions() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let vector_type = replicated_vector_type(&mesh, 2);
        let other_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("y", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![mesh.devices()[0]],
        )
        .unwrap();
        let other_target =
            Placement::new(other_mesh.clone(), Sharding::replicated(other_mesh.logical_mesh().clone(), 1)).unwrap();

        // The transpose region of a `linear_call` is a dormant rule that lowering never emits, so a placement there on
        // the devices and the topology of another mesh neither fails the validation of placements nor contributes its
        // mesh to the module.
        let build_region = |place: bool| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let residual = builder.add_input(vector_type.clone().into());
            let linear = builder.add_input(vector_type.clone().into());
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![residual, linear], None).unwrap()[0];
            if place {
                builder
                    .add_instruction(PlaceOperation::new(other_target.clone(), None), Vec::new(), vec![linear], None)
                    .unwrap();
            }
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        let (forward, transpose) = (build_region(false), build_region(true));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(vector_type.clone().into());
        let linear = builder.add_input(vector_type.clone().into());
        let output = builder
            .add_instruction(
                Operation::LinearCall(LinearCallOperation::new(1)),
                vec![forward, transpose],
                vec![residual, linear],
                None,
            )
            .unwrap()[0];
        let program: FlatProgram<Session> =
            builder.build(vec![output], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session
            .execute_program(
                &compiled,
                vec![f32_vector(&session, &mesh, &[2.0, 3.0]), f32_vector(&session, &mesh, &[4.0, 5.0])],
            )
            .unwrap();
        assert_eq!(outputs[0].copy_to_host().unwrap().elements::<f32>().unwrap(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_validate_placements_follows_the_regions_that_are_lowered() {
        // A named composite call whose body places its input on a device outside the mesh of the program is rejected
        // where lowering emits that body, and accepted on a target whose backend implementation replaces the whole
        // call (`ryft.tests.xla_named_square` version 1 on CUDA), which never lowers the body.
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![Device::new(0, 0)]).unwrap();
        let elsewhere = Device::new(1, 0);
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let target = PlacementRequest::from(elsewhere);
            let output =
                builder.add_instruction(PlaceOperation::new(target, None), Vec::new(), vec![input], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let key = CompositeKey::new("ryft.tests.xla_named_square".to_string(), 1);
        let named = NamedComposite::new(key, |types: &[Type]| Ok(types.to_vec()), |inputs| Ok(vec![inputs[0].clone()]));
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::new("placing").with_named_composite(named));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector_type.into());
        let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
        let regions = vec![builder.intern_callee(&body, None).unwrap()];
        let output = builder.add_instruction(operation, regions, vec![input], None).unwrap()[0];
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder]);
        let program = program.unwrap();
        let validate = |target_platform| {
            validate_placements(&program, &mesh, true, target_platform).map_err(|error| error.to_string())
        };
        assert_eq!(validate(Some("cuda")), Ok(()));
        assert_eq!(
            validate(Some("cpu")),
            Err(Error::Lowering(LoweringError::DeviceOutsideProgramMesh { device: elsewhere, mesh: mesh.clone() })
                .to_string()),
        );
        assert_eq!(validate(None), validate(Some("cpu")));
    }

    #[test]
    fn test_validate_placements_rejects_device_targets_outside_the_program_mesh() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2]);
        let place_error = |device: Device, mesh: &DeviceMesh| {
            compile::<_, ArrayType, ArrayType>(
                |input: CompileTracer| session.place(input, PlacementOptions::to(device)).unwrap(),
                input_type.clone(),
                &session,
                mesh.clone(),
            )
            .err()
            .unwrap()
        };

        // A device that is not the device of the mesh of the function is rejected.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let error = place_error(devices[1], &mesh);
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::DeviceOutsideProgramMesh { device, mesh: program_mesh })
                if device == &devices[1] && program_mesh == &mesh,
        ));
        assert_eq!(
            error.to_string(),
            format!(
                "`place` cannot move a value to the device {} inside a program compiled for the devices [{}] of mesh \
                 `{}`",
                devices[1].id(),
                devices[0].id(),
                mesh.logical_mesh(),
            ),
        );

        // So is a device of a mesh with more than one device, because the value is not placed on that device alone.
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let error = place_error(devices[0], &mesh);
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::DeviceOutsideProgramMesh { device, mesh: program_mesh })
                if device == &devices[0] && program_mesh == &mesh,
        ));
        assert_eq!(
            error.to_string(),
            format!(
                "`place` cannot move a value to the device {} inside a program compiled for the devices [{}, {}] of \
                 mesh `{}`",
                devices[0].id(),
                devices[0].id(),
                devices[1].id(),
                mesh.logical_mesh(),
            ),
        );

        // Only targets are validated, so the same device as the source of a value without a tracked sharding compiles
        // on a mesh of two devices, with or without a target, and the function returns its input on that mesh.
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let input =
            Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2.]))
                .unwrap();
        let cases = [
            (
                Some(PlacementRequest::from(Memory::Device)),
                indoc! {"
                    lambda %0:f32[2] .
                    let %1:f32[2] = place [source=device(0), target=in_place(memory=Device)] %0
                    in (%1)"
                },
            ),
            (
                None,
                indoc! {"
                    lambda %0:f32[2] .
                    let %1:f32[2] = place [source=device(0)] %0
                    in (%1)"
                },
            ),
        ];
        for (target, rendering) in cases {
            let source = Some(PlacementRequest::from(devices[0]));
            let options = PlacementOptions::<PlacementRequest>::new(target, source, None, None);
            let compiled = compile::<_, ArrayType, ArrayType>(
                |input: CompileTracer| session.place(input, options.clone()).unwrap(),
                input_type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            assert_eq!(compiled.source_program().program().to_string(), rendering);
            let output = compiled.executable().execute(&session, input.clone()).unwrap();
            assert_eq!(output.mesh(), &mesh);
            assert!(output.is_committed());
            assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2.]);
        }
    }

    #[test]
    fn test_validate_placements_skips_the_bodies_of_scans_without_iterations() {
        // A scan whose body places each slice on a device outside the mesh of the program is rejected when its length
        // admits an iteration, and accepted on every target when its length is zero, because lowering then emits
        // nothing of its body (refer to `omits_regions`).
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![Device::new(0, 0)]).unwrap();
        let elsewhere = Device::new(1, 0);
        let program = |length: usize| {
            let mut body = ProgramBuilder::<Constant, Operation>::new();
            body.add_input(ArrayType::scalar(DataType::I64).into());
            let slice = body.add_input(ArrayType::new_static(DataType::F32, [4]).into());
            let target = PlacementRequest::from(elsewhere);
            let placed =
                body.add_instruction(PlaceOperation::new(target, None), Vec::new(), vec![slice], None).unwrap()[0];
            let body: FlatProgram<Session> = body.build(vec![placed], vec![Placeholder; 2], vec![Placeholder]).unwrap();
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let body = builder.import_program(body);
            let stack = builder.add_input(ArrayType::new_static(DataType::F32, [length, 4]).into());
            let scan = ScanOperation::<Type>::new(0, length);
            let output = builder.add_instruction(scan, vec![body], vec![stack], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let validate = |length, target_platform| {
            validate_placements(&program(length), &mesh, true, target_platform).map_err(|error| error.to_string())
        };
        let rejected =
            Err(Error::Lowering(LoweringError::DeviceOutsideProgramMesh { device: elsewhere, mesh: mesh.clone() })
                .to_string());
        assert_eq!(validate(0, Some("cpu")), Ok(()));
        assert_eq!(validate(0, None), Ok(()));
        assert_eq!(validate(1, Some("cpu")), rejected);
        assert_eq!(validate(1, None), rejected);
    }

    #[test]
    fn test_has_only_static_array_types_follows_the_regions_that_are_lowered() {
        // A bounded intermediate inside the body of a named composite call selects the replica path without Shardy
        // where lowering emits that body, and not on a target whose backend implementation replaces the whole call
        // (`ryft.tests.xla_named_square` version 1 on CUDA), which never lowers the body. Before, such a body disabled
        // Shardy on every target, which then rejected a valid split boundary elsewhere in the program. The `shard_map`
        // check and the declaration of the module mesh read the same traversal (refer to `emitted_regions`).
        let size_type = ArrayType::scalar(DataType::I64);
        let value_type = ArrayType::scalar(DataType::F32);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let size = builder.add_input(size_type.clone().into());
            let value = builder.add_input(value_type.clone().into());
            let dimension = builder
                .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
                .unwrap()[0];
            let broadcast = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, dimension], None)
                .unwrap()[0];
            let output = builder
                .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![broadcast], None)
                .unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap())
        };
        let key = CompositeKey::new("ryft.tests.xla_named_square".to_string(), 1);
        let named =
            NamedComposite::new(key, |types: &[Type]| Ok(vec![types[1].clone()]), |inputs| Ok(vec![inputs[1].clone()]));
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::new("broadcasting").with_named_composite(named));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let size = builder.add_input(size_type.into());
        let value = builder.add_input(value_type.into());
        let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
        let regions = vec![builder.intern_callee(&body, None).unwrap()];
        let output = builder.add_instruction(operation, regions, vec![size, value], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert!(has_only_static_array_types(&program, Some("cuda")));
        assert!(!has_only_static_array_types(&program, Some("cpu")));
        assert!(!has_only_static_array_types(&program, None));
    }

    #[test]
    fn test_validate_placements_admits_sources_off_the_program_mesh() {
        // A source states where the caller expects the input to live, and the forward program moves nothing to it, so
        // a source on devices outside the mesh of the program, of an ordinary value or of one that varies along a
        // manual axis, and a split source of a bounded dynamic program are admitted. Before, all three were rejected
        // because transposition would return the cotangent to them.
        let manual = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let explicit = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let program = |sharding: Sharding, source_devices: Vec<Device>| {
            let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
            let source_mesh = DeviceMesh::new(sharding.mesh().clone(), source_devices).unwrap();
            let source = Placement::new(source_mesh, sharding).unwrap();
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(input_type.into());
            let operation = PlaceOperation::new(None, source);
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let (first, second) = (vec![Device::new(0, 0), Device::new(1, 0)], vec![Device::new(2, 0), Device::new(3, 0)]);
        let replicated = Sharding::replicated(manual.clone(), 1);
        let varying = replicated.clone().with_varying_manual_axes(["x"]).unwrap();
        let split = Sharding::new(explicit.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        for (sharding, source_devices, logical_mesh, use_shardy_partitioner) in [
            (replicated, second.clone(), manual.clone(), true),
            (varying, second, manual, true),
            (split, first.clone(), explicit.clone(), false),
        ] {
            let mesh = DeviceMesh::new(logical_mesh, first.clone()).unwrap();
            let program = program(sharding, source_devices);
            assert_eq!(
                validate_placements(&program, &mesh, use_shardy_partitioner, None).map_err(|error| error.to_string()),
                Ok(()),
            );
        }
    }

    #[test]
    fn test_session_compile_place_with_sources_on_the_program_mesh() {
        // A device source on the sole device of the program mesh, and a resolved source over the program mesh, are
        // accepted in the forward function and in its reverse mode derivative inside the same compiled function, whose
        // transposed placement moves the cotangent onto those devices.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let upload = |values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(values)).unwrap()
        };
        let cases = [
            (
                PlacementRequest::from(devices[1]),
                indoc! {"
                    lambda %0:f32[4] .
                    let %1:f32[4] = place [source=device(1), target=in_place(memory=Device)] %0
                    in (%1)"
                },
                indoc! {"
                    lambda %0:f32[4], %1:f32[4] .
                    let %2:f32[4] = place [target=device(1, memory=Device), may_alias=false] %1
                    in (%2)"
                },
            ),
            (
                PlacementRequest::from(Placement::new(mesh.clone(), Sharding::replicated(logical_mesh, 1)).unwrap()),
                indoc! {"
                    lambda %0:f32[4] .
                    let %1:f32[4] = place [
                        source={mesh<['a'=1:auto]>, devices=[1], sharding={mesh<['a'=1:auto]>, [{}]}},
                        target=in_place(memory=Device),
                    ] %0
                    in (%1)"
                },
                indoc! {"
                    lambda %0:f32[4], %1:f32[4] .
                    let %2:f32[4][sharding={mesh<['a'=1:auto]>, [{}]}] = place [
                        target={mesh<['a'=1:auto]>, devices=[1], sharding={mesh<['a'=1:auto]>, [{}]}, memory=Device},
                        may_alias=false,
                    ] %1
                        %3:f32[4] = broadcast [output_type=f32[4], output_axes=[0]] %2
                    in (%3)"
                },
            ),
        ];
        for (source, forward_rendering, reverse_rendering) in cases {
            let options = || PlacementOptions::to(PlacementRequest::from(Memory::Device)).with_source(source.clone());
            let forward = compile::<_, ArrayType, ArrayType>(
                |x: CompileTracer| session.place(x, options()).unwrap(),
                r#type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            assert_eq!(forward.source_program().program().to_string(), forward_rendering);
            let output = forward.executable().execute(&session, upload(&[1., 2., 3., 4.])).unwrap();
            assert_eq!(output.mesh(), &mesh);
            assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2., 3., 4.]);

            let reverse = compile::<_, (ArrayType, ArrayType), ArrayType>(
                |(x, seed): (CompileTracer, CompileTracer)| {
                    let (_, pullback) = differentiate_at(x).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
                    pullback.apply_materialized(seed).unwrap()
                },
                (r#type.clone(), r#type.clone()),
                &session,
                mesh.clone(),
            )
            .unwrap();
            assert_eq!(reverse.source_program().program().to_string(), reverse_rendering);
            let cotangent = reverse
                .executable()
                .execute(&session, (upload(&[1., 2., 3., 4.]), upload(&[5., 6., 7., 8.])))
                .unwrap();
            assert_eq!(cotangent.mesh(), &mesh);
            assert_eq!(cotangent.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![5., 6., 7., 8.]);
        }
    }

    #[test]
    fn test_session_compile_place_with_sources_off_the_program_mesh() {
        // A targetless `place` whose source names devices outside the mesh that a function is compiled for leaves the
        // input where it is, so the compiled function returns what the eager placement returns (devices, commitment,
        // and values), as JAX compiles `device_put(x, src=...)` inside `jit` for sources on any devices. Before, it
        // was rejected because transposition would return the cotangent to the source. Its reverse mode program holds
        // the transposed placement, whose target is that source, so it is still rejected as a target outside the
        // program mesh, and a source that does not match the input type is rejected by the type rule on both paths.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let elsewhere = DeviceMesh::new(mesh.logical_mesh().clone(), devices[2..4].to_vec()).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let bytes = values_to_bytes(&[1f32, 2., 3., 4.]);
        let committed_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated.clone()).unwrap();
        let committed = Array::from_host_buffer(&session, committed_type, mesh.clone(), None, &bytes).unwrap();
        let uncommitted =
            Array::from_host_buffer(&session, ArrayType::new_static(DataType::F32, [4]), None, None, &bytes).unwrap();
        let observe = |array: &Array| {
            let values = array.copy_to_host().unwrap().elements::<f32>().unwrap();
            (array.mesh().devices().to_vec(), array.is_committed(), values)
        };

        // Returns the eager result and the compiled result of a placement of `input` without a target and with
        // `source`, compiled for the mesh of `input`.
        let options = |source: &PlacementRequest| PlacementOptions::default().with_source(source.clone());
        let place = |input: &Array, source: &PlacementRequest| {
            let options = || options(source);
            let eager = session.place(input.clone(), options()).map(|output| observe(&output));
            let error = RefCell::new(None);
            let compiled = compile::<_, ArrayType, ArrayType>(
                |x: CompileTracer| {
                    session.place(x.clone(), options()).unwrap_or_else(|failure| {
                        error.replace(Some(failure));
                        x
                    })
                },
                input.r#type().into_owned(),
                &session,
                input.mesh().clone(),
            );
            let compiled = match (error.take(), compiled) {
                (Some(error), _) | (None, Err(error)) => Err(error.to_string()),
                (None, Ok(function)) => function
                    .executable()
                    .execute(&session, input.clone())
                    .map(|output| observe(&output))
                    .map_err(|error| error.to_string()),
            };
            (eager.map_err(|error| error.to_string()), compiled)
        };

        // Returns the error of compiling the reverse mode program of the same placement for the mesh of `input`.
        let reverse = |input: &Array, source: &PlacementRequest| {
            compile::<_, (ArrayType, ArrayType), ArrayType>(
                |(x, seed): (CompileTracer, CompileTracer)| {
                    let (_, pullback) =
                        differentiate_at(x).vjp(|x| Ok(session.place(x, options(source)).unwrap())).unwrap();
                    pullback.apply_materialized(seed).unwrap()
                },
                (input.r#type().into_owned(), input.r#type().into_owned()),
                &session,
                input.mesh().clone(),
            )
            .err()
            .map(|error| error.to_string())
        };

        let source = PlacementRequest::from(Placement::new(elsewhere.clone(), replicated).unwrap());
        let (eager, compiled) = place(&committed, &source);
        assert_eq!(eager, Ok((devices[..2].to_vec(), true, vec![1., 2., 3., 4.])));
        assert_eq!(compiled, eager);
        let rejection = LoweringError::PlacementOutsideProgramMesh { target: elsewhere.clone(), mesh };
        assert_eq!(reverse(&committed, &source), Some(Error::Lowering(rejection).to_string()));

        let source = PlacementRequest::from(devices[3]);
        let (eager, compiled) = place(&uncommitted, &source);
        assert_eq!(eager, Ok((vec![devices[0]], false, vec![1., 2., 3., 4.])));
        assert_eq!(compiled, eager);
        // The untracked input leaves the transposed placement's source as a bare device request.
        let rejection =
            LoweringError::DeviceOutsideProgramMesh { device: devices[3], mesh: uncommitted.mesh().clone() };
        assert_eq!(reverse(&uncommitted, &source), Some(Error::Lowering(rejection).to_string()));

        let (eager, compiled) =
            place(&committed, &PlacementRequest::from(Placement::new(elsewhere, split.clone()).unwrap()));
        let mismatch = format!(
            "`place` source sharding `{split}` does not match the input type `{}`",
            committed.r#type().into_owned(),
        );
        assert_eq!((eager, compiled), (Err(mismatch.clone()), Err(mismatch)));
    }

    #[test]
    fn test_validate_placements_ignores_inferred_sources() {
        // Linearizing a placement of an array on the first device infers that device as the source to which
        // transposition returns the cotangent, which constrains nothing forward, so the pushforward compiles and runs
        // on the second device, where its tangent lives, with and without a target there. Before, both were rejected
        // there. A source that the caller gives constrains the forward program no more (refer to
        // `test_validate_placements_admits_sources_off_the_program_mesh`).
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let second = DeviceMesh::new(LogicalMesh::single_device(), vec![devices[1]]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let upload = |values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), None, None, values_to_bytes(values)).unwrap()
        };
        let primal = upload(&[1.0, 2.0]);
        let tangent = upload(&[3.0, 4.0]).place(&devices[1], None).unwrap();
        for target in [None, Some(PlacementRequest::from(devices[1]))] {
            let (_, pushforward) = session
                .differentiate_at(Value::Array(primal.clone()))
                .linearize(|input| {
                    let input = ValueProjection::<ArrayType>::into_projected(input)?;
                    Ok(input.place(&target, None)?.into_value())
                })
                .unwrap();
            let program = pushforward.program();
            assert_eq!(validate_placements(program, &second, true, None).map_err(|error| error.to_string()), Ok(()));
            let lowered = session.lower_program(program, 0, &Options::new(second.clone())).unwrap();
            let compiled = session.compile_program(&lowered).unwrap();
            let outputs = CompilationDomain::execute(&session, &compiled, vec![Value::Array(tangent.clone())]).unwrap();
            let [Value::Array(output)] = outputs.as_slice() else {
                panic!("expected one array output");
            };
            let values = output.copy_to_host().unwrap().elements::<f32>().unwrap();
            assert_eq!((output.mesh().devices().to_vec(), values), (vec![devices[1]], vec![3.0, 4.0]));
        }
    }

    #[test]
    fn test_session_lower_program_ignores_dead_placements() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![devices[0]]).unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let predicate_type = replicated_scalar_type(&mesh, DataType::Boolean);

        // A `place` that no result uses is not part of the program that lowering emits, as JAX removes a dead
        // `device_put` before it looks at shardings, whether a branch drops its result or the condition returns it as
        // an output that the program does not use. It neither fails the validation of placements onto another device
        // nor commits the results, and the program agrees with the analysis of its placement requests.
        for returns_placement in [false, true] {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(scalar_type.clone().into());
            let placed = builder
                .add_instruction(
                    PlaceOperation::new(PlacementRequest::from(devices[1]), None),
                    Vec::new(),
                    vec![input],
                    None,
                )
                .unwrap()[0];
            let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
            let (outputs, output_count) =
                if returns_placement { (vec![negated, placed], 2) } else { (vec![negated], 1) };
            let branch = builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; output_count])
                .unwrap();
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(scalar_type.clone().into());
            let predicate = builder.add_input(predicate_type.clone().into());
            let true_region = builder.import_region(branch.entry_region_ref());
            let false_region = builder.import_region(branch.entry_region_ref());
            let output = builder
                .add_instruction(
                    ConditionOperation::new(),
                    vec![true_region, false_region],
                    vec![predicate, input],
                    None,
                )
                .unwrap()[0];
            let program: FlatProgram<Session> =
                builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap();
            assert!(!program.has_placement_requests(None, true));
            let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            assert_eq!(lowered.committed_outputs(), [false]);
            let compiled = session.compile_program(&lowered).unwrap();
            let outputs = session
                .execute_program(
                    &compiled,
                    vec![f32_scalar(&session, &mesh, 2.0), boolean_scalar(&session, &mesh, true)],
                )
                .unwrap();
            assert_eq!(outputs[0].copy_to_host().unwrap().elements::<f32>().unwrap(), vec![-2.0]);
        }
    }

    #[test]
    fn test_session_lower_program_ignores_dynamic_types_in_dormant_regions() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let vector_type = replicated_vector_type(&mesh, 2);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(4)).unwrap());

        // The transpose region of a `linear_call` is a dormant rule that lowering never emits, so a bounded dynamic
        // broadcast and sum there leaves the Shardy partitioner enabled for the static split of the primal.
        let build_region = |dynamic: bool| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let residual = builder.add_input(vector_type.clone().into());
            let linear = builder.add_input(vector_type.clone().into());
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![residual, linear], None).unwrap()[0];
            if dynamic {
                let size = CoreArray::from_elements(replicated_scalar_type(&mesh, DataType::I64), &[2i64]).unwrap();
                let size =
                    builder.add_instruction(ConstantOperation::new(size), Vec::new(), Vec::new(), None).unwrap()[0];
                let dimension = builder
                    .add_instruction(
                        DimensionFromScalarOperation::new(extent.clone().into()),
                        Vec::new(),
                        vec![size],
                        None,
                    )
                    .unwrap()[0];
                let sum = builder
                    .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![linear], None)
                    .unwrap()[0];
                let broadcast = builder
                    .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![sum, dimension], None)
                    .unwrap()[0];
                builder
                    .add_instruction(
                        ReduceOperation::new(vec![0], ReductionKind::Sum),
                        Vec::new(),
                        vec![broadcast],
                        None,
                    )
                    .unwrap();
            }
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        for dynamic in [false, true] {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let forward = builder.import_region(build_region(false).entry_region_ref());
            let transpose = builder.import_region(build_region(dynamic).entry_region_ref());
            let residual = builder.add_input(vector_type.clone().into());
            let linear = builder.add_input(vector_type.clone().into());
            let output = builder
                .add_instruction(
                    Operation::LinearCall(LinearCallOperation::new(1)),
                    vec![forward, transpose],
                    vec![residual, linear],
                    None,
                )
                .unwrap()[0];
            let output = builder
                .add_instruction(PlaceOperation::new(target.clone(), None), Vec::new(), vec![output], None)
                .unwrap()[0];
            let program: FlatProgram<Session> =
                builder.build(vec![output], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap();
            let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            let executable_options = lowered.compilation_options().executable_build_options.as_ref().unwrap();
            assert!(executable_options.use_shardy_partitioner);
            let compiled = session.compile_program(&lowered).unwrap();
            let outputs = session
                .execute_program(
                    &compiled,
                    vec![f32_vector(&session, &mesh, &[2.0, 3.0]), f32_vector(&session, &mesh, &[4.0, 5.0])],
                )
                .unwrap();
            assert_eq!(outputs[0].sharding(), &sharded);
            assert_eq!(outputs[0].copy_to_host().unwrap().elements::<f32>().unwrap(), vec![8.0, 15.0]);
        }
    }

    #[test]
    fn test_session_lower_program_uses_the_replica_path_for_internal_dynamic_tensors() {
        let client = cpu_client(4);
        let mesh = domain_mesh(client, "x", 4);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let value_type = replicated_scalar_type(&mesh, DataType::F32);

        // The boundary of this program is fully static, but its interior derives a dynamic extent from a runtime
        // scalar, broadcasts to it, and reduces back to a scalar.
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let size = builder.add_input(size_type.clone().into());
        let value = builder.add_input(value_type.clone().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let broadcast = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, dimension], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![broadcast], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let executable_options = lowered.compilation_options().executable_build_options.as_ref().unwrap();
        assert_eq!(executable_options.replica_count, 4);
        assert_eq!(executable_options.partition_count, 1);
        assert!(!executable_options.use_spmd_partitioning);
        assert!(!executable_options.use_shardy_partitioner);
        assert!(!lowered.stable_hlo().contains("sdy."), "{}", lowered.stable_hlo());

        let compiled = session.compile_program(&lowered).unwrap();
        let size =
            Array::from_host_buffer(&session, size_type.clone(), mesh.clone(), None, 3i64.to_ne_bytes().as_slice())
                .unwrap();
        let value = Array::from_host_buffer(
            &session,
            value_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let outputs = session.execute_program(&compiled, vec![size, value]).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![6.0]);

        // The replica path is a hard requirement rather than a preference, because annotating the very same program
        // for Shardy and compiling it with the Shardy partitioner enabled fails on the internal dynamic tensor, even
        // though every boundary type is static.
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let annotated = crate::lowering::modules::lower_mlir_module_for_program(
            &program,
            &[],
            &vec![size_type, value_type.clone()],
            &vec![value_type],
            "main",
            Some(&[replicated.clone(), replicated.clone()]),
            Some(std::slice::from_ref(&replicated)),
            None,
        )
        .unwrap();
        let mut shardy_options = jit_compilation_options(session.compilation_options(), mesh.devices(), true, None);
        resolve_shardy_debug_options(client, &mut shardy_options);
        let shardy_lowered =
            lowered.clone().with_stable_hlo(annotated.into_parts().0).with_compilation_options(shardy_options);
        let Err(error) = session.compile_program(&shardy_lowered) else {
            panic!("Shardy unexpectedly accepted a module containing an internal dynamic tensor");
        };
        assert!(error.to_string().contains("only supports ranked tensors with a static shape"), "{error}");
    }

    #[cfg(feature = "mps")]
    #[test]
    fn test_session_lower_program_omits_shardy_for_the_mps_plugin() {
        let client = mps_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let vector_type = replicated_vector_type(&mesh, 2);

        // The MPS plugin does not register the Shardy dialect, so the module of this program has neither its
        // constraint, nor the mesh, nor the shardings of the signature, and it is compiled without the Shardy
        // partitioner. The plugin does not report the layouts of its executables either, so the input is passed to
        // the execution as it is.
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector_type.clone().into());
        let constrained = builder
            .add_instruction(
                ConstrainShardingOperation::new(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let output = builder.add_instruction(NegOperation::new(), Vec::new(), vec![constrained], None).unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let executable_options = lowered.compilation_options().executable_build_options.as_ref().unwrap();
        assert_eq!(executable_options.replica_count, 1);
        assert_eq!(executable_options.partition_count, 1);
        assert!(!executable_options.use_spmd_partitioning);
        assert!(!executable_options.use_shardy_partitioner);
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.negate %arg0 : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "},
        );

        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session.execute_program(&compiled, vec![f32_vector(&session, &mesh, &[2.0, -3.0])]).unwrap();
        assert_eq!(outputs[0].copy_to_host().unwrap().elements::<f32>().unwrap(), vec![-2.0, 3.0]);
    }

    #[test]
    fn test_session_lower_program_rejects_unsupported_external_reference_storage_classes() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let replicated = |rank| Sharding::replicated(mesh.logical_mesh().clone(), rank);

        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(replicated(1))
            .unwrap();
        assert!(matches!(
            session.lower_program(
                &external_reference_read_program(zero_type),
                0,
                &Options::new(mesh.clone()),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference state at logical input 0 cannot use a zero-space type because it has \
                              no executable buffer",
        ));

        let host_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(replicated(1))
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        assert!(matches!(
            session.lower_program(
                &external_reference_read_program(host_type),
                0,
                &Options::new(mesh.clone()),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference state at logical input 0 must use device memory",
        ));

        let unbounded = DimensionVariable::new("unbounded", DimensionBounds::unbounded());
        let unbounded_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(unbounded)]))
            .with_sharding(replicated(1))
            .unwrap();
        assert!(matches!(
            session.lower_program(
                &external_reference_read_program(unbounded_type),
                0,
                &Options::new(mesh.clone()),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference state at logical input 0 has an unbounded dynamic dimension",
        ));

        // Types record placement over explicit mesh axes only, so the sharded state type uses an explicit axis.
        let explicit_logical_mesh =
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let explicit_mesh = DeviceMesh::new(explicit_logical_mesh.clone(), mesh.devices().to_vec()).unwrap();
        let bounded = DimensionVariable::new("bounded", DimensionBounds::new(1, Some(5)).unwrap());
        let sharded_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(bounded)]))
            .with_sharding(Sharding::new(explicit_logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert!(matches!(
            session.lower_program(&external_reference_read_program(sharded_type), 0, &Options::new(explicit_mesh)),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "bounded-dynamic external state input 0 must use replicated sharding",
        ));

        let foreign_logical_mesh =
            LogicalMesh::new(vec![MeshAxis::new("foreign", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let foreign_sharding_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(foreign_logical_mesh.clone(), 0))
            .unwrap();
        assert!(matches!(
            session.lower_program(
                &external_reference_read_program(foreign_sharding_type),
                0,
                &Options::new(mesh.clone()),
            ),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Sharding(ShardingError::MeshMismatch { expected, actual })
                    if expected == mesh.logical_mesh() && actual == &foreign_logical_mesh,
            ),
        ));

        let local_device = &mesh.devices()[0];
        let foreign_mesh = DeviceMesh::new(
            mesh.logical_mesh().clone(),
            vec![Device::new(local_device.id(), local_device.process_index() + 1)],
        )
        .unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(foreign_mesh.logical_mesh().clone(), 0))
            .unwrap();
        // A mesh that misattributes the devices of this process is rejected before the reference ABI is considered
        // (refer to `test_session_lower_program_rejects_misattributed_mesh_devices`).
        let (device_id, process_index) = (local_device.id(), local_device.process_index());
        assert_eq!(
            session
                .lower_program(&external_reference_read_program(scalar_type), 0, &Options::new(foreign_mesh))
                .unwrap_err()
                .to_string(),
            format!("device {device_id} reports process index {process_index}, but expected {}", process_index + 1),
        );

        let non_addressable_mesh = DeviceMesh::new(
            mesh.logical_mesh().clone(),
            vec![Device::new(local_device.id() + 1_000, local_device.process_index())],
        )
        .unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(non_addressable_mesh.logical_mesh().clone(), 0))
            .unwrap();
        assert_eq!(
            session
                .lower_program(&external_reference_read_program(scalar_type), 0, &Options::new(non_addressable_mesh))
                .unwrap_err()
                .to_string(),
            format!("device {} is not addressable by the PJRT client in process {process_index}", device_id + 1_000),
        );
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(scalar_type.clone().into());
        let replacement = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, input], None)
            .unwrap();
        let final_value =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program: FlatProgram<Session> = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![snapshot, final_value],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.output_types(), &[scalar_type.clone(), scalar_type]);
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> ",
            "(tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) {",
        );
        let expected_stable_hlo = indoc! {r#"
            module {
              sdy.mesh @mesh = <["x"=1]>
            @SIGNATURE@
                %0 = stablehlo.add %arg1, %arg0 : tensor<f32>
                return %arg1, %0 : tensor<f32>, tensor<f32>
              }
            }
        "#}
        .replace("@SIGNATURE@", expected_signature);
        assert_eq!(lowered.stable_hlo(), expected_stable_hlo,);
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session
            .execute_program(&compiled, vec![f32_scalar(&session, &mesh, 3.0), f32_scalar(&session, &mesh, 5.0)])
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![5.0]);
        assert_eq!(read_first_shard::<f32>(client, &outputs[1]), vec![8.0]);

        let eager_input = ArrayIrValue::Array(CoreArray::scalar(3.0f32).unwrap());
        let eager_replacement = ArrayIrValue::Array(CoreArray::scalar(5.0f32).unwrap());
        let eager_reference = eager_input.reference_new().unwrap();
        eager_reference.write(&eager_replacement).unwrap();
        let eager_snapshot = eager_reference.read().unwrap();
        eager_reference.add_update(&eager_input).unwrap();
        let eager_final = eager_reference.freeze().unwrap();
        assert_eq!(eager_snapshot, ArrayIrValue::Array(CoreArray::scalar(5.0f32).unwrap()));
        assert_eq!(eager_final, ArrayIrValue::Array(CoreArray::scalar(8.0f32).unwrap()));
    }

    #[test]
    fn test_session_lower_program_discharges_pinned_host_reference_state_in_shard_maps() {
        // A pinned host allocation that a `shard_map` mutates through a sharded reference input discharges to a manual
        // computation over its state. Every device updates its own shard, the final state keeps the pinned host
        // placement of the allocation, and the mapped output stays in device memory.
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
        let sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let device_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let host_type = device_type.clone().with_memory(Memory::Host { pinned: true });
        let boundary = ShardMapBoundary::new(
            logical_mesh,
            vec![sharding.clone(), sharding.clone()],
            vec![sharding],
            vec!["x".to_string()],
        )
        .unwrap();
        let local_type = boundary.local_input_type(1, &device_type).unwrap();
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let reference = builder.add_input(ReferenceType::new(local_type.clone()).into());
            let update = builder.add_input(local_type.into());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![state], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let operation = ShardMapOperation::from_program(
            &body,
            vec![ReferenceType::new(host_type.clone()).into(), device_type.clone().into()],
            boundary,
        )
        .unwrap();
        let program = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let initial = builder.add_input(host_type.clone().into());
            let update = builder.add_input(device_type.clone().into());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
            let body = builder.import_program(body);
            let mapped = builder
                .add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], vec![reference, update], None)
                .unwrap()[0];
            let state =
                builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![mapped, state], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let initial = Array::from_host_buffer(
            &session,
            host_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let update = Array::from_host_buffer(
            &session,
            device_type.clone(),
            mesh,
            None,
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0, 40.0]).as_slice(),
        )
        .unwrap();
        let outputs = session.execute_program(&compiled, vec![initial, update]).unwrap();
        let shards = |array: &Array| {
            array
                .addressable_shards()
                .map(|shard| {
                    let buffer = shard.buffer().unwrap();
                    let bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
                    (buffer.memory().unwrap().kind().unwrap().to_string(), values_from_bytes::<f32>(bytes.as_slice()))
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(outputs[0].r#type().as_ref(), &device_type);
        assert_eq!(
            shards(&outputs[0]),
            vec![("device".to_string(), vec![11.0, 22.0]), ("device".to_string(), vec![33.0, 44.0])],
        );
        assert_eq!(outputs[1].r#type().as_ref(), &host_type);
        assert_eq!(
            shards(&outputs[1]),
            vec![("pinned_host".to_string(), vec![11.0, 22.0]), ("pinned_host".to_string(), vec![33.0, 44.0])],
        );
    }

    #[test]
    fn test_session_lower_program_accumulates_slice_cotangents_into_existing_state() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(replicated_vector_type(&mesh, 4).into());
        let output = builder
            .add_instruction(SliceOperation::new(vec![1], vec![3]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(
            pullback.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["reference_add_update"],
        );

        // Discharge reads only the selected buffer entries, adds the seed, and writes that slice back. The lowered
        // graph contains no zero gradient of full size or dense addition before the indexed update.
        let lowered = session.lower_program(&pullback, 0, &Options::new(mesh)).unwrap();
        let operations = lowered
            .stable_hlo()
            .lines()
            .filter_map(|line| line.split_once(" = stablehlo."))
            .map(|(_, operation)| operation.split_whitespace().next().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(operations, vec!["slice", "add", "constant", "dynamic_update_slice"]);
    }

    #[test]
    fn test_session_lower_program_discharges_local_index_and_slice_reference_transforms() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 4);
        let pair_type = replicated_vector_type(&mesh, 2);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let initial = builder.add_input(vector_type.into());
        let replacement = builder.add_input(pair_type.clone().into());
        let update = builder.add_input(pair_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let indexed_snapshot = builder
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(3),
                }]),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let transforms = vec![
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 3, 1)] },
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] },
        ];
        let old = builder
            .add_instruction(
                ReferenceSwapOperation::new().with_transforms(transforms.clone()),
                Vec::new(),
                vec![reference, replacement],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                ReferenceAddUpdateOperation::new().with_transforms(transforms),
                Vec::new(),
                vec![reference, update],
                None,
            )
            .unwrap();
        let final_snapshot =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![indexed_snapshot, old, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, ",
            "%arg1: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, ",
            "%arg2: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> ",
            "(tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, ",
            "tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {",
        );
        let expected_update = |result, target, update, index, target_size, update_size| {
            format!(
                "    %{result} = stablehlo.dynamic_update_slice %{target}, %{update}, %{index} : \
                 (tensor<{target_size}xf32>, tensor<{update_size}xf32>, tensor<i64>) -> \
                 tensor<{target_size}xf32>",
            )
        };
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                @SIGNATURE@
                    %0 = stablehlo.slice %arg0 [3:4] : (tensor<4xf32>) -> tensor<1xf32>
                    %1 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
                    %2 = stablehlo.slice %arg0 [1:4] : (tensor<4xf32>) -> tensor<3xf32>
                    %3 = stablehlo.slice %2 [0:2] : (tensor<3xf32>) -> tensor<2xf32>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                @UPDATE0@
                    %c_0 = stablehlo.constant dense<1> : tensor<i64>
                @UPDATE1@
                    %6 = stablehlo.slice %5 [1:4] : (tensor<4xf32>) -> tensor<3xf32>
                    %7 = stablehlo.slice %6 [0:2] : (tensor<3xf32>) -> tensor<2xf32>
                    %8 = stablehlo.add %7, %arg2 : tensor<2xf32>
                    %c_1 = stablehlo.constant dense<0> : tensor<i64>
                @UPDATE2@
                    %c_2 = stablehlo.constant dense<1> : tensor<i64>
                @UPDATE3@
                    return %1, %3, %10 : tensor<f32>, tensor<2xf32>, tensor<4xf32>
                  }
                }
            "#}
            .replace("@SIGNATURE@", expected_signature)
            .replace("@UPDATE0@", &expected_update("4", "2", "arg1", "c", 3, 2))
            .replace("@UPDATE1@", &expected_update("5", "arg0", "4", "c_0", 4, 3))
            .replace("@UPDATE2@", &expected_update("9", "6", "8", "c_1", 3, 2))
            .replace("@UPDATE3@", &expected_update("10", "5", "9", "c_2", 4, 3)),
        );
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session
            .execute_program(
                &compiled,
                vec![
                    f32_vector(&session, &mesh, &[1.0, 2.0, 3.0, 4.0]),
                    f32_vector(&session, &mesh, &[10.0, 20.0]),
                    f32_vector(&session, &mesh, &[1.0, 2.0]),
                ],
            )
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![4.0]);
        assert_eq!(read_first_shard::<f32>(client, &outputs[1]), vec![2.0, 3.0]);
        assert_eq!(read_first_shard::<f32>(client, &outputs[2]), vec![1.0, 11.0, 22.0, 4.0]);
    }

    #[test]
    fn test_session_lower_program_discharges_writes_through_reference_views() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 4);
        let pair_type = replicated_vector_type(&mesh, 2);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let initial = builder.add_input(vector_type.into());
        let replacement = builder.add_input(pair_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(
                ReferenceWriteOperation::new()
                    .with_transforms(vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] }]),
                Vec::new(),
                vec![reference, replacement],
                None,
            )
            .unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program: FlatProgram<Session> = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session
            .execute_program(
                &compiled,
                vec![f32_vector(&session, &mesh, &[1.0, 2.0, 3.0, 4.0]), f32_vector(&session, &mesh, &[10.0, 20.0])],
            )
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![1.0, 10.0, 20.0, 4.0]);
    }

    #[test]
    fn test_session_lower_program_indexed_reference_mutation() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let initial = builder.add_input(vector_type.into());
        let replacement = builder.add_input(scalar_type.clone().into());
        let update = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) }];
        let old = builder
            .add_instruction(
                ReferenceSwapOperation::new().with_transforms(transforms.clone()),
                Vec::new(),
                vec![reference, replacement],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                ReferenceAddUpdateOperation::new().with_transforms(transforms),
                Vec::new(),
                vec![reference, update],
                None,
            )
            .unwrap();
        let final_snapshot =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![old, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, ",
            "%arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg2: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> ",
            "(tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {",
        );
        let expected_update = |result, target, update, index| {
            format!(
                "    %{result} = stablehlo.dynamic_update_slice %{target}, %{update}, %{index} : \
                 (tensor<3xf32>, tensor<1xf32>, tensor<i64>) -> tensor<3xf32>",
            )
        };
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                @SIGNATURE@
                    %0 = stablehlo.slice %arg0 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
                    %1 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
                    %2 = stablehlo.reshape %arg1 : (tensor<f32>) -> tensor<1xf32>
                    %c = stablehlo.constant dense<1> : tensor<i64>
                @UPDATE0@
                    %4 = stablehlo.slice %3 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
                    %5 = stablehlo.reshape %4 : (tensor<1xf32>) -> tensor<f32>
                    %6 = stablehlo.add %5, %arg2 : tensor<f32>
                    %7 = stablehlo.reshape %6 : (tensor<f32>) -> tensor<1xf32>
                    %c_0 = stablehlo.constant dense<1> : tensor<i64>
                @UPDATE1@
                    return %1, %8 : tensor<f32>, tensor<3xf32>
                  }
                }
            "#}
            .replace("@SIGNATURE@", expected_signature)
            .replace("@UPDATE0@", &expected_update("3", "arg0", "2", "c"))
            .replace("@UPDATE1@", &expected_update("8", "3", "7", "c_0")),
        );
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session
            .execute_program(
                &compiled,
                vec![
                    f32_vector(&session, &mesh, &[1.0, 2.0, 3.0]),
                    f32_scalar(&session, &mesh, 10.0),
                    f32_scalar(&session, &mesh, 3.0),
                ],
            )
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![2.0]);
        assert_eq!(read_first_shard::<f32>(client, &outputs[1]), vec![1.0, 13.0, 3.0]);
    }

    #[test]
    fn test_session_lower_program_preserves_capture_scope_inside_condition_regions() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let predicate_type = replicated_scalar_type(&mesh, DataType::Boolean);

        let branch = || {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let capture =
                builder.add_constant(Constant::Captured(CaptureReference::new(0, scalar_type.clone().into())));
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![capture], Vec::new(), vec![Placeholder]).unwrap()
        };
        let true_branch = branch();
        let false_branch = branch();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        builder.add_input(scalar_type.clone().into());
        let predicate = builder.add_input(predicate_type.into());
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![true_region, false_region], vec![predicate], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let lowered = session.lower_program(&program, 1, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session
            .execute_program(&compiled, vec![f32_scalar(&session, &mesh, 7.0), boolean_scalar(&session, &mesh, true)])
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![7.0]);
    }

    #[test]
    fn test_session_lower_program_validates_capture_count() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(scalar_type.into());
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            session.lower_program(&program, 2, &Options::new(mesh)),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidArgument { message })
                    if *message == "`capture_count` is 2, but the program has only 1 flat input(s)",
            ),
        ));
    }

    #[test]
    fn test_session_lower_program_keeps_pending_reductions_of_input_overrides() {
        // An input sharding override states the placement with which its argument arrives, a pending sum over an auto
        // axis included, so the boundary of the executable keeps it and the program can complete it (e.g., the reshard
        // that `Array::to_placement` compiles from such a source onto a sharding without unreduced axes). A runtime
        // placement with the same pending sum is dropped instead, because no value of the program can carry it, and
        // argument preparation then rejects an argument with that sum pending.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [1]);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let contributions = || {
            let buffers = client
                .addressable_devices()
                .unwrap()
                .into_iter()
                .zip([1.0f32, 2.0])
                .map(|(device, value)| {
                    client.buffer(value.to_ne_bytes().as_slice(), BufferType::F32, [1], None, device, None).unwrap()
                })
                .collect();
            Array::from_addressable_buffers(
                &session,
                vector_type.clone(),
                mesh.clone(),
                Some(unreduced.clone()),
                buffers,
            )
            .unwrap()
        };
        let options = Options::new(mesh.clone())
            .with_input_shardings(vec![unreduced.clone()])
            .with_output_shardings(vec![replicated.clone()]);
        let function: crate::CompiledFunction<ArrayType, ArrayType> =
            crate::compile_with_options(|x| x, vector_type.clone(), &session, options).unwrap();
        let output = function.executable().execute(&session, contributions()).unwrap();
        assert_eq!(output.sharding(), &replicated);
        assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![3.0]);

        // `Array::to_placement` and `Array::into_placement` take the compiled path for such a source, and each compiles
        // its reshard once.
        let placement = Placement::new(mesh.clone(), replicated.clone()).unwrap();
        for consumes in [false, true] {
            let mut program_counts = Vec::new();
            for _ in 0..2 {
                let output = match consumes {
                    false => contributions().to_placement(&session, placement.clone()),
                    true => contributions().into_placement(&session, placement.clone()),
                };
                assert_eq!(output.unwrap().copy_to_host().unwrap().elements::<f32>().unwrap(), vec![3.0]);
                program_counts.push(session.cached_program_count());
            }
            assert_eq!(program_counts[0], program_counts[1]);
        }
    }

    #[test]
    fn test_session_lower_program_uses_retained_array_sharding() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2f32]),
        )
        .unwrap();
        let retained = RetainedArray::new(&array);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let output = builder.add_constant(Constant::Session(retained.clone()));
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();

        // The hidden argument uses the retained runtime sharding without creating another logical array.
        let statistics = session.array_statistics();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh)).unwrap();
        assert_eq!(lowered.expected_argument_shardings(), &[sharding]);
        assert_eq!(session.array_statistics(), statistics);

        let foreign_client = other_cpu_client(2);
        let foreign_session = Session::builder().client(foreign_client).build().unwrap();
        let options = Options::new(domain_mesh(foreign_client, "x", 2));
        let expected = CoreError::Program(TypeError::invalid(format!(
            "session constant `{retained}` retains an array of a different PJRT client than the client of this session",
        ))
        .into());
        assert!(matches!(
            foreign_session.lower_program(&program, 0, &options),
            Err(Error::Core(error)) if *error == expected,
        ));
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state_through_condition() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let true_branch = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let replacement = builder.add_input(scalar_type.clone().into());
            let snapshot = builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let reference = builder.add_input(reference_type.into());
            builder.add_input(scalar_type.clone().into());
            let snapshot =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(replicated_scalar_type(&mesh, DataType::Boolean).into());
        let initial = builder.add_input(scalar_type.clone().into());
        let replacement = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<Type>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, replacement],
                None,
            )
            .unwrap()[0];
        let final_value =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![snapshot, final_value],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.output_types(), &[scalar_type.clone(), scalar_type]);
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<i1> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg2: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> ",
            "(tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) {",
        );
        let expected_stable_hlo = indoc! {r#"
            module {
              sdy.mesh @mesh = <["x"=1]>
            @SIGNATURE@
                %0:2 = "stablehlo.if"(%arg0) ({
                  stablehlo.return %arg1, %arg2 : tensor<f32>, tensor<f32>
                }, {
                  stablehlo.return %arg1, %arg1 : tensor<f32>, tensor<f32>
                }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)
                return %0#0, %0#1 : tensor<f32>, tensor<f32>
              }
            }
        "#}
        .replace("@SIGNATURE@", expected_signature);
        assert_eq!(lowered.stable_hlo(), expected_stable_hlo,);
        let compiled = session.compile_program(&lowered).unwrap();

        let true_outputs = session
            .execute_program(
                &compiled,
                vec![
                    boolean_scalar(&session, &mesh, true),
                    f32_scalar(&session, &mesh, 10.0),
                    f32_scalar(&session, &mesh, 7.0),
                ],
            )
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &true_outputs[0]), vec![10.0]);
        assert_eq!(read_first_shard::<f32>(client, &true_outputs[1]), vec![7.0]);

        let false_outputs = session
            .execute_program(
                &compiled,
                vec![
                    boolean_scalar(&session, &mesh, false),
                    f32_scalar(&session, &mesh, 10.0),
                    f32_scalar(&session, &mesh, 7.0),
                ],
            )
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &false_outputs[0]), vec![10.0]);
        assert_eq!(read_first_shard::<f32>(client, &false_outputs[1]), vec![10.0]);
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state_through_while() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let condition = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap();
            let predicate =
                CoreArray::from_elements(replicated_scalar_type(&mesh, DataType::Boolean), &[true]).unwrap();
            let predicate =
                builder.add_instruction(ConstantOperation::new(predicate), Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let reference = builder.add_input(reference_type.into());
            let update = builder
                .add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<Type>::new().with_iteration_bound(3).unwrap();
        let reference = builder.add_instruction(operation, vec![condition, body], vec![reference], None).unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.while").count(), 1, "{}", lowered.stable_hlo());
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session.execute_program(&compiled, vec![f32_scalar(&session, &mesh, 2.0)]).unwrap();

        // Three bounded iterations accumulate the constant `1.0` into the initial `2.0` state.
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![5.0]);
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state_through_static_scan() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let reference = builder.add_input(reference_type.into());
            let update = builder
                .add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let value =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![reference, value],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };

        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let scan_outputs = builder
            .add_instruction(ScanOperation::new(1, 3), vec![body], vec![reference], None)
            .unwrap()
            .to_vec();
        let final_value = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![scan_outputs[0]], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![final_value, scan_outputs[1]],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.while").count(), 1, "{}", lowered.stable_hlo());
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session.execute_program(&compiled, vec![f32_scalar(&session, &mesh, 2.0)]).unwrap();

        // Three scan iterations accumulate the constant `1.0` into the initial `2.0` state, and the body's second
        // output stacks the state observed after each iteration.
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![5.0]);
        assert_eq!(read_first_shard::<f32>(client, &outputs[1]), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state_through_dynamic_scan() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let length_variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let length = DimensionValue::new(DimensionType::from(length_variable.clone()), 3).unwrap();

        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let reference = builder.add_input(reference_type.into());
            let update = builder
                .add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![reference],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let runtime_length = builder.add_constant(Constant::Dimension(length.clone()));
        let reference = builder
            .add_instruction(
                ScanOperation::new(1, Dimension::Dynamic(length_variable.clone())),
                vec![body],
                vec![reference, runtime_length],
                None,
            )
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let scan_instruction = &program.entry_region_ref().instructions()[1];
        let Operation::Scan(scan) = scan_instruction.operation() else {
            panic!("expected dynamic scan operation");
        };
        assert_eq!(scan.carry_count(), 1);
        assert_eq!(scan.iteration_count(), &Dimension::Dynamic(length_variable));
        let runtime_length = *scan_instruction.inputs().last().unwrap();
        assert_eq!(
            program.entry_region_ref().atoms()[runtime_length.index()].as_constant(),
            Some(&Constant::Dimension(length)),
        );
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.while").count(), 1, "{}", lowered.stable_hlo());
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session.execute_program(&compiled, vec![f32_scalar(&session, &mesh, 2.0)]).unwrap();

        // The runtime length is three, so the scan accumulates the constant `1.0` into the initial `2.0` state three
        // times.
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![5.0]);
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state_through_nested_call() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let callee = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let reference = builder.add_input(reference_type.into());
            let update = builder.add_input(scalar_type.clone().into());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone().into());
        let update = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let reference = builder
            .add_instruction(Operation::Call(CallOperation::new(0)), vec![callee], vec![reference, update], None)
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            program.entry_region_ref().instructions()[1].operation(),
            Operation::Call(operation) if operation.capture_count() == 0,
        ));
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.add").count(), 1, "{}", lowered.stable_hlo());
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session
            .execute_program(&compiled, vec![f32_scalar(&session, &mesh, 2.0), f32_scalar(&session, &mesh, 4.0)])
            .unwrap();

        let immutable_oracle = 2.0f32 + 4.0;
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![immutable_oracle]);
    }

    #[test]
    fn test_session_lower_program_carries_external_reference_state_metadata() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();

        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let output = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program: FlatProgram<Session> = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh)).unwrap();
        assert_eq!(lowered.output_count(), 1);
        assert_eq!(lowered.reference_states().len(), 1);
        assert_eq!(lowered.reference_states()[0].source().flat_input_index(lowered.capture_count()).unwrap(), 0);
        assert_eq!(lowered.reference_states()[0].output_index(), None);
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> (tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) {
                    return %arg0 : tensor<f32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_session_lower_program_preserves_provenance_locations() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::F64);

        // Builds `-(x * x)` with the multiplication carrying the provided provenance.
        fn build_program(input_type: &ArrayType, provenance: Provenance) -> FlatProgram<Session> {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(input_type.clone().into());
            let product = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![input, input], Some(provenance))
                .unwrap()[0];
            let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
            builder.build(vec![negated], vec![Placeholder], vec![Placeholder]).unwrap()
        }

        let coordinate_basis = Provenance::scope(
            ProvenanceScope::new("ryft"),
            Provenance::scope(
                ProvenanceScope::new("differentiation"),
                Provenance::scope(ProvenanceScope::new("coordinate_basis"), Provenance::unknown()),
            ),
        );
        let scoped = build_program(&input_type, coordinate_basis);
        let options = Options::new(mesh.clone());
        let lowered = session.lower_program(&scoped, 0, &options).unwrap();

        // The nested framework scopes survive lowering as a chain of nested name locations that is visible in the
        // serialized module and therefore in dumped MLIR and HLO metadata, while the negation with unknown provenance
        // and the module scaffolding keep the base location.
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                #loc = loc(unknown)
                module {
                  sdy.mesh @mesh = <["x"=1]> loc(#loc)
                  func.func @main(%arg0: tensor<f64> {sdy.sharding = #sdy.sharding<@mesh, []>} loc(unknown)) -> (tensor<f64> {sdy.sharding = #sdy.sharding<@mesh, []>}) {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64> loc(#loc3)
                    %1 = stablehlo.negate %0 : tensor<f64> loc(#loc)
                    return %1 : tensor<f64> loc(#loc)
                  } loc(#loc)
                } loc(#loc)
                #loc1 = loc("coordinate_basis")
                #loc2 = loc("differentiation"(#loc1))
                #loc3 = loc("ryft"(#loc2))
            "#},
        );

        // Provenance does not affect numeric results.
        let compiled = session.compile_program(&lowered).unwrap();
        let value =
            Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, 3.0f64.to_ne_bytes().as_slice())
                .unwrap();
        let outputs = session.execute_program(&compiled, vec![value]).unwrap();
        assert_eq!(read_first_shard::<f64>(client, &outputs[0]), vec![-9.0]);

        // Two semantically equal programs with different scope names serialize differently and therefore key
        // differently, so neither can receive the other's labeled executable, while a program without provenance keeps
        // the serialization without locations, and hence the cache key that it had before provenance existed.
        let relabeled =
            build_program(&input_type, Provenance::scope(ProvenanceScope::new("other_label"), Provenance::unknown()));
        let relabeled = session.lower_program(&relabeled, 0, &options).unwrap();
        assert_ne!(lowered.compilation_key().unwrap(), relabeled.compilation_key().unwrap(),);
        let unlabeled = build_program(&input_type, Provenance::unknown());
        let unlabeled = session.lower_program(&unlabeled, 0, &options).unwrap();
        assert_eq!(
            unlabeled.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<f64> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> (tensor<f64> {sdy.sharding = #sdy.sharding<@mesh, []>}) {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %1 = stablehlo.negate %0 : tensor<f64>
                    return %1 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_session_lower_program_recreates_reference_views_inside_condition_branches() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(vector_type.clone());
        let true_branch = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let update =
                builder.add_instruction(OneOperation::new(scalar_type), Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .add_instruction(
                    ReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Static(1),
                    }]),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let reference = builder.add_input(reference_type.into());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(replicated_scalar_type(&mesh, DataType::Boolean).into());
        let initial = builder.add_input(vector_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let reference = builder
            .add_instruction(
                ConditionOperation::<Type>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<i1> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg1: tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> ",
            "(tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {",
        );
        let expected_update = concat!(
            "      %6 = stablehlo.dynamic_update_slice %arg1, %5, %c : ",
            "(tensor<3xf32>, tensor<1xf32>, tensor<i64>) -> tensor<3xf32>",
        );
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                @SIGNATURE@
                    %0 = "stablehlo.if"(%arg0) ({
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %1 = sdy.sharding_constraint %cst <@mesh, []> : tensor<f32>
                      %2 = stablehlo.slice %arg1 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
                      %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
                      %4 = stablehlo.add %3, %1 : tensor<f32>
                      %5 = stablehlo.reshape %4 : (tensor<f32>) -> tensor<1xf32>
                      %c = stablehlo.constant dense<1> : tensor<i64>
                @UPDATE@
                      stablehlo.return %6 : tensor<3xf32>
                    }, {
                      stablehlo.return %arg1 : tensor<3xf32>
                    }) : (tensor<i1>) -> tensor<3xf32>
                    return %0 : tensor<3xf32>
                  }
                }
            "#}
            .replace("@SIGNATURE@", expected_signature)
            .replace("@UPDATE@", expected_update),
        );
        let compiled = session.compile_program(&lowered).unwrap();
        for (predicate, expected) in [(true, vec![1.0, 3.0, 3.0]), (false, vec![1.0, 2.0, 3.0])] {
            let outputs = session
                .execute_program(
                    &compiled,
                    vec![boolean_scalar(&session, &mesh, predicate), f32_vector(&session, &mesh, &[1.0, 2.0, 3.0])],
                )
                .unwrap();
            assert_eq!(read_first_shard::<f32>(client, &outputs[0]), expected);
        }
    }

    #[test]
    fn test_session_compile_program_keeps_singleton_axes_of_open_outputs() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("e", 1, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let target_sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["e"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::I32, [4]).with_sharding(replicated.clone()).unwrap();
        let target = Placement::new(mesh.clone(), target_sharding).unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let placed =
            builder.add_instruction(PlaceOperation::new(target, None), Vec::new(), vec![input], None).unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![placed, placed], None).unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // The output is left open over the auto axis `a`, and the compiler reports its placement without the explicit
        // axis `e` of size one that the output type records, because that axis does not change how the output is
        // stored. The two placements are physically the same along `e`, so the output keeps `e` from its type.
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.open_outputs(), &[true]);
        let compiled = session.compile_program(&lowered).unwrap();
        assert_eq!(compiled.output_shardings()[0].to_string(), "{mesh<['e'=1:explicit, 'a'=2:auto]>, [{'e'}]}");
        let input =
            Array::from_host_buffer(&session, input_type, mesh, Some(replicated), values_to_bytes(&[1i32, 2, 3, 4]))
                .unwrap();
        let outputs = session.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(outputs[0].sharding(), &compiled.output_shardings()[0]);
        assert_eq!(
            outputs[0].shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
            vec!["[4]", "[4]"],
        );
        assert_eq!(outputs[0].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_session_reuse_cached_program_rejects_foreign_invocation_contracts() {
        // Exchange on the follower side that answers every key with one fixed artifact. This is the reachable
        // stand-in for a colliding or corrupted persistent entry, because the disk tier keys entries by the digest of
        // the canonical key bytes and re-validates that digest inside the stored envelope, so a planted disk entry
        // can only ever be rejected as corrupt, never resolved as a foreign executable.
        struct CollidingArtifactExchange {
            /// Serialized executable handed back for every requested key.
            artifact: Vec<u8>,
        }

        impl CompilationArtifactExchange for CollidingArtifactExchange {
            fn process_index(&self) -> usize {
                1
            }

            fn process_count(&self) -> usize {
                2
            }

            fn publish(&self, key: &[u8], artifact: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
                let _ = (key, artifact);
                Ok(())
            }

            fn receive(
                &self,
                key: &[u8],
                timeout: Duration,
            ) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError> {
                let _ = (key, timeout);
                Ok(Some(self.artifact.clone()))
            }
        }

        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        // The foreign executable keeps the requesting lowering's boundary types so that it loads and passes every
        // output type check in the core crate, and differs only in the donation contract the runtime call path trusts.
        let producer = Session::builder().client(client).build().unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |x| x.sin().unwrap(),
            input_type.clone(),
            Options::new(mesh.clone()).with_donate(true),
        )
        .unwrap()
        .into_inner();
        let foreign: CompiledFunction<Session, Type, Type> =
            staged.lower(&producer).unwrap().compile(&producer).unwrap();
        let artifact = producer.serialize_program(foreign.compiled_program()).unwrap().unwrap();

        let consumer = |policy| {
            let exchange: Arc<dyn CompilationArtifactExchange> =
                Arc::new(CollidingArtifactExchange { artifact: artifact.clone() });
            Session::builder()
                .client(client)
                .compilation_cache(CompilationCache::new().with_artifact_exchange(exchange, policy))
                .build()
                .unwrap()
        };
        let staged = crate::stage::<_, ArrayType, ArrayType>(|x| x.sin().unwrap(), input_type, Options::new(mesh))
            .unwrap()
            .into_inner();

        // The received executable loads cleanly and reports matching boundary types, so only the validation against
        // the requesting lowering catches the foreign invocation contract. The executable is then an incompatible
        // miss, so a follower that may fall back compiles its own executable.
        let fallback_consumer =
            consumer(CompilationArtifactExchangePolicy::PreferSharing { timeout: Duration::from_secs(5) });
        let lowered = staged.lower(&fallback_consumer).unwrap();
        let compiled = lowered.compile(&fallback_consumer).unwrap();
        assert!(
            fallback_consumer
                .reuse_cached_program(lowered.lowered_program(), Arc::new(compiled.compiled_program().clone()))
                .is_ok()
        );
        let statistics = fallback_consumer.compilation_cache().statistics();
        assert_eq!(statistics.exchange_hits, 0);
        assert_eq!(statistics.exchange_fallbacks, 1);
        assert_eq!(statistics.compilations, 1);

        // A follower that requires sharing fails instead.
        let required_consumer =
            consumer(CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(5) });
        assert!(matches!(
            staged.lower(&required_consumer).unwrap().compile(&required_consumer),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidArgument { message })
                    if *message == "compilation artifact exchange requirement was not satisfied: received compilation \
                        artifact is incompatible: incompatible XLA executable: cached executable has \
                        incompatible donation flags",
            ),
        ));
        assert_eq!(required_consumer.compilation_cache().statistics().compilations, 0);
    }

    #[test]
    fn test_session_reuse_cached_program_rejects_foreign_output_shardings() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let along_x = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let along_y = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["y"])]).unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [4]).into());
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let lower = |options: Options| session.lower_program(&program, 0, &options).unwrap();

        // Output types do not record placements along auto axes, so executables that place an output along `x` and
        // along `y`, which have the same size, return buffers of the same shapes and differ only in the placements
        // that their invocation contracts record for that output.
        let options = Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone()]);
        let along_x_lowered = lower(options.clone().with_output_shardings(vec![along_x.clone()]));
        let along_y_lowered = lower(options.with_output_shardings(vec![along_y]));
        let along_x_compiled = Arc::new(session.compile_program(&along_x_lowered).unwrap());
        let reused = session.reuse_cached_program(&along_x_lowered, Arc::clone(&along_x_compiled)).unwrap();
        assert!(Arc::ptr_eq(&reused, &along_x_compiled));
        assert!(matches!(
            session.reuse_cached_program(&along_y_lowered, along_x_compiled),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "cached executable has incompatible output shardings",
        ));

        // The compiler chooses the placement of an open output, so its placement in the executable may differ from
        // the one that the lowering declares.
        let open_lowered = lower(Options::new(mesh).with_input_shardings(vec![along_x.clone()]));
        assert_eq!(open_lowered.open_outputs(), &[true]);
        assert_eq!(open_lowered.output_shardings(), &[replicated]);
        let open_compiled = Arc::new(session.compile_program(&open_lowered).unwrap());
        assert_eq!(open_compiled.output_shardings(), &[along_x]);
        let reused = session.reuse_cached_program(&open_lowered, Arc::clone(&open_compiled)).unwrap();
        assert!(Arc::ptr_eq(&reused, &open_compiled));
    }

    #[test]
    fn test_placed_constant_places_captures_on_the_devices_that_the_function_names() {
        // An uncommitted capture that this process uploaded on its first device is placed on both devices of a
        // function compiled for them, and that placement, which the executable passes and exposes among its captures,
        // has the facts of an array that is placed on those devices directly: it is committed to them, so that an
        // executable on one of them rejects it, and every process knows its devices, so that a move to a device of
        // another process plans a transfer (which the CPU plugin cannot make) instead of returning no shards.
        let client = cpu_client(2);
        let session = session_with_remote_devices(client, [Device::new(99, 1)]);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let r#type = ArrayType::scalar(DataType::F32);
        let upload = |mesh: Option<DeviceMesh>| {
            Array::from_host_buffer(&session, r#type.clone(), mesh, None, 3f32.to_ne_bytes()).unwrap()
        };
        let capture = upload(None);
        assert!(!capture.is_committed());
        assert!(capture.is_locally_assigned());
        let function = crate::compile_statefully_with_captures::<_, Type, Type>(
            |captures, _| Ok(captures[0].clone()),
            vec![Value::Array(capture)],
            Type::Array(r#type.clone()),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let [Value::Array(placed)] = function.executable().captures() else {
            panic!("expected one captured array");
        };
        let direct = upload(Some(mesh.clone()));
        let first = DeviceMesh::new(LogicalMesh::single_device(), vec![mesh.devices()[0]]).unwrap();
        let identity =
            crate::compile::<_, ArrayType, ArrayType>(|input| input, r#type.clone(), &session, first.clone()).unwrap();
        let facts = |array: &Array| {
            let values = array.copy_to_host().unwrap().elements::<f32>().unwrap();
            let executed = identity.executable().execute(&session, array.clone()).map_err(|error| error.to_string());
            let moved =
                array.to_placement(&session, Device::new(99, 1)).map(|array| array.addressable_shards().count());
            (
                array.mesh().devices().to_vec(),
                array.is_committed(),
                array.is_locally_assigned(),
                values,
                executed.map(|_| ()),
                moved.map_err(|error| error.to_string()),
            )
        };
        assert_eq!(facts(placed), facts(&direct));
        assert_eq!(
            facts(placed),
            (
                mesh.devices().to_vec(),
                true,
                false,
                vec![3.0],
                Err("public argument #0 is committed to devices [0, 1], but the executable requires [0]".to_string()),
                Err("the cross-host transfers extension is not provided by the PJRT plugin".to_string()),
            ),
        );

        // A capture that already lives on the devices of the function is not copied, but it is handed out with the same
        // facts as a placement onto those devices, so that every process holds the same facts whether the capture
        // started on the devices of the function or elsewhere (e.g., the same capture without a target, on the first
        // device of the process that holds the function and on the device of another process).
        let capture = upload(None);
        let function = crate::compile_statefully_with_captures::<_, Type, Type>(
            |captures, _| Ok(captures[0].clone()),
            vec![Value::Array(capture.clone())],
            Type::Array(r#type.clone()),
            &session,
            Options::new(first.clone()),
        )
        .unwrap();
        let [Value::Array(found)] = function.executable().captures() else {
            panic!("expected one captured array");
        };
        assert!(Arc::ptr_eq(found.shards()[0].buffer().unwrap(), capture.shards()[0].buffer().unwrap()));
        assert_eq!(facts(found), facts(&upload(Some(first.clone()))));
        assert_eq!(
            facts(found),
            (
                vec![mesh.devices()[0]],
                true,
                false,
                vec![3.0],
                Ok(()),
                Err("the cross-host transfers extension is not provided by the PJRT plugin".to_string()),
            ),
        );
    }

    #[test]
    fn test_placed_constant_rejects_another_client_before_the_placement_attempt() {
        // A device names portable coordinates, so a session of another client with the same devices requests the
        // placement of a retention with the same mesh and sharding as its own client. That request submits nothing,
        // so it is rejected before it can consume the one placement attempt (refer to `RetainedArray::placed`) of
        // the client that owns the retention. Before, the owning client then failed to place its own capture.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let foreign_session = Session::builder().client(other_cpu_client(2)).build().unwrap();
        let r#type = ArrayType::scalar(DataType::F32);
        let upload = Array::from_host_buffer(&session, r#type.clone(), None, None, 3f32.to_ne_bytes()).unwrap();
        let retained = RetainedArray::new(&upload).with_declared_type(r#type);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("q", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh, vec![device]).unwrap();
        let expected = CoreError::Program(TypeError::invalid(format!(
            "session constant `{retained}` retains an array of a different PJRT client than the client of this session",
        ))
        .into());
        assert!(matches!(
            placed_constant(&foreign_session, &retained, &mesh),
            Err(Error::Core(error)) if *error == expected,
        ));
        let placed = placed_constant(&session, &retained, &mesh).unwrap().unwrap().to_array(&session).unwrap();
        let values = placed.copy_to_host().unwrap().elements::<f32>().unwrap();
        assert_eq!((placed.mesh().devices().to_vec(), placed.is_committed(), values), (vec![device], true, vec![3.0]));
    }

    #[test]
    fn test_placed_constant_validates_source_devices_independently_of_cached_placement() {
        let client = cpu_client(1);
        let narrow = Session::builder().client(client).build().unwrap();
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote = Device::new(local.id() + 1, local.process_index() + 1);
        let broad = session_with_remote_devices(client, [remote]);
        let source = DeviceMesh::new(LogicalMesh::single_device(), vec![remote]).unwrap();
        let target = DeviceMesh::new(LogicalMesh::single_device(), vec![local]).unwrap();
        let r#type = ArrayType::scalar(DataType::Zero);

        // A zero-space capture exercises the shared placement without requiring a remote transfer.
        let retain = || {
            let array = Array::from_host_buffer(&broad, r#type.clone(), source.clone(), None, []).unwrap();
            RetainedArray::new(&array)
        };
        let reject = |capture: &RetainedArray| {
            assert!(matches!(
                placed_constant(&narrow, capture, &target),
                Err(Error::Array(error)) if matches!(
                    *error,
                    ArrayError::UnknownDevice { device_id, process_index }
                        if device_id == remote.id() && process_index == remote.process_index(),
                ),
            ));
            let message =
                ArrayError::UnknownDevice { device_id: remote.id(), process_index: remote.process_index() }.to_string();
            let lift = narrow.with_device_mesh(target.clone()).lift(Constant::Session(capture.clone()));
            for error in [capture.to_array(&narrow).unwrap_err(), lift.unwrap_err()] {
                assert!(matches!(error, ProgramError::Custom(_)));
                assert_eq!(error.to_string(), message);
            }
        };
        let place = |capture: &RetainedArray| {
            let placed = placed_constant(&broad, capture, &target).unwrap().unwrap();
            assert_eq!(placed.mesh(), &target);
        };

        let narrow_first = retain();
        reject(&narrow_first);
        place(&narrow_first);
        let broad_first = retain();
        place(&broad_first);
        reject(&broad_first);
    }

    #[test]
    fn test_session_lower_staged_program_rejects_invalid_targets_before_placing_captures() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let capture_device = Device::from_pjrt(&devices[0]).unwrap();
        let target_device = Device::from_pjrt(&devices[1]).unwrap();
        let capture_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![capture_device]).unwrap();
        let target_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![target_device]).unwrap();
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let bytes = values_to_bytes(&[1f32, 2.]);
        let capture = Array::from_host_buffer(&session, array_type.clone(), capture_mesh, None, bytes).unwrap();
        let statistics = session.array_statistics();

        for _ in 0..2 {
            let attempted = crate::compile_with_captures::<_, ArrayType, ArrayType>(
                |captures, _| captures[0].place(&capture_device, None).unwrap(),
                vec![capture.clone()],
                array_type.clone(),
                &session,
                target_mesh.clone(),
            );
            assert!(matches!(
                attempted,
                Err(Error::Lowering(LoweringError::DeviceOutsideProgramMesh { device, mesh }))
                    if device == capture_device && mesh == target_mesh
            ));
            assert_eq!(session.array_statistics(), statistics);
        }
    }

    #[test]
    fn test_session_lower_staged_program_rejects_used_captures_of_another_client() {
        // A capture of another client that the function uses is rejected at compilation with the error with which a
        // session constant of another client is rejected (refer to
        // `test_placed_constant_rejects_another_client_before_the_placement_attempt`), whereas staging prunes one that
        // the function does not use, which therefore compiles. Before, a split capture of another client, which stays
        // an ordinary capture, compiled into an executable whose every execution failed with `CaptureClientMismatch`.
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let foreign_session = Session::builder().client(other_cpu_client(2)).build().unwrap();
        let foreign_mesh = two_device_mesh(foreign_session.client(), MeshAxisType::Explicit);
        let dimensions = vec![ShardingDimension::sharded(["x"])];
        let sharding = Sharding::new(foreign_mesh.logical_mesh().clone(), dimensions).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let bytes = values_to_bytes(&[1f32, 2f32]);
        let capture = Array::from_host_buffer(&foreign_session, r#type, foreign_mesh, Some(sharding), bytes).unwrap();
        let mesh = two_device_mesh(session.client(), MeshAxisType::Explicit);
        let input_type = Type::Array(ArrayType::scalar(DataType::F32));
        let used = compile_statefully_with_captures::<_, Type, Type>(
            |captures, _| Ok(captures[0].clone()),
            vec![Value::Array(capture.clone())],
            input_type.clone(),
            &session,
            Options::new(mesh.clone()),
        );
        let message = format!(
            "session constant `{}` retains an array of a different PJRT client than the client of this session",
            RetainedArray::new(&capture),
        );
        assert_eq!(used.err().map(|error| error.to_string()), Some(message));
        let unused = compile_statefully_with_captures::<_, Type, Type>(
            |_, input| Ok(input),
            vec![Value::Array(capture)],
            input_type,
            &session,
            Options::new(mesh),
        )
        .unwrap();
        assert!(unused.executable().captures().is_empty());
    }

    #[test]
    fn test_session_lower_staged_program_skips_captures_that_only_omitted_regions_read() {
        // A split capture that only the body of a scan imports is checked when the scan admits an iteration, and not
        // when its length is zero, because lowering then emits nothing of that body and the executable has no argument
        // for the capture (refer to `emitted_capture_reads`). This holds for a populated capture of another client,
        // whose admission rejects it, and for a capture of this client without buffers, which only an execution
        // rejects. At length zero both compile and run through the executable. It holds alike for a capture of another
        // client of the zero data type, which has no buffer either, and for a capture of this client that is split
        // unevenly, which the executable could not receive. Every entry point agrees: an execution, a call, the
        // interpretation rule of a call, an execution of a persisted copy, and an execution of a caller that is compiled
        // around a call, which passes the capture on to the callee and so reads it only where the callee reads it.
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let foreign_session = Session::builder().client(other_cpu_client(2)).build().unwrap();
        let mesh = two_device_mesh(session.client(), MeshAxisType::Explicit);
        let foreign_mesh = two_device_mesh(foreign_session.client(), MeshAxisType::Explicit);
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let split = |mesh: &DeviceMesh| {
            Some(Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
        };
        let foreign = Array::from_host_buffer(
            &foreign_session,
            r#type.clone(),
            foreign_mesh.clone(),
            split(&foreign_mesh),
            values_to_bytes(&[0f32; 4]),
        )
        .unwrap();
        let empty =
            Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), split(&mesh), Vec::new()).unwrap();
        // A populated capture of this client on the devices of the function in the opposite order, which the function
        // moves onto its mesh when it reads it, and which says nothing about where a call runs when it does not.
        let reversed_devices = mesh.devices().iter().rev().copied().collect();
        let reversed_mesh = DeviceMesh::new(mesh.logical_mesh().clone(), reversed_devices).unwrap();
        let reversed = Array::from_host_buffer(
            &session,
            r#type,
            reversed_mesh.clone(),
            split(&reversed_mesh),
            values_to_bytes(&[1f32, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        let zero_type = ArrayType::new_static(DataType::Zero, [4]);
        let zero = Array::from_host_buffer(&foreign_session, zero_type, foreign_mesh.clone(), split(&foreign_mesh), [])
            .unwrap();
        let uneven_type = ArrayType::new_static(DataType::F32, [3]);
        let uneven = Array::from_host_buffer(
            &session,
            uneven_type,
            mesh.clone(),
            split(&mesh),
            values_to_bytes(&[1f32, 2.0, 3.0]),
        )
        .unwrap();
        let compiled = |capture: &Array, length: usize| {
            let extent = capture.r#type().shape().dimensions()[0].value().unwrap();
            let is_zero = capture.data_type().is_zero();
            let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
            let input_type = ArrayType::new_static(DataType::F32, [length, extent]);
            let function = crate::compile_with_options::<_, ArrayType, ArrayType>(
                |stack: crate::CompileTracer| {
                    let scanned = scan(
                        |_, (), slice| {
                            let capture = ImportingContext::<Session, ThroughCapture>::import(
                                slice.context(),
                                Value::Array(capture.clone()),
                            )?;
                            // A capture of the zero data type is read as the output of the body.
                            if is_zero {
                                return Ok(((), capture));
                            }
                            // A capture that is split unevenly is replicated where it is read, so that only the
                            // capture and not the output of the function is split unevenly.
                            let capture = if extent % 2 == 1 { capture.reshard(&replicated)? } else { capture };
                            let slice = ValueProjection::<ArrayType>::into_projected(slice)?;
                            let capture = ValueProjection::<ArrayType>::into_projected(capture)?;
                            Ok(((), (slice + capture).into_value()))
                        },
                        (),
                        stack.into_value(),
                        ScanOptions::new(),
                    );
                    ValueProjection::<ArrayType>::into_projected(scanned.unwrap().1).unwrap()
                },
                input_type.clone(),
                &session,
                Options::new(mesh.clone()),
            )
            .map_err(|error| error.to_string())?;
            let bytes = values_to_bytes(&vec![0f32; extent * length]);
            let input = Array::from_host_buffer(&session, input_type, mesh.clone(), None, bytes).unwrap();
            Ok::<_, String>((function, input))
        };
        let read = |output: Array| {
            let values = values_from_bytes::<f32>(&output.copy_to_host_bytes().unwrap());
            (output.r#type().shape().clone(), values)
        };
        let executed = |capture: &Array, length: usize| {
            let (function, input) = compiled(capture, length)?;
            function.executable().execute(&session, input).map(read).map_err(|error| error.to_string())
        };
        let called = |capture: &Array, length: usize| {
            let (function, input) = compiled(capture, length)?;
            function.call(input).map(read).map_err(|error| error.to_string())
        };
        // The interpretation rule of the call that a caller stages.
        let interpreted = |capture: &Array, length: usize| {
            let (function, input) = compiled(capture, length)?;
            let caller: crate::StagedFunction<ArrayType, ArrayType> =
                crate::stage(|x| function.call(x).unwrap(), input.r#type().into_owned(), Options::new(mesh.clone()))
                    .unwrap();
            let program = caller.source_program().to_program_with_lifted_captures().unwrap();
            let mut inputs = caller.source_program().captures().to_vec();
            inputs.push(Value::Array(input));
            let call = program
                .instructions()
                .iter()
                .find(|instruction| matches!(instruction.operation(), Operation::Call(_)))
                .unwrap();
            let Operation::Call(operation) = call.operation() else { unreachable!("selected a `call`") };
            let callee = Arc::new(program.entry_region_ref().with_id(call.regions()[0]).unwrap().to_program());
            let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
            operation
                .interpret(&session, &EagerInterpretationDriver::new(&regions), &inputs)
                .map(|mut outputs| read(ValueProjection::<ArrayType>::into_projected(outputs.remove(0)).unwrap()))
                .map_err(|error| error.to_string())
        };
        // An execution of a copy of the executable that is persisted and loaded again on its own.
        let persisted = |capture: &Array, length: usize| {
            let (function, input) = compiled(capture, length)?;
            let lowered = function.staged().into_inner().lower(&session).map_err(|error| error.to_string())?;
            let compiled = lowered.compile(&session).map_err(|error| error.to_string())?;
            let bytes = session.serialize_program(compiled.compiled_program()).unwrap().unwrap();
            let restored = session.deserialize_program(&bytes).unwrap().unwrap();
            let mut inputs = function.source_program().captures().to_vec();
            inputs.push(Value::Array(input));
            session
                .execute(&restored, inputs)
                .map(|mut outputs| read(ValueProjection::<ArrayType>::into_projected(outputs.remove(0)).unwrap()))
                .map_err(|error| error.to_string())
        };
        // An execution of a caller that is compiled around a call of the function, whose program passes the captures
        // of the function on to the call.
        let caller = |capture: &Array, length: usize| {
            let (function, input) = compiled(capture, length)?;
            let caller = crate::compile_with_options::<_, ArrayType, ArrayType>(
                |x: crate::CompileTracer| function.call(x).unwrap(),
                input.r#type().into_owned(),
                &session,
                Options::new(mesh.clone()),
            )
            .map_err(|error| error.to_string())?;
            caller.executable().execute(&session, input).map(read).map_err(|error| error.to_string())
        };
        let foreign_message = format!(
            "session constant `{}` retains an array of a different PJRT client than the client of this session",
            RetainedArray::new(&foreign),
        );
        assert_eq!(executed(&foreign, 1), Err(foreign_message));
        assert_eq!(executed(&empty, 1), Err("input array #0 has no addressable shard for device 0".to_string()));
        let nothing = Ok((Shape::new(vec![Dimension::Static(0), Dimension::Static(4)]), Vec::new()));
        // A call requires of a capture what an execution requires of it (refer to `EagerCallDecision::unread_captures`),
        // so the two agree for every capture and length.
        let results = [executed(&foreign, 0), called(&foreign, 0), executed(&empty, 0), called(&empty, 0)];
        assert_eq!(results, [nothing.clone(), nothing.clone(), nothing.clone(), nothing.clone()]);
        assert_eq!([interpreted(&foreign, 0), interpreted(&empty, 0)], [nothing.clone(), nothing.clone()]);
        assert_eq!(called(&foreign, 1), executed(&foreign, 1));
        assert_eq!(called(&empty, 1), executed(&empty, 1));
        let results = [executed(&reversed, 0), called(&reversed, 0), interpreted(&reversed, 0)];
        assert_eq!(results, [nothing.clone(), nothing.clone(), nothing.clone()]);
        let read = Ok((Shape::new(vec![Dimension::Static(1), Dimension::Static(4)]), vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!([executed(&reversed, 1), called(&reversed, 1)], [read.clone(), read.clone()]);
        // A capture of another client of the zero data type and a capture of this client that is split unevenly are
        // rejected like the captures above when the function reads them, and require nothing when it does not. The
        // signature records which captures are unread rather than leaving execution to infer it from a missing buffer,
        // which a capture of the zero data type never has (refer to `ExecutableSignature::is_unread_input`), and a
        // capture is placed only once it is known to be read.
        let zero_message = format!(
            "session constant `{}` retains an array of a different PJRT client than the client of this session",
            RetainedArray::new(&zero),
        );
        let uneven_message = "the sharding of capture 0 splits dimension 0 of extent 3 into 2 shards, but XLA requires \
                              the number of shards to divide the extent";
        for (capture, message) in [(&zero, zero_message), (&uneven, uneven_message.to_string())] {
            assert_eq!([executed(capture, 1), called(capture, 1)], [Err(message.clone()), Err(message)]);
        }
        let nothing_of_three = Ok((Shape::new(vec![Dimension::Static(0), Dimension::Static(3)]), Vec::new()));
        for (capture, nothing) in [(&zero, &nothing), (&uneven, &nothing_of_three)] {
            let results = [executed(capture, 0), called(capture, 0), interpreted(capture, 0)];
            assert_eq!(results, [nothing.clone(), nothing.clone(), nothing.clone()]);
        }
        // A persisted copy keeps the record of the captures that its executable never reads.
        let unread = [(&foreign, &nothing), (&empty, &nothing), (&zero, &nothing), (&uneven, &nothing_of_three)];
        for (capture, nothing) in unread {
            assert_eq!(persisted(capture, 0), *nothing);
        }
        // A caller that passes a capture on to a call whose callee never reads it reads it neither, so its executable
        // has no argument for it and places nothing, and it agrees with the function at every length.
        for (capture, nothing) in unread.into_iter().chain([(&reversed, &nothing)]) {
            assert_eq!(caller(capture, 0), *nothing);
        }
        for capture in [&foreign, &empty, &zero, &uneven] {
            assert_eq!(caller(capture, 1), executed(capture, 1));
        }
        assert_eq!(caller(&reversed, 1), read);

        // A function that also calls a function whose own capture is numbered zero as well reads its own capture zero
        // only where its own regions read it, and the capture of its callee only through the operand of the call
        // (refer to `emitted_regions_entering_calls`). So a capture of another client that only a scan without
        // iterations reads stays unread when the callee reads its own capture zero, and a capture that the function
        // reads stays required when the callee never reads its own capture zero, at every entry point. Before, the
        // constant of the capture of the callee was counted as a read of the capture zero of the function, which
        // rejected the first function with the error of the second.
        let input_type = |length: usize| ArrayType::new_static(DataType::F32, [length, 4]);
        let scan_adding = |stack: crate::CompileTracer, capture: &Array| {
            let scanned = scan(
                |_, (), slice| {
                    let capture = ImportingContext::<Session, ThroughCapture>::import(
                        slice.context(),
                        Value::Array(capture.clone()),
                    )?;
                    let slice = ValueProjection::<ArrayType>::into_projected(slice)?;
                    let capture = ValueProjection::<ArrayType>::into_projected(capture)?;
                    Ok(((), (slice + capture).into_value()))
                },
                (),
                stack.into_value(),
                ScanOptions::new(),
            );
            ValueProjection::<ArrayType>::into_projected(scanned.unwrap().1).unwrap()
        };
        let read_array = |output: Array| {
            let values = values_from_bytes::<f32>(&output.copy_to_host_bytes().unwrap());
            (output.r#type().shape().clone(), values)
        };
        let read_pair = |(first, second): (Array, Array)| (read_array(first), read_array(second));
        let read_values = |mut outputs: Vec<Value>| {
            let second = ValueProjection::<ArrayType>::into_projected(outputs.remove(1)).unwrap();
            read_pair((ValueProjection::<ArrayType>::into_projected(outputs.remove(0)).unwrap(), second))
        };
        let nested = |outer: &Array, outer_length: usize, inner: &Array, inner_length: usize| {
            let callee = crate::compile_with_options::<_, ArrayType, ArrayType>(
                |stack: crate::CompileTracer| scan_adding(stack, inner),
                input_type(inner_length),
                &session,
                Options::new(mesh.clone()),
            )
            .map_err(|error| error.to_string())?;
            let types = (input_type(outer_length), input_type(inner_length));
            let function = crate::compile_with_options::<_, (ArrayType, ArrayType), (ArrayType, ArrayType)>(
                |(first, second): (crate::CompileTracer, crate::CompileTracer)| {
                    (scan_adding(first, outer), callee.call(second).unwrap())
                },
                types.clone(),
                &session,
                Options::new(mesh.clone()),
            )
            .map_err(|error| error.to_string())?;
            let input = |length: usize| {
                let bytes = values_to_bytes(&vec![0f32; 4 * length]);
                Array::from_host_buffer(&session, input_type(length), mesh.clone(), None, bytes).unwrap()
            };
            Ok::<_, String>((function, types, (input(outer_length), input(inner_length))))
        };
        // An execution, a call, the interpretation rule of a call, an execution of a persisted copy, and an execution of
        // a caller.
        let nested_results = |outer: &Array, outer_length: usize, inner: &Array, inner_length: usize| {
            let compiled = || nested(outer, outer_length, inner, inner_length);
            let executed = compiled().and_then(|(function, _, inputs)| {
                function.executable().execute(&session, inputs).map(read_pair).map_err(|error| error.to_string())
            });
            let called = compiled().and_then(|(function, _, inputs)| {
                function.call(inputs).map(read_pair).map_err(|error| error.to_string())
            });
            let interpreted = compiled().and_then(|(function, types, (first, second))| {
                let caller: crate::StagedFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
                    crate::stage(|inputs| function.call(inputs).unwrap(), types, Options::new(mesh.clone())).unwrap();
                let program = caller.source_program().to_program_with_lifted_captures().unwrap();
                let mut inputs = caller.source_program().captures().to_vec();
                inputs.extend([Value::Array(first), Value::Array(second)]);
                let call = program
                    .instructions()
                    .iter()
                    .find(|instruction| matches!(instruction.operation(), Operation::Call(_)))
                    .unwrap();
                let Operation::Call(operation) = call.operation() else { unreachable!("selected a `call`") };
                let callee = Arc::new(program.entry_region_ref().with_id(call.regions()[0]).unwrap().to_program());
                let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
                operation
                    .interpret(&session, &EagerInterpretationDriver::new(&regions), &inputs)
                    .map(read_values)
                    .map_err(|error| error.to_string())
            });
            let persisted = compiled().and_then(|(function, _, (first, second))| {
                let lowered = function.staged().into_inner().lower(&session).map_err(|error| error.to_string())?;
                let compiled = lowered.compile(&session).map_err(|error| error.to_string())?;
                let bytes = session.serialize_program(compiled.compiled_program()).unwrap().unwrap();
                let restored = session.deserialize_program(&bytes).unwrap().unwrap();
                let mut inputs = function.source_program().captures().to_vec();
                inputs.extend([Value::Array(first), Value::Array(second)]);
                session.execute(&restored, inputs).map(read_values).map_err(|error| error.to_string())
            });
            let caller = compiled().and_then(|(function, types, inputs)| {
                let caller = crate::compile_with_options::<_, (ArrayType, ArrayType), (ArrayType, ArrayType)>(
                    |inputs: (crate::CompileTracer, crate::CompileTracer)| function.call(inputs).unwrap(),
                    types,
                    &session,
                    Options::new(mesh.clone()),
                )
                .map_err(|error| error.to_string())?;
                caller.executable().execute(&session, inputs).map(read_pair).map_err(|error| error.to_string())
            });
            [executed, called, interpreted, persisted, caller]
        };
        let nothing = (Shape::new(vec![Dimension::Static(0), Dimension::Static(4)]), Vec::new());
        let read = (Shape::new(vec![Dimension::Static(1), Dimension::Static(4)]), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(nested_results(&foreign, 0, &reversed, 1), [(); 5].map(|()| Ok((nothing.clone(), read.clone()))));
        let foreign_message = format!(
            "session constant `{}` retains an array of a different PJRT client than the client of this session",
            RetainedArray::new(&foreign),
        );
        assert_eq!(nested_results(&foreign, 1, &empty, 0), [(); 5].map(|()| Err(foreign_message.clone())));
    }

    #[test]
    fn test_placed_constant_fills_missing_local_replicas_once() {
        // A replicated capture on the devices of a function that holds the buffer of only one of its two local replicas
        // (e.g., a partial import) is placed onto its own mesh and sharding once, which copies that replica, so that the
        // executable holds a fully populated capture and every execution creates only its output. Before, it was passed
        // as it was, and every execution copied the same replica again. A capture without elements is placed once as
        // well, as execution would otherwise create its missing empty buffer on every call.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let input_placement = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 0));
        let input = session.place(0f32, PlacementOptions::to(input_placement.unwrap())).unwrap();
        for values in [(0..8).map(|value| value as f32).collect::<Vec<_>>(), Vec::new()] {
            let device = client.addressable_devices().unwrap()[0].clone();
            let bytes = values_to_bytes(&values);
            let dimensions = [values.len() as u64];
            let buffer = client.buffer(bytes.as_slice(), BufferType::F32, dimensions, None, device, None).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [values.len()]);
            let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
            let capture =
                Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(sharding), vec![buffer]).unwrap();
            let function = compile_statefully_with_captures::<_, Type, Type>(
                |captures, _| Ok(captures[0].clone()),
                vec![Value::Array(capture.clone())],
                Type::Array(ArrayType::scalar(DataType::F32)),
                &session,
                Options::new(mesh.clone()),
            )
            .unwrap();
            let [Value::Array(placed)] = function.executable().captures() else {
                panic!("expected one captured array");
            };
            let held = placed.shards().iter().map(|shard| shard.buffer().is_some()).collect::<Vec<_>>();
            assert_eq!(held, [true, true]);
            // A held buffer with elements is kept, whereas a placement may give an empty one a new empty buffer.
            let held_buffer = capture.shards()[0].buffer().unwrap();
            let keeps_held_buffer = Arc::ptr_eq(placed.shards()[0].buffer().unwrap(), held_buffer);
            assert_eq!(keeps_held_buffer, !values.is_empty());
            let executions = (0..3)
                .map(|_| {
                    let statistics = session.array_statistics();
                    let output = function.execute_statefully_and_wait(&session, Value::Array(input.clone())).unwrap();
                    let created = session.array_statistics().created_arrays - statistics.created_arrays;
                    let Value::Array(output) = output else { panic!("expected an array output") };
                    (created, output.copy_to_host().unwrap().elements::<f32>().unwrap())
                })
                .collect::<Vec<_>>();
            assert_eq!(executions, vec![(1, values); 3]);
        }
    }

    #[test]
    fn test_placed_constant_never_waits_for_pending_work() {
        // A request consults its own placement before it looks for another source, and never waits for the work of
        // another copy, so two identical requests for a capture of a pending upload and a request under another axis
        // name all return before the upload is filled. Before, the second request waited for the copy of the first.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap()[0].clone();
        let specification = BufferSpecification::new(BufferType::F32, [1u64]);
        let manager = client.host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap());
        let manager = manager.unwrap();
        let origin_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let origin = DeviceMesh::new(LogicalMesh::single_device(), vec![origin_device]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [1]);
        let buffers = vec![manager.retrieve_buffer(0).unwrap()];
        let upload = Array::from_addressable_buffers(&session, r#type.clone(), origin, None, buffers).unwrap();
        let retained = RetainedArray::new(&upload).with_declared_type(r#type);
        let (returned_sender, returned) = mpsc::channel();
        let requests = std::thread::spawn({
            let session = session.clone();
            move || {
                for name in ["q", "q", "r"] {
                    let placed = placed_constant(&session, &retained, &domain_mesh(client, name, 1)).unwrap();
                    returned_sender.send(placed.unwrap()).unwrap();
                }
            }
        });
        let placed = (0..3).map(|_| returned.recv_timeout(Duration::from_secs(10)).ok()).collect::<Vec<_>>();
        manager.transfer_data(0, Arc::new(3f32.to_ne_bytes().to_vec()), 0, true).unwrap().r#await().unwrap();
        requests.join().unwrap();
        let placed = placed
            .into_iter()
            .map(|placed| placed.expect("a request waited for the pending upload"))
            .collect::<Vec<_>>();
        assert_eq!(placed[0], placed[1]);
        let values = placed
            .iter()
            .map(|placed| placed.to_array(&session).unwrap().copy_to_host().unwrap().elements::<f32>().unwrap());
        assert_eq!(values.collect::<Vec<_>>(), vec![vec![3.0]; 3]);
    }

    #[test]
    fn test_placed_constant_shares_one_copy_of_a_locally_assigned_capture_between_names() {
        // A capture with locally assigned devices is placed by every process with local work only, so its placements
        // cannot communicate, even onto devices of two processes, and placements onto meshes that differ only in their
        // axis names share one copy once it has completed. Before, each of eight names held a copy of its own. The
        // capture lives on the first device and the destination on the second, so that the copy is a buffer of its own.
        let client = cpu_client(2);
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let remote = Device::new(local.id() + 1, local.process_index() + 1);
        let session = session_with_remote_devices(client, [remote]);
        let r#type = ArrayType::scalar(DataType::F32);
        let upload = Array::from_host_buffer(&session, r#type.clone(), None, None, 3f32.to_ne_bytes()).unwrap();
        assert!(upload.is_locally_assigned());
        let retained = RetainedArray::new(&upload).with_declared_type(r#type);
        let mesh = |name: String| {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(name, 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            DeviceMesh::new(logical_mesh, vec![local, remote]).unwrap()
        };
        let place = |index: usize| {
            let placed = placed_constant(&session, &retained, &mesh(format!("n{index}"))).unwrap().unwrap();
            placed.to_array(&session).unwrap()
        };
        let buffer = |array: &Array| Arc::clone(array.addressable_shards().next().unwrap().buffer().unwrap());
        let target = mesh("n0".to_string());
        assert!(!retained.placement_communicates(&target, &Sharding::replicated(target.logical_mesh().clone(), 0)));
        let first = place(0);
        first.block_until_ready().unwrap();
        let buffers = (0..8).map(|index| buffer(&place(index))).collect::<Vec<_>>();
        assert!(buffers.iter().all(|placed| Arc::ptr_eq(placed, &buffer(&first))));
        assert!(!Arc::ptr_eq(&buffer(&first), &buffer(&upload)));
    }

    #[test]
    fn test_placed_constant_preserves_capture_completion_with_shared_buffers() {
        // Two captures share the buffer of one array but complete with different fences (e.g., a result that aliases an
        // existing array, refer to `Array::with_execution_fence`). Placing either one reuses the placed buffer of the
        // other while each placement observes the fence of its own capture.
        let client = cpu_client(3);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let device = |index: usize| Device::from_pjrt(&devices[index]).unwrap();
        let mesh = |index| DeviceMesh::new(LogicalMesh::single_device(), vec![device(index)]).unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let healthy =
            Array::from_host_buffer(&session, scalar_type.clone(), mesh(0), None, 4f32.to_ne_bytes()).unwrap();
        let (event, promise) = client.event(()).unwrap();
        promise.set(Some(PjrtError::aborted("capture producer failed"))).unwrap();
        let dependent = healthy.clone().with_execution_fence(ExecutionFence::new(vec![event]));
        let buffer = |array: &Array| Arc::clone(array.shards()[0].buffer().unwrap());
        assert!(Arc::ptr_eq(&buffer(&dependent), &buffer(&healthy)));

        let healthy = RetainedArray::new(&healthy).with_declared_type(scalar_type.clone());
        let dependent = RetainedArray::new(&dependent).with_declared_type(scalar_type);
        assert_ne!(healthy, dependent);
        let is_failed = |result: Result<(), Error>| {
            assert!(matches!(result, Err(Error::Pjrt(PjrtError::Aborted { message, .. }))
                if message == "capture producer failed"));
        };

        // Lifts share the physical placement in either request order while each lift observes its own fence.
        let lift = |capture: &RetainedArray, target: &DeviceMesh| {
            let Value::Array(array) =
                session.with_device_mesh(target.clone()).lift(Constant::Session(capture.clone())).unwrap()
            else {
                panic!("expected a lifted capture")
            };
            array
        };
        let warm_healthy = lift(&healthy, &mesh(1));
        warm_healthy.block_until_ready().unwrap();
        let warm_failed = lift(&dependent, &mesh(1));
        assert!(Arc::ptr_eq(&buffer(&warm_healthy), &buffer(&warm_failed)));
        is_failed(warm_failed.block_until_ready());

        let cold_failed = lift(&dependent, &mesh(2));
        is_failed(cold_failed.block_until_ready());
        let cold_healthy = lift(&healthy, &mesh(2));
        assert!(Arc::ptr_eq(&buffer(&cold_failed), &buffer(&cold_healthy)));
        assert_eq!(cold_healthy.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![4.0]);

        // A different mesh name misses the exact key but can reuse the completed copy's buffers.
        let renamed_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let renamed = DeviceMesh::new(renamed_mesh, vec![device(2)]).unwrap();
        let reused = lift(&dependent, &renamed);
        assert!(Arc::ptr_eq(&buffer(&cold_healthy), &buffer(&reused)));
        is_failed(reused.block_until_ready());

        // Compiled constants have separate hidden slots even when both captures share the same placed buffer.
        let execute = |target: DeviceMesh, captures: &[RetainedArray]| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let outputs = captures
                .iter()
                .map(|capture| builder.add_constant(Constant::Session(capture.clone())))
                .collect::<Vec<_>>();
            let program: FlatProgram<Session> =
                builder.build(outputs, Vec::new(), vec![Placeholder; captures.len()]).unwrap();
            let lowered = session.lower_program(&program, 0, &Options::new(target)).unwrap();
            assert_eq!(lowered.session_constants().len(), captures.len());
            let (program, captures) = CompilationDomain::compile(&session, &lowered).unwrap();
            let arguments = captures
                .into_iter()
                .map(|value| {
                    let Value::Array(array) = value else { panic!("expected an executable capture") };
                    array
                })
                .collect();
            session.execute_program(&program, arguments).and_then(|arrays| {
                arrays
                    .into_iter()
                    .map(|array| array.copy_to_host()?.elements::<f32>().map_err(Error::from))
                    .collect::<Result<Vec<_>, Error>>()
            })
        };
        assert_eq!(execute(mesh(1), std::slice::from_ref(&healthy)).unwrap(), vec![vec![4.0]]);
        is_failed(execute(mesh(1), &[healthy, dependent]).map(|_| ()));
    }

    #[test]
    fn test_placed_constant_rejects_misattributed_devices_before_the_placement_attempt() {
        // Like direct placement (refer to `test_array_to_placement_rejects_misattributed_target_devices`), this
        // rejects a target that assigns this process's device to another process or assigns an unaddressable device
        // to this process. Every process rejects the target before attempting placement. Previously, the process
        // that owns the device accepted it.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let (device_id, process_index) = (local.id(), local.process_index());
        let r#type = ArrayType::scalar(DataType::F32);
        let upload = Array::from_host_buffer(&session, r#type.clone(), None, None, 3f32.to_ne_bytes()).unwrap();
        let retained = RetainedArray::new(&upload).with_declared_type(r#type);
        let mesh = |device: Device| {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("q", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            DeviceMesh::new(logical_mesh, vec![device]).unwrap()
        };
        let error = |mesh: &DeviceMesh| placed_constant(&session, &retained, mesh).unwrap_err().to_string();
        let misattributed = mesh(Device::new(device_id, process_index + 1));
        let message =
            format!("device {device_id} reports process index {process_index}, but expected {}", process_index + 1);
        assert_eq!([error(&misattributed), error(&misattributed)], [message.clone(), message]);
        let missing = mesh(Device::new(device_id + 999, process_index));
        let message =
            format!("device {} is not addressable by the PJRT client in process {process_index}", device_id + 999);
        assert_eq!(error(&missing), message);
    }

    #[test]
    fn test_session_lower_program_rejects_misattributed_mesh_devices() {
        // A function compiled for a mesh that assigns the device of this process to another process, a device that
        // this process cannot address to this process, or a device of another process that the session does not know
        // (e.g., a mistyped identifier) is rejected before anything is lowered or compiled, as a placement of a capture
        // rejects it (refer to `test_placed_constant_rejects_misattributed_devices_before_the_placement_attempt`), so
        // every process rejects it alike, also for a function without captures. A session that declares that device
        // passes this check, and a CPU client, which lists only its own devices, then cannot assign the program to that
        // device. Before, it compiled, and an execution either failed for a missing shard or returned a
        // result on the misattributed mesh, and a device that the session does not know compiled as well.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let (device_id, process_index) = (local.id(), local.process_index());
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let program: FlatProgram<Session> = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let mesh = |device: Device| {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("q", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            DeviceMesh::new(logical_mesh, vec![device]).unwrap()
        };
        let error =
            |device: Device| session.lower_program(&program, 0, &Options::new(mesh(device))).unwrap_err().to_string();
        let unknown = Device::new(999, process_index + 1);
        assert_eq!(
            [
                error(Device::new(device_id, process_index + 1)),
                error(Device::new(device_id + 999, process_index)),
                error(unknown),
            ],
            [
                format!("device {device_id} reports process index {process_index}, but expected {}", process_index + 1),
                format!("device {} is not addressable by the PJRT client in process {process_index}", device_id + 999),
                format!(
                    "device 999 of process {} is not a device of the session; a session learns the devices of other \
                     processes that its PJRT client does not list from `SessionBuilder::distributed_runtime`",
                    process_index + 1,
                ),
            ],
        );
        let declared = session_with_remote_devices(client, [unknown]);
        assert!(matches!(
            declared.lower_program(&program, 0, &Options::new(mesh(unknown))),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "device 999 is not visible to the live client",
        ));
    }

    #[test]
    fn test_session_eager_dispatch_describes_remote_singleton_outputs() {
        let client = cpu_client(1);
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote = Device::new(999, local.process_index() + 1);
        let session = session_with_remote_devices(client, [remote]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("q", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(logical_mesh.clone(), 0);
        let mesh = DeviceMesh::new(logical_mesh, vec![remote]).unwrap();
        let input = Array::from_addressable_buffers(
            &session,
            ArrayType::scalar(DataType::F32),
            mesh.clone(),
            Some(sharding.clone()),
            Vec::new(),
        )
        .unwrap();

        let output = input.sin().unwrap();
        assert_eq!(output.r#type(), input.r#type());
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(output.shards().len(), 1);
        assert_eq!(output.addressable_shards().count(), 0);
        assert_eq!(session.compilation_cache().statistics().compilations, 0);
    }

    #[test]
    fn test_placed_constant_takes_the_layout_of_the_capture_whatever_the_index_holds() {
        // A capture of two column shards with byte strides `[8, 4]` lives on two devices of another process, which a
        // metadata-only import describes, because the CPU plugin cannot allocate strided buffers. A replicated copy on
        // another device pair drops the layout, which no longer describes its shards. A later placement as the
        // original split on that pair keeps the layout as a placement of the capture itself does, rather than dropping
        // it because it started from that copy. Before, it dropped it.
        let client = cpu_client(1);
        let session = session_with_remote_devices(client, (0..4).map(|device| Device::new(2048 + device, 1)));
        let mesh = |name: &str, devices: [usize; 2]| {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(name, 2, MeshAxisType::Auto).unwrap()]).unwrap();
            DeviceMesh::new(logical_mesh, devices.map(|device| Device::new(2048 + device, 1)).to_vec()).unwrap()
        };
        let source_mesh = mesh("a", [0, 1]);
        let dimensions = vec![ShardingDimension::replicated(), ShardingDimension::sharded(["a"])];
        let sharding = Sharding::new(source_mesh.logical_mesh().clone(), dimensions).unwrap();
        let layout = Layout::Strided(StridedLayout::new(vec![8, 4]));
        let r#type = ArrayType::new_static(DataType::F32, [4, 4]).with_layout(layout.clone());
        let placed_layout = |after_replicated_copy: bool| {
            let (r#type, source_mesh, sharding) = (r#type.clone(), source_mesh.clone(), Some(sharding.clone()));
            let capture = Array::from_addressable_buffers(&session, r#type, source_mesh, sharding, Vec::new()).unwrap();
            let retained = RetainedArray::new(&capture);
            let retained = match captured_type(&capture.r#type()) {
                Some(captured_type) => retained.with_declared_type(captured_type),
                None => retained,
            };
            if after_replicated_copy {
                let copy = placed_constant(&session, &retained, &mesh("b", [2, 3])).unwrap().unwrap();
                assert_eq!(copy.r#type().layout(), None);
            }
            let placed = placed_constant(&session, &retained, &mesh("a", [2, 3])).unwrap().unwrap();
            placed.r#type().layout().cloned()
        };
        assert_eq!([placed_layout(false), placed_layout(true)], [Some(layout.clone()), Some(layout)]);
    }

    #[test]
    fn test_jit_compilation_options() {
        let devices = [Device::new(3, 0), Device::new(1, 0)];
        let base =
            CompilationOptions { matrix_unit_operand_precision: Precision::Highest as i32, ..Default::default() };

        // Static programs compile as one replica that Shardy partitions over the devices, in the provided order.
        let options = jit_compilation_options(&base, &devices, true, None);
        assert_eq!(options.matrix_unit_operand_precision, Precision::Highest as i32);
        assert_eq!(
            options.executable_build_options,
            Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 2,
                use_spmd_partitioning: true,
                use_shardy_partitioner: true,
                device_assignment: Some(DeviceAssignment {
                    replica_count: 1,
                    computation_count: 2,
                    computation_devices: vec![
                        ComputationDeviceAssignment { replica_device_ids: vec![3] },
                        ComputationDeviceAssignment { replica_device_ids: vec![1] },
                    ],
                }),
                ..Default::default()
            }),
        );

        // Bounded dynamic programs compile as one computation with one replica per device.
        let options = jit_compilation_options(&base, &devices, false, None);
        assert_eq!(
            options.executable_build_options,
            Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 2,
                partition_count: 1,
                device_assignment: Some(DeviceAssignment {
                    replica_count: 2,
                    computation_count: 1,
                    computation_devices: vec![ComputationDeviceAssignment { replica_device_ids: vec![3, 1] }],
                }),
                ..Default::default()
            }),
        );

        // Portable executables are compiled for no particular device, so they are not pinned.
        let portable = CompilationOptions { compile_portable_executable: true, ..Default::default() };
        let options = jit_compilation_options(&portable, &devices[..1], true, None);
        assert_eq!(options.executable_build_options.unwrap().device_assignment, None);
    }
}
