//! Compilation lifecycle of the XLA [`Session`], which implements [`CompilationDomain`] and [`CompilationCacheDomain`].
//!
//! `Session::lower_staged_program` validates the boundary of a staged program against the sharding overrides of its
//! [`Options`]. `Session::lower_program` then selects kernels, discharges references, places every executable argument
//! and result, and emits the StableHLO module together with its invocation contract. `Session::compile_program`
//! compiles that module through the PJRT client of the session and derives the reusable execution plan.
//! [`with_input_shardings`] applies input sharding overrides to input types before staging.

use std::sync::Arc;
use std::time::Instant;

use ryft_core::{
    ArrayIrType, ArrayOperation, ArrayType, CompilationCacheDomain, CompilationDomain, Device, DispatchInputs,
    ExternalReferenceBinding, FlatProgram as CoreFlatProgram, Memory, MeshAxisType, Parameterized, ProgramError,
    ReferenceDischargeResult, Sharding, ShardingDimension, StagedProgram, Type, TypeError, Typed, TypedProgram,
};
use ryft_pjrt::protos::{
    CompilationOptions, ComputationDeviceAssignment, DeviceAssignment, ExecutableCompilationOptions, OpShardingType,
};
use ryft_pjrt::{LoadedExecutable, Program as PjrtProgram};

use crate::compilation::CompilationError;
use crate::compilation::keys::{CompilationKey, DispatchKey, dispatch_signature};
use crate::compilation::options::{FeedbackDirectedProfile, Options};
use crate::compilation::programs::{CompiledProgram, InvocationMetadata, LoweredProgram};
use crate::execution::ExecutionPlan;
use crate::lowering::admission::validate_data_dependent_compilation;
use crate::lowering::{collect_session_constants, contains_unresolved_references, contains_unresolved_state};
use crate::{Constant, Error, ExecutionError, FlatProgram, OpShardingExtension, Operation, Session, Value};

// TODO(eaplatanios): Review this module.

impl CompilationDomain for Session {
    type DispatchKey = DispatchKey;
    type LoweredProgram = LoweredProgram;
    type CompiledProgram = CompiledProgram;
    type Options = Options;
    type Error = Error;

    fn dispatch_signature(
        inputs: DispatchInputs<'_, '_, Self::Value>,
        options: &Self::Options,
    ) -> Result<(Self::DispatchKey, Arc<[ArrayIrType]>, Option<Self::Options>), Self::Error> {
        dispatch_signature(inputs, options)
    }

    fn lower(&self, staged: &StagedProgram<Self>) -> Result<LoweredProgram, Self::Error> {
        self.lower_staged_program(staged)
    }

    // The cached executable depends only on the types of the session constants of `program`, so their arrays are
    // resolved through this session's session for every request, including cache hits, as `SessionConstant` explains.
    fn compile(&self, program: &LoweredProgram) -> Result<(Arc<CompiledProgram>, Vec<Value>), Self::Error> {
        let session_arrays = program
            .session_constants()
            .iter()
            .map(|constant| Ok(Value::Array(self.retained_constant(constant)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        let compiled = self.compilation_cache().compile(self, program)?;

        // A program that was compiled for another lowering (e.g., an earlier one in this session or a persisted
        // executable) has the boundary of `program` up to the identities of its dimension variables, which the cache
        // validated. Callers compare the types of the result with those of `program` by identity, so the result takes
        // its boundary from `program` and shares everything else with the cached program.
        let compiled = CompiledProgram::rebound_to(compiled, program);
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
        let capture_count = staged.source_program().captures().len();
        let input_types = staged.input_types();
        let output_types = staged.output_types();
        let options = staged.options();
        if apply_input_shardings(input_types.to_vec(), options)? != input_types {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: "staged input types do not carry the shardings requested by `input_shardings`; apply them to \
                         the input types before staging (e.g., with `with_input_shardings`)"
                    .to_string(),
            }));
        }
        validate_reference_outputs(program.as_ref(), capture_count, output_types)?;
        validate_output_shardings(output_types, options)?;

        // Captured arrays are placed as they are at runtime, including over the auto mesh axes that their types do not
        // record, so that executions need not reshard them.
        let capture_placements = staged
            .source_program()
            .captures()
            .iter()
            .map(|capture| match capture {
                Value::Array(array) if array.sharding().references_auto_axis() => Some(array.sharding().clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let placed_options;
        let options = if capture_placements.iter().any(Option::is_some) {
            placed_options = options.clone().with_capture_placements(capture_placements.into());
            &placed_options
        } else {
            options
        };
        let lowered_program = self.lower_program(program.as_ref(), capture_count, options)?;
        let lowered_output_types = lowered_program.output_types().to_vec();
        // Lowering gives the program outputs the types of the output sharding overrides without their auto mesh axes,
        // so the staged output types are checked against the lowered ones with those same overrides applied.
        let declared_output_types =
            apply_signature_shardings(output_types.to_vec(), options.output_shardings(), "output")?;
        validate_output_types(&declared_output_types, &lowered_output_types)?;
        Ok(lowered_program)
    }

    /// Lowers a staged program to StableHLO and derives its complete logical and physical invocation contract for this
    /// session.
    pub(crate) fn lower_program(
        &self,
        program: &FlatProgram,
        capture_count: usize,
        options: &Options,
    ) -> Result<LoweredProgram, Error> {
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
            );
        }
        // Discharge the exact lifted program so capture references of attached regions resolve against the same lexical
        // capture prefixes used during staging. Programs that actually contain references are cloned because
        // discharge consumes and rewrites them; the reference-free identity path above remains borrowed.
        let discharged = program.clone().discharge_references_in_capture_lifted_program(capture_count)?;
        self.lower_discharged_program(&discharged, options)
    }

    /// Lowers one validated reference discharge artifact through the ordinary array-only XLA path.
    fn lower_discharged_program(
        &self,
        discharged: &ReferenceDischargeResult<Constant, Operation>,
        options: &Options,
    ) -> Result<LoweredProgram, Error> {
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
        )
    }

    /// Lowers one program after an independent verifier has established that it contains no references.
    fn lower_verified_reference_free_program(
        &self,
        program: &FlatProgram,
        capture_count: usize,
        output_count: usize,
        reference_states: &[ExternalReferenceBinding],
        options: &Options,
    ) -> Result<LoweredProgram, Error> {
        if contains_unresolved_state(program) {
            return Err(ProgramError::UnsupportedOperation {
                message: "discharged program still contains ordered state, which XLA lowering cannot represent"
                    .to_string(),
            }
            .into());
        }
        validate_data_dependent_compilation(program)?;
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
        // Arrays that session constants refer to enter the executable as hidden arguments that follow the captures, so
        // that the executable depends only on their types. Execution supplies them with the captures, as
        // `CompilationDomain::compile` shows.
        let session_constants = collect_session_constants(program);
        effective_capture_types.extend(session_constants.iter().map(|constant| constant.r#type().into_owned()));
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
                return Err(ryft_core::ShardingError::MeshMismatch {
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
        let public_output_types = apply_signature_shardings(
            program_output_types[..output_count].to_vec(),
            options.output_shardings(),
            "output",
        )?;
        // Executable arguments are placed as their runtime arrays are, or as an input override requests, including
        // over the auto mesh axes that their types never record. Session constants are placed as their retained arrays.
        let source_capture_count = lowered_capture_count - session_constants.len();
        let logical_argument_shardings = effective_input_types
            .iter()
            .enumerate()
            .map(|(index, array_type)| {
                let placement = if index < source_capture_count {
                    options.capture_placements().and_then(|placements| placements.get(index).cloned().flatten())
                } else if index < lowered_capture_count {
                    let constant = &session_constants[index - source_capture_count];
                    Some(self.retained_constant(constant)?.sharding().clone())
                } else {
                    let index = index - lowered_capture_count;
                    options.input_shardings().and_then(|shardings| shardings.get(index).cloned()).or_else(|| {
                        options.input_placements().and_then(|placements| placements.get(index).cloned().flatten())
                    })
                };
                Ok(placement.or_else(|| array_type.sharding().cloned()).unwrap_or_else(|| {
                    Sharding::replicated(options.mesh().logical_mesh().clone(), array_type.shape().rank())
                }))
            })
            .collect::<Result<Vec<_>, Error>>()?;
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
        // Outputs are placed as an output override requests, including over auto mesh axes, and otherwise as their
        // types are.
        let output_placement = |index: usize, array_type: &ArrayType| {
            options
                .output_shardings()
                .and_then(|shardings| shardings.get(index).cloned())
                .or_else(|| array_type.sharding().cloned())
        };
        // An output without a placement is replicated. These placements are both the result annotations that lowering
        // emits and the placements from which execution reconstructs the outputs, so that the two always agree. In
        // particular, a result is never left without an annotation, which would let the compiler place it differently
        // (e.g., as the output of a `shard_map` that it is derived from) than execution expects.
        let mut output_shardings = output_types
            .iter()
            .enumerate()
            .map(|(index, array_type)| {
                output_placement(index, array_type).unwrap_or_else(|| {
                    Sharding::replicated(options.mesh().logical_mesh().clone(), array_type.shape().rank())
                })
            })
            .collect::<Vec<_>>();
        // Mutated bindings cover the hidden output suffix in order, as validated above. Derive both the hidden state
        // type and its placement from the same input binding in one pass.
        for (state, &logical_input_index) in reference_states.iter().zip(&reference_state_input_indices) {
            if state.is_mutated() {
                output_types.push(effective_input_types[logical_input_index].clone());
                output_shardings.push(logical_argument_shardings[logical_input_index].clone());
            }
        }
        // Shardy rejects bounded dynamic tensors anywhere in the module, not only at the computation boundary, so a
        // program whose boundary is static but whose interior derives a dynamic extent through a gateway operation
        // still has to take the replica path. This is why the whole region arena is inspected instead of just the
        // boundary types, which it already subsumes.
        let use_shardy_partitioner = has_only_static_array_types(program);
        // A `shard_map` body only ever has static boundary types, so it can coexist with dynamic tensors staged
        // elsewhere in the same program. Its `sdy.manual_computation` lowering is meaningless without Shardy on any
        // device count, and bounded dynamic tensors force Shardy off for the whole module.
        if !use_shardy_partitioner && contains_shard_map(program) {
            return Err(Error::Lowering(crate::LoweringError::UnsupportedOperation {
                operation: ryft_core::SHARD_MAP_OPERATION_NAME.to_string(),
                reason: "bounded-dynamic programs cannot contain `shard_map` regions because `sdy.manual_computation` \
                         requires the Shardy partitioner"
                    .to_string(),
            }));
        }
        if !use_shardy_partitioner
            && options.mesh().devices().len() > 1
            && logical_argument_shardings
                .iter()
                .chain(output_types.iter().filter_map(ArrayType::sharding))
                .any(|sharding| !sharding.is_replicated())
        {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: "bounded-dynamic multi-device programs currently require fully replicated shardings \
                             because Shardy rejects dynamic tensors"
                    .to_string(),
            }));
        }
        let compilation_options = jit_compilation_options(
            options.compilation_options().unwrap_or_else(|| self.compilation_options()),
            options.mesh().devices(),
            use_shardy_partitioner,
            options.feedback_directed_profile(),
        );
        // The target platform gates platform-specific lowerings such as fused attention. A client whose platform
        // cannot be queried cannot compile the program either, so a failed query fails the lowering.
        let target = self.target()?;
        let target_platform = Some(target.platform_name());
        // Shardy currently rejects bounded dynamic tensors. Fully replicated dynamic programs need no partitioning,
        // so lower them without Shardy metadata and execute one replica per mesh device instead.
        let lowered_argument_shardings = use_shardy_partitioner.then_some(logical_argument_shardings.as_slice());
        let lowered_result_shardings = use_shardy_partitioner.then_some(output_shardings.as_slice());
        // Public outputs without an override leave their placement over auto mesh axes to the compiler, like JAX
        // outputs without `out_shardings`, and compilation reads that placement back from the executable. Axes of size
        // one admit only one placement, so meshes without larger auto axes keep every result closed. Outputs with
        // reduction state or manual variation, and hidden reference states, are always placed as declared, and
        // outputs with unconstrained dimensions already leave their whole placement to the compiler, as
        // `ExecutionPlan` explains.
        let open_outputs = output_shardings
            .iter()
            .enumerate()
            .map(|(index, sharding)| {
                lowered_result_shardings.is_some()
                    && index < output_count
                    && options.output_shardings().is_none()
                    && sharding.mesh().axes().iter().any(|axis| axis.r#type() == MeshAxisType::Auto && axis.size() > 1)
                    && sharding.unreduced_axes().is_empty()
                    && sharding.reduced_axes().is_empty()
                    && sharding.varying_manual_axes().is_empty()
                    && !sharding.dimensions().contains(&ShardingDimension::Unconstrained)
            })
            .collect::<Vec<_>>();
        let lowered_module = crate::lowering::lower_mlir_module_for_program_with_reference_state(
            program,
            effective_capture_types.as_slice(),
            &effective_public_input_types,
            &output_types,
            "main",
            lowered_argument_shardings,
            lowered_result_shardings,
            target_platform,
            reference_states,
            options.ragged_dot_lowering_strategy(),
            &open_outputs,
        )?;
        // The module mesh assigns the devices of the target mesh to mesh coordinates, so a program over another
        // topology cannot be placed on it.
        if let Some(mesh) = lowered_module.mesh()
            && !mesh.has_same_topology(options.mesh().logical_mesh())
        {
            return Err(ryft_core::ShardingError::MeshMismatch {
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
        let embeds_kernels = program.entry_region_ref().computation_regions().any(|region| {
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
        LoweredProgram::new(
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
            options.mesh().clone(),
            target,
            kernel_execution_facts,
        )
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
        InvocationMetadata::from(current).validate_replacement(replacement.into())
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

    fn validate_cached_program(
        &self,
        program: &Self::LoweredProgram,
        compiled: &Self::CompiledProgram,
    ) -> Result<(), Self::Error> {
        // Fresh compilations copy this metadata verbatim, so this rejects only reused programs whose invocation
        // contract does not match the requesting lowering (e.g., a corrupted or colliding persistent or exchanged
        // artifact). The stateful transaction trusts this metadata, so a mismatch must be caught before the first
        // call.
        match InvocationMetadata::from(program).incompatible_field(compiled.into()) {
            Some(field) => Err(Error::Compilation(CompilationError::IncompatibleExecutable {
                reason: format!("cached executable has incompatible {field}"),
            })),
            None => Ok(()),
        }
    }

    #[inline]
    fn persistent_cache_key(&self, key: &Self::CacheKey) -> Option<Vec<u8>> {
        Some(key.canonical_bytes().to_vec())
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

/// Returns `input_types` with the input sharding overrides of `options` (i.e., [`Options::input_shardings`])
/// applied, or `input_types` unchanged when `options` has none. The overrides must be applied before tracing, because
/// tracers carry their input types and sharding-aware operations read them. Retained dispatch applies them through
/// [`CompilationDomain::dispatch_signature`], the staging entry
/// points of this crate apply them through [`with_input_shardings`], and lowering rejects staged functions whose input
/// types do not carry them. External reference inputs are rejected when overrides are present, because holder state
/// derives its effective sharding from the compiled reference ABI.
pub(super) fn apply_input_shardings(
    input_types: Vec<ArrayIrType>,
    options: &Options,
) -> Result<Vec<ArrayIrType>, Error> {
    let Some(input_shardings) = options.input_shardings() else {
        return Ok(input_types);
    };
    if input_types.iter().any(|r#type| r#type.is_reference()) {
        return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
            reason: "external reference inputs do not support explicit input shardings".to_string(),
        }));
    }
    apply_signature_shardings(input_types, Some(input_shardings), "input")
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
///     ArrayIrType, ArrayType, DataType, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding,
///     ShardingDimension,
/// };
/// use ryft_xla::Options;
/// use ryft_xla::compilation::with_input_shardings;
///
/// let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit)?])?;
/// let mesh = DeviceMesh::new(logical_mesh, vec![Device::new(0, 0), Device::new(1, 0)])?;
/// let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])])?;
/// let options = Options::new(mesh).with_input_shardings(vec![sharding.clone()]);
/// let input_type = ArrayType::new_static(DataType::F32, [4]);
/// let input_types = (ArrayIrType::Array(input_type.clone()),);
/// let input_types = with_input_shardings(input_types, &options)?;
/// assert_eq!(input_types, (ArrayIrType::Array(input_type.with_sharding(sharding)?),));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn with_input_shardings<In: Parameterized<ArrayIrType>>(input_types: In, options: &Options) -> Result<In, Error> {
    if options.input_shardings().is_none() {
        return Ok(input_types);
    }
    let input_structure = input_types.parameter_structure();
    let input_types = apply_input_shardings(input_types.into_parameters().collect(), options)?;
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

/// Validates that the staged `output_types` carry the output sharding overrides of `options` (i.e.,
/// [`Options::output_shardings`]). The staging entry points of this crate stage the overrides after tracing (i.e., a
/// `reshard` over explicit mesh axes and a `constrain_sharding` over auto mesh axes), so that calls of a staged
/// function return the output types that its executions return. Types never record placement over auto mesh axes, as
/// [`ArrayType::with_sharding`] documents, so only the placement over the other mesh axes is compared, and lowering
/// then places the outputs over auto mesh axes as the overrides request. Outputs that are not arrays are left for the
/// later array-only conversions to reject.
fn validate_output_shardings(output_types: &[ArrayIrType], options: &Options) -> Result<(), Error> {
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
        let ArrayIrType::Array(array_type) = r#type else {
            continue;
        };
        if array_type.sharding() != Some(&sharding.without_auto_axes()) {
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
    program: &CoreFlatProgram<Session>,
    capture_count: usize,
    output_types: &[ArrayIrType],
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
fn validate_output_types(declared: &[ArrayIrType], actual: &[ArrayType]) -> Result<(), Error> {
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
fn apply_signature_shardings(
    mut types: Vec<ArrayIrType>,
    shardings: Option<&[Sharding]>,
    kind: &'static str,
) -> Result<Vec<ArrayIrType>, Error> {
    let Some(shardings) = shardings else {
        return Ok(types);
    };
    if shardings.len() != types.len() {
        return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
            kind,
            expected: types.len(),
            actual: shardings.len(),
        }));
    }
    for (r#type, sharding) in types.iter_mut().zip(shardings) {
        *r#type = match r#type {
            ArrayIrType::Array(array_type) => array_type.clone().with_sharding(sharding.without_auto_axes())?.into(),
            ArrayIrType::Dimension(_) | ArrayIrType::Reference(_) => {
                return Err(ProgramError::from(TypeError::invalid(format!(
                    "cannot apply an array sharding to type `{type}`",
                )))
                .into());
            }
        };
    }
    Ok(types)
}

/// Returns whether every array type that occurs anywhere in `program` has a static shape. The walk covers the whole
/// region arena, and therefore every region boundary, constant, and instruction result, recursively through attached
/// regions. First-class atoms of dimension type are ignored because they lower to ordinary scalars. Only dynamically
/// shaped *array* types make the lowered module unacceptable to the Shardy partitioner.
fn has_only_static_array_types(program: &FlatProgram) -> bool {
    program.regions().iter().all(|region| {
        region.atoms().iter().all(|atom| match atom.r#type().as_ref() {
            ArrayIrType::Array(array_type) => array_type.static_shape().is_some(),
            ArrayIrType::Dimension(_) => true,
            // The predicate answers its own question rather than relying on the reference rejection during lowering
            // firing first. A dynamically shaped referent counts as dynamic so that partitioner selection and its
            // guards stay correct even for programs whose only dynamic type is a reference referent.
            ArrayIrType::Reference(reference_type) => reference_type.referent().static_shape().is_some(),
        })
    })
}

/// Returns whether `program` stages a `shard_map` anywhere, including inside nested regions. The documentation of
/// [`ShardMapOperation`](ryft_core::ShardMapOperation) requires `shard_map` boundaries to be static, so a `shard_map`
/// can still appear in a program that is bounded dynamic elsewhere, which this predicate detects.
fn contains_shard_map(program: &FlatProgram) -> bool {
    program.regions().iter().any(|region| {
        region
            .instructions()
            .iter()
            .any(|instruction| matches!(instruction.operation(), Operation::ShardMap(_)))
    })
}

/// Returns the runtime placement of every output of `program` compiled as `executable`. The placement of an output
/// that lowering left open over auto mesh axes, as recorded by [`LoweredProgram::open_outputs`], is the placement that
/// the compiler chose for it, which must agree with the placement that the output type records over the other mesh
/// axes.
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, Array as CoreArray, ArrayIrType, ArrayReferenceTransform, ArrayReferenceTransformIndex,
        ArraySliceAxis, ArrayType, CaptureReference, CompilationArtifactExchange, CompilationArtifactExchangeError,
        CompilationArtifactExchangePolicy, CompilationCache, CompilationCacheDomain, CompilationTracer,
        ConditionOperation, ConstantOperation, CotangentDestinationKind, CumulativeKind, CumulativeOperation, DataType,
        Device, DeviceMesh, Dimension as CoreDimension, DimensionAddOperation, DimensionBounds,
        DimensionFromScalarOperation, DimensionOperation, DimensionSizeOperation, DimensionType, DimensionValue,
        DimensionVariable, DynamicBroadcastOperation, DynamicReshapeOperation, LogicalMesh, Memory, MeshAxis,
        MeshAxisType, MulOperation, NegOperation, OneOperation, Operation as CoreOperation, Placeholder, ProgramError,
        Provenance, ProvenanceScope, ReduceOperation, ReductionKind, ReferenceAddUpdate, ReferenceAddUpdateOperation,
        ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation, ReferenceRead,
        ReferenceReadOperation, ReferenceSwapOperation, ReferenceType, ReferenceWrite, ReferenceWriteOperation,
        Reshard, ScanOperation, Shape, ShardMapBoundary, ShardMapOperation, Sharding, ShardingDimension, Sin,
        SliceOperation, StagedFunction as CoreStagedFunction, WhileOperation,
    };
    use ryft_pjrt::protos::Precision;

    use crate::compilation::functions::StatefulCompileTracer;
    use crate::sessions::tests::{
        boolean_scalar, domain_mesh, external_reference_read_program, f32_scalar, f32_vector, read_f32s, read_f64s,
        replicated_scalar_type, replicated_vector_type, session_with_client,
    };
    use crate::tests::{cpu_client, execution_client, values_from_bytes, values_to_bytes};
    use crate::{
        Array, Constant, Error, FlatProgram, FromPjrt, JitCallOperation, Operation, ProgramBuilder, Session, Value,
    };

    use super::*;

    #[test]
    fn test_session_compilation_domain_pipeline() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let engine = session_with_client(client).with_device_mesh(mesh.clone());

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let options = Options::new(mesh.clone());
        let staged = crate::stage::<_, ArrayType, ArrayType>(|x| x.sin().unwrap(), input_type.clone(), options)
            .unwrap()
            .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<Session, ArrayIrType, ArrayIrType> =
            staged.lower(&engine).unwrap().compile(&engine).unwrap();

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
            let _: ryft_core::compilation::CompiledFunction<Session, ArrayIrType, ArrayIrType> =
                staged.lower(&engine).unwrap().compile(&engine).unwrap();
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
        let session = session_with_client(client);
        let scalar_sharding = Sharding::new(mesh.logical_mesh().clone(), Vec::new()).unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

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
            CoreStagedFunction::<Session, ArrayIrType, ArrayIrType>::stage(forward,
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
            CoreStagedFunction::<Session, ArrayIrType, ArrayIrType>::stage(allocate,
                Vec::new(),
                ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
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
            CoreStagedFunction::<Session, ArrayIrType, ArrayIrType>::stage(forward,
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
        let session = session_with_client(client);
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
        let staged = CoreStagedFunction::<Session, ArrayIrType, ArrayIrType>::stage(
            allocate_reference,
            Vec::new(),
            ArrayIrType::Array(array_type),
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
        let session = session_with_client(client);
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
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
        let staged = CoreStagedFunction::<Session, ArrayIrType, ArrayIrType>::stage(
            forward,
            Vec::new(),
            scalar_type.clone(),
            options.clone(),
        )
        .unwrap();
        assert!(matches!(
            staged.lower(&session).map(|_| ()),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "staged input types do not carry the shardings requested by `input_shardings`; apply them \
                              to the input types before staging (e.g., with `with_input_shardings`)",
        ));

        let input_types = with_input_shardings(scalar_type, &options).unwrap();
        let staged =
            CoreStagedFunction::<Session, ArrayIrType, ArrayIrType>::stage(forward, Vec::new(), input_types, options)
                .unwrap();
        staged.lower(&session).unwrap();
    }

    #[test]
    fn test_session_lower_staged_program_requires_output_shardings() {
        let client = cpu_client(2);
        let session = session_with_client(client);
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
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated).unwrap())
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
        let staged = CoreStagedFunction::<Session, ArrayIrType, ArrayIrType>::stage(
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
        let staged = CoreStagedFunction::<Session, ArrayIrType, ArrayIrType>::stage(
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
        let staged = CoreStagedFunction::<Session, ArrayIrType, ArrayIrType>::stage(
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
        let session = session_with_client(client);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let operation = CumulativeOperation::new(0, CumulativeKind::Sum);
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program: FlatProgram = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
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
        assert_eq!(values_from_bytes::<f32>(&outputs[0].copy_to_host().unwrap()), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_session_lower_program_dynamic_dimension_arithmetic_broadcast_and_reshape() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client).with_device_mesh(mesh.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let build_program = |vector_type: &ArrayType, increment: usize| {
            let mut builder = ProgramBuilder::new();
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
        assert_eq!(read_f32s(client, &outputs[0]), vec![7.0; 4]);
    }

    #[test]
    fn test_session_lower_program_rejects_shard_maps_in_bounded_dynamic_programs() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client).with_device_mesh(mesh.clone());
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Manual).unwrap()]).unwrap();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let value_type = replicated_scalar_type(&mesh, DataType::F32);
        let global_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let manual_sharding = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let boundary =
            ShardMapBoundary::new(manual_mesh, vec![manual_sharding.clone()], vec![manual_sharding], Vec::new())
                .unwrap();
        let local_type = boundary.local_input_type(0, &global_type).unwrap();

        // A `shard_map` body always has static boundary types, so it can coexist with dynamic tensors that are staged
        // beside it. Here the gateway makes the program bounded dynamic while the `shard_map` stays fully static.
        let mut body_builder = ProgramBuilder::new();
        let local = body_builder.add_input(local_type.into());
        let doubled =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![local, local], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::new();
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
    fn test_session_lower_program_uses_the_replica_path_for_internal_dynamic_tensors() {
        let client = cpu_client(4);
        let mesh = domain_mesh(client, "x", 4);
        let session = session_with_client(client).with_device_mesh(mesh.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let value_type = replicated_scalar_type(&mesh, DataType::F32);

        // The boundary of this program is fully static, but its interior derives a dynamic extent from a runtime
        // scalar, broadcasts to it, and reduces back to a scalar.
        let mut builder = ProgramBuilder::new();
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
        assert_eq!(read_f32s(client, &outputs[0]), vec![6.0]);

        // The replica path is a hard requirement rather than a preference, because annotating the very same program
        // for Shardy and compiling it with the Shardy partitioner enabled fails on the internal dynamic tensor, even
        // though every boundary type is static.
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let annotated = crate::lowering::lower_mlir_module_for_program(
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
        let shardy_lowered = lowered.clone().with_stable_hlo(annotated.into_parts().0).with_compilation_options(
            jit_compilation_options(session.compilation_options(), mesh.devices(), true, None),
        );
        let Err(error) = session.compile_program(&shardy_lowered) else {
            panic!("Shardy unexpectedly accepted a module containing an internal dynamic tensor");
        };
        assert!(error.to_string().contains("only supports ranked tensors with a static shape"), "{error}");
    }

    #[test]
    fn test_session_lower_program_rejects_unsupported_external_reference_storage_classes() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let replicated = |rank| Sharding::replicated(mesh.logical_mesh().clone(), rank);

        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(2)]))
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

        let host_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]))
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
        let unbounded_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(unbounded)]))
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
        let sharded_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(bounded)]))
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
                ryft_core::Error::Sharding(ryft_core::ShardingError::MeshMismatch { expected, actual })
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
        assert!(matches!(
            session.lower_program(
                &external_reference_read_program(scalar_type),
                0,
                &Options::new(foreign_mesh),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference state requires a mesh of devices that this client sees under the \
                              processes that own them, at least one of which this process addresses",
        ));

        let non_addressable_mesh = DeviceMesh::new(
            mesh.logical_mesh().clone(),
            vec![Device::new(local_device.id() + 1_000, local_device.process_index())],
        )
        .unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(non_addressable_mesh.logical_mesh().clone(), 0))
            .unwrap();
        assert!(matches!(
            session.lower_program(
                &external_reference_read_program(scalar_type),
                0,
                &Options::new(non_addressable_mesh),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference state requires a mesh of devices that this client sees under the \
                              processes that own them, at least one of which this process addresses",
        ));
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        let mut builder = ProgramBuilder::new();
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
        let program: FlatProgram = builder
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
        assert_eq!(read_f32s(client, &outputs[0]), vec![5.0]);
        assert_eq!(read_f32s(client, &outputs[1]), vec![8.0]);

        let eager_input = ryft_core::ArrayIrValue::Array(CoreArray::scalar(3.0f32).unwrap());
        let eager_replacement = ryft_core::ArrayIrValue::Array(CoreArray::scalar(5.0f32).unwrap());
        let eager_reference = eager_input.reference_new().unwrap();
        eager_reference.write(&eager_replacement).unwrap();
        let eager_snapshot = eager_reference.read().unwrap();
        eager_reference.add_update(&eager_input).unwrap();
        let eager_final = eager_reference.freeze().unwrap();
        assert_eq!(eager_snapshot, ryft_core::ArrayIrValue::Array(CoreArray::scalar(5.0f32).unwrap()));
        assert_eq!(eager_final, ryft_core::ArrayIrValue::Array(CoreArray::scalar(8.0f32).unwrap()));
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
        let session = session_with_client(client);
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
            let mut builder = ProgramBuilder::new();
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
            let mut builder = ProgramBuilder::new();
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
        let session = session_with_client(client);
        let mut builder = ProgramBuilder::new();
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
        let session = session_with_client(client);
        let vector_type = replicated_vector_type(&mesh, 4);
        let pair_type = replicated_vector_type(&mesh, 2);
        let mut builder = ProgramBuilder::new();
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
        assert_eq!(read_f32s(client, &outputs[0]), vec![4.0]);
        assert_eq!(read_f32s(client, &outputs[1]), vec![2.0, 3.0]);
        assert_eq!(read_f32s(client, &outputs[2]), vec![1.0, 11.0, 22.0, 4.0]);
    }

    #[test]
    fn test_session_lower_program_discharges_writes_through_reference_views() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let vector_type = replicated_vector_type(&mesh, 4);
        let pair_type = replicated_vector_type(&mesh, 2);
        let mut builder = ProgramBuilder::new();
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
        let program: FlatProgram = builder
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
        assert_eq!(read_f32s(client, &outputs[0]), vec![1.0, 10.0, 20.0, 4.0]);
    }

    #[test]
    fn test_session_lower_program_indexed_reference_mutation() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let vector_type = replicated_vector_type(&mesh, 3);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let mut builder = ProgramBuilder::new();
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
        assert_eq!(read_f32s(client, &outputs[0]), vec![2.0]);
        assert_eq!(read_f32s(client, &outputs[1]), vec![1.0, 13.0, 3.0]);
    }

    #[test]
    fn test_session_lower_program_preserves_capture_scope_inside_condition_regions() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let predicate_type = replicated_scalar_type(&mesh, DataType::Boolean);

        let branch = || {
            let mut builder = ProgramBuilder::new();
            let capture =
                builder.add_constant(Constant::Captured(CaptureReference::new(0, scalar_type.clone().into())));
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![capture], Vec::new(), vec![Placeholder]).unwrap()
        };
        let true_branch = branch();
        let false_branch = branch();
        let mut builder = ProgramBuilder::new();
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
        assert_eq!(read_f32s(client, &outputs[0]), vec![7.0]);
    }

    #[test]
    fn test_session_lower_program_validates_capture_count() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(scalar_type.into());
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            session.lower_program(&program, 2, &Options::new(mesh)),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::InvalidArgument { message })
                    if *message == "`capture_count` is 2, but the program has only 1 flat input(s)",
            ),
        ));
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state_through_condition() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let true_branch = {
            let mut builder = ProgramBuilder::new();
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
            let mut builder = ProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            builder.add_input(scalar_type.clone().into());
            let snapshot =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(replicated_scalar_type(&mesh, DataType::Boolean).into());
        let initial = builder.add_input(scalar_type.clone().into());
        let replacement = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
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
        assert_eq!(read_f32s(client, &true_outputs[0]), vec![10.0]);
        assert_eq!(read_f32s(client, &true_outputs[1]), vec![7.0]);

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
        assert_eq!(read_f32s(client, &false_outputs[0]), vec![10.0]);
        assert_eq!(read_f32s(client, &false_outputs[1]), vec![10.0]);
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state_through_while() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let condition = {
            let mut builder = ProgramBuilder::new();
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
            let mut builder = ProgramBuilder::new();
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

        let mut builder = ProgramBuilder::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(3).unwrap();
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
        assert_eq!(read_f32s(client, &outputs[0]), vec![5.0]);
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state_through_static_scan() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let body = {
            let mut builder = ProgramBuilder::new();
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

        let mut builder = ProgramBuilder::new();
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
        assert_eq!(read_f32s(client, &outputs[0]), vec![5.0]);
        assert_eq!(read_f32s(client, &outputs[1]), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state_through_dynamic_scan() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let length_variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let length = DimensionValue::new(DimensionType::from(length_variable.clone()), 3).unwrap();

        let body = {
            let mut builder = ProgramBuilder::new();
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

        let mut builder = ProgramBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let runtime_length = builder.add_constant(Constant::Dimension(length.clone()));
        let reference = builder
            .add_instruction(
                ScanOperation::new(1, CoreDimension::Dynamic(length_variable.clone())),
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
        assert_eq!(scan.iteration_count(), &CoreDimension::Dynamic(length_variable));
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
        assert_eq!(read_f32s(client, &outputs[0]), vec![5.0]);
    }

    #[test]
    fn test_session_lower_program_discharges_local_reference_state_through_nested_jit_call() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let callee = {
            let mut builder = ProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            let update = builder.add_input(scalar_type.clone().into());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone().into());
        let update = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let reference = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![reference, update], None)
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            program.entry_region_ref().instructions()[1].operation(),
            Operation::JitCall(operation) if operation.capture_count() == 0,
        ));
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.add").count(), 1, "{}", lowered.stable_hlo());
        let compiled = session.compile_program(&lowered).unwrap();
        let outputs = session
            .execute_program(&compiled, vec![f32_scalar(&session, &mesh, 2.0), f32_scalar(&session, &mesh, 4.0)])
            .unwrap();

        let immutable_oracle = 2.0f32 + 4.0;
        assert_eq!(read_f32s(client, &outputs[0]), vec![immutable_oracle]);
    }

    #[test]
    fn test_session_lower_program_carries_external_reference_state_metadata() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);

        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let output = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program: FlatProgram = builder
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
        let session = session_with_client(client).with_device_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::F64);

        // Builds `-(x * x)` with the multiplication carrying the provided provenance.
        fn build_program(
            input_type: &ArrayType,
            provenance: Provenance,
        ) -> crate::operations::Program<Vec<Constant>, Vec<Constant>> {
            let mut builder = ProgramBuilder::new();
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
        assert_eq!(read_f64s(client, &outputs[0]), vec![-9.0]);

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
        let session = session_with_client(client);
        let vector_type = replicated_vector_type(&mesh, 3);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(vector_type.clone());
        let true_branch = {
            let mut builder = ProgramBuilder::new();
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
            let mut builder = ProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(replicated_scalar_type(&mesh, DataType::Boolean).into());
        let initial = builder.add_input(vector_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let reference = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
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
            assert_eq!(read_f32s(client, &outputs[0]), expected);
        }
    }

    #[test]
    fn test_session_validate_cached_program_rejects_foreign_invocation_contracts() {
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
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        // The foreign executable keeps the requesting lowering's boundary types so that it loads and passes every
        // output type check in the core crate, and differs only in the donation contract the runtime call path trusts.
        let producer = session_with_client(client);
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |x| x.sin().unwrap(),
            input_type.clone(),
            Options::new(mesh.clone()).with_donate(true),
        )
        .unwrap()
        .into_inner();
        let foreign: ryft_core::compilation::CompiledFunction<Session, ArrayIrType, ArrayIrType> =
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
        assert!(matches!(
            fallback_consumer.validate_cached_program(lowered.lowered_program(), compiled.compiled_program()),
            Ok(()),
        ));
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
                ryft_core::Error::Program(ProgramError::InvalidArgument { message })
                    if *message == "compilation artifact exchange requirement was not satisfied: received compilation \
                        artifact is incompatible: incompatible XLA executable: cached executable has \
                        incompatible donation flags",
            ),
        ));
        assert_eq!(required_consumer.compilation_cache().statistics().compilations, 0);
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
