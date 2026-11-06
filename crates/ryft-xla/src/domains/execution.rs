use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;

use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayReference, ArrayType, Cast, DataType, Device, DeviceId, DeviceMesh, Dimension as CoreDimension,
    Layout, ProgramError, ReadyOrPendingReferenceGuard, ReferenceCompletion as CoreReferenceCompletion,
    ReferenceCompletionBackend, ReferenceId, ReferenceReplacementPreparation, ReferenceType, Shape, Sharding,
    ShardingDimension, StatefulCompilationDomain, StatefulExecution, StaticShape, Type, TypeRefinements, Typed,
    ValueProjection, validate_reference_boundary,
};
use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Buffer, BufferSpecification, BufferType, Execution, ExecutionFence, LoadedExecutable};

use crate::arrays::{ArrayOutputLayout, ArrayTypeExtension};
use crate::arrays_v0::host::{
    DenseArrayHostCopy, begin_materialize_dense_array_bytes, materialize_dense_array_bytes,
    reject_unreduced_host_round_trip, relayout_shards_through_host,
};
use crate::arrays_v0::{
    BoundedMaterializationKey, BoundedMaterializationProbe, BoundedMaterializationProducer,
    BoundedMaterializationWaiter, ExecuteArguments,
};
use crate::domains::Domain;
use crate::domains::compilation::CompilationError;
use crate::domains::effects::{ensure_effect_dispatch_allowed, format_effect_failures};
use crate::domains::programs::CompiledProgram;
use crate::domains::runtimes::RuntimeRequirement;
#[cfg(test)]
use crate::domains::tests::{
    StatefulFailureInjection, take_stateful_completion_injection, take_stateful_failure_injection,
};
use crate::experimental::lowering::signature::ExecutableSignature;
use crate::{Array, ArrayError, Error, FromPjrt, ToPjrt, Value};

// TODO(eaplatanios): Review this module.

/// XLA invocation, runtime capability, completion, and effect-recovery failures.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ExecutionError {
    /// The executable belongs to a different live PJRT client.
    #[error("xla executable belongs to a different PJRT client")]
    ClientMismatch,

    /// Runtime donation metadata has a different arity than the invocation inputs.
    #[error("got {actual_count} donation flag(s), but expected {expected_count}")]
    DonationFlagCountMismatch { expected_count: usize, actual_count: usize },

    /// An input array has no buffered shard for a device required by the executable assignment.
    #[error("input array #{array_index} has no addressable shard for device {device_id}")]
    MissingArrayShardForDevice { array_index: usize, device_id: DeviceId },

    /// An input array has a buffered shard outside the executable's device assignment.
    #[error("input array #{array_index} has an unexpected addressable shard for device {device_id}")]
    UnexpectedArrayShardDevice { array_index: usize, device_id: DeviceId },

    /// An executable with external mutable reference state was called through a stateless API.
    #[error("external reference state requires the stateful invocation API")]
    StatefulInvocationRequired,

    /// Assertion callbacks are unavailable on the selected platform or in this feature configuration.
    #[error("compiled runtime assertions are not supported on XLA platform `{platform}`")]
    UnsupportedAssertionPlatform { platform: String },

    /// The selected platform has no qualified native ordered-I/O token transport.
    #[error("native ordered I/O transport is not yet verified on platform `{platform}`")]
    UnsupportedEffectPlatform { platform: String },

    /// Ordered I/O cannot execute over the selected distributed or non-addressable assignment.
    #[error("ordered I/O requires a fixed fully addressable single-process device assignment")]
    UnsupportedEffectDeviceAssignment,

    /// Multi-device ordered I/O was declared without the required device-ordering contract.
    #[error("multi-device ordered I/O requires every ordered occurrence to declare `DeviceOrderedIo`")]
    MissingDeviceOrderedEffect,

    /// Previously failed ordered work must be observed and acknowledged before further ordered submission.
    #[error("ordered I/O has failed; call `effects_barrier()` and then `acknowledge_effect_errors()` before retrying")]
    ScopeNeedsRecovery,

    /// An existing ordered effect chain cannot change its device assignment.
    #[error("ordered I/O device changed from {expected:?} to {actual:?}; use `fork_effect_scope()`")]
    ScopeDeviceMismatch { expected: Vec<DeviceId>, actual: Vec<DeviceId> },

    /// Execution or an effect barrier was entered from a managed callback.
    #[error("execution and effect barriers cannot be entered from a managed callback")]
    ReentrantDispatch,

    /// Submitted work failed asynchronously, including invocations that have no external reference state.
    #[error("asynchronous XLA execution failed: {reason}")]
    AsynchronousFailure { reason: Arc<str> },

    /// An effect barrier observed failed submissions, retaining their reservation order and diagnostics.
    #[error(
        "{}; call `effects_barrier()` again if more work was submitted, then `acknowledge_effect_errors()` to recover \
         once affected work is complete",
        format_effect_failures(.failures),
    )]
    EffectSubmissionFailures { failures: Vec<(u64, Arc<str>)> },

    /// Pending or unobserved failed work prevents acknowledgment of the current effect scope.
    #[error(
        "effect failures are pending or unreported; call `effects_barrier()` and then `acknowledge_effect_errors()` \
         again"
    )]
    PrematureAcknowledgment,
}

/// Backend-neutral completion adapter for one submitted PJRT execution.
struct ReferenceCompletion {
    /// Whole-execution fence joining every addressable device event.
    fence: ExecutionFence,
}

impl ReferenceCompletionBackend for ReferenceCompletion {
    fn r#await(&self) -> Result<(), Arc<str>> {
        self.fence.block_until_ready().map_err(|error| Arc::from(error.to_string()))
    }

    fn is_ready(&self) -> Result<bool, Arc<str>> {
        self.fence.is_ready().map_err(|error| Arc::from(error.to_string()))
    }
}

/// Fully materialized physical arguments and logical refinements prepared before PJRT submission.
struct PreparedExecution<'c> {
    /// Fully materialized physical arrays in executable argument order.
    inputs: Vec<Array<'c>>,

    /// Addressable devices in executable order.
    addressable_device_ids: Arc<[DeviceId]>,

    /// Effective physical donation request for every input.
    donation_flags: Vec<bool>,

    /// Runtime refinements established from the unmaterialized logical inputs.
    input_refinements: <ArrayType as Type>::Refinements,

    /// Physical result count expected from every addressable device.
    physical_output_count: usize,
}

impl<'c> PreparedExecution<'c> {
    /// Consumes the materialized arrays into device-major PJRT arguments.
    fn into_arguments(self) -> Result<(ExecuteArguments<'c>, <ArrayType as Type>::Refinements, usize), Error> {
        let arguments = ExecuteArguments::from_arrays_with_donation(
            self.inputs,
            self.addressable_device_ids,
            self.donation_flags.as_slice(),
        )?;
        Ok((arguments, self.input_refinements, self.physical_output_count))
    }
}

/// Execution metadata of an [`CompiledProgram`] that does not depend on the values that it is called with. It is
/// computed once when the program is compiled or loaded, so that each execution only reads it instead of re-deriving
/// physical types, device assignments, donation requests, and output placements, and instead of querying PJRT.
pub(super) struct ExecutionPlan {
    /// Logical input index of each physical array input, excluding hidden extent scalars and tokens.
    physical_input_logical_indices: Vec<usize>,

    /// Physical array input types, including hidden extent scalars.
    physical_input_types: Vec<ArrayType>,

    /// Physical donation request of each physical array input when the caller overrides no donation.
    default_donation_flags: Vec<bool>,

    /// Identifiers of the executable's addressable devices, in execution order.
    addressable_device_ids: Arc<[DeviceId]>,

    /// Whether every logical input type has a static shape. Refining such inputs reduces to comparing types.
    static_inputs: bool,

    /// Whether any physical array input has the zero-space data type and thus needs a materialized carrier.
    zero_space_inputs: bool,

    /// Placement of every logical output, present when every output has a static type, every output without a
    /// physical result is zero-space, and the program runs on a single device or declares a sharding without
    /// `Unconstrained` dimensions for every output (which XLA must then honor). The outputs of such programs are built
    /// directly from the returned buffers.
    output_layouts: Option<Vec<ArrayOutputLayout>>,
}

impl ExecutionPlan {
    /// Computes the execution plan of a program with the provided metadata, loaded as `executable`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        executable: &LoadedExecutable<'_>,
        input_types: &[ArrayType],
        output_types: &[ArrayType],
        signature: &ExecutableSignature,
        donation_flags: &[bool],
        capture_count: usize,
        mesh: &DeviceMesh,
    ) -> Result<Self, Error> {
        let physical_input_logical_indices = signature.project_inputs(&(0..input_types.len()).collect::<Vec<_>>());
        let mut physical_input_types = signature.physical_input_types(input_types);
        let compiled_executable = executable.executable()?;
        let input_layouts =
            if physical_input_types.is_empty() { Vec::new() } else { compiled_executable.input_layouts()? };
        let physical_output_types = signature.physical_output_types(output_types);
        let output_layouts = if physical_output_types.iter().any(|array_type| array_type.layout().is_some()) {
            compiled_executable.output_layouts()?
        } else {
            Vec::new()
        };
        // Low-level compile options can override MLIR layout attributes. Validate the compiled ABI before trusting
        // declared layouts, including when rebuilding a plan for a persistent executable. Unspecified layouts impose
        // no constraint, and tokens are excluded from these physical array type lists.
        for (kind, types, layouts) in [
            ("input", physical_input_types.as_slice(), input_layouts.as_slice()),
            ("output", physical_output_types.as_slice(), output_layouts.as_slice()),
        ] {
            for (index, array_type) in types.iter().enumerate() {
                if let Some(expected) = array_type.layout() {
                    let layout = layouts.get(index).ok_or_else(|| {
                        Error::Compilation(CompilationError::IncompatibleExecutable {
                            reason: format!("executable {kind} {index} has no physical layout"),
                        })
                    })?;
                    let actual = Layout::from_pjrt(ryft_pjrt::Layout::from_proto(layout.serialize()?.proto()?)?)?;
                    if &actual != expected {
                        return Err(Error::Compilation(CompilationError::IncompatibleExecutable {
                            reason: format!(
                                "executable {kind} {index} uses layout `{actual}` but its type declares `{expected}`",
                            ),
                        }));
                    }
                }
            }
        }
        // These types describe the physical executable signature, including layouts chosen for unconstrained
        // logical inputs. Keep the logical declarations unchanged so they still admit other representations.
        for (index, array_type) in physical_input_types.iter_mut().enumerate() {
            let layout = input_layouts.get(index).ok_or_else(|| {
                Error::Compilation(CompilationError::IncompatibleExecutable {
                    reason: format!("executable input {index} has no physical layout"),
                })
            })?;
            let layout = Layout::from_pjrt(ryft_pjrt::Layout::from_proto(layout.serialize()?.proto()?)?)?;
            *array_type = array_type.clone().with_layout(layout);
        }
        let logical_donation_flags =
            std::iter::repeat_n(false, capture_count).chain(donation_flags.iter().copied()).collect::<Vec<_>>();
        let mut default_donation_flags = signature.project_inputs(logical_donation_flags.as_slice());
        default_donation_flags.extend(std::iter::repeat_n(false, signature.input_dimensions().len()));
        for (donation, input_type) in default_donation_flags.iter_mut().zip(physical_input_types.iter()) {
            if input_type.data_type().is_zero() {
                *donation = false;
            }
        }
        let addressable_device_ids = executable
            .addressable_devices()?
            .iter()
            .map(|device| device.id().map_err(Error::from))
            .collect::<Result<Arc<[_]>, _>>()?;
        let static_inputs = input_types.iter().all(|input_type| input_type.static_shape().is_some());
        let zero_space_inputs = physical_input_types.iter().any(|input_type| input_type.data_type().is_zero());
        let static_outputs = signature.output_dimensions().is_empty()
            && output_types.iter().zip(signature.output_mapping()).all(|(output_type, physical_index)| {
                output_type.static_shape().is_some() && (physical_index.is_some() || output_type.data_type().is_zero())
            });
        // Declared shardings only fix output placement when XLA must honor them. Lowering emits no result shardings
        // when any output lacks one, and Shardy propagation may shard the open dimensions that `Unconstrained` lowers
        // to, so XLA chooses the placement of such outputs. Their buffers are validated by the general constructor
        // instead. Every placement on a single device holds the whole array, so it is always fixed.
        let declared_placement = mesh.devices().len() == 1
            || output_types.iter().all(|output_type| {
                output_type.sharding().is_some_and(|sharding| {
                    !sharding.dimensions().iter().any(|dimension| matches!(dimension, ShardingDimension::Unconstrained))
                })
            });
        let output_layouts = if static_outputs && declared_placement {
            let mesh = Arc::new(mesh.clone());
            Some(
                output_types
                    .iter()
                    .map(|output_type| {
                        let output_type = match output_type.sharding() {
                            Some(_) => output_type.clone(),
                            None => output_type.replicated(mesh.as_ref())?,
                        };
                        Ok(ArrayOutputLayout::new(output_type, Arc::clone(&mesh), &addressable_device_ids)?)
                    })
                    .collect::<Result<Vec<_>, Error>>()?,
            )
        } else {
            None
        };
        Ok(Self {
            physical_input_logical_indices,
            physical_input_types,
            default_donation_flags,
            addressable_device_ids,
            static_inputs,
            zero_space_inputs,
            output_layouts,
        })
    }
}

impl<'c> Domain<'c> {
    /// Initializes required extensions on the invoking session before compilation or native loading.
    pub(super) fn ensure_runtime_requirements(&self, requirements: &[RuntimeRequirement]) -> Result<(), Error> {
        if requirements.is_empty() {
            return Ok(());
        }
        self.session.runtimes.ensure(requirements, self.target()?)
    }

    /// Validates that this domain's PJRT client owns `program`.
    pub(super) fn validate_program_owner(&self, program: &CompiledProgram<'c>) -> Result<(), Error> {
        if program.executable.is_owned_by(self.client()) {
            Ok(())
        } else {
            Err(Error::Execution(ExecutionError::ClientMismatch))
        }
    }

    /// Enqueues a stateless `program` and returns its possibly still pending flat outputs together with a
    /// whole-execution fence. Programs with external reference state must flow through the stateful call surface
    /// instead, which owns the reference transaction and the hidden final-state outputs.
    ///
    /// The backend may execute inline or return pending arrays and a completion fence. This function adds no wait
    /// solely because public outputs are empty. Bounded-dynamic boundaries can introduce additional waits:
    /// below-bound inputs perform device-to-host copies and bound-shaped uploads before launch, and each dynamic
    /// output reads its hidden extent scalar before constructing the logical output type. Erased zero-space inputs
    /// await producer completion before launch because they have no physical buffer to carry that dependency.
    pub(crate) fn execute_compiled_async(
        &self,
        program: &CompiledProgram<'c>,
        inputs: Vec<Array<'c>>,
    ) -> Result<Execution<Vec<Array<'c>>>, Error> {
        // Public call surfaces screen stateful programs out before reaching this entry point: only the stateful call
        // path owns holder leases and pending generations, so a stateful program arriving here must already carry
        // state values as plain arrays (as ABI-level tests do), and its hidden final-state outputs are returned
        // verbatim.
        let prepared = self.prepare_compiled_execution(program, inputs, &[])?;
        let (arguments, input_refinements, physical_output_count) = prepared.into_arguments()?;
        let execution = self.execute_pjrt_buffers(program, arguments, physical_output_count, |_| {})?;
        let (physical_outputs, fence) = execution.into_parts();
        let mut physical_outputs = physical_outputs.into_iter().map(Some).collect::<Vec<_>>();
        let outputs = reconstruct_compiled_outputs(
            self,
            program,
            &mut physical_outputs,
            fence.clone(),
            0..program.output_types.len(),
        )?;
        // Static outputs cannot disagree with the input refinements, so only builds with debug assertions re-validate
        // them.
        if program.plan.output_layouts.is_none() || cfg!(debug_assertions) {
            validate_compiled_output_refinements(program, &input_refinements, program.output_types.as_ref(), &outputs)?;
        }
        if physical_outputs.iter().any(Option::is_some) {
            return Err(
                ProgramError::MalformedProgram("executable returned an unclaimed physical output".to_string()).into()
            );
        }
        Ok(Execution::new(outputs, fence))
    }

    /// Materializes and validates physical execution arguments without submitting device work.
    fn prepare_compiled_execution(
        &self,
        program: &CompiledProgram<'c>,
        inputs: Vec<Array<'c>>,
        donation_overrides: &[(usize, bool)],
    ) -> Result<PreparedExecution<'c>, Error> {
        ensure_effect_dispatch_allowed()?;
        self.validate_program_owner(program)?;
        if program.kernel_execution_facts.is_some() {
            crate::kernels::distributed::validate_kernel_participants(self.client(), &program.mesh)?;
        }
        self.ensure_runtime_requirements(&program.runtime_requirements)?;
        if inputs.len() != program.input_types.len() {
            return Err(Error::from(ProgramError::InvalidInputCount {
                expected: program.input_types.len(),
                actual: inputs.len(),
            }));
        }
        let plan = program.plan.as_ref();

        // Static input types establish no refinements, so refining them reduces to comparing every component except
        // the sharding, which is normalized separately at the executable boundary. Inputs that differ, as well as
        // dynamic input types, take the general path below, which also produces the precise diagnostic.
        let static_inputs_match = plan.static_inputs
            && program.input_types.iter().zip(inputs.iter()).all(|(declared, input)| {
                let actual = input.r#type();
                declared.data_type() == actual.data_type()
                    && declared.shape() == actual.shape()
                    && declared.layout() == actual.layout()
                    && declared.memory() == actual.memory()
            });
        let (input_refinements, actual_input_types) = if static_inputs_match {
            (<ArrayType as Type>::Refinements::default(), None)
        } else {
            let actual_input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            let refinement_input_types = program
                .input_types
                .iter()
                .zip(actual_input_types.iter())
                .map(|(declared, actual)| {
                    // Preserve every observed type component except the sharding, so that refinement still rejects a
                    // wrong data type, layout, memory space, rank, or extent.
                    actual.clone().with_sharding(declared.sharding().cloned()).map_err(Error::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let input_refinements = <ArrayType as Type>::Refinements::from_local_declarations(
                program.input_types.iter(),
                refinement_input_types.iter(),
            )
            .map_err(ProgramError::from)?;
            (input_refinements, Some(actual_input_types))
        };

        // Erased zero-space inputs have no buffers through which PJRT could order consumer work after their
        // producers. Observe completion before dropping those inputs, including before any consumer effects run.
        for (input, physical_index) in inputs.iter().zip(program.signature.input_mapping()) {
            if physical_index.is_none() {
                input.block_until_ready()?;
            }
        }

        // Without hidden extent scalars, physical inputs are a projection of the logical inputs (which is the identity
        // for most programs); bounded-dynamic inputs are padded to their physical bounds instead.
        let inputs = if program.signature.input_dimensions().is_empty() {
            if plan.physical_input_logical_indices.iter().copied().eq(0..inputs.len()) {
                inputs
            } else {
                plan.physical_input_logical_indices.iter().map(|&index| inputs[index].clone()).collect()
            }
        } else {
            let actual_input_types = match actual_input_types {
                Some(actual_input_types) => actual_input_types,
                None => inputs.iter().map(|input| input.r#type().into_owned()).collect(),
            };
            materialize_bounded_dynamic_inputs(
                self,
                &program.signature,
                &program.input_types,
                actual_input_types.as_slice(),
                inputs,
            )?
            .inputs
        };
        let inputs = if plan.zero_space_inputs {
            materialize_zero_space_carriers(self, plan.physical_input_types.as_slice(), inputs)?
        } else {
            inputs
        };
        let mut inputs = reshard_inputs_if_needed(self, &program.mesh, &program.expected_argument_shardings, inputs)?;
        // A logical input without a layout constraint admits any physical representation. Match the compiled
        // signature before handing buffers to PJRT, which otherwise reads their storage in its expected order.
        // Known layouts convert on device through the existing capability. Unknown layouts require observing the
        // shard buffers; only a mismatch takes the dense host fallback, which also handles differently laid-out shards.
        for (input, physical_type) in inputs.iter_mut().zip(&plan.physical_input_types) {
            if input.data_type().is_zero() {
                continue;
            }
            let expected = physical_type.layout().expect("the execution plan records every physical input layout");
            if let Some(actual) = input.layout() {
                if actual != expected {
                    // A representation adapter owns its small compilation signature, so do not inherit a caller's
                    // low-level parameter/result shape overrides. Keep the same session, mesh and effect scope.
                    let conversion_domain = self.with_compilation_options(CompilationOptions::default());
                    let source = input.clone().associate(&conversion_domain)?;
                    *input = source.cast(source.r#type().into_owned().with_layout(expected.clone()), &[])?;
                }
            } else {
                let expected_buffer_layout = expected.to_pjrt()?;
                let mut needs_conversion = false;
                for shard in input.addressable_shards() {
                    if let Some(buffer) = shard.buffer()
                        && buffer.specification()?.layout != expected_buffer_layout
                    {
                        needs_conversion = true;
                        break;
                    }
                }
                if needs_conversion && !input.sharding().unreduced_axes().is_empty() {
                    // The devices of an unreduced array hold separate summands, which a dense round trip would replace.
                    *input = relayout_shards_through_host(self, input, expected)?;
                } else if needs_conversion {
                    let bytes = materialize_dense_array_bytes(input)?;
                    *input = Array::from_host_buffer(
                        self,
                        input.r#type().into_owned().with_layout(expected.clone()),
                        input.mesh().clone(),
                        bytes,
                    )?;
                }
            }
        }
        let donation_flags = if donation_overrides.is_empty() {
            plan.default_donation_flags.clone()
        } else {
            let mut logical_donation_flags = std::iter::repeat_n(false, program.capture_count)
                .chain(program.donation_flags.iter().copied())
                .collect::<Vec<_>>();
            for &(logical_input_index, donation) in donation_overrides {
                let flag = logical_donation_flags.get_mut(logical_input_index).ok_or_else(|| {
                    Error::from(ProgramError::InvalidArgument {
                        message: format!("donation override input {logical_input_index} is out of range"),
                    })
                })?;
                *flag = donation;
            }
            let mut donation_flags = program.signature.project_inputs(logical_donation_flags.as_slice());
            donation_flags.extend(std::iter::repeat_n(false, program.signature.input_dimensions().len()));
            for (donation, input_type) in donation_flags.iter_mut().zip(plan.physical_input_types.iter()) {
                if input_type.data_type().is_zero() {
                    *donation = false;
                }
            }
            donation_flags
        };
        Ok(PreparedExecution {
            inputs,
            addressable_device_ids: Arc::clone(&plan.addressable_device_ids),
            donation_flags,
            input_refinements,
            physical_output_count: program.signature.array_output_count(),
        })
    }

    /// Executes a compiled program without external reference state on flat `inputs` (refer to
    /// [`CompilationDomain::execute`](ryft_core::CompilationDomain::execute)) and returns its public outputs together
    /// with the whole-execution completion fence.
    ///
    /// The returned output arrays already carry the fence individually, so
    /// [`CompilationDomain::execute`](ryft_core::CompilationDomain::execute) drops the returned fence; the stateful
    /// asynchronous surface threads it into a completion instead so that awaiting a reference-free invocation still
    /// observes asynchronous execution errors. When the program has no outputs, the effect scope retains effectful
    /// completion. Only pure zero-output calls explicitly wait.
    pub(super) fn execute_stateless(
        &self,
        program: &CompiledProgram<'c>,
        inputs: Vec<Value<'c>>,
    ) -> Result<(Vec<Value<'c>>, ExecutionFence), Error> {
        if program.requires_stateful_call() {
            return Err(Error::Execution(ExecutionError::StatefulInvocationRequired));
        }
        check_count!("input", inputs, program.input_types.len(), ProgramError);
        for (declared, input) in program.input_types.iter().zip(&inputs) {
            // Array inputs borrow their types, while the array IR type of other inputs is built only to report them.
            match input {
                Value::Array(array) => validate_input_type(declared, &array.r#type())?,
                input => validate_input_type(
                    declared,
                    <&ArrayType>::try_from(input.r#type().as_ref()).map_err(ProgramError::from)?,
                )?,
            }
        }
        let inputs = inputs
            .into_iter()
            .map(ValueProjection::<ArrayType>::into_projected)
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let execution = self.execute_compiled_async(program, inputs)?;
        if execution.output().is_empty() && !program.signature.has_effects() {
            execution.fence().block_until_ready()?;
        }
        let (outputs, fence) = execution.into_parts();
        validate_runtime_outputs(program.output_types(), &outputs)?;
        Ok((outputs.into_iter().map(Value::Array).collect(), fence))
    }

    /// Locks one stateful reference while preserving reference lifecycle errors through [`Error::Core`].
    fn lock_stateful_reference<'g>(
        reference: &'g ArrayReference<Array<'c>>,
    ) -> Result<ReadyOrPendingReferenceGuard<'g, Array<'c>>, Error> {
        reference.lock_storage().map_err(|error| {
            if let Some(reference_error) = error.downcast_custom::<ryft_core::ReferenceError>().cloned() {
                Error::Core(ryft_core::Error::Reference(reference_error))
            } else {
                Error::Core(ryft_core::Error::Program(error))
            }
        })
    }

    /// Submits a stateful execution of a compiled program on flat `inputs` (refer to
    /// [`StatefulCompilationDomain::execute_statefully`]), publishes its dependencies, and commits hidden final
    /// states without waiting for device completion when the logical boundary is static.
    fn execute_stateful_async(
        &self,
        program: &CompiledProgram<'c>,
        inputs: Vec<Value<'c>>,
    ) -> StatefulExecution<Vec<Value<'c>>, Error> {
        if let Err(error) = ensure_effect_dispatch_allowed() {
            return StatefulExecution::ready(Err(error));
        }
        if program.reference_states.is_empty() {
            // A reference-free invocation binds no holders, but the completion contract still promises that awaiting
            // the result observes whole-invocation completion, so the execution fence rides along as the completion.
            return match self.execute_stateless(program, inputs) {
                Ok((output, fence)) => StatefulExecution::pending(
                    Ok(output),
                    CoreReferenceCompletion::new(ReferenceCompletion { fence }),
                    |reason| Error::Execution(ExecutionError::AsynchronousFailure { reason }),
                ),
                Err(error) => StatefulExecution::ready(Err(error)),
            };
        }
        let prepared = (|| -> Result<_, Error> {
            check_count!("input", inputs, program.input_types.len(), ProgramError);
            let reference_state_input_indices = program
                .reference_states
                .iter()
                .map(|state| state.source().flat_input_index(program.capture_count))
                .collect::<Result<Vec<_>, _>>()?;

            // Discharge retains each reference's position but replaces its type with the referent's array type.
            // The compiled binding table restores that distinction without a caller-supplied signature. Its indices
            // are validated in increasing order during lowering and persistent executable restoration.
            for (index, (declared, input)) in program.input_types.iter().zip(&inputs).enumerate() {
                let expects_reference = reference_state_input_indices.binary_search(&index).is_ok();
                match (expects_reference, input) {
                    (false, Value::Array(array)) => validate_input_type(declared, &array.r#type())?,
                    (true, Value::Reference(reference)) => {
                        validate_input_type(declared, reference.r#type().referent())?;
                    }
                    _ => {
                        let actual = input.r#type();
                        let declared = if expects_reference {
                            ArrayIrType::Reference(ReferenceType::new(declared.clone()))
                        } else {
                            ArrayIrType::Array(declared.clone())
                        };
                        return Err(ProgramError::InvalidArgument {
                            message: format!("runtime input type `{actual}` does not match declared type `{declared}`"),
                        }
                        .into());
                    }
                }
            }

            let (captures, public_inputs) = inputs.split_at(program.capture_count.min(inputs.len()));
            validate_reference_boundary(public_inputs.iter(), captures.iter()).map_err(ProgramError::from)?;
            let mut bindings =
                Vec::<(ReferenceId, ArrayReference<Array<'c>>)>::with_capacity(program.reference_states.len());
            for &logical_input_index in &reference_state_input_indices {
                let reference = match inputs.get(logical_input_index) {
                    Some(Value::Reference(reference)) => reference.handle(),
                    Some(value) => {
                        return Err(Error::from(ProgramError::InvalidArgument {
                            message: format!(
                                "external state input {logical_input_index} must be a reference, but received `{}`",
                                value.r#type(),
                            ),
                        }));
                    }
                    None => {
                        return Err(ProgramError::InvalidInputCount {
                            expected: program.input_types.len(),
                            actual: inputs.len(),
                        }
                        .into());
                    }
                };
                let physical_input_index = program.signature.input_mapping()[logical_input_index].unwrap();
                let expected_sharding = &program.expected_argument_shardings[physical_input_index];
                let declared_type = &program.input_types[logical_input_index];
                let effective_type = if declared_type.sharding().is_some() {
                    declared_type.clone()
                } else {
                    declared_type.clone().with_sharding(expected_sharding.clone())?
                };
                let actual_type = reference.r#type();
                if !effective_type.is_refined_by(actual_type.referent()) {
                    return Err(Error::from(ProgramError::InvalidArgument {
                        message: format!(
                            "external state input {logical_input_index} reference type `{}` does not refine effective \
                             compiled state type `{effective_type}`",
                            actual_type.referent(),
                        ),
                    }));
                }
                if !reference.is_storage_root() {
                    return Err(Error::from(ProgramError::InvalidArgument {
                        message: format!(
                            "external state input {logical_input_index} must be a root reference handle that uses its \
                             allocation's stored type identities",
                        ),
                    }));
                }
                bindings.push((reference.id(), reference.clone()));
            }
            // Every in-repo multi-holder path acquires guards in ascending identity order. This ordering is the
            // deadlock-prevention invariant for overlapping stateful calls and must be preserved while guards remain
            // held from submission through hidden-state replacement commit below.
            bindings.sort_by_key(|(id, _)| *id);
            let guard_indices =
                bindings.iter().enumerate().map(|(index, (id, _))| (*id, index)).collect::<BTreeMap<_, _>>();

            let mut mutated_outputs = BTreeMap::new();
            for (state, &logical_input_index) in program.reference_states.iter().zip(&reference_state_input_indices) {
                let Value::Reference(reference) = &inputs[logical_input_index] else {
                    unreachable!("reference bindings were validated before holder acquisition")
                };
                let guard_index = guard_indices[&reference.handle().id()];
                if let Some(logical_output_index) = state.output_index() {
                    mutated_outputs.insert(guard_index, logical_output_index);
                }
            }
            let mut inputs = Some(inputs);

            let donation_overrides =
                reference_state_input_indices.iter().map(|&index| (index, false)).collect::<Vec<_>>();
            let (
                execution,
                input_refinements,
                publication,
                replacement_transactions,
                read_only_guards,
                prepared_replacements,
            ) = 'snapshot: loop {
                let guards = bindings
                    .iter()
                    .map(|(_, reference)| Self::lock_stateful_reference(reference))
                    .collect::<Result<Vec<ReadyOrPendingReferenceGuard<'_, Array<'c>>>, Error>>()?;
                let observations =
                    guards.iter().map(ReadyOrPendingReferenceGuard::observe).collect::<Result<Vec<_>, _>>()?;
                let logical_inputs = inputs.as_ref().unwrap();
                let mut execution_inputs = logical_inputs.clone();
                for &logical_input_index in &reference_state_input_indices {
                    let Value::Reference(reference) = &logical_inputs[logical_input_index] else {
                        unreachable!("reference bindings were validated before holder acquisition")
                    };
                    let snapshot = observations[guard_indices[&reference.handle().id()]].snapshot().clone();
                    if snapshot.mesh() != &program.mesh {
                        return Err(Error::from(ProgramError::InvalidArgument {
                            message: format!(
                                "external state input {logical_input_index} reference mesh does not match the compiled \
                                 device mesh",
                            ),
                        }));
                    }
                    execution_inputs[logical_input_index] = Value::Array(snapshot);
                }
                drop(guards);

                let arrays = execution_inputs
                    .into_iter()
                    .map(ValueProjection::<ArrayType>::into_projected)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(ProgramError::from)?;
                let prepared_execution =
                    self.prepare_compiled_execution(program, arrays, donation_overrides.as_slice())?;

                let guards = bindings
                    .iter()
                    .map(|(_, reference)| Self::lock_stateful_reference(reference))
                    .collect::<Result<Vec<ReadyOrPendingReferenceGuard<'_, Array<'c>>>, Error>>()?;
                if !guards.iter().zip(&observations).all(|(guard, observation)| observation.is_current(guard)) {
                    continue;
                }
                let mut read_only_guards = Vec::with_capacity(bindings.len() - mutated_outputs.len());
                let mut prepared_replacements = Vec::with_capacity(mutated_outputs.len());
                let mut read_leases = Vec::new();
                for (guard_index, guard) in guards.into_iter().enumerate() {
                    if let Some(&logical_output_index) = mutated_outputs.get(&guard_index) {
                        match guard.prepare_replacement()? {
                            ReferenceReplacementPreparation::Prepared(prepared) => {
                                prepared_replacements.push((logical_output_index, prepared));
                            }
                            ReferenceReplacementPreparation::Waiting(mut leases) => read_leases.append(&mut leases),
                        }
                    } else {
                        read_only_guards.push(guard);
                    }
                }
                if !read_leases.is_empty() {
                    drop(prepared_replacements);
                    drop(read_only_guards);
                    for lease in read_leases {
                        match lease.r#await() {
                            Ok(()) | Err(_) => {
                                // A completed read lease releases its input snapshot regardless of execution outcome.
                            }
                        }
                    }
                    continue;
                }

                // Current observations still describe these snapshots. A pending observation may have become ready at
                // the same generation; retaining its now-completed dependency is harmless and keeps one coherent
                // value-generation-dependency tuple throughout the retry attempt.
                let dependencies =
                    observations.iter().filter_map(|observation| observation.dependency().cloned()).collect::<Vec<_>>();
                // Observation validation under the ordered guards commits this attempt. Dropping the original logical
                // inputs here releases only retry ownership, allowing ordinary inputs to remain donatable.
                drop(inputs.take());
                // Nothing has been submitted or published yet, so a failed projection simply releases the guards.
                let (execution_inputs, input_refinements, physical_output_count) =
                    prepared_execution.into_arguments()?;
                let read_only_guard_storage = RefCell::new(read_only_guards);
                let prepared_replacement_storage = RefCell::new(prepared_replacements);
                let publication: RefCell<Option<CoreReferenceCompletion>> = RefCell::new(None);
                let replacement_transactions = RefCell::new(Vec::with_capacity(mutated_outputs.len()));
                let execution = self.execute_pjrt_buffers(program, execution_inputs, physical_output_count, |fence| {
                    let mut completions = dependencies.clone();
                    completions.push(CoreReferenceCompletion::new(ReferenceCompletion { fence: fence.clone() }));
                    #[cfg(test)]
                    if let Some(injected) = take_stateful_completion_injection() {
                        completions.push(injected);
                    }
                    let completion = CoreReferenceCompletion::joined(completions);
                    for guard in read_only_guard_storage.borrow_mut().iter_mut() {
                        guard.preserve_value_until(completion.clone());
                    }
                    let mut transactions = replacement_transactions.borrow_mut();
                    for (logical_output_index, prepared) in
                        std::mem::take(&mut *prepared_replacement_storage.borrow_mut())
                    {
                        transactions.push((logical_output_index, prepared.begin(completion.clone())));
                    }
                    *publication.borrow_mut() = Some(completion);
                });
                break 'snapshot (
                    execution,
                    input_refinements,
                    publication.into_inner(),
                    replacement_transactions.into_inner(),
                    read_only_guard_storage.into_inner(),
                    prepared_replacement_storage.into_inner(),
                );
            };
            let Some(completion) = publication else {
                drop(prepared_replacements);
                drop(read_only_guards);
                return match execution {
                    Err(error) => Ok(StatefulExecution::ready(Err(error))),
                    Ok(_) => unreachable!("successful PJRT submission must publish reference state"),
                };
            };
            let execution = match execution {
                Ok(execution) => execution,
                Err(error) => {
                    let reason = error.to_string();
                    for (_, transaction) in replacement_transactions {
                        transaction.poison(reason.clone());
                    }
                    drop(read_only_guards);
                    return Ok(StatefulExecution::pending(Err(error), completion, |reason| {
                        Error::Execution(ExecutionError::AsynchronousFailure { reason })
                    }));
                }
            };
            let (physical_outputs, fence) = execution.into_parts();
            let mut physical_outputs = physical_outputs.into_iter().map(Some).collect::<Vec<_>>();
            debug_assert!(prepared_replacements.is_empty());
            let replacements = (|| -> Result<Vec<_>, Error> {
                #[cfg(test)]
                if take_stateful_failure_injection(StatefulFailureInjection::BeforeHiddenReplacementCommit) {
                    return Err(Error::Pjrt(ryft_pjrt::Error::internal(
                        "injected failure before hidden state replacement commit".to_string(),
                    )));
                }
                let hidden_outputs = reconstruct_compiled_outputs(
                    self,
                    program,
                    &mut physical_outputs,
                    fence.clone(),
                    program.output_count..program.output_types.len(),
                )?;
                let mut hidden_outputs =
                    (program.output_count..program.output_types.len()).zip(hidden_outputs).collect::<BTreeMap<_, _>>();
                if replacement_transactions.len() != mutated_outputs.len() {
                    return Err(ProgramError::MalformedProgram(
                        "submitted reference replacement count does not match mutated reference count".to_string(),
                    )
                    .into());
                }
                let replacements = replacement_transactions
                    .iter()
                    .map(|(logical_output_index, _)| {
                        hidden_outputs.remove(logical_output_index).ok_or_else(|| {
                            ProgramError::MalformedProgram("hidden state output was claimed twice".to_string())
                        })
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                if !hidden_outputs.is_empty() {
                    return Err(ProgramError::MalformedProgram(
                        "hidden state output was not claimed by a mutated reference".to_string(),
                    )
                    .into());
                }
                Ok(replacements)
            })();
            let replacements = match replacements {
                Ok(replacements) => replacements,
                Err(error) => {
                    let reason = error.to_string();
                    for (_, transaction) in replacement_transactions {
                        transaction.poison(reason.clone());
                    }
                    drop(read_only_guards);
                    return Ok(StatefulExecution::pending(Err(error), completion, |reason| {
                        Error::Execution(ExecutionError::AsynchronousFailure { reason })
                    }));
                }
            };
            let mut submitted_replacements = replacement_transactions.into_iter().zip(replacements);
            let mut validated_replacements = Vec::with_capacity(mutated_outputs.len());
            while let Some(((_, transaction), replacement)) = submitted_replacements.next() {
                match transaction.validate(replacement) {
                    Ok(replacement) => validated_replacements.push(replacement),
                    Err(error) => {
                        let reason = error.to_string();
                        for replacement in validated_replacements {
                            replacement.poison(reason.clone());
                        }
                        for ((_, transaction), _) in submitted_replacements {
                            transaction.poison(reason.clone());
                        }
                        drop(read_only_guards);
                        return Ok(StatefulExecution::pending(Err(error.into()), completion, |reason| {
                            Error::Execution(ExecutionError::AsynchronousFailure { reason })
                        }));
                    }
                }
            }
            let committed_replacement_guards =
                validated_replacements.into_iter().map(|replacement| replacement.commit()).collect::<Vec<_>>();
            // Mutated hidden outputs are statically shaped under the current ABI, so neither buffer logical-shape
            // queries nor runtime-extent materialization can wait for device execution during replacement commit.
            // Release every reference guard before public reconstruction, where dynamic outputs may require either
            // wait.
            drop(committed_replacement_guards);
            drop(read_only_guards);

            let public_result = (|| -> Result<_, Error> {
                #[cfg(test)]
                if take_stateful_failure_injection(StatefulFailureInjection::BeforePublicReconstruction) {
                    return Err(Error::Pjrt(ryft_pjrt::Error::internal(
                        "injected failure before public output reconstruction".to_string(),
                    )));
                }
                let public_outputs =
                    reconstruct_compiled_outputs(self, program, &mut physical_outputs, fence, 0..program.output_count)?;
                validate_compiled_output_refinements(
                    program,
                    &input_refinements,
                    program.output_types(),
                    &public_outputs,
                )?;
                if physical_outputs.iter().any(Option::is_some) {
                    return Err(ProgramError::MalformedProgram(
                        "executable returned an unclaimed physical output".to_string(),
                    )
                    .into());
                }
                validate_runtime_outputs(program.output_types(), &public_outputs)?;
                Ok(public_outputs.into_iter().map(Value::Array).collect())
            })();
            Ok(StatefulExecution::pending(public_result, completion, |reason| {
                Error::Execution(ExecutionError::AsynchronousFailure { reason })
            }))
        })();
        match prepared {
            Ok(execution) => execution,
            Err(error) => StatefulExecution::ready(Err(error)),
        }
    }
}

impl<'c> StatefulCompilationDomain for Domain<'c> {
    fn execute_statefully(
        &self,
        program: &CompiledProgram<'c>,
        inputs: Vec<Value<'c>>,
    ) -> StatefulExecution<Vec<Value<'c>>, Self::Error> {
        self.execute_stateful_async(program, inputs)
    }
}

impl<'c> Domain<'c> {
    /// Executes a program with array inputs. Output arrays retain completion, while an execution without outputs or
    /// effects waits here so failures remain observable.
    pub(super) fn execute_program(
        &self,
        program: &CompiledProgram<'c>,
        inputs: Vec<Array<'c>>,
    ) -> Result<Vec<Array<'c>>, Error> {
        let execution = self.execute_compiled_async(program, inputs)?;
        if execution.output().is_empty() && !program.signature.has_effects() {
            // No output array carries the execution fence out of this call, so an asynchronous execution error would
            // otherwise be unobservable. Block here so this synchronous entry point still reports it.
            let (outputs, fence) = execution.into_parts();
            fence.block_until_ready()?;
            return Ok(outputs);
        }
        Ok(execution.into_output())
    }
}

/// Returns device IDs in replica-major, then computation-partition-major order.
pub(super) fn flatten_device_assignment(assignment: &ryft_pjrt::DeviceAssignment) -> Result<Vec<u64>, Error> {
    let mut devices = Vec::with_capacity(assignment.replica_count() * assignment.computation_count());
    for replica in 0..assignment.replica_count() {
        for partition in 0..assignment.computation_count() {
            devices.push(assignment.device_id(replica, partition)? as u64);
        }
    }
    Ok(devices)
}

/// Validates runtime refinement of a declared logical input type, excluding sharding because placement is resolved
/// separately at the call boundary.
pub(crate) fn validate_input_type(declared: &ArrayType, actual: &ArrayType) -> Result<(), Error> {
    // Calls of a specialization ordinarily pass inputs of exactly its declared types, which refine them trivially.
    if declared == actual {
        return Ok(());
    }
    let declared_without_sharding = declared.clone().with_sharding(None)?;
    let actual_without_sharding = actual.clone().with_sharding(None)?;
    if declared_without_sharding.is_refined_by(&actual_without_sharding) {
        Ok(())
    } else {
        Err(ProgramError::InvalidArgument {
            message: format!("runtime input type {actual} does not refine declared type {declared}"),
        }
        .into())
    }
}

/// Validates output arity and refinement of the declared logical output types, including sharding.
pub(crate) fn validate_runtime_outputs(declared: &[ArrayType], outputs: &[Array<'_>]) -> Result<(), Error> {
    if declared.len() != outputs.len() {
        return Err(ProgramError::InvalidOutputCount { expected: declared.len(), actual: outputs.len() }.into());
    }
    for (declared, actual) in declared.iter().zip(outputs.iter().map(Typed::r#type)) {
        if declared != actual.as_ref() && !declared.is_refined_by(actual.as_ref()) {
            return Err(ProgramError::InvalidArgument {
                message: format!("runtime output type {actual} does not refine declared type {declared}"),
            }
            .into());
        }
    }
    Ok(())
}

/// Reshards `inputs` to match `expected_shardings` at the call boundary. Inputs that already
/// match skip the reshard entirely; the implicit-reshard path is the cold path.
fn reshard_inputs_if_needed<'c>(
    domain: &Domain<'c>,
    mesh: &DeviceMesh,
    expected_shardings: &[Sharding],
    inputs: Vec<Array<'c>>,
) -> Result<Vec<Array<'c>>, Error> {
    let needs_reshard = inputs.iter().zip(expected_shardings).any(|(array, expected)| array.sharding() != expected);
    if !needs_reshard {
        return Ok(inputs);
    }
    inputs
        .into_iter()
        .zip(expected_shardings)
        .map(|(array, expected)| {
            if array.sharding() == expected {
                Ok(array)
            } else {
                crate::arrays_v0::compiled_reshard::reshard(&array, domain, mesh, expected)
            }
        })
        .collect()
}

/// Host copy that has been issued for one bounded input requiring padding.
struct PendingBoundedInputHostCopy<'c> {
    /// Logical input index used to recover the runtime array metadata.
    logical_input_index: usize,

    /// Physical executable input slot to populate.
    physical_input_index: usize,

    /// Static executable type defining the padding target.
    physical_type: ArrayType,

    /// Concrete logical shape copied from the runtime input.
    actual_shape: StaticShape,

    /// Static executable-bound shape used for physical storage.
    physical_shape: StaticShape,

    /// Already-issued host copy for the logical input.
    host_copy: DenseArrayHostCopy,

    /// Single-flight reservation that retains success.
    producer: BoundedMaterializationProducer<'c>,
}

/// Bound-shaped upload issued for one padded input but not yet published to its cache or executable slot.
struct PendingBoundedInputPublication<'c> {
    /// Physical executable input slot to populate after readiness succeeds.
    physical_input_index: usize,

    /// Uploaded physical array whose asynchronous transfer must become ready before publication.
    physical: Array<'c>,

    /// Single-flight reservation that retains ready success.
    producer: BoundedMaterializationProducer<'c>,
}

/// Cache waiter retained until all producer copies in this call have completed.
struct WaitingBoundedInputMaterialization<'c> {
    /// Logical input index used to retry production if the prior producer fails.
    logical_input_index: usize,

    /// Physical executable input slot to populate.
    physical_input_index: usize,

    /// Static executable type defining the cache key and padding target.
    physical_type: ArrayType,

    /// Concrete logical shape copied from the runtime input on a retry.
    actual_shape: StaticShape,

    /// Static executable-bound shape used for physical storage on a retry.
    physical_shape: StaticShape,

    /// Non-spinning single-flight wait handle.
    waiter: BoundedMaterializationWaiter<'c>,
}

/// Deterministic work and transport report for one bounded-input materialization boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BoundedInputMaterializationReport {
    /// Inputs that reused original device buffers because the runtime shape equaled the bound.
    at_bound_reuses: usize,

    /// Ready retained materializations reused without padding work or transport.
    cache_hits: usize,

    /// Missing cache entries whose successful production was retained.
    retained_misses: usize,

    /// Device-to-host shard copies issued by padding work.
    device_to_host_shard_copies: usize,

    /// Full-global host merge buffers allocated after shard copies.
    host_merge_buffer_allocations: usize,

    /// O(bound) host padding payloads allocated.
    host_padding_payload_allocations: usize,

    /// Bound-shaped host-to-device shard uploads.
    host_to_device_shard_uploads: usize,

    /// Hidden logical-extent scalar host-to-device uploads.
    extent_scalar_uploads: usize,

    /// Sum of logical payload bytes for bounded inputs.
    actual_bytes: usize,

    /// Sum of physical-bound payload bytes for bounded inputs.
    bound_bytes: usize,
}

/// Physical inputs and their deterministic bounded-materialization work report.
struct MaterializedBoundedInputs<'c> {
    /// Physical executable inputs, including hidden extent scalars.
    inputs: Vec<Array<'c>>,

    /// Path-local work and transport counts.
    #[cfg_attr(not(test), allow(dead_code))]
    report: BoundedInputMaterializationReport,
}

/// Selects whether a bounded input can reuse its device buffer or requires host padding.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BoundedInputPacking {
    /// The runtime shape already equals the executable's physical bound.
    Reuse,

    /// At least one runtime extent is smaller than its executable bound.
    Pad,
}

impl BoundedInputPacking {
    /// Determines the bounded-input packing tier from concrete runtime and executable shapes.
    fn from_shapes(actual_shape: &StaticShape, physical_shape: &StaticShape) -> Self {
        if actual_shape == physical_shape { Self::Reuse } else { Self::Pad }
    }
}

/// Completes one already-issued host copy and issues its bound-shaped upload without publishing it.
fn upload_bounded_input_host_copy<'c>(
    domain: &Domain<'c>,
    input: &Array<'c>,
    pending: PendingBoundedInputHostCopy<'c>,
    report: &mut BoundedInputMaterializationReport,
) -> Result<PendingBoundedInputPublication<'c>, Error> {
    report.device_to_host_shard_copies += pending.host_copy.shard_copy_count();
    let (source_bytes, allocated_merge_buffer) = pending.host_copy.finish_with_measurements()?;
    report.host_merge_buffer_allocations += usize::from(allocated_merge_buffer);
    let (padded_bytes, padding_payload_allocations) = pad_dense_array_bytes_with_target_allocation_count(
        source_bytes.as_slice(),
        pending.actual_shape.as_slice(),
        pending.physical_shape.as_slice(),
        pending.physical_type.data_type(),
    )?;
    report.host_padding_payload_allocations += padding_payload_allocations;
    let physical = Array::from_host_buffer(domain, pending.physical_type, input.mesh().clone(), padded_bytes)?;
    report.host_to_device_shard_uploads += physical.shards().iter().filter(|shard| shard.buffer().is_some()).count();
    Ok(PendingBoundedInputPublication {
        physical_input_index: pending.physical_input_index,
        physical,
        producer: pending.producer,
    })
}

/// Publishes one batch only after every asynchronously uploaded physical array passes `check_readiness`.
fn publish_bounded_input_uploads<'c>(
    publications: Vec<PendingBoundedInputPublication<'c>>,
    physical_inputs: &mut [Array<'c>],
    check_readiness: &mut impl FnMut(&[PendingBoundedInputPublication<'c>]) -> Result<(), Error>,
) -> Result<(), Error> {
    // Readiness is checked for the complete batch before any producer publishes. An error therefore drops every
    // reservation in the batch, caches no failed upload, and leaves all concurrent H2D work issued.
    check_readiness(publications.as_slice())?;
    for publication in publications {
        physical_inputs[publication.physical_input_index] = publication.producer.complete(publication.physical);
    }
    Ok(())
}

/// Pads bounded-dynamic inputs to static executable storage and appends their logical extents.
///
/// Dispatch has two tiers in this boundary: (1) an actual shape equal to the physical bound reuses the source array;
/// and (2) a smaller value uses a clone-shared retained materialization, with one single-flight producer per
/// structural `(ZeroOriginV1, physical ArrayType)` key. Below-bound zero-space values are deferred to
/// [`materialize_zero_space_carriers`], which synthesizes their bound-shaped carriers directly, and below-bound values
/// with non-addressable shards are rejected because padding must read every shard's bytes on this process. Every
/// producer D2H copy is issued during the first scan. The resulting H2D uploads are also all issued before any upload
/// readiness wait, and retained values are published only after the complete batch succeeds. Cold work is O(bound)
/// bytes plus O(shard-segment-count) overlap bookkeeping; ready hits perform no pad-path allocation or transport. A
/// failed asynchronous upload stores no cache entry and a later call retries the ordinary cold path.
fn materialize_bounded_dynamic_inputs<'c>(
    domain: &Domain<'c>,
    signature: &ExecutableSignature,
    declared_types: &[ArrayType],
    actual_types: &[ArrayType],
    inputs: Vec<Array<'c>>,
) -> Result<MaterializedBoundedInputs<'c>, Error> {
    materialize_bounded_dynamic_inputs_with_readiness(
        domain,
        signature,
        declared_types,
        actual_types,
        inputs,
        |publications| {
            for publication in publications {
                publication.physical.block_until_ready()?;
            }
            Ok(())
        },
    )
}

/// Implements bounded-input materialization with an injectable batch readiness check for deterministic failure tests.
fn materialize_bounded_dynamic_inputs_with_readiness<'c>(
    domain: &Domain<'c>,
    signature: &ExecutableSignature,
    declared_types: &[ArrayType],
    actual_types: &[ArrayType],
    inputs: Vec<Array<'c>>,
    mut check_readiness: impl FnMut(&[PendingBoundedInputPublication<'c>]) -> Result<(), Error>,
) -> Result<MaterializedBoundedInputs<'c>, Error> {
    assert_eq!(declared_types.len(), inputs.len());
    assert_eq!(actual_types.len(), inputs.len());
    let physical_types = signature.physical_input_types(declared_types);
    let mut physical_inputs = signature.project_inputs(inputs.as_slice());
    let mut pending_host_copies = Vec::new();
    let mut waiting_materializations = Vec::new();
    let mut report = BoundedInputMaterializationReport::default();
    for logical_input_index in 0..inputs.len() {
        if !signature
            .input_dimensions()
            .iter()
            .any(|input_dimension| input_dimension.logical_input_index() == logical_input_index)
        {
            continue;
        }
        let physical_input_index = signature.input_mapping()[logical_input_index].unwrap();
        let physical_type = physical_types[physical_input_index].clone();
        let physical_shape = physical_type.static_shape().ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("bounded-dynamic executable input {logical_input_index} has no static upper shape"),
        })?;
        let actual_shape =
            actual_types[logical_input_index].static_shape().ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("runtime executable input {logical_input_index} must have a static shape"),
            })?;
        report.actual_bytes += actual_types[logical_input_index].size_in_bytes()?;
        report.bound_bytes += physical_type.size_in_bytes()?;
        if BoundedInputPacking::from_shapes(&actual_shape, &physical_shape) == BoundedInputPacking::Reuse {
            report.at_bound_reuses += 1;
            continue;
        }

        // A below-bound zero-space input carries no payload bytes to pad or cache: its dense host copy is empty, so
        // the retained-cache and byte-pad tiers below must not run. The zero-space carrier materialization that
        // follows this boundary synthesizes its bound-shaped all-false predicate directly instead.
        if physical_type.data_type().is_zero() {
            continue;
        }

        let input = &inputs[logical_input_index];
        if input.supports_bounded_materialization_cache() {
            match input.probe_bounded_materialization(BoundedMaterializationKey::new(physical_type.clone())) {
                BoundedMaterializationProbe::Hit(physical) => {
                    report.cache_hits += 1;
                    physical_inputs[physical_input_index] = physical;
                }
                BoundedMaterializationProbe::Produce(producer) => {
                    reject_unreduced_host_round_trip(input, "pad")?;
                    report.retained_misses += 1;
                    pending_host_copies.push(PendingBoundedInputHostCopy {
                        logical_input_index,
                        physical_input_index,
                        physical_type,
                        actual_shape,
                        physical_shape,
                        host_copy: begin_materialize_dense_array_bytes(input)?,
                        producer,
                    });
                }
                BoundedMaterializationProbe::Wait(waiter) => {
                    waiting_materializations.push(WaitingBoundedInputMaterialization {
                        logical_input_index,
                        physical_input_index,
                        physical_type,
                        actual_shape,
                        physical_shape,
                        waiter,
                    });
                }
            }
        } else {
            // Padding reads every shard's bytes on this process, so an input with a non-addressable shard cannot be
            // materialized here at all; reject it explicitly instead of failing deep inside the host copy.
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "bounded-dynamic executable input {logical_input_index} is below its bound but has \
                     non-addressable shards, so it cannot be padded on this process",
                ),
            }
            .into());
        }
    }

    let mut pending_publications = Vec::with_capacity(pending_host_copies.len());
    for pending in pending_host_copies {
        let input = &inputs[pending.logical_input_index];
        pending_publications.push(upload_bounded_input_host_copy(domain, input, pending, &mut report)?);
    }
    publish_bounded_input_uploads(pending_publications, &mut physical_inputs, &mut check_readiness)?;

    for waiting in waiting_materializations {
        let WaitingBoundedInputMaterialization {
            logical_input_index,
            physical_input_index,
            physical_type,
            actual_shape,
            physical_shape,
            waiter,
        } = waiting;
        match waiter.resolve() {
            Ok(physical) => {
                report.cache_hits += 1;
                physical_inputs[physical_input_index] = physical;
            }
            Err(producer) => {
                // The prior producer failed and removed its reservation. This waiter is now the retry producer. Its
                // copy necessarily starts after that failure; the ordinary all-miss path above stays fully concurrent.
                report.retained_misses += 1;
                let input = &inputs[logical_input_index];
                reject_unreduced_host_round_trip(input, "pad")?;
                let pending = PendingBoundedInputHostCopy {
                    logical_input_index,
                    physical_input_index,
                    physical_type,
                    actual_shape,
                    physical_shape,
                    host_copy: begin_materialize_dense_array_bytes(input)?,
                    producer,
                };
                let publication = upload_bounded_input_host_copy(domain, input, pending, &mut report)?;
                publish_bounded_input_uploads(vec![publication], &mut physical_inputs, &mut check_readiness)?;
            }
        }
    }

    for input_dimension in signature.input_dimensions() {
        let input = &inputs[input_dimension.logical_input_index()];
        let extent = actual_types[input_dimension.logical_input_index()].shape().dimensions()[input_dimension.axis()]
            .value()
            .unwrap();
        let extent = i32::try_from(extent).map_err(|_| ProgramError::InvalidArgument {
            message: format!("runtime dimension extent {extent} does not fit in a StableHLO i32 scalar"),
        })?;
        let (extent_scalar, uploaded) = input.logical_extent_scalar(domain, extent)?;
        report.extent_scalar_uploads += usize::from(uploaded);
        physical_inputs.push(extent_scalar);
    }
    assert_eq!(physical_inputs.len(), signature.array_input_count());
    Ok(MaterializedBoundedInputs { inputs: physical_inputs, report })
}

/// Zero-pads one dense row-major array into a larger shape of the same rank.
#[cfg(test)]
fn pad_dense_array_bytes(
    source: &[u8],
    source_shape: &[usize],
    target_shape: &[usize],
    data_type: DataType,
) -> Result<Vec<u8>, Error> {
    pad_dense_array_bytes_with_target_allocation_count(source, source_shape, target_shape, data_type)
        .map(|(bytes, _)| bytes)
}

/// Zero-pads one dense row-major array and reports the number of target payload allocations.
///
/// The count deliberately covers the O(bound) payload allocation, which dominates this path. Small O(rank) stride
/// vectors are implementation metadata and are not included.
fn pad_dense_array_bytes_with_target_allocation_count(
    source: &[u8],
    source_shape: &[usize],
    target_shape: &[usize],
    data_type: DataType,
) -> Result<(Vec<u8>, usize), Error> {
    if source_shape.len() != target_shape.len()
        || source_shape.iter().zip(target_shape).any(|(source, target)| source > target)
    {
        return Err(ProgramError::InvalidArgument {
            message: format!(
                "cannot pad runtime shape {:?} into bounded physical shape {:?}",
                source_shape, target_shape,
            ),
        }
        .into());
    }
    let element_size = data_type.to_pjrt().element_size_in_bytes()?;
    let mut source_strides = vec![1usize; source_shape.len()];
    let mut target_strides = vec![1usize; target_shape.len()];
    for axis in (0..source_shape.len().saturating_sub(1)).rev() {
        source_strides[axis] = source_strides[axis + 1].checked_mul(source_shape[axis + 1]).ok_or_else(|| {
            ProgramError::InvalidArgument { message: "runtime input shape strides exceed usize".to_string() }
        })?;
        target_strides[axis] = target_strides[axis + 1].checked_mul(target_shape[axis + 1]).ok_or_else(|| {
            ProgramError::InvalidArgument { message: "bounded physical shape strides exceed usize".to_string() }
        })?;
    }
    let target_elements = target_shape.iter().try_fold(1usize, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| ProgramError::InvalidArgument {
            message: "bounded physical input element count exceeds usize".to_string(),
        })
    })?;
    let mut target = vec![
        0u8;
        target_elements.checked_mul(element_size).ok_or_else(|| ProgramError::InvalidArgument {
            message: "bounded physical input byte count exceeds usize".to_string(),
        })?
    ];
    if source_shape.is_empty() {
        target.copy_from_slice(source);
        let allocation_count = usize::from(!target.is_empty());
        return Ok((target, allocation_count));
    }
    copy_dense_array_into_padding(
        source,
        &mut target,
        source_shape,
        source_strides.as_slice(),
        target_strides.as_slice(),
        element_size,
        0,
        0,
        0,
    );
    let allocation_count = usize::from(!target.is_empty());
    Ok((target, allocation_count))
}

/// Recursively copies one dense row-major source into the origin-aligned region of a padded target.
#[allow(clippy::too_many_arguments)]
fn copy_dense_array_into_padding(
    source: &[u8],
    target: &mut [u8],
    source_shape: &[usize],
    source_strides: &[usize],
    target_strides: &[usize],
    element_size: usize,
    axis: usize,
    source_offset: usize,
    target_offset: usize,
) {
    if axis + 1 == source_shape.len() {
        let byte_count = source_shape[axis] * element_size;
        let source_offset = source_offset * element_size;
        let target_offset = target_offset * element_size;
        target[target_offset..target_offset + byte_count]
            .copy_from_slice(&source[source_offset..source_offset + byte_count]);
        return;
    }
    for index in 0..source_shape[axis] {
        copy_dense_array_into_padding(
            source,
            target,
            source_shape,
            source_strides,
            target_strides,
            element_size,
            axis + 1,
            source_offset + index * source_strides[axis],
            target_offset + index * target_strides[axis],
        );
    }
}

/// Materializes private false-predicate carriers for zero-space inputs that remain in the physical executable
/// signature, currently because their declared type contains dynamic dimensions. Static zero-space inputs are erased
/// before this point and never allocate a carrier.
///
/// Each carrier is allocated at the input's *physical* (bound-shaped) type from `physical_input_types`, not at the
/// value's runtime type: the compiled module declares the bound-shaped `i1` argument, and the hidden extent scalar
/// transports the logical size, so a below-bound zero-space input must still hand PJRT a bound-shaped buffer.
fn materialize_zero_space_carriers<'c>(
    domain: &Domain<'c>,
    physical_input_types: &[ArrayType],
    inputs: Vec<Array<'c>>,
) -> Result<Vec<Array<'c>>, Error> {
    assert_eq!(physical_input_types.len(), inputs.len());
    physical_input_types
        .iter()
        .zip(inputs)
        .map(|(physical_type, input)| {
            if !physical_type.data_type().is_zero() {
                return Ok(input);
            }
            input.block_until_ready()?;
            let carrier_type = physical_type.clone().with_data_type(DataType::Boolean);
            let element_count =
                carrier_type.element_count()?.expect("physical bounded input types should only have static shapes");
            Ok(Array::from_host_buffer(domain, carrier_type, input.mesh().clone(), vec![0u8; element_count])?)
        })
        .collect()
}

/// Reconstructs selected logical outputs from one submitted execution's physical buffers.
fn reconstruct_compiled_outputs<'c>(
    domain: &Domain<'c>,
    program: &CompiledProgram<'c>,
    physical_outputs: &mut [Option<Vec<Buffer<'c>>>],
    fence: ryft_pjrt::ExecutionFence,
    logical_indices: impl IntoIterator<Item = usize>,
) -> Result<Vec<Array<'c>>, Error> {
    // Programs with only static outputs build each output directly from its planned layout, without querying the
    // returned buffers or re-deriving their placement.
    if let Some(layouts) = &program.plan.output_layouts {
        return logical_indices
            .into_iter()
            .map(|logical_index| {
                let buffers = match program.signature.output_mapping()[logical_index] {
                    Some(physical_index) => {
                        physical_outputs.get_mut(physical_index).and_then(Option::take).ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "physical output {physical_index} for logical output {logical_index} is missing or \
                                 was claimed twice",
                            ))
                        })?
                    }
                    None => Vec::new(),
                };
                Ok(Array::from_execution_output(domain, &layouts[logical_index], buffers, fence.clone()))
            })
            .collect();
    }
    let logical_indices = logical_indices.into_iter().collect::<Vec<_>>();
    let scalar_type = ArrayType::scalar(DataType::I64).replicated(&program.mesh)?;
    let mut output_extents = BTreeMap::new();
    for output_dimension in program
        .signature
        .output_dimensions()
        .iter()
        .filter(|dimension| logical_indices.contains(&dimension.logical_output_index()))
    {
        let physical_output_index = output_dimension.physical_output_index();
        let physical_output =
            physical_outputs.get_mut(physical_output_index).and_then(Option::take).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "physical output {physical_output_index} for runtime extent of output {} axis {} is missing or was \
                     claimed twice",
                    output_dimension.logical_output_index(),
                    output_dimension.axis(),
                ))
            })?;
        let scalar = Array::from_canonical_addressable_buffers(
            domain,
            scalar_type.clone(),
            program.mesh.clone(),
            physical_output,
        )?
        .with_execution_fence(fence.clone());
        let bytes = materialize_dense_array_bytes(&scalar)?;
        let extent = bytes
            .get(..size_of::<i64>())
            .map(|bytes| i64::from_ne_bytes(bytes.try_into().unwrap()))
            .and_then(|extent| usize::try_from(extent).ok())
            .ok_or_else(|| ProgramError::InvalidArgument {
                message: format!(
                    "runtime extent for output {} axis {} is negative, missing, or exceeds usize",
                    output_dimension.logical_output_index(),
                    output_dimension.axis(),
                ),
            })?;
        output_extents.insert((output_dimension.logical_output_index(), output_dimension.axis()), extent);
    }

    logical_indices
        .into_iter()
        .map(|logical_index| {
            let output_type = &program.output_types[logical_index];
            match program.signature.output_mapping()[logical_index] {
                Some(physical_index) => {
                    let dimensions = output_type
                        .shape()
                        .dimensions()
                        .iter()
                        .enumerate()
                        .map(|(axis, dimension)| {
                            dimension
                                .value()
                                .or_else(|| output_extents.get(&(logical_index, axis)).copied())
                                .map_or_else(
                                    || {
                                        Err(ProgramError::MalformedProgram(format!(
                                            "executable omitted runtime extent for output {logical_index} axis {axis}",
                                        )))
                                    },
                                    |extent| Ok(CoreDimension::Static(extent)),
                                )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let output_type = output_type.clone().with_shape(Shape::new(dimensions));
                    let output_type = match output_type.sharding() {
                        Some(_) => output_type,
                        None => output_type.replicated(&program.mesh)?,
                    };
                    Ok(Array::from_canonical_addressable_buffers(
                        domain,
                        output_type,
                        program.mesh.clone(),
                        physical_outputs.get_mut(physical_index).and_then(Option::take).ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "physical output {physical_index} for logical output {logical_index} is missing or was \
                                 claimed twice",
                            ))
                        })?,
                    )?
                    .with_execution_fence(fence.clone()))
                }
                None => Ok(Array::from_zero_space(domain, output_type.clone(), program.mesh.clone())?
                    .with_execution_fence(fence.clone())),
            }
        })
        .collect()
}

/// Validates reconstructed logical outputs against the runtime input refinements.
fn validate_compiled_output_refinements(
    program: &CompiledProgram<'_>,
    input_refinements: &<ArrayType as Type>::Refinements,
    declared_output_types: &[ArrayType],
    outputs: &[Array<'_>],
) -> Result<(), Error> {
    let mut closed_identities = Vec::new();
    for (_, identity) in program.input_types.iter().chain(program.output_types.iter()).flat_map(Type::identities) {
        if !closed_identities.contains(identity) {
            closed_identities.push(identity.clone());
        }
    }
    input_refinements
        .validate_refinement(
            declared_output_types.iter(),
            outputs.iter().map(|output| output.r#type().into_owned()),
            closed_identities.as_slice(),
        )
        .map_err(ProgramError::from)
        .map_err(Into::into)
}

impl<'c> Domain<'c> {
    /// Submits at the common physical boundary and transposes array results, retaining hidden effect tokens in the
    /// active scope. No scope lock is held across PJRT, which may execute callbacks inline on this thread.
    fn execute_pjrt_buffers(
        &self,
        program: &CompiledProgram<'c>,
        mut arguments: ExecuteArguments<'c>,
        array_output_count: usize,
        after_submission: impl FnOnce(&ExecutionFence),
    ) -> Result<Execution<Vec<Vec<Buffer<'c>>>>, Error> {
        ensure_effect_dispatch_allowed()?;
        let addressable_device_count = arguments.addressable_device_ids().len();
        let ordered = program.signature.has_ordered_io();
        if ordered {
            if !self.target()?.is_fully_addressable_single_process_mesh(&program.mesh) {
                return Err(ExecutionError::UnsupportedEffectDeviceAssignment.into());
            }
            if addressable_device_count > 1 && !program.signature.has_device_ordered_io() {
                return Err(ExecutionError::MissingDeviceOrderedEffect.into());
            }
        }
        let device = ordered.then(|| arguments.addressable_device_ids().to_vec());
        let mut reservation =
            if program.signature.has_effects() { Some(self.effect_scope.reserve(device)?) } else { None };
        if let Some(reservation) = &reservation {
            if ordered {
                let predecessor = reservation
                    .predecessor
                    .as_ref()
                    .map_or(Ok(None), |previous| previous.carrier())
                    .map_err(|reason| Error::Execution(ExecutionError::AsynchronousFailure { reason }))?;
                let tokens = match predecessor {
                    Some(tokens) => tokens,
                    None => {
                        let client = self.client();
                        let devices = client.addressable_devices()?;
                        let process_index = client.process_index()?;
                        let mut tokens = Vec::with_capacity(addressable_device_count);
                        for &device_id in arguments.addressable_device_ids() {
                            let selected = devices
                                .iter()
                                .find(|candidate| {
                                    Device::from_pjrt(*candidate).is_ok_and(|candidate| candidate.id() == device_id)
                                })
                                .ok_or(ArrayError::NonAddressableDevice { device_id, process_index })?;
                            let manager = client.host_to_device_transfer_manager(
                                vec![BufferSpecification::new(BufferType::Token, [])],
                                selected.default_memory()?,
                            )?;
                            let token = manager.retrieve_buffer(0)?;
                            // Only initial zero-byte token creation is awaited. Predecessor carriers remain pending
                            // inputs.
                            manager.transfer_data(0, Arc::new([] as [u8; 0]), 0, true)?.r#await()?;
                            tokens.push(Arc::new(token));
                        }
                        tokens
                    }
                };
                arguments.push_input_buffers(tokens);
            }
        }
        debug_assert!(
            arguments
                .inputs_by_device()
                .iter()
                .all(|inputs| inputs.len() == program.signature.physical_input_count())
        );
        #[cfg(test)]
        if take_stateful_failure_injection(StatefulFailureInjection::BeforeSubmission) {
            return Err(Error::Pjrt(ryft_pjrt::Error::internal(
                "injected failure before execution submission".to_string(),
            )));
        }
        let resources = self.session.runtimes.prepare(&program.runtime_requirements, self.target()?, &program.mesh)?;
        let (context, retained_resources) = resources.into_parts();
        // These leases stay in this stack frame on every error path. PJRT destroys the context before returning an
        // error, and if submission succeeded it first waits for every device, so the leases outlive all native use.
        let (mut device_outputs, fence) = program
            .executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, context, None, None, None, &[])?
            .into_parts();
        // The fence releases the context before it reports readiness, and the session releases these leases only after
        // that, so pointees of the context's user data outlive native context destruction.
        self.session.runtimes.retain(&fence, retained_resources);
        // Reference publication must happen after every successful handoff, including a subsequent marshalling error.
        after_submission(&fence);
        let output_count = program.signature.physical_output_count();
        debug_assert_eq!(array_output_count, program.signature.array_output_count());
        let invalid_count = device_outputs
            .iter()
            .find(|outputs| outputs.outputs.len() != output_count)
            .map(|outputs| outputs.outputs.len());
        if let Some(reservation) = &mut reservation {
            if let Some(actual) = invalid_count {
                let reason: Arc<str> =
                    Arc::from(format!("expected {output_count} output(s) per device, but got {actual}"));
                reservation.submission.publish(Err(Arc::clone(&reason)));
                // Retain completion until the submitted work is done, then report the boundary failure.
                let completion = Arc::clone(&reservation.submission.completion);
                fence.on_ready(move |_| completion.complete(Err(reason)));
            } else {
                let carrier = if ordered {
                    Some(
                        device_outputs
                            .iter_mut()
                            .map(|outputs| {
                                Arc::new(outputs.outputs.remove(program.signature.ordered_io_output_index().unwrap()))
                            })
                            .collect(),
                    )
                } else {
                    None
                };
                reservation.submission.publish(Ok(carrier));
                let completion = Arc::clone(&reservation.submission.completion);
                fence.on_ready(move |result| completion.complete(result.map_err(|error| Arc::from(error.to_string()))));
            }
            reservation.published = true;
        }
        if let Some(actual) = invalid_count {
            return Err(Error::Pjrt(ryft_pjrt::Error::invalid_argument(format!(
                "expected {output_count} output(s) per device, but got {actual}",
            ))));
        }

        #[cfg(test)]
        if take_stateful_failure_injection(StatefulFailureInjection::AfterHandoff) {
            return Err(Error::Pjrt(ryft_pjrt::Error::internal(
                "injected failure after execution handoff".to_string(),
            )));
        }
        let mut per_output_buffers =
            (0..array_output_count).map(|_| Vec::with_capacity(addressable_device_count)).collect::<Vec<_>>();
        for device_output in device_outputs {
            for (output_index, buffer) in device_output.outputs.into_iter().enumerate() {
                per_output_buffers[output_index].push(buffer);
            }
        }
        Ok(Execution::new(per_output_buffers, fence))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AndOperation, ArrayIrBatch, ArrayIrType, ArrayType, Assert, AssertOperation, BatchAxis, BatchableOperation,
        BatchingContext, CompareOperation, ComparisonDirection, CompilationCacheDomain, Context,
        ConvertElementTypeOperation, DataType, Dimension as CoreDimension, DimensionBounds, DimensionDivOperation,
        DimensionFromScalarOperation, DimensionOperation, DimensionRemOperation, DimensionSubOperation,
        DimensionToScalarOperation, DimensionType, DimensionValue, DimensionVariable, DynamicBroadcastOperation,
        DynamicSliceOperation, EmptyRegionDriver, IotaOperation, LogicalMesh, MeshAxis, MeshAxisType,
        Operation as CoreOperation, Placeholder, ReduceOperation, ReductionKind, ReferenceAddUpdateOperation,
        ReferenceReadOperation, ReferenceType, Reshape, Shape, Sharding, ShardingDimension,
        StagedFunction as CoreStagedFunction, StagingContext, StaticShape, TiledLayout, TracingContext, Typed, batch,
    };
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::domains::compilation::jit_compilation_options;
    use crate::domains::options::Options;
    use crate::domains::sessions::Session;
    use crate::domains::tests::*;
    use crate::experimental::ops::{Constant, FlatProgram, Operation, ProgramBuilder};
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{Array, Value};

    use super::*;

    #[test]
    fn test_compiled_zero_space_identity_preserves_logical_type_and_canonical_value() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged =
            crate::stage::<_, ArrayType, ArrayType>(|input| input, input_type.clone(), Options::new(mesh.clone()))
                .unwrap()
                .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            staged.lower(&domain).unwrap().compile(&domain).unwrap();
        assert_eq!(compiled.compiled_program().output_types(), std::slice::from_ref(&input_type));

        let donated_staged: CoreStagedFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            crate::stage::<_, ArrayType, ArrayType>(
                |input| input,
                input_type.clone(),
                Options { donation_flags: Some(vec![true]), ..Options::new(mesh.clone()) },
            )
            .unwrap()
            .into_inner();
        let donated_lowered = donated_staged.lower(&domain).unwrap();
        assert_eq!(donated_lowered.lowered_program().donation_flags.as_ref(), &[false]);
        assert_eq!(
            domain.compilation_key(compiled.lowered().lowered_program()).unwrap(),
            domain.compilation_key(donated_lowered.lowered_program()).unwrap(),
        );

        // Boolean values retain physical `i1` arguments/results while the zero-space identity has an empty physical
        // signature. Their retained logical metadata also keeps their compilation identities distinct.
        let boolean_type = input_type.clone().with_data_type(DataType::Boolean);
        let boolean_staged: CoreStagedFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            crate::stage::<_, ArrayType, ArrayType>(|input| input, boolean_type, Options::new(mesh.clone()))
                .unwrap()
                .into_inner();
        let boolean_lowered = boolean_staged.lower(&domain).unwrap();
        assert_ne!(
            domain.compilation_key(compiled.lowered().lowered_program()).unwrap(),
            domain.compilation_key(boolean_lowered.lowered_program()).unwrap(),
        );

        let input = Array::from_host_buffer(&domain, input_type, mesh, []).unwrap();
        let output = compiled.executable().execute(&domain, Value::Array(input)).unwrap();
        let Value::Array(output) = output else {
            panic!("array-only compiled function returned a first-class dimension");
        };
        assert_eq!(output.data_type(), DataType::Zero);
        assert!(output.addressable_shards().next().unwrap().buffer().is_none());
    }

    #[test]
    fn test_compiled_mixed_zero_space_signature_projects_and_reconstructs_logical_values() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let value_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(3)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]))
            .with_sharding(sharding)
            .unwrap();
        let options = Options { donation_flags: Some(vec![false, true]), ..Options::new(mesh.clone()) };
        let staged = crate::stage::<_, (ArrayType, ArrayType), (ArrayType, ArrayType)>(
            |(value, zero)| (zero, value),
            (value_type.clone(), zero_type.clone()),
            options,
        )
        .unwrap()
        .into_inner();
        let lowered = staged.lower(&domain).unwrap();
        assert!(lowered.lowered_program().stable_hlo().contains("func.func @main(%arg0: tensor<3xf32>"));
        let compiled: ryft_core::compilation::CompiledFunction<
            Domain<'_>,
            (ArrayIrType, ArrayIrType),
            (ArrayIrType, ArrayIrType),
        > = lowered.compile(&domain).unwrap();
        let value = Array::from_host_buffer(
            &domain,
            value_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0]),
        )
        .unwrap();
        let zero = Array::from_host_buffer(&domain, zero_type.clone(), mesh, []).unwrap();

        let (zero_output, value_output) =
            compiled.executable().execute(&domain, (Value::Array(value), Value::Array(zero))).unwrap();
        let Value::Array(zero_output) = zero_output else {
            panic!("array-only compiled function returned a first-class dimension");
        };
        let Value::Array(value_output) = value_output else {
            panic!("array-only compiled function returned a first-class dimension");
        };

        assert_eq!(zero_output.r#type().as_ref(), &zero_type);
        assert!(zero_output.addressable_shards().next().unwrap().buffer().is_none());
        assert_eq!(value_output.r#type().as_ref(), &value_type);
        let value_bytes = value_output
            .addressable_shards()
            .next()
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(value_bytes.as_slice()), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_bounded_dynamic_input_padding_is_dense_row_major() {
        let source = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);

        let padded = pad_dense_array_bytes(source.as_slice(), &[2, 2], &[3, 4], DataType::F32).unwrap();

        assert_eq!(
            values_from_bytes::<f32>(padded.as_slice()),
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn test_bounded_dynamic_input_packing_cost_guard() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        // An input already at its bound takes the reuse tier: the physical payload preserves storage identity and the
        // only new value is its hidden logical-extent scalar.
        let exact_type = replicated_vector_type(&mesh, 8);
        let exact = Array::from_host_buffer(
            &domain,
            exact_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[0.0; 8]).as_slice(),
        )
        .unwrap();
        assert_eq!(
            BoundedInputPacking::from_shapes(&exact_type.static_shape().unwrap(), &StaticShape::new(vec![8])),
            BoundedInputPacking::Reuse,
        );
        let exact_storage = exact.clone();
        let at_bound = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&exact_type),
            vec![exact],
        )
        .unwrap();
        assert_eq!(at_bound.inputs.len(), 2);
        assert_eq!(at_bound.inputs[0], exact_storage);
        assert_eq!(
            at_bound.report,
            BoundedInputMaterializationReport {
                at_bound_reuses: 1,
                extent_scalar_uploads: 1,
                actual_bytes: 32,
                bound_bytes: 32,
                ..Default::default()
            },
        );

        // A smaller single-shard input pays the cold padding cost exactly once and retains the physical buffer. The
        // second call reuses identical PJRT buffer Arcs and reports zero pad-path allocation and transport. Explicit
        // path-local counts avoid process-global allocator noise from PJRT and Rust's parallel test runner.
        let smaller_type = replicated_vector_type(&mesh, 4);
        assert_eq!(
            BoundedInputPacking::from_shapes(&smaller_type.static_shape().unwrap(), &StaticShape::new(vec![8])),
            BoundedInputPacking::Pad,
        );
        let smaller = Array::from_host_buffer(
            &domain,
            smaller_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let cold = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&smaller_type),
            vec![smaller.clone()],
        )
        .unwrap();
        assert_eq!(
            cold.report,
            BoundedInputMaterializationReport {
                retained_misses: 1,
                device_to_host_shard_copies: 1,
                host_padding_payload_allocations: 1,
                host_to_device_shard_uploads: 1,
                extent_scalar_uploads: 1,
                actual_bytes: 16,
                bound_bytes: 32,
                ..Default::default()
            },
        );
        let retained_storage = cold.inputs[0].clone();
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let donation_arguments =
            Array::into_execute_arguments_with_donation(vec![cold.inputs[0].clone()], &[device_id], &[true]).unwrap();
        assert!(!donation_arguments.inputs_by_device()[0][0].donatable);
        let warm =
            materialize_bounded_dynamic_inputs(&domain, &signature, &[declared_type], &[smaller_type], vec![smaller])
                .unwrap();
        assert_eq!(warm.inputs[0], retained_storage);
        assert_eq!(
            warm.report,
            BoundedInputMaterializationReport {
                cache_hits: 1,
                extent_scalar_uploads: 0,
                actual_bytes: 16,
                bound_bytes: 32,
                ..Default::default()
            },
        );
    }

    #[test]
    fn test_independent_bounded_materialization_misses_share_one_issue_phase() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let actual_type = replicated_vector_type(&mesh, 4);
        let left = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let right = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[5.0, 6.0, 7.0, 8.0]).as_slice(),
        )
        .unwrap();
        let declared_types = [declared_type.clone(), declared_type];
        let actual_types = [actual_type.clone(), actual_type];
        let signature = ExecutableSignature::new(&declared_types, &[]);

        // The materializer's first scan reserves both cache entries and issues both D2H copies before its completion
        // loop awaits either one. The path-local report guards the two independent cold jobs and the warm zero-work
        // result without relying on scheduler timing.
        let cold = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            &declared_types,
            &actual_types,
            vec![left.clone(), right.clone()],
        )
        .unwrap();
        assert_eq!(
            cold.report,
            BoundedInputMaterializationReport {
                retained_misses: 2,
                device_to_host_shard_copies: 2,
                host_padding_payload_allocations: 2,
                host_to_device_shard_uploads: 2,
                extent_scalar_uploads: 2,
                actual_bytes: 32,
                bound_bytes: 64,
                ..Default::default()
            },
        );
        let retained = [cold.inputs[0].clone(), cold.inputs[1].clone()];

        let warm =
            materialize_bounded_dynamic_inputs(&domain, &signature, &declared_types, &actual_types, vec![left, right])
                .unwrap();
        assert_eq!(warm.inputs[0], retained[0]);
        assert_eq!(warm.inputs[1], retained[1]);
        assert_eq!(
            warm.report,
            BoundedInputMaterializationReport {
                cache_hits: 2,
                actual_bytes: 32,
                bound_bytes: 64,
                ..Default::default()
            },
        );
    }

    #[test]
    fn test_sharded_bounded_materialization_reuses_every_device_buffer() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 4);
        let domain = Session::new(&client).domain();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let actual_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let source = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        let cold = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&actual_type),
            vec![source.clone()],
        )
        .unwrap();
        assert_eq!(cold.report.device_to_host_shard_copies, 4);
        // Each shard of the leading dimension is a contiguous range of the dense source, so the shards are copied
        // directly into one destination and no merge buffer is allocated.
        assert_eq!(cold.report.host_merge_buffer_allocations, 0);
        assert_eq!(cold.report.host_padding_payload_allocations, 1);
        assert_eq!(cold.report.host_to_device_shard_uploads, 4);
        let retained = cold.inputs[0].clone();

        let warm =
            materialize_bounded_dynamic_inputs(&domain, &signature, &[declared_type], &[actual_type], vec![source])
                .unwrap();
        assert_eq!(warm.inputs[0], retained);
        assert_eq!(warm.report.cache_hits, 1);
        assert_eq!(warm.report.device_to_host_shard_copies, 0);
        assert_eq!(warm.report.host_to_device_shard_uploads, 0);
    }

    #[test]
    fn test_bounded_materialization_rejects_unreduced_inputs() {
        // Padding copies a bounded input through the host as one dense array, which would replace the separate
        // summands `[1, 2]` and `[3, 4]` that the two devices of this unreduced input hold, so it is rejected.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let devices = client.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let domain = Session::new(&client).domain();
        let sharding = Sharding::replicated(logical_mesh, 1).with_unreduced_axes(["x"]).unwrap();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let actual_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding).unwrap();
        let buffers = [[1f32, 2.], [3., 4.]]
            .iter()
            .zip(&devices)
            .map(|(values, device)| {
                client
                    .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let source = Array::from_addressable_buffers(&domain, actual_type.clone(), mesh, buffers).unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);
        let Err(error) =
            materialize_bounded_dynamic_inputs(&domain, &signature, &[declared_type], &[actual_type], vec![source])
        else {
            panic!("bounded materialization unexpectedly padded an unreduced input");
        };
        assert_eq!(
            error.to_string(),
            "cannot pad an array of type `f32[2][sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}}]` through \
             the host, because its devices hold separate summands along its unreduced axes, which a dense host copy \
             would not preserve",
        );
    }

    #[test]
    fn test_execution_plan_rejects_compilation_options_with_conflicting_layouts() {
        use ryft_core::CastOperation;

        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let input_type = ArrayType::new_static(DataType::F64, [2, 3]);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(
                CastOperation::<ArrayIrType>::unchecked(
                    input_type.clone().with_layout(layout.clone()).into(),
                    vec![input_type.into()],
                )
                .unwrap(),
                vec![],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let row_major = BufferSpecification::new(BufferType::F64, [2, 3]).proto().unwrap();
        // Explicit argument layouts bypass the MLIR layout attributes in PJRT. A conflicting result override must
        // fail before its buffers can be wrapped in types that claim another layout.
        lowered.compilation_options.argument_layouts = vec![row_major.clone()];
        lowered.compilation_options.executable_build_options.as_mut().unwrap().result_shape = Some(row_major);
        let Err(error) = domain.compile_program(&lowered) else {
            panic!("conflicting executable layout unexpectedly accepted");
        };
        assert!(matches!(&error, Error::Compilation(CompilationError::IncompatibleExecutable { .. })));
        assert_eq!(
            error.to_string(),
            "incompatible XLA executable: executable output 0 uses layout `tiled{1,0}` but its type declares \
             `tiled{0,1}`",
        );
        let column_major = BufferSpecification::with_layout(BufferType::F64, [2, 3], layout.to_pjrt().unwrap())
            .proto()
            .unwrap();
        lowered.compilation_options.executable_build_options.as_mut().unwrap().result_shape =
            Some(column_major.clone());
        domain.compile_program(&lowered).unwrap();
        lowered.compilation_options.argument_layouts = vec![column_major];
        let compiled = domain.compile_program(&lowered).unwrap();
        let values = vec![1f64, 2., 3., 4., 5., 6.];
        let input = Array::from_host_buffer(
            &domain,
            ArrayType::new_static(DataType::F64, [2, 3]),
            mesh,
            values_to_bytes(&values),
        )
        .unwrap();
        let outputs = domain.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(values_from_bytes::<f64>(&materialize_dense_array_bytes(&outputs[0]).unwrap()), values);
        let configured = domain.with_compilation_options(lowered.compilation_options.clone());
        let input = Array::from_host_buffer(
            &configured,
            ArrayType::new_static(DataType::F64, [2, 3])
                .with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]))),
            compiled.mesh().clone(),
            values_to_bytes(&values),
        )
        .unwrap();
        let outputs = domain.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(values_from_bytes::<f64>(&materialize_dense_array_bytes(&outputs[0]).unwrap()), values);
    }

    #[test]
    fn test_concurrent_bounded_materialization_is_single_flight() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let actual_type = replicated_vector_type(&mesh, 4);
        let source = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        let (left, right) = std::thread::scope(|scope| {
            let left_source = source.clone();
            let left_declared_type = declared_type.clone();
            let left_actual_type = actual_type.clone();
            let left = scope.spawn(|| {
                materialize_bounded_dynamic_inputs(
                    &domain,
                    &signature,
                    &[left_declared_type],
                    &[left_actual_type],
                    vec![left_source],
                )
                .unwrap()
            });
            let right = scope.spawn(|| {
                materialize_bounded_dynamic_inputs(
                    &domain,
                    &signature,
                    std::slice::from_ref(&declared_type),
                    std::slice::from_ref(&actual_type),
                    vec![source],
                )
                .unwrap()
            });
            (left.join().unwrap(), right.join().unwrap())
        });

        assert_eq!(left.inputs[0], right.inputs[0]);
        let retained_misses = left.report.retained_misses + right.report.retained_misses;
        let cache_hits = left.report.cache_hits + right.report.cache_hits;
        assert_eq!(retained_misses, 1);
        assert_eq!(cache_hits, 1);
        assert_eq!(left.report.device_to_host_shard_copies + right.report.device_to_host_shard_copies, 1);
        assert_eq!(left.report.host_to_device_shard_uploads + right.report.host_to_device_shard_uploads, 1);
    }

    #[test]
    fn test_materialize_zero_space_carriers_allocate_the_bounded_physical_shape() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        // The compiled module declares the bound-shaped `[3]` argument, so a below-bound `[2]` runtime value must
        // still produce a `[3]` carrier; the hidden extent scalar transports the logical size.
        let physical_zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let runtime_zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(sharding)
            .unwrap();
        let zero = Array::from_host_buffer(&domain, runtime_zero_type, mesh, []).unwrap();

        let carriers = materialize_zero_space_carriers(&domain, &[physical_zero_type], vec![zero]).unwrap();

        assert_eq!(carriers.len(), 1);
        assert_eq!(carriers[0].data_type(), DataType::Boolean);
        assert_eq!(carriers[0].shape().as_slice(), &[3]);
        let carrier_bytes = carriers[0]
            .addressable_shards()
            .next()
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(carrier_bytes.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn test_below_bound_zero_space_inputs_skip_the_padding_tiers_and_carry_the_bounded_shape() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let declared_type = ArrayType::new(
            DataType::Zero,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::non_negative(Some(4)).unwrap()).into()]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);
        let actual_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(sharding)
            .unwrap();
        let zero = Array::from_host_buffer(&domain, actual_type.clone(), mesh, []).unwrap();

        // The below-bound zero-space input must bypass the retained-cache and byte-pad tiers (its dense host copy is
        // empty) and rely on the carrier materialization to synthesize the bound-shaped predicate argument.
        let materialized = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&actual_type),
            vec![zero],
        )
        .unwrap();
        assert_eq!(materialized.inputs.len(), 2);
        assert_eq!(
            materialized.report,
            BoundedInputMaterializationReport { extent_scalar_uploads: 1, ..Default::default() },
        );

        let physical_types = signature.physical_input_types(std::slice::from_ref(&declared_type));
        let carriers = materialize_zero_space_carriers(&domain, &physical_types, materialized.inputs).unwrap();
        assert_eq!(carriers[0].data_type(), DataType::Boolean);
        assert_eq!(carriers[0].shape().as_slice(), &[3]);
        assert_eq!(carriers[1].data_type(), DataType::I32);
    }

    #[test]
    fn test_reference_transaction_donation_respects_effective_buffer_ownership() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();

        let unique = ArrayReference::new(f32_scalar(&domain, &mesh, 1.0));
        let unique_guard = unique.lock_storage().unwrap();
        let mut inputs = vec![unique_guard.observe().unwrap().snapshot().clone()];
        drop(Array::into_execute_arguments_with_donation(inputs.clone(), &[device_id], &[true]).unwrap());
        let (value, unique_taken) = unique_guard.wait_until_ready().unwrap().take().unwrap();
        inputs[0] = value;
        let arguments = Array::into_execute_arguments_with_donation(inputs, &[device_id], &[true]).unwrap();
        assert!(arguments.inputs_by_device()[0][0].donatable);
        unique_taken.poison("test transaction consumed unique input");

        let shared_value = f32_scalar(&domain, &mesh, 2.0);
        let retained = shared_value.clone();
        let shared = ArrayReference::new(shared_value);
        let shared_guard = shared.lock_storage().unwrap();
        let mut inputs = vec![shared_guard.observe().unwrap().snapshot().clone()];
        drop(Array::into_execute_arguments_with_donation(inputs.clone(), &[device_id], &[true]).unwrap());
        let (value, shared_taken) = shared_guard.wait_until_ready().unwrap().take().unwrap();
        inputs[0] = value;
        let arguments = Array::into_execute_arguments_with_donation(inputs, &[device_id], &[true]).unwrap();
        assert!(!arguments.inputs_by_device()[0][0].donatable);
        shared_taken.poison("test transaction consumed shared input");
        assert_eq!(read_f32s(&client, &retained), vec![2.0]);
    }

    #[test]
    fn test_stateful_compilation_domain_execute_uses_compiled_input_types() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let mut builder = ProgramBuilder::new();
        let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(scalar_type.clone())));
        let update = builder.add_input(ArrayIrType::Array(scalar_type));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program: FlatProgram = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_program(&program, 1, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let bytes = compiled.serialize().unwrap().unwrap();
        let restored = domain.deserialize_compiled_program(&bytes).unwrap().unwrap();

        for program in [&compiled, &restored] {
            let holder = ArrayReference::new(f32_scalar(&domain, &mesh, 1.0));
            let reference = Value::Reference(crate::Reference::new(holder.clone(), domain.clone()));
            let update = Value::Array(f32_scalar(&domain, &mesh, 2.0));

            // Kind checks use the stored binding positions, including captures, before acquiring mutable state.
            let expected = format!(
                "runtime input type `{}` does not match declared type `{}`",
                update.r#type(),
                reference.r#type(),
            );
            assert!(matches!(
                domain.execute_statefully(program, vec![update.clone(), update.clone()]).r#await(),
                Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidArgument { message })))
                    if message == expected,
            ));
            let expected = format!(
                "runtime input type `{}` does not match declared type `{}`",
                reference.r#type(),
                update.r#type(),
            );
            assert!(matches!(
                domain.execute_statefully(program, vec![reference.clone(), reference.clone()]).r#await(),
                Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidArgument { message })))
                    if message == expected,
            ));
            assert!(matches!(
                domain.execute_statefully(program, Vec::new()).r#await(),
                Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidInputCount { expected: 2, actual: 0 }))),
            ));
            assert_eq!(read_f32s(&client, &holder.read().unwrap()), vec![1.0]);

            let outputs = domain.execute_statefully(program, vec![reference, update]).r#await().unwrap();
            assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![3.0]);
            assert_eq!(read_f32s(&client, &holder.read().unwrap()), vec![3.0]);
        }
    }

    #[test]
    fn test_prepare_compiled_execution_kernel_participants() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "device", 1);
        let domain = Session::new(&client).domain();
        let input = f32_vector(&domain, &mesh, &[2.0, 5.0]);
        let mut builder = ProgramBuilder::new();
        let argument = builder.add_input(input.r#type().into_owned().into());
        let source = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![argument], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_program(&source, 0, &Options::new(mesh.clone())).unwrap();
        let mut program = domain.compile_program(&lowered).unwrap();
        // Attach the same validated marker carried by native-kernel executables. The identity program isolates
        // submission admission from any backend-specific kernel compiler or custom-call handler.
        program.kernel_execution_facts = Some(
            crate::kernels::KernelExecutionFacts::from_target(domain.target().unwrap(), &mesh)
                .unwrap()
                .configuration_key()
                .unwrap(),
        );
        let execution = domain.execute_compiled_async(&program, vec![input.clone()]).unwrap();
        execution.fence().block_until_ready().unwrap();
        assert_eq!(read_f32s(&client, &execution.output()[0]), vec![2.0, 5.0]);
        program.mesh = DeviceMesh::new(
            mesh.logical_mesh().clone(),
            vec![Device::new(mesh.devices()[0].id(), client.process_index().unwrap() + 1)],
        )
        .unwrap();
        // A topology change cannot reach resharding, donation, or PJRT execution; the retained input stays readable.
        assert!(matches!(
            domain.prepare_compiled_execution(&program, vec![input.clone()], &[]),
            Err(error)
                if matches!(error.downcast_custom::<crate::kernels::KernelEmbeddingError>(),
                    Some(crate::kernels::KernelEmbeddingError::Invalid { message })
                        if message == "cross-process kernel execution requires a qualified collective ordering and \
                                       completion contract"),
        ));
        assert_eq!(read_f32s(&client, &input), vec![2.0, 5.0]);
    }

    #[test]
    fn test_prepare_compiled_execution_preserves_erased_input_producer_completion() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let zero_type = ArrayType::new_static(DataType::Zero, [2]);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        // Exercise both the ordinary input projection and the branch that materializes bounded inputs. The failed
        // producer belongs to an erased zero-space input, so no physical input buffer can carry its failure.
        for value_type in
            [ArrayType::new_static(DataType::F32, [2]), ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]))]
        {
            let mut builder = ProgramBuilder::new();
            let zero_input = builder.add_input(zero_type.clone().into());
            let value_input = builder.add_input(value_type.into());
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![zero_input, value_input],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap();
            let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            assert_eq!(lowered.signature.input_mapping(), &[None, Some(0)]);
            let compiled = domain.compile_program(&lowered).unwrap();
            let (event, promise) = client.event(()).unwrap();
            promise.set(Some(ryft_pjrt::Error::invalid_argument("erased producer failed"))).unwrap();
            let zero = Array::from_host_buffer(&domain, zero_type.clone(), mesh.clone(), [])
                .unwrap()
                .with_execution_fence(ExecutionFence::new(vec![event]));
            let value = Array::from_host_buffer(
                &domain,
                ArrayType::new_static(DataType::F32, [2]),
                mesh.clone(),
                values_to_bytes::<f32>(&[1.0, 2.0]),
            )
            .unwrap();
            assert!(matches!(
                domain.execute_compiled_async(&compiled, vec![zero, value]),
                Err(Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. }))
                    if message == "erased producer failed",
            ));
        }
    }

    #[test]
    fn test_failed_bounded_upload_batch_is_not_published_and_retries() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let actual_type = replicated_vector_type(&mesh, 4);
        let left = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let right = Array::from_host_buffer(
            &domain,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[5.0, 6.0, 7.0, 8.0]).as_slice(),
        )
        .unwrap();
        let declared_types = [declared_type.clone(), declared_type];
        let actual_types = [actual_type.clone(), actual_type];
        let signature = ExecutableSignature::new(&declared_types, &[]);
        let mut observed_publications = 0usize;
        let mut observed_uploads = 0usize;

        // The injected failure runs only after both physical arrays and their H2D uploads exist. Returning the error
        // before publication drops both producer reservations, so neither failed result can become a warm cache hit.
        let error = match materialize_bounded_dynamic_inputs_with_readiness(
            &domain,
            &signature,
            &declared_types,
            &actual_types,
            vec![left.clone(), right.clone()],
            |publications| {
                observed_publications = publications.len();
                observed_uploads = publications
                    .iter()
                    .flat_map(|publication| publication.physical.shards())
                    .filter(|shard| shard.buffer().is_some())
                    .count();
                Err(ProgramError::InvalidArgument { message: "injected bounded upload readiness failure".to_string() }
                    .into())
            },
        ) {
            Ok(_) => panic!("injected upload readiness failure must reject materialization"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("injected bounded upload readiness failure"));
        assert_eq!(observed_publications, 2);
        assert_eq!(observed_uploads, 2);

        let retry = materialize_bounded_dynamic_inputs(
            &domain,
            &signature,
            &declared_types,
            &actual_types,
            vec![left.clone(), right.clone()],
        )
        .unwrap();
        assert_eq!(retry.report.retained_misses, 2);
        assert_eq!(retry.report.cache_hits, 0);
        assert_eq!(retry.report.device_to_host_shard_copies, 2);
        assert_eq!(retry.report.host_to_device_shard_uploads, 2);
        let retained = [retry.inputs[0].clone(), retry.inputs[1].clone()];

        let warm =
            materialize_bounded_dynamic_inputs(&domain, &signature, &declared_types, &actual_types, vec![left, right])
                .unwrap();
        assert_eq!(warm.inputs[0], retained[0]);
        assert_eq!(warm.inputs[1], retained[1]);
        assert_eq!(warm.report.cache_hits, 2);
        assert_eq!(warm.report.device_to_host_shard_copies, 0);
        assert_eq!(warm.report.host_to_device_shard_uploads, 0);
    }

    #[test]
    fn test_static_program_outputs_take_their_planned_layouts_on_multiple_cpu_devices() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 4);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]))
            .with_sharding(sharding)
            .unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |input| input.clone() + input,
            input_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            staged.lower(&domain).unwrap().compile(&domain).unwrap();
        assert!(compiled.compiled_program().plan.output_layouts.is_some());

        // Outputs are built from the program's planned layouts, which place every shard like the validating
        // constructors do and are shared by the outputs of every execution.
        let values = (0..8).map(|value| value as f32).collect::<Vec<_>>();
        let input = Array::from_host_buffer(
            &domain,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes(values.as_slice()).as_slice(),
        )
        .unwrap();
        let call = || {
            let output = compiled.executable().execute(&domain, Value::Array(input.clone())).unwrap();
            let Value::Array(output) = output else {
                panic!("array-only compiled function returned a first-class dimension");
            };
            output
        };
        let output = call();
        let other_output = call();
        output.block_until_ready().unwrap();
        assert_eq!(output.r#type().as_ref(), &input_type);
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(
            output.shards().iter().map(|shard| shard.descriptor()).collect::<Vec<_>>(),
            input.shards().iter().map(|shard| shard.descriptor()).collect::<Vec<_>>(),
        );
        assert_eq!(
            other_output.shards().iter().map(|shard| shard.descriptor()).collect::<Vec<_>>(),
            output.shards().iter().map(|shard| shard.descriptor()).collect::<Vec<_>>(),
        );
        let shards = output
            .shards()
            .iter()
            .map(|shard| {
                values_from_bytes::<f32>(
                    shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap().as_slice(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(shards, vec![vec![0.0, 2.0], vec![4.0, 6.0], vec![8.0, 10.0], vec![12.0, 14.0]]);
    }

    #[test]
    fn test_static_program_outputs_without_declared_placement_are_validated_on_multiple_cpu_devices() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 4);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let unconstrained_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::unconstrained()]).unwrap();
        let unconstrained_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]))
            .with_sharding(unconstrained_sharding)
            .unwrap();
        let unsharded_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]));

        // XLA chooses the placement of outputs with open (`Unconstrained`) dimensions and of outputs without declared
        // shardings, so their programs plan no trusted output layouts and validate the returned buffers instead.
        for input_type in [unconstrained_type, unsharded_type] {
            let staged = crate::stage::<_, ArrayType, ArrayType>(
                |input| input.clone() + input,
                input_type,
                Options::new(mesh.clone()),
            )
            .unwrap()
            .into_inner();
            let compiled: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
                staged.lower(&domain).unwrap().compile(&domain).unwrap();
            assert!(compiled.compiled_program().plan.output_layouts.is_none());
        }
    }

    #[test]
    fn test_internal_dynamic_tensor_behind_a_static_boundary_takes_the_replica_path() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 4);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
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

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let executable_options = lowered.compilation_options.executable_build_options.as_ref().unwrap();
        assert_eq!(executable_options.replica_count, 4);
        assert_eq!(executable_options.partition_count, 1);
        assert!(!executable_options.use_spmd_partitioning);
        assert!(!executable_options.use_shardy_partitioner);
        assert!(!lowered.stable_hlo().contains("sdy."), "{}", lowered.stable_hlo());

        let compiled = domain.compile_program(&lowered).unwrap();
        let size =
            Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), 3i64.to_ne_bytes().as_slice()).unwrap();
        let value = Array::from_host_buffer(
            &domain,
            value_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.execute_program(&compiled, vec![size, value]).unwrap();
        assert_eq!(read_f32s(&client, &outputs[0]), vec![6.0]);

        // The replica path is a hard requirement rather than a preference: annotating the very same program for
        // Shardy and compiling it with the Shardy partitioner enabled fails on the internal dynamic tensor, even
        // though every boundary type is static.
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let annotated = crate::experimental::lowering::lower_mlir_module_for_program(
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
        let mut shardy_lowered = lowered.clone();
        shardy_lowered.stable_hlo = annotated.into_parts().0.into();
        shardy_lowered.compilation_options =
            jit_compilation_options(domain.compilation_options.as_ref(), 4, true, None);
        let Err(error) = domain.compile_program(&shardy_lowered) else {
            panic!("Shardy unexpectedly accepted a module containing an internal dynamic tensor");
        };
        assert!(error.to_string().contains("only supports ranked tensors with a static shape"), "{error}");
    }

    #[test]
    fn test_bounded_dynamic_program_executes_on_multiple_cpu_devices() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 4);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 4);
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(4)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(2, Some(5)).unwrap());
        let depth = DimensionVariable::new("depth", DimensionBounds::new(4, Some(7)).unwrap());
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                CoreDimension::Static(4),
                CoreDimension::Dynamic(rows),
                CoreDimension::Dynamic(columns),
                CoreDimension::Dynamic(depth),
            ]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |input| input.clone() + input,
            declared_type,
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            staged.lower(&domain).unwrap().compile(&domain).unwrap();
        let executable_options =
            compiled.compiled_program().compilation_options.executable_build_options.as_ref().unwrap();
        assert_eq!(executable_options.replica_count, 4);
        assert_eq!(executable_options.partition_count, 1);
        assert!(!executable_options.use_spmd_partitioning);
        assert!(!executable_options.use_shardy_partitioner);
        assert!(compiled.compiled_program().plan.output_layouts.is_none());

        for shape in [[4usize, 2, 3, 5], [4, 3, 4, 6]] {
            let actual_type =
                ArrayType::new(DataType::F32, Shape::new(shape.into_iter().map(CoreDimension::Static).collect()))
                    .with_sharding(sharding.clone())
                    .unwrap();
            let values = (0..shape.iter().product()).map(|value| value as f32).collect::<Vec<_>>();
            let input = Array::from_host_buffer(
                &domain,
                actual_type,
                mesh.clone(),
                values_to_bytes(values.as_slice()).as_slice(),
            )
            .unwrap();
            let output = compiled.executable().execute(&domain, Value::Array(input)).unwrap();
            let Value::Array(output) = output else {
                panic!("array-only compiled function returned a first-class dimension");
            };
            output.block_until_ready().unwrap();

            assert_eq!(output.shape().as_slice(), shape.as_slice());
            assert_eq!(read_f32s(&client, &output), values.iter().map(|value| value * 2.0).collect::<Vec<_>>());
        }
        assert_eq!(domain.program_count(), 1);

        // Dynamic tensor partitioning remains an explicit backend limitation rather than silently changing a user's
        // requested placement to replication.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let sharded_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4), CoreDimension::Dynamic(extent)]))
                .with_sharding(
                    Sharding::new(
                        mesh.logical_mesh().clone(),
                        vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                    )
                    .unwrap(),
                )
                .unwrap();
        let staged =
            crate::stage::<_, ArrayType, ArrayType>(|input| input.clone() + input, sharded_type, Options::new(mesh))
                .unwrap()
                .into_inner();
        let Err(error) = staged.lower(&domain) else {
            panic!("bounded-dynamic sharded program unexpectedly lowered");
        };
        assert_eq!(
            error.to_string(),
            "invalid compilation options: bounded-dynamic multi-device programs currently require fully replicated \
             shardings because Shardy rejects dynamic tensors",
        );
    }

    #[test]
    fn test_data_derived_prefix_slice_executes_at_multiple_extents() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let mask_type = ArrayType::new(DataType::Boolean, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(input_sharding.clone())
            .unwrap();
        let values_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(input_sharding)
            .unwrap();
        let count_variable = DimensionVariable::new("count", DimensionBounds::new(0, Some(5)).unwrap());

        // This is the compiled counterpart of the Phase 4 prefix-take fixture: the mask determines a device-side
        // count, the checked gateway turns that count into a bounded dimension, and the slice returns that many
        // leading values without a host readback.
        let mut builder = ProgramBuilder::new();
        let mask = builder.add_input(mask_type.into());
        let values = builder.add_input(values_type.into());
        let mask = builder
            .add_instruction(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::I64, false),
                Vec::new(),
                vec![mask],
                None,
            )
            .unwrap()[0];
        let count = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![mask], None)
            .unwrap()[0];
        let count = builder
            .add_instruction(DimensionFromScalarOperation::new(count_variable.into()), Vec::new(), vec![count], None)
            .unwrap()[0];
        let start = builder.add_constant(Constant::Dimension(DimensionValue::constant(0).unwrap()));
        let output = builder
            .add_instruction(
                DynamicSliceOperation::<ArrayIrType>::from_rank(1),
                Vec::new(),
                vec![values, start, count],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let values = || f32_vector(&domain, &mesh, &[10.0, 20.0, 30.0, 40.0]);
        let execute = |mask: &[bool]| {
            domain.execute_program(&compiled, vec![boolean_vector(&domain, &mesh, mask), values()]).unwrap()
        };

        let two = execute(&[true, false, true, false]);
        assert_eq!(two[0].shape(), StaticShape::new(vec![2]));
        assert_eq!(read_f32s(&client, &two[0]), vec![10.0, 20.0]);
        let four = execute(&[true, true, true, true]);
        assert_eq!(four[0].shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, &four[0]), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_data_derived_prefix_slice_executes_with_duplicate_and_equal_outputs() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let mask_type = ArrayType::new(DataType::Boolean, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(input_sharding.clone())
            .unwrap();
        let values_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(input_sharding)
            .unwrap();
        let count_variable = DimensionVariable::new("count", DimensionBounds::new(0, Some(5)).unwrap());

        // XLA merges bounded dynamic results that it proves equal and, on CPU, fills the duplicate result with a plain
        // copy that leaves its runtime-size trailer at zero (CUDA preserves the trailer). The program returns the same
        // prefix slice twice and then an equal slice staged separately, which covers duplicate entry results and
        // results merged by XLA's CSE.
        let mut builder = ProgramBuilder::new();
        let mask = builder.add_input(mask_type.into());
        let values = builder.add_input(values_type.into());
        let mask = builder
            .add_instruction(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::I64, false),
                Vec::new(),
                vec![mask],
                None,
            )
            .unwrap()[0];
        let count = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![mask], None)
            .unwrap()[0];
        let count = builder
            .add_instruction(DimensionFromScalarOperation::new(count_variable.into()), Vec::new(), vec![count], None)
            .unwrap()[0];
        let start = builder.add_constant(Constant::Dimension(DimensionValue::constant(0).unwrap()));
        let mut prefix = || {
            builder
                .add_instruction(
                    DynamicSliceOperation::<ArrayIrType>::from_rank(1),
                    Vec::new(),
                    vec![values, start, count],
                    None,
                )
                .unwrap()[0]
        };
        let output = prefix();
        let equal_output = prefix();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![output, output, equal_output],
                vec![Placeholder, Placeholder],
                vec![Placeholder; 3],
            )
            .unwrap();

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let values = || f32_vector(&domain, &mesh, &[10.0, 20.0, 30.0, 40.0]);
        let execute = |mask: &[bool]| {
            domain.execute_program(&compiled, vec![boolean_vector(&domain, &mesh, mask), values()]).unwrap()
        };

        let two = execute(&[true, false, true, false]);
        assert_eq!(two.len(), 3);
        for output in &two {
            assert_eq!(output.shape(), StaticShape::new(vec![2]));
            assert_eq!(read_f32s(&client, output), vec![10.0, 20.0]);
        }
        let four = execute(&[true, true, true, true]);
        assert_eq!(four.len(), 3);
        for output in &four {
            assert_eq!(output.shape(), StaticShape::new(vec![4]));
            assert_eq!(read_f32s(&client, output), vec![10.0, 20.0, 30.0, 40.0]);
        }
    }

    #[test]
    fn test_bounded_dynamic_bitcast_element_type_preserves_prefix_extents() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![CoreDimension::Dynamic(DimensionVariable::new(
                "count",
                DimensionBounds::new(0, Some(5)).unwrap(),
            ))]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let mut builder = ProgramBuilder::new();
        let values = builder.add_input(declared_type.into());
        let halves = builder
            .add_instruction(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::U16, true),
                Vec::new(),
                vec![values],
                None,
            )
            .unwrap()[0];
        let restored = builder
            .add_instruction(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, true),
                Vec::new(),
                vec![halves],
                None,
            )
            .unwrap()[0];
        let bits = builder
            .add_instruction(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::I1, true),
                Vec::new(),
                vec![values],
                None,
            )
            .unwrap()[0];
        let restored_bits = builder
            .add_instruction(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, true),
                Vec::new(),
                vec![bits],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![halves, restored, bits, restored_bits],
                vec![Placeholder],
                vec![Placeholder; 4],
            )
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        // Splitting and joining the static bit-width axis preserves an independent dynamic prefix and its live bytes.
        for count in [2usize, 4] {
            let outputs =
                domain.execute_program(&compiled, vec![f32_vector(&domain, &mesh, &vec![1.0; count])]).unwrap();
            assert_eq!(outputs[0].shape(), StaticShape::new(vec![count, 2]));
            assert_eq!(outputs[1].shape(), StaticShape::new(vec![count]));
            assert_eq!(read_f32s(&client, &outputs[1]), vec![1.0; count]);
            assert_eq!(outputs[2].shape(), StaticShape::new(vec![count, 32]));
            assert_eq!(outputs[3].shape(), StaticShape::new(vec![count]));
            assert_eq!(read_f32s(&client, &outputs[3]), vec![1.0; count]);
            let bytes = outputs[0]
                .addressable_shards()
                .next()
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            assert_eq!(bytes, values_to_bytes(&vec![1.0f32; count]));
        }
    }

    #[test]
    fn test_data_derived_dynamic_slice_checks_runtime_input_and_result_extents() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let value_type = replicated_scalar_type(&mesh, DataType::F32);
        let input_extent = DimensionVariable::new("input_extent", DimensionBounds::new(1, Some(5)).unwrap());
        let result_extent = DimensionVariable::new("result_extent", DimensionBounds::new(0, Some(5)).unwrap());

        // The result extent is independent of the input extent at the type level. Lowering uses their finite physical
        // bounds for one specialization and preserves the eager `result_extent <= input_extent` contract with an
        // ordered runtime assertion.
        let mut builder = ProgramBuilder::new();
        let input_size = builder.add_input(size_type.clone().into());
        let result_size = builder.add_input(size_type.clone().into());
        let value = builder.add_input(value_type.clone().into());
        let input_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(input_extent.into()), Vec::new(), vec![input_size], None)
            .unwrap()[0];
        let result_dimension = builder
            .add_instruction(
                DimensionFromScalarOperation::new(result_extent.into()),
                Vec::new(),
                vec![result_size],
                None,
            )
            .unwrap()[0];
        let input = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, input_dimension], None)
            .unwrap()[0];
        let start = builder.add_constant(Constant::Dimension(DimensionValue::constant(0).unwrap()));
        let output = builder
            .add_instruction(
                DynamicSliceOperation::<ArrayIrType>::from_rank(1),
                Vec::new(),
                vec![input, start, result_dimension],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert!(lowered.stable_hlo().contains("@ryft.assert"));
        let compiled = domain.compile_program(&lowered).unwrap();
        let scalar =
            |r#type: ArrayType, bytes: &[u8]| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap();
        let inputs = |input_size: i64, result_size: i64| {
            vec![
                scalar(size_type.clone(), input_size.to_ne_bytes().as_slice()),
                scalar(size_type.clone(), result_size.to_ne_bytes().as_slice()),
                scalar(value_type.clone(), 7.0f32.to_ne_bytes().as_slice()),
            ]
        };

        let eager = program
            .interpret_in_context(&domain, inputs(4, 2).into_iter().map(Value::Array).collect::<Vec<_>>())
            .unwrap();
        let compiled_valid = domain.execute_program(&compiled, inputs(4, 2)).unwrap();
        assert_eq!(read_f32s(&client, program_array(&eager[0])), vec![7.0, 7.0]);
        assert_eq!(read_f32s(&client, &compiled_valid[0]), read_f32s(&client, program_array(&eager[0])));

        let error = match domain.execute_program(&compiled, inputs(2, 4)) {
            Ok(invalid) => invalid[0].block_until_ready().unwrap_err().to_string(),
            Err(error) => error.to_string(),
        };
        assert_eq!(error, "`dynamic_slice` limit 4 exceeds input axis 0 extent 2");
    }

    #[test]
    fn test_compiled_dimension_from_scalar_reports_observed_bounds_failure_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());

        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![dimension], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("@ryft.assert").count(), 1, "{}", lowered.stable_hlo());
        let compiled = domain.compile_program(&lowered).unwrap();
        assert!(compiled.requires_assertion_handler);
        // Some PJRT plugins do not serialize executables containing host callbacks. When serialization is available,
        // exercise the cache-hit path and prove its runtime-feature metadata survives restoration.
        let compiled = match domain.serialize_program(&compiled).unwrap() {
            Some(bytes) => domain.deserialize_program(bytes.as_slice()).unwrap().unwrap(),
            None => compiled,
        };
        assert!(compiled.requires_assertion_handler);

        let valid =
            Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), 4i64.to_ne_bytes().as_slice()).unwrap();
        let valid = domain.execute_program(&compiled, vec![valid]).unwrap();
        assert_eq!(read_i64s(&client, &valid[0]), vec![4]);

        let invalid = Array::from_host_buffer(&domain, input_type, mesh, 9i64.to_ne_bytes().as_slice()).unwrap();
        let error = domain.execute_program(&compiled, vec![invalid]).unwrap_err();
        assert!(
            error.to_string().contains(
                "`dimension_from_scalar` failed: input dimension `extent` = 9 is outside its declared bounds [1, 9)"
            ),
            "{error}",
        );
    }

    #[test]
    fn test_compiled_assertions_report_the_first_same_class_failure_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let bounds = DimensionBounds::new(0, Some(10)).unwrap();
        let first_type = DimensionType::new("first", bounds);
        let second_type = DimensionType::new("second", bounds);
        let third_type = DimensionType::new("third", bounds);

        let mut builder = ProgramBuilder::new();
        let first_input = builder.add_input(input_type.clone().into());
        let second_input = builder.add_input(input_type.clone().into());
        let third_input = builder.add_input(input_type.clone().into());
        let first = builder
            .add_instruction(DimensionFromScalarOperation::new(first_type.clone()), Vec::new(), vec![first_input], None)
            .unwrap()[0];
        let second = builder
            .add_instruction(
                DimensionFromScalarOperation::new(second_type.clone()),
                Vec::new(),
                vec![second_input],
                None,
            )
            .unwrap()[0];
        let third = builder
            .add_instruction(DimensionFromScalarOperation::new(third_type.clone()), Vec::new(), vec![third_input], None)
            .unwrap()[0];
        let predicate = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThanOrEqual),
                Vec::new(),
                vec![first, second],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new("`first` <= `second`")
                    .with_labels(vec!["first".to_owned(), "second".to_owned()]),
                Vec::new(),
                vec![predicate, first, second],
                None,
            )
            .unwrap();
        let predicate = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThanOrEqual),
                Vec::new(),
                vec![second, third],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new("`second` <= `third`")
                    .with_labels(vec!["second".to_owned(), "third".to_owned()]),
                Vec::new(),
                vec![predicate, second, third],
                None,
            )
            .unwrap();
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![third], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("@ryft.assert").count(), 5, "{}", lowered.stable_hlo());
        let compiled = domain.compile_program(&lowered).unwrap();
        let input = |value: i64| {
            Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
        };

        let error = domain.execute_program(&compiled, vec![input(4), input(3), input(2)]).unwrap_err();
        assert!(error.to_string().contains("assertion failed: `first` <= `second`; first=4, second=3"), "{error}",);
        assert!(!error.to_string().contains("`second` <= `third`"), "{error}");
    }

    #[test]
    fn test_compiled_assertions_preserve_diagnostics_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let bounds = DimensionBounds::new(0, Some(20)).unwrap();
        let left_type = DimensionType::new("left", bounds);
        let right_type = DimensionType::new("right", bounds);
        for (left_value, right_value, divisible) in [(3i64, 4i64, false), (7, 3, true), (7, 0, true)] {
            let mut builder = ProgramBuilder::new();
            let left_input = builder.add_input(input_type.clone().into());
            let right_input = builder.add_input(input_type.clone().into());
            let left = builder
                .add_instruction(
                    DimensionFromScalarOperation::new(left_type.clone()),
                    Vec::new(),
                    vec![left_input],
                    None,
                )
                .unwrap()[0];
            let right = builder
                .add_instruction(
                    DimensionFromScalarOperation::new(right_type.clone()),
                    Vec::new(),
                    vec![right_input],
                    None,
                )
                .unwrap()[0];
            let mut output = right;
            let (predicate, message) = if divisible {
                let zero = builder.add_constant(Constant::Dimension(DimensionValue::constant(0).unwrap()));
                let one = builder.add_constant(Constant::Dimension(DimensionValue::constant(1).unwrap()));
                let positive = builder
                    .add_instruction(
                        CompareOperation::<ArrayIrType>::new(ComparisonDirection::GreaterThan),
                        Vec::new(),
                        vec![right, zero],
                        None,
                    )
                    .unwrap()[0];
                builder
                    .add_instruction(
                        AssertOperation::<ArrayIrType>::new("divisor must be positive")
                            .with_labels(vec!["right".to_owned()]),
                        Vec::new(),
                        vec![positive, right],
                        None,
                    )
                    .unwrap();
                let one_type = DimensionValue::constant(1).unwrap().r#type().into_owned();
                let maximum = ryft_core::DimensionMaxOperation::new(&right_type, &one_type).unwrap();
                let safe_type = maximum.infer_output_types(&[right_type.clone(), one_type], &[]).unwrap().remove(0);
                let safe = builder
                    .add_instruction(ryft_core::DimensionOperation::Max(maximum), Vec::new(), vec![right, one], None)
                    .unwrap()[0];
                let remainder = builder
                    .add_instruction(
                        ryft_core::DimensionOperation::Rem(DimensionRemOperation::new(&left_type, &safe_type).unwrap()),
                        Vec::new(),
                        vec![left, safe],
                        None,
                    )
                    .unwrap()[0];
                output = builder
                    .add_instruction(
                        ryft_core::DimensionOperation::Div(DimensionDivOperation::new(&left_type, &safe_type).unwrap()),
                        Vec::new(),
                        vec![left, safe],
                        None,
                    )
                    .unwrap()[0];
                (
                    builder
                        .add_instruction(
                            CompareOperation::<ArrayIrType>::new(ComparisonDirection::Equal),
                            Vec::new(),
                            vec![remainder, zero],
                            None,
                        )
                        .unwrap()[0],
                    "`left` must be divisible by `right`",
                )
            } else {
                (
                    builder
                        .add_instruction(
                            CompareOperation::<ArrayIrType>::new(ComparisonDirection::Equal),
                            Vec::new(),
                            vec![left, right],
                            None,
                        )
                        .unwrap()[0],
                    "dimensions must agree",
                )
            };
            builder
                .add_instruction(
                    AssertOperation::<ArrayIrType>::new(message)
                        .with_labels(vec!["left".to_owned(), "right".to_owned()]),
                    Vec::new(),
                    vec![predicate, left, right],
                    None,
                )
                .unwrap();
            let output =
                builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![output], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap();
            let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            let compiled = domain.compile_program(&lowered).unwrap();
            let input = |value: i64| {
                Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice())
                    .unwrap()
            };
            let error = domain.execute_program(&compiled, vec![input(left_value), input(right_value)]).unwrap_err();
            let expected = if right_value == 0 {
                ryft_core::AssertionError::Failed {
                    message: "divisor must be positive".to_owned(),
                    observations: vec![("right".to_owned(), "0".to_owned())],
                }
            } else {
                ryft_core::AssertionError::Failed {
                    message: message.to_owned(),
                    observations: vec![
                        ("left".to_owned(), left_value.to_string()),
                        ("right".to_owned(), right_value.to_string()),
                    ],
                }
            };
            assert!(error.to_string().contains(&expected.to_string()), "{error}");
        }
    }

    #[test]
    fn test_compiled_per_element_validation_on_cpu() {
        use ryft_core::Compare;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let values = f32_vector(&domain, &mesh, &[1., -2., 3., -4.]);
        let limits = f32_vector(&domain, &mesh, &[0., 0., 0., 0.]);
        let staged = crate::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| vec![inputs[0].greater_than(&inputs[1]).unwrap()],
            vec![values.r#type().into_owned(), limits.r#type().into_owned()],
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let lowered = domain.lower_program(staged.source_program().program(), 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        assert!(!compiled.requires_assertion_handler);
        let outputs = domain.execute_program(&compiled, vec![values, limits]).unwrap();
        assert_eq!(
            outputs[0]
                .addressable_shards()
                .next()
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap(),
            vec![1, 0, 1, 0]
        );
    }

    #[test]
    fn test_compiled_assertion_bounded_elements_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        for mode in [0, 1, 2] {
            let predicates = boolean_vector(&domain, &mesh, &[true, false, false, false]);
            let observations = f32_vector(&domain, &mesh, &[10., 20., 30., 40.]);
            let staged = crate::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
                |inputs| {
                    if mode == 1 {
                        batch(
                            |(predicate, observation)| {
                                predicate.assert_with_limit(
                                    "bounded check",
                                    &[("value", observation)],
                                    std::num::NonZeroUsize::new(2).unwrap(),
                                )
                            },
                            (inputs[0].clone(), inputs[1].clone()),
                            (BatchAxis::new(0), BatchAxis::new(0)),
                            (),
                            None,
                        )
                        .unwrap();
                    } else {
                        let predicate = if mode == 2 { inputs[0].reshape([2, 2]).unwrap() } else { inputs[0].clone() };
                        let observation =
                            if mode == 2 { inputs[1].reshape([2, 2]).unwrap() } else { inputs[1].clone() };
                        predicate
                            .assert_with_limit(
                                "bounded check",
                                &[("value", observation)],
                                std::num::NonZeroUsize::new(2).unwrap(),
                            )
                            .unwrap();
                    }
                    vec![]
                },
                vec![predicates.r#type().into_owned(), observations.r#type().into_owned()],
                Options::new(mesh.clone()),
            )
            .unwrap()
            .into_inner();
            let lowered =
                domain.lower_program(staged.source_program().program(), 0, &Options::new(mesh.clone())).unwrap();
            let compiled = domain.compile_program(&lowered).unwrap();
            let error = domain.execute_program(&compiled, vec![predicates, observations.clone()]).unwrap_err();
            let expected = ryft_core::AssertionError::FailedElements {
                message: "bounded check".to_owned(),
                failures: vec![
                    ryft_core::AssertionFailure {
                        index: if mode == 2 { vec![0, 1] } else { vec![1] },
                        observations: vec![("value".to_owned(), "20".to_owned())],
                    },
                    ryft_core::AssertionFailure {
                        index: if mode == 2 { vec![1, 0] } else { vec![2] },
                        observations: vec![("value".to_owned(), "30".to_owned())],
                    },
                ],
                omitted: 1,
            };
            assert!(error.to_string().contains(&expected.to_string()), "{error}");
            let predicates = boolean_vector(&domain, &mesh, &[true; 4]);
            assert!(domain.execute_program(&compiled, vec![predicates, observations]).unwrap().is_empty());
        }
    }

    #[test]
    fn test_compiled_assertion_bounded_sub_byte_observations_on_cpu() {
        use ryft_core::ConvertElementType;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        for (data_type, values) in [
            (DataType::I1, [0., -1., 0., -1.]),
            (DataType::U1, [0., 1., 0., 1.]),
            (DataType::I2, [0., -2., -1., 1.]),
            (DataType::U2, [0., 1., 2., 3.]),
            (DataType::I4, [0., -8., -1., 7.]),
            (DataType::U4, [0., 1., 8., 15.]),
        ] {
            let predicates = boolean_vector(&domain, &mesh, &[false; 4]);
            let observations = f32_vector(&domain, &mesh, &values);
            let staged = crate::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
                |inputs| {
                    let observation = inputs[1].convert_element_type(data_type).unwrap();
                    inputs[0]
                        .assert_with_limit(
                            "narrow values",
                            &[("value", observation)],
                            std::num::NonZeroUsize::new(3).unwrap(),
                        )
                        .unwrap();
                    vec![]
                },
                vec![predicates.r#type().into_owned(), observations.r#type().into_owned()],
                Options::new(mesh.clone()),
            )
            .unwrap()
            .into_inner();
            let lowered =
                domain.lower_program(staged.source_program().program(), 0, &Options::new(mesh.clone())).unwrap();
            let compiled = domain.compile_program(&lowered).unwrap();
            let error = domain.execute_program(&compiled, vec![predicates, observations]).unwrap_err();
            let expected = ryft_core::AssertionError::FailedElements {
                message: "narrow values".to_owned(),
                failures: values
                    .iter()
                    .take(3)
                    .enumerate()
                    .map(|(index, value)| ryft_core::AssertionFailure {
                        index: vec![index],
                        observations: vec![("value".to_owned(), value.to_string())],
                    })
                    .collect(),
                omitted: 1,
            };
            assert!(error.to_string().contains(&expected.to_string()), "{data_type}: {error}");
        }
    }

    #[test]
    fn test_compiled_assertion_scalar_observations_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let predicate_type = replicated_scalar_type(&mesh, DataType::Boolean);
        for observations in [
            vec![],
            vec![
                ("unsigned", DataType::U64, u64::MAX.to_ne_bytes().to_vec(), u64::MAX.to_string()),
                ("signed", DataType::I64, i64::MIN.to_ne_bytes().to_vec(), i64::MIN.to_string()),
                ("float", DataType::F32, (-0.5f32).to_ne_bytes().to_vec(), "-0.5".to_owned()),
                ("boolean", DataType::Boolean, vec![1], "true".to_owned()),
            ],
        ] {
            let mut builder = ProgramBuilder::new();
            let predicate = builder.add_input(predicate_type.clone().into());
            let predicate =
                builder.add_instruction(AndOperation::new(), vec![], vec![predicate, predicate], None).unwrap()[0];
            let mut inputs = vec![predicate];
            for (_, data_type, _, _) in &observations {
                inputs.push(builder.add_input(replicated_scalar_type(&mesh, *data_type).into()));
            }
            builder
                .add_instruction(
                    AssertOperation::<ArrayIrType>::new("scalar check")
                        .with_labels(observations.iter().map(|(label, _, _, _)| (*label).to_owned()).collect()),
                    Vec::new(),
                    inputs,
                    None,
                )
                .unwrap();
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![], vec![Placeholder; observations.len() + 1], vec![])
                .unwrap();
            let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            let compiled = domain.compile_program(&lowered).unwrap();
            for passed in [true, false] {
                let mut inputs = vec![
                    Array::from_host_buffer(&domain, predicate_type.clone(), mesh.clone(), &[u8::from(passed)])
                        .unwrap(),
                ];
                for (_, data_type, bytes, _) in &observations {
                    inputs.push(
                        Array::from_host_buffer(
                            &domain,
                            replicated_scalar_type(&mesh, *data_type),
                            mesh.clone(),
                            bytes,
                        )
                        .unwrap(),
                    );
                }
                let output = domain.execute_program(&compiled, inputs);
                if passed {
                    assert!(output.unwrap().is_empty());
                } else {
                    let expected = ryft_core::AssertionError::Failed {
                        message: "scalar check".to_owned(),
                        observations: observations
                            .iter()
                            .map(|(label, _, _, rendered)| ((*label).to_owned(), rendered.clone()))
                            .collect(),
                    };
                    let error = output.unwrap_err();
                    assert!(error.to_string().contains(&expected.to_string()), "{error}");
                }
            }
        }

        let predicates = boolean_vector(&domain, &mesh, &[true, false, false]);
        let observations = f32_vector(&domain, &mesh, &[10., 20., 30.]);
        let staged = crate::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                batch(
                    |(predicate, observation)| predicate.assert("mapped check", &[("value", observation)]),
                    (inputs[0].clone(), inputs[1].clone()),
                    (BatchAxis::new(0), BatchAxis::new(0)),
                    (),
                    None,
                )
                .unwrap();
                vec![]
            },
            vec![predicates.r#type().into_owned(), observations.r#type().into_owned()],
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let program = staged.source_program().program();
        let lowered = domain.lower_program(program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let error = domain.execute_program(&compiled, vec![predicates, observations]).unwrap_err();
        let expected = ryft_core::AssertionError::Failed {
            message: "mapped check".to_owned(),
            observations: vec![("value".to_owned(), "20".to_owned()), ("batch_index".to_owned(), "1".to_owned())],
        };
        assert!(error.to_string().contains(&expected.to_string()), "{error}");
    }

    #[test]
    fn test_compiled_assertion_dynamic_batching_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        for bounded in [false, true] {
            let extent = DimensionVariable::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
            let context = TracingContext::<Constant, Operation>::new();
            let size = context.input(input_type.clone().into());
            let limit = context.input(input_type.clone().into());
            let dimension = context
                .bind(DimensionFromScalarOperation::new(extent.clone().into()), Vec::new(), &[size])
                .unwrap()
                .remove(0);
            let indices = context
                .bind(
                    IotaOperation::new(ArrayType::new(DataType::I64, Shape::new(vec![extent.into()])), 0).unwrap(),
                    Vec::new(),
                    std::slice::from_ref(&dimension),
                )
                .unwrap()
                .remove(0);
            let limits = context
                .bind(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), &[limit, dimension.clone()])
                .unwrap()
                .remove(0);
            let predicates = context
                .bind(
                    Operation::Array(ryft_core::ArrayOperation::Compare(CompareOperation::new(
                        ComparisonDirection::LessThan,
                    ))),
                    Vec::new(),
                    &[indices.clone(), limits],
                )
                .unwrap()
                .remove(0);
            let batching = BatchingContext::new(context.clone(), dimension);
            let operation = AssertOperation::new("dynamic mapped check").with_labels(vec!["value".to_owned()]);
            let operation =
                if bounded { operation.with_failure_limit(std::num::NonZeroUsize::new(2).unwrap()) } else { operation };
            operation
                .batch(
                    &batching,
                    &EmptyRegionDriver,
                    &[ArrayIrBatch::new(predicates, 0).unwrap(), ArrayIrBatch::new(indices, 0).unwrap()],
                )
                .unwrap();
            let program = context
                .builder()
                .borrow()
                .clone()
                .build::<Vec<Constant>, Vec<Constant>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            let compiled = domain.compile_program(&lowered).unwrap();
            for (size, limit) in [(0i64, 0i64), (1, 1), (3, 3)] {
                let inputs = [size, limit]
                    .into_iter()
                    .map(|value| {
                        Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), &value.to_ne_bytes())
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                assert!(domain.execute_program(&compiled, inputs).unwrap().is_empty());
            }
            let inputs = [3i64, 1i64]
                .into_iter()
                .map(|value| {
                    Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), &value.to_ne_bytes()).unwrap()
                })
                .collect::<Vec<_>>();
            let error = domain.execute_program(&compiled, inputs).unwrap_err();
            let expected = if bounded {
                ryft_core::AssertionError::FailedElements {
                    message: "dynamic mapped check".to_owned(),
                    failures: vec![
                        ryft_core::AssertionFailure {
                            index: vec![1],
                            observations: vec![("value".to_owned(), "1".to_owned())],
                        },
                        ryft_core::AssertionFailure {
                            index: vec![2],
                            observations: vec![("value".to_owned(), "2".to_owned())],
                        },
                    ],
                    omitted: 0,
                }
            } else {
                ryft_core::AssertionError::Failed {
                    message: "dynamic mapped check".to_owned(),
                    observations: vec![
                        ("value".to_owned(), "1".to_owned()),
                        ("batch_index".to_owned(), "1".to_owned()),
                    ],
                }
            };
            assert!(error.to_string().contains(&expected.to_string()), "{error}");
        }
    }

    #[test]
    fn test_compiled_assertion_dynamic_replicated_batching_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        for bounded in [false, true] {
            let context = TracingContext::<Constant, Operation>::new();
            let size = context.input(input_type.clone().into());
            let dimension = context
                .bind(
                    DimensionFromScalarOperation::new(
                        DimensionVariable::new("batch", DimensionBounds::new(0, Some(5)).unwrap()).into(),
                    ),
                    Vec::new(),
                    std::slice::from_ref(&size),
                )
                .unwrap()
                .remove(0);
            let condition = context.lift(Constant::Boolean(false)).unwrap();
            let batching = BatchingContext::new(context.clone(), dimension);
            let operation = AssertOperation::new("dynamic replicated check").with_labels(vec!["size".to_owned()]);
            let operation =
                if bounded { operation.with_failure_limit(std::num::NonZeroUsize::new(2).unwrap()) } else { operation };
            operation
                .batch(
                    &batching,
                    &EmptyRegionDriver,
                    &[ArrayIrBatch::replicated(condition), ArrayIrBatch::replicated(size)],
                )
                .unwrap();
            let program = context
                .builder()
                .borrow()
                .clone()
                .build::<Vec<Constant>, Vec<Constant>>(vec![], vec![Placeholder], vec![])
                .unwrap();
            let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            let compiled = domain.compile_program(&lowered).unwrap();
            let empty =
                Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), &0i64.to_ne_bytes()).unwrap();
            assert!(domain.execute_program(&compiled, vec![empty]).unwrap().is_empty());
            let nonempty =
                Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), &3i64.to_ne_bytes()).unwrap();
            let error = domain.execute_program(&compiled, vec![nonempty]).unwrap_err();
            let expected = if bounded {
                ryft_core::AssertionError::FailedElements {
                    message: "dynamic replicated check".to_owned(),
                    failures: vec![
                        ryft_core::AssertionFailure {
                            index: vec![0],
                            observations: vec![("size".to_owned(), "3".to_owned())],
                        },
                        ryft_core::AssertionFailure {
                            index: vec![1],
                            observations: vec![("size".to_owned(), "3".to_owned())],
                        },
                    ],
                    omitted: 1,
                }
            } else {
                ryft_core::AssertionError::Failed {
                    message: "dynamic replicated check".to_owned(),
                    observations: vec![("size".to_owned(), "3".to_owned())],
                }
            };
            assert!(error.to_string().contains(&expected.to_string()), "{error}");
        }
    }

    #[test]
    fn test_compiled_dimension_arithmetic_preserves_checked_host_diagnostics_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let bounds = DimensionBounds::new(0, Some(10)).unwrap();
        let left_type = DimensionType::new("left", bounds);
        let right_type = DimensionType::new("right", bounds);
        let check = |operation: DimensionOperation<DimensionValue>,
                     valid_operands: (i64, i64),
                     expected_output: i64,
                     invalid_operands: (i64, i64),
                     expected_error: &str| {
            let mut builder = ProgramBuilder::new();
            let left_input = builder.add_input(input_type.clone().into());
            let right_input = builder.add_input(input_type.clone().into());
            let left = builder
                .add_instruction(
                    DimensionFromScalarOperation::new(left_type.clone()),
                    Vec::new(),
                    vec![left_input],
                    None,
                )
                .unwrap()[0];
            let right = builder
                .add_instruction(
                    DimensionFromScalarOperation::new(right_type.clone()),
                    Vec::new(),
                    vec![right_input],
                    None,
                )
                .unwrap()[0];
            let result = builder.add_instruction(operation, Vec::new(), vec![left, right], None).unwrap()[0];
            let output =
                builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![result], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap();
            let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            assert_eq!(lowered.stable_hlo().matches("@ryft.assert").count(), 3, "{}", lowered.stable_hlo());
            let compiled = domain.compile_program(&lowered).unwrap();
            let input = |value: i64| {
                Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice())
                    .unwrap()
            };

            let output =
                domain.execute_program(&compiled, vec![input(valid_operands.0), input(valid_operands.1)]).unwrap();
            assert_eq!(read_i64s(&client, &output[0]), vec![expected_output]);
            let error = domain
                .execute_program(&compiled, vec![input(invalid_operands.0), input(invalid_operands.1)])
                .unwrap_err();
            assert!(error.to_string().contains(expected_error), "{error}");
        };

        check(
            DimensionOperation::Sub(DimensionSubOperation::new(&left_type, &right_type).unwrap()),
            (5, 2),
            3,
            (2, 5),
            "`dimension_sub` requires left >= right; got left=2, right=5",
        );
        check(
            DimensionOperation::Div(DimensionDivOperation::new(&left_type, &right_type).unwrap()),
            (7, 3),
            2,
            (7, 0),
            "`dimension_div` requires right > 0; got left=7, right=0",
        );
        check(
            DimensionOperation::Rem(DimensionRemOperation::new(&left_type, &right_type).unwrap()),
            (7, 3),
            1,
            (7, 0),
            "`dimension_rem` requires right > 0; got left=7, right=0",
        );
    }
}
