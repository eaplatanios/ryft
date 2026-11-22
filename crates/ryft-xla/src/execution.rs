//! Execution of compiled XLA programs through PJRT.
//!
//! Every [`CompiledProgram`] carries an `ExecutionPlan` that is derived once when the program is compiled or loaded.
//! The plan records the physical input types, the default donation requests, the addressable devices, and, when every
//! output has a static type and a fixed placement, the output plans, so that each execution only reads it.
//!
//! Programs without external reference state execute through `Session::execute_compiled_async`, which validates the
//! inputs, materializes their physical form (i.e., pads bounded dynamic inputs, allocates carriers for inputs of the
//! zero data type, reshards them, and converts their layouts), submits them to PJRT, and reconstructs the logical
//! outputs from the returned buffers. Programs with external reference state execute through
//! [`StatefulCompilationDomain::execute_statefully`], which locks the bound references in ascending identity order,
//! submits the execution with snapshots of their states, and commits the hidden final state outputs as reference
//! replacements. [`ExecutionArguments`] groups physical input buffers by device, and [`ExecutionError`] describes the
//! failures of this lifecycle, including those of ordered effects.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ryft_core::macros::check_count;
use ryft_core::{
    ArrayReference, ArrayType, Cast, DataType, Device, DeviceId, DeviceMesh, Dimension, Layout, Memory, Pad, Placement,
    ProgramError, ReadyOrPendingReferenceGuard, ReferenceCompletion as CoreReferenceCompletion,
    ReferenceCompletionBackend, ReferenceError, ReferenceId, ReferenceReplacementPreparation, ReferenceType, Shape,
    Sharding, ShardingDimension, ShardingError, StatefulCompilationDomain, StatefulExecution, Type as CoreType,
    TypeRefinements, Typed, TypedProgram, UnreducedKind, ValueProjection, validate_reference_boundary,
};
use ryft_pjrt::{
    Buffer, BufferSpecification, BufferType, Error as PjrtError, Execution, ExecutionDeviceInputs, ExecutionFence,
    ExecutionInput, LoadedExecutable,
};

use crate::arrays::metadata::ArrayOutputPlan;
use crate::devices::MemoryExtension;
use crate::lowering::signature::ExecutableSignature;
use crate::sessions::effects::ensure_effect_dispatch_is_allowed;
use crate::types::DataTypeExtension;
use crate::{
    Array, ArrayError, ArrayShard, CompilationError, CompiledProgram, Error, ExecutionResources, FromPjrt,
    RuntimeRequirement, Session, ToPjrt, Type, Value,
};

// TODO(eaplatanios): Review this module.

/// Represents errors produced while invoking a [`CompiledProgram`], waiting for submitted work to finish, or
/// recovering an effect scope.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ExecutionError {
    /// The executable belongs to a different live PJRT client.
    #[error("XLA executable belongs to a different PJRT client")]
    ClientMismatch,

    /// Runtime donation metadata has a different arity than the invocation inputs.
    #[error("got {actual_count} donation flag(s), but expected {expected_count}")]
    DonationFlagCountMismatch { expected_count: usize, actual_count: usize },

    /// A public argument is committed to a physical device assignment incompatible with the executable.
    #[error("public argument #{index} is committed to devices {actual:?}, but the executable requires {expected:?}")]
    CommittedInputDeviceMismatch { index: usize, actual: Vec<DeviceId>, expected: Vec<DeviceId> },

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

    /// The selected platform has no qualified native token transport for ordered I/O.
    #[error("native ordered I/O transport is not yet verified on platform `{platform}`")]
    UnsupportedEffectPlatform { platform: String },

    /// Ordered I/O cannot execute over the selected distributed or non-addressable assignment.
    #[error("ordered I/O requires a fixed fully addressable single-process device assignment")]
    UnsupportedEffectDeviceAssignment,

    /// Multi-device ordered I/O was declared without the required device ordering contract.
    #[error("multi-device ordered I/O requires every ordered occurrence to declare `DeviceOrderedIo`")]
    MissingDeviceOrderedEffect,

    /// Previously failed ordered work must be observed and acknowledged before further ordered submission.
    #[error("ordered I/O has failed; call `effects_barrier()` and then `acknowledge_effect_errors()` before retrying")]
    ScopeNeedsRecovery,

    /// An existing ordered effect chain cannot change its device assignment.
    #[error("ordered I/O device changed from {expected:?} to {actual:?}; use `with_new_effect_scope()`")]
    ScopeDeviceMismatch { expected: Vec<DeviceId>, actual: Vec<DeviceId> },

    /// Execution or an effect barrier was entered from a managed callback.
    #[error("execution and effect barriers cannot be entered from a managed callback")]
    ReentrantDispatch,

    /// Submitted work failed asynchronously, including invocations that have no external reference state.
    #[error("asynchronous XLA execution failed: {reason}")]
    AsynchronousFailure { reason: Arc<str> },

    /// Session synchronization observed failed executions or transfers in registration order.
    #[error(
        "session work failed: {}",
        .failures
            .iter()
            .map(|(identity, error)| format!("submission {identity}: {error}"))
            .collect::<Vec<_>>()
            .join("; "),
    )]
    SynchronizationFailures { failures: Vec<(u64, Error)> },

    /// An effect barrier observed failed submissions, retaining their reservation order and diagnostics.
    #[error(
        "{}; call `effects_barrier()` again if more work was submitted, then `acknowledge_effect_errors()` to recover \
         once affected work is complete",
        .failures
            .iter()
            .map(|(identity, error)| format!("submission {identity}: {error}"))
            .collect::<Vec<_>>()
            .join("; "),
    )]
    EffectSubmissionFailures { failures: Vec<(u64, Arc<str>)> },

    /// Pending or unobserved failed work prevents acknowledgment of the current effect scope.
    #[error(
        "effect failures are pending or unreported; call `effects_barrier()` and then `acknowledge_effect_errors()` \
         again"
    )]
    PrematureAcknowledgment,
}

/// Execution metadata of a [`CompiledProgram`] that does not depend on the values that it is called with. It is
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

    /// Whether any physical array input has the zero data type (i.e., [`DataType::Zero`]) and thus needs a
    /// materialized carrier.
    zero_space_inputs: bool,

    /// Placement of every logical output, present when every output has a static type, every output without a
    /// physical result has the zero data type, the program runs on a single device or declares a sharding without
    /// `Unconstrained` dimensions for every output, which XLA must then honor, and the executable reports the memory
    /// of every output, which [`Self::new`] compares with the declared one. The outputs of such programs are built
    /// directly from the returned buffers.
    output_plans: Option<Vec<ArrayOutputPlan>>,
}

impl ExecutionPlan {
    /// Computes the execution plan of a program with the provided metadata, loaded as `executable`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        executable: &LoadedExecutable<'_>,
        input_types: &[ArrayType],
        output_types: &[ArrayType],
        output_shardings: &[Sharding],
        signature: &ExecutableSignature,
        donation_flags: &[bool],
        capture_count: usize,
        mesh: &DeviceMesh,
    ) -> Result<Self, Error> {
        let physical_input_logical_indices = signature.project_inputs(&(0..input_types.len()).collect::<Vec<_>>());
        let mut physical_input_types = signature.physical_input_types(input_types);
        let compiled_executable = executable.executable()?;
        // A plugin without the layouts extension (e.g., the MPS plugin) does not report the layouts of its executables.
        // The validation below then rejects an input type that declares a layout, which it cannot confirm, and an
        // execution passes the inputs without a declared layout as they are.
        let input_layouts = if physical_input_types.is_empty() {
            Some(Vec::new())
        } else {
            match compiled_executable.input_layouts() {
                Ok(layouts) => Some(layouts),
                Err(PjrtError::Unimplemented { .. } | PjrtError::MissingFunction { .. }) => None,
                Err(error) => return Err(error.into()),
            }
        };
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
            ("input", physical_input_types.as_slice(), input_layouts.as_deref().unwrap_or_default()),
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
        // The same compile options can override the MLIR memory attributes, so the memories of the compiled ABI are
        // validated like its layouts. A plugin that does not report them leaves the inputs to the execution itself and
        // the outputs to the validation of their returned buffers, which the output plans below skip. Device
        // placement means the default memory of the plugin, whose kind is platform-dependent.
        let devices = executable.addressable_devices()?;
        let default_memory_kind = match devices.first() {
            Some(device) => device.default_memory()?.kind()?.into_owned(),
            None => Memory::Device.placement_kind().to_string(),
        };
        let input_memory_kinds = if physical_input_types.is_empty() {
            Some(Vec::new())
        } else {
            match compiled_executable.input_memory_kinds() {
                Ok(memory_kinds) => Some(memory_kinds),
                Err(PjrtError::Unimplemented { .. } | PjrtError::MissingFunction { .. }) => None,
                Err(error) => return Err(error.into()),
            }
        };
        let output_memory_kinds = if physical_output_types.is_empty() {
            Some(Vec::new())
        } else {
            match compiled_executable.output_memory_kinds() {
                Ok(memory_kinds) => Some(memory_kinds),
                Err(PjrtError::Unimplemented { .. } | PjrtError::MissingFunction { .. }) => None,
                Err(error) => return Err(error.into()),
            }
        };
        for (kind, types, memory_kinds) in [
            ("input", physical_input_types.as_slice(), &input_memory_kinds),
            ("output", physical_output_types.as_slice(), &output_memory_kinds),
        ] {
            let Some(memory_kinds) = memory_kinds else {
                continue;
            };
            for (index, array_type) in types.iter().enumerate() {
                let expected = match array_type.memory() {
                    Memory::Device => default_memory_kind.as_str(),
                    memory => memory.placement_kind(),
                };
                let actual = memory_kinds.get(index).ok_or_else(|| {
                    Error::Compilation(CompilationError::IncompatibleExecutable {
                        reason: format!("executable {kind} {index} has no memory kind"),
                    })
                })?;
                if actual.as_ref() != expected {
                    return Err(Error::Compilation(CompilationError::IncompatibleExecutable {
                        reason: format!(
                            "executable {kind} {index} uses memory kind `{actual}` but its type declares `{expected}`",
                        ),
                    }));
                }
            }
        }
        // These types describe the physical executable signature, including layouts chosen for unconstrained
        // logical inputs. Keep the logical declarations unchanged so they still admit other representations.
        if let Some(input_layouts) = &input_layouts {
            for (index, array_type) in physical_input_types.iter_mut().enumerate() {
                let layout = input_layouts.get(index).ok_or_else(|| {
                    Error::Compilation(CompilationError::IncompatibleExecutable {
                        reason: format!("executable input {index} has no physical layout"),
                    })
                })?;
                let layout = Layout::from_pjrt(ryft_pjrt::Layout::from_proto(layout.serialize()?.proto()?)?)?;
                *array_type = array_type.clone().with_layout(layout);
            }
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
        let addressable_device_ids =
            devices.iter().map(|device| device.id().map_err(Error::from)).collect::<Result<Arc<[_]>, _>>()?;
        let static_inputs = input_types.iter().all(|input_type| input_type.static_shape().is_some());
        let zero_space_inputs = physical_input_types.iter().any(|input_type| input_type.data_type().is_zero());
        let static_outputs = signature.output_dimensions().is_empty()
            && output_types.iter().zip(signature.output_mapping()).all(|(output_type, physical_index)| {
                output_type.static_shape().is_some() && (physical_index.is_some() || output_type.data_type().is_zero())
            });
        // Declared shardings only fix output placement when XLA must honor them. Lowering emits no result shardings
        // when any output lacks one, so XLA chooses the placement of such outputs. Outputs whose placements record
        // `Unconstrained` dimensions, which their result annotations replicate, are not planned either. The buffers of
        // these outputs are validated by the general constructor instead. Every placement on a single device holds the
        // whole array, so it is always fixed.
        let declared_placement = mesh.devices().len() == 1
            || output_types.iter().all(|output_type| output_type.sharding().is_some())
                && output_shardings.iter().all(|sharding| {
                    !sharding.dimensions().iter().any(|dimension| matches!(dimension, ShardingDimension::Unconstrained))
                });
        let output_plans = if static_outputs && declared_placement && output_memory_kinds.is_some() {
            Some(
                output_types
                    .iter()
                    .zip(output_shardings)
                    .map(|(output_type, sharding)| {
                        ArrayOutputPlan::new(output_type.clone(), Some(sharding.clone()), mesh, &addressable_device_ids)
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
            output_plans,
        })
    }
}

/// Physical input buffers of one PJRT execution, grouped by addressable device in the order in which
/// [`LoadedExecutable::execute`] consumes them.
pub struct ExecutionArguments {
    /// Addressable devices in the same order as [`Self::inputs_by_device`].
    addressable_device_ids: Arc<[DeviceId]>,

    /// Execution inputs grouped by addressable device.
    inputs_by_device: Vec<Vec<ExecutionInput<'static>>>,
}

impl ExecutionArguments {
    /// Creates execution arguments that pass `arrays` to [`ryft_pjrt::LoadedExecutable::execute`] without donating any
    /// of them. The inputs of each device are in `addressable_device_ids` order, and
    /// [`Self::as_execution_device_inputs`] converts the result to the form that PJRT consumes.
    pub fn from_arrays(arrays: Vec<Array>, addressable_device_ids: &[DeviceId]) -> Result<Self, Error> {
        let donation_flags = vec![false; arrays.len()];
        Self::from_arrays_with_donation(arrays, addressable_device_ids.into(), donation_flags.as_slice())
    }

    /// Creates execution arguments from `arrays`, honoring `donation_flags` only for arrays whose shard buffers are
    /// uniquely owned. A requested donation is silently downgraded to a non-donating input when any shard buffer is
    /// shared (e.g., by another clone of the array). Callers that need to
    /// observe the effective decision can inspect [`ExecutionInput::donatable`] on the result.
    pub fn from_arrays_with_donation(
        arrays: Vec<Array>,
        addressable_device_ids: Arc<[DeviceId]>,
        donation_flags: &[bool],
    ) -> Result<Self, Error> {
        if donation_flags.len() != arrays.len() {
            return Err(Error::Execution(ExecutionError::DonationFlagCountMismatch {
                expected_count: arrays.len(),
                actual_count: donation_flags.len(),
            }));
        }
        for (position, &device_id) in addressable_device_ids.iter().enumerate() {
            if addressable_device_ids[..position].contains(&device_id) {
                return Err(ShardingError::DuplicateDeviceId { id: device_id }.into());
            }
        }

        // Every array must have exactly one buffer on each addressable device and none elsewhere. Its shards are
        // looked up by device in constant time, and counting its buffers detects buffers on other devices.
        let mut inputs_by_device = (0..addressable_device_ids.len())
            .map(|_| Vec::with_capacity(arrays.len()))
            .collect::<Vec<Vec<ExecutionInput<'static>>>>();
        for (array_index, (array, requested)) in arrays.iter().zip(donation_flags).enumerate() {
            let donatable = *requested && array.has_unique_shard_buffers();
            for (device_inputs, &device_id) in inputs_by_device.iter_mut().zip(addressable_device_ids.iter()) {
                let buffer = array
                    .device_shard(device_id)
                    .and_then(ArrayShard::buffer)
                    .ok_or(Error::Execution(ExecutionError::MissingArrayShardForDevice { array_index, device_id }))?;
                device_inputs.push(ExecutionInput { buffer: Arc::clone(buffer), donatable });
            }
            if array.shards().iter().filter(|shard| shard.buffer().is_some()).count() != addressable_device_ids.len() {
                let device_id = array
                    .shards()
                    .iter()
                    .filter(|shard| shard.buffer().is_some())
                    .map(|shard| shard.device().id())
                    .find(|device_id| !addressable_device_ids.contains(device_id))
                    .unwrap();
                return Err(Error::Execution(ExecutionError::UnexpectedArrayShardDevice { array_index, device_id }));
            }
        }

        // Dropping the consumed arrays leaves the execution inputs as the only owners of donated buffers.
        drop(arrays);
        Ok(Self { addressable_device_ids, inputs_by_device })
    }

    /// Returns addressable device IDs corresponding to [`Self::inputs_by_device`].
    pub fn addressable_device_ids(&self) -> &[DeviceId] {
        &self.addressable_device_ids
    }

    /// Returns execution inputs grouped by device.
    pub fn inputs_by_device(&self) -> &[Vec<ExecutionInput<'static>>] {
        self.inputs_by_device.as_slice()
    }

    /// Creates PJRT [`ExecutionDeviceInputs`] in the same device order as [`Self::addressable_device_ids`].
    pub fn as_execution_device_inputs<'l>(&'l self) -> Vec<ExecutionDeviceInputs<'static, 'l>> {
        self.inputs_by_device.iter().map(|inputs| ExecutionDeviceInputs::from(inputs.as_slice())).collect()
    }

    /// Appends one non-donatable hidden input per addressable device. The execution boundary validates the device
    /// assignment before calling this function, and hidden tokens never pass through numerical array construction.
    fn push_input_buffers(&mut self, buffers: Vec<Arc<Buffer<'static>>>) {
        assert_eq!(buffers.len(), self.inputs_by_device.len());
        for (inputs, buffer) in self.inputs_by_device.iter_mut().zip(buffers) {
            inputs.push(ExecutionInput::from(buffer));
        }
    }
}

impl std::fmt::Debug for ExecutionArguments {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input_counts = self.inputs_by_device.iter().map(Vec::len).collect::<Vec<_>>();
        formatter
            .debug_struct("ExecutionArguments")
            .field("addressable_device_ids", &self.addressable_device_ids)
            .field("input_counts_per_device", &input_counts)
            .finish()
    }
}

impl Session {
    /// Initializes required extensions on the invoking session before compilation or native loading.
    pub(super) fn ensure_runtime_requirements(&self, requirements: &[RuntimeRequirement]) -> Result<(), Error> {
        self.runtimes().ensure(requirements)
    }

    /// Validates that this session's PJRT client owns `program`.
    pub(super) fn validate_program_owner(&self, program: &CompiledProgram) -> Result<(), Error> {
        if program.executable().is_owned_by(self.client()) {
            Ok(())
        } else {
            Err(Error::Execution(ExecutionError::ClientMismatch))
        }
    }

    /// Enqueues a stateless `program` and returns its possibly still pending flat outputs together with the fence of
    /// the whole execution. Programs with external reference state must flow through the stateful call surface
    /// instead, which owns the reference transaction and the hidden outputs that carry final reference states.
    ///
    /// The backend may execute inline or return pending arrays and a completion fence. This function adds no wait
    /// solely because public outputs are empty. Bounded dynamic boundaries can introduce additional waits. Inputs
    /// below their bounds perform device-to-host copies and uploads padded to those bounds before launch, and each
    /// dynamic output reads its hidden extent scalar before constructing the logical output type. Erased inputs of the
    /// zero data type await producer completion before launch because they have no physical buffer to carry that
    /// dependency.
    pub(crate) fn execute_compiled_async(
        &self,
        program: &CompiledProgram,
        inputs: Vec<Array>,
    ) -> Result<Execution<Vec<Array>>, Error> {
        // Public call surfaces screen stateful programs out before reaching this entry point, because only the
        // stateful call path owns holder leases and pending generations. A stateful program arriving here must
        // therefore already carry state values as plain arrays (e.g., as ABI tests do), and its hidden outputs that
        // carry final reference states are returned verbatim.
        let mut resources = Some(self.prepare_execution_resources(program)?);
        let committed = inputs.iter().skip(program.capture_count()).any(Array::is_committed);
        let prepared = self.prepare_compiled_execution(program, inputs, &[])?;
        let (arguments, input_refinements, physical_output_count) = prepared.into_arguments()?;
        let execution = self.execute_pjrt_buffers(program, arguments, physical_output_count, &mut resources, |_| {})?;
        let (physical_outputs, fence) = execution.into_parts();
        let mut physical_outputs = physical_outputs.into_iter().map(Some).collect::<Vec<_>>();
        let outputs = self.reconstruct_compiled_outputs(
            program,
            &mut physical_outputs,
            fence.clone(),
            0..program.all_output_types().len(),
            committed,
        )?;
        // Static outputs cannot disagree with the input refinements, so only builds with debug assertions re-validate
        // them.
        if program.plan().output_plans.is_none() || cfg!(debug_assertions) {
            validate_compiled_output_refinements(program, &input_refinements, program.all_output_types(), &outputs)?;
        }
        if physical_outputs.iter().any(Option::is_some) {
            return Err(
                ProgramError::MalformedProgram("executable returned an unclaimed physical output".to_string()).into()
            );
        }
        Ok(Execution::new(outputs, fence))
    }

    /// Executes a compiled program without external reference state on flat `inputs`, which implements
    /// [`CompilationDomain::execute`](ryft_core::CompilationDomain::execute), and returns its public outputs together
    /// with the completion fence of the whole execution.
    ///
    /// The returned output arrays already carry the fence individually, so `CompilationDomain::execute` drops the
    /// returned fence. The stateful asynchronous surface threads it into a completion instead, so that awaiting a
    /// reference-free invocation still observes asynchronous execution errors. When the program has no outputs, the
    /// effect scope retains effectful completion. Only pure calls without outputs explicitly wait.
    pub(super) fn execute_stateless(
        &self,
        program: &CompiledProgram,
        inputs: Vec<Value>,
    ) -> Result<(Vec<Value>, ExecutionFence), Error> {
        if program.requires_stateful_call() {
            return Err(Error::Execution(ExecutionError::StatefulInvocationRequired));
        }
        check_count!("input", inputs, program.input_types().len(), ProgramError);
        for (declared, input) in program.input_types().iter().zip(&inputs) {
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
        if execution.output().is_empty() && !program.signature().has_effects() {
            execution.fence().block_until_ready()?;
        }
        let (outputs, fence) = execution.into_parts();
        validate_runtime_outputs(program.output_types(), &outputs)?;
        Ok((outputs.into_iter().map(Value::Array).collect(), fence))
    }

    /// Executes a program with array inputs. Output arrays retain completion, while an execution without outputs or
    /// effects waits here so failures remain observable.
    pub(crate) fn execute_program(&self, program: &CompiledProgram, inputs: Vec<Array>) -> Result<Vec<Array>, Error> {
        let execution = self.execute_compiled_async(program, inputs)?;
        if execution.output().is_empty() && !program.signature().has_effects() {
            // No output array carries the execution fence out of this call, so an asynchronous execution error would
            // otherwise be unobservable. Block here so this synchronous entry point still reports it.
            let (outputs, fence) = execution.into_parts();
            fence.block_until_ready()?;
            return Ok(outputs);
        }
        Ok(execution.into_output())
    }

    /// Validates the program and prepares its runtime resources before argument materialization and effect reservation.
    fn prepare_execution_resources(&self, program: &CompiledProgram) -> Result<ExecutionResources, Error> {
        ensure_effect_dispatch_is_allowed()?;
        self.validate_program_owner(program)?;
        if program.kernel_execution_facts().is_some() {
            crate::kernels::distributed::validate_kernel_participants(self.client(), program.mesh())?;
        }
        self.runtimes().prepare(program.runtime_requirements(), program.mesh())
    }

    /// Materializes and validates physical execution arguments without submitting the executable.
    fn prepare_compiled_execution(
        &self,
        program: &CompiledProgram,
        inputs: Vec<Array>,
        donation_overrides: &[(usize, bool)],
    ) -> Result<PreparedExecution, Error> {
        if inputs.len() != program.input_types().len() {
            return Err(Error::from(ProgramError::InvalidInputCount {
                expected: program.input_types().len(),
                actual: inputs.len(),
            }));
        }
        let plan = program.plan();

        // Static input types establish no refinements, so refining them reduces to comparing every component except
        // the sharding, which is normalized separately at the executable boundary. Inputs that differ, as well as
        // dynamic input types, take the general path below, which also produces the precise diagnostic.
        let static_inputs_match = plan.static_inputs
            && program.input_types().iter().zip(inputs.iter()).all(|(declared, input)| {
                let actual = input.r#type();
                declared.data_type() == actual.data_type()
                    && declared.shape() == actual.shape()
                    && declared.layout() == actual.layout()
                    && declared.memory() == actual.memory()
            });
        let (input_refinements, actual_input_types) = if static_inputs_match {
            (<ArrayType as CoreType>::Refinements::default(), None)
        } else {
            let actual_input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            let refinement_input_types = program
                .input_types()
                .iter()
                .zip(actual_input_types.iter())
                .map(|(declared, actual)| {
                    // Preserve every observed type component except the sharding, so that refinement still rejects a
                    // wrong data type, layout, memory space, rank, or extent.
                    actual.clone().with_sharding(declared.sharding().cloned()).map_err(Error::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let input_refinements = <ArrayType as CoreType>::Refinements::from_local_declarations(
                program.input_types().iter(),
                refinement_input_types.iter(),
            )
            .map_err(ProgramError::from)?;
            (input_refinements, Some(actual_input_types))
        };

        // Validate the logical public arguments before padding, erasing, donating, or moving any buffers.
        // Captures, including hidden session constants, are placed on the executable's mesh.
        for (index, input) in inputs.iter().skip(program.capture_count()).enumerate() {
            validate_committed_input_devices(input, index, program.mesh())?;
        }

        // Erased inputs of the zero data type have no buffers through which PJRT could order consumer work after their
        // producers. Observe completion before dropping those inputs, including before any consumer effects run.
        for (input, physical_index) in inputs.iter().zip(program.signature().input_mapping()) {
            if physical_index.is_none() {
                input.block_until_ready()?;
            }
        }

        // Without hidden extent scalars, physical inputs are a projection of the logical inputs, which is the identity
        // for most programs. Bounded dynamic inputs are padded to their physical bounds instead.
        let inputs = if program.signature().input_dimensions().is_empty() {
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
            self.materialize_bounded_dynamic_inputs(
                program.signature(),
                program.input_types(),
                actual_input_types.as_slice(),
                inputs,
            )?
        };
        let inputs = if plan.zero_space_inputs {
            self.materialize_zero_space_carriers(plan.physical_input_types.as_slice(), inputs)?
        } else {
            inputs
        };
        let mut inputs =
            self.reshard_inputs_if_needed(program.mesh(), program.expected_argument_shardings(), inputs)?;
        // A logical input without a layout constraint admits any physical representation. Match the compiled
        // signature before handing buffers to PJRT, which otherwise reads their storage in its expected order.
        // Known layouts convert on device through the existing capability. Unknown layouts require observing the
        // shard buffers, whose layouts may differ. When one of them differs from the expected layout, the storage
        // placement of the input converts every mismatching shard on its own and keeps the others, on the device or
        // through host memory in the background, so that preparation never waits for the producer of a shard. It
        // keeps the placement of the input, because the input type does not record the splits along auto mesh axes
        // (e.g., of an input placed by the compiler) and a dense readback would combine the values of unreduced
        // shards, so neither can be rebuilt from the whole array, and its result joins the execution fence of the
        // input with the completion of every conversion.
        for (input, physical_type) in inputs.iter_mut().zip(&plan.physical_input_types) {
            if input.data_type().is_zero() {
                continue;
            }
            // The executable of a plugin without the layouts extension reports no layout to match.
            let Some(expected) = physical_type.layout() else {
                continue;
            };
            if let Some(actual) = input.layout() {
                if actual != expected {
                    let source = input.clone().with_session(self)?;
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
                if needs_conversion {
                    // Keep the commitment of the input, which the placement would otherwise set, because outputs
                    // are committed when an input of the execution is.
                    let committed = input.is_committed();
                    let target =
                        Placement::new(input.mesh().clone(), input.sharding().clone())?.with_layout(expected.clone());
                    *input = input.clone().run_placement_dispatch(self, &target, false)?.with_committed(committed);
                }
            }
        }
        let donation_flags = if donation_overrides.is_empty() {
            plan.default_donation_flags.clone()
        } else {
            let mut logical_donation_flags = std::iter::repeat_n(false, program.capture_count())
                .chain(program.donation_flags().iter().copied())
                .collect::<Vec<_>>();
            for &(logical_input_index, donation) in donation_overrides {
                let flag = logical_donation_flags.get_mut(logical_input_index).ok_or_else(|| {
                    Error::from(ProgramError::InvalidArgument {
                        message: format!("donation override input {logical_input_index} is out of range"),
                    })
                })?;
                *flag = donation;
            }
            let mut donation_flags = program.signature().project_inputs(logical_donation_flags.as_slice());
            donation_flags.extend(std::iter::repeat_n(false, program.signature().input_dimensions().len()));
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
            physical_output_count: program.signature().array_output_count(),
        })
    }

    /// Locks one stateful reference while preserving reference lifecycle errors through [`Error::Core`].
    fn lock_stateful_reference<'g>(
        reference: &'g ArrayReference<Array>,
    ) -> Result<ReadyOrPendingReferenceGuard<'g, Array>, Error> {
        reference.lock_storage().map_err(|error| {
            if let Some(reference_error) = error.downcast_custom::<ReferenceError>().cloned() {
                Error::from(reference_error)
            } else {
                Error::from(error)
            }
        })
    }

    /// Submits a stateful execution of a compiled program on flat `inputs`, which implements
    /// [`StatefulCompilationDomain::execute_statefully`], publishes its dependencies, and commits hidden final states
    /// without waiting for device completion when the logical boundary is static.
    fn execute_stateful_async(
        &self,
        program: &CompiledProgram,
        inputs: Vec<Value>,
    ) -> StatefulExecution<Vec<Value>, Error> {
        if let Err(error) = ensure_effect_dispatch_is_allowed() {
            return StatefulExecution::ready(Err(error));
        }
        if program.reference_states().is_empty() {
            // A reference-free invocation binds no holders, but the completion contract still promises that awaiting
            // the result observes completion of the whole invocation, so the execution fence rides along as the
            // completion.
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
            check_count!("input", inputs, program.input_types().len(), ProgramError);
            let reference_state_input_indices = program
                .reference_states()
                .iter()
                .map(|state| state.source().flat_input_index(program.capture_count()))
                .collect::<Result<Vec<_>, _>>()?;

            // Discharge retains each reference's position but replaces its type with the referent's array type.
            // The compiled binding table restores that distinction without a signature supplied by the caller. Its
            // indices are validated in increasing order during lowering and persistent executable restoration.
            for (index, (declared, input)) in program.input_types().iter().zip(&inputs).enumerate() {
                let expects_reference = reference_state_input_indices.binary_search(&index).is_ok();
                match (expects_reference, input) {
                    (false, Value::Array(array)) => validate_input_type(declared, &array.r#type())?,
                    (true, Value::Reference(reference)) => {
                        validate_input_type(declared, reference.r#type().referent())?;
                    }
                    _ => {
                        let actual = input.r#type();
                        let declared = if expects_reference {
                            Type::Reference(ReferenceType::new(declared.clone()))
                        } else {
                            Type::Array(declared.clone())
                        };
                        return Err(ProgramError::InvalidArgument {
                            message: format!("runtime input type `{actual}` does not match declared type `{declared}`"),
                        }
                        .into());
                    }
                }
            }

            let (captures, public_inputs) = inputs.split_at(program.capture_count().min(inputs.len()));
            validate_reference_boundary(public_inputs.iter(), captures.iter()).map_err(ProgramError::from)?;
            let mut bindings =
                Vec::<(ReferenceId, ArrayReference<Array>)>::with_capacity(program.reference_states().len());
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
                            expected: program.input_types().len(),
                            actual: inputs.len(),
                        }
                        .into());
                    }
                };
                let physical_input_index = program.signature().input_mapping()[logical_input_index].unwrap();
                let expected_sharding = &program.expected_argument_shardings()[physical_input_index];
                let declared_type = &program.input_types()[logical_input_index];
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
            // Every path in this repository that locks multiple holders acquires their guards in ascending identity
            // order. This ordering is the invariant that prevents deadlocks between overlapping stateful calls, and it
            // must be preserved while guards remain held from submission through the commit of hidden state
            // replacements below.
            bindings.sort_by_key(|(id, _)| *id);
            let guard_indices =
                bindings.iter().enumerate().map(|(index, (id, _))| (*id, index)).collect::<BTreeMap<_, _>>();

            let mut mutated_outputs = BTreeMap::new();
            for (state, &logical_input_index) in program.reference_states().iter().zip(&reference_state_input_indices) {
                let Value::Reference(reference) = &inputs[logical_input_index] else {
                    unreachable!("reference bindings were validated before holder acquisition")
                };
                let guard_index = guard_indices[&reference.handle().id()];
                if let Some(logical_output_index) = state.output_index() {
                    mutated_outputs.insert(guard_index, logical_output_index);
                }
            }
            let mut inputs = Some(inputs);

            // Snapshot retries reuse the same invocation resources. Prepare them before acquiring reference guards.
            let mut resources = Some(self.prepare_execution_resources(program)?);
            let donation_overrides =
                reference_state_input_indices.iter().map(|&index| (index, false)).collect::<Vec<_>>();

            // The outputs are committed when a public input of the attempt that is submitted, including a reference
            // snapshot, is committed. Captures never commit them. Each attempt sets this before it can be submitted.
            let mut committed;
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
                    .collect::<Result<Vec<ReadyOrPendingReferenceGuard<'_, Array>>, Error>>()?;
                let observations =
                    guards.iter().map(ReadyOrPendingReferenceGuard::observe).collect::<Result<Vec<_>, _>>()?;
                let logical_inputs = inputs.as_ref().unwrap();
                let mut execution_inputs = logical_inputs.clone();
                for &logical_input_index in &reference_state_input_indices {
                    let Value::Reference(reference) = &logical_inputs[logical_input_index] else {
                        unreachable!("reference bindings were validated before holder acquisition")
                    };
                    let snapshot = observations[guard_indices[&reference.handle().id()]].snapshot().clone();
                    if snapshot.mesh() != program.mesh() {
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
                committed = arrays.iter().skip(program.capture_count()).any(Array::is_committed);
                let prepared_execution =
                    self.prepare_compiled_execution(program, arrays, donation_overrides.as_slice())?;

                let guards = bindings
                    .iter()
                    .map(|(_, reference)| Self::lock_stateful_reference(reference))
                    .collect::<Result<Vec<ReadyOrPendingReferenceGuard<'_, Array>>, Error>>()?;
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
                // the same generation. Retaining its dependency, which has now completed, is harmless and keeps the
                // value, generation, and dependency of each observation coherent throughout the retry attempt.
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
                let after_submission = |fence: &ExecutionFence| {
                    let mut completions = dependencies.clone();
                    completions.push(CoreReferenceCompletion::new(ReferenceCompletion { fence: fence.clone() }));
                    #[cfg(test)]
                    if let Some(injected) = tests::take_stateful_completion_injection() {
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
                };
                let execution = self.execute_pjrt_buffers(
                    program,
                    execution_inputs,
                    physical_output_count,
                    &mut resources,
                    after_submission,
                );
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
                if tests::take_stateful_failure_injection(
                    tests::StatefulFailureInjection::BeforeHiddenReplacementCommit,
                ) {
                    return Err(Error::Pjrt(ryft_pjrt::Error::internal(
                        "injected failure before hidden state replacement commit".to_string(),
                    )));
                }
                let hidden_outputs = self.reconstruct_compiled_outputs(
                    program,
                    &mut physical_outputs,
                    fence.clone(),
                    program.output_count()..program.all_output_types().len(),
                    committed,
                )?;
                let mut hidden_outputs = (program.output_count()..program.all_output_types().len())
                    .zip(hidden_outputs)
                    .collect::<BTreeMap<_, _>>();
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
            // Mutated hidden outputs are statically shaped under the current ABI, so neither queries of logical buffer
            // shapes nor materialization of runtime extents can wait for device execution during the replacement
            // commit. Release every reference guard before public reconstruction, where dynamic outputs may require
            // either wait.
            drop(committed_replacement_guards);
            drop(read_only_guards);

            let public_result = (|| -> Result<_, Error> {
                #[cfg(test)]
                if tests::take_stateful_failure_injection(tests::StatefulFailureInjection::BeforePublicReconstruction) {
                    return Err(Error::Pjrt(ryft_pjrt::Error::internal(
                        "injected failure before public output reconstruction".to_string(),
                    )));
                }
                let public_outputs = self.reconstruct_compiled_outputs(
                    program,
                    &mut physical_outputs,
                    fence,
                    0..program.output_count(),
                    committed,
                )?;
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

    /// Submits at the common physical boundary and transposes array results, retaining hidden effect tokens in the
    /// active scope. No scope lock is held across PJRT, which may execute callbacks inline on this thread.
    ///
    /// # Parameters
    ///
    ///   - `program`: Compiled program to execute.
    ///   - `arguments`: Physical array inputs grouped by device, to which this function appends the hidden effect
    ///     tokens of ordered programs.
    ///   - `array_output_count`: Number of physical array outputs that every device returns.
    ///   - `resources`: Prepared invocation resources. This function takes them at the PJRT handoff so the caller
    ///     can release any reference guards before dropping the resources after an earlier failure.
    ///   - `after_submission`: Callback that runs synchronously on this thread, exactly once, with the fence of the
    ///     whole execution right after PJRT accepts the submission and before the results are checked. It does not
    ///     run when submission fails.
    fn execute_pjrt_buffers(
        &self,
        program: &CompiledProgram,
        mut arguments: ExecutionArguments,
        array_output_count: usize,
        resources: &mut Option<ExecutionResources>,
        after_submission: impl FnOnce(&ExecutionFence),
    ) -> Result<Execution<Vec<Vec<Buffer<'static>>>>, Error> {
        ensure_effect_dispatch_is_allowed()?;
        let addressable_device_count = arguments.addressable_device_ids().len();
        let ordered = program.signature().has_ordered_io();
        if ordered {
            if !self.target()?.is_fully_addressable_single_process_mesh(program.mesh()) {
                return Err(ExecutionError::UnsupportedEffectDeviceAssignment.into());
            }
            if addressable_device_count > 1 && !program.signature().has_device_ordered_io() {
                return Err(ExecutionError::MissingDeviceOrderedEffect.into());
            }
        }
        let devices = ordered.then_some(arguments.addressable_device_ids());
        let mut work = self.reserve_work();
        let reservation = if program.signature().has_effects() { Some(self.reserve_effects(devices)?) } else { None };
        if let Some(reservation) = &reservation
            && ordered
        {
            let predecessor_tokens = reservation
                .wait_for_predecessor_tokens()
                .map_err(|reason| Error::Execution(ExecutionError::AsynchronousFailure { reason }))?;
            let tokens = match predecessor_tokens {
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
                        // Only the initial creation of empty tokens is awaited. Predecessor token buffers can be used
                        // as pending inputs.
                        manager.transfer_data(0, Arc::new([] as [u8; 0]), 0, true)?.r#await()?;
                        tokens.push(Arc::new(token));
                    }
                    tokens
                }
            };
            arguments.push_input_buffers(tokens);
        }
        debug_assert!(
            arguments
                .inputs_by_device()
                .iter()
                .all(|inputs| inputs.len() == program.signature().physical_input_count())
        );
        #[cfg(test)]
        if tests::take_stateful_failure_injection(tests::StatefulFailureInjection::BeforeSubmission) {
            return Err(Error::Pjrt(ryft_pjrt::Error::internal(
                "injected failure before execution submission".to_string(),
            )));
        }
        let (context, retained_resources) = resources.take().unwrap().into_parts();
        // These leases stay in this stack frame on every error path. PJRT destroys the context before returning an
        // error, and if submission succeeded it first waits for every device, so the leases outlive all native use.
        let (mut device_outputs, fence) = program
            .executable()
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, context, None, None, None, &[])?
            .into_parts();
        work.track_fence(fence.clone());
        // The fence releases the context before it reports readiness, and the session releases these leases only after
        // that, so pointees of the context's user data outlive native context destruction.
        self.runtimes().retain(&fence, retained_resources);
        // Reference publication must happen after every successful handoff, including a subsequent marshalling error.
        after_submission(&fence);
        let output_count = program.signature().physical_output_count();
        debug_assert_eq!(array_output_count, program.signature().array_output_count());
        let invalid_outputs = device_outputs
            .iter()
            .find(|outputs| outputs.outputs.len() != output_count)
            .map(|outputs| format!("expected {output_count} output(s) per device, but got {}", outputs.outputs.len()));
        let boundary_error = invalid_outputs.as_ref().map(|message| {
            let error = Error::Pjrt(ryft_pjrt::Error::invalid_argument(message.clone()));
            work.track_error(error.clone());
            error
        });
        if let Some(reservation) = reservation {
            let tokens = if let Some(message) = invalid_outputs {
                Err(Arc::from(message))
            } else if ordered {
                Ok(Some(
                    device_outputs
                        .iter_mut()
                        .map(|outputs| {
                            Arc::new(outputs.outputs.remove(program.signature().ordered_io_output_index().unwrap()))
                        })
                        .collect(),
                ))
            } else {
                Ok(None)
            };
            reservation.publish(tokens, &fence);
        }
        if let Some(error) = boundary_error {
            return Err(error);
        }

        #[cfg(test)]
        if tests::take_stateful_failure_injection(tests::StatefulFailureInjection::AfterHandoff) {
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

    /// Places `inputs` on `mesh` according to `expected_shardings` at the call boundary. Inputs that already match skip
    /// the placement entirely, and the implicit placement is the cold path. An input matches only if it is also on
    /// `mesh`, because the same sharding assigns its shards to other devices on another device mesh (e.g., one that
    /// lists the same devices in another order).
    fn reshard_inputs_if_needed(
        &self,
        mesh: &DeviceMesh,
        expected_shardings: &[Sharding],
        inputs: Vec<Array>,
    ) -> Result<Vec<Array>, Error> {
        let matches = |array: &Array, expected: &Sharding| array.sharding() == expected && array.mesh() == mesh;
        if inputs.iter().zip(expected_shardings).all(|(array, expected)| matches(array, expected)) {
            return Ok(inputs);
        }
        inputs
            .into_iter()
            .zip(expected_shardings)
            .map(|(array, expected)| {
                if matches(&array, expected) {
                    Ok(array)
                } else {
                    array.reshard_on_devices(self, mesh, expected, false)
                }
            })
            .collect()
    }

    /// Pads bounded-dynamic inputs to static executable storage and appends their logical extents. An executable
    /// compiled for bounded dynamic shapes takes every such input with its static bound shape, followed by a scalar
    /// argument for each logical extent, and this function produces those physical inputs in that order.
    ///
    /// Every bounded input takes one of three paths:
    ///
    ///   - An input whose actual shape equals its physical bound is passed through as it is.
    ///   - A smaller input is padded on its devices with zeros after its last element along every axis, by applying
    ///     [`Pad`] to it through this session. The padding is an ordinary eager operation, so its compiled program is
    ///     cached by the session per input type and padding amounts, no data moves through the host, and the padded
    ///     array is owned by the returned inputs alone. Nothing is retained on the source array, so the copy with the
    ///     bound shape is released as soon as the execution that consumes it lets go of it.
    ///   - A zero-space input below its bound is passed through as well, because
    ///     [`Self::materialize_zero_space_carriers`] synthesizes its carrier with the bound shape directly.
    ///
    /// The padded arrays may still be pending when this function returns. The executable that takes them as inputs
    /// waits for them, and a failure of the padding surfaces as a failure of that execution.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] if a bounded input has no static bound shape or no static runtime
    /// shape, exceeds its bound, has shards in other processes while it is below its bound, or has an extent that does
    /// not fit in an `i32`. Errors of the padding operation and of the uploads of the scalars are returned unchanged.
    fn materialize_bounded_dynamic_inputs(
        &self,
        signature: &ExecutableSignature,
        declared_types: &[ArrayType],
        actual_types: &[ArrayType],
        inputs: Vec<Array>,
    ) -> Result<Vec<Array>, Error> {
        assert_eq!(declared_types.len(), inputs.len());
        assert_eq!(actual_types.len(), inputs.len());
        let physical_types = signature.physical_input_types(declared_types);
        let mut physical_inputs = signature.project_inputs(inputs.as_slice());
        for logical_input_index in 0..inputs.len() {
            if !signature
                .input_dimensions()
                .iter()
                .any(|input_dimension| input_dimension.logical_input_index() == logical_input_index)
            {
                continue;
            }
            let physical_input_index = signature.input_mapping()[logical_input_index].unwrap();
            let physical_type = &physical_types[physical_input_index];
            let physical_shape = physical_type.static_shape().ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("bounded-dynamic executable input {logical_input_index} has no static upper shape"),
            })?;
            let actual_shape =
                actual_types[logical_input_index].static_shape().ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("runtime executable input {logical_input_index} must have a static shape"),
                })?;

            // A zero-space input below its bound carries no payload to pad. The zero-space carrier materialization
            // that follows this boundary synthesizes its all-false predicate with the bound shape directly instead.
            if actual_shape == physical_shape || physical_type.data_type().is_zero() {
                continue;
            }

            let input = &inputs[logical_input_index];
            if !input.shards().iter().all(ArrayShard::is_addressable) {
                // Padding below runs on the devices of this process alone. Reject an input with a shard in another
                // process explicitly instead of failing deep inside the execution of the padding.
                return Err(ProgramError::InvalidArgument {
                    message: format!(
                        "bounded-dynamic executable input {logical_input_index} is below its bound but has \
                         non-addressable shards, so it cannot be padded on this process",
                    ),
                }
                .into());
            }

            // The elements stay at the origin, so all padding follows the last element along every axis.
            let actual_extents = actual_shape.as_slice();
            let bound_extents = physical_shape.as_slice();
            let padding = (actual_extents.len() == bound_extents.len())
                .then(|| {
                    actual_extents
                        .iter()
                        .zip(bound_extents)
                        .map(|(actual, bound)| {
                            bound.checked_sub(*actual).and_then(|padding| i64::try_from(padding).ok())
                        })
                        .collect::<Option<Vec<_>>>()
                })
                .flatten()
                .ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!(
                        "cannot pad runtime shape `{actual_extents:?}` into bounded physical shape `{bound_extents:?}`",
                    ),
                })?;
            let rank = padding.len();
            let data_type = input.data_type();
            let zero_type =
                ArrayType::scalar(data_type).with_sharding(Sharding::replicated(input.sharding().mesh().clone(), 0))?;
            let zero_bytes = vec![0u8; data_type.size_in_bytes()?];
            let zero = Array::from_host_buffer(self, zero_type, input.mesh().clone(), None, zero_bytes)?;

            // The operation runs in this session, which executes the program and is not necessarily the session of
            // the input.
            let input = input.clone().with_session(self)?;
            physical_inputs[physical_input_index] =
                input.pad(&zero, vec![0i64; rank].as_slice(), padding.as_slice(), vec![0usize; rank].as_slice())?;
        }

        for input_dimension in signature.input_dimensions() {
            let input = &inputs[input_dimension.logical_input_index()];
            let extent = actual_types[input_dimension.logical_input_index()].shape().dimensions()
                [input_dimension.axis()]
            .value()
            .unwrap();
            let extent = i32::try_from(extent).map_err(|_| ProgramError::InvalidArgument {
                message: format!("runtime dimension extent {extent} does not fit in a StableHLO `i32` scalar"),
            })?;

            // The scalar is uploaded for this call alone and replicated over the mesh of its input.
            let extent_type = ArrayType::scalar(DataType::I32)
                .with_sharding(Sharding::replicated(input.sharding().mesh().clone(), 0))?;
            let extent_scalar =
                Array::from_host_buffer(self, extent_type, input.mesh().clone(), None, extent.to_ne_bytes())?;
            physical_inputs.push(extent_scalar);
        }
        assert_eq!(physical_inputs.len(), signature.array_input_count());
        Ok(physical_inputs)
    }

    /// Materializes private carriers of false predicates for inputs of the zero data type that remain in the physical
    /// executable signature, currently because their declared type contains dynamic dimensions. Static inputs of the
    /// zero data type are erased before this point and never allocate a carrier.
    ///
    /// Each carrier is allocated at the *physical* type of the input (i.e., its type padded to its bounds) from
    /// `physical_input_types`, not at the runtime type of the value. The compiled module declares the `i1` argument at
    /// the bounds and the hidden extent scalar transports the logical size, so an input of the zero data type that is
    /// smaller than its bounds must still hand PJRT a buffer of the bounded shape.
    fn materialize_zero_space_carriers(
        &self,
        physical_input_types: &[ArrayType],
        inputs: Vec<Array>,
    ) -> Result<Vec<Array>, Error> {
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
                let element_count = carrier_type.element_count()?.unwrap();
                Array::from_host_buffer(self, carrier_type, input.mesh().clone(), None, vec![0u8; element_count])
            })
            .collect()
    }

    /// Reconstructs selected logical outputs from one submitted execution's physical buffers. The outputs are
    /// [committed](Array::is_committed) to their devices when `committed` is `true`, which is the case when any public
    /// array input of the execution is committed. The arrays that the program captures never commit them. The outputs
    /// that [`CompiledProgram::committed_outputs`] flags (e.g., the public outputs of a program that requests a
    /// placement explicitly, or all of them when the program runs on more than one device) are committed in either
    /// case, which both the plain and the stateful execution paths rely on. Eager operations and compiled reshards
    /// replace the commitment of the arrays that this function returns with their own.
    fn reconstruct_compiled_outputs(
        &self,
        program: &CompiledProgram,
        physical_outputs: &mut [Option<Vec<Buffer<'static>>>],
        fence: ExecutionFence,
        logical_indices: impl IntoIterator<Item = usize>,
        committed: bool,
    ) -> Result<Vec<Array>, Error> {
        let committed = |logical_index: usize| committed || program.committed_outputs()[logical_index];
        // Programs with only static outputs build each output directly from its plan, without querying the returned
        // buffers or re-deriving their placement.
        if let Some(output_plans) = &program.plan().output_plans {
            return logical_indices
                .into_iter()
                .map(|logical_index| {
                    let buffers = match program.signature().output_mapping()[logical_index] {
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
                    let output =
                        Array::from_execution_output(self, &output_plans[logical_index], buffers, fence.clone());
                    Ok(output.with_committed(committed(logical_index)))
                })
                .collect();
        }
        let logical_indices = logical_indices.into_iter().collect::<Vec<_>>();
        let scalar_type = ArrayType::scalar(DataType::I64).replicated(program.mesh())?;
        let mut output_extents = BTreeMap::new();
        for output_dimension in program
            .signature()
            .output_dimensions()
            .iter()
            .filter(|dimension| logical_indices.contains(&dimension.logical_output_index()))
        {
            let physical_output_index = output_dimension.physical_output_index();
            let physical_output =
                physical_outputs.get_mut(physical_output_index).and_then(Option::take).ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "physical output {physical_output_index} for runtime extent of output {} axis {} is missing \
                         or was claimed twice",
                        output_dimension.logical_output_index(),
                        output_dimension.axis(),
                    ))
                })?;
            let scalar = Array::from_canonical_addressable_buffers(
                self,
                scalar_type.clone(),
                None,
                program.mesh().clone(),
                physical_output,
            )?
            .with_execution_fence(fence.clone());
            let bytes = scalar.copy_to_host_bytes()?;
            let extent = bytes
                .get(..size_of::<i64>())
                .map(|bytes| i64::from_ne_bytes(bytes.try_into().unwrap()))
                .and_then(|extent| usize::try_from(extent).ok())
                .ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!(
                        "runtime extent for output {} axis {} is negative, missing, or exceeds `usize`",
                        output_dimension.logical_output_index(),
                        output_dimension.axis(),
                    ),
                })?;
            output_extents.insert((output_dimension.logical_output_index(), output_dimension.axis()), extent);
        }

        logical_indices
            .into_iter()
            .map(|logical_index| {
                let output_type = &program.all_output_types()[logical_index];
                match program.signature().output_mapping()[logical_index] {
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
                                                "executable omitted runtime extent for output {logical_index} \
                                                 axis {axis}",
                                            )))
                                        },
                                        |extent| Ok(Dimension::Static(extent)),
                                    )
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let output_type = output_type.clone().with_shape(Shape::new(dimensions));
                        Ok(Array::from_canonical_addressable_buffers(
                            self,
                            output_type,
                            Some(program.output_shardings()[logical_index].clone()),
                            program.mesh().clone(),
                            physical_outputs.get_mut(physical_index).and_then(Option::take).ok_or_else(|| {
                                ProgramError::MalformedProgram(format!(
                                    "physical output {physical_index} for logical output {logical_index} is \
                                     missing or was claimed twice",
                                ))
                            })?,
                        )?
                        .with_execution_fence(fence.clone())
                        .with_committed(committed(logical_index)))
                    }
                    None => Ok(Array::zero(
                        self,
                        output_type.clone(),
                        Some(program.output_shardings()[logical_index].clone()),
                        program.mesh().clone(),
                    )?
                    .with_execution_fence(fence.clone())
                    .with_committed(committed(logical_index))),
                }
            })
            .collect()
    }
}

impl StatefulCompilationDomain for Session {
    fn execute_statefully(
        &self,
        program: &CompiledProgram,
        inputs: Vec<Value>,
    ) -> StatefulExecution<Vec<Value>, Self::Error> {
        self.execute_stateful_async(program, inputs)
    }
}

/// Returns the device IDs of `assignment` ordered by replica first and by computation partition second.
pub(super) fn flatten_device_assignment(assignment: &ryft_pjrt::DeviceAssignment) -> Result<Vec<u64>, Error> {
    let mut devices = Vec::with_capacity(assignment.replica_count() * assignment.computation_count());
    for replica in 0..assignment.replica_count() {
        for partition in 0..assignment.computation_count() {
            devices.push(assignment.device_id(replica, partition)? as u64);
        }
    }
    Ok(devices)
}

/// Checks the commitment of a public argument against the physical devices of a fixed call boundary.
/// The order identifies the device assigned to each partition. Logical mesh names and shardings can differ.
pub(crate) fn validate_committed_input_devices(
    input: &Array,
    index: usize,
    mesh: &DeviceMesh,
) -> Result<(), ExecutionError> {
    if input.is_committed() && input.mesh().devices() != mesh.devices() {
        return Err(ExecutionError::CommittedInputDeviceMismatch {
            index,
            actual: input.mesh().devices().iter().map(Device::id).collect(),
            expected: mesh.devices().iter().map(Device::id).collect(),
        });
    }
    Ok(())
}

/// Validates runtime refinement of a declared logical input type, excluding the placement that a sharding describes,
/// because an execution reshards its arguments as its executable expects. A reshard cannot change a pending reduction
/// (i.e., the unreduced axes of a sharding and their kind), so the two types must agree on it. This is the rule by
/// which an executable accepts its arguments. The type rule of a `jit_call` accepts the inputs of a call by a stricter
/// rule, because a call does not reshard its inputs (refer to `validate_call_input_types`).
pub(crate) fn validate_input_type(declared: &ArrayType, actual: &ArrayType) -> Result<(), Error> {
    // Calls of a specialization ordinarily pass inputs of exactly its declared types, which refine them trivially.
    if declared == actual {
        return Ok(());
    }
    fn pending_reduction(r#type: &ArrayType) -> Option<(&BTreeSet<String>, UnreducedKind)> {
        r#type
            .sharding()
            .filter(|sharding| !sharding.unreduced_axes().is_empty())
            .map(|sharding| (sharding.unreduced_axes(), sharding.unreduced_kind()))
    }
    let declared_without_sharding = declared.clone().with_sharding(None)?;
    let actual_without_sharding = actual.clone().with_sharding(None)?;
    if pending_reduction(declared) == pending_reduction(actual)
        && declared_without_sharding.is_refined_by(&actual_without_sharding)
    {
        Ok(())
    } else {
        Err(ProgramError::InvalidArgument {
            message: format!("runtime input type `{actual}` does not refine declared type `{declared}`"),
        }
        .into())
    }
}

/// Validates output arity and refinement of the declared logical output types, including sharding.
pub(crate) fn validate_runtime_outputs(declared: &[ArrayType], outputs: &[Array]) -> Result<(), Error> {
    if declared.len() != outputs.len() {
        return Err(ProgramError::InvalidOutputCount { expected: declared.len(), actual: outputs.len() }.into());
    }
    for (declared, actual) in declared.iter().zip(outputs.iter().map(Typed::r#type)) {
        if declared != actual.as_ref() && !declared.is_refined_by(actual.as_ref()) {
            return Err(ProgramError::InvalidArgument {
                message: format!("runtime output type `{actual}` does not refine declared type `{declared}`"),
            }
            .into());
        }
    }
    Ok(())
}

/// Completion adapter for one submitted PJRT execution that implements the backend neutral
/// [`ReferenceCompletionBackend`].
struct ReferenceCompletion {
    /// Fence of the whole execution, which joins every addressable device event.
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
struct PreparedExecution {
    /// Fully materialized physical arrays in executable argument order.
    inputs: Vec<Array>,

    /// Addressable devices in executable order.
    addressable_device_ids: Arc<[DeviceId]>,

    /// Effective physical donation request for every input.
    donation_flags: Vec<bool>,

    /// Runtime refinements established from the unmaterialized logical inputs.
    input_refinements: <ArrayType as CoreType>::Refinements,

    /// Physical result count expected from every addressable device.
    physical_output_count: usize,
}

impl PreparedExecution {
    /// Consumes the materialized arrays into PJRT arguments grouped by device.
    fn into_arguments(self) -> Result<(ExecutionArguments, <ArrayType as CoreType>::Refinements, usize), Error> {
        let arguments = ExecutionArguments::from_arrays_with_donation(
            self.inputs,
            self.addressable_device_ids,
            self.donation_flags.as_slice(),
        )?;
        Ok((arguments, self.input_refinements, self.physical_output_count))
    }
}

/// Validates reconstructed logical outputs against the runtime input refinements.
fn validate_compiled_output_refinements(
    program: &CompiledProgram,
    input_refinements: &<ArrayType as CoreType>::Refinements,
    declared_output_types: &[ArrayType],
    outputs: &[Array],
) -> Result<(), Error> {
    let mut closed_identities = Vec::new();
    for (_, identity) in
        program.input_types().iter().chain(program.all_output_types().iter()).flat_map(CoreType::identities)
    {
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

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;
    use std::sync::{Arc, Barrier, Condvar, Mutex, mpsc};
    use std::time::Duration;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayReferenceTransform, ArrayReferenceTransformIndex, ArrayType, CastOperation, CompilationCacheDomain,
        CompiledFunction, ConstrainSharding, Context, ConvertElementTypeOperation, DataType, Device, DeviceMesh,
        Dimension, DimensionBounds, DimensionFromScalarOperation, DimensionValue, DimensionVariable,
        DynamicSliceOperation, Error as CoreError, LogicalMesh, LoweredFunction, Memory, MeshAxis, MeshAxisType, Place,
        Placeholder, Placement, PrintOperation, ReduceOperation, ReductionKind, ReferenceAddUpdate,
        ReferenceAddUpdateOperation, ReferenceError, ReferenceRead, ReferenceReadOperation, ReferenceType, Shape,
        ShardMapBoundary, Sharding, ShardingDimension, ShardingError, StagedFunction, StaticShape, TiledLayout, Typed,
    };
    use ryft_mlir::{Block, Context as MlirContext};
    use ryft_pjrt::protos::Shape as ShapeProto;
    use ryft_pjrt::{BufferType, Client, ClientOptions, CpuClientOptions, Program};

    use crate::execution::ExecutionArguments;
    use crate::lowering::tests::static_sharded_array_type;
    use crate::sessions::tests::{
        boolean_vector, domain_mesh, f32_scalar, f32_vector, program_array, read_first_shard, read_sharded_f32_array,
        replicated_scalar_type, replicated_vector_type, two_device_mesh,
    };
    use crate::tests::{cpu_client, execution_client, test_compilation_options, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayPlacement, CompileTracer, Constant, Error, ExecutionError, FlatProgram, Options,
        PRINT_CUSTOM_CALL_TARGET, Platform, ProgramBuilder, Reference, RetainedArray, RuntimeError, RuntimeExtension,
        RuntimeExtensions, RuntimeProvider, Session, ToMlir, Type, Value, compile_statefully,
        compile_statefully_with_captures,
    };

    use super::*;

    /// Deterministic failure point of stateful execution, used by regression tests of its transaction protocol.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub(super) enum StatefulFailureInjection {
        /// Fails immediately before PJRT submission, after argument preparation but before publication.
        BeforeSubmission,

        /// Fails after mutable inputs cross the execution handoff boundary.
        AfterHandoff,

        /// Fails after state publication but before the hidden replacements of final states are reconstructed and
        /// committed.
        BeforeHiddenReplacementCommit,

        /// Fails after the hidden replacements of final states are committed but before public outputs are
        /// reconstructed.
        BeforePublicReconstruction,
    }

    thread_local! {
        /// One-shot failure injection scoped to the invoking test thread.
        static STATEFUL_FAILURE_INJECTION: Cell<Option<StatefulFailureInjection>> = const {
            Cell::new(None)
        };

        /// One-shot dependency, independent of the backend, joined into the next stateful execution on the current
        /// test thread.
        static STATEFUL_COMPLETION_INJECTION: RefCell<Option<CoreReferenceCompletion>> = const { RefCell::new(None) };
    }

    /// Consumes the pending failure only when execution reaches its selected injection point.
    pub(super) fn take_stateful_failure_injection(expected: StatefulFailureInjection) -> bool {
        STATEFUL_FAILURE_INJECTION.with(|injection| {
            if injection.get() != Some(expected) {
                return false;
            }
            injection.set(None);
            true
        })
    }

    /// Takes the dependency to join into the next stateful execution on the current test thread.
    pub(super) fn take_stateful_completion_injection() -> Option<CoreReferenceCompletion> {
        STATEFUL_COMPLETION_INJECTION.with(|injection| injection.borrow_mut().take())
    }

    /// Injects one deterministic failure into the next stateful call on the current test thread.
    fn inject_stateful_failure(failure: StatefulFailureInjection) {
        STATEFUL_FAILURE_INJECTION.with(|injection| injection.set(Some(failure)));
    }

    /// Joins one deterministic dependency into the next stateful call on the current test thread.
    fn inject_stateful_completion(completion: CoreReferenceCompletion) {
        STATEFUL_COMPLETION_INJECTION.with(|injection| *injection.borrow_mut() = Some(completion));
    }

    /// Deterministic completion gate used by stateful asynchronous integration tests.
    #[derive(Clone)]
    struct ControlledReferenceCompletion {
        /// Shared terminal result and wait notification.
        state: Arc<(Mutex<ControlledReferenceCompletionState>, Condvar)>,
    }

    /// Mutable state of one [`ControlledReferenceCompletion`].
    struct ControlledReferenceCompletionState {
        /// Terminal result, absent while the completion is pending.
        result: Option<Result<(), Arc<str>>>,

        /// Optional single notification emitted when a caller starts waiting, carrying this gate's label so that
        /// tests sharing one channel across several gates can pin which gate was awaited first.
        await_started: Option<(&'static str, mpsc::Sender<&'static str>)>,
    }

    impl ControlledReferenceCompletion {
        /// Creates a pending completion gate.
        fn new() -> Self {
            Self {
                state: Arc::new((
                    Mutex::new(ControlledReferenceCompletionState { result: None, await_started: None }),
                    Condvar::new(),
                )),
            }
        }

        /// Creates a pending completion that reports `label` when its blocking wait begins.
        fn with_await_notification(label: &'static str, await_started: mpsc::Sender<&'static str>) -> Self {
            Self {
                state: Arc::new((
                    Mutex::new(ControlledReferenceCompletionState {
                        result: None,
                        await_started: Some((label, await_started)),
                    }),
                    Condvar::new(),
                )),
            }
        }

        /// Completes this gate exactly once and wakes blocking waiters.
        fn complete(&self, result: Result<(), Arc<str>>) {
            let (state, ready) = &*self.state;
            let mut state = state.lock().unwrap();
            assert!(state.result.is_none());
            state.result = Some(result);
            ready.notify_all();
        }
    }

    impl ReferenceCompletionBackend for ControlledReferenceCompletion {
        fn r#await(&self) -> Result<(), Arc<str>> {
            let (state, ready) = &*self.state;
            let mut state = state.lock().unwrap();
            if let Some((label, await_started)) = state.await_started.take() {
                await_started.send(label).unwrap();
            }
            while state.result.is_none() {
                state = ready.wait(state).unwrap();
            }
            state.result.clone().unwrap()
        }

        fn is_ready(&self) -> Result<bool, Arc<str>> {
            let state = self.state.0.lock().unwrap();
            state.result.clone().map_or(Ok(false), |result| result.map(|_| true))
        }
    }

    // Returns the type of an `f32` vector with the placement `sharding` whose extent is dynamic and at most eight.
    fn bounded_vector_type(sharding: Sharding) -> ArrayType {
        ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(sharding)
        .unwrap()
    }

    #[test]
    fn test_execution_plan_new_rejects_compilation_options_with_conflicting_layouts() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [2, 3]);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(
                CastOperation::<Type>::unchecked(
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
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let row_major = BufferSpecification::new(BufferType::F64, [2, 3]).proto().unwrap();
        // Explicit argument layouts bypass the MLIR layout attributes in PJRT. A conflicting result override must
        // fail before its buffers can be wrapped in types that claim another layout.
        let mut compilation_options = lowered.compilation_options().clone();
        compilation_options.argument_layouts = vec![row_major.clone()];
        compilation_options.executable_build_options.as_mut().unwrap().result_shape = Some(row_major);
        let lowered = lowered.with_compilation_options(compilation_options.clone());
        let Err(error) = session.compile_program(&lowered) else {
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
        compilation_options.executable_build_options.as_mut().unwrap().result_shape = Some(column_major.clone());
        let lowered = lowered.with_compilation_options(compilation_options.clone());
        session.compile_program(&lowered).unwrap();
        compilation_options.argument_layouts = vec![column_major];
        let lowered = lowered.with_compilation_options(compilation_options);
        let compiled = session.compile_program(&lowered).unwrap();
        let values = vec![1f64, 2., 3., 4., 5., 6.];
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F64, [2, 3]),
            mesh,
            None,
            values_to_bytes(&values),
        )
        .unwrap();
        let outputs = session.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(outputs[0].copy_to_host().unwrap().elements::<f64>().unwrap(), values);
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F64, [2, 3])
                .with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]))),
            compiled.mesh().clone(),
            None,
            values_to_bytes(&values),
        )
        .unwrap();
        let outputs = session.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(outputs[0].copy_to_host().unwrap().elements::<f64>().unwrap(), values);
    }

    #[test]
    fn test_execution_plan_new_rejects_compilation_options_with_conflicting_memories() {
        // CPU accepts the conflicting layouts at compilation, so this exercises validation of the compiled ABI.
        // CUDA can reject the conflicting memory placements during compilation.
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let device_type = ArrayType::new_static(DataType::F64, [2]);
        let host_type = device_type.clone().with_memory(Memory::Host { pinned: true });
        let lower = |r#type: &ArrayType| {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(r#type.clone().into());
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap();
            session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap()
        };
        // XLA marks host memory in a layout with the memory space `Layout::kHostMemorySpace`, which is five.
        let device_shape = BufferSpecification::new(BufferType::F64, [2]).proto().unwrap();
        let mut host_shape = device_shape.clone();
        host_shape.layout.as_mut().unwrap().memory_space = Some(5);
        // Explicit argument and result layouts bypass the MLIR memory attributes in PJRT. A conflicting override must
        // fail before its buffers can be wrapped in types that claim another memory, in either direction.
        let compile = |r#type: &ArrayType, argument_shape: &ShapeProto, result_shape: &ShapeProto| {
            let lowered = lower(r#type);
            let mut compilation_options = lowered.compilation_options().clone();
            compilation_options.argument_layouts = vec![argument_shape.clone()];
            compilation_options.executable_build_options.as_mut().unwrap().result_shape = Some(result_shape.clone());
            session.compile_program(&lowered.with_compilation_options(compilation_options))
        };
        let message = |result: Result<_, Error>| match result {
            Err(error @ Error::Compilation(CompilationError::IncompatibleExecutable { .. })) => error.to_string(),
            Err(error) => panic!("unexpected error: {error}"),
            Ok(_) => panic!("conflicting executable memory unexpectedly accepted"),
        };
        assert_eq!(
            message(compile(&host_type, &device_shape, &host_shape)),
            "incompatible XLA executable: executable input 0 uses memory kind `device` but its type declares \
             `pinned_host`",
        );
        assert_eq!(
            message(compile(&device_type, &host_shape, &device_shape)),
            "incompatible XLA executable: executable input 0 uses memory kind `pinned_host` but its type declares \
             `device`",
        );
        assert_eq!(
            message(compile(&host_type, &host_shape, &device_shape)),
            "incompatible XLA executable: executable output 0 uses memory kind `device` but its type declares \
             `pinned_host`",
        );
        assert_eq!(
            message(compile(&device_type, &device_shape, &host_shape)),
            "incompatible XLA executable: executable output 0 uses memory kind `pinned_host` but its type declares \
             `device`",
        );

        // Overrides that agree with the declared memories are accepted, and the result occupies the declared memory.
        let compiled = compile(&host_type, &host_shape, &host_shape).unwrap();
        let values = [1f64, 2.];
        let input = Array::from_host_buffer(&session, host_type.clone(), mesh, None, values_to_bytes(&values)).unwrap();
        let outputs = session.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(outputs[0].r#type().memory(), Memory::Host { pinned: true });
        let buffer = outputs[0].addressable_shards().next().unwrap().buffer().unwrap();
        assert_eq!(buffer.memory().unwrap().kind().unwrap(), "pinned_host");
        assert_eq!(outputs[0].copy_to_host().unwrap().elements::<f64>().unwrap(), values);
    }

    #[test]
    fn test_execution_plan_new_plans_static_outputs_on_multiple_cpu_devices() {
        let client = cpu_client(4);
        let mesh = domain_mesh(client, "x", 4);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        // Types record no placement over auto mesh axes, so the placements are input and output sharding overrides.
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |input| input.clone() + input,
            input_type.clone(),
            Options::new(mesh.clone())
                .with_input_shardings(vec![sharding.clone()])
                .with_output_shardings(vec![sharding.clone()]),
        )
        .unwrap()
        .into_inner();
        let compiled: CompiledFunction<Session, Type, Type> =
            staged.lower(&session).unwrap().compile(&session).unwrap();
        assert_eq!(compiled.compiled_program().plan().output_plans.as_ref().map(Vec::len), Some(1));

        // Outputs are built from the program's output plans, which place every shard like the validating
        // constructors do and are shared by the outputs of every execution.
        let values = (0..8).map(|value| value as f32).collect::<Vec<_>>();
        let input = Array::from_host_buffer(
            &session,
            input_type,
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(values.as_slice()).as_slice(),
        )
        .unwrap();
        let call = || {
            let output = compiled.executable().execute(&session, Value::Array(input.clone())).unwrap();
            let Value::Array(output) = output else {
                panic!("array-only compiled function returned a first-class dimension");
            };
            output
        };
        let output = call();
        let other_output = call();
        output.block_until_ready().unwrap();
        assert_eq!(output.r#type(), input.r#type());
        assert_eq!(output.sharding(), &sharding);
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
    fn test_execution_plan_new_plans_no_outputs_without_declared_placement() {
        let client = cpu_client(4);
        let mesh = domain_mesh(client, "x", 4);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let unconstrained_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::unconstrained()]).unwrap();
        let unconstrained_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(unconstrained_sharding)
            .unwrap();
        let unsharded_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // Outputs with open (i.e., `Unconstrained`) dimensions and outputs without declared shardings have no output
        // plans, so their programs validate the returned buffers instead.
        for input_type in [unconstrained_type, unsharded_type] {
            let staged = crate::stage::<_, ArrayType, ArrayType>(
                |input| input.clone() + input,
                input_type,
                Options::new(mesh.clone()),
            )
            .unwrap()
            .into_inner();
            let compiled: CompiledFunction<Session, Type, Type> =
                staged.lower(&session).unwrap().compile(&session).unwrap();
            assert_eq!(compiled.compiled_program().plan().output_plans.as_ref().map(Vec::len), None);

            // Without output plans, the outputs are built from the returned buffers by the validating constructor.
            let values = (0..8).map(|value| value as f32).collect::<Vec<_>>();
            let input = f32_vector(&session, &mesh, values.as_slice());
            let Value::Array(output) = compiled.executable().execute(&session, Value::Array(input)).unwrap() else {
                panic!("array-only compiled function returned a first-class dimension");
            };
            assert_eq!(output.mesh(), &mesh);
            assert_eq!(
                output.copy_to_host().unwrap().elements::<f32>().unwrap(),
                values.iter().map(|value| value * 2.0).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn test_execution_arguments_from_arrays() {
        // Use the same 8-device CPU setup as `ryft_pjrt` tests.
        let client = cpu_client(8);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 8);

        // Build the mesh used for runtime arrays. Just-in-time compilation derives StableHLO Shardy annotations
        // directly from these arrays.
        let mesh = domain_mesh(client, "x", 8);

        let left_sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let right_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2);

        // The global left-hand side input matrix is 8x4 and split by rows across 8 devices, so each shard is 1x4.
        // Row i is [i, i+1, i+2, i+3].
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

        // The global right-hand side input matrix is replicated on each device.
        // [[1, 2], [0, 1], [1, 0], [2, 1]]
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

        let left_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(4)]));
        let right_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)]));
        let left_array =
            Array::from_addressable_buffers(&session, left_array_type, mesh.clone(), Some(left_sharding), left_buffers)
                .unwrap();
        let right_array = Array::from_addressable_buffers(
            &session,
            right_array_type,
            mesh.clone(),
            Some(right_sharding),
            right_buffers,
        )
        .unwrap();

        assert_eq!(left_array.data_type(), DataType::F32);
        assert_eq!(right_array.data_type(), DataType::F32);
        assert_eq!(left_array.addressable_shards().count(), 8);
        assert!(left_array.shards().iter().all(|shard| shard.buffer().is_some()));

        // Derive Shardy attributes from runtime arrays, as just-in-time compilation does.
        let context = ryft_mlir::Context::new();
        let location = context.unknown_location();
        let mesh_module = context.module(location).unwrap();
        let mesh_operation = mesh_module
            .body()
            .unwrap()
            .append_operation(left_array.sharding().mesh().to_mlir(location).unwrap())
            .unwrap()
            .to_string();
        let left_sharding_attribute = left_array.sharding().to_mlir(location).unwrap().to_string();
        let right_sharding_attribute = right_array.sharding().to_mlir(location).unwrap().to_string();

        assert_eq!(mesh_operation, "sdy.mesh @mesh = <[\"x\"=8]>");
        assert_eq!(left_sharding_attribute, "#sdy.sharding<@mesh, [{\"x\"}, {}]>");
        assert_eq!(right_sharding_attribute, "#sdy.sharding<@mesh, [{}, {}]>");

        let mlir_program = format!(
            r#"
                    module {{
                        {mesh_operation}
                        func.func @main(
                            %arg0: tensor<8x4xf32> {{sdy.sharding = {left_sharding_attribute}}},
                            %arg1: tensor<4x2xf32> {{sdy.sharding = {right_sharding_attribute}}}
                        ) -> (tensor<8x2xf32> {{sdy.sharding = {left_sharding_attribute}}}) {{
                            %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0]
                                : (tensor<8x4xf32>, tensor<4x2xf32>) -> tensor<8x2xf32>
                            return %0 : tensor<8x2xf32>
                        }}
                    }}
                "#
        );
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
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

        // Validate each output shard, whose row r should be [4r + 8, 4r + 4].
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
    fn test_execution_arguments_from_arrays_executes_manual_computation_on_cpu() {
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

        let input_array = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            None,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
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
    fn test_execution_arguments_from_arrays_with_donation_respects_buffer_ownership() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();

        let unique = ArrayReference::new(f32_scalar(&session, &mesh, 1.0));
        let unique_guard = unique.lock_storage().unwrap();
        let mut inputs = vec![unique_guard.observe().unwrap().snapshot().clone()];
        drop(ExecutionArguments::from_arrays_with_donation(inputs.clone(), [device_id].into(), &[true]).unwrap());
        let (value, unique_taken) = unique_guard.wait_until_ready().unwrap().take().unwrap();
        inputs[0] = value;
        let arguments = ExecutionArguments::from_arrays_with_donation(inputs, [device_id].into(), &[true]).unwrap();
        assert!(arguments.inputs_by_device()[0][0].donatable);
        unique_taken.poison("test transaction consumed unique input");

        let shared_value = f32_scalar(&session, &mesh, 2.0);
        let retained = shared_value.clone();
        let shared = ArrayReference::new(shared_value);
        let shared_guard = shared.lock_storage().unwrap();
        let mut inputs = vec![shared_guard.observe().unwrap().snapshot().clone()];
        drop(ExecutionArguments::from_arrays_with_donation(inputs.clone(), [device_id].into(), &[true]).unwrap());
        let (value, shared_taken) = shared_guard.wait_until_ready().unwrap().take().unwrap();
        inputs[0] = value;
        let arguments = ExecutionArguments::from_arrays_with_donation(inputs, [device_id].into(), &[true]).unwrap();
        assert!(!arguments.inputs_by_device()[0][0].donatable);
        shared_taken.poison("test transaction consumed shared input");
        assert_eq!(read_first_shard::<f32>(client, &retained), vec![2.0]);
    }

    #[test]
    fn test_execution_arguments_from_arrays_with_donation_rejects_flag_count() {
        assert!(matches!(
            ExecutionArguments::from_arrays_with_donation(Vec::new(), Arc::from([]), &[true]),
            Err(Error::Execution(ExecutionError::DonationFlagCountMismatch { expected_count: 0, actual_count: 1 })),
        ));
    }

    #[test]
    fn test_execution_arguments_from_arrays_with_donation_rejects_duplicate_devices() {
        assert!(matches!(
            ExecutionArguments::from_arrays_with_donation(Vec::new(), Arc::from([7, 7]), &[]),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Sharding(ShardingError::DuplicateDeviceId { id: 7 }),
            ),
        ));
    }

    #[test]
    fn test_session_execute_compiled_async_prepares_runtime_before_arguments() {
        /// Print runtime that initializes successfully but rejects invocation preparation.
        struct RejectingPrintRuntime;

        impl RuntimeProvider for RejectingPrintRuntime {
            fn identifier(&self) -> &str {
                PRINT_CUSTOM_CALL_TARGET
            }

            fn initialize(
                &self,
                client: &'static Client<'static>,
                requirement: &RuntimeRequirement,
            ) -> Result<Arc<dyn RuntimeExtension>, Error> {
                let _ = (client, requirement);
                Ok(Arc::new(Self))
            }
        }

        impl RuntimeExtension for RejectingPrintRuntime {
            fn prepare(&self, resources: &mut ExecutionResources) -> Result<(), Error> {
                let _ = resources;
                Err(RuntimeError::InvalidConfiguration {
                    identifier: PRINT_CUSTOM_CALL_TARGET.to_owned(),
                    reason: "test runtime rejected the invocation".to_owned(),
                }
                .into())
            }
        }

        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(replicated_scalar_type(&mesh, DataType::F32).into());
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("test value"), vec![], vec![input], None)
            .unwrap();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh)).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let extensions = RuntimeExtensions::new().with_replaced_provider(Arc::new(RejectingPrintRuntime)).unwrap();
        let session = Session::builder().client(client).runtime_extensions(extensions).build().unwrap();

        // The missing input makes the error distinguish preparation from argument validation.
        assert!(matches!(
            session.execute_compiled_async(&compiled, vec![]),
            Err(Error::Runtime(RuntimeError::InvalidConfiguration { identifier, reason }))
                if identifier == PRINT_CUSTOM_CALL_TARGET && reason == "test runtime rejected the invocation",
        ));
    }

    #[test]
    fn test_session_execute_compiled_async_commits_outputs_of_committed_inputs() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let wide_mesh = two_device_mesh(client, MeshAxisType::Auto);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let second_mesh = DeviceMesh::new(logical_mesh, vec![wide_mesh.devices()[1]]).unwrap();
        let scalar_type = replicated_scalar_type(&second_mesh, DataType::F32);
        let staged = crate::stage::<_, (ArrayType, ArrayType), (ArrayType, ArrayType)>(
            |(left, right)| (right, left),
            (scalar_type.clone(), scalar_type.clone()),
            Options::new(second_mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled: CompiledFunction<Session, (Type, Type), (Type, Type)> =
            staged.lower(&session).unwrap().compile(&session).unwrap();
        let committed = f32_scalar(&session, &second_mesh, 1.0);
        let uncommitted =
            Array::from_host_buffer(&session, scalar_type, None, None, 2f32.to_ne_bytes().as_slice()).unwrap();
        assert_eq!(uncommitted.mesh().devices(), [wide_mesh.devices()[0]]);
        assert!(!uncommitted.is_committed());

        // The function moves both inputs to the devices that it was compiled for, and its outputs are committed when
        // one of its inputs is, whichever of them an output comes from.
        let inputs = (Value::Array(committed.clone()), Value::Array(uncommitted.clone()));
        let (first, second) = compiled.executable().execute(&session, inputs).unwrap();
        for output in [program_array(&first), program_array(&second)] {
            assert_eq!(output.mesh(), &second_mesh);
            assert!(output.is_committed());
        }
        assert_eq!(read_sharded_f32_array(program_array(&first)), vec![2.0]);
        assert_eq!(read_sharded_f32_array(program_array(&second)), vec![1.0]);

        // Without a committed input, the outputs are uncommitted.
        let inputs = (Value::Array(committed.with_committed(false)), Value::Array(uncommitted));
        let (first, second) = compiled.executable().execute(&session, inputs).unwrap();
        assert!(!program_array(&first).is_committed());
        assert!(!program_array(&second).is_committed());
    }

    #[test]
    fn test_session_execute_compiled_async_commits_outputs_of_explicit_placements() {
        // Every output of a function compiled for more than one device is committed, so the function runs on a single
        // device here, where only the rules below commit its outputs.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [2]);
        let uncommitted =
            Array::from_host_buffer(&session, vector_type.clone(), None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        assert!(!uncommitted.is_committed());

        // Staged types record no placement over auto mesh axes, so overrides and runtime placements over them leave
        // the same staged input type.
        let staged_type = vector_type.with_sharding(sharded.without_auto_axes()).unwrap();
        let lower = |function: fn(CompileTracer, &Placement) -> CompileTracer, options: Options| {
            let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
            crate::stage::<_, ArrayType, ArrayType>(move |input| function(input, &target), staged_type.clone(), options)
                .unwrap()
                .into_inner()
                .lower(&session)
                .unwrap()
        };
        let execute = |lowered: &LoweredFunction<Session, Type, Type>| {
            let compiled: CompiledFunction<Session, Type, Type> = lowered.compile(&session).unwrap();
            let output = compiled.executable().execute(&session, Value::Array(uncommitted.clone())).unwrap();
            assert_eq!(program_array(&output).copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.0, 4.0]);
            program_array(&output).is_committed()
        };
        let double = |input: CompileTracer, _: &Placement| input.clone() + input;
        let double_and_place =
            |input: CompileTracer, target: &Placement| (input.clone() + input).place(target, None).unwrap();
        let place_and_double = |input: CompileTracer, target: &Placement| {
            let placed = input.place(target, None).unwrap();
            placed.clone() + placed
        };
        let constrain_and_double = |input: CompileTracer, target: &Placement| {
            let constrained = input.constrain_sharding(target.sharding()).unwrap();
            constrained.clone() + constrained
        };

        // The outputs of a function on uncommitted inputs are committed when the function requests a placement
        // explicitly anywhere, through a `place` with a target, a sharding constraint, or any sharding override, as in
        // JAX, whether or not that request produces the outputs.
        assert!(execute(&lower(double_and_place, Options::new(mesh.clone()))));
        assert!(execute(&lower(place_and_double, Options::new(mesh.clone()))));
        assert!(execute(&lower(constrain_and_double, Options::new(mesh.clone()))));
        assert!(execute(&lower(double, Options::new(mesh.clone()).with_output_shardings(vec![sharded.clone()]))));
        assert!(execute(&lower(double, Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]))));

        // A request commits every output of the function, including outputs that it does not reach.
        let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
        let staged = crate::stage::<_, ArrayType, (ArrayType, ArrayType)>(
            move |input: CompileTracer| {
                let placed = input.clone().place(&target, None).unwrap();
                (placed.clone() + placed, input)
            },
            staged_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled: CompiledFunction<Session, Type, (Type, Type)> =
            staged.lower(&session).unwrap().compile(&session).unwrap();
        let (first, second) = compiled.executable().execute(&session, Value::Array(uncommitted.clone())).unwrap();
        assert_eq!(program_array(&first).copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.0, 4.0]);
        assert_eq!(program_array(&second).copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0]);
        assert!(program_array(&first).is_committed());
        assert!(program_array(&second).is_committed());

        // Without an explicit request, the outputs keep the rule for the commitment of the inputs.
        assert!(!execute(&lower(double, Options::new(mesh.clone()))));

        // A function that places its input as an override requests and one that places it as its runtime array is
        // placed compile to the same executable, which the second one reuses from the cache, but only the first one
        // requests the placement of its output explicitly.
        let requested = lower(double, Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]));
        let observed =
            lower(double, Options::new(mesh.clone()).with_input_placements(vec![Some(sharded.clone())].into()));
        assert_eq!(
            session.compilation_key(requested.lowered_program()).unwrap(),
            session.compilation_key(observed.lowered_program()).unwrap(),
        );
        session.clear_cache();
        assert!(execute(&requested));
        assert!(!execute(&observed));
        session.clear_cache();
        assert!(!execute(&observed));
        assert!(execute(&requested));
    }

    #[test]
    fn test_session_execute_compiled_async_commits_outputs_of_nested_boundary_shardings() {
        // The function runs on a single device, where only explicit placement requests commit its outputs.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let vector_type = ArrayType::new_static(DataType::I32, [2]);
        let uncommitted =
            Array::from_host_buffer(&session, vector_type.clone(), None, None, values_to_bytes(&[1i32, 2])).unwrap();
        assert!(!uncommitted.is_committed());
        let execute = |options: Options| {
            let double = crate::compile_with_options::<_, ArrayType, ArrayType>(
                |input| input.clone() + input,
                vector_type.clone(),
                &session,
                options,
            )
            .unwrap();
            let wrapper = crate::compile::<_, ArrayType, (ArrayType, ArrayType)>(
                |input| (double.call(input.clone()).unwrap(), input),
                vector_type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let (doubled, forwarded) = wrapper.executable().execute(&session, uncommitted.clone()).unwrap();
            [(doubled, vec![2i32, 4]), (forwarded, vec![1i32, 2])]
                .into_iter()
                .map(|(output, values)| {
                    output.block_until_ready().unwrap();
                    assert_eq!(output.mesh(), &mesh);
                    assert_eq!(output.sharding(), &replicated);
                    assert_eq!(
                        output.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
                        ["[2]"]
                    );
                    assert_eq!(output.copy_to_host().unwrap().elements::<i32>().unwrap(), values);
                    output.is_committed()
                })
                .collect::<Vec<_>>()
        };

        // A call of a compiled function with a sharding override requests the placement of the boundary of that
        // function, like the shardings of a nested `jit` call in JAX, although a replicated override over an auto mesh
        // axis stages nothing. Every output of the calling function is therefore committed, including one that the
        // call does not produce.
        assert_eq!(execute(Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone()])), [true, true]);
        assert_eq!(execute(Options::new(mesh.clone()).with_output_shardings(vec![replicated.clone()])), [true, true]);

        // A call without overrides requests nothing, so the outputs keep the rule for the commitment of the inputs.
        assert_eq!(execute(Options::new(mesh.clone())), [false, false]);
    }

    #[test]
    fn test_session_execute_compiled_async_commits_outputs_on_multiple_devices() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let wide_mesh = two_device_mesh(client, MeshAxisType::Auto);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let narrow_mesh = DeviceMesh::new(logical_mesh, vec![wide_mesh.devices()[0]]).unwrap();
        let uncommitted = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert!(!uncommitted.is_committed());
        let execute = |mesh: &DeviceMesh| {
            let staged = crate::stage::<_, ArrayType, (ArrayType, ArrayType)>(
                |input: CompileTracer| (input.clone() + input.clone(), input),
                replicated_vector_type(mesh, 2),
                Options::new(mesh.clone()),
            )
            .unwrap()
            .into_inner();
            let compiled: CompiledFunction<Session, Type, (Type, Type)> =
                staged.lower(&session).unwrap().compile(&session).unwrap();
            let (first, second) = compiled.executable().execute(&session, Value::Array(uncommitted.clone())).unwrap();
            let (first, second) = (program_array(&first), program_array(&second));
            assert_eq!(first.mesh(), mesh);
            assert_eq!(second.mesh(), mesh);
            assert_eq!(first.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.0, 4.0]);
            assert_eq!(second.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0]);
            (first.is_committed(), second.is_committed())
        };

        // A function on uncommitted inputs that requests no placement commits every output when it is compiled for
        // more than one device, as JAX commits every result of a computation over more than one device, and it keeps
        // the rule for the commitment of its inputs when it is compiled for a single device.
        assert_eq!(execute(&wide_mesh), (true, true));
        assert_eq!(execute(&narrow_mesh), (false, false));
    }

    #[test]
    fn test_session_execute_compiled_async_ignores_the_commitment_of_captures() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let uncommitted =
            Array::from_host_buffer(&session, scalar_type.clone(), None, None, 1f32.to_ne_bytes().as_slice()).unwrap();
        let placement = ArrayPlacement::mesh(uncommitted.mesh().clone(), uncommitted.sharding().clone()).unwrap();
        let committed = uncommitted.to_placement(&session, placement).unwrap();
        assert!(!uncommitted.is_committed());
        assert!(committed.is_committed());
        assert_eq!(RetainedArray::new(&uncommitted), RetainedArray::new(&committed));
        let execute = |first: &Array, second: &Array| {
            let (first, second) = (RetainedArray::new(first), RetainedArray::new(second));
            let staged = crate::stage::<_, ArrayType, ArrayType>(
                move |input: CompileTracer| {
                    let context = input.value().context().clone();
                    let first = context.lift(Constant::Session(first)).unwrap();
                    let second = context.lift(Constant::Session(second)).unwrap();
                    let first = ValueProjection::<ArrayType>::into_projected(first).unwrap();
                    let second = ValueProjection::<ArrayType>::into_projected(second).unwrap();
                    input + first + second
                },
                scalar_type.clone(),
                Options::new(uncommitted.mesh().clone()),
            )
            .unwrap()
            .into_inner();
            let compiled: CompiledFunction<Session, Type, Type> =
                staged.lower(&session).unwrap().compile(&session).unwrap();
            let output = compiled.executable().execute(&session, Value::Array(uncommitted.clone())).unwrap();
            assert_eq!(program_array(&output).copy_to_host().unwrap().elements::<f32>().unwrap(), vec![3.0]);
            program_array(&output).is_committed()
        };

        // A capture never commits the outputs, whichever of two aliases of the same buffers the function captures
        // first, as JAX returns `jit(lambda x: x + c)(x)` uncommitted for an uncommitted `x` and a committed `c`, and
        // also after the second function reuses the executable of the first one from the cache.
        assert!(!execute(&uncommitted, &committed));
        assert!(!execute(&committed, &uncommitted));
        session.clear_cache();
        assert!(!execute(&committed, &uncommitted));
        assert!(!execute(&uncommitted, &committed));
        assert!(!execute(&uncommitted, &uncommitted));
    }

    #[test]
    fn test_session_execute_stateless_zero_identity_preserves_logical_type_and_canonical_value() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let input_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged =
            crate::stage::<_, ArrayType, ArrayType>(|input| input, input_type.clone(), Options::new(mesh.clone()))
                .unwrap()
                .into_inner();
        let compiled: CompiledFunction<Session, Type, Type> =
            staged.lower(&session).unwrap().compile(&session).unwrap();
        assert_eq!(compiled.compiled_program().output_types(), std::slice::from_ref(&input_type));

        let donated_staged: StagedFunction<Session, Type, Type> = crate::stage::<_, ArrayType, ArrayType>(
            |input| input,
            input_type.clone(),
            Options::new(mesh.clone()).with_donate(true),
        )
        .unwrap()
        .into_inner();
        let donated_lowered = donated_staged.lower(&session).unwrap();
        assert_eq!(donated_lowered.lowered_program().donation_flags(), &[false]);
        assert_eq!(
            session.compilation_key(compiled.lowered().lowered_program()).unwrap(),
            session.compilation_key(donated_lowered.lowered_program()).unwrap(),
        );

        // Boolean values retain physical `i1` arguments and results while the identity over the zero data type has an
        // empty physical signature. Their retained logical metadata also keeps their compilation identities distinct.
        let boolean_type = input_type.clone().with_data_type(DataType::Boolean);
        let boolean_staged: StagedFunction<Session, Type, Type> =
            crate::stage::<_, ArrayType, ArrayType>(|input| input, boolean_type, Options::new(mesh.clone()))
                .unwrap()
                .into_inner();
        let boolean_lowered = boolean_staged.lower(&session).unwrap();
        assert_ne!(
            session.compilation_key(compiled.lowered().lowered_program()).unwrap(),
            session.compilation_key(boolean_lowered.lowered_program()).unwrap(),
        );

        let input = Array::from_host_buffer(&session, input_type, mesh, None, []).unwrap();
        let output = compiled.executable().execute(&session, Value::Array(input)).unwrap();
        let Value::Array(output) = output else {
            panic!("array-only compiled function returned a first-class dimension");
        };
        assert_eq!(output.data_type(), DataType::Zero);
        assert!(output.addressable_shards().next().unwrap().buffer().is_none());
    }

    #[test]
    fn test_session_execute_stateless_projects_and_reconstructs_mixed_zero_signatures() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let value_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(sharding)
            .unwrap();
        let options = Options::new(mesh.clone()).with_donate((false, true));
        let staged = crate::stage::<_, (ArrayType, ArrayType), (ArrayType, ArrayType)>(
            |(value, zero)| (zero, value),
            (value_type.clone(), zero_type.clone()),
            options,
        )
        .unwrap()
        .into_inner();
        let lowered = staged.lower(&session).unwrap();
        assert!(lowered.lowered_program().stable_hlo().contains("func.func @main(%arg0: tensor<3xf32>"));
        let compiled: CompiledFunction<Session, (Type, Type), (Type, Type)> = lowered.compile(&session).unwrap();
        let value = Array::from_host_buffer(
            &session,
            value_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0]),
        )
        .unwrap();
        let zero = Array::from_host_buffer(&session, zero_type.clone(), mesh, None, []).unwrap();

        let (zero_output, value_output) =
            compiled.executable().execute(&session, (Value::Array(value), Value::Array(zero))).unwrap();
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
    fn test_session_execute_program_bounded_dynamic_program_on_multiple_cpu_devices() {
        let client = cpu_client(4);
        let mesh = domain_mesh(client, "x", 4);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 4);
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(4)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(2, Some(5)).unwrap());
        let depth = DimensionVariable::new("depth", DimensionBounds::new(4, Some(7)).unwrap());
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Static(4),
                Dimension::Dynamic(rows),
                Dimension::Dynamic(columns),
                Dimension::Dynamic(depth),
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
        let compiled: CompiledFunction<Session, Type, Type> =
            staged.lower(&session).unwrap().compile(&session).unwrap();
        let executable_options =
            compiled.compiled_program().compilation_options().executable_build_options.as_ref().unwrap();
        assert_eq!(executable_options.replica_count, 4);
        assert_eq!(executable_options.partition_count, 1);
        assert!(!executable_options.use_spmd_partitioning);
        assert!(!executable_options.use_shardy_partitioner);
        assert_eq!(compiled.compiled_program().plan().output_plans.as_ref().map(Vec::len), None);

        for shape in [[4usize, 2, 3, 5], [4, 3, 4, 6]] {
            let actual_type =
                ArrayType::new(DataType::F32, Shape::new(shape.into_iter().map(Dimension::Static).collect()))
                    .with_sharding(sharding.clone())
                    .unwrap();
            let values = (0..shape.iter().product()).map(|value| value as f32).collect::<Vec<_>>();
            let input = Array::from_host_buffer(
                &session,
                actual_type,
                mesh.clone(),
                None,
                values_to_bytes(values.as_slice()).as_slice(),
            )
            .unwrap();
            let output = compiled.executable().execute(&session, Value::Array(input)).unwrap();
            let Value::Array(output) = output else {
                panic!("array-only compiled function returned a first-class dimension");
            };
            output.block_until_ready().unwrap();

            assert_eq!(output.shape().as_slice(), shape.as_slice());
            assert_eq!(
                read_first_shard::<f32>(client, &output),
                values.iter().map(|value| value * 2.0).collect::<Vec<_>>()
            );
        }

        // Besides the compiled function, the session caches the padding program of the one input shape that is below
        // its bound.
        assert_eq!(session.cached_program_count(), 2);

        // Dynamic tensor partitioning remains an explicit backend limitation rather than silently changing a user's
        // requested placement to replication. Types record no placement over auto mesh axes, so that placement is an
        // input sharding override.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Dynamic(extent)]));
        let sharded = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |input| input.clone() + input,
            dynamic_type,
            Options::new(mesh).with_input_shardings(vec![sharded]),
        )
        .unwrap()
        .into_inner();
        let Err(error) = staged.lower(&session) else {
            panic!("bounded-dynamic sharded program unexpectedly lowered");
        };
        assert_eq!(
            error.to_string(),
            "invalid compilation options: bounded-dynamic multi-device programs currently require fully replicated \
             shardings because Shardy rejects dynamic tensors",
        );
    }

    #[test]
    fn test_session_execute_program_data_derived_prefix_slice_at_multiple_extents() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let input_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let mask_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(input_sharding.clone())
            .unwrap();
        let values_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(input_sharding)
            .unwrap();
        let count_variable = DimensionVariable::new("count", DimensionBounds::new(0, Some(5)).unwrap());

        // The mask determines a count on the device, the checked gateway turns that count into a bounded dimension, and
        // the slice returns that many leading values without a host readback.
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
            .add_instruction(DynamicSliceOperation::<Type>::from_rank(1), Vec::new(), vec![values, start, count], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let values = || f32_vector(&session, &mesh, &[10.0, 20.0, 30.0, 40.0]);
        let execute = |mask: &[bool]| {
            session.execute_program(&compiled, vec![boolean_vector(&session, &mesh, mask), values()]).unwrap()
        };

        let two = execute(&[true, false, true, false]);
        assert_eq!(two[0].shape(), StaticShape::new(vec![2]));
        assert_eq!(read_first_shard::<f32>(client, &two[0]), vec![10.0, 20.0]);
        let four = execute(&[true, true, true, true]);
        assert_eq!(four[0].shape(), StaticShape::new(vec![4]));
        assert_eq!(read_first_shard::<f32>(client, &four[0]), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_session_execute_program_data_derived_prefix_slice_with_duplicate_and_equal_outputs() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let input_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let mask_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(input_sharding.clone())
            .unwrap();
        let values_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(input_sharding)
            .unwrap();
        let count_variable = DimensionVariable::new("count", DimensionBounds::new(0, Some(5)).unwrap());

        // XLA merges bounded dynamic results that it proves equal and, on CPU, fills the duplicate result with a plain
        // copy that leaves its runtime size trailer at zero, while CUDA preserves the trailer. The program returns the
        // same prefix slice twice and then an equal slice built separately, which covers duplicate entry results and
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
                    DynamicSliceOperation::<Type>::from_rank(1),
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

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let values = || f32_vector(&session, &mesh, &[10.0, 20.0, 30.0, 40.0]);
        let execute = |mask: &[bool]| {
            session.execute_program(&compiled, vec![boolean_vector(&session, &mesh, mask), values()]).unwrap()
        };

        let two = execute(&[true, false, true, false]);
        assert_eq!(two.len(), 3);
        for output in &two {
            assert_eq!(output.shape(), StaticShape::new(vec![2]));
            assert_eq!(read_first_shard::<f32>(client, output), vec![10.0, 20.0]);
        }
        let four = execute(&[true, true, true, true]);
        assert_eq!(four.len(), 3);
        for output in &four {
            assert_eq!(output.shape(), StaticShape::new(vec![4]));
            assert_eq!(read_first_shard::<f32>(client, output), vec![10.0, 20.0, 30.0, 40.0]);
        }
    }

    #[test]
    fn test_session_prepare_execution_resources_validates_kernel_participants() {
        let client = execution_client();
        let mesh = domain_mesh(client, "device", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input = f32_vector(&session, &mesh, &[2.0, 5.0]);
        let mut builder = ProgramBuilder::new();
        let argument = builder.add_input(input.r#type().into_owned().into());
        let source = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![argument], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = session.lower_program(&source, 0, &Options::new(mesh.clone())).unwrap();
        let program = session.compile_program(&lowered).unwrap();
        // Attach the same validated marker carried by executables with native kernels. The identity program isolates
        // submission admission from any backend-specific kernel compiler or custom-call handler.
        let program = program.with_kernel_execution_facts(Some(
            crate::kernels::KernelExecutionFacts::from_target(session.target().unwrap(), &mesh)
                .unwrap()
                .configuration_key()
                .unwrap(),
        ));
        let execution = session.execute_compiled_async(&program, vec![input.clone()]).unwrap();
        execution.fence().block_until_ready().unwrap();
        assert_eq!(read_first_shard::<f32>(client, &execution.output()[0]), vec![2.0, 5.0]);
        let program = program.with_mesh(
            DeviceMesh::new(
                mesh.logical_mesh().clone(),
                vec![Device::new(mesh.devices()[0].id(), client.process_index().unwrap() + 1)],
            )
            .unwrap(),
        );
        // A topology change cannot reach resharding, donation, or PJRT execution, and the retained input stays
        // readable.
        assert!(matches!(
            session.prepare_execution_resources(&program),
            Err(error)
                if matches!(error.downcast_custom::<crate::kernels::KernelEmbeddingError>(),
                    Some(crate::kernels::KernelEmbeddingError::Invalid { message })
                        if message == "cross-process kernel execution requires a qualified collective ordering and \
                                       completion contract"),
        ));
        assert_eq!(read_first_shard::<f32>(client, &input), vec![2.0, 5.0]);
    }

    #[test]
    fn test_session_prepare_compiled_execution_preserves_erased_input_producer_completion() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let zero_type = ArrayType::new_static(DataType::Zero, [2]);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        // Exercise both the ordinary input projection and the branch that materializes bounded inputs. The failed
        // producer belongs to an erased input of the zero data type, so no physical input buffer can carry its failure.
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
            let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            assert_eq!(lowered.signature().input_mapping(), &[None, Some(0)]);
            let compiled = session.compile_program(&lowered).unwrap();
            let (event, promise) = client.event(()).unwrap();
            promise.set(Some(ryft_pjrt::Error::invalid_argument("erased producer failed"))).unwrap();
            let zero = Array::from_host_buffer(&session, zero_type.clone(), mesh.clone(), None, [])
                .unwrap()
                .with_execution_fence(ExecutionFence::new(vec![event]));
            let value = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [2]),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(&[1.0, 2.0]),
            )
            .unwrap();
            assert!(matches!(
                session.execute_compiled_async(&compiled, vec![zero, value]),
                Err(Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. }))
                    if message == "erased producer failed",
            ));
        }
    }

    #[test]
    fn test_session_prepare_compiled_execution_preserves_unreduced_shards_during_relayout() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2).with_unreduced_axes(["x"]).unwrap();
        let shape = [2, 2];
        let input_type = ArrayType::new_static(DataType::F32, shape).with_sharding(sharding.clone()).unwrap();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();

        // Supply buffers in a different layout from the executable's, with no layout declared on the input type.
        let expected = compiled.plan().physical_input_types[0].layout().unwrap();
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let layout = if expected == &row_major { column_major } else { row_major }.to_pjrt().unwrap();
        let summands = [[1.0f32, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]].map(|values| values_to_bytes(&values));
        let buffers = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .zip(&summands)
            .map(|(device, bytes)| {
                client
                    .buffer(
                        bytes.as_slice(),
                        BufferType::F32,
                        shape.map(|extent| extent as u64),
                        None,
                        device,
                        Some(layout.clone()),
                    )
                    .unwrap()
            })
            .collect();
        let input = Array::from_addressable_buffers(&session, input_type, mesh, None, buffers).unwrap();
        let prepared = session.prepare_compiled_execution(&compiled, vec![input], &[]).unwrap();

        let input = &prepared.inputs[0];
        assert_eq!(input.layout(), Some(expected));
        assert_eq!(input.sharding(), &sharding);
        for (shard, bytes) in input.addressable_shards().zip(summands) {
            assert_eq!(shard.copy_to_host_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn test_session_prepare_compiled_execution_preserves_auto_axis_shards_during_relayout() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::U8, [4, 4]);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let options = Options::new(mesh.clone()).with_input_shardings(vec![sharding.clone()]);
        let lowered = session.lower_program(&program, 0, &options).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();

        // Supply row shards in column-major order, which differs from the executable's, with no layout declared on the
        // input type, whose sharding does not record the split along the auto axis `x`. Each shard is converted on its
        // own, so the input keeps the split instead of being uploaded again as replicated.
        let expected = compiled.plan().physical_input_types[0].layout().unwrap();
        assert_eq!(expected, &Layout::Tiled(TiledLayout::new(vec![1, 0], vec![])));
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])).to_pjrt().unwrap();
        let buffers = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .zip([(0..8).collect::<Vec<u8>>(), (8..16).collect::<Vec<u8>>()])
            .map(|(device, bytes)| {
                client
                    .buffer(bytes.as_slice(), BufferType::U8, [2, 4], None, device, Some(column_major.clone()))
                    .unwrap()
            })
            .collect();
        let input =
            Array::from_addressable_buffers(&session, input_type, mesh, Some(sharding.clone()), buffers).unwrap();
        let prepared = session.prepare_compiled_execution(&compiled, vec![input], &[]).unwrap();

        let input = &prepared.inputs[0];
        input.block_until_ready().unwrap();
        assert_eq!(input.layout(), Some(expected));
        assert_eq!(input.sharding(), &sharding);
        assert_eq!(
            input.addressable_shards().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
            vec!["[2, 4]", "[2, 4]"],
        );
        assert_eq!(
            input.addressable_shards().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>(),
            vec![(0..8).collect::<Vec<u8>>(), (8..16).collect::<Vec<u8>>()],
        );
    }

    #[test]
    fn test_session_prepare_compiled_execution_relayouts_only_mismatching_shards_without_waiting() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::U8, [4, 4]);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let options = Options::new(mesh.clone()).with_input_shardings(vec![sharding.clone()]);
        let lowered = session.lower_program(&program, 0, &options).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let expected = compiled.plan().physical_input_types[0].layout().unwrap().clone();
        assert_eq!(expected, Layout::Tiled(TiledLayout::new(vec![1, 0], vec![])));

        // The first row shard is ready in column-major order, and the second one is still pending in the row-major
        // order of the executable. The source also carries the execution fence of its producer.
        let devices = client.addressable_devices().unwrap();
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])).to_pjrt().unwrap();
        let first = client
            .buffer(
                (0..8).collect::<Vec<u8>>().as_slice(),
                BufferType::U8,
                [2, 4],
                None,
                devices[0].clone(),
                Some(column_major),
            )
            .unwrap();
        let specification = BufferSpecification::new(BufferType::U8, [2u64, 4]);
        let manager = client
            .host_to_device_transfer_manager(vec![specification], devices[1].default_memory().unwrap())
            .unwrap();
        let second = manager.retrieve_buffer(0).unwrap();
        let (producer, producer_promise) = client.event(()).unwrap();
        let input =
            Array::from_addressable_buffers(&session, input_type, mesh, Some(sharding.clone()), vec![first, second])
                .unwrap()
                .with_execution_fence(ExecutionFence::new(vec![producer]));
        let second = Arc::clone(input.addressable_shards().nth(1).unwrap().buffer().unwrap());

        // Preparation returns before the pending shard has its data, so a caller that supplies that data only after
        // preparation returns can proceed. The pending shard already has the expected layout and is kept as it is.
        let (sender, receiver) = mpsc::channel();
        let preparing_session = session.clone();
        let preparing_compiled = compiled.clone();
        std::thread::spawn(move || {
            let prepared = preparing_session.prepare_compiled_execution(&preparing_compiled, vec![input], &[]);
            sender.send(prepared.map(|prepared| prepared.inputs.into_iter().next().unwrap())).unwrap();
        });
        let input = receiver.recv_timeout(Duration::from_secs(60)).unwrap().unwrap();
        assert_eq!(input.layout(), Some(&expected));
        assert_eq!(input.sharding(), &sharding);
        assert_eq!(input.r#type().memory(), Memory::Device);
        assert_eq!(
            input.addressable_shards().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
            vec!["[2, 4]", "[2, 4]"],
        );
        assert!(Arc::ptr_eq(input.addressable_shards().nth(1).unwrap().buffer().unwrap(), &second));
        assert_eq!(input.execution_fence().unwrap().is_ready(), Ok(false));

        // Once the pending shard and the producer complete, the converted input holds the values of both shards and
        // reports the failure of the producer of its source.
        manager
            .transfer_data(0, Arc::new((8..16).collect::<Vec<u8>>()), 0, true)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(
            input.addressable_shards().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>(),
            vec![(0..8).collect::<Vec<u8>>(), (8..16).collect::<Vec<u8>>()],
        );
        producer_promise.set(Some(ryft_pjrt::Error::aborted("producer failed"))).unwrap();
        assert!(matches!(
            input.block_until_ready(),
            Err(Error::Pjrt(ryft_pjrt::Error::Aborted { message, .. })) if message == "producer failed",
        ));
    }

    #[test]
    fn test_session_reshard_inputs_if_needed() {
        let client = cpu_client(4);
        let mesh = domain_mesh(client, "x", 4);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let device_values = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| {
                    let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                    values_from_bytes::<f32>(bytes.as_slice())
                })
                .collect::<Vec<_>>()
        };

        // An input that already has the expected sharding passes through.
        let input = f32_vector(&session, &mesh, &[1., 2., 3., 4.]);
        let inputs = session
            .reshard_inputs_if_needed(&mesh, std::slice::from_ref(&replicated), vec![input.clone()])
            .unwrap();
        assert_eq!(inputs, vec![input.clone()]);

        // An input on one device of the mesh is placed on every device by copying its shard, which compiles nothing.
        let single_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![mesh.devices()[0]],
        )
        .unwrap();
        let single_device_input = f32_vector(&session, &single_device_mesh, &[1., 2., 3., 4.]);
        let inputs = session
            .reshard_inputs_if_needed(
                &mesh,
                &[replicated.clone(), replicated],
                vec![input.clone(), single_device_input.clone()],
            )
            .unwrap();
        assert_eq!(inputs[0], input);
        assert_eq!(inputs[1].r#type(), input.r#type());
        assert_eq!(inputs[1].mesh(), &mesh);
        assert!(Arc::ptr_eq(
            inputs[1].shards()[0].buffer().unwrap(),
            single_device_input.shards()[0].buffer().unwrap(),
        ));
        assert_eq!(device_values(&inputs[1]), vec![vec![1., 2., 3., 4.]; 4]);
        assert_eq!(session.cached_program_count(), 0);

        // An input whose shards have to be sliced goes through a compiled reshard.
        let inputs = session.reshard_inputs_if_needed(&mesh, std::slice::from_ref(&sharded), vec![input]).unwrap();
        assert_eq!(inputs[0].sharding(), &sharded);
        assert_eq!(device_values(&inputs[0]), vec![vec![1.], vec![2.], vec![3.], vec![4.]]);
        assert_eq!(session.cached_program_count(), 1);

        // An input with the expected sharding on a mesh that lists the devices in another order holds each slice on
        // another device, so its shards are moved to the devices that the program expects them on.
        let reversed_mesh =
            DeviceMesh::new(mesh.logical_mesh().clone(), mesh.devices().iter().rev().copied().collect()).unwrap();
        let reversed_input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            reversed_mesh.clone(),
            Some(sharded.clone()),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        assert_eq!(reversed_input.sharding(), &sharded);
        let inputs = session
            .reshard_inputs_if_needed(&mesh, std::slice::from_ref(&sharded), vec![reversed_input])
            .unwrap();
        assert_eq!(inputs[0].mesh(), &mesh);
        assert_eq!(device_values(&inputs[0]), vec![vec![1.], vec![2.], vec![3.], vec![4.]]);
        assert_eq!(session.cached_program_count(), 1);
    }

    #[test]
    fn test_session_materialize_bounded_dynamic_inputs() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let declared_type = bounded_vector_type(Sharding::replicated(mesh.logical_mesh().clone(), 1));
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        // An input that is already at its bound is passed through with its storage, followed by its extent scalar.
        let exact_type = replicated_vector_type(&mesh, 8);
        let exact_values = values_to_bytes::<f32>(&[0.0; 8]);
        let exact =
            Array::from_host_buffer(&session, exact_type.clone(), mesh.clone(), None, exact_values.as_slice()).unwrap();
        let at_bound = session
            .materialize_bounded_dynamic_inputs(
                &signature,
                std::slice::from_ref(&declared_type),
                std::slice::from_ref(&exact_type),
                vec![exact.clone()],
            )
            .unwrap();
        assert_eq!(at_bound.len(), 2);
        assert_eq!(at_bound[0], exact);
        assert_eq!(at_bound[1].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![8]);

        // A smaller input is padded with zeros after its elements up to the bound, and its extent scalar records its
        // actual extent.
        let smaller_type = replicated_vector_type(&mesh, 4);
        let smaller_values = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let smaller =
            Array::from_host_buffer(&session, smaller_type.clone(), mesh, None, smaller_values.as_slice()).unwrap();
        let padded = session
            .materialize_bounded_dynamic_inputs(
                &signature,
                std::slice::from_ref(&declared_type),
                std::slice::from_ref(&smaller_type),
                vec![smaller.clone()],
            )
            .unwrap();
        assert_eq!(padded.len(), 2);
        assert_eq!(
            padded[0].copy_to_host().unwrap().elements::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(padded[1].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![4]);

        // The padded array and the extent scalar are owned by the result alone. The source array retains neither, so
        // the source can still donate its buffers, and dropping the result releases both.
        assert!(padded[0].has_unique_shard_buffers());
        assert!(smaller.has_unique_shard_buffers());
        let live_arrays = session.array_statistics().live_arrays;
        drop(padded);
        assert_eq!(session.array_statistics().live_arrays, live_arrays - 2);
    }

    #[test]
    fn test_session_materialize_bounded_dynamic_inputs_pads_every_bounded_input() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let declared_type = bounded_vector_type(Sharding::replicated(mesh.logical_mesh().clone(), 1));
        let left_type = replicated_vector_type(&mesh, 4);
        let right_type = replicated_vector_type(&mesh, 2);
        let left_values = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let right_values = values_to_bytes::<f32>(&[5.0, 6.0]);
        let left =
            Array::from_host_buffer(&session, left_type.clone(), mesh.clone(), None, left_values.as_slice()).unwrap();
        let right = Array::from_host_buffer(&session, right_type.clone(), mesh, None, right_values.as_slice()).unwrap();
        let declared_types = [declared_type.clone(), declared_type];
        let signature = ExecutableSignature::new(&declared_types, &[]);

        // The padded inputs come first, in input order, followed by one extent scalar per bounded dimension.
        let padded = session
            .materialize_bounded_dynamic_inputs(
                &signature,
                &declared_types,
                &[left_type, right_type],
                vec![left, right],
            )
            .unwrap();
        assert_eq!(padded.len(), 4);
        assert_eq!(
            padded[0].copy_to_host().unwrap().elements::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(
            padded[1].copy_to_host().unwrap().elements::<f32>().unwrap(),
            vec![5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(padded[2].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![4]);
        assert_eq!(padded[3].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![2]);
    }

    #[test]
    fn test_session_materialize_bounded_dynamic_inputs_pads_sharded_inputs() {
        let client = cpu_client(4);
        // Executable signatures record placement over explicit mesh axes only, so the sharded input uses one.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap(),
            client
                .addressable_devices()
                .unwrap()
                .iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect(),
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let declared_type = bounded_vector_type(sharding.clone());
        let actual_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let source_values = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let source =
            Array::from_host_buffer(&session, actual_type.clone(), mesh, None, source_values.as_slice()).unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        // The padding runs on the four devices and keeps the result sharded over them.
        let padded = session
            .materialize_bounded_dynamic_inputs(&signature, &[declared_type], &[actual_type], vec![source])
            .unwrap();
        assert_eq!(
            padded[0].copy_to_host().unwrap().elements::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(padded[0].shards().len(), 4);
        assert_eq!(padded[0].shards()[0].shape(), StaticShape::new(vec![2]));
    }

    #[test]
    fn test_session_materialize_bounded_dynamic_inputs_pads_every_axis() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let declared_type = ArrayType::new(
            DataType::I32,
            Shape::new(vec![
                DimensionVariable::new("rows", DimensionBounds::new(0, Some(4)).unwrap()).into(),
                DimensionVariable::new("columns", DimensionBounds::new(0, Some(5)).unwrap()).into(),
            ]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let actual_type = ArrayType::new_static(DataType::I32, [2, 2]).with_sharding(sharding).unwrap();
        let values = values_to_bytes::<i32>(&[1, 2, 3, 4]);
        let input = Array::from_host_buffer(&session, actual_type.clone(), mesh, None, values.as_slice()).unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        // The elements stay at the origin and zeros follow them along every axis, with one extent scalar per axis.
        let padded = session
            .materialize_bounded_dynamic_inputs(&signature, &[declared_type], &[actual_type], vec![input])
            .unwrap();
        assert_eq!(padded.len(), 3);
        assert_eq!(padded[0].shape(), StaticShape::new(vec![3, 4]));
        assert_eq!(
            padded[0].copy_to_host().unwrap().elements::<i32>().unwrap(),
            vec![1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(padded[1].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![2]);
        assert_eq!(padded[2].copy_to_host().unwrap().elements::<i32>().unwrap(), vec![2]);
    }

    #[test]
    fn test_session_materialize_zero_space_carriers() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        // The compiled module declares the `[3]` argument at its bounds, so a `[2]` runtime value below those bounds
        // must still produce a `[3]` carrier. The hidden extent scalar transports the logical size.
        let physical_zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let runtime_zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(sharding)
            .unwrap();
        let zero = Array::from_host_buffer(&session, runtime_zero_type, mesh, None, []).unwrap();

        let carriers = session.materialize_zero_space_carriers(&[physical_zero_type], vec![zero]).unwrap();

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
    fn test_session_materialize_zero_space_carriers_for_inputs_below_their_bounds() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let declared_type = ArrayType::new(
            DataType::Zero,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::non_negative(Some(4)).unwrap()).into()]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let signature = ExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);
        let actual_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(sharding)
            .unwrap();
        let zero = Array::from_host_buffer(&session, actual_type.clone(), mesh, None, []).unwrap();

        // The input of the zero data type below its bounds must bypass the padding, because it has no payload, and
        // rely on the carrier materialization to synthesize the predicate argument at its bounds.
        let inputs = session
            .materialize_bounded_dynamic_inputs(
                &signature,
                std::slice::from_ref(&declared_type),
                std::slice::from_ref(&actual_type),
                vec![zero.clone()],
            )
            .unwrap();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], zero);

        let physical_types = signature.physical_input_types(std::slice::from_ref(&declared_type));
        let carriers = session.materialize_zero_space_carriers(&physical_types, inputs).unwrap();
        assert_eq!(carriers[0].data_type(), DataType::Boolean);
        assert_eq!(carriers[0].shape().as_slice(), &[3]);
        assert_eq!(carriers[1].data_type(), DataType::I32);
    }

    #[test]
    fn test_session_execute_statefully_uses_compiled_input_types() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let mut builder = ProgramBuilder::new();
        let reference = builder.add_input(Type::Reference(ReferenceType::new(scalar_type.clone())));
        let update = builder.add_input(Type::Array(scalar_type));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program: FlatProgram = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let lowered = session.lower_program(&program, 1, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let bytes = compiled.serialize().unwrap().unwrap();
        let restored = session.deserialize_compiled_program(&bytes).unwrap().unwrap();

        for program in [&compiled, &restored] {
            let holder = ArrayReference::new(f32_scalar(&session, &mesh, 1.0));
            let reference = Value::Reference(crate::Reference::new(holder.clone(), session.clone()));
            let update = Value::Array(f32_scalar(&session, &mesh, 2.0));

            // Kind checks use the stored binding positions, including captures, before acquiring mutable state.
            let expected = format!(
                "runtime input type `{}` does not match declared type `{}`",
                update.r#type(),
                reference.r#type(),
            );
            assert!(matches!(
                session.execute_statefully(program, vec![update.clone(), update.clone()]).r#await(),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    CoreError::Program(ProgramError::InvalidArgument { message })
                        if *message == expected,
                ),
            ));
            let expected = format!(
                "runtime input type `{}` does not match declared type `{}`",
                reference.r#type(),
                update.r#type(),
            );
            assert!(matches!(
                session.execute_statefully(program, vec![reference.clone(), reference.clone()]).r#await(),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    CoreError::Program(ProgramError::InvalidArgument { message })
                        if *message == expected,
                ),
            ));
            assert!(matches!(
                session.execute_statefully(program, Vec::new()).r#await(),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    CoreError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
                ),
            ));
            assert_eq!(read_first_shard::<f32>(client, &holder.read().unwrap()), vec![1.0]);

            let outputs = session.execute_statefully(program, vec![reference, update]).r#await().unwrap();
            assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![3.0]);
            assert_eq!(read_first_shard::<f32>(client, &holder.read().unwrap()), vec![3.0]);
        }
    }

    #[test]
    fn test_session_execute_statefully_rejects_non_root_views() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 2);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled = compile_statefully::<_, Type, Type>(
            |reference| reference.read().map_err(Into::into),
            Type::Reference(ReferenceType::new(scalar_type)),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let root = ArrayReference::new(
            Array::from_host_buffer(&session, vector_type, mesh, None, values_to_bytes::<f32>(&[4.0, 9.0]).as_slice())
                .unwrap(),
        );
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) })
            .unwrap();
        assert!(matches!(
            compiled.execute_statefully_and_wait(&session, Value::Reference(Reference::new(view, session.clone()))),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidArgument { message: reason })
                    if *reason
                        == "external state input 0 must be a root reference handle that uses its allocation's stored \
                            type identities",
            ),
        ));
        assert_eq!(read_first_shard::<f32>(client, &root.read().unwrap()), vec![4.0, 9.0]);
    }

    #[test]
    fn test_session_execute_statefully_validates_holder_sharding() {
        let client = execution_client();
        // Types record placement over explicit mesh axes only, so the alternate static sharding uses an explicit axis.
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let shape = Shape::new(vec![Dimension::Static(2)]);
        let replicated_type = ArrayType::new(DataType::F32, shape.clone())
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let alternate_type = ArrayType::new(DataType::F32, shape)
            .with_sharding(Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let sharded = compile_statefully::<_, Type, Type>(
            |reference| reference.read().map_err(Into::into),
            Type::Reference(ReferenceType::new(alternate_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(
                &session,
                alternate_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(&[1.0, 2.0]).as_slice(),
            )
            .unwrap(),
        );
        let Value::Array(output) = sharded
            .execute_statefully_and_wait(&session, Value::Reference(Reference::new(reference.clone(), session.clone())))
            .unwrap()
        else {
            panic!("stateful read output must be an array")
        };
        assert_eq!(read_first_shard::<f32>(client, &output), vec![1.0, 2.0]);
        let compiled = compile_statefully::<_, (Type, Type), Type>(
            |(reference, update)| {
                reference.add_update(&update)?;
                reference.read().map_err(Into::into)
            },
            (Type::Reference(ReferenceType::new(replicated_type.clone())), Type::Array(replicated_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let update = Array::from_host_buffer(
            &session,
            replicated_type.clone(),
            mesh,
            None,
            values_to_bytes::<f32>(&[3.0, 4.0]).as_slice(),
        )
        .unwrap();
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &session,
                (Value::Reference(Reference::new(reference.clone(), session.clone())), Value::Array(update)),
            ),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidArgument { message: reason })
                    if *reason == format!(
                        "external state input 0 reference type `{alternate_type}` does not refine effective \
                         compiled state type `{replicated_type}`",
                    ),
            ),
        ));
        assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![1.0, 2.0]);
    }

    #[test]
    fn test_session_execute_statefully_updates_sharded_holder() {
        let client = cpu_client(2);
        // Types record placement over explicit mesh axes only, so the statically sharded state uses an explicit axis.
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let compiled = compile_statefully::<_, (Type, Type), Type>(
            |(reference, update)| {
                reference.add_update(&update)?;
                reference.read().map_err(Into::into)
            },
            (Type::Reference(ReferenceType::new(array_type.clone())), Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(
                &session,
                array_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
            )
            .unwrap(),
        );
        let update = Array::from_host_buffer(
            &session,
            array_type.clone(),
            mesh.clone(),
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
            panic!("stateful sharded output must be an array")
        };
        let expected = vec![11.0, 22.0, 33.0, 44.0];
        assert_eq!(read_sharded_f32_array(&output), expected);
        assert_eq!(read_sharded_f32_array(&reference.read().unwrap()), expected);

        let reversed_mesh =
            DeviceMesh::new(mesh.logical_mesh().clone(), mesh.devices().iter().rev().cloned().collect()).unwrap();
        let reversed_reference = ArrayReference::new(
            Array::from_host_buffer(
                &session,
                array_type.clone(),
                reversed_mesh,
                None,
                values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
            )
            .unwrap(),
        );
        let update = Array::from_host_buffer(
            &session,
            array_type,
            mesh,
            None,
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0, 40.0]).as_slice(),
        )
        .unwrap();
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &session,
                (
                    Value::Reference(Reference::new(reversed_reference.clone(), session.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidArgument { message: reason })
                    if *reason == "external state input 0 reference mesh does not match the compiled device mesh",
            ),
        ));
        assert_eq!(read_sharded_f32_array(&reversed_reference.read().unwrap()), vec![1.0, 2.0, 3.0, 4.0],);
    }

    #[test]
    fn test_session_execute_statefully_rejects_capture_aliasing_input() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, array_type, mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice()).unwrap(),
        );
        let compiled = compile_statefully_with_captures::<_, Type, Type>(
            |captures, public_reference| {
                let update = public_reference.read()?;
                captures[0].add_update(&update)?;
                captures[0].read().map_err(Into::into)
            },
            vec![Value::Reference(Reference::new(reference.clone(), session.clone()))],
            reference_type.clone(),
            &session,
            Options::new(mesh),
        )
        .unwrap();
        let error = compiled
            .execute_statefully_and_wait(&session, Value::Reference(Reference::new(reference.clone(), session.clone())))
            .err();
        assert!(
            matches!(
                &error,
                Some(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    CoreError::Program(ProgramError::InvalidArgument { message })
                        if *message == "capture 0 and input 0 bind the same reference allocation",
                ),
            ),
            "{error:?}",
        );
        assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![1.0]);
    }

    #[test]
    fn test_session_execute_statefully_commits_references_in_reverse_identity_order() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));
        let compiled = compile_statefully::<_, (Type, Type, Type), (Type, Type)>(
            |(first, second, update)| {
                first.add_update(&update)?;
                second.add_update(&update)?;
                Ok((first.read()?, second.read()?))
            },
            (reference_type.clone(), reference_type, Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let first = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let second = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let (lower_reference, higher_reference) =
            if first.id() < second.id() { (first, second) } else { (second, first) };
        let update =
            Array::from_host_buffer(&session, array_type, mesh, None, 2.0f32.to_ne_bytes().as_slice()).unwrap();

        let (higher_output, lower_output) = compiled
            .execute_statefully_and_wait(
                &session,
                (
                    Value::Reference(Reference::new(higher_reference.clone(), session.clone())),
                    Value::Reference(Reference::new(lower_reference.clone(), session.clone())),
                    Value::Array(update),
                ),
            )
            .unwrap();
        let Value::Array(higher_output) = higher_output else {
            panic!("first stateful public output must be an array")
        };
        let Value::Array(lower_output) = lower_output else { panic!("second stateful public output must be an array") };
        assert_eq!(read_first_shard::<f32>(client, &higher_output), vec![3.0]);
        assert_eq!(read_first_shard::<f32>(client, &lower_output), vec![3.0]);
        assert_eq!(read_first_shard::<f32>(client, &higher_reference.read().unwrap()), vec![3.0]);
        assert_eq!(read_first_shard::<f32>(client, &lower_reference.read().unwrap()), vec![3.0]);
    }

    #[test]
    fn test_session_execute_statefully_locks_shared_holders_in_identity_order() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));
        let compiled = compile_statefully::<_, (Type, Type, Type), ()>(
            |(first, second, update)| {
                first.add_update(&update)?;
                second.add_update(&update).map_err(Into::into)
            },
            (reference_type.clone(), reference_type, Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let first = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let second = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let (lower_reference, higher_reference) =
            if first.id() < second.id() { (first, second) } else { (second, first) };
        let higher_first_update =
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        let lower_first_update =
            Array::from_host_buffer(&session, array_type, mesh, None, 2.0f32.to_ne_bytes().as_slice()).unwrap();

        // The two calls present the same holders in opposite argument order and start together. Both must finish,
        // because internal identity ordering serializes their retained guard windows without creating a lock cycle.
        std::thread::scope(|scope| {
            let start = Arc::new(Barrier::new(3));
            let (finished, results) = mpsc::channel();
            for (first, second, update) in [
                (higher_reference.clone(), lower_reference.clone(), higher_first_update),
                (lower_reference.clone(), higher_reference.clone(), lower_first_update),
            ] {
                let start = Arc::clone(&start);
                let finished = finished.clone();
                let compiled = &compiled;
                let session = &session;
                scope.spawn(move || {
                    start.wait();
                    finished
                        .send(compiled.execute_statefully_and_wait(
                            session,
                            (
                                Value::Reference(Reference::new(first, session.clone())),
                                Value::Reference(Reference::new(second, session.clone())),
                                Value::Array(update),
                            ),
                        ))
                        .unwrap();
                });
            }
            drop(finished);
            start.wait();
            for _ in 0..2 {
                results.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
            }
        });
        assert_eq!(read_first_shard::<f32>(client, &lower_reference.read().unwrap()), vec![5.0]);
        assert_eq!(read_first_shard::<f32>(client, &higher_reference.read().unwrap()), vec![5.0]);
    }

    #[test]
    fn test_session_execute_statefully_failure_boundaries() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));
        let compiled = compile_statefully::<_, (Type, Type, Type, Type), Type>(
            |(first, second, read_only, update)| {
                let output = read_only.read()?;
                first.add_update(&update)?;
                second.add_update(&update)?;
                Ok(output)
            },
            (reference_type.clone(), reference_type.clone(), reference_type, Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference_new = |value: f32| {
            ArrayReference::new(
                Array::from_host_buffer(
                    &session,
                    array_type.clone(),
                    mesh.clone(),
                    None,
                    value.to_ne_bytes().as_slice(),
                )
                .unwrap(),
            )
        };

        // Failures during holder validation before execution leave every holder Ready and unchanged.
        let pre_handoff_first = reference_new(1.0);
        let pre_handoff_second = reference_new(10.0);
        let pre_handoff_read_only = reference_new(20.0);
        let wrong_update_type = replicated_scalar_type(&mesh, DataType::F64);
        let wrong_update =
            Array::from_host_buffer(&session, wrong_update_type, mesh.clone(), None, 2.0f64.to_ne_bytes().as_slice())
                .unwrap();
        let declared_update_type = <&ArrayType>::try_from(&compiled.executable().input_types()[3]).unwrap();
        let expected_error = format!(
            "runtime input type `{}` does not refine declared type `{declared_update_type}`",
            wrong_update.r#type(),
        );
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &session,
                (
                    Value::Reference(Reference::new(pre_handoff_first.clone(), session.clone())),
                    Value::Reference(Reference::new(pre_handoff_second.clone(), session.clone())),
                    Value::Reference(Reference::new(pre_handoff_read_only.clone(), session.clone())),
                    Value::Array(wrong_update),
                ),
            ),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidArgument { message })
                    if *message == expected_error,
            ),
        ));
        assert_eq!(read_first_shard::<f32>(client, &pre_handoff_first.read().unwrap()), vec![1.0]);
        assert_eq!(read_first_shard::<f32>(client, &pre_handoff_second.read().unwrap()), vec![10.0]);
        assert_eq!(read_first_shard::<f32>(client, &pre_handoff_read_only.read().unwrap()), vec![20.0]);

        // A fully prepared call that fails before PJRT submission publishes no lease or pending mutation.
        let pre_submission_first = reference_new(1.0);
        let pre_submission_second = reference_new(10.0);
        let pre_submission_read_only = reference_new(20.0);
        let update =
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        inject_stateful_failure(StatefulFailureInjection::BeforeSubmission);
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &session,
                (
                    Value::Reference(Reference::new(pre_submission_first.clone(), session.clone())),
                    Value::Reference(Reference::new(pre_submission_second.clone(), session.clone())),
                    Value::Reference(Reference::new(pre_submission_read_only.clone(), session.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Pjrt(ryft_pjrt::Error::Internal { message, .. }))
                if message == "injected failure before execution submission",
        ));
        assert_eq!(read_first_shard::<f32>(client, &pre_submission_first.read().unwrap()), vec![1.0]);
        assert_eq!(read_first_shard::<f32>(client, &pre_submission_second.read().unwrap()), vec![10.0]);
        assert_eq!(read_first_shard::<f32>(client, &pre_submission_read_only.read().unwrap()), vec![20.0]);

        // Once mutable buffers cross the execution handoff, every mutated holder is poisoned together while a
        // read-only peer remains Ready.
        let post_handoff_first = reference_new(1.0);
        let post_handoff_second = reference_new(10.0);
        let post_handoff_read_only = reference_new(20.0);
        let update =
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        inject_stateful_failure(StatefulFailureInjection::AfterHandoff);
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &session,
                (
                    Value::Reference(Reference::new(post_handoff_first.clone(), session.clone())),
                    Value::Reference(Reference::new(post_handoff_second.clone(), session.clone())),
                    Value::Reference(Reference::new(post_handoff_read_only.clone(), session.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Pjrt(ryft_pjrt::Error::Internal { message, .. }))
                if message == "injected failure after execution handoff",
        ));
        for reference in [&post_handoff_first, &post_handoff_second] {
            let Err(error) = reference.read() else { panic!("mutated holder must be poisoned after handoff") };
            assert_eq!(
                error.downcast_custom::<ReferenceError>(),
                Some(&ReferenceError::ExecutionPoisoned {
                    reason: "injected failure after execution handoff".to_string(),
                }),
            );
        }
        assert_eq!(read_first_shard::<f32>(client, &post_handoff_read_only.read().unwrap()), vec![20.0]);

        // Every ordinary error after publication but before the boundary that validates and then commits all holders
        // poisons the complete mutated batch with that exact error. The published read lease on a read-only peer
        // remains valid and releases normally when the submitted execution completes.
        let pre_commit_first = reference_new(1.0);
        let pre_commit_second = reference_new(10.0);
        let pre_commit_read_only = reference_new(20.0);
        let update =
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        inject_stateful_failure(StatefulFailureInjection::BeforeHiddenReplacementCommit);
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &session,
                (
                    Value::Reference(Reference::new(pre_commit_first.clone(), session.clone())),
                    Value::Reference(Reference::new(pre_commit_second.clone(), session.clone())),
                    Value::Reference(Reference::new(pre_commit_read_only.clone(), session.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Pjrt(ryft_pjrt::Error::Internal { message, .. }))
                if message == "injected failure before hidden state replacement commit",
        ));
        for reference in [&pre_commit_first, &pre_commit_second] {
            let Err(error) = reference.read() else { panic!("mutated reference must be poisoned before commit") };
            assert_eq!(
                error.downcast_custom::<ReferenceError>(),
                Some(&ReferenceError::ExecutionPoisoned {
                    reason: "injected failure before hidden state replacement commit".to_string(),
                }),
            );
        }
        assert_eq!(read_first_shard::<f32>(client, &pre_commit_read_only.read().unwrap()), vec![20.0]);

        // A failure confined to public reconstruction occurs only after every hidden final state replacement commits.
        let committed_first = reference_new(1.0);
        let committed_second = reference_new(10.0);
        let committed_read_only = reference_new(20.0);
        let update =
            Array::from_host_buffer(&session, array_type, mesh, None, 2.0f32.to_ne_bytes().as_slice()).unwrap();
        inject_stateful_failure(StatefulFailureInjection::BeforePublicReconstruction);
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &session,
                (
                    Value::Reference(Reference::new(committed_first.clone(), session.clone())),
                    Value::Reference(Reference::new(committed_second.clone(), session.clone())),
                    Value::Reference(Reference::new(committed_read_only.clone(), session.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Pjrt(ryft_pjrt::Error::Internal { message, .. }))
                if message == "injected failure before public output reconstruction",
        ));
        assert_eq!(read_first_shard::<f32>(client, &committed_first.read().unwrap()), vec![3.0]);
        assert_eq!(read_first_shard::<f32>(client, &committed_second.read().unwrap()), vec![12.0]);
        assert_eq!(read_first_shard::<f32>(client, &committed_read_only.read().unwrap()), vec![20.0]);
    }

    #[test]
    fn test_session_execute_statefully_validation_failure_leaves_peer_unchanged() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));
        let compiled = compile_statefully::<_, (Type, Type, Type), Type>(
            |(first, second, update)| {
                first.add_update(&update)?;
                second.add_update(&update)?;
                first.read().map_err(Into::into)
            },
            (reference_type.clone(), reference_type, Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let first = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let frozen = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 2.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        frozen.freeze().unwrap();
        let update =
            Array::from_host_buffer(&session, array_type, mesh, None, 3.0f32.to_ne_bytes().as_slice()).unwrap();
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &session,
                (
                    Value::Reference(Reference::new(first.clone(), session.clone())),
                    Value::Reference(Reference::new(frozen, session.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Reference(ReferenceError::Frozen),
            ),
        ));
        assert_eq!(read_first_shard::<f32>(client, &first.read().unwrap()), vec![1.0]);
    }

    #[test]
    fn test_session_execute_statefully_read_leases_overlap_and_block_mutation() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));
        let read = compile_statefully::<_, Type, Type>(
            |reference| reference.read().map_err(Into::into),
            reference_type.clone(),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let mutate = compile_statefully::<_, (Type, Type), ()>(
            |(reference, update)| reference.add_update(&update).map_err(Into::into),
            (reference_type, Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        // Both gates report into one labelled channel, so the assertions below pin which read lease the mutation
        // blocks on first instead of silently depending on the order the transaction awaits its leases.
        let (await_started, await_observed) = mpsc::channel();
        let first_gate = ControlledReferenceCompletion::with_await_notification("first read", await_started.clone());
        inject_stateful_completion(CoreReferenceCompletion::new(first_gate.clone()));
        let first_read =
            read.execute_statefully(&session, Value::Reference(Reference::new(reference.clone(), session.clone())));
        let second_gate = ControlledReferenceCompletion::with_await_notification("second read", await_started);
        inject_stateful_completion(CoreReferenceCompletion::new(second_gate.clone()));
        let second_read =
            read.execute_statefully(&session, Value::Reference(Reference::new(reference.clone(), session.clone())));

        std::thread::scope(|scope| {
            let (mutation_finished, mutation_result) = mpsc::channel();
            let mutation_reference = reference.clone();
            let session = &session;
            let mutate = &mutate;
            scope.spawn(move || {
                let update =
                    Array::from_host_buffer(session, array_type, mesh, None, 2.0f32.to_ne_bytes().as_slice()).unwrap();
                mutation_finished
                    .send(mutate.execute_statefully_and_wait(
                        session,
                        (Value::Reference(Reference::new(mutation_reference, session.clone())), Value::Array(update)),
                    ))
                    .unwrap();
            });

            assert_eq!(await_observed.recv_timeout(Duration::from_secs(5)).unwrap(), "first read");
            assert!(matches!(mutation_result.try_recv(), Err(mpsc::TryRecvError::Empty)));
            first_gate.complete(Ok(()));
            assert_eq!(await_observed.recv_timeout(Duration::from_secs(5)).unwrap(), "second read");
            assert!(matches!(mutation_result.try_recv(), Err(mpsc::TryRecvError::Empty)));
            second_gate.complete(Ok(()));
            mutation_result.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        });
        first_read.r#await().unwrap();
        second_read.r#await().unwrap();
        assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![3.0]);
    }

    #[test]
    fn test_session_execute_statefully_without_public_outputs_waits_for_commit() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled = compile_statefully::<_, (Type, Type), ()>(
            |(reference, update)| reference.add_update(&update).map_err(Into::into),
            (Type::Reference(ReferenceType::new(array_type.clone())), Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let update =
            Array::from_host_buffer(&session, array_type, mesh, None, 2.0f32.to_ne_bytes().as_slice()).unwrap();
        compiled
            .execute_statefully(
                &session,
                (Value::Reference(Reference::new(reference.clone(), session.clone())), Value::Array(update)),
            )
            .r#await()
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![3.0]);
    }

    #[test]
    fn test_session_execute_statefully_chains_unawaited_mutations_and_preserves_prior_failure() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled = compile_statefully::<_, (Type, Type), ()>(
            |(reference, update)| reference.add_update(&update).map_err(Into::into),
            (Type::Reference(ReferenceType::new(array_type.clone())), Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let predecessor = ControlledReferenceCompletion::new();
        inject_stateful_completion(CoreReferenceCompletion::new(predecessor.clone()));
        let first_update =
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        let first = compiled.execute_statefully(
            &session,
            (Value::Reference(Reference::new(reference.clone(), session.clone())), Value::Array(first_update)),
        );
        let second_update =
            Array::from_host_buffer(&session, array_type, mesh, None, 2.0f32.to_ne_bytes().as_slice()).unwrap();
        let second = compiled.execute_statefully(
            &session,
            (Value::Reference(Reference::new(reference.clone(), session.clone())), Value::Array(second_update)),
        );

        predecessor.complete(Err(Arc::from("injected predecessor failure")));
        for execution in [first, second] {
            assert!(matches!(
                execution.r#await(),
                Err(Error::Execution(ExecutionError::AsynchronousFailure { reason }))
                    if reason.as_ref() == "injected predecessor failure",
            ));
        }
        let error = reference.read().unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ExecutionPoisoned { reason: "injected predecessor failure".to_string() }),
        );
    }

    #[test]
    fn test_session_execute_statefully_drop_reconciles_pending_holder_on_access() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled = compile_statefully::<_, (Type, Type), ()>(
            |(reference, update)| reference.add_update(&update).map_err(Into::into),
            (Type::Reference(ReferenceType::new(array_type.clone())), Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let completion = ControlledReferenceCompletion::new();
        inject_stateful_completion(CoreReferenceCompletion::new(completion.clone()));
        let update =
            Array::from_host_buffer(&session, array_type, mesh, None, 2.0f32.to_ne_bytes().as_slice()).unwrap();
        drop(compiled.execute_statefully(
            &session,
            (Value::Reference(Reference::new(reference.clone(), session.clone())), Value::Array(update)),
        ));

        completion.complete(Ok(()));
        assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![3.0]);
    }
}
