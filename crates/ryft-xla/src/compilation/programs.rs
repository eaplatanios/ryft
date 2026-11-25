//! Lowered and compiled program artifacts of the XLA [`Session`].
//!
//! Lowering a staged program produces a [`LoweredProgram`], which holds the StableHLO module, the effective PJRT
//! compilation options, and the invocation contract that maps logical values to physical executable slots. Compiling
//! it produces a [`CompiledProgram`], which retains the loaded PJRT executable together with the same invocation
//! contract and a reusable execution plan. Both artifacts implement [`TypedProgram`] for their public logical
//! signature, and replacement executables must preserve the complete invocation contract of the program they replace.

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use ryft_core::{ArrayType, DeviceMesh, ExternalReferenceBinding, ProgramError, Sharding, TypedProgram};
use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions};
use ryft_pjrt::{ExecutionFence, LoadedExecutable};

use crate::arrays::metadata::ArrayOutputPlan;
use crate::compilation::CompilationError;
use crate::compilation::analysis::{CompilationAnalysis, OptimizedProgram};
use crate::compilation::domains::declared_input_sharding;
use crate::compilation::keys::{COMPILER_IDENTITY, canonical_boundary_types};
use crate::compilation::options::FeedbackDirectedProfile;
use crate::execution::{ExecutionPlan, validate_input_reduction};
use crate::lowering::LoweredModule;
use crate::lowering::signature::ExecutableSignature;
use crate::{Array, Error, RetainedArray, RuntimeRequirement, Session, Target, Type};

// TODO(eaplatanios): Review this module.

/// A hidden argument that binds an original IR constant to the array supplied to the executable.
#[derive(Clone, Debug)]
pub(crate) struct SessionConstantBinding {
    /// Constant that defines the declared type, identity, and placement behavior of the capture in the program.
    source: RetainedArray,

    /// Retained array at the placement where the executable receives this argument.
    argument: RetainedArray,
}

impl SessionConstantBinding {
    /// Binds `source` to the retained array `argument` that supplies its executable buffers.
    pub(super) fn new(source: RetainedArray, argument: RetainedArray) -> Self {
        Self { source, argument }
    }

    /// Returns the original constant as it appears in the IR.
    pub(crate) fn source(&self) -> &RetainedArray {
        &self.source
    }

    /// Returns the array that supplies the executable buffers.
    pub(super) fn argument(&self) -> &RetainedArray {
        &self.argument
    }
}

/// How an invocation resolves the ownership of its fixed execution mesh.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutputDeviceAssignment {
    /// The staged function names its devices, which every process shares.
    #[default]
    Fixed,

    /// The applying session and its inputs determine whether the execution mesh is assigned locally.
    FromCaller,
}

/// The StableHLO module and invocation contract shared by native compilation and remote eager dispatch.
#[derive(Clone, Debug)]
pub(super) struct ProgramLowering {
    /// Textual StableHLO and Shardy module accepted by PJRT.
    stable_hlo: Arc<str>,

    /// Invocation contract prior to native compilation.
    invocation: InvocationMetadata,

    /// Binding for each session constant of the program in hidden argument order, following the staged capture table.
    session_constants: Arc<[SessionConstantBinding]>,
}

impl ProgramLowering {
    /// Combines the lowered module with the logical and physical invocation contract that produced it.
    ///
    /// # Parameters
    ///
    ///   - `module`: Lowered module, which provides the StableHLO text, the executable signature, the runtime
    ///     requirements, and whether executions need the runtime assertion handler.
    ///   - `compilation_options`: Exact options that PJRT compiles the module with.
    ///   - `input_types`: Effective flat input types, starting with the lowered captures.
    ///   - `output_types`: Flat output types, including the hidden final reference states.
    ///   - `output_count`: Number of public outputs.
    ///   - `reference_states`: External reference states of the program.
    ///   - `donation_flags`: Donation declaration of each public input.
    ///   - `capture_count`: Number of leading inputs that are lowered captures.
    ///   - `session_constants`: Bindings of session constants to the arrays that execution passes as captures.
    ///   - `expected_argument_shardings`: Sharding that each physical argument must have at the call boundary.
    ///   - `output_shardings`: Declared sharding of each output.
    ///   - `open_outputs`: Whether the compiler chooses the placement of each output over auto mesh axes.
    ///   - `committed_outputs`: Whether executions commit each output whatever the commitment of their inputs.
    ///   - `output_device_assignment`: Whether this invocation takes output assignment from the caller.
    ///   - `mesh`: Device mesh that the program is compiled for and executes on.
    ///   - `kernel_execution_facts`: Configuration key of the live execution facts when the module embeds kernels.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        module: LoweredModule,
        compilation_options: CompilationOptions,
        input_types: Arc<[ArrayType]>,
        output_types: Arc<[ArrayType]>,
        output_count: usize,
        reference_states: Arc<[ExternalReferenceBinding]>,
        donation_flags: Arc<[bool]>,
        capture_count: usize,
        session_constants: Arc<[SessionConstantBinding]>,
        expected_argument_shardings: Arc<[Sharding]>,
        output_shardings: Arc<[Sharding]>,
        open_outputs: Arc<[bool]>,
        committed_outputs: Arc<[bool]>,
        output_device_assignment: OutputDeviceAssignment,
        mesh: DeviceMesh,
        kernel_execution_facts: Option<Vec<u8>>,
    ) -> Self {
        let (stable_hlo, signature, requires_assertion_handler, runtime_requirements) = module.into_parts();
        Self {
            stable_hlo: stable_hlo.into(),
            invocation: InvocationMetadata {
                compilation_options,
                output_types,
                output_count,
                reference_states,
                input_types,
                signature,
                runtime_requirements,
                requires_assertion_handler,
                donation_flags,
                capture_count,
                expected_argument_shardings,
                output_shardings,
                open_outputs: Some(open_outputs),
                committed_outputs,
                output_device_assignment,
                mesh,
                kernel_execution_facts,
            },
            session_constants,
        }
    }

    /// Returns the session constant bindings after the staged capture table, in hidden argument order.
    pub(super) fn session_constants(&self) -> &[SessionConstantBinding] {
        &self.session_constants
    }
}

/// Plans a pure, statically shaped eager result on a process that owns no device of its singleton mesh.
pub(crate) struct RemoteEagerProgram {
    /// Expected runtime sharding for every logical input, with `None` for unread inputs.
    input_shardings: Vec<Option<Sharding>>,

    /// Number of leading arguments sourced from session constants.
    capture_count: usize,

    /// Result plans in logical output order, with no local buffer positions.
    outputs: Vec<ArrayOutputPlan>,

    /// Whether each result's lowering commits it independently of its inputs.
    committed_outputs: Arc<[bool]>,
}

impl RemoteEagerProgram {
    /// Builds a description using the same invocation contract that the owner compiles.
    pub(super) fn new(lowering: &ProgramLowering) -> Result<Self, Error> {
        let program = &lowering.invocation;
        let signature = &program.signature;
        let output_types = &program.output_types[..program.output_count];
        if signature.has_effects() || output_types.iter().any(|r#type| r#type.static_shape().is_none()) {
            return Err(ProgramError::UnsupportedOperation {
                message: "eager operations on a remote singleton require static outputs and no effects".to_string(),
            }
            .into());
        }
        // Erased zero arrays have a sharding and can carry a failed producer. Only unread arguments have no dependency.
        let input_shardings = signature
            .input_mapping()
            .iter()
            .zip(program.input_types.iter())
            .enumerate()
            .map(|(index, (physical_index, input_type))| {
                if signature.is_unread_input(index) {
                    return None;
                }
                Some(match physical_index {
                    Some(index) => program.expected_argument_shardings[*index].clone(),
                    None => declared_input_sharding(input_type, program.mesh.logical_mesh()),
                })
            })
            .collect();
        let outputs = output_types
            .iter()
            .zip(program.output_shardings.iter())
            .map(|(r#type, sharding)| ArrayOutputPlan::new(r#type.clone(), Some(sharding.clone()), &program.mesh, &[]))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            input_shardings,
            capture_count: program.capture_count,
            outputs,
            committed_outputs: Arc::clone(&program.committed_outputs),
        })
    }

    /// Returns the explicit commitment flags of the lowering.
    fn committed_outputs(&self) -> &[bool] {
        &self.committed_outputs
    }

    /// Reconstructs results with no local buffers, preserving producer dependencies of the admitted inputs.
    pub(crate) fn execute(&self, session: &Session, inputs: &[Array]) -> Result<Vec<Array>, Error> {
        debug_assert_eq!(inputs.len(), self.input_shardings.len());
        let mut fences = Vec::new();
        for (index, (input, sharding)) in inputs.iter().zip(&self.input_shardings).enumerate() {
            if let Some(sharding) = sharding {
                validate_input_reduction(input, sharding, index, self.capture_count)?;
                fences.extend(input.execution_fence().cloned());
            }
        }
        let fence = ExecutionFence::join(fences);
        Ok(self
            .outputs
            .iter()
            .map(|plan| Array::from_execution_output(session, plan, Vec::new(), fence.clone()))
            .collect())
    }
}

/// How this process participates in an eager operation.
#[derive(Clone)]
pub(crate) enum EagerDispatchProgram {
    /// This process loads an executable and submits it to its local devices.
    Local(Arc<CompiledProgram>),

    /// This process reconstructs remote results of a singleton without submitting device work.
    Remote(Arc<RemoteEagerProgram>),
}

impl EagerDispatchProgram {
    /// Returns the outputs committed by the lowering independently of the actual inputs.
    pub(crate) fn committed_outputs(&self) -> &[bool] {
        match self {
            Self::Local(program) => program.committed_outputs(),
            Self::Remote(program) => program.committed_outputs(),
        }
    }
}

/// StableHLO lowering and metadata of the call boundary produced before PJRT compilation.
///
/// A [`Session`] lowers a staged program into this artifact with its effective compilation
/// options and the mapping between logical values and physical executable slots. PJRT compilation produces a
/// [`CompiledProgram`] retaining the same invocation contract.
///
/// This flat artifact implements [`TypedProgram`] for its logical signature. Typed function handles in
/// [`functions`](crate::compilation::functions) additionally retain the Rust input and output container structures.
/// Hidden final reference states remain in the internal metadata but are excluded from the public output types.
#[derive(Clone, Debug)]
pub struct LoweredProgram {
    /// Lowered module and invocation contract.
    lowering: ProgramLowering,

    /// PJRT platform name of the client against which this program was lowered.
    platform_name: Arc<str>,

    /// PJRT platform version of the client against which this program was lowered.
    platform_version: Arc<str>,

    /// Device kinds in the exact mesh order used by this lowering.
    device_kinds: Arc<[String]>,

    /// Exact Ryft, OpenXLA, and JAX build identity used by this lowering.
    compiler_identity: Arc<str>,

    /// XLA flags of the process in effect when this program was lowered.
    xla_flags: Arc<str>,
}

impl LoweredProgram {
    /// Adds the concrete target identity needed to compile this lowering and reuse its executable.
    pub(super) fn new(lowering: ProgramLowering, target: &Target) -> Result<Self, Error> {
        let device_kinds = target.mesh_device_kinds(&lowering.invocation.mesh)?;
        Ok(Self {
            lowering,
            platform_name: target.platform_name().into(),
            platform_version: target.platform_version().into(),
            device_kinds: device_kinds.into(),
            compiler_identity: COMPILER_IDENTITY.as_str().into(),
            xla_flags: std::env::var("XLA_FLAGS").unwrap_or_default().into(),
        })
    }

    /// Returns this lowering with an opaque feedback-directed profile installed in its exact PJRT compilation
    /// options.
    ///
    /// The StableHLO and runtime signature are unchanged. Because canonical compilation identity includes the
    /// complete PJRT options, the returned lowering receives a distinct in-memory, persistent, and distributed cache
    /// key from the baseline executable.
    pub fn with_feedback_directed_profile(mut self, profile: FeedbackDirectedProfile) -> Self {
        self.lowering
            .invocation
            .compilation_options
            .executable_build_options
            .get_or_insert_with(ExecutableCompilationOptions::default)
            .fdo_profile = profile.bytes().to_vec();
        self.lowering.invocation.compilation_options.profile_version = profile.version();
        self
    }

    /// Returns this program with `stable_hlo` in place of its lowered module, for tests that compile a module which
    /// lowering itself would never produce.
    #[cfg(test)]
    pub(crate) fn with_stable_hlo(mut self, stable_hlo: impl Into<Arc<str>>) -> Self {
        self.lowering.stable_hlo = stable_hlo.into();
        self
    }

    /// Returns this program with `compilation_options` in place of its effective compilation options, for tests
    /// that force a compilation mode which lowering would not select.
    #[cfg(test)]
    pub(crate) fn with_compilation_options(mut self, compilation_options: CompilationOptions) -> Self {
        self.lowering.invocation.compilation_options = compilation_options;
        self
    }

    /// Returns the textual StableHLO and Shardy module.
    #[inline]
    pub fn stable_hlo(&self) -> &str {
        &self.lowering.stable_hlo
    }

    /// Returns the effective public logical flat output types, including leaves of zero size erased from the
    /// executable ABI but excluding the hidden final state output suffix appended by reference discharge.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.lowering.invocation.output_types[..self.lowering.invocation.output_count]
    }

    /// Returns every logical flat output type (i.e., the public outputs that [`Self::output_types`] returns followed by
    /// the hidden final state outputs appended by reference discharge).
    #[inline]
    pub(super) fn all_output_types(&self) -> &[ArrayType] {
        &self.lowering.invocation.output_types
    }

    /// Returns the runtime contracts that the invoking session must provide before compilation or execution.
    #[inline]
    pub fn runtime_requirements(&self) -> &[RuntimeRequirement] {
        &self.lowering.invocation.runtime_requirements
    }

    /// Returns the device mesh this lowering targets.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.lowering.invocation.mesh
    }

    /// Returns the bindings of the session constants after the staged capture table, in hidden argument order.
    #[inline]
    pub(crate) fn session_constants(&self) -> &[SessionConstantBinding] {
        self.lowering.session_constants()
    }

    /// Returns the effective PJRT compilation options that this program is compiled with.
    #[inline]
    pub fn compilation_options(&self) -> &CompilationOptions {
        &self.lowering.invocation.compilation_options
    }

    /// Returns the default donation flag of each public logical input, excluding captures.
    #[inline]
    pub fn donation_flags(&self) -> &[bool] {
        &self.lowering.invocation.donation_flags
    }

    /// Returns the name of the PJRT platform that this program was lowered for.
    #[inline]
    pub fn platform_name(&self) -> &str {
        &self.platform_name
    }

    /// Returns the version of the PJRT platform that this program was lowered for.
    #[inline]
    pub fn platform_version(&self) -> &str {
        &self.platform_version
    }

    /// Returns the PJRT device kinds of the mesh devices in mesh order.
    #[inline]
    pub fn device_kinds(&self) -> &[String] {
        &self.device_kinds
    }

    /// Returns the mapping between the logical boundary of the program and the physical boundary of its executable.
    pub(crate) fn signature(&self) -> &ExecutableSignature {
        &self.lowering.invocation.signature
    }

    /// Returns the number of leading logical outputs exposed through the structured public function result.
    #[inline]
    pub(super) fn output_count(&self) -> usize {
        self.lowering.invocation.output_count
    }

    /// Returns the logical binding recipes of external reference states in canonical source order.  Flat input
    /// positions are derived from each binding's source and [`Self::capture_count`].
    #[inline]
    pub(super) fn reference_states(&self) -> &[ExternalReferenceBinding] {
        &self.lowering.invocation.reference_states
    }

    /// Returns the effective logical input types, including captures first.
    #[inline]
    pub(super) fn input_types(&self) -> &[ArrayType] {
        &self.lowering.invocation.input_types
    }

    /// Returns whether execution requires Ryft's host runtime assertion handler.
    #[inline]
    pub(super) fn requires_assertion_handler(&self) -> bool {
        self.lowering.invocation.requires_assertion_handler
    }

    /// Returns the number of leading logical inputs supplied by the staged capture table, followed by the hidden
    /// arguments that carry [`Self::session_constants`].
    #[inline]
    pub(super) fn capture_count(&self) -> usize {
        self.lowering.invocation.capture_count
    }

    /// Returns the effective sharding expected for every physical executable argument, including materialized captures
    /// first.
    #[inline]
    pub(super) fn expected_argument_shardings(&self) -> &[Sharding] {
        &self.lowering.invocation.expected_argument_shardings
    }

    /// Returns the runtime placement of every logical output over all mesh axes, including the auto mesh axes that
    /// output types never record. It also covers the hidden final reference states after the public result. The
    /// placement of an output flagged in [`Self::open_outputs`] is only the placement that its type records, and
    /// compilation resolves its placement over auto mesh axes from the executable.
    #[inline]
    pub(super) fn output_shardings(&self) -> &[Sharding] {
        &self.lowering.invocation.output_shardings
    }

    /// Returns whether lowering left each logical output open over auto mesh axes, so that the compiler chooses its
    /// placement over them, as JAX does for outputs without `out_shardings`.
    #[inline]
    pub(super) fn open_outputs(&self) -> &[bool] {
        self.lowering.invocation.open_outputs.as_deref().unwrap()
    }

    /// Returns whether executions of this lowering commit each logical output to its devices whatever the commitment
    /// of their inputs (refer to [`CompiledProgram::committed_outputs`]).
    #[inline]
    pub(super) fn committed_outputs(&self) -> &[bool] {
        &self.lowering.invocation.committed_outputs
    }

    /// Returns the exact live execution facts for selected kernels, absent for ordinary executables.
    #[inline]
    pub(crate) fn kernel_execution_facts(&self) -> Option<&[u8]> {
        self.lowering.invocation.kernel_execution_facts.as_deref()
    }

    /// Returns the invocation contract for validating cached and replacement executables.
    pub(super) fn invocation(&self) -> &InvocationMetadata {
        &self.lowering.invocation
    }

    /// Returns the exact Ryft, OpenXLA, and JAX build identity used by this lowering.
    #[inline]
    pub(super) fn compiler_identity(&self) -> &str {
        &self.compiler_identity
    }

    /// Returns the XLA flags of the process in effect when this program was lowered.
    #[inline]
    pub(super) fn xla_flags(&self) -> &str {
        &self.xla_flags
    }
}

impl TypedProgram for LoweredProgram {
    type Type = Type;

    // Core reads the public logical signature in the type universe of the domain, which excludes the hidden
    // final state output suffix appended by reference discharge, like the inherent `output_types` function.
    fn input_count(&self) -> usize {
        self.lowering.invocation.input_types.len()
    }

    fn input_types(&self) -> Cow<'_, [Type]> {
        Cow::Owned(self.lowering.invocation.input_types.iter().cloned().map(Into::into).collect())
    }

    fn output_count(&self) -> usize {
        self.lowering.invocation.output_count
    }

    fn output_types(&self) -> Cow<'_, [Type]> {
        Cow::Owned(self.output_types().iter().cloned().map(Into::into).collect())
    }
}

/// Loaded PJRT executable plus the metadata required to invoke it safely.
///
/// Compilation retains the [`LoweredProgram`]'s logical inputs, outputs, captures, reference bindings, and physical
/// boundary mapping alongside a reusable execution plan. This flat artifact implements [`TypedProgram`] for its
/// logical signature, excluding hidden final reference states from public output types. Typed function handles in
/// [`functions`](crate::compilation::functions) retain the Rust container structures separately. Reused and
/// replacement executables must preserve the complete invocation contract before execution.
///
/// The invoking [`Session`] validates inputs, resolves placement and runtime resources,
/// materializes bounded dynamic arguments, and reconstructs logical outputs. Ordinary calls carry completion through
/// their outputs. Stateful calls coordinate external references through [`ryft_core::StatefulExecution`], publishing
/// mutations only after submission and tracking completion of the whole invocation.
#[derive(Clone)]
pub struct CompiledProgram {
    /// Loaded PJRT executable, shared by cloned program handles.
    executable: Arc<LoadedExecutable<'static>>,

    /// Invocation contract with output shardings resolved by the executable.
    invocation: InvocationMetadata,

    /// PJRT platform name recorded at compilation.
    platform_name: Arc<str>,

    /// PJRT platform version recorded at compilation.
    platform_version: Arc<str>,

    /// Device kinds in the exact mesh order used for compilation.
    device_kinds: Arc<[String]>,

    /// Exact Ryft, OpenXLA, and JAX build identity used for compilation.
    compiler_identity: Arc<str>,

    /// XLA flags of the process recorded at compilation.
    xla_flags: Arc<str>,

    /// Wall-clock duration of the PJRT compilation request, if retained by the artifact.
    compilation_duration: Option<Duration>,

    /// Shared lazy analysis result, retaining both successful queries and their original failures.
    analysis: Arc<OnceLock<Result<CompilationAnalysis, Error>>>,

    /// Execution metadata that does not depend on runtime values, computed once when the program is compiled or
    /// loaded.
    plan: Arc<ExecutionPlan>,
}

impl CompiledProgram {
    /// Creates the compiled program of `program` from the `executable` that the backend compiled for it. The
    /// invocation contract is shared with `program`.
    ///
    /// # Parameters
    ///
    ///   - `program`: Lowered program that `executable` was compiled from.
    ///   - `executable`: Loaded PJRT executable.
    ///   - `output_shardings`: Resolved sharding of each output, including the placements that the compiler chose.
    ///   - `plan`: Execution plan derived from `executable` and the invocation contract.
    ///   - `compilation_duration`: Time that the backend compilation took.
    pub(super) fn new(
        program: &LoweredProgram,
        executable: LoadedExecutable<'static>,
        output_shardings: Arc<[Sharding]>,
        plan: ExecutionPlan,
        compilation_duration: Duration,
    ) -> Self {
        Self {
            executable: Arc::new(executable),
            invocation: InvocationMetadata {
                output_shardings,
                open_outputs: None,
                ..program.lowering.invocation.clone()
            },
            platform_name: Arc::clone(&program.platform_name),
            platform_version: Arc::clone(&program.platform_version),
            device_kinds: Arc::clone(&program.device_kinds),
            compiler_identity: Arc::clone(&program.compiler_identity),
            xla_flags: Arc::clone(&program.xla_flags),
            compilation_duration: Some(compilation_duration),
            analysis: Arc::new(OnceLock::new()),
            plan: Arc::new(plan),
        }
    }

    /// Creates a compiled program from an `executable` that was loaded from its serialized form and from the
    /// invocation contract that was persisted with it, including the outputs that its executions always commit, which
    /// are part of its compilation key (refer to [`Self::committed_outputs`]).
    ///
    /// # Parameters
    ///
    ///   - `executable`: Loaded PJRT executable.
    ///   - `input_types`: Flat input types, starting with the lowered captures.
    ///   - `output_types`: Flat output types, including the hidden final reference states.
    ///   - `output_count`: Number of public outputs.
    ///   - `reference_states`: External reference states of the program.
    ///   - `signature`: Mapping between the logical boundary and the physical executable arguments and results.
    ///   - `runtime_requirements`: Runtime contracts that executions of the program need.
    ///   - `requires_assertion_handler`: Whether executions need the runtime assertion handler.
    ///   - `donation_flags`: Donation declaration of each public input.
    ///   - `capture_count`: Number of leading inputs that are lowered captures.
    ///   - `expected_argument_shardings`: Sharding that each physical argument must have at the call boundary.
    ///   - `output_shardings`: Resolved sharding of each output.
    ///   - `committed_outputs`: Whether executions commit each output whatever the commitment of their inputs.
    ///   - `output_device_assignment`: Whether this invocation takes output assignment from the caller.
    ///   - `mesh`: Device mesh that the program executes on.
    ///   - `compilation_options`: Options that the executable was compiled with.
    ///   - `platform_name`: Name of the platform that compiled the executable.
    ///   - `platform_version`: Version of that platform.
    ///   - `device_kinds`: Kinds of the mesh devices that the executable was compiled for.
    ///   - `kernel_execution_facts`: Configuration key of the execution facts of embedded kernels, if any.
    ///   - `compiler_identity`: Identity of the compiler that produced the executable.
    ///   - `xla_flags`: Value of `XLA_FLAGS` at compilation time.
    ///   - `compilation_duration`: Time that the original backend compilation took, if it was recorded.
    ///   - `plan`: Execution plan derived from `executable` and the invocation contract.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_persisted_parts(
        executable: LoadedExecutable<'static>,
        input_types: Arc<[ArrayType]>,
        output_types: Arc<[ArrayType]>,
        output_count: usize,
        reference_states: Arc<[ExternalReferenceBinding]>,
        signature: ExecutableSignature,
        runtime_requirements: Arc<[RuntimeRequirement]>,
        requires_assertion_handler: bool,
        donation_flags: Arc<[bool]>,
        capture_count: usize,
        expected_argument_shardings: Arc<[Sharding]>,
        output_shardings: Arc<[Sharding]>,
        committed_outputs: Arc<[bool]>,
        output_device_assignment: OutputDeviceAssignment,
        mesh: DeviceMesh,
        compilation_options: CompilationOptions,
        platform_name: Arc<str>,
        platform_version: Arc<str>,
        device_kinds: Arc<[String]>,
        kernel_execution_facts: Option<Vec<u8>>,
        compiler_identity: Arc<str>,
        xla_flags: Arc<str>,
        compilation_duration: Option<Duration>,
        plan: ExecutionPlan,
    ) -> Self {
        Self {
            executable: Arc::new(executable),
            invocation: InvocationMetadata {
                input_types,
                output_types,
                output_count,
                reference_states,
                signature,
                runtime_requirements,
                requires_assertion_handler,
                donation_flags,
                capture_count,
                expected_argument_shardings,
                output_shardings,
                committed_outputs,
                output_device_assignment,
                mesh,
                compilation_options,
                kernel_execution_facts,
                open_outputs: None,
            },
            platform_name,
            platform_version,
            device_kinds,
            compiler_identity,
            xla_flags,
            compilation_duration,
            analysis: Arc::new(OnceLock::new()),
            plan: Arc::new(plan),
        }
    }

    /// Reuses the native executable with the requesting lowering's boundary and output assignment. Dimension
    /// identities can differ across lowerings of the same executable, and output assignment belongs to each request.
    /// When both already agree, returns the same shared program.
    pub(super) fn rebound_to(mut program: Arc<Self>, lowered: &LoweredProgram) -> Arc<Self> {
        let lowered = &lowered.lowering.invocation;
        let compiled = &program.invocation;
        let same_inputs =
            Arc::ptr_eq(&compiled.input_types, &lowered.input_types) || compiled.input_types == lowered.input_types;
        let same_outputs =
            Arc::ptr_eq(&compiled.output_types, &lowered.output_types) || compiled.output_types == lowered.output_types;
        if !same_inputs || !same_outputs || compiled.output_device_assignment != lowered.output_device_assignment {
            let compiled = &mut Arc::make_mut(&mut program).invocation;
            compiled.input_types = Arc::clone(&lowered.input_types);
            compiled.output_types = Arc::clone(&lowered.output_types);
            compiled.reference_states = Arc::clone(&lowered.reference_states);
            compiled.signature = lowered.signature.clone();
            compiled.output_device_assignment = lowered.output_device_assignment;
        }
        program
    }

    /// Returns this program with `kernel_execution_facts` in place of its recorded kernel execution facts, for tests
    /// of execution admission that need no kernel compiler.
    #[cfg(test)]
    pub(crate) fn with_kernel_execution_facts(mut self, kernel_execution_facts: Option<Vec<u8>>) -> Self {
        self.invocation.kernel_execution_facts = kernel_execution_facts;
        self
    }

    /// Returns this program with `mesh` in place of the mesh that it was compiled for, for tests of execution
    /// admission against a changed topology.
    #[cfg(test)]
    pub(crate) fn with_mesh(mut self, mesh: DeviceMesh) -> Self {
        self.invocation.mesh = mesh;
        self
    }

    /// Returns the runtime contracts resolved against the invoking session on every call.
    #[inline]
    pub fn runtime_requirements(&self) -> &[RuntimeRequirement] {
        &self.invocation.runtime_requirements
    }

    /// Returns the loaded PJRT executable.
    #[inline]
    pub fn executable(&self) -> &LoadedExecutable<'static> {
        &self.executable
    }

    /// Returns the canonical ordinary input boundary retained by the compiled executable.
    pub(crate) fn input_types(&self) -> &[ArrayType] {
        &self.invocation.input_types
    }

    /// Returns the public logical flat output types in the order that users see, including reconstructed leaves of
    /// zero size but excluding the hidden final state output suffix appended by reference discharge.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.invocation.output_types[..self.invocation.output_count]
    }

    /// Returns the mesh the compiled program runs against.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.invocation.mesh
    }

    /// Returns every logical flat output type, i.e., the public outputs that [`Self::output_types`] returns followed by
    /// the hidden final state outputs appended by reference discharge.
    #[inline]
    pub(crate) fn all_output_types(&self) -> &[ArrayType] {
        &self.invocation.output_types
    }

    /// Returns the external reference bindings of the executable, whose final states are its hidden outputs.
    #[inline]
    pub(crate) fn reference_states(&self) -> &[ExternalReferenceBinding] {
        &self.invocation.reference_states
    }

    /// Returns the mapping between the logical boundary of the program and the physical boundary of its executable.
    #[inline]
    pub(crate) fn signature(&self) -> &ExecutableSignature {
        &self.invocation.signature
    }

    /// Returns the default donation flag of each public logical input, excluding captures.
    #[inline]
    pub(crate) fn donation_flags(&self) -> &[bool] {
        &self.invocation.donation_flags
    }

    /// Returns the number of leading logical inputs that captures supply.
    #[inline]
    pub(crate) fn capture_count(&self) -> usize {
        self.invocation.capture_count
    }

    /// Returns the sharding that the executable expects of each physical argument, in argument order.
    #[inline]
    pub(crate) fn expected_argument_shardings(&self) -> &[Sharding] {
        &self.invocation.expected_argument_shardings
    }

    /// Returns the runtime placement of each logical output, including over auto mesh axes.
    #[inline]
    pub(crate) fn output_shardings(&self) -> &[Sharding] {
        &self.invocation.output_shardings
    }

    /// Returns whether executions of this program commit each logical output to its devices whatever the commitment of
    /// their inputs. This is the case for every public output of a lowering that requests a
    /// placement explicitly anywhere (i.e., through an input or output sharding override, or through a `place` with a
    /// target or a `constrain_sharding` in any region that it emits), as JAX commits the results of `jit` functions
    /// with explicit shardings. It is also the case for every output, including the hidden final reference states, of
    /// a lowering for a mesh of more than one device, as JAX commits every result of a computation on more than one
    /// device. The arrays that a lowering captures never commit its outputs, as JAX ignores the commitment of the
    /// arrays that a `jit` function closes over. Lowerings of the same module can differ in these flags (e.g., the
    /// identity and a placement onto the memory that a device array is already in, which lowers to nothing), so they
    /// are part of the compilation key, and a cached program serves only lowerings that commit the same outputs.
    #[inline]
    pub(crate) fn committed_outputs(&self) -> &[bool] {
        &self.invocation.committed_outputs
    }

    /// Returns how an invocation assigns its output devices.
    #[inline]
    pub(crate) fn output_device_assignment(&self) -> OutputDeviceAssignment {
        self.invocation.output_device_assignment
    }

    /// Returns the execution facts that selected kernels were compiled against, or [`None`] for a program without
    /// selected kernels.
    #[inline]
    pub(crate) fn kernel_execution_facts(&self) -> Option<&[u8]> {
        self.invocation.kernel_execution_facts.as_deref()
    }

    /// Returns the execution metadata that was derived once when this program was compiled or loaded.
    #[inline]
    pub(crate) fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Returns the effective PJRT compilation options that this program was compiled with.
    #[inline]
    pub fn compilation_options(&self) -> &CompilationOptions {
        &self.invocation.compilation_options
    }

    /// Returns whether the executable requires the host runtime assertion handler.
    #[inline]
    pub fn requires_assertion_handler(&self) -> bool {
        self.invocation.requires_assertion_handler
    }

    /// Returns the backend's optimized program only when explicitly requested.
    pub fn optimized_program(&self) -> Result<OptimizedProgram, Error> {
        Ok(OptimizedProgram::new(self.executable.executable()?.optimized_program()?))
    }

    /// Returns whether this executable requires a stateful reference invocation boundary.
    #[inline]
    pub(crate) fn requires_stateful_call(&self) -> bool {
        !self.invocation.reference_states.is_empty()
    }

    /// Returns whether this executable has no captures, needs no stateful call, and has a public boundary that matches
    /// `inputs` and `outputs`. It is checked before an independently loaded executable is exposed.
    pub(crate) fn matches_stateless_signature(&self, inputs: &[ArrayType], outputs: &[ArrayType]) -> bool {
        // An independently loaded executable recreates its dimension variables, and `ArrayType` equality compares
        // their identities, so boundaries that are not equal are compared through their canonical encoding.
        let input_types = self.invocation.input_types.as_ref();
        self.invocation.capture_count == 0
            && !self.requires_stateful_call()
            && ((input_types == inputs && self.output_types() == outputs)
                || canonical_boundary_types(input_types, self.output_types())
                    == canonical_boundary_types(inputs, outputs))
    }

    /// Returns the number of leading logical outputs exposed through the structured public result.
    #[inline]
    pub(super) fn output_count(&self) -> usize {
        self.invocation.output_count
    }

    /// Returns the PJRT platform name recorded at compilation.
    #[inline]
    pub(super) fn platform_name(&self) -> &str {
        &self.platform_name
    }

    /// Returns the PJRT platform version recorded at compilation.
    #[inline]
    pub(super) fn platform_version(&self) -> &str {
        &self.platform_version
    }

    /// Returns the device kinds in the exact mesh order used for compilation.
    #[inline]
    pub(super) fn device_kinds(&self) -> &[String] {
        &self.device_kinds
    }

    /// Returns the invocation contract for validating cached and replacement executables.
    pub(super) fn invocation(&self) -> &InvocationMetadata {
        &self.invocation
    }

    /// Returns the exact Ryft, OpenXLA, and JAX build identity used for compilation.
    #[inline]
    pub(super) fn compiler_identity(&self) -> &str {
        &self.compiler_identity
    }

    /// Returns the XLA flags of the process recorded at compilation.
    #[inline]
    pub(super) fn xla_flags(&self) -> &str {
        &self.xla_flags
    }

    /// Returns the wall-clock duration of the PJRT compilation request, if retained by the artifact.
    #[inline]
    pub(super) fn compilation_duration(&self) -> Option<Duration> {
        self.compilation_duration
    }

    /// Returns the shared lazy analysis result, retaining both successful queries and their original failures.
    #[inline]
    pub(super) fn analysis(&self) -> &Arc<OnceLock<Result<CompilationAnalysis, Error>>> {
        &self.analysis
    }
}

impl TypedProgram for CompiledProgram {
    type Type = Type;

    // Core reads the public logical signature in the type universe of the domain, which excludes the hidden
    // final state output suffix appended by reference discharge, like the inherent `output_types` function.
    fn input_count(&self) -> usize {
        self.invocation.input_types.len()
    }

    fn input_types(&self) -> Cow<'_, [Type]> {
        Cow::Owned(self.invocation.input_types.iter().cloned().map(Into::into).collect())
    }

    fn output_count(&self) -> usize {
        self.invocation.output_count
    }

    fn output_types(&self) -> Cow<'_, [Type]> {
        Cow::Owned(self.output_types().iter().cloned().map(Into::into).collect())
    }
}

/// Owned invocation contract shared by lowered and compiled artifacts.
///
/// A compiled artifact can also be loaded independently of its source, so StableHLO text and live constant bindings
/// belong only to the lowering.
#[derive(Clone, Debug)]
pub(super) struct InvocationMetadata {
    /// Effective PJRT compilation options, including SPMD partition count.
    compilation_options: CompilationOptions,

    /// Effective logical output types after applying output sharding overrides.
    output_types: Arc<[ArrayType]>,

    /// Number of leading logical outputs exposed through the structured public function result.
    output_count: usize,

    /// Logical binding recipes of external reference states in canonical source order.
    ///
    /// Flat input positions are derived from each binding's source and [`LoweredProgram::capture_count`].
    reference_states: Arc<[ExternalReferenceBinding]>,

    /// Effective logical input types, including captures first.
    input_types: Arc<[ArrayType]>,

    /// Mapping from the logical boundary to the physical StableHLO and PJRT boundary.
    signature: ExecutableSignature,

    /// Canonical runtime contracts resolved against the invoking session.
    runtime_requirements: Arc<[RuntimeRequirement]>,

    /// Whether execution requires Ryft's host runtime assertion handler.
    requires_assertion_handler: bool,

    /// Donation declarations for public inputs. Captures are never donatable.
    donation_flags: Arc<[bool]>,

    /// Number of leading logical inputs supplied by the staged capture table and [`LoweredProgram::session_constants`].
    capture_count: usize,

    /// Effective sharding expected for every physical executable argument, including materialized captures first.
    expected_argument_shardings: Arc<[Sharding]>,

    /// Runtime placement of every logical output over all mesh axes, including the auto mesh axes that output types
    /// never record. It also covers the hidden final reference states after the public result. The placement of an
    /// output flagged in [`LoweredProgram::open_outputs`] is only the placement that its type records. Compilation
    /// resolves its placement over auto mesh axes from the executable.
    output_shardings: Arc<[Sharding]>,

    /// Whether lowering left each logical output open over auto mesh axes, so that the compiler chooses its placement
    /// over them, as JAX does for outputs without `out_shardings`. Compiled programs have `None` here.
    open_outputs: Option<Arc<[bool]>>,

    /// Whether executions commit each logical output to its devices whatever the commitment of their inputs (refer to
    /// [`CompiledProgram::committed_outputs`]).
    committed_outputs: Arc<[bool]>,

    /// Whether this invocation takes output assignment from the caller or names its devices explicitly.
    output_device_assignment: OutputDeviceAssignment,

    /// Concrete mesh against which the module was lowered.
    mesh: DeviceMesh,

    /// Exact live execution facts for selected kernels, absent for ordinary executables.
    kernel_execution_facts: Option<Vec<u8>>,
}

impl InvocationMetadata {
    /// Returns the first field of the invocation metadata on which the two programs disagree, if any. Output placements
    /// are compared for the outputs that neither program leaves to the compiler, because the placement of an open
    /// output is only known once the compiler has chosen it, and they are not compared when neither program is a
    /// lowering, because a compiled program does not record which of its outputs were open. Output types do not record
    /// placements along auto mesh axes, so this comparison is what distinguishes executables that place an output
    /// along different auto axes of the same size, whose buffers have the same shapes.
    pub(super) fn incompatible_field(&self, other: &Self) -> Option<&'static str> {
        // Structurally equal boundary types have equal canonical encodings, so only unequal ones pay for encoding both.
        let equal_boundary_types = self.input_types == other.input_types && self.output_types == other.output_types;
        if !equal_boundary_types
            && canonical_boundary_types(&self.input_types, &self.output_types)
                != canonical_boundary_types(&other.input_types, &other.output_types)
        {
            Some("boundary types")
        } else if self.output_count != other.output_count {
            Some("output count")
        } else if self.reference_states != other.reference_states {
            Some("reference-state signature")
        } else if self.signature != other.signature {
            Some("executable signature")
        } else if self.runtime_requirements != other.runtime_requirements {
            Some("runtime requirements")
        } else if self.donation_flags != other.donation_flags {
            Some("donation flags")
        } else if self.capture_count != other.capture_count {
            Some("capture count")
        } else if self.expected_argument_shardings != other.expected_argument_shardings {
            Some("argument shardings")
        } else if !self.has_compatible_output_shardings(other) {
            Some("output shardings")
        } else if self.committed_outputs != other.committed_outputs {
            Some("committed outputs")
        } else if self.mesh != other.mesh {
            Some("device mesh")
        } else if self.kernel_execution_facts != other.kernel_execution_facts {
            Some("kernel execution facts")
        } else if self.requires_assertion_handler != other.requires_assertion_handler {
            Some("assertion-handler requirement")
        } else {
            None
        }
    }

    /// Returns whether the two programs place the outputs that neither of them leaves open the same way, as
    /// [`Self::incompatible_field`] describes.
    fn has_compatible_output_shardings(&self, other: &Self) -> bool {
        if self.open_outputs.is_none() && other.open_outputs.is_none() {
            return true;
        }
        let is_open = |open_outputs: Option<&[bool]>, index: usize| {
            open_outputs.is_some_and(|open_outputs| open_outputs.get(index).copied().unwrap_or(false))
        };
        self.output_shardings.len() == other.output_shardings.len()
            && self.output_shardings.iter().zip(other.output_shardings.iter()).enumerate().all(
                |(index, (sharding, other_sharding))| {
                    is_open(self.open_outputs.as_deref(), index)
                        || is_open(other.open_outputs.as_deref(), index)
                        || sharding == other_sharding
                },
            )
    }

    /// Validates that `replacement` preserves this program's complete invocation contract, naming the first
    /// incompatible metadata field otherwise.
    pub(super) fn validate_replacement(&self, replacement: &Self) -> Result<(), Error> {
        let incompatible = self
            .incompatible_field(replacement)
            .or((self.output_device_assignment != replacement.output_device_assignment)
                .then_some("output device assignment"));
        match incompatible {
            Some(field) => Err(Error::Compilation(CompilationError::IncompatibleExecutable {
                reason: format!("replacement executable has incompatible {field}"),
            })),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use ryft_core::{
        ArrayType, CompiledFunction, DataType, Dimension, DimensionBounds, DimensionVariable, ExternalReferenceBinding,
        ReferenceSource, Shape,
    };

    use crate::compilation::functions::stage;
    use crate::lowering::signature::ExecutableSignature;
    use crate::sessions::tests::domain_mesh;
    use crate::tests::{device_mesh_2x2, execution_client};
    use crate::{Error, Options, Session};

    use super::*;

    #[test]
    fn test_compiled_program_matches_stateless_signature() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let compile = |upper: usize| {
            let extent = DimensionVariable::new("n", DimensionBounds::new(1, Some(upper)).unwrap());
            let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]))
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
                .unwrap();
            let staged =
                stage::<_, ArrayType, ArrayType>(|input| input.clone() + input, input_type, Options::new(mesh.clone()))
                    .unwrap()
                    .into_inner();
            let compiled: CompiledFunction<Session, Type, Type> =
                staged.lower(&session).unwrap().compile(&session).unwrap();
            compiled
        };

        // Two compilations of one function create distinct dimension variables, like a compilation and the executable
        // that is later loaded for it, and their boundaries still match. A boundary with another bound does not. The
        // second compilation reuses the executable of the first and keeps its own boundary types.
        let (first, second, other) = (compile(8), compile(8), compile(9));
        assert_eq!(session.cached_program_count(), 2);
        let (first, second, other) = (first.compiled_program(), second.compiled_program(), other.compiled_program());
        assert_ne!(first.input_types(), second.input_types());
        assert!(first.matches_stateless_signature(first.input_types(), first.output_types()));
        assert!(first.matches_stateless_signature(second.input_types(), second.output_types()));
        assert!(!first.matches_stateless_signature(other.input_types(), other.output_types()));
    }

    #[test]
    fn test_invocation_metadata_incompatible_field() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let other_mesh = device_mesh_2x2();
        let signature = ExecutableSignature::new(&[], &[]);
        let other_signature = ExecutableSignature::new(&[ArrayType::scalar(DataType::F32)], &[]);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let current = InvocationMetadata {
            compilation_options: CompilationOptions::default(),
            input_types: [].into(),
            output_types: [].into(),
            output_count: 0,
            reference_states: [].into(),
            signature,
            runtime_requirements: [].into(),
            donation_flags: [false].into(),
            capture_count: 0,
            expected_argument_shardings: [].into(),
            output_shardings: [].into(),
            committed_outputs: [].into(),
            output_device_assignment: OutputDeviceAssignment::Fixed,
            open_outputs: None,
            mesh,
            requires_assertion_handler: false,
            kernel_execution_facts: None,
        };
        assert_eq!(current.incompatible_field(&current), None);
        assert_eq!(
            current.incompatible_field(&InvocationMetadata { signature: other_signature, ..current.clone() }),
            Some("executable signature"),
        );
        assert_eq!(
            current.incompatible_field(&InvocationMetadata { committed_outputs: [true].into(), ..current.clone() }),
            Some("committed outputs"),
        );
        assert_eq!(
            current.incompatible_field(&InvocationMetadata { capture_count: 1, ..current.clone() }),
            Some("capture count")
        );
        assert_eq!(
            current.incompatible_field(&InvocationMetadata {
                expected_argument_shardings: [sharding.clone()].into(),
                ..current.clone()
            }),
            Some("argument shardings"),
        );
        assert_eq!(
            current.incompatible_field(&InvocationMetadata { mesh: other_mesh, ..current.clone() }),
            Some("device mesh")
        );

        // Output placements are compared for the outputs that a lowering leaves closed, because a compiled program
        // records the placement that the compiler chose for an open output and does not record which outputs were open.
        let other_sharding = Sharding::replicated(current.mesh.logical_mesh().clone(), 1);
        let lowered = InvocationMetadata {
            output_shardings: [sharding].into(),
            open_outputs: Some([false].into()),
            ..current.clone()
        };
        let compiled =
            InvocationMetadata { output_shardings: [other_sharding].into(), open_outputs: None, ..current.clone() };
        assert_eq!(lowered.incompatible_field(&lowered), None);
        assert_eq!(lowered.incompatible_field(&compiled), Some("output shardings"));
        assert_eq!(compiled.incompatible_field(&lowered), Some("output shardings"));
        assert_eq!(
            InvocationMetadata { open_outputs: Some([true].into()), ..lowered.clone() }.incompatible_field(&compiled),
            None
        );
        assert_eq!(compiled.incompatible_field(&InvocationMetadata { open_outputs: None, ..lowered.clone() }), None);
    }

    #[test]
    fn test_invocation_metadata_validate_replacement() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let signature = ExecutableSignature::new(&[], &[]);
        let current = InvocationMetadata {
            compilation_options: CompilationOptions::default(),
            input_types: [].into(),
            output_types: [].into(),
            output_count: 0,
            reference_states: [].into(),
            signature,
            runtime_requirements: [].into(),
            donation_flags: [false].into(),
            capture_count: 0,
            expected_argument_shardings: [].into(),
            output_shardings: [].into(),
            committed_outputs: [].into(),
            output_device_assignment: OutputDeviceAssignment::Fixed,
            open_outputs: None,
            mesh,
            requires_assertion_handler: false,
            kernel_execution_facts: None,
        };
        let replacement = InvocationMetadata { kernel_execution_facts: Some(b"changed".to_vec()), ..current.clone() };
        assert!(matches!(
            current.validate_replacement(&replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible kernel execution facts",
        ));
        let requirements = [RuntimeRequirement::builtin("test.runtime")];
        let replacement = InvocationMetadata { runtime_requirements: requirements.into(), ..current.clone() };
        assert_eq!(current.incompatible_field(&replacement), Some("runtime requirements"));
        assert!(matches!(
            current.validate_replacement(&replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible runtime requirements",
        ));
        let replacement = InvocationMetadata { donation_flags: [true].into(), ..current.clone() };
        assert!(matches!(
            current.validate_replacement(&replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible donation flags",
        ));
        let replacement = InvocationMetadata { output_count: 1, ..current.clone() };
        assert!(matches!(
            current.validate_replacement(&replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible output count",
        ));
        let state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None);
        let replacement = InvocationMetadata { reference_states: [state].into(), ..current.clone() };
        assert!(matches!(
            current.validate_replacement(&replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible reference-state signature",
        ));
        let replacement = InvocationMetadata { requires_assertion_handler: true, ..current.clone() };
        assert!(matches!(
            current.validate_replacement(&replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible assertion-handler requirement",
        ));

        // Boundary types compare through the persistent encoding, which numbers dimension variables canonically.
        // Structurally identical bounded dynamic boundaries built from fresh dimension variables must be compatible
        // even though `ArrayType` equality compares dimension variables by identity, while a genuinely different
        // boundary must be rejected.
        let dynamic_boundary = || {
            vec![ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                    "n",
                    DimensionBounds::new(0, Some(8)).unwrap(),
                ))]),
            )]
        };
        let left_types = dynamic_boundary();
        let right_types = dynamic_boundary();
        assert_ne!(left_types, right_types);
        let left = InvocationMetadata { input_types: left_types.into(), ..current.clone() };
        let right = InvocationMetadata { input_types: right_types.into(), ..current.clone() };
        assert!(matches!(left.validate_replacement(&right), Ok(())));
        let static_types = vec![ArrayType::scalar(DataType::F32)];
        let different = InvocationMetadata { input_types: static_types.into(), ..current.clone() };
        assert!(matches!(
            left.validate_replacement(&different),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible boundary types",
        ));
    }
}
