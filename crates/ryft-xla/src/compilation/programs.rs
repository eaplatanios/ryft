//! Lowered and compiled program artifacts of the XLA [`Session`](crate::sessions::Session).
//!
//! Lowering a staged program produces a [`LoweredProgram`], which holds the StableHLO module, the effective PJRT
//! compilation options, and the invocation contract that maps logical values to physical executable slots. Compiling
//! it produces a [`CompiledProgram`], which retains the loaded PJRT executable together with the same invocation
//! contract and a reusable execution plan. Both artifacts implement [`TypedProgram`] for their public logical
//! signature, and replacement executables must preserve the complete invocation contract of the program they replace.

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use ryft_core::{ArrayType, DeviceMesh, ExternalReferenceBinding, Sharding, TypedProgram};
use ryft_pjrt::LoadedExecutable;
use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions};

use crate::compilation::CompilationError;
use crate::compilation::analysis::{CompilationAnalysis, OptimizedProgram};
use crate::compilation::keys::{COMPILER_IDENTITY, canonical_boundary_types};
use crate::compilation::options::FeedbackDirectedProfile;
use crate::execution::ExecutionPlan;
use crate::lowering::LoweredModule;
use crate::lowering::signature::ExecutableSignature;
use crate::{Error, RetainedArray, RuntimeRequirement, Target, Type};

// TODO(eaplatanios): Review this module.

/// StableHLO lowering and metadata of the call boundary produced before PJRT compilation.
///
/// A [`Session`](crate::sessions::Session) lowers a staged program into this artifact with its effective compilation
/// options and the mapping between logical values and physical executable slots. PJRT compilation produces a
/// [`CompiledProgram`] retaining the same invocation contract.
///
/// This flat artifact implements [`TypedProgram`] for its logical signature. Typed function handles in
/// [`functions`](crate::compilation::functions) additionally retain the Rust input and output container structures.
/// Hidden final reference states remain in the internal metadata but are excluded from the public output types.
#[derive(Clone, Debug)]
pub struct LoweredProgram {
    /// Textual StableHLO and Shardy module accepted by PJRT.
    stable_hlo: Arc<str>,

    /// Effective PJRT compilation options, including SPMD partition count.
    compilation_options: CompilationOptions,

    /// Effective logical output types after applying output sharding overrides.
    output_types: Arc<[ArrayType]>,

    /// Number of leading logical outputs exposed through the structured public function result.
    output_count: usize,

    /// Logical binding recipes of external reference states in canonical source order.
    ///
    /// Flat input positions are derived from each binding's source and [`Self::capture_count`].
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

    /// Number of leading logical inputs supplied by the staged capture table, followed by the hidden arguments that
    /// carry [`Self::session_constants`].
    capture_count: usize,

    /// Session constants of the lowered program, whose arrays the trailing [`Self::session_constants`]`.len()` of the
    /// [`Self::capture_count`] capture arguments carry, in order.
    session_constants: Arc<[RetainedArray]>,

    /// Effective sharding expected for every physical executable argument, including materialized captures first.
    expected_argument_shardings: Arc<[Sharding]>,

    /// Runtime placement of every logical output over all mesh axes, including the auto mesh axes that output types
    /// never record. It also covers the hidden final reference states after the public result. The placement of an
    /// output flagged in [`Self::open_outputs`] is only the placement that its type records, and compilation resolves
    /// its placement over auto mesh axes from the executable.
    output_shardings: Arc<[Sharding]>,

    /// Whether lowering left each logical output open over auto mesh axes, so that the compiler chooses its placement
    /// over them, as JAX does for outputs without `out_shardings`.
    open_outputs: Arc<[bool]>,

    /// Whether executions commit each logical output to its devices whatever the commitment of their inputs (refer to
    /// [`CompiledProgram::committed_outputs`]).
    committed_outputs: Arc<[bool]>,

    /// Concrete mesh against which the module was lowered.
    mesh: DeviceMesh,

    /// PJRT platform name of the client against which this program was lowered.
    platform_name: Arc<str>,

    /// PJRT platform version of the client against which this program was lowered.
    platform_version: Arc<str>,

    /// Device kinds in the exact mesh order used by this lowering.
    device_kinds: Arc<[String]>,

    /// Exact live execution facts for selected kernels, absent for ordinary executables.
    kernel_execution_facts: Option<Vec<u8>>,

    /// Exact Ryft, OpenXLA, and JAX build identity used by this lowering.
    compiler_identity: Arc<str>,

    /// XLA flags of the process in effect when this program was lowered.
    xla_flags: Arc<str>,
}

impl LoweredProgram {
    /// Creates a lowered program from the module that lowering rendered and the invocation contract of that module.
    /// The platform and device facts are read from `target`, and the compiler identity and the `XLA_FLAGS` value of
    /// the process are recorded as they are at the time of the call.
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
    ///   - `session_constants`: Session constants whose arrays every execution passes as captures.
    ///   - `expected_argument_shardings`: Sharding that each physical argument must have at the call boundary.
    ///   - `output_shardings`: Declared sharding of each output.
    ///   - `open_outputs`: Whether the compiler chooses the placement of each output over auto mesh axes.
    ///   - `committed_outputs`: Whether executions commit each output whatever the commitment of their inputs.
    ///   - `mesh`: Device mesh that the program is compiled for and executes on.
    ///   - `target`: Target of the session that lowered the program.
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
        session_constants: Arc<[RetainedArray]>,
        expected_argument_shardings: Arc<[Sharding]>,
        output_shardings: Arc<[Sharding]>,
        open_outputs: Arc<[bool]>,
        committed_outputs: Arc<[bool]>,
        mesh: DeviceMesh,
        target: &Target,
        kernel_execution_facts: Option<Vec<u8>>,
    ) -> Result<Self, Error> {
        let (stable_hlo, signature, requires_assertion_handler, runtime_requirements) = module.into_parts();
        Ok(Self {
            stable_hlo: stable_hlo.into(),
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
            session_constants,
            expected_argument_shardings,
            output_shardings,
            open_outputs,
            committed_outputs,
            platform_name: target.platform_name().into(),
            platform_version: target.platform_version().into(),
            device_kinds: target.mesh_device_kinds(&mesh)?.into(),
            mesh,
            kernel_execution_facts,
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
        self.compilation_options
            .executable_build_options
            .get_or_insert_with(ExecutableCompilationOptions::default)
            .fdo_profile = profile.bytes().to_vec();
        self.compilation_options.profile_version = profile.version();
        self
    }

    /// Returns this program with `stable_hlo` in place of its lowered module, for tests that compile a module which
    /// lowering itself would never produce.
    #[cfg(test)]
    pub(crate) fn with_stable_hlo(mut self, stable_hlo: impl Into<Arc<str>>) -> Self {
        self.stable_hlo = stable_hlo.into();
        self
    }

    /// Returns this program with `compilation_options` in place of its effective compilation options, for tests
    /// that force a compilation mode which lowering would not select.
    #[cfg(test)]
    pub(crate) fn with_compilation_options(mut self, compilation_options: CompilationOptions) -> Self {
        self.compilation_options = compilation_options;
        self
    }

    /// Returns the textual StableHLO and Shardy module.
    #[inline]
    pub fn stable_hlo(&self) -> &str {
        &self.stable_hlo
    }

    /// Returns the effective public logical flat output types, including leaves of zero size erased from the
    /// executable ABI but excluding the hidden final state output suffix appended by reference discharge.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.output_types[..self.output_count]
    }

    /// Returns every logical flat output type (i.e., the public outputs that [`Self::output_types`] returns followed by
    /// the hidden final state outputs appended by reference discharge).
    #[inline]
    pub(super) fn all_output_types(&self) -> &[ArrayType] {
        &self.output_types
    }

    /// Returns the runtime contracts that the invoking session must provide before compilation or execution.
    #[inline]
    pub fn runtime_requirements(&self) -> &[RuntimeRequirement] {
        &self.runtime_requirements
    }

    /// Returns the device mesh this lowering targets.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Returns the handles of the session constants that the executable receives as hidden arguments after its
    /// captures. The handles are shared so that callers can retain them without copying.
    #[inline]
    pub(crate) fn session_constants(&self) -> &Arc<[RetainedArray]> {
        &self.session_constants
    }

    /// Returns the effective PJRT compilation options that this program is compiled with.
    #[inline]
    pub fn compilation_options(&self) -> &CompilationOptions {
        &self.compilation_options
    }

    /// Returns the default donation flag of each public logical input, excluding captures.
    #[inline]
    pub fn donation_flags(&self) -> &[bool] {
        &self.donation_flags
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

    /// Returns the PJRT device kinds of the mesh devices, in mesh order.
    #[inline]
    pub fn device_kinds(&self) -> &[String] {
        &self.device_kinds
    }

    /// Returns the mapping between the logical boundary of the program and the physical boundary of its executable.
    pub(crate) fn signature(&self) -> &ExecutableSignature {
        &self.signature
    }

    /// Returns the number of leading logical outputs exposed through the structured public function result.
    #[inline]
    pub(super) fn output_count(&self) -> usize {
        self.output_count
    }

    /// Returns the logical binding recipes of external reference states in canonical source order.  Flat input
    /// positions are derived from each binding's source and [`Self::capture_count`].
    #[inline]
    pub(super) fn reference_states(&self) -> &[ExternalReferenceBinding] {
        &self.reference_states
    }

    /// Returns the effective logical input types, including captures first.
    #[inline]
    pub(super) fn input_types(&self) -> &[ArrayType] {
        &self.input_types
    }

    /// Returns whether execution requires Ryft's host runtime assertion handler.
    #[inline]
    pub(super) fn requires_assertion_handler(&self) -> bool {
        self.requires_assertion_handler
    }

    /// Returns the number of leading logical inputs supplied by the staged capture table, followed by the hidden
    /// arguments that carry [`Self::session_constants`].
    #[inline]
    pub(super) fn capture_count(&self) -> usize {
        self.capture_count
    }

    /// Returns the effective sharding expected for every physical executable argument, including materialized captures
    /// first.
    #[inline]
    pub(super) fn expected_argument_shardings(&self) -> &[Sharding] {
        &self.expected_argument_shardings
    }

    /// Returns the runtime placement of every logical output over all mesh axes, including the auto mesh axes that
    /// output types never record. It also covers the hidden final reference states after the public result. The
    /// placement of an output flagged in [`Self::open_outputs`] is only the placement that its type records, and
    /// compilation resolves its placement over auto mesh axes from the executable.
    #[inline]
    pub(super) fn output_shardings(&self) -> &[Sharding] {
        &self.output_shardings
    }

    /// Returns whether lowering left each logical output open over auto mesh axes, so that the compiler chooses its
    /// placement over them, as JAX does for outputs without `out_shardings`.
    #[inline]
    pub(super) fn open_outputs(&self) -> &[bool] {
        &self.open_outputs
    }

    /// Returns the exact live execution facts for selected kernels, absent for ordinary executables.
    #[inline]
    pub(crate) fn kernel_execution_facts(&self) -> Option<&[u8]> {
        self.kernel_execution_facts.as_deref()
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
        self.input_types.len()
    }

    fn input_types(&self) -> Cow<'_, [Type]> {
        Cow::Owned(self.input_types.iter().cloned().map(Into::into).collect())
    }

    fn output_count(&self) -> usize {
        self.output_count
    }

    fn output_types(&self) -> Cow<'_, [Type]> {
        Cow::Owned(self.output_types[..self.output_count].iter().cloned().map(Into::into).collect())
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
/// The invoking [`Session`](crate::sessions::Session) validates inputs, resolves placement and runtime resources,
/// materializes bounded dynamic arguments, and reconstructs logical outputs. Ordinary calls carry completion through
/// their outputs. Stateful calls coordinate external references through [`ryft_core::StatefulExecution`], publishing
/// mutations only after submission and tracking completion of the whole invocation.
#[derive(Clone)]
pub struct CompiledProgram {
    /// Loaded PJRT executable, shared by cloned program handles.
    executable: Arc<LoadedExecutable<'static>>,

    /// Logical input types, including captures before public inputs.
    input_types: Arc<[ArrayType]>,

    /// Logical output types, including hidden final reference states after the public result.
    output_types: Arc<[ArrayType]>,

    /// Number of leading logical outputs exposed through the structured public result.
    output_count: usize,

    /// Logical external reference binding recipes in canonical source order.
    reference_states: Arc<[ExternalReferenceBinding]>,

    /// Mapping between logical arrays and physical executable arguments, results, extents, and effects.
    signature: ExecutableSignature,

    /// Canonical runtime contracts resolved against the invoking session.
    runtime_requirements: Arc<[RuntimeRequirement]>,

    /// Whether execution requires the host runtime assertion handler.
    requires_assertion_handler: bool,

    /// Donation declarations for public inputs. Captures are never donatable.
    donation_flags: Arc<[bool]>,

    /// Number of leading logical inputs supplied by the staged capture table.
    capture_count: usize,

    /// Effective sharding of every physical executable argument, including materialized captures first.
    expected_argument_shardings: Arc<[Sharding]>,

    /// Runtime placement of every logical output over all mesh axes, including the auto mesh axes that output types
    /// never record. It also covers the hidden final reference states after the public result.
    output_shardings: Arc<[Sharding]>,

    /// Whether executions of the lowering that this program serves commit each logical output to its devices whatever
    /// the commitment of their inputs (refer to [`Self::committed_outputs`]).
    committed_outputs: Arc<[bool]>,

    /// Concrete device mesh used for compilation and execution.
    mesh: DeviceMesh,

    /// Effective PJRT compilation options used to build the executable.
    compilation_options: CompilationOptions,

    /// PJRT platform name recorded at compilation.
    platform_name: Arc<str>,

    /// PJRT platform version recorded at compilation.
    platform_version: Arc<str>,

    /// Device kinds in the exact mesh order used for compilation.
    device_kinds: Arc<[String]>,

    /// Exact live execution facts for selected kernels, absent for ordinary executables.
    kernel_execution_facts: Option<Vec<u8>>,

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
            input_types: Arc::clone(&program.input_types),
            output_types: Arc::clone(&program.output_types),
            output_count: program.output_count,
            reference_states: Arc::clone(&program.reference_states),
            signature: program.signature.clone(),
            runtime_requirements: Arc::clone(&program.runtime_requirements),
            requires_assertion_handler: program.requires_assertion_handler,
            donation_flags: Arc::clone(&program.donation_flags),
            capture_count: program.capture_count,
            expected_argument_shardings: Arc::clone(&program.expected_argument_shardings),
            output_shardings,
            committed_outputs: Arc::clone(&program.committed_outputs),
            mesh: program.mesh.clone(),
            compilation_options: program.compilation_options.clone(),
            platform_name: Arc::clone(&program.platform_name),
            platform_version: Arc::clone(&program.platform_version),
            device_kinds: Arc::clone(&program.device_kinds),
            kernel_execution_facts: program.kernel_execution_facts.clone(),
            compiler_identity: Arc::clone(&program.compiler_identity),
            xla_flags: Arc::clone(&program.xla_flags),
            compilation_duration: Some(compilation_duration),
            analysis: Arc::new(OnceLock::new()),
            plan: Arc::new(plan),
        }
    }

    /// Creates a compiled program from an `executable` that was loaded from its serialized form and from the
    /// invocation contract that was persisted with it. Which outputs executions always commit is a property of the
    /// lowering that a program serves rather than of its executable (refer to [`Self::committed_outputs`]), so it is
    /// not persisted, and the result commits no output by itself until it is rebound to a lowering (refer to
    /// [`Self::rebound_to`]).
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
            committed_outputs: vec![false; output_shardings.len()].into(),
            output_shardings,
            mesh,
            compilation_options,
            platform_name,
            platform_version,
            device_kinds,
            kernel_execution_facts,
            compiler_identity,
            xla_flags,
            compilation_duration,
            analysis: Arc::new(OnceLock::new()),
            plan: Arc::new(plan),
        }
    }

    /// Returns `program` itself if it already has the boundary of `lowered` and commits the same outputs, and otherwise
    /// a program that takes its boundary types, reference states, signature, and committed outputs from `lowered` and
    /// shares everything else, including its executable, with `program`. A program that was compiled for another
    /// lowering has the same boundary up to the identities of its dimension variables, and callers compare boundary
    /// types by identity. Which outputs executions always commit is not part of the compilation key, because it does
    /// not change the executable (e.g., an input sharding override and a runtime input placement with the same
    /// sharding, or two aliases of the same captured buffers of which only one is committed, compile to the same
    /// executable), so it comes from `lowered` as well.
    pub(super) fn rebound_to(program: Arc<Self>, lowered: &LoweredProgram) -> Arc<Self> {
        let same_inputs =
            Arc::ptr_eq(&program.input_types, &lowered.input_types) || program.input_types == lowered.input_types;
        let same_outputs =
            Arc::ptr_eq(&program.output_types, &lowered.output_types) || program.output_types == lowered.output_types;
        let same_committed_outputs = program.committed_outputs == lowered.committed_outputs;
        if same_inputs && same_outputs && same_committed_outputs {
            program
        } else {
            Arc::new(Self {
                input_types: Arc::clone(&lowered.input_types),
                output_types: Arc::clone(&lowered.output_types),
                reference_states: Arc::clone(&lowered.reference_states),
                signature: lowered.signature.clone(),
                committed_outputs: Arc::clone(&lowered.committed_outputs),
                ..Self::clone(&program)
            })
        }
    }

    /// Returns this program with `kernel_execution_facts` in place of its recorded kernel execution facts, for tests
    /// of execution admission that need no kernel compiler.
    #[cfg(test)]
    pub(crate) fn with_kernel_execution_facts(mut self, kernel_execution_facts: Option<Vec<u8>>) -> Self {
        self.kernel_execution_facts = kernel_execution_facts;
        self
    }

    /// Returns this program with `mesh` in place of the mesh that it was compiled for, for tests of execution
    /// admission against a changed topology.
    #[cfg(test)]
    pub(crate) fn with_mesh(mut self, mesh: DeviceMesh) -> Self {
        self.mesh = mesh;
        self
    }

    /// Returns the runtime contracts resolved against the invoking session on every call.
    #[inline]
    pub fn runtime_requirements(&self) -> &[RuntimeRequirement] {
        &self.runtime_requirements
    }

    /// Returns the loaded PJRT executable.
    #[inline]
    pub fn executable(&self) -> &LoadedExecutable<'static> {
        &self.executable
    }

    /// Returns the canonical ordinary input boundary retained by the compiled executable.
    pub(crate) fn input_types(&self) -> &[ArrayType] {
        &self.input_types
    }

    /// Returns the public logical flat output types in the order that users see, including reconstructed leaves of
    /// zero size but excluding the hidden final state output suffix appended by reference discharge.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.output_types[..self.output_count]
    }

    /// Returns the mesh the compiled program runs against.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Returns every logical flat output type, i.e., the public outputs that [`Self::output_types`] returns followed by
    /// the hidden final state outputs appended by reference discharge.
    #[inline]
    pub(crate) fn all_output_types(&self) -> &[ArrayType] {
        &self.output_types
    }

    /// Returns the external reference bindings of the executable, whose final states are its hidden outputs.
    #[inline]
    pub(crate) fn reference_states(&self) -> &[ExternalReferenceBinding] {
        &self.reference_states
    }

    /// Returns the mapping between the logical boundary of the program and the physical boundary of its executable.
    #[inline]
    pub(crate) fn signature(&self) -> &ExecutableSignature {
        &self.signature
    }

    /// Returns the default donation flag of each public logical input, excluding captures.
    #[inline]
    pub(crate) fn donation_flags(&self) -> &[bool] {
        &self.donation_flags
    }

    /// Returns the number of leading logical inputs that captures supply.
    #[inline]
    pub(crate) fn capture_count(&self) -> usize {
        self.capture_count
    }

    /// Returns the sharding that the executable expects of each physical argument, in argument order.
    #[inline]
    pub(crate) fn expected_argument_shardings(&self) -> &[Sharding] {
        &self.expected_argument_shardings
    }

    /// Returns the runtime placement of each logical output, including over auto mesh axes.
    #[inline]
    pub(crate) fn output_shardings(&self) -> &[Sharding] {
        &self.output_shardings
    }

    /// Returns whether executions of the lowering that this program serves commit each logical output to its devices
    /// whatever the commitment of their inputs. This is the case for every public output of a lowering that requests a
    /// placement explicitly anywhere (i.e., through an input or output sharding override, or through a `place` with a
    /// target or a `constrain_sharding` in any region that it emits), as JAX commits the results of `jit` functions
    /// with explicit shardings. It is also the case for every output, including the hidden final reference states, of
    /// a lowering for a mesh of more than one device, as JAX commits every result of a computation on more than one
    /// device, and of a lowering that captures a committed array through any of its handles, like the outputs of an
    /// operation that has a committed input.
    #[inline]
    pub(crate) fn committed_outputs(&self) -> &[bool] {
        &self.committed_outputs
    }

    /// Returns the execution facts that selected kernels were compiled against, or [`None`] for a program without
    /// selected kernels.
    #[inline]
    pub(crate) fn kernel_execution_facts(&self) -> Option<&[u8]> {
        self.kernel_execution_facts.as_deref()
    }

    /// Returns the execution metadata that was derived once when this program was compiled or loaded.
    #[inline]
    pub(crate) fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Returns the effective PJRT compilation options that this program was compiled with.
    #[inline]
    pub fn compilation_options(&self) -> &CompilationOptions {
        &self.compilation_options
    }

    /// Returns whether the executable requires the host runtime assertion handler.
    #[inline]
    pub fn requires_assertion_handler(&self) -> bool {
        self.requires_assertion_handler
    }

    /// Returns the backend's optimized program only when explicitly requested.
    pub fn optimized_program(&self) -> Result<OptimizedProgram, Error> {
        Ok(OptimizedProgram::new(self.executable.executable()?.optimized_program()?))
    }

    /// Returns whether this executable requires a stateful reference invocation boundary.
    #[inline]
    pub(crate) fn requires_stateful_call(&self) -> bool {
        !self.reference_states.is_empty()
    }

    /// Returns whether this executable has no captures, needs no stateful call, and has a public boundary that matches
    /// `inputs` and `outputs`. It is checked before an independently loaded executable is exposed.
    pub(crate) fn matches_stateless_signature(&self, inputs: &[ArrayType], outputs: &[ArrayType]) -> bool {
        // An independently loaded executable recreates its dimension variables, and `ArrayType` equality compares
        // their identities, so boundaries that are not equal are compared through their canonical encoding.
        let input_types = self.input_types.as_ref();
        self.capture_count == 0
            && !self.requires_stateful_call()
            && ((input_types == inputs && self.output_types() == outputs)
                || canonical_boundary_types(input_types, self.output_types())
                    == canonical_boundary_types(inputs, outputs))
    }

    /// Returns the number of leading logical outputs exposed through the structured public result.
    #[inline]
    pub(super) fn output_count(&self) -> usize {
        self.output_count
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
        self.input_types.len()
    }

    fn input_types(&self) -> Cow<'_, [Type]> {
        Cow::Owned(self.input_types.iter().cloned().map(Into::into).collect())
    }

    fn output_count(&self) -> usize {
        self.output_count
    }

    fn output_types(&self) -> Cow<'_, [Type]> {
        Cow::Owned(self.output_types[..self.output_count].iter().cloned().map(Into::into).collect())
    }
}

/// Borrowed invocation contract shared by lowered and compiled artifacts for compatibility validation.
#[derive(Copy, Clone)]
pub(super) struct InvocationMetadata<'o> {
    /// Logical input types, including captures before public inputs.
    input_types: &'o [ArrayType],

    /// Logical output types, including hidden final reference states after the public result.
    output_types: &'o [ArrayType],

    /// Number of leading logical outputs exposed through the structured public result.
    output_count: usize,

    /// Logical external reference binding recipes in canonical source order.
    reference_states: &'o [ExternalReferenceBinding],

    /// Mapping between logical arrays and physical executable arguments, results, extents, and effects.
    signature: &'o ExecutableSignature,

    /// Canonical runtime contracts that the invoking session must provide.
    runtime_requirements: &'o [RuntimeRequirement],

    /// Donation declarations for public inputs. Captures are never donatable.
    donation_flags: &'o [bool],

    /// Number of leading logical inputs supplied by the staged capture table.
    capture_count: usize,

    /// Effective sharding of every physical executable argument, including materialized captures first.
    expected_argument_shardings: &'o [Sharding],

    /// Concrete device mesh used for compilation and execution.
    mesh: &'o DeviceMesh,

    /// Whether execution requires the host runtime assertion handler.
    requires_assertion_handler: bool,

    /// Exact live execution facts for selected kernels, absent for ordinary executables.
    kernel_execution_facts: Option<&'o [u8]>,
}

impl InvocationMetadata<'_> {
    /// Returns the first field of the invocation metadata on which the two programs disagree, if any.
    pub(super) fn incompatible_field(self, other: Self) -> Option<&'static str> {
        // Structurally equal boundary types have equal canonical encodings, so only unequal ones pay for encoding both.
        let equal_boundary_types = self.input_types == other.input_types && self.output_types == other.output_types;
        if !equal_boundary_types
            && canonical_boundary_types(self.input_types, self.output_types)
                != canonical_boundary_types(other.input_types, other.output_types)
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

    /// Validates that `replacement` preserves this program's complete invocation contract, naming the first
    /// incompatible metadata field otherwise.
    pub(super) fn validate_replacement(self, replacement: Self) -> Result<(), Error> {
        match self.incompatible_field(replacement) {
            Some(field) => Err(Error::Compilation(CompilationError::IncompatibleExecutable {
                reason: format!("replacement executable has incompatible {field}"),
            })),
            None => Ok(()),
        }
    }
}

impl<'o> From<&'o CompiledProgram> for InvocationMetadata<'o> {
    fn from(program: &'o CompiledProgram) -> Self {
        Self {
            input_types: &program.input_types,
            output_types: &program.output_types,
            output_count: program.output_count,
            reference_states: &program.reference_states,
            signature: &program.signature,
            runtime_requirements: &program.runtime_requirements,
            donation_flags: &program.donation_flags,
            capture_count: program.capture_count,
            expected_argument_shardings: &program.expected_argument_shardings,
            mesh: &program.mesh,
            requires_assertion_handler: program.requires_assertion_handler,
            kernel_execution_facts: program.kernel_execution_facts.as_deref(),
        }
    }
}

impl<'o> From<&'o LoweredProgram> for InvocationMetadata<'o> {
    fn from(program: &'o LoweredProgram) -> Self {
        Self {
            input_types: &program.input_types,
            output_types: &program.output_types,
            output_count: program.output_count,
            reference_states: &program.reference_states,
            signature: &program.signature,
            runtime_requirements: &program.runtime_requirements,
            donation_flags: &program.donation_flags,
            capture_count: program.capture_count,
            expected_argument_shardings: &program.expected_argument_shardings,
            mesh: &program.mesh,
            requires_assertion_handler: program.requires_assertion_handler,
            kernel_execution_facts: program.kernel_execution_facts.as_deref(),
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
            input_types: &[],
            output_types: &[],
            output_count: 0,
            reference_states: &[],
            signature: &signature,
            runtime_requirements: &[],
            donation_flags: &[false],
            capture_count: 0,
            expected_argument_shardings: &[],
            mesh: &mesh,
            requires_assertion_handler: false,
            kernel_execution_facts: None,
        };
        assert_eq!(current.incompatible_field(current), None);
        assert_eq!(
            current.incompatible_field(InvocationMetadata { signature: &other_signature, ..current }),
            Some("executable signature"),
        );
        assert_eq!(
            current.incompatible_field(InvocationMetadata { capture_count: 1, ..current }),
            Some("capture count")
        );
        assert_eq!(
            current.incompatible_field(InvocationMetadata {
                expected_argument_shardings: std::slice::from_ref(&sharding),
                ..current
            }),
            Some("argument shardings"),
        );
        assert_eq!(
            current.incompatible_field(InvocationMetadata { mesh: &other_mesh, ..current }),
            Some("device mesh")
        );
    }

    #[test]
    fn test_invocation_metadata_validate_replacement() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let signature = ExecutableSignature::new(&[], &[]);
        let current = InvocationMetadata {
            input_types: &[],
            output_types: &[],
            output_count: 0,
            reference_states: &[],
            signature: &signature,
            runtime_requirements: &[],
            donation_flags: &[false],
            capture_count: 0,
            expected_argument_shardings: &[],
            mesh: &mesh,
            requires_assertion_handler: false,
            kernel_execution_facts: None,
        };
        let replacement = InvocationMetadata { kernel_execution_facts: Some(b"changed"), ..current };
        assert!(matches!(
            current.validate_replacement(replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible kernel execution facts",
        ));
        let requirements = [RuntimeRequirement::builtin("test.runtime")];
        let replacement = InvocationMetadata { runtime_requirements: &requirements, ..current };
        assert_eq!(current.incompatible_field(replacement), Some("runtime requirements"));
        assert!(matches!(
            current.validate_replacement(replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible runtime requirements",
        ));
        let replacement = InvocationMetadata { donation_flags: &[true], ..current };
        assert!(matches!(
            current.validate_replacement(replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible donation flags",
        ));
        let replacement = InvocationMetadata { output_count: 1, ..current };
        assert!(matches!(
            current.validate_replacement(replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible output count",
        ));
        let state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None);
        let replacement = InvocationMetadata { reference_states: std::slice::from_ref(&state), ..current };
        assert!(matches!(
            current.validate_replacement(replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible reference-state signature",
        ));
        let replacement = InvocationMetadata { requires_assertion_handler: true, ..current };
        assert!(matches!(
            current.validate_replacement(replacement),
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
        let left = InvocationMetadata { input_types: left_types.as_slice(), ..current };
        let right = InvocationMetadata { input_types: right_types.as_slice(), ..current };
        assert!(matches!(left.validate_replacement(right), Ok(())));
        let static_types = vec![ArrayType::scalar(DataType::F32)];
        let left = InvocationMetadata { input_types: left_types.as_slice(), ..current };
        let different = InvocationMetadata { input_types: static_types.as_slice(), ..current };
        assert!(matches!(
            left.validate_replacement(different),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible boundary types",
        ));
    }
}
