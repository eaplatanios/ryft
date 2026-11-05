use std::borrow::Cow;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use ryft_core::{ArrayIrType, ArrayType, DeviceMesh, ExternalReferenceBinding, Sharding, TypedProgram};
use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{LoadedExecutable, Program as PjrtProgram};

use crate::Error;
use crate::domains::analysis::{CompilationAnalysis, OptimizedProgram};
use crate::domains::compilation::CompilationError;
use crate::domains::execution::ExecutionPlan;
use crate::domains::options::FeedbackDirectedProfile;
use crate::domains::persistence::canonical_boundary_types;
use crate::domains::runtimes::RuntimeRequirement;
use crate::experimental::lowering::signature::ExecutableSignature;

// TODO(eaplatanios): Review this module.

/// StableHLO lowering and call-boundary metadata produced before PJRT compilation.
///
/// A [`Domain`](crate::domains::Domain) lowers a staged program into this artifact with its effective compilation
/// options and the mapping between logical values and physical executable slots. PJRT compilation produces a
/// [`CompiledProgram`] retaining the same invocation contract.
///
/// This flat artifact implements [`TypedProgram`] for its logical signature. Typed function handles in
/// [`crate::compilation`] additionally retain the Rust input and output container structures. Hidden final reference
/// states remain in the internal metadata but are excluded from the public output types.
#[derive(Clone, Debug)]
pub struct LoweredProgram {
    /// Textual StableHLO/Shardy module accepted by PJRT.
    pub(super) stable_hlo: Arc<str>,

    /// Effective PJRT compilation options, including SPMD partition count.
    pub(super) compilation_options: CompilationOptions,

    /// Effective logical output types after applying output-sharding overrides.
    pub(super) output_types: Arc<[ArrayType]>,

    /// Number of leading logical outputs exposed through the structured public function result.
    pub(super) output_count: usize,

    /// Logical external reference-state binding recipes in canonical source order.
    ///
    /// Flat input positions are derived from each binding's source and [`Self::capture_count`].
    pub(super) reference_states: Arc<[ExternalReferenceBinding]>,

    /// Effective logical input types, including captures first.
    pub(super) input_types: Arc<[ArrayType]>,

    /// Logical-to-physical StableHLO/PJRT boundary mapping.
    pub(super) signature: ExecutableSignature,

    /// Canonical runtime contracts resolved against the invoking session.
    pub(super) runtime_requirements: Arc<[RuntimeRequirement]>,

    /// Whether execution requires Ryft's host runtime assertion handler.
    pub(super) requires_assertion_handler: bool,

    /// Donation declarations for public inputs. Captures are always non-donatable.
    pub(super) donation_flags: Arc<[bool]>,

    /// Number of leading logical inputs supplied by the staged capture table.
    pub(super) capture_count: usize,

    /// Effective sharding expected for every physical executable argument, including materialized captures first.
    pub(super) expected_argument_shardings: Arc<[Sharding]>,

    /// Concrete mesh against which the module was lowered.
    pub(super) mesh: DeviceMesh,

    /// PJRT platform name of the client against which this program was lowered.
    pub(super) platform_name: Arc<str>,

    /// PJRT platform version of the client against which this program was lowered.
    pub(super) platform_version: Arc<str>,

    /// Device kinds in the exact mesh order used by this lowering.
    pub(super) device_kinds: Arc<[String]>,

    /// Exact live execution facts for selected kernels; ordinary executables leave this absent.
    pub(super) kernel_execution_facts: Option<Vec<u8>>,

    /// Exact Ryft, OpenXLA, and JAX build identity used by this lowering.
    pub(super) compiler_identity: Arc<str>,

    /// Process-level XLA flags in effect when this program was lowered.
    pub(super) xla_flags: Arc<str>,
}

impl LoweredProgram {
    /// Returns this lowering with an opaque feedback-directed profile installed in its exact PJRT compilation
    /// options.
    ///
    /// The StableHLO and runtime signature are unchanged. Because canonical compilation identity includes the
    /// complete PJRT options, the returned lowering receives a distinct in-memory, persistent, and distributed cache
    /// key from the baseline executable.
    pub fn with_feedback_directed_profile(mut self, profile: FeedbackDirectedProfile) -> Self {
        use ryft_pjrt::protos::ExecutableCompilationOptions;

        self.compilation_options
            .executable_build_options
            .get_or_insert_with(ExecutableCompilationOptions::default)
            .fdo_profile = profile.bytes().to_vec();
        self.compilation_options.profile_version = profile.version();
        self
    }

    /// Returns the textual StableHLO/Shardy module.
    #[inline]
    pub fn stable_hlo(&self) -> &str {
        &self.stable_hlo
    }

    /// Returns the effective public logical flat output types, including zero-space leaves erased from the
    /// executable ABI but excluding the hidden final-state output suffix appended by reference discharge.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.output_types[..self.output_count]
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
}

impl TypedProgram for LoweredProgram {
    type Type = ArrayIrType;

    // Core reads the public logical signature in the type universe of the domain, which excludes the hidden
    // final-state output suffix appended by reference discharge, like the inherent `output_types` function.
    fn input_count(&self) -> usize {
        self.input_types.len()
    }

    fn input_types(&self) -> Cow<'_, [ArrayIrType]> {
        Cow::Owned(self.input_types.iter().cloned().map(Into::into).collect())
    }

    fn output_count(&self) -> usize {
        self.output_count
    }

    fn output_types(&self) -> Cow<'_, [ArrayIrType]> {
        Cow::Owned(self.output_types[..self.output_count].iter().cloned().map(Into::into).collect())
    }
}

/// Loaded PJRT executable plus the metadata required to invoke it safely.
///
/// Compilation retains the [`LoweredProgram`]'s logical inputs, outputs, captures, reference bindings, and physical
/// boundary mapping alongside a reusable execution plan. This flat artifact implements [`TypedProgram`] for its
/// logical signature, excluding hidden final reference states from public output types. Typed function handles in
/// [`crate::compilation`] retain the Rust container structures separately. Reused and replacement executables must
/// preserve the complete invocation contract before execution.
///
/// The invoking [`Domain`](crate::domains::Domain) validates inputs, resolves placement and runtime resources,
/// materializes bounded dynamic arguments, and reconstructs logical outputs. Ordinary calls carry completion through
/// their outputs. Stateful calls coordinate external references through [`ryft_core::StatefulExecution`], publishing
/// mutations only after submission and tracking completion of the whole invocation.
#[derive(Clone)]
pub struct CompiledProgram<'c> {
    /// Loaded PJRT executable, shared by cloned program handles.
    pub(super) executable: Arc<LoadedExecutable<'c>>,

    /// Logical input types, including captures before public inputs.
    pub(super) input_types: Arc<[ArrayType]>,

    /// Logical output types, including hidden final reference states after the public result.
    pub(super) output_types: Arc<[ArrayType]>,

    /// Number of leading logical outputs exposed through the structured public result.
    pub(super) output_count: usize,

    /// Logical external reference binding recipes in canonical source order.
    pub(super) reference_states: Arc<[ExternalReferenceBinding]>,

    /// Mapping between logical arrays and physical executable arguments, results, extents, and effects.
    pub(super) signature: ExecutableSignature,

    /// Canonical runtime contracts resolved against the invoking session.
    pub(super) runtime_requirements: Arc<[RuntimeRequirement]>,

    /// Whether execution requires the host runtime assertion handler.
    pub(super) requires_assertion_handler: bool,

    /// Donation declarations for public inputs. Captures are always non-donatable.
    pub(super) donation_flags: Arc<[bool]>,

    /// Number of leading logical inputs supplied by the staged capture table.
    pub(super) capture_count: usize,

    /// Effective sharding of every physical executable argument, including materialized captures first.
    pub(super) expected_argument_shardings: Arc<[Sharding]>,

    /// Concrete device mesh used for compilation and execution.
    pub(super) mesh: DeviceMesh,

    /// Effective PJRT compilation options used to build the executable.
    pub(super) compilation_options: CompilationOptions,

    /// PJRT platform name recorded at compilation.
    pub(super) platform_name: Arc<str>,

    /// PJRT platform version recorded at compilation.
    pub(super) platform_version: Arc<str>,

    /// Device kinds in the exact mesh order used for compilation.
    pub(super) device_kinds: Arc<[String]>,

    /// Exact live execution facts for selected kernels; ordinary executables leave this absent.
    pub(super) kernel_execution_facts: Option<Vec<u8>>,

    /// Exact Ryft, OpenXLA, and JAX build identity used for compilation.
    pub(super) compiler_identity: Arc<str>,

    /// Process-level XLA flags recorded at compilation.
    pub(super) xla_flags: Arc<str>,

    /// Wall-clock duration of the PJRT compilation request, if retained by the artifact.
    pub(super) compilation_duration: Option<Duration>,

    /// Shared lazy analysis result, retaining both successful queries and their original failures.
    pub(super) analysis: Arc<OnceLock<Result<CompilationAnalysis, Error>>>,

    /// Value-independent execution metadata, computed once when the program is compiled or loaded.
    pub(super) plan: Arc<ExecutionPlan>,
}

impl<'c> CompiledProgram<'c> {
    /// Returns the runtime contracts resolved against the invoking session on every call.
    #[inline]
    pub fn runtime_requirements(&self) -> &[RuntimeRequirement] {
        &self.runtime_requirements
    }

    /// Returns the loaded PJRT executable.
    #[inline]
    pub fn executable(&self) -> &LoadedExecutable<'c> {
        &self.executable
    }

    /// Returns the canonical ordinary input boundary retained by the compiled executable.
    pub(crate) fn input_types(&self) -> &[ArrayType] {
        &self.input_types
    }

    /// Returns the public logical flat output types in user-visible order, including reconstructed zero-space
    /// leaves but excluding the hidden final-state output suffix appended by reference discharge.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.output_types[..self.output_count]
    }

    /// Returns the mesh the compiled program runs against.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Returns the backend's optimized program only when explicitly requested.
    pub fn optimized_program(&self) -> Result<OptimizedProgram, Error> {
        match self.executable.executable()?.optimized_program()? {
            PjrtProgram::Mlir { bytecode } => Ok(OptimizedProgram { format: "mlir", bytes: bytecode }),
            PjrtProgram::Hlo { proto } => Ok(OptimizedProgram { format: "hlo", bytes: proto }),
            PjrtProgram::HloWithConfig { proto } => Ok(OptimizedProgram { format: "hlo_with_config", bytes: proto }),
        }
    }

    /// Returns whether this executable requires a stateful reference invocation boundary.
    #[inline]
    pub(crate) fn requires_stateful_call(&self) -> bool {
        !self.reference_states.is_empty()
    }

    /// Checks a capture-free, stateless public boundary before an independently loaded executable is exposed.
    pub(crate) fn matches_stateless_signature(&self, inputs: &[ArrayType], outputs: &[ArrayType]) -> bool {
        self.capture_count == 0
            && !self.requires_stateful_call()
            && self.input_types.as_ref() == inputs
            && self.output_types() == outputs
    }
}

impl TypedProgram for CompiledProgram<'_> {
    type Type = ArrayIrType;

    // Core reads the public logical signature in the type universe of the domain, which excludes the hidden
    // final-state output suffix appended by reference discharge, like the inherent `output_types` function.
    fn input_count(&self) -> usize {
        self.input_types.len()
    }

    fn input_types(&self) -> Cow<'_, [ArrayIrType]> {
        Cow::Owned(self.input_types.iter().cloned().map(Into::into).collect())
    }

    fn output_count(&self) -> usize {
        self.output_count
    }

    fn output_types(&self) -> Cow<'_, [ArrayIrType]> {
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

    /// Donation declarations for public inputs. Captures are always non-donatable.
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
    /// Returns the first invocation-metadata field on which the two programs disagree, if any.
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

impl<'o, 'c> From<&'o CompiledProgram<'c>> for InvocationMetadata<'o> {
    fn from(program: &'o CompiledProgram<'c>) -> Self {
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
        ArrayType, DataType, Dimension as CoreDimension, DimensionBounds, DimensionVariable, ExternalReferenceBinding,
        ReferenceSource, Shape,
    };
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::Error;
    use crate::domains::tests::*;
    use crate::experimental::lowering::signature::ExecutableSignature;

    use super::*;

    #[test]
    fn test_replacement_metadata_rejects_changed_invocation_contract() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
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
        assert!(matches!(current.validate_replacement(replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible kernel execution facts"));
        let requirements = [crate::domains::runtimes::builtin_requirement("test.runtime")];
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

        // Boundary types compare through the persistent encoding with canonical dimension-variable numbering:
        // structurally identical bounded-dynamic boundaries built from fresh dimension variables must be compatible
        // even though `ArrayType` equality is dimension-variable identity equality, while a genuinely different
        // boundary must be rejected.
        let dynamic_boundary = || {
            vec![ArrayType::new(
                DataType::F32,
                Shape::new(vec![CoreDimension::Dynamic(DimensionVariable::new(
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
        assert!(left.validate_replacement(right).is_ok());
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
