//! Lowered and compiled XLA artifacts and their invocation contracts.

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
use crate::experimental::lowering::signature::ExecutableSignature;

// TODO(eaplatanios): Review this module.

/// StableHLO lowering and call-boundary metadata produced before PJRT compilation.
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

    /// Returns the device mesh this lowering targets.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

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
#[derive(Clone)]
pub struct CompiledProgram<'c> {
    pub(super) executable: Arc<LoadedExecutable<'c>>,
    pub(super) input_types: Arc<[ArrayType]>,
    pub(super) output_types: Arc<[ArrayType]>,
    pub(super) output_count: usize,
    pub(super) reference_states: Arc<[ExternalReferenceBinding]>,
    pub(super) signature: ExecutableSignature,
    pub(super) requires_assertion_handler: bool,
    pub(super) donation_flags: Arc<[bool]>,
    pub(super) capture_count: usize,
    pub(super) expected_argument_shardings: Arc<[Sharding]>,
    pub(super) mesh: DeviceMesh,
    pub(super) compilation_options: CompilationOptions,
    pub(super) platform_name: Arc<str>,
    pub(super) platform_version: Arc<str>,
    pub(super) device_kinds: Arc<[String]>,

    /// Exact live execution facts for selected kernels; ordinary executables leave this absent.
    pub(super) kernel_execution_facts: Option<Vec<u8>>,
    pub(super) compiler_identity: Arc<str>,
    pub(super) xla_flags: Arc<str>,
    pub(super) compilation_duration: Option<Duration>,
    pub(super) analysis: Arc<OnceLock<Result<CompilationAnalysis, Error>>>,

    /// Value-independent execution metadata, computed once when the program is compiled or loaded.
    pub(super) plan: Arc<ExecutionPlan>,
}

#[derive(Copy, Clone)]
pub(super) struct InvocationMetadata<'a> {
    input_types: &'a [ArrayType],
    output_types: &'a [ArrayType],
    output_count: usize,
    reference_states: &'a [ExternalReferenceBinding],
    signature: &'a ExecutableSignature,
    donation_flags: &'a [bool],
    capture_count: usize,
    expected_argument_shardings: &'a [Sharding],
    mesh: &'a DeviceMesh,
    requires_assertion_handler: bool,
    kernel_execution_facts: Option<&'a [u8]>,
}

impl<'a, 'c> From<&'a CompiledProgram<'c>> for InvocationMetadata<'a> {
    fn from(program: &'a CompiledProgram<'c>) -> Self {
        Self {
            input_types: &program.input_types,
            output_types: &program.output_types,
            output_count: program.output_count,
            reference_states: &program.reference_states,
            signature: &program.signature,
            donation_flags: &program.donation_flags,
            capture_count: program.capture_count,
            expected_argument_shardings: &program.expected_argument_shardings,
            mesh: &program.mesh,
            requires_assertion_handler: program.requires_assertion_handler,
            kernel_execution_facts: program.kernel_execution_facts.as_deref(),
        }
    }
}

impl<'a> From<&'a LoweredProgram> for InvocationMetadata<'a> {
    fn from(program: &'a LoweredProgram) -> Self {
        Self {
            input_types: &program.input_types,
            output_types: &program.output_types,
            output_count: program.output_count,
            reference_states: &program.reference_states,
            signature: &program.signature,
            donation_flags: &program.donation_flags,
            capture_count: program.capture_count,
            expected_argument_shardings: &program.expected_argument_shardings,
            mesh: &program.mesh,
            requires_assertion_handler: program.requires_assertion_handler,
            kernel_execution_facts: program.kernel_execution_facts.as_deref(),
        }
    }
}

/// Returns the first invocation-metadata field on which the two programs disagree, if any.
pub(super) fn incompatible_invocation_field(
    current: InvocationMetadata<'_>,
    other: InvocationMetadata<'_>,
) -> Option<&'static str> {
    // Structurally equal boundary types have equal canonical encodings, so only unequal ones pay for encoding both.
    let equal_boundary_types = current.input_types == other.input_types && current.output_types == other.output_types;
    if !equal_boundary_types
        && canonical_boundary_types(current.input_types, current.output_types)
            != canonical_boundary_types(other.input_types, other.output_types)
    {
        Some("boundary types")
    } else if current.output_count != other.output_count {
        Some("output count")
    } else if current.reference_states != other.reference_states {
        Some("reference-state signature")
    } else if current.signature != other.signature {
        Some("executable signature")
    } else if current.donation_flags != other.donation_flags {
        Some("donation flags")
    } else if current.capture_count != other.capture_count {
        Some("capture count")
    } else if current.expected_argument_shardings != other.expected_argument_shardings {
        Some("argument shardings")
    } else if current.mesh != other.mesh {
        Some("device mesh")
    } else if current.kernel_execution_facts != other.kernel_execution_facts {
        Some("kernel execution facts")
    } else if current.requires_assertion_handler != other.requires_assertion_handler {
        Some("assertion-handler requirement")
    } else {
        None
    }
}

/// Validates that `replacement` preserves `current`'s complete invocation contract, naming the first incompatible
/// metadata field otherwise.
pub(super) fn validate_replacement_metadata(
    current: InvocationMetadata<'_>,
    replacement: InvocationMetadata<'_>,
) -> Result<(), Error> {
    match incompatible_invocation_field(current, replacement) {
        Some(field) => Err(Error::Compilation(CompilationError::IncompatibleExecutable {
            reason: format!("replacement executable has incompatible {field}"),
        })),
        None => Ok(()),
    }
}

impl<'c> CompiledProgram<'c> {
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
            donation_flags: &[false],
            capture_count: 0,
            expected_argument_shardings: &[],
            mesh: &mesh,
            requires_assertion_handler: false,
            kernel_execution_facts: None,
        };
        let replacement = InvocationMetadata { kernel_execution_facts: Some(b"changed"), ..current };
        assert!(matches!(validate_replacement_metadata(current, replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible kernel execution facts"));
        let cuda_signature = signature.clone().with_cuda_kernel_runtime(true);
        let replacement = InvocationMetadata { signature: &cuda_signature, ..current };
        assert!(matches!(validate_replacement_metadata(current, replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible executable signature"));
        let rocm_signature = signature.clone().with_rocm_kernel_runtime(true);
        let replacement = InvocationMetadata { signature: &rocm_signature, ..current };
        assert!(matches!(validate_replacement_metadata(current, replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible executable signature"));
        let replacement = InvocationMetadata { donation_flags: &[true], ..current };

        assert!(matches!(
            validate_replacement_metadata(current, replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible donation flags",
        ));
        let replacement = InvocationMetadata { output_count: 1, ..current };
        assert!(matches!(
            validate_replacement_metadata(current, replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible output count",
        ));
        let state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None);
        let replacement = InvocationMetadata { reference_states: std::slice::from_ref(&state), ..current };
        assert!(matches!(
            validate_replacement_metadata(current, replacement),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible reference-state signature",
        ));
        let replacement = InvocationMetadata { requires_assertion_handler: true, ..current };
        assert!(matches!(
            validate_replacement_metadata(current, replacement),
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
        assert!(validate_replacement_metadata(left, right).is_ok());
        let static_types = vec![ArrayType::scalar(DataType::F32)];
        let left = InvocationMetadata { input_types: left_types.as_slice(), ..current };
        let different = InvocationMetadata { input_types: static_types.as_slice(), ..current };
        assert!(matches!(
            validate_replacement_metadata(left, different),
            Err(Error::Compilation(CompilationError::IncompatibleExecutable { reason }))
                if reason == "replacement executable has incompatible boundary types",
        ));
    }
}
