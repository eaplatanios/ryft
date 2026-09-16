//! Direct typed TTIR compilation of verified portable kernels.
//!
//! The adapter admits static dense F32 arrays, masked global windows, arithmetic, reductions and bounded control
//! flow. Floating-point dot uses IEEE input precision. Unsupported effects and memory contracts fail before native
//! compilation. The linked native bridge compiles without creating a GPU context. Artifact deployment uses the
//! corresponding runtime adapter.

use ryft_core::{ArrayType, ProgramError, TypeError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Configuration, portable admission, native compilation and artifact failures.
#[derive(Debug, Error)]
pub enum Error {
    /// An option, target or compiler installation violates the admitted contract.
    #[error("invalid Triton configuration: {message}")]
    Invalid {
        /// Failed invariant.
        message: String,
    },

    /// The linked native archive does not provide the requested compiler target.
    #[error("triton compiler is unavailable: {message}")]
    Unavailable {
        /// Required compiler capability.
        message: String,
    },

    /// A canonical operation requires semantics unavailable in this adapter.
    #[error("triton cannot lower `{operation}`: {reason}")]
    Unsupported {
        /// Canonical operation name.
        operation: &'static str,

        /// Missing semantic contract.
        reason: String,
    },

    /// Native compilation returned an error.
    #[error("triton compilation failed: {message}")]
    Compilation {
        /// Bounded native diagnostics.
        message: String,
    },

    /// Cancellation was requested before compilation or before accepting its result.
    #[error("triton compilation cancelled")]
    Cancelled,

    /// Compiler products disagree with the requested signature, target or resource contract.
    #[error("invalid Triton artifact: {message}")]
    Artifact {
        /// Failed artifact invariant.
        message: String,
    },

    /// Native metadata could not be decoded.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// Typed native construction failed.
    #[error(transparent)]
    Mlir(#[from] ryft_mlir::Error),

    /// Canonical program access failed.
    #[error(transparent)]
    Program(#[from] ProgramError),

    /// Canonical type validation failed.
    #[error(transparent)]
    Type(#[from] TypeError),

    /// Concrete CUDA artifact validation failed.
    #[error(transparent)]
    Cuda(#[from] ryft_cuda::Error),

    /// Concrete ROCm artifact validation failed.
    #[error(transparent)]
    Rocm(#[from] ryft_rocm::Error),
}

mod compiler;
mod lowering;

pub use compiler::Compiler;

/// Version of the typed lowering, native bridge and physical argument contract.
pub const COMPILER_SCHEMA_VERSION: u32 = 2;

/// Source revision that owns the native pipeline and its Triton patches.
pub const XLA_VERSION: &str = "eb6b90ed013f511eca088c52f541f3c0819f919e";

/// Triton revision selected by the pinned XLA workspace.
pub const TRITON_VERSION: &str = "a77e7c793abc0d0c923a9afb275058e2fe57a198";

/// Explicit compiler architecture. Construction creates no runtime context; admission validates the target.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Target {
    /// NVIDIA CUDA compute capability.
    Cuda {
        /// Major compute capability.
        major: u32,

        /// Minor compute capability.
        minor: u32,
    },

    /// AMD GPU architecture with a qualified native code-object contract.
    Rocm {
        /// Exact processor name, without feature overrides.
        architecture: String,
    },
}

impl Target {
    /// Checks the source and compiler-qualified architecture set.
    fn validate(&self) -> Result<(), Error> {
        match self {
            Self::Cuda { major, minor } if matches!((*major, *minor), (8, 0) | (12, 1)) => Ok(()),
            Self::Rocm { architecture } if matches!(architecture.as_str(), "gfx908" | "gfx90a" | "gfx942") => Ok(()),
            Self::Rocm { architecture } => {
                Err(Error::Invalid { message: format!("unqualified ROCm architecture `{architecture}`") })
            }
            Self::Cuda { major, minor } => {
                Err(Error::Invalid { message: format!("unqualified CUDA compute capability `{major}.{minor}`") })
            }
        }
    }
}

/// Compiler choices and bounded construction and output resources.
///
/// Warp count affects generated code and compiler identity. Size limits affect admission and failure behavior only.
/// Pipeline stages come from the canonical kernel schedule, defaulting to two.
#[derive(Clone, Debug)]
pub struct Options {
    /// Requested warps per program; final native metadata may require additional warps.
    warp_count: u32,

    /// Maximum instructions traversed while constructing TTIR.
    maximum_instructions: usize,

    /// Maximum physical elements in any power-of-two tile.
    maximum_tile_elements: usize,

    /// Maximum native artifact size.
    maximum_artifact_bytes: usize,

    /// Maximum bytes retained from native diagnostics.
    maximum_diagnostic_bytes: usize,
}

impl Options {
    /// Returns the requested number of warps per program.
    pub fn warp_count(&self) -> u32 {
        self.warp_count
    }

    /// Returns the construction instruction limit.
    pub fn maximum_instructions(&self) -> usize {
        self.maximum_instructions
    }

    /// Returns the physical tile element limit.
    pub fn maximum_tile_elements(&self) -> usize {
        self.maximum_tile_elements
    }

    /// Returns the maximum artifact size in bytes.
    pub fn maximum_artifact_bytes(&self) -> usize {
        self.maximum_artifact_bytes
    }

    /// Returns the diagnostic capture limit in bytes.
    pub fn maximum_diagnostic_bytes(&self) -> usize {
        self.maximum_diagnostic_bytes
    }

    /// Selects one, two, four or eight warps per program.
    pub fn with_warp_count(mut self, count: u32) -> Result<Self, Error> {
        if !matches!(count, 1 | 2 | 4 | 8) {
            return Err(Error::Invalid { message: "warp count must be one, two, four or eight".into() });
        }
        self.warp_count = count;
        Ok(self)
    }

    /// Sets finite positive construction limits before native allocation.
    pub fn with_construction_limits(mut self, instructions: usize, tile_elements: usize) -> Result<Self, Error> {
        if instructions == 0 || tile_elements == 0 || instructions > 1_000_000 || tile_elements > 1_048_576 {
            return Err(Error::Invalid {
                message: "construction limits must be positive and at most one million instructions \
                          and 1048576 tile elements"
                    .into(),
            });
        }
        self.maximum_instructions = instructions;
        self.maximum_tile_elements = tile_elements;
        Ok(self)
    }

    /// Sets bounded artifact and diagnostic capture sizes.
    pub fn with_output_limits(mut self, artifact_bytes: usize, diagnostic_bytes: usize) -> Result<Self, Error> {
        if artifact_bytes == 0
            || diagnostic_bytes == 0
            || artifact_bytes > 256 * 1024 * 1024
            || diagnostic_bytes > 8 * 1024 * 1024
        {
            return Err(Error::Invalid {
                message: "output limits must be positive and at most 256 MiB for artifacts and 8 MiB for diagnostics"
                    .into(),
            });
        }
        self.maximum_artifact_bytes = artifact_bytes;
        self.maximum_diagnostic_bytes = diagnostic_bytes;
        Ok(self)
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            warp_count: 4,
            maximum_instructions: 65_536,
            maximum_tile_elements: 65_536,
            maximum_artifact_bytes: 64 * 1024 * 1024,
            maximum_diagnostic_bytes: 1024 * 1024,
        }
    }
}

/// Validated device image and launch ABI owned by the corresponding runtime adapter.
#[derive(Clone, Debug)]
pub enum Artifact {
    /// NVIDIA PTX artifact.
    Cuda(ryft_cuda::CudaKernelArtifact),

    /// AMD HSA code object.
    Rocm(ryft_rocm::RocmKernelArtifact),
}

/// Validated concrete producer output. Each logical parameter corresponds to one physical global pointer.
#[derive(Clone, Debug)]
pub struct CompiledKernel {
    /// Immutable concrete image and physical ABI.
    artifact: Artifact,

    /// Canonical parameter types, in physical pointer order.
    parameter_types: Vec<ArrayType>,

    /// Exact portable definition identity.
    semantic_key: String,

    /// Compiler, target and schedule identity verified during compilation.
    configuration_key: Vec<u8>,

    /// Bounded native diagnostics.
    diagnostics: String,
}

impl CompiledKernel {
    /// Returns the concrete artifact used by its platform launcher.
    pub fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    /// Returns logical parameter types in physical argument order.
    pub fn parameter_types(&self) -> &[ArrayType] {
        &self.parameter_types
    }

    /// Returns the portable definition identity.
    pub fn semantic_key(&self) -> &str {
        &self.semantic_key
    }

    /// Returns exact compiler, target and schedule identity.
    pub fn configuration_key(&self) -> &[u8] {
        &self.configuration_key
    }

    /// Returns the compiler's bounded diagnostics.
    pub fn diagnostics(&self) -> &str {
        &self.diagnostics
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::DataType;
    use ryft_cuda::{
        CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaKernelParameterType,
    };

    use super::*;

    /// Constructs metadata without invoking a compiler or GPU runtime.
    fn output() -> CompiledKernel {
        CompiledKernel {
            artifact: Artifact::Cuda(
                CudaKernelArtifact::new(
                    CudaArtifactFormat::Ptx,
                    b".version 9.0".to_vec(),
                    "ryft_kernel",
                    "sm_121",
                    CudaKernelLaunchDimensions::new([1; 3], [128, 1, 1], 0).unwrap(),
                    CudaKernelAbi::new("triton global pointers", 1, vec![CudaKernelParameterType::DevicePointer])
                        .unwrap(),
                )
                .unwrap(),
            ),
            parameter_types: vec![ArrayType::new_static(DataType::F32, [1])],
            semantic_key: "semantic".into(),
            configuration_key: vec![1, 2],
            diagnostics: "diagnostic".into(),
        }
    }

    #[test]
    fn test_target_validate() {
        for architecture in ["gfx908", "gfx90a", "gfx942"] {
            assert!(Target::Rocm { architecture: architecture.into() }.validate().is_ok());
        }
        assert!(matches!(Target::Rocm { architecture: "gfx942:xnack+".into() }.validate(),
            Err(Error::Invalid { message }) if message == "unqualified ROCm architecture `gfx942:xnack+`"));
        assert!(Target::Cuda { major: 8, minor: 0 }.validate().is_ok());
        assert!(Target::Cuda { major: 12, minor: 1 }.validate().is_ok());
        assert!(matches!(Target::Cuda { major: 9, minor: 0 }.validate(), Err(Error::Invalid { message })
            if message == "unqualified CUDA compute capability `9.0`"));
    }

    #[test]
    fn test_options_warp_count() {
        assert_eq!(Options::default().warp_count(), 4);
    }

    #[test]
    fn test_options_maximum_instructions() {
        assert_eq!(Options::default().maximum_instructions(), 65_536);
    }

    #[test]
    fn test_options_maximum_tile_elements() {
        assert_eq!(Options::default().maximum_tile_elements(), 65_536);
    }

    #[test]
    fn test_options_maximum_artifact_bytes() {
        assert_eq!(Options::default().maximum_artifact_bytes(), 64 * 1024 * 1024);
    }

    #[test]
    fn test_options_maximum_diagnostic_bytes() {
        assert_eq!(Options::default().maximum_diagnostic_bytes(), 1024 * 1024);
    }

    #[test]
    fn test_options_with_warp_count() {
        assert_eq!(Options::default().with_warp_count(8).unwrap().warp_count(), 8);
        assert!(matches!(Options::default().with_warp_count(3), Err(Error::Invalid { message })
            if message == "warp count must be one, two, four or eight"));
    }

    #[test]
    fn test_options_with_construction_limits() {
        let options = Options::default().with_construction_limits(1_000_000, 1_048_576).unwrap();
        assert_eq!((options.maximum_instructions(), options.maximum_tile_elements()), (1_000_000, 1_048_576));
        for (instructions, elements) in [(0, 1), (1, 0), (1_000_001, 1), (1, 1_048_577)] {
            assert!(matches!(
                Options::default().with_construction_limits(instructions, elements),
                Err(Error::Invalid { message })
                    if message == "construction limits must be positive and at most one million instructions \
                                   and 1048576 tile elements"
            ));
        }
    }

    #[test]
    fn test_options_with_output_limits() {
        let options = Options::default().with_output_limits(256 * 1024 * 1024, 8 * 1024 * 1024).unwrap();
        assert_eq!(
            (options.maximum_artifact_bytes(), options.maximum_diagnostic_bytes()),
            (256 * 1024 * 1024, 8 * 1024 * 1024)
        );
        for (artifact, diagnostic) in [(0, 1), (1, 0), (256 * 1024 * 1024 + 1, 1), (1, 8 * 1024 * 1024 + 1)] {
            assert!(
                matches!(Options::default().with_output_limits(artifact, diagnostic), Err(Error::Invalid { message })
                if message == "output limits must be positive and at most 256 MiB for artifacts \
                               and 8 MiB for diagnostics")
            );
        }
    }

    #[test]
    fn test_compiled_kernel_artifact() {
        let output = output();
        let Artifact::Cuda(artifact) = output.artifact() else { panic!("expected CUDA artifact") };
        assert_eq!(artifact.symbol(), "ryft_kernel");
    }

    #[test]
    fn test_compiled_kernel_parameter_types() {
        assert_eq!(output().parameter_types(), &[ArrayType::new_static(DataType::F32, [1])]);
    }

    #[test]
    fn test_compiled_kernel_semantic_key() {
        assert_eq!(output().semantic_key(), "semantic");
    }

    #[test]
    fn test_compiled_kernel_configuration_key() {
        assert_eq!(output().configuration_key(), &[1, 2]);
    }

    #[test]
    fn test_compiled_kernel_diagnostics() {
        assert_eq!(output().diagnostics(), "diagnostic");
    }
}
