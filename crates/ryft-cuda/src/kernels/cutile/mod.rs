//! Verified portable kernels compiled through the official cuTile Python AOT interface.
//!
//! Physical array arguments follow `cutile_python_v2`: pointer, all shape components, then all element strides.
//! Static constraints do not remove shape or stride arguments. Writable references bind to the execution integration's
//! result buffers; repeated read-only arguments may alias. No compiler is selected implicitly.
//!
//! The `cutile` feature exposes artifact validation and physical argument mappings. Enable `cutile-compiler` to
//! generate source and invoke the pinned Python compiler. Compilation owns no CUDA runtime resources; the shared
//! [`CudaKernelArtifact`] and launcher APIs retain artifact loading and execution responsibilities.

use std::path::PathBuf;
use std::time::Duration;

use ryft_core::kernels::{GridExecution, NoKernelExtension, VerifiedKernel};
use ryft_core::{
    ArrayType, DataType, Dimension, Layout, Memory, MeshAxisType, ProgramError, ShardingDimension, TypeError, Typed,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaKernelParameterType,
    CudaScalarType,
};

/// Compiler configuration, capability, subprocess, and artifact failures.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid target, resource bound, or compiler option.
    #[error("invalid cuTile configuration: {message}")]
    Invalid {
        /// Specific configuration failure.
        message: String,
    },

    /// Canonical semantics cannot be lowered by this adapter.
    #[error("cuTile cannot lower `{operation}`: {reason}")]
    Unsupported {
        /// Canonical operation requiring support.
        operation: &'static str,
        /// Semantic contract absent from this compiler.
        reason: String,
    },

    /// A compiler subprocess failed, timed out, or was cancelled. Both streams are retained, including partial output.
    #[error("cuTile compiler {reason}\nstdout:\n{stdout}\nstderr:\n{stderr}")]
    Tool {
        /// Termination or validation cause.
        reason: String,
        /// Retained standard output, including partial output on failure.
        stdout: String,
        /// Retained diagnostic output, including partial output on failure.
        stderr: String,
    },

    /// The configured tool could not be started.
    #[error("cuTile tool `{path}` is unavailable: {source}")]
    Unavailable {
        /// Explicit executable that could not be started.
        path: PathBuf,
        /// Operating-system launch failure.
        source: std::io::Error,
    },

    /// A required Python compiler distribution is absent.
    #[error("cuTile compiler dependency is unavailable: {message}")]
    MissingDependency {
        /// Missing distribution reported by the worker.
        message: String,
    },

    /// Installed packages do not implement the pinned contract.
    #[error("incompatible cuTile compiler: {message}")]
    Incompatible {
        /// Observed version or platform mismatch.
        message: String,
    },

    /// Tool output does not agree with the requested canonical ABI and artifact.
    #[error("invalid cuTile artifact: {message}")]
    Artifact {
        /// Exact failed artifact invariant.
        message: String,
    },

    /// File or child-process management failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Manifest decoding failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// Canonical program access failed.
    #[error(transparent)]
    Program(#[from] ProgramError),

    /// Canonical type or exact semantic-key validation failed.
    #[error(transparent)]
    Type(#[from] TypeError),

    /// Canonical CUDA artifact validation failed.
    #[error(transparent)]
    Cuda(#[from] crate::Error),
}

#[cfg(feature = "cutile-compiler")]
mod compiler;
#[cfg(feature = "cutile-compiler")]
mod lowering;
#[cfg(feature = "cutile-compiler")]
pub use compiler::Compiler;

/// Supported Python distribution version; changing this requires requalifying the source and AOT contract.
pub const CUDA_TILE_VERSION: &str = "1.5.0";
/// Supported native TileIR assembler distribution version.
pub const TILEIRAS_VERSION: &str = "13.3.36";
/// Supported CUDA compiler companion distribution version.
pub const NVCC_VERSION: &str = "13.3.73";
/// Supported NVVM companion distribution version.
pub const NVVM_VERSION: &str = "13.3.73";
/// Version of the source generator and validated output contract.
pub const COMPILER_SCHEMA_VERSION: u32 = 1;

/// Exact architecture, independent of any runtime device or CUDA context.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Target {
    /// CUDA architecture number, for example `(12, 1)`.
    compute_capability: (u32, u32),
}

impl Target {
    /// Constructs an architecture admitted by the pinned compiler contract.
    pub fn new(major: u32, minor: u32) -> Result<Self, Error> {
        if !matches!((major, minor), (10, 0) | (10, 3) | (11, 0) | (12, 0) | (12, 1)) {
            return Err(Error::Invalid { message: format!("unsupported compute capability `{major}.{minor}`") });
        }
        Ok(Self { compute_capability: (major, minor) })
    }

    /// Returns the exact CUDA compute capability.
    pub fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }

    /// Returns the target spelling consumed by the pinned AOT exporter.
    pub fn architecture(&self) -> String {
        format!("sm_{}{}", self.compute_capability.0, self.compute_capability.1)
    }
}

/// Bounded source construction and tool invocation controls.
#[derive(Clone, Debug)]
pub struct Options {
    /// Maximum admitted canonical instruction and literal-element work.
    maximum_instructions: usize,
    /// Inner native compiler timeout, shorter than the process deadline.
    compiler_timeout: Duration,
    /// Total wall-clock deadline for each child, including Python startup.
    process_timeout: Duration,
    /// Maximum bytes retained from each child output stream.
    maximum_diagnostic_bytes: usize,
    /// Maximum accepted cubin size, checked before reading it.
    maximum_artifact_bytes: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            maximum_instructions: 16_384,
            compiler_timeout: Duration::from_secs(120),
            process_timeout: Duration::from_secs(150),
            maximum_diagnostic_bytes: 1024 * 1024,
            maximum_artifact_bytes: 64 * 1024 * 1024,
        }
    }
}

impl Options {
    /// Returns the source construction budget.
    pub fn maximum_instructions(&self) -> usize {
        self.maximum_instructions
    }

    /// Returns the native compiler timeout.
    pub fn compiler_timeout(&self) -> Duration {
        self.compiler_timeout
    }

    /// Returns the outer child-process deadline.
    pub fn process_timeout(&self) -> Duration {
        self.process_timeout
    }

    /// Returns the maximum retained bytes per output stream.
    pub fn maximum_diagnostic_bytes(&self) -> usize {
        self.maximum_diagnostic_bytes
    }

    /// Returns the maximum accepted cubin size.
    pub fn maximum_artifact_bytes(&self) -> usize {
        self.maximum_artifact_bytes
    }

    /// Bounds admitted instructions and literal expansion.
    pub fn with_maximum_instructions(mut self, maximum: usize) -> Result<Self, Error> {
        if maximum == 0 {
            return Err(Error::Invalid { message: "instruction budget must be positive".into() });
        }
        self.maximum_instructions = maximum;
        Ok(self)
    }

    /// Sets native and outer timeouts. The outer deadline must exceed the nonzero native timeout.
    pub fn with_timeouts(mut self, compiler: Duration, process: Duration) -> Result<Self, Error> {
        if compiler.is_zero() || process <= compiler {
            return Err(Error::Invalid { message: "process timeout must exceed the positive compiler timeout".into() });
        }
        self.compiler_timeout = compiler;
        self.process_timeout = process;
        Ok(self)
    }
}

/// One physical driver argument. Array indices name canonical kernel parameters, not flattened input/result slots.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Argument {
    /// Device pointer resolved through the canonical parameter's access mode.
    Array(usize),

    /// Static signed shape or element-stride component.
    I32(i32),
}

/// Immutable compiler output. Construction is private so mappings cannot become detached from validated artifact ABI.
#[derive(Clone, Debug)]
pub struct CompiledKernel {
    /// Canonical CUDA binary and launch contract.
    artifact: CudaKernelArtifact,
    /// Validated physical argument sequence.
    arguments: Vec<Argument>,
    /// Canonical full array types in logical parameter order.
    parameter_types: Vec<ArrayType>,
    /// Exact producer compatibility key, including installed tool versions.
    configuration_key: Vec<u8>,
    /// Exact body identity checked during artifact reconstruction.
    semantic_key: String,
    /// Adapter-authored manifest retained for runtime-only reload.
    manifest: Vec<u8>,
    /// Successful compiler diagnostics retained for inspection.
    diagnostics: String,
}

impl CompiledKernel {
    /// Reconstructs an artifact from an adapter-authored manifest and cubin without invoking Python.
    ///
    /// The caller supplies the verified body whose ABI and exact semantic identity must agree with the manifest.
    /// Integrity checks detect corruption and mismatched metadata; they do not authenticate untrusted native code.
    /// Only trusted compiler output may subsequently be executed.
    pub fn from_manifest(
        kernel: &VerifiedKernel<'_, NoKernelExtension>,
        manifest: &[u8],
        cubin: Vec<u8>,
    ) -> Result<Self, Error> {
        if manifest.len() > 1024 * 1024 || cubin.len() > 64 * 1024 * 1024 {
            return Err(Error::Artifact { message: "manifest or cubin exceeds the artifact size limit".into() });
        }
        let parsed: Manifest = serde_json::from_slice(manifest)?;
        let parameter_types = kernel
            .definition()
            .operation()
            .parameters()
            .iter()
            .map(|parameter| parameter.r#type().into_owned())
            .collect::<Vec<_>>();
        let parameters = parameter_types.iter().map(Parameter::from_type).collect::<Result<Vec<_>, _>>()?;
        let semantic_key = kernel.definition().semantic_key()?;
        let configuration: Configuration = serde_json::from_slice(&parsed.configuration_key)?;
        let target = Target::new(configuration.target.compute_capability.0, configuration.target.compute_capability.1)?;
        let grid = launch_grid(kernel)?;
        if parsed.schema != COMPILER_SCHEMA_VERSION
            || parsed.versions.cuda_tile != CUDA_TILE_VERSION
            || parsed.versions.tileiras != TILEIRAS_VERSION
            || parsed.versions.nvcc != NVCC_VERSION
            || parsed.versions.nvvm != NVVM_VERSION
            || parsed.versions.binary_version != TILEIRAS_VERSION
            || parsed.semantic_key != semantic_key
            || parsed.symbol != "ryft_cutile_kernel"
            || parsed.calling_convention != "cutile_python_v2"
            || parsed.parameters != parameters
            || parsed.block != [1, 1, 1]
            || parsed.shared_memory_bytes != 0
            || parsed.size_bytes != cubin.len()
            || parsed.sha256 != format!("{:x}", Sha256::digest(&cubin))
            || parsed.grid != grid
            || parsed.target != target.architecture()
            || configuration.schema != COMPILER_SCHEMA_VERSION
            || configuration.cuda_tile != CUDA_TILE_VERSION
            || configuration.tileiras != TILEIRAS_VERSION
            || configuration.nvcc != NVCC_VERSION
            || configuration.nvvm != NVVM_VERSION
            || configuration.worker_sha256 != format!("{:x}", Sha256::digest(include_bytes!("export.py")))
        {
            return Err(Error::Artifact {
                message: "manifest does not match the kernel, ABI, toolchain, or bytes".into(),
            });
        }
        let mut arguments = Vec::new();
        let mut abi = Vec::new();
        for (index, parameter) in parameters.iter().enumerate() {
            arguments.push(Argument::Array(index));
            abi.push(CudaKernelParameterType::DevicePointer);
            for value in parameter.shape.iter().chain(&parameter.strides) {
                arguments.push(Argument::I32(*value));
                abi.push(CudaKernelParameterType::Scalar(CudaScalarType::I32));
            }
        }
        let artifact = CudaKernelArtifact::new(
            CudaArtifactFormat::Cubin,
            cubin,
            parsed.symbol,
            parsed.target,
            CudaKernelLaunchDimensions::new(parsed.grid, [1, 1, 1], 0)?,
            CudaKernelAbi::new("cutile_python_v2", 2, abi)?,
        )?;
        Ok(Self {
            artifact,
            arguments,
            parameter_types,
            configuration_key: parsed.configuration_key,
            semantic_key,
            manifest: manifest.to_vec(),
            diagnostics: String::new(),
        })
    }

    /// Returns the exact checked portable body identity.
    pub fn semantic_key(&self) -> &str {
        &self.semantic_key
    }

    /// Returns the adapter-authored manifest for Python-free artifact reload.
    pub fn manifest(&self) -> &[u8] {
        &self.manifest
    }

    /// Returns the shared runtime's validated artifact.
    pub fn artifact(&self) -> &CudaKernelArtifact {
        &self.artifact
    }

    /// Returns the physical argument mapping into canonical parameter order.
    pub fn arguments(&self) -> &[Argument] {
        &self.arguments
    }

    /// Returns the full canonical parameter types used to construct the signature.
    pub fn parameter_types(&self) -> &[ArrayType] {
        &self.parameter_types
    }

    /// Returns the producer configuration identity established before compilation.
    pub fn configuration_key(&self) -> &[u8] {
        &self.configuration_key
    }

    /// Returns retained successful subprocess output.
    pub fn diagnostics(&self) -> &str {
        &self.diagnostics
    }
}

/// Explicit array constraints, shared by source generation and runtime artifact reconstruction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Parameter {
    /// Public cuTile dtype attribute.
    pub dtype: String,
    /// Static shape in signed ABI units.
    pub shape: Vec<i32>,
    /// Dense element strides in signed ABI units.
    pub strides: Vec<i32>,
}

impl Parameter {
    /// Builds pointer-independent constraints from canonical static array metadata.
    pub(super) fn from_type(array_type: &ArrayType) -> Result<Self, Error> {
        if array_type.memory() != Memory::Device
            || array_type.layout().is_some_and(|layout| !matches!(layout, Layout::Tiled(layout)
                if layout.tiles().is_empty() && layout.minor_to_major().iter().copied().eq((0..array_type.rank()).rev()))) {
            return Err(Error::Unsupported { operation: "parameter", reason: "requires dense row-major device arrays".into() });
        }
        if let Some(sharding) = array_type.sharding() {
            let manual = |name: &String| sharding.mesh().axis_type(name) == Some(MeshAxisType::Manual);
            let dimensions_local = sharding.dimensions().iter().all(|dimension| match dimension {
                ShardingDimension::Replicated => true,
                ShardingDimension::Sharded(names) => names.iter().all(manual),
                ShardingDimension::Unconstrained => false,
            });
            if !dimensions_local
                || !sharding.varying_manual_axes().iter().all(manual)
                || !sharding.unreduced_axes().iter().all(manual)
                || !sharding.reduced_axes().iter().all(manual)
            {
                return Err(Error::Unsupported {
                    operation: "parameter",
                    reason: "requires local manual-only sharding".into(),
                });
            }
        }
        let dtype = data_type_name(array_type.data_type())
            .ok_or_else(|| Error::Unsupported {
                operation: "parameter",
                reason: format!("unsupported dtype `{}`", array_type.data_type()),
            })?
            .to_owned();
        let shape = array_type.static_shape().ok_or_else(|| Error::Unsupported {
            operation: "parameter",
            reason: "requires static array shapes".into(),
        })?;
        if shape.dimensions().contains(&0) {
            return Err(Error::Unsupported {
                operation: "parameter",
                reason: "zero-sized external arrays have no admitted CUDA pointer contract".into(),
            });
        }
        let shape = shape
            .dimensions()
            .iter()
            .map(|&dimension| {
                i32::try_from(dimension)
                    .map_err(|_| Error::Invalid { message: "array shape exceeds the signed I32 ABI".into() })
            })
            .collect::<Result<Vec<_>, _>>()?;
        // The public AOT partition-view contract requires positive rank. A scalar is one physical element;
        // canonical logical types and source values retain rank zero throughout the execution integration.
        let shape = if shape.is_empty() { vec![1] } else { shape };
        let mut strides = vec![1; shape.len()];
        let mut stride = 1i32;
        for (index, &extent) in shape.iter().enumerate().rev() {
            strides[index] = stride;
            stride = stride
                .checked_mul(extent)
                .ok_or_else(|| Error::Invalid { message: "array size exceeds the signed I32 ABI".into() })?;
        }
        Ok(Self { dtype, shape, strides })
    }
}

/// Single supported dtype spelling table shared by body lowering and physical array constraints.
fn data_type_name(data_type: DataType) -> Option<&'static str> {
    match data_type {
        DataType::Boolean => Some("bool_"),
        DataType::I32 => Some("int32"),
        DataType::U32 => Some("uint32"),
        DataType::I64 => Some("int64"),
        DataType::U64 => Some("uint64"),
        DataType::F16 => Some("float16"),
        DataType::BF16 => Some("bfloat16"),
        DataType::F32 => Some("float32"),
        DataType::F64 => Some("float64"),
        _ => None,
    }
}

/// Actual producer versions, rechecked before invoking the exporter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Versions {
    /// Python frontend distribution.
    cuda_tile: String,
    /// Native TileIR compiler distribution.
    tileiras: String,
    /// CUDA compiler companion distribution version.
    nvcc: String,
    /// NVVM companion distribution version.
    nvvm: String,
    /// Actual selected TileIR compiler executable version.
    binary_version: String,
    /// Python interpreter version.
    python: String,
}

/// Adapter-authored manifest, not an upstream cuTile compiler output format.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    /// Adapter manifest schema version.
    schema: u32,
    /// Actual compiler installation.
    versions: Versions,
    /// Exact canonical body identity.
    semantic_key: String,
    /// Pinned producer and target identity.
    configuration_key: Vec<u8>,
    /// Requested native GPU architecture.
    target: String,
    /// Exported entry symbol.
    symbol: String,
    /// Versioned public AOT argument convention.
    calling_convention: String,
    /// Explicit array constraints in canonical parameter order.
    parameters: Vec<Parameter>,
    /// Physical tile grid.
    grid: [u32; 3],
    /// Required CUDA Tile launch block dimensions.
    block: [u32; 3],
    /// Dynamic shared-memory launch size.
    shared_memory_bytes: u32,
    /// Expected cubin byte count.
    size_bytes: usize,
    /// Expected cubin digest.
    sha256: String,
}

/// Producer identity serialized independently of process controls and invocation values.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Configuration {
    /// Source generator and contract schema.
    schema: u32,
    /// Pinned frontend distribution version.
    cuda_tile: String,
    /// Pinned native assembler distribution version.
    tileiras: String,
    /// CUDA compiler companion version.
    nvcc: String,
    /// NVVM companion version.
    nvvm: String,
    /// Exact compiler target.
    target: Target,
    /// Explicit compiler installation path.
    python: PathBuf,
    /// Requested result-preserving pipeline hint.
    pipeline_stages: Option<usize>,
    /// Requested result-preserving buffering hint.
    buffering_depth: Option<usize>,
    /// Explicit scratch resource bound.
    maximum_scratch_bytes: Option<usize>,
    /// Exact bundled exporter identity.
    worker_sha256: String,
}

/// Computes the physical tile grid shared by lowering and runtime manifest verification.
fn launch_grid(kernel: &VerifiedKernel<'_, NoKernelExtension>) -> Result<[u32; 3], Error> {
    let mut parallel = 1usize;
    for dimension in kernel.definition().operation().grid().dimensions() {
        let Dimension::Static(extent) = dimension.extent() else {
            return Err(Error::Unsupported {
                operation: "kernel_call",
                reason: "requires a statically specialized grid".into(),
            });
        };
        if dimension.execution() == GridExecution::Parallel {
            parallel = parallel.checked_mul(*extent).filter(|value| *value <= i32::MAX as usize).ok_or_else(|| {
                Error::Invalid { message: "parallel grid exceeds the signed native launch range".into() }
            })?;
        }
    }
    Ok([parallel.max(1) as u32, 1, 1])
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::kernels::{
        Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, whole_array_parameter,
    };
    use ryft_core::{ReferenceRead, ReferenceWrite};

    use super::*;

    /// Creates a verified mutable tile with identical ABI and two distinguishable effectful bodies.
    pub(super) fn definition(twice: bool) -> KernelDefinition {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::I32, [2, 3]), KernelParameterAccess::ReadWrite)
                    .unwrap(),
            ],
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, _)| {
            let value = references[0].read()?;
            references[0].write(&value)?;
            if twice {
                references[0].write(&value)?;
            }
            Ok(())
        })
        .unwrap()
    }

    /// Minimal CUDA ELF metadata fixture; it is never loaded or executed.
    pub(super) fn cubin() -> Vec<u8> {
        let mut bytes = vec![0; 64];
        bytes[..6].copy_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1]);
        bytes[18..20].copy_from_slice(&190u16.to_le_bytes());
        bytes[48..52].copy_from_slice(&121u32.to_le_bytes());
        bytes
    }

    /// Builds adapter-owned metadata independently of the exporter subprocess.
    pub(super) fn manifest(kernel: &VerifiedKernel<'_>, bytes: &[u8]) -> Manifest {
        let target = Target::new(12, 1).unwrap();
        let configuration = Configuration {
            schema: 1,
            cuda_tile: CUDA_TILE_VERSION.into(),
            tileiras: TILEIRAS_VERSION.into(),
            nvcc: NVCC_VERSION.into(),
            nvvm: NVVM_VERSION.into(),
            target,
            python: "/explicit/python".into(),
            pipeline_stages: None,
            buffering_depth: None,
            maximum_scratch_bytes: None,
            worker_sha256: format!("{:x}", Sha256::digest(include_bytes!("export.py"))),
        };
        Manifest {
            schema: 1,
            versions: Versions {
                cuda_tile: CUDA_TILE_VERSION.into(),
                tileiras: TILEIRAS_VERSION.into(),
                nvcc: NVCC_VERSION.into(),
                nvvm: NVVM_VERSION.into(),
                binary_version: TILEIRAS_VERSION.into(),
                python: "3.12.3".into(),
            },
            semantic_key: kernel.definition().semantic_key().unwrap(),
            configuration_key: serde_json::to_vec(&configuration).unwrap(),
            target: "sm_121".into(),
            symbol: "ryft_cutile_kernel".into(),
            calling_convention: "cutile_python_v2".into(),
            parameters: vec![Parameter { dtype: "int32".into(), shape: vec![2, 3], strides: vec![3, 1] }],
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_memory_bytes: 0,
            size_bytes: bytes.len(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    }

    #[test]
    fn test_target_new() {
        assert_eq!(Target::new(12, 1).unwrap().architecture(), "sm_121");
        assert_eq!(Target::new(10, 0).unwrap().compute_capability(), (10, 0));
        assert_eq!(
            Target::new(9, 9).unwrap_err().to_string(),
            "invalid cuTile configuration: unsupported compute capability `9.9`"
        );
    }

    #[test]
    fn test_options_with_timeouts() {
        let options = Options::default().with_timeouts(Duration::from_millis(1), Duration::from_millis(2)).unwrap();
        assert_eq!(options.process_timeout(), Duration::from_millis(2));
        for (inner, outer) in [(0, 1), (1, 1), (2, 1)] {
            assert_eq!(
                Options::default()
                    .with_timeouts(Duration::from_secs(inner), Duration::from_secs(outer))
                    .unwrap_err()
                    .to_string(),
                "invalid cuTile configuration: process timeout must exceed the positive compiler timeout"
            );
        }
    }

    #[test]
    fn test_parameter_from_type() {
        assert_eq!(
            Parameter::from_type(&ArrayType::new_static(DataType::F32, [2, 3])).unwrap(),
            Parameter { dtype: "float32".into(), shape: vec![2, 3], strides: vec![3, 1] }
        );
        assert_eq!(
            Parameter::from_type(&ArrayType::scalar(DataType::I64)).unwrap(),
            Parameter { dtype: "int64".into(), shape: vec![1], strides: vec![1] }
        );
        assert_eq!(
            Parameter::from_type(&ArrayType::new_static(DataType::I32, [i32::MAX as usize, 2]))
                .unwrap_err()
                .to_string(),
            "invalid cuTile configuration: array size exceeds the signed I32 ABI"
        );
    }

    #[test]
    fn test_parameter_from_type_empty() {
        assert_eq!(
            Parameter::from_type(&ArrayType::new_static(DataType::F32, [2, 0])).unwrap_err().to_string(),
            "cuTile cannot lower `parameter`: zero-sized external arrays have no admitted CUDA pointer contract"
        );
    }

    #[test]
    fn test_compiled_kernel_from_manifest() {
        let definition = definition(false);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let bytes = cubin();
        let manifest = serde_json::to_vec(&manifest(&kernel, &bytes)).unwrap();
        let output = CompiledKernel::from_manifest(&kernel, &manifest, bytes.clone()).unwrap();
        assert_eq!(
            output.arguments(),
            &[Argument::Array(0), Argument::I32(2), Argument::I32(3), Argument::I32(3), Argument::I32(1)]
        );
        assert_eq!(
            output.artifact().abi().parameters(),
            &[
                CudaKernelParameterType::DevicePointer,
                CudaKernelParameterType::Scalar(CudaScalarType::I32),
                CudaKernelParameterType::Scalar(CudaScalarType::I32),
                CudaKernelParameterType::Scalar(CudaScalarType::I32),
                CudaKernelParameterType::Scalar(CudaScalarType::I32)
            ]
        );
        assert_eq!(output.parameter_types(), &[ArrayType::new_static(DataType::I32, [2, 3])]);
        assert_eq!(output.artifact().bytes(), bytes);
        assert_eq!(output.manifest(), manifest);
        let reloaded =
            CompiledKernel::from_manifest(&kernel, output.manifest(), output.artifact().bytes().to_vec()).unwrap();
        assert_eq!(reloaded.configuration_key(), output.configuration_key());
    }

    #[test]
    fn test_compiled_kernel_from_manifest_scalar_abi() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            references[0].write(&references[0].read()?)?;
            Ok(())
        })
        .unwrap();
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let bytes = cubin();
        let mut metadata = manifest(&kernel, &bytes);
        metadata.parameters = vec![Parameter { dtype: "int32".into(), shape: vec![1], strides: vec![1] }];
        let output = CompiledKernel::from_manifest(&kernel, &serde_json::to_vec(&metadata).unwrap(), bytes).unwrap();
        assert_eq!(output.parameter_types(), &[ArrayType::scalar(DataType::I32)]);
        assert_eq!(output.arguments(), &[Argument::Array(0), Argument::I32(1), Argument::I32(1)]);
        assert_eq!(
            output.artifact().abi().parameters(),
            &[
                CudaKernelParameterType::DevicePointer,
                CudaKernelParameterType::Scalar(CudaScalarType::I32),
                CudaKernelParameterType::Scalar(CudaScalarType::I32)
            ]
        );
    }

    #[test]
    fn test_compiled_kernel_from_manifest_rejects_mismatch() {
        let definition = definition(false);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let bytes = cubin();
        for field in ["semantic_key", "target", "calling_convention", "sha256"] {
            let mut value = serde_json::to_value(manifest(&kernel, &bytes)).unwrap();
            value[field] = "changed".into();
            assert!(matches!(
                CompiledKernel::from_manifest(&kernel, &serde_json::to_vec(&value).unwrap(), bytes.clone()),
                Err(Error::Artifact { .. })
            ));
        }
        for field in ["grid", "block"] {
            let mut value = serde_json::to_value(manifest(&kernel, &bytes)).unwrap();
            value[field] = serde_json::json!([2, 1, 1]);
            assert!(matches!(
                CompiledKernel::from_manifest(&kernel, &serde_json::to_vec(&value).unwrap(), bytes.clone()),
                Err(Error::Artifact { .. })
            ));
        }
        let different = super::tests::definition(true);
        let different = VerifiedKernel::new(&different, 1).unwrap();
        assert!(matches!(
            CompiledKernel::from_manifest(&different, &serde_json::to_vec(&manifest(&kernel, &bytes)).unwrap(), bytes),
            Err(Error::Artifact { .. })
        ));
    }
    #[test]
    fn test_parameter_from_type_manual_sharding() {
        use ryft_core::{LogicalMesh, MeshAxis, Sharding};
        let base = ArrayType::new_static(DataType::F32, [4]);
        for axis_type in [MeshAxisType::Manual, MeshAxisType::Auto, MeshAxisType::Explicit] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
            let array_type = base.clone().with_sharding(sharding).unwrap();
            if axis_type == MeshAxisType::Manual {
                assert_eq!(Parameter::from_type(&array_type).unwrap(), Parameter::from_type(&base).unwrap());
                assert_eq!(array_type.sharding().unwrap().mesh().axis_type("x"), Some(MeshAxisType::Manual));
            } else {
                assert_eq!(
                    Parameter::from_type(&array_type).unwrap_err().to_string(),
                    "cuTile cannot lower `parameter`: requires local manual-only sharding"
                );
            }
        }
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1).with_varying_manual_axes(["x"]).unwrap();
        assert_eq!(
            Parameter::from_type(&base.clone().with_sharding(sharding).unwrap()).unwrap(),
            Parameter::from_type(&base).unwrap()
        );
    }
}
