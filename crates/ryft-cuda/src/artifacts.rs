//! Immutable CUDA images, parameter ABIs, and launch resource metadata.

use std::fmt::{Display, Formatter};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::Error;

/// Compute capability reported by a CUDA device (e.g., `9.0` for Hopper or `10.0` for Blackwell).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct CudaComputeCapability {
    /// Major architecture revision.
    pub(super) major: u32,

    /// Minor architecture revision.
    pub(super) minor: u32,
}

impl CudaComputeCapability {
    /// Returns the SM number that names this capability in `sm_<N>` and `compute_<N>` targets (e.g., `90` for `9.0`).
    pub(super) fn number(self) -> u32 {
        self.major * 10 + self.minor
    }
}

impl Display for CudaComputeCapability {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}", self.major, self.minor)
    }
}

/// Binary representation carried by a [`CudaKernelArtifact`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CudaArtifactFormat {
    /// Device machine code packaged as an ELF image.
    Cubin,

    /// Virtual assembly compiled by the CUDA driver when loaded.
    Ptx,
}

/// Size in bytes of an ELF64 file header (`Elf64_Ehdr`).
const ELF64_HEADER_SIZE: usize = 64;

/// `e_machine` value that identifies NVIDIA CUDA objects (`EM_CUDA` in LLVM's `BinaryFormat/ELF.h`).
const ELF_MACHINE_CUDA: u16 = 190;

/// `EI_ABIVERSION` value of the CUDA ELF ABI introduced with Blackwell that stores the SM number in the second byte of
/// `e_flags` (`ELFABIVERSION_CUDA_V2` in LLVM's `BinaryFormat/ELF.h`). Older cubins use `ELFABIVERSION_CUDA_V1` (7)
/// and store the SM number in the low byte.
const ELF_ABI_VERSION_CUDA_V2: u8 = 8;

/// Returns the SM number recorded in the ELF64 header of a cubin.
///
/// A cubin is a little-endian ELF64 image whose `e_machine` is `EM_CUDA` (190). NVIDIA records the target SM number
/// in `e_flags`: for `EI_ABIVERSION == ELFABIVERSION_CUDA_V1` (7) it occupies the low byte (`EF_CUDA_SM = 0xff`, e.g.
/// `0x5a` for `sm_90`), and for `EI_ABIVERSION == ELFABIVERSION_CUDA_V2` (8) it occupies the second byte
/// (`EF_CUDA_SM_MASK = 0xff00`, e.g. `0x6400` for `sm_100`). Feature-specific variants such as `sm_90a` set separate
/// accelerator flag bits that this function ignores. The encoding follows the `EM_CUDA` handling of LLVM's
/// [`BinaryFormat/ELF.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/BinaryFormat/ELF.h)
/// and [`llvm-readobj`](https://github.com/llvm/llvm-project/blob/main/llvm/tools/llvm-readobj/ELFDumper.cpp).
fn cubin_elf_architecture(bytes: &[u8]) -> Result<u32, Error> {
    // ELF identification: magic, `EI_CLASS == ELFCLASS64`, and `EI_DATA == ELFDATA2LSB`.
    if bytes.len() < ELF64_HEADER_SIZE || bytes[..4] != [0x7f, b'E', b'L', b'F'] || bytes[4] != 2 || bytes[5] != 1 {
        return Err(Error::invalid_argument("cuda cubin bytes do not start with a little-endian ELF64 header"));
    }
    let machine = u16::from_le_bytes([bytes[18], bytes[19]]);
    if machine != ELF_MACHINE_CUDA {
        return Err(Error::invalid_argument(format!(
            "cuda cubin ELF header has `e_machine` {machine}, expected {ELF_MACHINE_CUDA} (NVIDIA CUDA)",
        )));
    }
    let flags = u32::from_le_bytes([bytes[48], bytes[49], bytes[50], bytes[51]]);
    // Cubins that predate the versioned CUDA ABIs use the `ELFABIVERSION_CUDA_V1` encoding.
    if bytes[8] == ELF_ABI_VERSION_CUDA_V2 { Ok((flags >> 8) & 0xff) } else { Ok(flags & 0xff) }
}

/// Producer-recorded target architecture parsed from strings such as `sm_100`, `compute_90`, or `sm_90a`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CudaTargetArchitecture {
    /// A real architecture (`sm_<N>`) that a cubin was assembled for.
    Real(u32),

    /// A virtual architecture (`compute_<N>`) that PTX was generated for.
    Virtual(u32),

    /// An architecture-specific or family-specific target whose restrictions are checked by CUDA.
    Restricted(u32),
}

impl CudaTargetArchitecture {
    /// Parses `sm_<N>` and `compute_<N>` targets, tolerating a trailing feature suffix such as `a` or `f`.
    /// Returns `None` for other representations so that unknown target encodings remain accepted.
    fn parse(target: &str) -> Option<Self> {
        let (constructor, digits): (fn(u32) -> Self, &str) = if let Some(digits) = target.strip_prefix("sm_") {
            (Self::Real, digits)
        } else if let Some(digits) = target.strip_prefix("compute_") {
            (Self::Virtual, digits)
        } else {
            return None;
        };
        let digit_count = digits.bytes().take_while(u8::is_ascii_digit).count();
        if digit_count == 0 || !digits[digit_count..].bytes().all(|byte| byte.is_ascii_lowercase()) {
            return None;
        }
        digits[..digit_count]
            .parse()
            .ok()
            .map(|number| if digit_count == digits.len() { constructor(number) } else { Self::Restricted(number) })
    }

    /// Returns the SM number of this target.
    fn number(self) -> u32 {
        match self {
            Self::Real(number) | Self::Virtual(number) | Self::Restricted(number) => number,
        }
    }
}

/// Scalar types accepted by CUDA kernel launch arguments.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CudaScalarType {
    /// Signed 8-bit integer.
    I8,

    /// Signed 16-bit integer.
    I16,

    /// Signed 32-bit integer.
    I32,

    /// Signed 64-bit integer.
    I64,

    /// Unsigned 8-bit integer.
    U8,

    /// Unsigned 16-bit integer.
    U16,

    /// Unsigned 32-bit integer.
    U32,

    /// Unsigned 64-bit integer.
    U64,

    /// Finite-only 4-bit floating-point value with 2 exponent bits and 1 mantissa bit.
    F4E2M1FN,

    /// Finite-only 6-bit floating-point value with 2 exponent bits and 3 mantissa bits.
    F6E2M3FN,

    /// Finite-only 6-bit floating-point value with 3 exponent bits and 2 mantissa bits.
    F6E3M2FN,

    /// 8-bit floating-point value with 3 exponent bits and 4 mantissa bits.
    F8E3M4,

    /// 8-bit floating-point value with 4 exponent bits and 3 mantissa bits.
    F8E4M3,

    /// Finite-only 8-bit floating-point value with 4 exponent bits and 3 mantissa bits.
    F8E4M3FN,

    /// Finite-only, unsigned-zero 8-bit floating-point value with 4 exponent bits and 3 mantissa bits.
    F8E4M3FNUZ,

    /// Finite-only, unsigned-zero 8-bit floating-point value with 4 exponent bits, 3 mantissa bits, and bias 11.
    F8E4M3B11FNUZ,

    /// 8-bit floating-point value with 5 exponent bits and 2 mantissa bits.
    F8E5M2,

    /// Finite-only, unsigned-zero 8-bit floating-point value with 5 exponent bits and 2 mantissa bits.
    F8E5M2FNUZ,

    /// Finite-only, unsigned 8-bit floating-point value with 8 exponent bits and no mantissa bits.
    F8E8M0FNU,

    /// 16-bit floating-point value with 8 exponent bits and 7 mantissa bits.
    BF16,

    /// IEEE 16-bit floating-point value with 5 exponent bits and 10 mantissa bits.
    F16,

    /// IEEE 32-bit floating-point value.
    F32,

    /// IEEE 64-bit floating-point value.
    F64,
}

/// Immutable ABI type of one flattened CUDA kernel parameter.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CudaKernelParameterType {
    /// A non-null device address represented by one native pointer parameter.
    DevicePointer,

    /// One by-value scalar with the specified native representation.
    Scalar(CudaScalarType),
}

/// Immutable description of a CUDA kernel's launch resources.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CudaKernelLaunchDimensions {
    /// CUDA grid dimensions.
    grid: [u32; 3],

    /// CUDA thread-block dimensions.
    block: [u32; 3],

    /// Dynamic shared memory requested for each thread block.
    dynamic_shared_memory_bytes: u32,
}

impl CudaKernelLaunchDimensions {
    /// Creates launch dimensions after validating that all grid and thread-block dimensions are nonzero.
    ///
    /// Device-specific limits and kernel resource requirements are checked by CUDA at launch time.
    pub fn new(grid: [u32; 3], block: [u32; 3], dynamic_shared_memory_bytes: u32) -> Result<Self, Error> {
        if grid.contains(&0) {
            return Err(Error::invalid_argument("cuda grid dimensions must be nonzero"));
        }
        if block.contains(&0) {
            return Err(Error::invalid_argument("cuda thread-block dimensions must be nonzero"));
        }
        Ok(Self { grid, block, dynamic_shared_memory_bytes })
    }

    /// Returns the CUDA grid dimensions.
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    /// Returns the CUDA thread-block dimensions.
    pub fn block(&self) -> [u32; 3] {
        self.block
    }

    /// Returns the dynamic shared memory requested for each thread block.
    pub fn dynamic_shared_memory_bytes(&self) -> u32 {
        self.dynamic_shared_memory_bytes
    }
}

/// Immutable, versioned ABI description for a flattened CUDA kernel parameter list.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CudaKernelAbi {
    /// Name of the ABI schema.
    schema: String,

    /// Version of the ABI schema.
    version: u32,

    /// Flattened kernel parameter types in launch order.
    parameters: Box<[CudaKernelParameterType]>,
}

impl CudaKernelAbi {
    /// Creates an immutable CUDA kernel ABI description with a nonempty schema name.
    ///
    /// The schema and version are producer metadata. No schema registry or executable-signature verification is
    /// performed; parameter types describe the flattened native argument order used during launch validation.
    pub fn new<S: Into<String>, P: Into<Box<[CudaKernelParameterType]>>>(
        schema: S,
        version: u32,
        parameters: P,
    ) -> Result<Self, Error> {
        let schema = schema.into();
        if schema.trim().is_empty() {
            return Err(Error::invalid_argument("cuda kernel ABI schema must not be empty"));
        }
        Ok(Self { schema, version, parameters: parameters.into() })
    }

    /// Returns the ABI schema name.
    pub fn schema(&self) -> &str {
        self.schema.as_str()
    }

    /// Returns the ABI schema version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns the flattened kernel parameter types in launch order.
    pub fn parameters(&self) -> &[CudaKernelParameterType] {
        self.parameters.as_ref()
    }
}

/// Immutable CUDA artifact and its backend-neutral launch metadata.
#[derive(Clone, Debug)]
pub struct CudaKernelArtifact {
    /// Shared immutable artifact state, including its loaded-resource content address.
    inner: Arc<CudaKernelArtifactInner>,

    /// Per-variant execution defaults, independent of the shared compiled image.
    launch_dimensions: CudaKernelLaunchDimensions,
}

/// Shared producer image and metadata that identify and describe one compiled function.
#[derive(Debug)]
struct CudaKernelArtifactInner {
    /// Loaded-resource identity computed once from the image, format, and symbol.
    content_address: CudaKernelContentAddress,

    /// Representation of the compiled image.
    format: CudaArtifactFormat,

    /// Exact producer-supplied image.
    bytes: Box<[u8]>,

    /// Exported kernel entry point.
    symbol: String,

    /// Producer-recorded target architecture.
    target_architecture: String,

    /// SM number decoded from a cubin header, absent for PTX.
    cubin_architecture: Option<u32>,

    /// Declared flattened parameter ABI.
    abi: CudaKernelAbi,
}

impl CudaKernelArtifact {
    /// Creates an immutable CUDA kernel artifact.
    ///
    /// Cubin bytes must form a little-endian ELF64 image for NVIDIA CUDA, and the SM number recorded in its header must
    /// agree with an `sm_<N>` `target_architecture`. Refer to [`Self::cubin_architecture`] for the header encoding.
    /// PTX syntax and the declared parameter ABI are not verified. Constructing an artifact does not establish the
    /// safety of launching it; callers must satisfy [`CudaKernelLauncher::launch`](crate::CudaKernelLauncher::launch).
    ///
    /// # Parameters
    ///
    ///   - `format`: Binary representation of `bytes`.
    ///   - `bytes`: Cubin or PTX bytes. PTX must not contain an interior NUL byte.
    ///   - `symbol`: Exported kernel symbol.
    ///   - `target_architecture`: Producer-recorded target such as `sm_100` or `compute_90`.
    ///   - `launch_dimensions`: Grid, thread-block, and dynamic shared-memory requirements.
    ///   - `abi`: Immutable flattened parameter ABI.
    pub fn new<B: Into<Box<[u8]>>, S: Into<String>, T: Into<String>>(
        format: CudaArtifactFormat,
        bytes: B,
        symbol: S,
        target_architecture: T,
        launch_dimensions: CudaKernelLaunchDimensions,
        abi: CudaKernelAbi,
    ) -> Result<Self, Error> {
        let bytes = bytes.into();
        if bytes.is_empty() {
            return Err(Error::invalid_argument("cuda kernel artifact bytes must not be empty"));
        }
        if format == CudaArtifactFormat::Ptx && bytes.contains(&0) {
            return Err(Error::invalid_argument("cuda PTX must not contain a NUL byte"));
        }
        let symbol = symbol.into();
        if symbol.is_empty() {
            return Err(Error::invalid_argument("cuda kernel symbol must not be empty"));
        }
        if symbol.as_bytes().contains(&0) {
            return Err(Error::invalid_argument("cuda kernel symbol must not contain a NUL byte"));
        }
        let target_architecture = target_architecture.into();
        if target_architecture.trim().is_empty() {
            return Err(Error::invalid_argument("cuda target architecture must not be empty"));
        }
        let cubin_architecture = match format {
            CudaArtifactFormat::Cubin => {
                let architecture = cubin_elf_architecture(&bytes)?;
                if let Some(CudaTargetArchitecture::Real(recorded) | CudaTargetArchitecture::Restricted(recorded)) =
                    CudaTargetArchitecture::parse(&target_architecture)
                    && target_architecture.starts_with("sm_")
                    && recorded != architecture
                {
                    return Err(Error::invalid_argument(format!(
                        "cuda cubin ELF header targets `sm_{architecture}`, but the artifact records target \
                         architecture `{target_architecture}`",
                    )));
                }
                Some(architecture)
            }
            CudaArtifactFormat::Ptx => None,
        };
        let content_address = CudaKernelContentAddress::new(format, &bytes, &symbol);
        Ok(Self {
            launch_dimensions,
            inner: Arc::new(CudaKernelArtifactInner {
                content_address,
                format,
                bytes,
                symbol,
                target_architecture,
                cubin_architecture,
                abi,
            }),
        })
    }

    /// Returns the artifact binary format.
    pub fn format(&self) -> CudaArtifactFormat {
        self.inner.format
    }

    /// Returns the artifact bytes exactly as supplied by its producer.
    pub fn bytes(&self) -> &[u8] {
        self.inner.bytes.as_ref()
    }

    /// Returns the exported CUDA kernel symbol.
    pub fn symbol(&self) -> &str {
        self.inner.symbol.as_str()
    }

    /// Returns the producer-recorded target architecture.
    pub fn target_architecture(&self) -> &str {
        self.inner.target_architecture.as_str()
    }

    /// Returns the SM number recorded in the ELF64 header of a cubin artifact (e.g., `90` for `sm_90`), or [`None`]
    /// for PTX artifacts.
    ///
    /// A cubin is a little-endian ELF64 image whose `e_machine` is `EM_CUDA` (190). Its `e_flags` field stores the SM
    /// number in the low byte for `EI_ABIVERSION` 7 (`ELFABIVERSION_CUDA_V1`, e.g. `0x5a` for `sm_90`) and in the
    /// second byte for `EI_ABIVERSION` 8 (`ELFABIVERSION_CUDA_V2`, e.g. `0x6400` for `sm_100`). Feature-specific
    /// variants such as `sm_90a` set separate accelerator flag bits that are not part of the returned number. This
    /// follows LLVM's
    /// [CUDA ELF definitions](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/BinaryFormat/ELF.h).
    pub fn cubin_architecture(&self) -> Option<u32> {
        self.inner.cubin_architecture
    }

    /// Validates that this artifact can be loaded on a device with the provided compute capability.
    ///
    /// Ordinary cubins require the same major architecture and an equal or newer minor revision. Ordinary PTX
    /// targets require at least the recorded capability. Restricted targets with feature suffixes and unrecognized
    /// representations defer their additional requirements to CUDA. This preflight check does not inspect instructions
    /// or accelerator flags and does not replace the driver's authoritative image compatibility checks.
    pub(super) fn validate_device_compatibility(
        &self,
        device: i32,
        capability: CudaComputeCapability,
    ) -> Result<(), Error> {
        let compatible =
            match (self.inner.cubin_architecture, CudaTargetArchitecture::parse(self.target_architecture())) {
                (_, Some(CudaTargetArchitecture::Restricted(_))) => true,
                (Some(architecture), _) => {
                    architecture / 10 == capability.major && architecture % 10 <= capability.minor
                }
                (None, Some(target)) => capability.number() >= target.number(),
                (None, None) => true,
            };
        if compatible {
            return Ok(());
        }
        let description = match self.inner.cubin_architecture {
            Some(architecture) => format!("cuda cubin targets `sm_{architecture}`"),
            None => format!("cuda PTX targets `{}`", self.target_architecture()),
        };
        Err(Error::invalid_argument(format!(
            "{description}, but cuda device {device} has compute capability {capability}",
        )))
    }

    /// Returns the default launch resources, which can be changed with [`Self::with_launch_dimensions`].
    pub fn launch_dimensions(&self) -> CudaKernelLaunchDimensions {
        self.launch_dimensions
    }

    /// Returns an artifact variant with different execution defaults, sharing the original image and content address.
    ///
    /// This operation neither copies nor hashes the compiled image and does not change loaded-resource cache identity.
    /// Device resource limits are checked when the variant is launched.
    pub fn with_launch_dimensions(&self, launch_dimensions: CudaKernelLaunchDimensions) -> Self {
        Self { inner: self.inner.clone(), launch_dimensions }
    }

    /// Returns the immutable flattened kernel ABI.
    pub fn abi(&self) -> &CudaKernelAbi {
        &self.inner.abi
    }

    /// Returns the content address used to identify the loaded CUDA resource.
    pub(super) fn content_address(&self) -> CudaKernelContentAddress {
        self.inner.content_address
    }
}

/// Content address of the fields that determine a loaded CUDA module/function resource.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct CudaKernelContentAddress([u8; 32]);

impl CudaKernelContentAddress {
    /// Hashes the unambiguously delimited fields that identify one loaded function.
    fn new(format: CudaArtifactFormat, bytes: &[u8], symbol: &str) -> Self {
        let mut digest = Sha256::new();
        digest.update(b"ryft.cuda.kernel.v1\0");
        digest.update([match format {
            CudaArtifactFormat::Cubin => 0,
            CudaArtifactFormat::Ptx => 1,
        }]);
        digest.update(u64::try_from(bytes.len()).unwrap().to_le_bytes());
        digest.update(bytes);
        digest.update(u64::try_from(symbol.len()).unwrap().to_le_bytes());
        digest.update(symbol.as_bytes());
        Self(digest.finalize().into())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use crate::tests::{test_artifact, test_artifact_with_bytes, test_cubin};

    use super::*;

    /// Builds a valid empty-parameter artifact for image and target validation.
    fn test_image(format: CudaArtifactFormat, bytes: Vec<u8>, target: &str) -> Result<CudaKernelArtifact, Error> {
        CudaKernelArtifact::new(
            format,
            bytes,
            "kernel",
            target,
            CudaKernelLaunchDimensions::new([1, 1, 1], [1, 1, 1], 0).unwrap(),
            CudaKernelAbi::new("test", 1, Vec::new()).unwrap(),
        )
    }

    /// Builds a minimal PTX metadata fixture without invoking the CUDA parser.
    fn test_ptx_artifact(target: &str) -> CudaKernelArtifact {
        test_image(CudaArtifactFormat::Ptx, b".version 8.0".to_vec(), target).unwrap()
    }

    #[test]
    fn test_cuda_compute_capability_number() {
        let capability = CudaComputeCapability { major: 12, minor: 1 };
        assert_eq!(capability.number(), 121);
        assert_eq!(capability.to_string(), "12.1");
        assert_eq!(format!("{capability:?}"), "CudaComputeCapability { major: 12, minor: 1 }");
    }

    #[test]
    fn test_cubin_elf_architecture() {
        assert_eq!(cubin_elf_architecture(&test_cubin(90, &[])), Ok(90));
        for (offset, value) in [(0, 0), (4, 1), (5, 2)] {
            let mut bytes = test_cubin(90, &[]);
            bytes[offset] = value;
            assert!(matches!(
                cubin_elf_architecture(&bytes),
                Err(Error::InvalidArgument { message, .. })

                    if message == "cuda cubin bytes do not start with a little-endian ELF64 header",
            ));
        }
        assert!(matches!(
            cubin_elf_architecture(&[0; 63]),
            Err(Error::InvalidArgument { message, .. })

                if message == "cuda cubin bytes do not start with a little-endian ELF64 header",
        ));
    }

    #[test]
    fn test_cuda_target_architecture_parse() {
        assert_eq!(CudaTargetArchitecture::parse("sm_100"), Some(CudaTargetArchitecture::Real(100)));
        assert_eq!(CudaTargetArchitecture::parse("compute_90"), Some(CudaTargetArchitecture::Virtual(90)));
        assert_eq!(CudaTargetArchitecture::parse("sm_90a"), Some(CudaTargetArchitecture::Restricted(90)));
        assert_eq!(CudaTargetArchitecture::parse("compute_120f"), Some(CudaTargetArchitecture::Restricted(120)));
        for target in ["sm_", "sm_90A", "sm_9_0", "gfx942", "sm_99999999999999"] {
            assert_eq!(CudaTargetArchitecture::parse(target), None);
        }
    }

    #[test]
    fn test_cuda_target_architecture_number() {
        for target in [
            CudaTargetArchitecture::Real(90),
            CudaTargetArchitecture::Virtual(90),
            CudaTargetArchitecture::Restricted(90),
        ] {
            assert_eq!(target.number(), 90);
        }
    }

    #[test]
    fn test_cuda_kernel_launch_dimensions_new() {
        assert_eq!(
            CudaKernelLaunchDimensions::new([1, 2, 3], [4, 5, 6], 128),
            Ok(CudaKernelLaunchDimensions { grid: [1, 2, 3], block: [4, 5, 6], dynamic_shared_memory_bytes: 128 }),
        );
        assert!(matches!(
            CudaKernelLaunchDimensions::new([0, 1, 1], [1, 1, 1], 0),
            Err(Error::InvalidArgument { message, .. })
                if message == "cuda grid dimensions must be nonzero",
        ));
        assert!(matches!(
            CudaKernelLaunchDimensions::new([1, 1, 1], [1, 0, 1], 0),
            Err(Error::InvalidArgument { message, .. })
                if message == "cuda thread-block dimensions must be nonzero",
        ));
    }

    #[test]
    fn test_cuda_kernel_launch_dimensions_grid() {
        assert_eq!(CudaKernelLaunchDimensions::new([1, 2, 3], [4, 5, 6], 128).unwrap().grid(), [1, 2, 3]);
    }

    #[test]
    fn test_cuda_kernel_launch_dimensions_block() {
        assert_eq!(CudaKernelLaunchDimensions::new([1, 2, 3], [4, 5, 6], 128).unwrap().block(), [4, 5, 6]);
    }

    #[test]
    fn test_cuda_kernel_launch_dimensions_dynamic_shared_memory_bytes() {
        assert_eq!(
            CudaKernelLaunchDimensions::new([1, 2, 3], [4, 5, 6], 128).unwrap().dynamic_shared_memory_bytes(),
            128,
        );
    }

    #[test]
    fn test_cuda_kernel_launch_dimensions_equality_hash_debug() {
        let dimensions = CudaKernelLaunchDimensions::new([1, 2, 3], [4, 5, 6], 128).unwrap();
        let other = CudaKernelLaunchDimensions::new([2, 2, 3], [4, 5, 6], 128).unwrap();
        assert_eq!(dimensions, dimensions);
        assert_ne!(dimensions, other);
        assert_eq!(HashMap::from([(dimensions, 7)]).get(&dimensions), Some(&7));
        assert_eq!(
            format!("{dimensions:?}"),
            "CudaKernelLaunchDimensions { grid: [1, 2, 3], block: [4, 5, 6], dynamic_shared_memory_bytes: 128 }",
        );
    }

    #[test]
    fn test_cuda_kernel_abi_new() {
        assert_eq!(
            CudaKernelAbi::new("test", 1, Vec::new()),
            Ok(CudaKernelAbi { schema: "test".into(), version: 1, parameters: Box::new([]) }),
        );
        for schema in ["", " \t"] {
            assert!(matches!(
                CudaKernelAbi::new(schema, 1, Vec::new()),
                Err(Error::InvalidArgument { message, .. })
                    if message == "cuda kernel ABI schema must not be empty",
            ));
        }
    }

    #[test]
    fn test_cuda_kernel_abi_schema() {
        let abi = CudaKernelAbi::new("test", 7, vec![CudaKernelParameterType::DevicePointer]).unwrap();
        assert_eq!(abi.schema(), "test");
    }

    #[test]
    fn test_cuda_kernel_abi_version() {
        let abi = CudaKernelAbi::new("test", 7, vec![CudaKernelParameterType::DevicePointer]).unwrap();
        assert_eq!(abi.version(), 7);
    }

    #[test]
    fn test_cuda_kernel_abi_parameters() {
        let abi = CudaKernelAbi::new("test", 7, vec![CudaKernelParameterType::DevicePointer]).unwrap();
        assert_eq!(abi.parameters(), &[CudaKernelParameterType::DevicePointer]);
    }

    #[test]
    fn test_cuda_kernel_abi_equality_hash_debug() {
        let abi = CudaKernelAbi::new("test", 1, Vec::new()).unwrap();
        assert_eq!(abi, abi.clone());
        assert_ne!(abi, CudaKernelAbi::new("test", 2, Vec::new()).unwrap());
        assert_eq!(HashMap::from([(abi.clone(), 7)]).get(&abi), Some(&7));
        assert_eq!(format!("{abi:?}"), "CudaKernelAbi { schema: \"test\", version: 1, parameters: [] }");
    }

    #[test]
    fn test_cuda_kernel_artifact_new() {
        let artifact = test_artifact(Vec::new());
        assert!(Arc::ptr_eq(&artifact.inner, &artifact.clone().inner));
    }

    #[test]
    fn test_cuda_kernel_artifact_new_concurrently_shareable() {
        let artifact = Arc::new(test_artifact(Vec::new()));
        let threads = (0..8)
            .map(|_| {
                let artifact = artifact.clone();
                std::thread::spawn(move || {
                    assert_eq!(artifact.symbol(), "test_kernel");
                    assert_eq!(artifact.target_architecture(), "sm_100");
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn test_cuda_kernel_artifact_new_invalid_metadata() {
        let dimensions = CudaKernelLaunchDimensions::new([1, 1, 1], [1, 1, 1], 0).unwrap();
        for (format, bytes, symbol, target, expected) in [
            (CudaArtifactFormat::Cubin, Vec::new(), "kernel", "sm_100", "cuda kernel artifact bytes must not be empty"),
            (CudaArtifactFormat::Ptx, b"a\0b".to_vec(), "kernel", "sm_100", "cuda PTX must not contain a NUL byte"),
            (CudaArtifactFormat::Cubin, test_cubin(100, &[]), "", "sm_100", "cuda kernel symbol must not be empty"),
            (
                CudaArtifactFormat::Cubin,
                test_cubin(100, &[]),
                "a\0b",
                "sm_100",
                "cuda kernel symbol must not contain a NUL byte",
            ),
            (
                CudaArtifactFormat::Cubin,
                test_cubin(100, &[]),
                "kernel",
                " ",
                "cuda target architecture must not be empty",
            ),
        ] {
            assert!(matches!(
                CudaKernelArtifact::new(
                        format,
                        bytes,
                        symbol,
                        target,
                        dimensions,
                        CudaKernelAbi::new("test", 1, Vec::new()).unwrap(),
                    ),
                Err(Error::InvalidArgument { message, .. })
                    if message == expected,
            ));
        }
    }

    #[test]
    fn test_cuda_kernel_artifact_new_invalid_cubin() {
        assert!(matches!(
            test_image(CudaArtifactFormat::Cubin, b"malformed".to_vec(), "sm_100"),
            Err(Error::InvalidArgument { message, .. })

                if message == "cuda cubin bytes do not start with a little-endian ELF64 header",
        ));
        let mut big_endian = test_cubin(100, &[]);
        big_endian[5] = 2;
        assert!(matches!(
            test_image(CudaArtifactFormat::Cubin, big_endian, "sm_100"),
            Err(Error::InvalidArgument { message, .. })

                if message == "cuda cubin bytes do not start with a little-endian ELF64 header",
        ));
        let mut x86_64 = test_cubin(100, &[]);
        x86_64[18..20].copy_from_slice(&62u16.to_le_bytes());
        assert!(matches!(
            test_image(CudaArtifactFormat::Cubin, x86_64, "sm_100"),
            Err(Error::InvalidArgument { message, .. })

                if message == "cuda cubin ELF header has `e_machine` 62, expected 190 (NVIDIA CUDA)",
        ));
        assert!(matches!(
            test_image(CudaArtifactFormat::Cubin, test_cubin(100, &[]), "sm_90"),
            Err(Error::InvalidArgument { message, .. })

                if message == "cuda cubin ELF header targets `sm_100`, but the artifact records target \
                               architecture `sm_90`",
        ));
    }

    #[test]
    fn test_cuda_kernel_artifact_format() {
        assert_eq!(test_artifact(Vec::new()).format(), CudaArtifactFormat::Cubin);
    }

    #[test]
    fn test_cuda_kernel_artifact_bytes() {
        assert_eq!(test_artifact(Vec::new()).bytes(), test_cubin(100, &[1, 2, 3]));
    }

    #[test]
    fn test_cuda_kernel_artifact_symbol() {
        assert_eq!(test_artifact(Vec::new()).symbol(), "test_kernel");
    }

    #[test]
    fn test_cuda_kernel_artifact_target_architecture() {
        assert_eq!(test_artifact(Vec::new()).target_architecture(), "sm_100");
    }

    #[test]
    fn test_cuda_kernel_artifact_cubin_architecture() {
        // The pre-Blackwell ABI stores the SM number in the low byte of `e_flags` and the Blackwell ABI in the second
        // byte; accelerator flag bits and feature suffixes do not participate in the comparison.
        assert_eq!(
            test_image(CudaArtifactFormat::Cubin, test_cubin(90, &[]), "sm_90").unwrap().cubin_architecture(),
            Some(90),
        );
        let mut blackwell = test_cubin(0, &[]);
        blackwell[8] = 8;
        blackwell[48..52].copy_from_slice(&(0x0000_6400u32 | 0x8).to_le_bytes());
        assert_eq!(
            test_image(CudaArtifactFormat::Cubin, blackwell, "sm_100a").unwrap().cubin_architecture(),
            Some(100),
        );
        let mut accelerated_hopper = test_cubin(90, &[]);
        accelerated_hopper[48..52].copy_from_slice(&(0x5au32 | 0x800).to_le_bytes());
        assert_eq!(
            test_image(CudaArtifactFormat::Cubin, accelerated_hopper, "sm_90a").unwrap().cubin_architecture(),
            Some(90),
        );
        assert_eq!(test_ptx_artifact("compute_90").cubin_architecture(), None);

        // Only `sm_<N>` targets are compared against the header; other representations remain open-ended.
        assert_eq!(
            test_image(CudaArtifactFormat::Cubin, test_cubin(90, &[]), "compute_90")
                .unwrap()
                .cubin_architecture(),
            Some(90),
        );
        assert_eq!(
            test_image(CudaArtifactFormat::Cubin, test_cubin(90, &[]), "hopper").unwrap().cubin_architecture(),
            Some(90),
        );
    }

    #[test]
    fn test_cuda_kernel_artifact_validate_device_compatibility() {
        let cubin = test_artifact(Vec::new());
        assert_eq!(cubin.validate_device_compatibility(0, CudaComputeCapability { major: 10, minor: 0 }), Ok(()));
        assert!(matches!(
            cubin.validate_device_compatibility(1, CudaComputeCapability { major: 9, minor: 0 }),
            Err(Error::InvalidArgument { message, .. })

                if message == "cuda cubin targets `sm_100`, but cuda device 1 has compute capability 9.0",
        ));
        assert!(matches!(
            cubin.validate_device_compatibility(0, CudaComputeCapability { major: 12, minor: 0 }),
            Err(Error::InvalidArgument { message, .. })

                if message == "cuda cubin targets `sm_100`, but cuda device 0 has compute capability 12.0",
        ));

        // Ordinary PTX can JIT forward; restricted target acceptance is decided by CUDA.
        let ptx = test_ptx_artifact("compute_90");
        assert_eq!(ptx.validate_device_compatibility(0, CudaComputeCapability { major: 9, minor: 0 }), Ok(()));
        assert_eq!(ptx.validate_device_compatibility(0, CudaComputeCapability { major: 12, minor: 1 }), Ok(()));
        assert!(matches!(
            ptx.validate_device_compatibility(2, CudaComputeCapability { major: 8, minor: 9 }),
            Err(Error::InvalidArgument { message, .. })

                if message == "cuda PTX targets `compute_90`, but cuda device 2 has compute capability 8.9",
        ));
        assert_eq!(
            test_ptx_artifact("sm_90").validate_device_compatibility(0, CudaComputeCapability { major: 10, minor: 0 }),
            Ok(()),
        );
        assert_eq!(
            test_ptx_artifact("hopper").validate_device_compatibility(0, CudaComputeCapability { major: 7, minor: 0 }),
            Ok(()),
        );
        let older_minor = test_image(CudaArtifactFormat::Cubin, test_cubin(80, &[]), "sm_80").unwrap();
        assert_eq!(older_minor.validate_device_compatibility(0, CudaComputeCapability { major: 8, minor: 6 }), Ok(()));
        let newer_minor = test_image(CudaArtifactFormat::Cubin, test_cubin(86, &[]), "sm_86").unwrap();
        assert!(matches!(
            newer_minor.validate_device_compatibility(0, CudaComputeCapability { major: 8, minor: 0 }),
            Err(Error::InvalidArgument { message, .. })
                if message == "cuda cubin targets `sm_86`, but cuda device 0 has compute capability 8.0",
        ));
        for target in ["sm_90a", "compute_120f"] {
            assert_eq!(
                test_ptx_artifact(target)
                    .validate_device_compatibility(0, CudaComputeCapability { major: 8, minor: 0 }),
                Ok(()),
            );
        }
        let restricted = test_image(CudaArtifactFormat::Cubin, test_cubin(90, &[]), "sm_90a").unwrap();
        assert_eq!(restricted.validate_device_compatibility(0, CudaComputeCapability { major: 10, minor: 0 }), Ok(()));
    }

    #[test]
    fn test_cuda_kernel_artifact_launch_dimensions() {
        assert_eq!(
            test_artifact(Vec::new()).launch_dimensions(),
            CudaKernelLaunchDimensions::new([1, 2, 3], [4, 5, 6], 128).unwrap(),
        );
    }

    #[test]
    fn test_cuda_kernel_artifact_with_launch_dimensions() {
        let artifact = test_artifact(Vec::new());
        let dimensions = CudaKernelLaunchDimensions::new([8, 1, 1], [32, 1, 1], 65536).unwrap();
        let variant = artifact.with_launch_dimensions(dimensions);
        assert_eq!(variant.launch_dimensions(), dimensions);
        assert_eq!(artifact.launch_dimensions().grid(), [1, 2, 3]);
        assert!(Arc::ptr_eq(&artifact.inner, &variant.inner));
        assert_eq!(artifact.content_address(), variant.content_address());
    }

    #[test]
    fn test_cuda_kernel_artifact_abi() {
        assert_eq!(
            test_artifact(vec![CudaKernelParameterType::DevicePointer]).abi(),
            &CudaKernelAbi::new("ryft.test", 1, vec![CudaKernelParameterType::DevicePointer]).unwrap(),
        );
    }

    #[test]
    fn test_cuda_kernel_artifact_content_address() {
        let artifact = test_artifact(Vec::new());
        assert_eq!(
            artifact.content_address(),
            test_artifact(vec![CudaKernelParameterType::DevicePointer]).content_address(),
        );
        assert_ne!(
            artifact.content_address(),
            test_artifact_with_bytes(test_cubin(100, &[4]), Vec::new()).content_address(),
        );
        let other_target = CudaKernelArtifact::new(
            artifact.format(),
            artifact.bytes(),
            artifact.symbol(),
            "blackwell",
            artifact.launch_dimensions(),
            artifact.abi().clone(),
        )
        .unwrap();
        assert_eq!(artifact.content_address(), other_target.content_address());
    }

    #[test]
    fn test_cuda_kernel_content_address_new() {
        let address = CudaKernelContentAddress::new(CudaArtifactFormat::Cubin, b"abc", "kernel");
        assert_eq!(address, CudaKernelContentAddress::new(CudaArtifactFormat::Cubin, b"abc", "kernel"));
        assert_ne!(address, CudaKernelContentAddress::new(CudaArtifactFormat::Ptx, b"abc", "kernel"));
        assert_ne!(address, CudaKernelContentAddress::new(CudaArtifactFormat::Cubin, b"abd", "kernel"));
        assert_ne!(address, CudaKernelContentAddress::new(CudaArtifactFormat::Cubin, b"abc", "other"));
        assert_ne!(
            CudaKernelContentAddress::new(CudaArtifactFormat::Cubin, b"ab", "c"),
            CudaKernelContentAddress::new(CudaArtifactFormat::Cubin, b"a", "bc"),
        );
        assert_eq!(HashMap::from([(address, 7)]).get(&address), Some(&7));
    }
}
