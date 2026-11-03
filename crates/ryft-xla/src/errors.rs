use std::sync::Arc;

use ryft_core::{
    ArrayType, BatchingError, BroadcastingError, DataType, DataTypeError, DeviceMesh, DifferentiationError,
    Error as CoreError, LayoutError, Memory, MeshAxisType, ParameterError, ProgramError, ReferenceError, Shape,
    Sharding, ShardingError, TypeError,
};
use ryft_mlir::Error as MlirError;
use ryft_pjrt::{DeviceId, Error as PjrtError};

use crate::arrays::ShardIndex;
use crate::kernels::KernelEmbeddingError;

/// Errors returned by array preparation, tracing, lowering, compilation, and execution in `ryft-xla`. Foreign Ryft core
/// errors have one canonical [`Core`](Self::Core) carrier. Batching and differentiation errors first convert through
/// [`ProgramError`], preserving the core pipeline's normalized typed payloads.
#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Core(#[from] CoreError),

    #[error(transparent)]
    Pjrt(#[from] PjrtError),

    #[error(transparent)]
    Mlir(#[from] MlirError),

    #[error(transparent)]
    Io(Arc<std::io::Error>),

    #[error("{message}")]
    SizeLimitExceeded { message: String },

    #[error("expected {expected} byte(s) but got {actual}")]
    ByteCountMismatch { expected: usize, actual: usize },

    #[error("missing required sharding metadata")]
    MissingSharding,

    #[error("domain was constructed without a device mesh")]
    MissingMesh,

    #[error("expected static shape but got {shape}")]
    DynamicShape { shape: Shape },

    #[error("device {device_id} is not addressable by the PJRT client in process {process_index}")]
    NonAddressableDevice { device_id: DeviceId, process_index: usize },

    #[error("got multiple buffers for device {device_id}")]
    MultipleBuffersOnDevice { device_id: DeviceId },

    #[error("device {device_id} is not present in the device mesh")]
    DeviceNotInMesh { device_id: DeviceId },

    #[error("device {device_id} reports process index {actual_process_index}, but expected {expected_process_index}")]
    DeviceProcessIndexMismatch { device_id: DeviceId, expected_process_index: usize, actual_process_index: usize },

    #[error("buffer has type {actual}, but expected {expected}")]
    BufferTypeMismatch { expected: ArrayType, actual: ArrayType },

    #[error("device {device_id} has no addressable memory for `{memory}`")]
    UnsupportedMemory { device_id: DeviceId, memory: Memory },

    #[error("buffer on device {device_id} uses memory kind `{actual}`, but expected `{expected}`")]
    BufferMemoryMismatch { device_id: DeviceId, expected: Memory, actual: String },

    #[error("zero-space buffer on device {device_id} contains a nonzero physical carrier value")]
    NonCanonicalZeroBuffer { device_id: DeviceId },

    #[error("shape rank {shape_rank} does not match shard slice rank {slice_rank}")]
    ShardSliceRankMismatch { shape_rank: usize, slice_rank: usize },

    // TODO(eaplatanios): Review from here onwards.
    /// Error returned when `device_put` is asked to shard an element type without a supported dense host encoding.
    #[error("device_put does not support dense host bytes for element type {element_type}")]
    UnsupportedDevicePutElementType { element_type: DataType },

    /// Error returned when the higher-level [`device_put`](fn@crate::arrays_v0::device_put) API needs a default device
    /// but the client has no local devices.
    #[error("device_put needs a default local device, but the PJRT client has no addressable devices")]
    MissingDefaultDevice,

    /// Error returned when a provided `src` placement does not match an array leaf's current placement.
    #[error(
        "device_put src mesh {expected_mesh:?} with sharding {expected_sharding:?} does not match the array's current \
         mesh {actual_mesh:?} with sharding {actual_sharding:?}"
    )]
    SourcePlacementMismatch {
        expected_mesh: DeviceMesh,
        expected_sharding: Sharding,
        actual_mesh: DeviceMesh,
        actual_sharding: Sharding,
    },

    /// Error returned when a device ID cannot be represented by the PJRT cross-host transfers extension.
    #[error("device {device_id} cannot be represented as a PJRT global device ID for cross-host transfers")]
    CrossHostTransferDeviceIdTooLarge { device_id: DeviceId },

    /// Error returned when a shard dimension cannot be represented by the PJRT cross-host transfers extension.
    #[error(
        "cross-host transfer for shard #{shard_index} has shape dimension #{dimension}={size}, which does not fit \
         in i64"
    )]
    CrossHostTransferShapeDimensionTooLarge { shard_index: ShardIndex, dimension: usize, size: usize },

    /// Error returned when an exact-shard cross-host transfer key cannot be represented in PJRT.
    #[error(
        "exact-shard transfer key for source shard #{source_shard_index} and destination shard \
         #{destination_shard_index} does not fit in i64"
    )]
    CrossHostTransferKeyTooLarge { source_shard_index: ShardIndex, destination_shard_index: ShardIndex },

    /// Error returned when [`XlaArray::to_placement`](crate::XlaArray::to_placement) needs a source shard that is not
    /// addressable locally.
    #[error(
        "array move requires shard #{shard_index} on device {device_id} to be addressable from the current process"
    )]
    MissingAddressableShardForMove { shard_index: ShardIndex, device_id: DeviceId },

    /// Error returned when copying a source shard to host yields an unexpected byte count.
    #[error(
        "copied shard #{shard_index} from device {device_id} to host and got {actual_byte_count} byte(s), but expected \
         {expected_byte_count}"
    )]
    CopiedShardByteCountMismatch {
        shard_index: ShardIndex,
        device_id: DeviceId,
        expected_byte_count: usize,
        actual_byte_count: usize,
    },

    /// Error returned when overlapping source shards disagree while materializing a dense host array.
    #[error("array move found inconsistent overlapping data while materializing shard #{shard_index}")]
    InconsistentOverlappingShardData { shard_index: ShardIndex },

    /// Error returned when the destination [`DeviceMesh`] has a mesh axis with
    /// [`MeshAxisType::Manual`]. `Manual` axes are
    /// managed explicitly by the user (e.g. inside
    /// [`shard_map`](fn@ryft_core::shard_map)); the compiled reshard path uses the SPMD
    /// partitioner from the top level and can only plan `Auto` and `Explicit` axes.
    #[error("compiled reshard does not support `Manual` mesh axis {axis_name} (type {axis_type:?})")]
    UnsupportedMeshAxisType { axis_name: String, axis_type: MeshAxisType },

    /// Error returned when a destination device of a compiled reshard is not addressable from the
    /// current process. Cross-host destinations are not yet supported.
    #[error(
        "compiled reshard destination device {device_id} is not addressable from the current process (its process \
         index is {process_index})"
    )]
    NonAddressableDestinationDevice { device_id: DeviceId, process_index: usize },

    /// Error returned when the number of donation flags does not match the number of arrays.
    #[error("got {actual_count} donation flag(s), but expected {expected_count}")]
    DonationFlagCountMismatch { expected_count: usize, actual_count: usize },

    /// Error returned when the device list for execution contains duplicate IDs.
    #[error("device {device_id} appears multiple times in the execution device order")]
    DuplicateExecutionDeviceId { device_id: DeviceId },

    /// Error returned when an array does not have an addressable shard for a required device.
    #[error("input array #{array_index} has no addressable shard for device {device_id}")]
    MissingArrayShardForDevice { array_index: usize, device_id: DeviceId },

    /// Error returned when an array has an addressable shard for a device that is not in the execution device order.
    #[error("input array #{array_index} has an unexpected addressable shard for device {device_id}")]
    UnexpectedArrayShardDevice { array_index: usize, device_id: DeviceId },

    /// Error returned when a lowered function name is empty or contains whitespace.
    #[error("invalid function name `{function_name}` used during XLA lowering")]
    InvalidFunctionName { function_name: String },

    /// Error returned when lowering encounters a traced tensor type that MLIR rejects.
    #[error("invalid tensor type `{array_type}` used during XLA lowering")]
    InvalidTensorType { array_type: ArrayType },

    /// Error returned when a reshape dimension cannot be represented by StableHLO's signed shape element type.
    #[error("reshape dimension {value} cannot be represented as a StableHLO i{bit_width} shape value")]
    ReshapeDimensionOutOfRange { value: usize, bit_width: u8 },

    /// Error returned when a dynamic allocation bound exceeds StableHLO's signed runtime size carrier.
    #[error("dynamic dimension bound {value} cannot be represented as a StableHLO i32 size value")]
    DynamicDimensionOutOfRange { value: usize },

    /// Error returned when a pad interior amount cannot be represented by StableHLO's signed attribute type.
    #[error("pad interior padding {value} cannot be represented as a StableHLO i64 attribute")]
    PadInteriorPaddingOutOfRange { value: usize },

    /// Error returned when lowering encounters a staged op that does not yet have StableHLO support.
    #[error("unsupported staged op `{op}` during XLA lowering")]
    UnsupportedOperation { op: String },

    /// Error returned when unresolved reference semantics reach XLA lowering before functionalization.
    #[error("unresolved reference construct `{construct}` must be discharged before XLA lowering")]
    UnresolvedReference { construct: String },

    /// Error returned when unresolved mutable state reaches ordinary XLA lowering before functionalization.
    #[error("unresolved state in `{construct}` must be discharged before XLA lowering")]
    UnresolvedState { construct: String },

    /// Error returned when lowering encounters a captured constant reference without a matching hidden argument.
    #[error("missing captured constant #{index} during XLA lowering")]
    MissingCapturedConstant { index: usize },

    /// Error returned when lowering tries to materialize abstract XLA type metadata as a literal value.
    #[error("abstract XLA value `{array_type}` cannot be materialized as a StableHLO literal")]
    AbstractValueLiteral { array_type: ArrayType },

    /// Error returned when signature sharding metadata does not match the lowered function signature.
    #[error("invalid {kind} sharding count during XLA lowering: expected {expected}, got {actual}")]
    InvalidShardingCount {
        /// Name of the sharding group being validated.
        kind: &'static str,

        /// Number of shardings required by the signature.
        expected: usize,

        /// Number of shardings provided.
        actual: usize,
    },

    /// Error returned when lowering encounters a type that does not have StableHLO support yet.
    #[error("unsupported data type `{data_type}` during XLA lowering")]
    UnsupportedDataType { data_type: DataType },

    /// Error returned when MLIR rejects the constructed dense-elements attribute.
    #[error("invalid dense elements attribute for data type `{data_type}` during XLA lowering")]
    InvalidDenseElementsAttribute { data_type: DataType },

    /// Error returned when the constructed MLIR module fails verification.
    #[error("constructed MLIR module failed verification during XLA lowering")]
    MlirVerificationFailure,

    /// Error returned when one traced XLA program mixes shard maps from incompatible meshes.
    #[error("traced XLA lowering requires all nested shard maps to use compatible logical meshes")]
    IncompatibleNestedMeshes,

    /// Error returned when logical reference-state metadata cannot map to a safe executable alias.
    #[error("invalid XLA reference-state ABI: {message}")]
    InvalidReferenceStateAbi { message: String },

    /// Error returned when [`to_mlir_module`](crate::operations::sharding::shard_map::to_mlir_module) lowers a traced
    /// shard map whose global input or output type varies along a mesh axis that is not one of its own manual axes.
    /// Only an enclosing manual region makes such an axis manual (e.g., for a body traced with
    /// [`ryft_core::trace_shard_map_with_named_axes`]), and the standalone module that
    /// [`to_mlir_module`](crate::operations::sharding::shard_map::to_mlir_module) produces has none, so the shard map
    /// can only be lowered as part of the program that encloses it.
    #[error(
        "{value_kind} type #{value_index} of a standalone `shard_map` varies along mesh axis `{axis_name}`, which is \
         not one of its manual axes; lower it as part of the program whose manual region makes that axis manual"
    )]
    StandaloneShardMapVariesAlongEnclosingManualAxis {
        /// Whether the varying value is an input or output.
        value_kind: &'static str,

        /// Position of the varying value in its input or output signature.
        value_index: usize,

        /// Mesh axis requiring an enclosing manual region.
        axis_name: String,
    },

    /// Error returned when [`to_mlir_module`](crate::operations::sharding::shard_map::to_mlir_module) lowers a traced
    /// shard map whose body applies a collective (e.g., `parallel_reduce` or `axis_index`) along a manual mesh axis
    /// that neither the shard map nor a shard map nested in its body makes manual. Only an enclosing manual region
    /// makes such an axis manual (e.g., for a body traced with [`ryft_core::trace_shard_map_with_named_axes`]), even
    /// when the boundary of the shard map does not vary along it (e.g., for a reduction of an invariant input), and the
    /// standalone module that [`to_mlir_module`](crate::operations::sharding::shard_map::to_mlir_module) produces has
    /// none.
    #[error(
        "`{operation_name}` in the body of a standalone `shard_map` communicates along mesh axis `{axis_name}`, which \
         is not one of its manual axes; lower it as part of the program whose manual region makes that axis manual"
    )]
    StandaloneShardMapCommunicatesAlongEnclosingManualAxis { operation_name: &'static str, axis_name: String },

    /// Error surfaced when [`compile_with_options`](crate::jit::compile_with_options) is given options
    /// that do not match the traced function's arity or shape — for example a
    /// `donate_argnums` index outside the flat input range, or an `in_shardings` length that
    /// doesn't match the number of flat inputs.
    #[error("invalid compilation options: {reason}")]
    InvalidCompilationOptions { reason: String },

    /// The runtime domain does not own the PJRT executable it was asked to invoke or install.
    #[error("xla executable belongs to a different PJRT client")]
    ExecutableClientMismatch,

    /// Error surfaced while encoding or decoding persistent compilation metadata.
    #[error("invalid persistent XLA executable: {reason}")]
    InvalidPersistentExecutable { reason: String },

    /// Error surfaced while normalizing backend analysis data.
    #[error("invalid XLA compilation analysis: {reason}")]
    InvalidCompilationAnalysis { reason: String },

    /// Runtime assertion host callbacks currently require host-accessible scalar buffers.
    #[error("compiled runtime assertions are not supported on XLA platform `{platform}`")]
    UnsupportedRuntimeAssertionPlatform { platform: String },

    /// The selected XLA invocation or reference ABI does not support the requested external-state configuration.
    #[error("unsupported XLA reference ABI: {reason}")]
    UnsupportedReferenceAbi { reason: String },

    /// Ordering, completion observation, or recovery of an effect scope failed.
    #[error("effect scope: {reason}")]
    EffectScope { reason: String },

    /// A submitted asynchronous XLA reference execution failed before its pending state became ready.
    #[error("asynchronous XLA reference execution failed: {reason}")]
    AsynchronousReferenceExecution { reason: String },

    /// A program statistics request names a case that is absent from the registry.
    #[error("unknown program statistics case `{case_id}`")]
    UnknownProgramStatisticsCase { case_id: String },

    /// Typed extension or experimental subsystem error, retaining its diagnostic and source chain.
    #[error("{0}")]
    Custom(#[source] Arc<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// Wraps a typed extension or experimental subsystem error in [`Self::Custom`]. The payload is shared when this
    /// error is cloned and can be recovered by [`Self::downcast_custom`].
    #[inline]
    pub fn custom(error: impl 'static + std::error::Error + Send + Sync) -> Self {
        Self::Custom(Arc::new(error))
    }

    /// Returns the original custom payload when this error is [`Self::Custom`] and its concrete type is `T`.
    /// Returns [`None`] for another payload type or any other variant.
    #[inline]
    pub fn downcast_custom<T: 'static + std::error::Error>(&self) -> Option<&T> {
        match self {
            // Downcast the payload rather than the `Arc` that shares it.
            Self::Custom(custom) => (&**custom as &dyn std::error::Error).downcast_ref::<T>(),
            _ => None,
        }
    }
}

impl From<ParameterError> for Error {
    fn from(error: ParameterError) -> Self {
        Self::Core(error.into())
    }
}

impl From<DataTypeError> for Error {
    fn from(error: DataTypeError) -> Self {
        Self::Core(error.into())
    }
}

impl From<LayoutError> for Error {
    fn from(error: LayoutError) -> Self {
        Self::Core(error.into())
    }
}

impl From<TypeError> for Error {
    fn from(error: TypeError) -> Self {
        Self::Core(error.into())
    }
}

impl From<BroadcastingError> for Error {
    fn from(error: BroadcastingError) -> Self {
        Self::Core(error.into())
    }
}

impl From<ShardingError> for Error {
    fn from(error: ShardingError) -> Self {
        Self::Core(error.into())
    }
}

impl From<ProgramError> for Error {
    fn from(error: ProgramError) -> Self {
        Self::Core(error.into())
    }
}

impl From<ReferenceError> for Error {
    fn from(error: ReferenceError) -> Self {
        Self::Core(error.into())
    }
}

impl From<BatchingError> for Error {
    fn from(error: BatchingError) -> Self {
        Self::Core(ProgramError::from(error).into())
    }
}

impl From<DifferentiationError> for Error {
    fn from(error: DifferentiationError) -> Self {
        Self::Core(ProgramError::from(error).into())
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(Arc::new(error))
    }
}

impl From<ryft_kernels::KernelError> for Error {
    fn from(error: ryft_kernels::KernelError) -> Self {
        Self::custom(error)
    }
}

impl From<KernelEmbeddingError> for Error {
    fn from(error: KernelEmbeddingError) -> Self {
        Self::custom(error)
    }
}

impl From<ryft_cuda::Error> for Error {
    fn from(error: ryft_cuda::Error) -> Self {
        Self::custom(error)
    }
}

#[cfg(feature = "rocm")]
impl From<ryft_rocm::Error> for Error {
    fn from(error: ryft_rocm::Error) -> Self {
        Self::custom(error)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StandardError;
    use std::io;

    use pretty_assertions::assert_eq;

    use crate::kernels::{DistributedKernelError, KernelAotError, KernelTuningError};

    use super::*;

    #[test]
    fn test_error_custom() {
        /// Extension payload that deliberately implements neither equality nor cloning.
        #[derive(Debug, thiserror::Error)]
        #[error("extension rejected input `{input}`")]
        struct ExtensionError {
            /// Exact invalid input retained for typed recovery.
            input: String,
        }

        let error = Error::custom(ExtensionError { input: "sample".into() });
        assert_eq!(error.to_string(), "extension rejected input `sample`");
        assert_eq!(format!("{error:?}"), "Custom(ExtensionError { input: \"sample\" })");
        assert_eq!(error.downcast_custom::<ExtensionError>().unwrap().input, "sample");

        let cloned = error.clone();
        assert!(std::ptr::eq(
            error.downcast_custom::<ExtensionError>().unwrap(),
            cloned.downcast_custom::<ExtensionError>().unwrap(),
        ));
        assert_eq!(error.source().unwrap().to_string(), "extension rejected input `sample`");
    }

    #[test]
    fn test_error_downcast_custom() {
        let error = Error::Custom(Arc::new(io::Error::new(io::ErrorKind::PermissionDenied, "extension is unreadable")));
        let payload = error.downcast_custom::<io::Error>().unwrap();
        assert_eq!(payload.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(payload.to_string(), "extension is unreadable");
        assert!(error.downcast_custom::<PjrtError>().is_none());
        assert!(Error::MissingMesh.downcast_custom::<io::Error>().is_none());
        assert!(Error::from(io::Error::other("ordinary I/O failure")).downcast_custom::<io::Error>().is_none());
    }

    #[test]
    fn test_error_from_foreign_errors() {
        let core = Error::from(ProgramError::PoisonedValue);
        assert!(matches!(core, Error::Core(CoreError::Program(ProgramError::PoisonedValue))));
        assert_eq!(core.to_string(), "encountered poisoned value where a live value was required");
        assert_eq!(format!("{core:?}"), "Core(Program(PoisonedValue))");

        let reference = Error::from(ReferenceError::Frozen);
        assert!(matches!(reference, Error::Core(CoreError::Reference(ReferenceError::Frozen))));
        assert_eq!(reference.to_string(), "reference is frozen");

        let pjrt = PjrtError::invalid_argument("runtime rejected input");
        let debug = format!("Pjrt({pjrt:?})");
        let pjrt = Error::from(pjrt);
        assert!(matches!(pjrt, Error::Pjrt(PjrtError::InvalidArgument { .. })));
        assert_eq!(pjrt.to_string(), "runtime rejected input");
        assert_eq!(format!("{pjrt:?}"), debug);

        let mlir = MlirError::invalid_argument("invalid tensor type");
        let debug = format!("Mlir({mlir:?})");
        let mlir = Error::from(mlir);
        assert!(matches!(mlir, Error::Mlir(MlirError::InvalidArgument { .. })));
        assert_eq!(mlir.to_string(), "invalid tensor type");
        assert_eq!(format!("{mlir:?}"), debug);
    }

    #[test]
    fn test_error_from_transform_errors() {
        let batching = Error::from(BatchingError::EmptyBatch);
        let Error::Core(CoreError::Program(program)) = batching else { unreachable!() };
        assert_eq!(program.downcast_custom::<BatchingError>(), Some(&BatchingError::EmptyBatch));

        let differentiation = Error::from(DifferentiationError::EmptyInput);
        let Error::Core(CoreError::Program(program)) = differentiation else { unreachable!() };
        assert_eq!(program.downcast_custom::<DifferentiationError>(), Some(&DifferentiationError::EmptyInput));

        for error in [
            Error::from(BatchingError::Program(ProgramError::PoisonedValue)),
            Error::from(DifferentiationError::Program(ProgramError::PoisonedValue)),
        ] {
            assert!(matches!(error, Error::Core(CoreError::Program(ProgramError::PoisonedValue))));
        }
    }

    #[test]
    fn test_error_from_io_error() {
        let source = io::Error::new(io::ErrorKind::PermissionDenied, "profile is unreadable");
        let debug = format!("Io({source:?})");
        let error = Error::from(source);
        let cloned = error.clone();
        assert_eq!(error.to_string(), "profile is unreadable");
        assert_eq!(format!("{error:?}"), debug);
        let (Error::Io(original), Error::Io(cloned)) = (error, cloned) else { unreachable!() };
        assert_eq!(original.kind(), io::ErrorKind::PermissionDenied);
        assert!(Arc::ptr_eq(&original, &cloned));
    }

    #[test]
    fn test_error_from_kernel_error() {
        let error = Error::from(ryft_kernels::KernelError::ParameterRank { array_rank: 3, block_rank: 2 });
        assert_eq!(error.to_string(), "kernel parameter has array rank 3 but block rank 2");
        assert_eq!(
            error.downcast_custom::<ryft_kernels::KernelError>(),
            Some(&ryft_kernels::KernelError::ParameterRank { array_rank: 3, block_rank: 2 }),
        );
    }

    #[test]
    fn test_error_from_kernel_embedding_error() {
        let error = Error::from(KernelEmbeddingError::Compiler(Box::new(io::Error::other("compiler failed"))));
        assert_eq!(error.to_string(), "kernel compilation failed: compiler failed");
        let kernel = error.downcast_custom::<KernelEmbeddingError>().unwrap();
        assert!(matches!(kernel, KernelEmbeddingError::Compiler(_)));
        assert_eq!(kernel.source().unwrap().to_string(), "compiler failed");
    }

    #[test]
    fn test_error_from_experimental_kernel_errors() {
        let aot = Error::from(KernelAotError::Invalid { message: "invalid manifest".into() });
        assert_eq!(aot.to_string(), "invalid kernel AOT bundle: invalid manifest");
        assert!(matches!(
            aot.downcast_custom::<KernelAotError>(),
            Some(KernelAotError::Invalid { message }) if message == "invalid manifest",
        ));

        let tuning = Error::from(KernelTuningError::Busy);
        assert_eq!(tuning.to_string(), "kernel tuner is busy");
        assert!(matches!(tuning.downcast_custom::<KernelTuningError>(), Some(KernelTuningError::Busy)));

        let distributed = Error::from(DistributedKernelError::Cancelled);
        assert_eq!(distributed.to_string(), "distributed kernel call cancelled");
        assert!(matches!(
            distributed.downcast_custom::<DistributedKernelError>(),
            Some(DistributedKernelError::Cancelled),
        ));
    }

    #[test]
    fn test_error_from_cuda_error() {
        let error = Error::from(ryft_cuda::Error::integration("missing CUDA context"));
        assert_eq!(error.to_string(), "missing CUDA context");
        assert!(matches!(
            error.downcast_custom::<ryft_cuda::Error>(),
            Some(ryft_cuda::Error::Integration { message, .. }) if message == "missing CUDA context",
        ));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_error_from_rocm_error() {
        let error = Error::from(ryft_rocm::Error::Unavailable { message: "missing HIP runtime".into() });
        assert_eq!(error.to_string(), "missing HIP runtime");
        assert!(matches!(
            error.downcast_custom::<ryft_rocm::Error>(),
            Some(ryft_rocm::Error::Unavailable { message }) if message == "missing HIP runtime",
        ));
    }
}
