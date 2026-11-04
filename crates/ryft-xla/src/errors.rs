use std::sync::Arc;

use ryft_core::{
    BatchingError, BroadcastingError, DataTypeError, DifferentiationError, DimensionError, Error as CoreError,
    LayoutError, ParameterError, ProgramError, ReferenceError, ShardingError, TypeError,
};
use ryft_mlir::Error as MlirError;
use ryft_pjrt::Error as PjrtError;

use crate::kernels::KernelEmbeddingError;
use crate::{ArrayError, CompilationError, ExecutionError, LoweringError};

/// Errors returned by array preparation, tracing, lowering, compilation, and execution in `ryft-xla`. XLA-owned
/// conditions are grouped by their owning contracts, while foreign errors keep their canonical carriers. Public entry
/// points return this type even when their failure is owned by a more specific group. Batching and differentiation
/// errors first convert through [`ProgramError`], preserving normalized core payloads.
#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Core(#[from] CoreError),

    #[error(transparent)]
    Pjrt(#[from] PjrtError),

    #[error(transparent)]
    Mlir(#[from] MlirError),

    #[error(transparent)]
    Array(#[from] ArrayError),

    #[error(transparent)]
    Lowering(#[from] LoweringError),

    #[error(transparent)]
    Compilation(#[from] CompilationError),

    #[error(transparent)]
    Execution(#[from] ExecutionError),

    #[error(transparent)]
    Io(Arc<std::io::Error>),

    #[error("{message}")]
    SizeLimitExceeded { message: String },

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

// TODO(eaplatanios): Review from here onwards.

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

impl From<DimensionError> for Error {
    fn from(error: DimensionError) -> Self {
        Self::Core(TypeError::from(error).into())
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
        assert!(Error::from(ArrayError::MissingMesh).downcast_custom::<io::Error>().is_none());
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
    fn test_error_from_grouped_errors() {
        let dimension =
            DimensionError::ExtentExceedsBackendWidth { value: i32::MAX as usize + 1, maximum: i32::MAX as usize };
        let array =
            Error::from(ArrayError::InvalidShardDimension { shard_index: 3, dimension: 1, source: dimension.clone() });
        assert!(matches!(&array,
            Error::Array(ArrayError::InvalidShardDimension { shard_index: 3, dimension: 1, source })
                if source == &dimension,
        ));
        assert_eq!(array.source().unwrap().downcast_ref::<DimensionError>(), Some(&dimension));
        assert!(array.downcast_custom::<DimensionError>().is_none());

        let lowering = Error::from(LoweringError::InvalidFunctionName { function_name: "bad name".into() });
        assert!(matches!(lowering,
            Error::Lowering(LoweringError::InvalidFunctionName { function_name }) if function_name == "bad name",
        ));

        let compilation =
            Error::from(CompilationError::ShardingCountMismatch { kind: "argument", expected: 2, actual: 1 });
        assert!(matches!(
            compilation,
            Error::Compilation(CompilationError::ShardingCountMismatch { kind: "argument", expected: 2, actual: 1 }),
        ));

        let execution = Error::from(ExecutionError::DonationFlagCountMismatch { expected_count: 2, actual_count: 1 });
        assert!(matches!(
            execution,
            Error::Execution(ExecutionError::DonationFlagCountMismatch { expected_count: 2, actual_count: 1 }),
        ));
    }

    #[test]
    fn test_error_from_dimension_error() {
        let dimension =
            DimensionError::ExtentExceedsBackendWidth { value: i32::MAX as usize + 1, maximum: i32::MAX as usize };
        let error = Error::from(dimension.clone());
        let Error::Core(CoreError::Type(type_error)) = error else { unreachable!() };
        assert_eq!(type_error.downcast_custom::<DimensionError>(), Some(&dimension));
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
