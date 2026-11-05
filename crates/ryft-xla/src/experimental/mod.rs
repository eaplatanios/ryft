use crate::Error;
use crate::kernels::KernelEmbeddingError;

// TODO(eaplatanios): Review this module.

/// StableHLO and Shardy lowering helpers for traced XLA programs.
pub mod lowering;
/// Backend-owned staged operation types for traced XLA programs.
pub mod ops;
/// General XLA program tracing: the [`trace`] entry point, its [`TracedProgram`] result, which lowers the traced
/// program to StableHLO/Shardy MLIR for compilation and execution, and the [`Error`](crate::Error) of both stages.
pub mod tracing;

pub use tracing::{ArrayTracer, TracedProgram, trace, trace_with_context};

// Experimental kernel integrations retain their concrete error payloads in the common custom carrier.
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

    use super::*;

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
