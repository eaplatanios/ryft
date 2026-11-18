//! Crate error type of `ryft-xla`. [`Error`] is the single error type that public `ryft-xla` entry points return. It
//! groups conditions owned by this crate (e.g., [`LoweringError`] and [`CompilationError`]), keeps foreign `ryft-core`,
//! PJRT, and MLIR errors in their canonical carriers, and carries errors of typed extensions in [`Error::Custom`].

use std::sync::Arc;

use ryft_core::{
    AxisError, BatchingError, BroadcastingError, DataTypeError, DifferentiationError, DimensionError,
    Error as CoreError, LayoutError, ParameterError, ProgramError, ReferenceError, ShardingError, TypeError,
};
use ryft_mlir::Error as MlirError;
use ryft_pjrt::Error as PjrtError;

use crate::{ArrayError, CompilationError, ExecutionError, LoweringError, RuntimeError, SessionError};

/// Represents errors that can occur in the Ryft XLA backend. This type groups errors such as [`ArrayError`] and
/// [`LoweringError`] in dedicated variants. Errors from `ryft-core`, PJRT, and MLIR retain their original types.
/// Batching and differentiation errors first convert through [`ProgramError`], preserving the normalized core error.
/// Core and array errors are boxed to keep this type compact. Converting either error allocates one box, and cloning
/// the error clones its boxed value.
#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Core(Box<CoreError>),

    #[error(transparent)]
    Pjrt(#[from] PjrtError),

    #[error(transparent)]
    Mlir(#[from] MlirError),

    #[error(transparent)]
    Array(Box<ArrayError>),

    #[error(transparent)]
    Lowering(#[from] LoweringError),

    #[error(transparent)]
    Compilation(#[from] CompilationError),

    #[error(transparent)]
    Execution(#[from] ExecutionError),

    #[error(transparent)]
    Runtime(#[from] RuntimeError),

    #[error(transparent)]
    Session(#[from] SessionError),

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

impl From<CoreError> for Error {
    #[inline]
    fn from(error: CoreError) -> Self {
        Self::Core(Box::new(error))
    }
}

impl From<ArrayError> for Error {
    #[inline]
    fn from(error: ArrayError) -> Self {
        Self::Array(Box::new(error))
    }
}

impl From<ParameterError> for Error {
    #[inline]
    fn from(error: ParameterError) -> Self {
        Self::from(CoreError::from(error))
    }
}

impl From<DataTypeError> for Error {
    #[inline]
    fn from(error: DataTypeError) -> Self {
        Self::from(CoreError::from(error))
    }
}

impl From<LayoutError> for Error {
    #[inline]
    fn from(error: LayoutError) -> Self {
        Self::from(CoreError::from(error))
    }
}

impl From<BroadcastingError> for Error {
    #[inline]
    fn from(error: BroadcastingError) -> Self {
        Self::from(CoreError::from(error))
    }
}

impl From<ShardingError> for Error {
    #[inline]
    fn from(error: ShardingError) -> Self {
        Self::from(CoreError::from(error))
    }
}

impl From<AxisError> for Error {
    #[inline]
    fn from(error: AxisError) -> Self {
        Self::from(CoreError::from(error))
    }
}

impl From<TypeError> for Error {
    #[inline]
    fn from(error: TypeError) -> Self {
        Self::from(CoreError::from(error))
    }
}

impl From<DimensionError> for Error {
    #[inline]
    fn from(error: DimensionError) -> Self {
        Self::from(TypeError::from(error))
    }
}

impl From<ReferenceError> for Error {
    #[inline]
    fn from(error: ReferenceError) -> Self {
        Self::from(CoreError::from(error))
    }
}

impl From<ProgramError> for Error {
    #[inline]
    fn from(error: ProgramError) -> Self {
        Self::from(CoreError::from(error))
    }
}

impl From<BatchingError> for Error {
    #[inline]
    fn from(error: BatchingError) -> Self {
        Self::from(ProgramError::from(error))
    }
}

impl From<DifferentiationError> for Error {
    #[inline]
    fn from(error: DifferentiationError) -> Self {
        Self::from(ProgramError::from(error))
    }
}

impl From<std::io::Error> for Error {
    #[inline]
    fn from(error: std::io::Error) -> Self {
        Self::Io(Arc::new(error))
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StandardError;
    use std::io;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_error_size() {
        // Keep the shared carrier below Clippy's default large-error threshold.
        assert!(size_of::<Error>() < 128, "XLA error occupies {} bytes", size_of::<Error>());
    }

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

        let error = Error::Custom(Arc::new(io::Error::new(io::ErrorKind::PermissionDenied, "extension is unreadable")));
        let payload = error.downcast_custom::<io::Error>().unwrap();
        assert_eq!(payload.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(payload.to_string(), "extension is unreadable");
        assert!(error.downcast_custom::<PjrtError>().is_none());
        assert!(Error::from(ArrayError::MissingMesh).downcast_custom::<io::Error>().is_none());
        assert!(Error::from(io::Error::other("ordinary I/O failure")).downcast_custom::<io::Error>().is_none());
    }

    #[test]
    fn test_error_from_core_errors() {
        let source = ParameterError::MissingParameters { expected_count: 1, paths: None };
        let original = CoreError::from(BroadcastingError::ParameterError(source.clone()));
        let message = original.to_string();
        let debug = format!("Core({original:?})");
        let boxed = Error::from(original.clone());
        let Error::Core(inner) = &boxed else { unreachable!() };
        assert_eq!(inner.as_ref(), &original);
        assert_eq!(boxed.to_string(), message);
        assert_eq!(format!("{boxed:?}"), debug);
        assert_eq!(boxed.source().unwrap().downcast_ref::<ParameterError>(), Some(&source));

        let cloned = boxed.clone();
        let Error::Core(cloned_inner) = &cloned else { unreachable!() };
        assert_eq!(cloned_inner, inner);
        assert!(!std::ptr::eq(cloned_inner.as_ref(), inner.as_ref()));
        assert_eq!(cloned.to_string(), message);
        assert_eq!(format!("{cloned:?}"), debug);
        assert_eq!(cloned.source().unwrap().downcast_ref::<ParameterError>(), Some(&source));

        let core = Error::from(ProgramError::PoisonedValue);
        assert!(matches!(
            &core,
            Error::Core(core_error) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::PoisonedValue),
            ),
        ));
        assert_eq!(core.to_string(), "encountered poisoned value where a live value was required");
        assert_eq!(format!("{core:?}"), "Core(Program(PoisonedValue))");

        let reference = Error::from(ReferenceError::Frozen);
        assert!(matches!(
            &reference,
            Error::Core(core_error) if matches!(
                core_error.as_ref(),
                CoreError::Reference(ReferenceError::Frozen),
            ),
        ));
        assert_eq!(reference.to_string(), "reference is frozen");

        // Errors of the individual core components convert into the corresponding core error.
        let parameter = ParameterError::MissingParameters { expected_count: 2, paths: None };
        let layout = LayoutError::InvalidLayout { message: "tile is empty".into() };
        let sharding = ShardingError::EmptyMeshAxisName;
        let axis = AxisError::DuplicateAxis { axis: 1 };
        let r#type = TypeError::invalid("type is invalid");
        for (error, expected) in [
            (Error::from(parameter.clone()), CoreError::from(parameter)),
            (Error::from(layout.clone()), CoreError::from(layout)),
            (Error::from(sharding.clone()), CoreError::from(sharding)),
            (Error::from(axis.clone()), CoreError::from(axis)),
            (Error::from(r#type.clone()), CoreError::from(r#type)),
        ] {
            let Error::Core(core_error) = error else { unreachable!() };
            assert_eq!(*core_error, expected);
        }

        let dimension =
            DimensionError::ExtentExceedsBackendWidth { value: i32::MAX as usize + 1, maximum: i32::MAX as usize };
        let error = Error::from(dimension.clone());
        let Error::Core(core_error) = error else { unreachable!() };
        let CoreError::Type(type_error) = *core_error else { unreachable!() };
        assert_eq!(type_error.downcast_custom::<DimensionError>(), Some(&dimension));

        let batching = Error::from(BatchingError::EmptyBatch);
        let Error::Core(core_error) = batching else { unreachable!() };
        let CoreError::Program(program) = *core_error else { unreachable!() };
        assert_eq!(program.downcast_custom::<BatchingError>(), Some(&BatchingError::EmptyBatch));

        let differentiation = Error::from(DifferentiationError::EmptyInput);
        let Error::Core(core_error) = differentiation else { unreachable!() };
        let CoreError::Program(program) = *core_error else { unreachable!() };
        assert_eq!(program.downcast_custom::<DifferentiationError>(), Some(&DifferentiationError::EmptyInput));

        for error in [
            Error::from(BatchingError::Program(ProgramError::PoisonedValue)),
            Error::from(DifferentiationError::Program(ProgramError::PoisonedValue)),
        ] {
            assert!(matches!(
                error,
                Error::Core(core_error) if matches!(
                    core_error.as_ref(),
                    CoreError::Program(ProgramError::PoisonedValue),
                ),
            ));
        }
    }

    #[test]
    fn test_error_from_backend_errors() {
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

        let dimension =
            DimensionError::ExtentExceedsBackendWidth { value: i32::MAX as usize + 1, maximum: i32::MAX as usize };
        let cause = ArrayError::InvalidShardDimension { shard_index: 3, dimension: 1, source: dimension.clone() };
        let debug = format!("Array({cause:?})");
        let array = Error::from(cause);
        let Error::Array(inner) = &array else { unreachable!() };
        assert!(matches!(inner.as_ref(),
            ArrayError::InvalidShardDimension { shard_index: 3, dimension: 1, source } if source == &dimension,
        ));
        assert_eq!(array.to_string(), format!("cross-host transfer for shard #3, dimension #1: {dimension}"));
        assert_eq!(format!("{array:?}"), debug);
        assert_eq!(array.source().unwrap().downcast_ref::<DimensionError>(), Some(&dimension));
        assert!(array.downcast_custom::<DimensionError>().is_none());

        let cloned = array.clone();
        assert_eq!(cloned.to_string(), array.to_string());
        assert_eq!(format!("{cloned:?}"), debug);
        assert_eq!(cloned.source().unwrap().downcast_ref::<DimensionError>(), Some(&dimension));

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
}
