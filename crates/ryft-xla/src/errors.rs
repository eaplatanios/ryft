use thiserror::Error;

use ryft_core::{
    ArrayType, BroadcastingError, DataTypeError, Error as CoreError, LayoutError, ParameterError, Shape, ShardingError,
    TypeError,
};
use ryft_pjrt::{DeviceId, Error as PjrtError};

/// Represents errors that can occur in `ryft-xla`.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum Error {
    #[error(transparent)]
    CoreError(#[from] CoreError),

    #[error(transparent)]
    PjrtError(#[from] PjrtError),

    #[error("size {size} exceeds the maximum allowed size of {}", usize::MAX)]
    SizeLimitExceeded { size: u64 },

    #[error("missing required sharding metadata")]
    MissingSharding,

    #[error("expected static shape but got {shape}")]
    DynamicShape { shape: Shape },

    #[error("got multiple buffers for device {device_id}")]
    MultipleBuffersOnDevice { device_id: DeviceId },

    #[error("device {device_id} is not present in the device mesh")]
    DeviceNotInMesh { device_id: DeviceId },

    #[error("device {device_id} reports process index {actual_process_index}, but expected {expected_process_index}")]
    DeviceProcessIndexMismatch { device_id: DeviceId, expected_process_index: usize, actual_process_index: usize },

    #[error("buffer has type {actual}, but expected {expected}")]
    BufferTypeMismatch { expected: ArrayType, actual: ArrayType },
}

impl From<ParameterError> for Error {
    fn from(error: ParameterError) -> Self {
        Self::CoreError(error.into())
    }
}

impl From<DataTypeError> for Error {
    fn from(error: DataTypeError) -> Self {
        Self::CoreError(error.into())
    }
}

impl From<LayoutError> for Error {
    fn from(error: LayoutError) -> Self {
        Self::CoreError(error.into())
    }
}

impl From<TypeError> for Error {
    fn from(error: TypeError) -> Self {
        Self::CoreError(error.into())
    }
}

impl From<BroadcastingError> for Error {
    fn from(error: BroadcastingError) -> Self {
        Self::CoreError(error.into())
    }
}

impl From<ShardingError> for Error {
    fn from(error: ShardingError) -> Self {
        Self::CoreError(error.into())
    }
}
