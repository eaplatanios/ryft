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

    #[error("{message}")]
    SizeLimitExceeded { message: String },

    #[error("expected {expected} byte(s) but got {got}")]
    ByteCountMismatch { expected: usize, got: usize },

    #[error("missing required sharding metadata")]
    MissingSharding,

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

    #[error("shape rank {shape_rank} does not match shard slice rank {slice_rank}")]
    ShardSliceRankMismatch { shape_rank: usize, slice_rank: usize },
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
