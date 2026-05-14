use thiserror::Error;

use ryft_core::{DataTypeError, ParameterError, ShardingError};
use ryft_pjrt::{DeviceId, Error as PjrtError};

/// Represents errors that can occur in `ryft-xla`.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum Error {
    #[error("{0}")]
    PjrtError(#[from] PjrtError),

    #[error("{0}")]
    DataTypeError(#[from] DataTypeError),

    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    #[error("{0}")]
    ParameterError(#[from] ParameterError),

    #[error("got multiple buffers for device {device_id}")]
    MultipleBuffersOnDevice { device_id: DeviceId },

    #[error("device {device_id} is not present in the device mesh")]
    DeviceNotInMesh { device_id: DeviceId },
}
