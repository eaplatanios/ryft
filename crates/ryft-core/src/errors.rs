use thiserror::Error;

use crate::batching::BatchingError;
use crate::broadcasting::BroadcastingError;
use crate::parameters::ParameterError;
use crate::sharding::ShardingError;
use crate::types::{DataTypeError, LayoutError, TypeError};

/// Represents errors that can occur in `ryft-core`.
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Error {
    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error(transparent)]
    DataType(#[from] DataTypeError),

    #[error(transparent)]
    Layout(#[from] LayoutError),

    #[error(transparent)]
    Type(#[from] TypeError),

    #[error(transparent)]
    Broadcasting(#[from] BroadcastingError),

    #[error(transparent)]
    Batching(#[from] BatchingError),

    #[error(transparent)]
    Sharding(#[from] ShardingError),
}
