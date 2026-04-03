use thiserror::Error;

use crate::broadcasting::BroadcastingError;
use crate::parameters::ParameterError;
use crate::sharding::ShardingError;
use crate::types::{DataTypeError, LayoutError};

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
    Broadcasting(#[from] BroadcastingError),

    #[error(transparent)]
    Sharding(#[from] ShardingError),
}
