use thiserror::Error;

use crate::parameters::ParameterError;
use crate::types::data_type::DataTypeError;

/// Represents errors that can occur in `ryft-core`.
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Error {
    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error(transparent)]
    DataType(#[from] DataTypeError),
}
