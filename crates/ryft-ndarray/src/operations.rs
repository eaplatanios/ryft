use std::convert::Infallible;

use ryft_core::tracing_v2::{ArrayOperation, LinearArrayOperation};
use ryft_core::types::ArrayType;

/// Ordinary staged operation type used by the ndarray backend.
pub type NdarrayOperation<V> = ArrayOperation<V, ArrayType>;

/// Linear staged operation type used by the ndarray backend.
pub type LinearNdarrayOperation<V, Factor = V> = LinearArrayOperation<V, ArrayType, Infallible, Factor>;
