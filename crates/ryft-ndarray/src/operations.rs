use ryft_core::tracing_v2::{ArrayOperation, LinearArrayOperation};
use ryft_core::types::ArrayType;

use crate::arrays::Array;

/// Ordinary staged operation carrier used by the ndarray backend.
pub type NdarrayOperation<V> = ArrayOperation<V, ArrayType>;

/// Ordinary staged operation carrier specialized to an ndarray element type.
pub type NdarrayElementOperation<T> = NdarrayOperation<Array<T>>;

/// Linear staged operation carrier used by the ndarray backend.
pub type LinearNdarrayOperation<V> = LinearArrayOperation<V, ArrayType>;

/// Linear staged operation carrier specialized to an ndarray element type.
pub type LinearNdarrayElementOperation<T> = LinearNdarrayOperation<Array<T>>;
