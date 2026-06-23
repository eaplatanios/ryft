use ryft_core::tracing_v2::{ArrayOperation, LinearArrayOperation};

/// Ordinary staged operation type used by the ndarray backend.
pub type NdarrayOperation<V> = ArrayOperation<V>;

/// Linear staged operation type used by the ndarray backend.
pub type LinearNdarrayOperation<V, C = V, Factor = V> = LinearArrayOperation<V, C, Factor, NdarrayOperation<C>>;
