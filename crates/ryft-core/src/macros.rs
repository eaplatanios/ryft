/// Checks that `inputs` contains exactly `expected` values. This macro is intended for traced operation and transform
/// helpers that already return `Result<_, TracingError>`. When the input arity does not match, it returns early from
/// the enclosing function with [`TracingError::InvalidInputCount`](crate::tracing_v2::TracingError::InvalidInputCount).
#[macro_export]
macro_rules! check_input_count {
    ($inputs:expr, $expected:expr $(,)?) => {{
        let inputs = &$inputs;
        let expected = $expected;
        if inputs.len() != expected {
            return Err($crate::tracing_v2::TracingError::InvalidInputCount { expected, got: inputs.len() });
        }
    }};
}

/// Checks that all batched inputs carry the same number of lanes. This macro is intended for traced batching rules that
/// already return `Result<_, TracingError>`. Empty or singleton input slices always pass. When any lane count differs
/// from the first batch, it returns early from the enclosing function with
/// [`TracingError::MismatchedBatchSize`](crate::tracing_v2::TracingError::MismatchedBatchSize).
#[macro_export]
macro_rules! check_batch_sizes {
    ($inputs:expr $(,)?) => {{
        let inputs = &$inputs;
        if let Some(first_input) = inputs.first() {
            let expected_lane_count = first_input.len();
            if inputs.iter().skip(1).any(|input| input.len() != expected_lane_count) {
                return Err($crate::tracing_v2::TracingError::MismatchedBatchSize);
            }
        }
    }};
}

pub use crate::{check_batch_sizes, check_input_count};
