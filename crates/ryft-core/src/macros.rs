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

pub use crate::check_input_count;
