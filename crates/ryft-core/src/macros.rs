/// Checks that `inputs` contains exactly `expected` values and, if not, returns an error of the specified type.
#[macro_export]
macro_rules! check_input_count {
    ($inputs:expr, $expected:expr, TracingError $(,)?) => {{
        let inputs = &$inputs;
        let expected = $expected;
        if inputs.len() != expected {
            return Err($crate::tracing::TracingError::InvalidInputCount { expected, got: inputs.len() });
        }
    }};
    ($inputs:expr, $expected:expr, TypeError $(,)?) => {{
        let inputs = &$inputs;
        let expected = $expected;
        if inputs.len() != expected {
            let input_count = inputs.len();
            let input_noun = if expected == 1 { "input" } else { "inputs" };
            return Err($crate::types::TypeError {
                message: format!("expected {expected} {input_noun} but got {input_count}"),
            });
        }
    }};
}

pub use crate::check_input_count;
