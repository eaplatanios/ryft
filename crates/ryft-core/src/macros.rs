/// Checks that `values` contains exactly `expected` entries and, if not, returns an error of the specified type.
#[macro_export]
macro_rules! check_count {
    ("input", $values:expr, $expected:expr, TracingError $(,)?) => {{
        let values = &$values;
        let expected = $expected;
        if values.len() != expected {
            return Err($crate::tracing::TracingError::InvalidInputCount { expected, got: values.len() }.into());
        }
    }};
    ("output", $values:expr, $expected:expr, TracingError $(,)?) => {{
        let values = &$values;
        let expected = $expected;
        if values.len() != expected {
            return Err($crate::tracing::TracingError::InvalidOutputCount { expected, got: values.len() }.into());
        }
    }};
    ($descriptor:expr, $values:expr, $expected:expr, TypeError $(,)?) => {{
        let values = &$values;
        let expected = $expected;
        if values.len() != expected {
            let count = values.len();
            let descriptor = $descriptor;
            let noun = if expected == 1 { descriptor.to_string() } else { format!("{descriptor}s") };
            return Err($crate::types::TypeError { message: format!("expected {expected} {noun} but got {count}") });
        }
    }};
}

pub use crate::check_count;
