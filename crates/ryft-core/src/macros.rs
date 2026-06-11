/// Checks that `values` contains exactly `expected` entries and, if not, returns an error of the specified type.
#[macro_export]
macro_rules! check_count {
    ("input", $values:expr, $expected:expr, ProgramError $(,)?) => {{
        let values = &$values;
        let expected = $expected;
        if values.len() != expected {
            return Err($crate::ProgramError::InvalidInputCount { expected, actual: values.len() }.into());
        }
    }};
    ("output", $values:expr, $expected:expr, ProgramError $(,)?) => {{
        let values = &$values;
        let expected = $expected;
        if values.len() != expected {
            return Err($crate::ProgramError::InvalidOutputCount { expected, actual: values.len() }.into());
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

/// Checks that two flat type signatures are identical and, if not, returns a [`TypeError`](crate::TypeError)
/// whose message names the mismatching descriptor.
///
/// # Parameters
///
///   - `descriptor`: Expression evaluating to a string that names the validated signature in the error message.
///   - `$left`: Expression evaluating to a slice of [`Type`](crate::Type)s.
///   - `$right`: Expression evaluating to a slice of [`Type`](crate::Type)s.
#[macro_export]
macro_rules! check_types {
    ($descriptor:expr, $left:expr, $right:expr $(,)?) => {{
        let left = &$left[..];
        let right = &$right[..];
        if left != right {
            return Err($crate::types::TypeError {
                message: format!(
                    "{} type signature mismatch: expected [{}] but got [{}]",
                    $descriptor,
                    left.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                    right.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                ),
            });
        }
    }};
}

/// Checks that a concrete [`DeviceMesh`](crate::DeviceMesh) and a [`Sharding`](crate::Sharding) refer to the same
/// [`LogicalMesh`](crate::LogicalMesh). If the logical meshes differ, the macro returns a
/// [`ShardingError::MeshMismatch`](crate::ShardingError::MeshMismatch) converted into the enclosing function's error
/// type using [`Into::into`]. Use this macro in functions that return a [`Result`] whose error type can be constructed
/// from [`ShardingError`](crate::ShardingError).
///
/// # Parameters
///
///   - `$mesh`: Expression evaluating to a [`DeviceMesh`](crate::DeviceMesh) or a reference to one.
///   - `$sharding`: Expression evaluating to a [`Sharding`](crate::Sharding) or a reference to one.
#[macro_export]
macro_rules! check_sharding {
    ($mesh:expr, $sharding:expr $(,)?) => {{
        let mesh = &$mesh;
        let sharding = &$sharding;
        if mesh.logical_mesh() != sharding.mesh() {
            return Err($crate::ShardingError::MeshMismatch {
                expected: mesh.logical_mesh().clone(),
                actual: sharding.mesh().clone(),
            }
            .into());
        }
    }};
}

pub use crate::{check_count, check_sharding, check_types};
