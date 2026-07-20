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
            return Err($crate::TypeError { message: format!("expected {expected} {noun} but got {count}") });
        }
    }};
}

/// Checks types against a structural or semantic type contract. All forms use an `@` selector and return
/// [`TypeError`](crate::TypeError)s as appropriate, converted into the enclosing function's error type, when the
/// selected contract is not satisfied. Data-type selectors compose by intersection when written next to one another.
/// For example, `@numeric @real` accepts real numeric types, while `@float @real` accepts real floating-point types.
/// The available selectors are:
///
///   - `@same`: Requires the provided expected and actual flat type signatures to be identical.
///   - `@numeric`: Accepts integer, floating-point, and complex [`DataType`](crate::DataType)s.
///   - `@float`: Accepts floating-point and complex [`DataType`](crate::DataType)s.
///   - `@real`: Excludes complex [`DataType`](crate::DataType)s and is intended to refine `@numeric` or `@float`.
///   - `@no_unreduced`: Rejects [`ArrayType`](crate::ArrayType)s carrying any unreduced mesh axes.
///   - `@same_unreduced_axes`: Requires exactly two [`ArrayType`](crate::ArrayType)s with matching unreduced-axis sets.
///   - `@same_reduced_axes`: Requires exactly two [`ArrayType`](crate::ArrayType)s with matching reduced-axis sets.
///
/// # Examples
///
/// Compose selectors to express the intersection of their contracts. Selector order does not affect the accepted types,
/// so both invocations below accept real numeric data types and reject Boolean, token, zero-space, and complex types:
///
/// ```rust,ignore
/// check_types!(@numeric @real, "maximum", input_types);
/// check_types!(@real @numeric, "maximum", input_types);
/// ```
///
/// # Parameters
///
///   - `$selectors`: One structural selector, or one or more composable [`DataType`](crate::DataType) selectors
///     identifying the contract to validate.
///   - `$descriptor`: Expression evaluating to a string that identifies the checked operation or signature in errors.
///   - `$types`: Expression evaluating to the data or array types checked by `$selector`.
///   - `$signatures`: Bracketed pair containing the expected and actual flat type signatures checked by `@same`.
#[macro_export]
macro_rules! check_types {
    (@same, $descriptor:expr, [$expected:expr, $actual:expr $(,)?] $(,)?) => {{
        let expected = &$expected[..];
        let actual = &$actual[..];
        if expected != actual {
            return Err($crate::TypeError {
                message: format!(
                    "{} type signature mismatch: expected [{}] but got [{}]",
                    $descriptor,
                    expected.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                    actual.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                ),
            });
        }
    }};

    (@no_unreduced, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if types.iter().any(|r#type| !r#type.unreduced_axes().is_empty()) {
            return Err(
                $crate::TypeError { message: format!("'{descriptor}' does not support unreduced operands") }.into()
            );
        }
    }};

    (@same_unreduced_axes, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if types.len() != 2 {
            return Err($crate::TypeError { message: format!("expected 2 inputs but got {}", types.len()) }.into());
        }
        if types[0].unreduced_axes() != types[1].unreduced_axes() {
            return Err($crate::TypeError {
                message: format!("'{descriptor}' operands must be unreduced over the same axes"),
            }
            .into());
        }
    }};

    (@same_reduced_axes, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if types.len() != 2 {
            return Err($crate::TypeError { message: format!("expected 2 inputs but got {}", types.len()) }.into());
        }
        if types[0].reduced_axes() != types[1].reduced_axes() {
            return Err($crate::TypeError {
                message: format!("'{descriptor}' operands must be reduced over the same axes"),
            }
            .into());
        }
    }};

    ($(@$selector:ident)+, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if let Some(input_type) = types.iter().find(|input_type| {
            !$crate::check_types!(@matches_data_type input_type; $(@$selector)+)
        }) {
            return Err($crate::TypeError {
                message: format!("'{descriptor}' does not support input data type {input_type}"),
            }
            .into());
        }
    }};

    // This internal helper terminates a composed data-type contract after every predicate has accepted the candidate.
    // It supplies the `true` identity needed to combine an arbitrary number of selectors with logical conjunction.
    (@matches_data_type $input_type:ident;) => {
        true
    };

    // This internal helper accepts the numeric universe: signed and unsigned integers, real floating-point values,
    // and complex values. It recurses so later selectors can refine that universe without duplicating its variant list.
    (@matches_data_type $input_type:ident; @numeric $($selectors:tt)*) => {
        !matches!(
            $input_type,
            $crate::DataType::Token | $crate::DataType::Zero | $crate::DataType::Boolean
        ) && $crate::check_types!(@matches_data_type $input_type; $($selectors)*)
    };

    // This internal helper accepts real floating-point and complex types as one float-capable universe. Keeping this
    // predicate independent from `@real` lets callers retain complex values with `@float` or exclude them by composing
    // `@float @real`.
    (@matches_data_type $input_type:ident; @float $($selectors:tt)*) => {
        matches!(
            $input_type,
            $crate::DataType::F4E2M1FN
                | $crate::DataType::F6E2M3FN
                | $crate::DataType::F6E3M2FN
                | $crate::DataType::F8E3M4
                | $crate::DataType::F8E4M3
                | $crate::DataType::F8E4M3FN
                | $crate::DataType::F8E4M3FNUZ
                | $crate::DataType::F8E4M3B11FNUZ
                | $crate::DataType::F8E5M2
                | $crate::DataType::F8E5M2FNUZ
                | $crate::DataType::F8E8M0FNU
                | $crate::DataType::BF16
                | $crate::DataType::F16
                | $crate::DataType::F32
                | $crate::DataType::F64
                | $crate::DataType::C64
                | $crate::DataType::C128
        ) && $crate::check_types!(@matches_data_type $input_type; $($selectors)*)
    };

    // This internal helper excludes complex types from the universe established by preceding or following selectors.
    // Its independent predicate makes selector order irrelevant and supports both `@numeric @real` and `@float @real`
    // without compound selector names.
    (@matches_data_type $input_type:ident; @real $($selectors:tt)*) => {
        !matches!($input_type, $crate::DataType::C64 | $crate::DataType::C128)
            && $crate::check_types!(@matches_data_type $input_type; $($selectors)*)
    };
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

/// Checks that [`ProgramBuilder`](crate::ProgramBuilder) handles refer to the same builder and returns a
/// [`ProgramError::MismatchedProgramBuilders`](crate::ProgramError::MismatchedProgramBuilders) if they do not.
///
/// # Parameters
///
///   - `$reference`: Expression evaluating to the reference [`ProgramBuilder`](crate::ProgramBuilder) handle.
///   - `$other`: Expression evaluating to a single [`ProgramBuilder`](crate::ProgramBuilder) handle, or bracketed
///     syntax `[$others]` where `$others` evaluates to an iterable of [`ProgramBuilder`](crate::ProgramBuilder)
///     handles.
#[macro_export]
macro_rules! check_builders {
    ($reference:expr, [$others:expr] $(,)?) => {{
        let reference = $reference;
        let mut result = ::std::result::Result::Ok(());
        for other in $others {
            if !::std::rc::Rc::ptr_eq(reference, other) {
                result = ::std::result::Result::Err($crate::ProgramError::MismatchedProgramBuilders);
                break;
            }
        }
        result
    }};
    ($reference:expr, $other:expr $(,)?) => {{
        let reference = $reference;
        let other = $other;
        if ::std::rc::Rc::ptr_eq(reference, other) {
            ::std::result::Result::Ok(())
        } else {
            ::std::result::Result::Err($crate::ProgramError::MismatchedProgramBuilders)
        }
    }};
}

pub use crate::{check_builders, check_count, check_sharding, check_types};
