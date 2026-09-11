/// Dispatches one runtime [`DataType`](crate::arrays::DataType) to its sealed Rust array element type, so that array
/// kernels can run generic element code without storing or allocating another dynamic element representation.
///
/// The macro takes a class selector, a [`DataType`](crate::arrays::DataType) expression, and an `|Element| body`
/// closure-like form, and expands to a `match` that binds the type alias `Element` to the matching element type from
/// [`arrays::encoding`](crate::arrays::encoding) in every selected arm. The body is instantiated once per selected
/// element type and may therefore use `Element` in any type position, including calls to generic functions bounded
/// by [`ArrayElement`](crate::arrays::encoding::ArrayElement) or by kernel capability traits:
///
/// ```
/// # use ryft_core::arrays::{DataType, dispatch_on_array_element_type};
///
/// fn element_byte_count(data_type: DataType) -> usize {
///     dispatch_on_array_element_type!(data_type, |Element| size_of::<Element>())
/// }
///
/// assert_eq!(element_byte_count(DataType::F8E4M3FN), 1);
/// assert_eq!(element_byte_count(DataType::C128), 16);
/// ```
///
/// Omitting the class selector dispatches on every element type, including Booleans and complex numbers. The class
/// selectors restrict which element types the body is instantiated for, so that bodies bounded by class-specific
/// capability traits still compile:
///
///   - `@numeric`: Every element type except Booleans.
///   - `@ordered`: Every partially ordered element type, namely every element type except the unordered complex ones.
///   - `@real`: Every integer and floating-point element type (no Booleans and no complex numbers).
///   - `@integer`: Every sub-byte and primitive integer element type.
///   - `@signed`: Every signed sub-byte and primitive integer element type.
///   - `@float`: Every low-precision, half-precision, and primitive floating-point element type.
///   - `@complex`: Every complex element type.
///   - `@boolean_or_integer`: Boolean and every integer element type (i.e., the bitwise and logical family).
///
/// Kernels dispatch after type inference has already validated the operand element class,
/// so a [`DataType`](crate::arrays::DataType) outside the selected class (including the payload-free
/// [`Token`](crate::arrays::DataType::Token) and [`Zero`](crate::arrays::DataType::Zero) types, which no selector
/// includes) is an internal invariant violation and panics with a descriptive message rather than forcing every body
/// to return a [`Result`].
#[macro_export]
macro_rules! dispatch_on_array_element_type {
    ($data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (Boolean, bool),
            (I1, $crate::arrays::encoding::i1),
            (I2, $crate::arrays::encoding::i2),
            (I4, $crate::arrays::encoding::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::encoding::u1),
            (U2, $crate::arrays::encoding::u2),
            (U4, $crate::arrays::encoding::u4),
            (U8, u8),
            (U16, u16),
            (U32, u32),
            (U64, u64),
            (F4E2M1FN, $crate::arrays::encoding::f4e2m1fn),
            (F6E2M3FN, $crate::arrays::encoding::f6e2m3fn),
            (F6E3M2FN, $crate::arrays::encoding::f6e3m2fn),
            (F8E3M4, $crate::arrays::encoding::f8e3m4),
            (F8E4M3, $crate::arrays::encoding::f8e4m3),
            (F8E4M3FN, $crate::arrays::encoding::f8e4m3fn),
            (F8E4M3FNUZ, $crate::arrays::encoding::f8e4m3fnuz),
            (F8E4M3B11FNUZ, $crate::arrays::encoding::f8e4m3b11fnuz),
            (F8E5M2, $crate::arrays::encoding::f8e5m2),
            (F8E5M2FNUZ, $crate::arrays::encoding::f8e5m2fnuz),
            (F8E8M0FNU, $crate::arrays::encoding::f8e8m0fnu),
            (BF16, $crate::arrays::encoding::bf16),
            (F16, $crate::arrays::encoding::f16),
            (F32, f32),
            (F64, f64),
            (C64, $crate::arrays::encoding::Complex<f32>),
            (C128, $crate::arrays::encoding::Complex<f64>),
        ) $data_type, |$element| $body)
    };

    (@numeric $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (I1, $crate::arrays::encoding::i1),
            (I2, $crate::arrays::encoding::i2),
            (I4, $crate::arrays::encoding::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::encoding::u1),
            (U2, $crate::arrays::encoding::u2),
            (U4, $crate::arrays::encoding::u4),
            (U8, u8),
            (U16, u16),
            (U32, u32),
            (U64, u64),
            (F4E2M1FN, $crate::arrays::encoding::f4e2m1fn),
            (F6E2M3FN, $crate::arrays::encoding::f6e2m3fn),
            (F6E3M2FN, $crate::arrays::encoding::f6e3m2fn),
            (F8E3M4, $crate::arrays::encoding::f8e3m4),
            (F8E4M3, $crate::arrays::encoding::f8e4m3),
            (F8E4M3FN, $crate::arrays::encoding::f8e4m3fn),
            (F8E4M3FNUZ, $crate::arrays::encoding::f8e4m3fnuz),
            (F8E4M3B11FNUZ, $crate::arrays::encoding::f8e4m3b11fnuz),
            (F8E5M2, $crate::arrays::encoding::f8e5m2),
            (F8E5M2FNUZ, $crate::arrays::encoding::f8e5m2fnuz),
            (F8E8M0FNU, $crate::arrays::encoding::f8e8m0fnu),
            (BF16, $crate::arrays::encoding::bf16),
            (F16, $crate::arrays::encoding::f16),
            (F32, f32),
            (F64, f64),
            (C64, $crate::arrays::encoding::Complex<f32>),
            (C128, $crate::arrays::encoding::Complex<f64>),
        ) $data_type, |$element| $body)
    };

    (@ordered $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (Boolean, bool),
            (I1, $crate::arrays::encoding::i1),
            (I2, $crate::arrays::encoding::i2),
            (I4, $crate::arrays::encoding::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::encoding::u1),
            (U2, $crate::arrays::encoding::u2),
            (U4, $crate::arrays::encoding::u4),
            (U8, u8),
            (U16, u16),
            (U32, u32),
            (U64, u64),
            (F4E2M1FN, $crate::arrays::encoding::f4e2m1fn),
            (F6E2M3FN, $crate::arrays::encoding::f6e2m3fn),
            (F6E3M2FN, $crate::arrays::encoding::f6e3m2fn),
            (F8E3M4, $crate::arrays::encoding::f8e3m4),
            (F8E4M3, $crate::arrays::encoding::f8e4m3),
            (F8E4M3FN, $crate::arrays::encoding::f8e4m3fn),
            (F8E4M3FNUZ, $crate::arrays::encoding::f8e4m3fnuz),
            (F8E4M3B11FNUZ, $crate::arrays::encoding::f8e4m3b11fnuz),
            (F8E5M2, $crate::arrays::encoding::f8e5m2),
            (F8E5M2FNUZ, $crate::arrays::encoding::f8e5m2fnuz),
            (F8E8M0FNU, $crate::arrays::encoding::f8e8m0fnu),
            (BF16, $crate::arrays::encoding::bf16),
            (F16, $crate::arrays::encoding::f16),
            (F32, f32),
            (F64, f64),
        ) $data_type, |$element| $body)
    };

    (@real $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (I1, $crate::arrays::encoding::i1),
            (I2, $crate::arrays::encoding::i2),
            (I4, $crate::arrays::encoding::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::encoding::u1),
            (U2, $crate::arrays::encoding::u2),
            (U4, $crate::arrays::encoding::u4),
            (U8, u8),
            (U16, u16),
            (U32, u32),
            (U64, u64),
            (F4E2M1FN, $crate::arrays::encoding::f4e2m1fn),
            (F6E2M3FN, $crate::arrays::encoding::f6e2m3fn),
            (F6E3M2FN, $crate::arrays::encoding::f6e3m2fn),
            (F8E3M4, $crate::arrays::encoding::f8e3m4),
            (F8E4M3, $crate::arrays::encoding::f8e4m3),
            (F8E4M3FN, $crate::arrays::encoding::f8e4m3fn),
            (F8E4M3FNUZ, $crate::arrays::encoding::f8e4m3fnuz),
            (F8E4M3B11FNUZ, $crate::arrays::encoding::f8e4m3b11fnuz),
            (F8E5M2, $crate::arrays::encoding::f8e5m2),
            (F8E5M2FNUZ, $crate::arrays::encoding::f8e5m2fnuz),
            (F8E8M0FNU, $crate::arrays::encoding::f8e8m0fnu),
            (BF16, $crate::arrays::encoding::bf16),
            (F16, $crate::arrays::encoding::f16),
            (F32, f32),
            (F64, f64),
        ) $data_type, |$element| $body)
    };

    (@integer $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (I1, $crate::arrays::encoding::i1),
            (I2, $crate::arrays::encoding::i2),
            (I4, $crate::arrays::encoding::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::encoding::u1),
            (U2, $crate::arrays::encoding::u2),
            (U4, $crate::arrays::encoding::u4),
            (U8, u8),
            (U16, u16),
            (U32, u32),
            (U64, u64),
        ) $data_type, |$element| $body)
    };

    (@signed $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (I1, $crate::arrays::encoding::i1),
            (I2, $crate::arrays::encoding::i2),
            (I4, $crate::arrays::encoding::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
        ) $data_type, |$element| $body)
    };

    (@float $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (F4E2M1FN, $crate::arrays::encoding::f4e2m1fn),
            (F6E2M3FN, $crate::arrays::encoding::f6e2m3fn),
            (F6E3M2FN, $crate::arrays::encoding::f6e3m2fn),
            (F8E3M4, $crate::arrays::encoding::f8e3m4),
            (F8E4M3, $crate::arrays::encoding::f8e4m3),
            (F8E4M3FN, $crate::arrays::encoding::f8e4m3fn),
            (F8E4M3FNUZ, $crate::arrays::encoding::f8e4m3fnuz),
            (F8E4M3B11FNUZ, $crate::arrays::encoding::f8e4m3b11fnuz),
            (F8E5M2, $crate::arrays::encoding::f8e5m2),
            (F8E5M2FNUZ, $crate::arrays::encoding::f8e5m2fnuz),
            (F8E8M0FNU, $crate::arrays::encoding::f8e8m0fnu),
            (BF16, $crate::arrays::encoding::bf16),
            (F16, $crate::arrays::encoding::f16),
            (F32, f32),
            (F64, f64),
        ) $data_type, |$element| $body)
    };

    (@complex $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (C64, $crate::arrays::encoding::Complex<f32>),
            (C128, $crate::arrays::encoding::Complex<f64>),
        ) $data_type, |$element| $body)
    };

    (@boolean_or_integer $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (Boolean, bool),
            (I1, $crate::arrays::encoding::i1),
            (I2, $crate::arrays::encoding::i2),
            (I4, $crate::arrays::encoding::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::encoding::u1),
            (U2, $crate::arrays::encoding::u2),
            (U4, $crate::arrays::encoding::u4),
            (U8, u8),
            (U16, u16),
            (U32, u32),
            (U64, u64),
        ) $data_type, |$element| $body)
    };

    (@arms($(($variant:ident, $element_type:ty)),+ $(,)?) $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        match $data_type {
            $(
                $crate::arrays::DataType::$variant => {
                    type $element = $element_type;
                    $body
                }
            )+
            other => unreachable!("unsupported element data type {other} for this dispatch"),
        }
    };
}

/// Implements a binary elementwise capability for [`Array`](crate::Array).
///
/// Generates `impl Capability for Array` with a function of the form `fn function(&self, rhs: &Self) -> Result<Self,
/// ProgramError>`. The generated function checks the operation's input contract and determines the common element type
/// and broadcast shape. Inputs already using the common element type remain borrowed. Other inputs are converted before
/// the scalar calculation runs. The calculation receives decoded values of the concrete Rust element type selected by
/// [`dispatch_on_array_element_type!`]. Its result has that same element type, inferred from the generated traversal;
/// no caller-visible type binding is needed.
///
/// Empty results undergo all input and metadata checks, but skip operand conversion and scalar evaluation.
/// The result buffer follows the inferred layout, including zero-initialized padding. Non-empty results use
/// [`Array::binary_elements`](crate::Array::binary_elements) to traverse broadcast operands directly through
/// their storage layouts. The scalar calculation may return an error, which is propagated without modification.
///
/// Unsupported input types and metadata, incompatible broadcast shapes, failed conversions, and scalar errors
/// are returned as [`ProgramError`](crate::ProgramError). Validation precedes broadcasting, even for empty results.
/// Operations with a different output element type, such as comparisons, or special promotion rules should use their
/// own preparation path instead.
///
/// # Examples
///
/// A local capability can implement complex-aware minimum selection using the element contract. The scalar right
/// operand broadcasts across the left vector. Downstream crates must use a locally defined capability trait because
/// Rust's orphan rules prevent implementing another crate's trait for [`Array`](crate::Array).
///
/// ```rust
/// # use ryft_core::{Array, ArrayElement, ProgramError};
/// # use ryft_core::arrays::macros::impl_array_binary_elementwise_operation;
/// trait Minimum: Sized {
///     fn minimum(&self, rhs: &Self) -> Result<Self, ProgramError>;
/// }
///
/// impl_array_binary_elementwise_operation!(
///     Minimum, minimum,
///     operation = "minimum",
///     inputs = @numeric,
///     checks = [@no_unreduced, @same_reduced_axes],
///     |lhs, rhs| Ok(ArrayElement::min(&lhs, &rhs)),
/// );
///
/// let result = Array::vector(vec![1i32, 5, 3]).minimum(&Array::scalar(2i32))?;
/// assert_eq!(result.elements::<i32>()?, vec![1, 2, 2]);
/// # Ok::<(), ProgramError>(())
/// ```
///
/// A real-only capability can reuse the same execution path. Here mixed `i16` and `i32` inputs promote to `i32`:
///
/// ```rust
/// # use ryft_core::{Array, ArrayElement, ProgramError};
/// # use ryft_core::arrays::macros::impl_array_binary_elementwise_operation;
/// trait Maximum: Sized {
///     fn maximum(&self, rhs: &Self) -> Result<Self, ProgramError>;
/// }
///
/// impl_array_binary_elementwise_operation!(
///     Maximum, maximum,
///     operation = "maximum",
///     inputs = @real,
///     checks = [@no_unreduced, @same_reduced_axes],
///     |lhs, rhs| Ok(ArrayElement::max(&lhs, &rhs)),
/// );
///
/// let result = Array::vector(vec![1i16, 5]).maximum(&Array::scalar(3i32))?;
/// assert_eq!(result.elements::<i32>()?, vec![3, 5]);
/// # Ok::<(), ProgramError>(())
/// ```
///
/// # Parameters
///
///   - `$capability`: Capability trait path.
///   - `$method`: Name of its binary function to implement.
///   - `operation = $operation`: Name used in diagnostics, such as `"min"`.
///   - `inputs = @$class`: `@numeric` accepts integers, real floating-point values, and complex values.
///     `@real` accepts integers and real floating-point values. Both reject Booleans, including for empty arrays.
///   - `checks = $checks`: Ordered list of array metadata checks from [`check_types!`](crate::check_types), typically
///     `@no_unreduced`, `@same_unreduced_axes`, or `@same_reduced_axes`. An empty list applies no additional checks.
///   - `|lhs, rhs| body`: Names for decoded scalar operands, followed by an expression returning their element type
///     wrapped in `Result<_, ProgramError>`. The body must compile for every type in `inputs`.
#[macro_export]
macro_rules! impl_array_binary_elementwise_operation {
    // Validates the numeric dispatch class, excluding Booleans.
    (@validate @numeric, $operation:expr, $types:expr $(,)?) => {
        $crate::macros::check_types!(@numeric, $operation, $types)
    };

    // Refines numeric inputs to real values; `check_types!` treats `@real` alone only as a complex exclusion.
    (@validate @real, $operation:expr, $types:expr $(,)?) => {
        $crate::macros::check_types!(@numeric @real, $operation, $types)
    };

    // Generates a capability implementation with validated promotion, broadcasting, and a fallible scalar kernel.
    (
        $capability:path, $method:ident,
        operation = $operation:expr,
        inputs = @$class:ident,
        checks = [$(@$check:ident),* $(,)?],
        |$lhs_element:ident, $rhs_element:ident| $body:expr $(,)?
    ) => {
        impl $capability for $crate::arrays::Array {
            fn $method(&self, rhs: &Self) -> Result<Self, $crate::programs::ProgramError> {
                use $crate::arrays::broadcasting::Broadcastable as _;
                use $crate::programs::Typed as _;

                let lhs = self;
                let operation = $operation;
                let lhs_type = lhs.r#type();
                let rhs_type = rhs.r#type();

                // Validate the original inputs before promotion or the empty result shortcut, so neither can
                // hide an unsupported input type or invalid reduction metadata.
                $crate::arrays::macros::impl_array_binary_elementwise_operation!(
                    @validate @$class, operation,
                    [lhs_type.data_type(), rhs_type.data_type()],
                );
                $($crate::macros::check_types!(@$check, operation, [lhs_type.as_ref(), rhs_type.as_ref()]);)*

                // Infer the common element type and broadcast result metadata without materializing broadcasts.
                let output_type = lhs_type.as_ref().broadcast(rhs_type.as_ref())
                    .map_err(|error| $crate::programs::TypeError::invalid(error.to_string()))?;

                // Empty results need only layout-sized, zero-initialized storage. Skip operand conversions
                // and scalar evaluation, since neither contributes any result elements.
                if Self::element_count(&output_type) == 0 {
                    let addressing = $crate::arrays::addressing::ArrayAddressing::new(output_type.clone())?;
                    return Self::new(output_type, vec![0; addressing.storage_byte_len()]);
                }

                // Borrow inputs already using the common element type and convert only those that differ.
                // Their shapes remain unchanged. The traversal below supplies broadcast indexing.
                let data_type = output_type.data_type();
                let lhs = lhs.promoted_to(data_type)?;
                let rhs = rhs.promoted_to(data_type)?;

                // Instantiate the scalar calculation for the selected Rust element type. The shared traversal
                // decodes addressed inputs and encodes each result directly into the output buffer.
                $crate::arrays::macros::dispatch_on_array_element_type!(@$class data_type, |Element| {
                    lhs.binary_elements::<Element, Element>(&rhs, output_type, |$lhs_element, $rhs_element| $body)
                })
            }
        }
    };
}

pub use crate::{dispatch_on_array_element_type, impl_array_binary_elementwise_operation};

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::encoding::{ArrayElement, Complex};
    use crate::arrays::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::programs::{ProgramError, TypeError, Typed};

    /// Numeric minimum used to test generated array implementations.
    trait TestMinimum: Sized {
        /// Computes the elementwise minimum of the operands.
        fn minimum(&self, rhs: &Self) -> Result<Self, ProgramError>;
    }

    impl_array_binary_elementwise_operation!(
        TestMinimum, minimum,
        operation = "min",
        inputs = @numeric,
        checks = [@no_unreduced, @same_reduced_axes],
        |lhs, rhs| Ok(ArrayElement::min(&lhs, &rhs)),
    );

    /// Returns the [`DataType`] represented by one array-element type.
    fn element_data_type<T: ArrayElement>() -> DataType {
        T::DATA_TYPE
    }

    #[test]
    fn test_dispatch_on_array_element_type() {
        // Every selected data type must dispatch to the element type that represents exactly that data type.
        let all = [
            DataType::Boolean,
            DataType::I1,
            DataType::I2,
            DataType::I4,
            DataType::I8,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::U1,
            DataType::U2,
            DataType::U4,
            DataType::U8,
            DataType::U16,
            DataType::U32,
            DataType::U64,
            DataType::F4E2M1FN,
            DataType::F6E2M3FN,
            DataType::F6E3M2FN,
            DataType::F8E3M4,
            DataType::F8E4M3,
            DataType::F8E4M3FN,
            DataType::F8E4M3FNUZ,
            DataType::F8E4M3B11FNUZ,
            DataType::F8E5M2,
            DataType::F8E5M2FNUZ,
            DataType::F8E8M0FNU,
            DataType::BF16,
            DataType::F16,
            DataType::F32,
            DataType::F64,
            DataType::C64,
            DataType::C128,
        ];

        all.iter().copied().for_each(|data_type| {
            assert_eq!(dispatch_on_array_element_type!(data_type, |Element| element_data_type::<Element>()), data_type);
        });

        // Every class selector covers exactly the data types of its class and dispatches each to its element type.
        all.iter().copied().filter(|data_type| *data_type != DataType::Boolean).for_each(|data_type| {
            assert_eq!(
                dispatch_on_array_element_type!(@numeric data_type, |Element| element_data_type::<Element>()),
                data_type,
            );
        });

        all.iter().copied().filter(|data_type| !data_type.is_complex()).for_each(|data_type| {
            assert_eq!(
                dispatch_on_array_element_type!(@ordered data_type, |Element| element_data_type::<Element>()),
                data_type,
            );
        });

        all.iter()
            .copied()
            .filter(|data_type| data_type.is_integer() || data_type.is_floating_point())
            .for_each(|data_type| {
                assert_eq!(
                    dispatch_on_array_element_type!(@real data_type, |Element| element_data_type::<Element>()),
                    data_type,
                );
            });

        all.iter().copied().filter(|data_type| data_type.is_integer()).for_each(|data_type| {
            assert_eq!(
                dispatch_on_array_element_type!(@integer data_type, |Element| element_data_type::<Element>()),
                data_type,
            );
        });

        all.iter().copied().filter(|data_type| data_type.is_signed()).for_each(|data_type| {
            assert_eq!(
                dispatch_on_array_element_type!(@signed data_type, |Element| element_data_type::<Element>()),
                data_type,
            );
        });

        all.iter().copied().filter(|data_type| data_type.is_floating_point()).for_each(|data_type| {
            assert_eq!(
                dispatch_on_array_element_type!(@float data_type, |Element| element_data_type::<Element>()),
                data_type,
            );
        });

        all.iter().copied().filter(|data_type| data_type.is_complex()).for_each(|data_type| {
            assert_eq!(
                dispatch_on_array_element_type!(@complex data_type, |Element| element_data_type::<Element>()),
                data_type,
            );
        });

        all.iter()
            .copied()
            .filter(|data_type| *data_type == DataType::Boolean || data_type.is_integer())
            .for_each(|data_type| {
                assert_eq!(
                    dispatch_on_array_element_type!(
                        @boolean_or_integer
                        data_type,
                        |Element| element_data_type::<Element>(),
                    ),
                    data_type,
                );
            });

        // The body is instantiated per element type, so type-position uses such as `size_of` resolve per arm.
        assert_eq!(dispatch_on_array_element_type!(DataType::F8E4M3FN, |Element| size_of::<Element>()), 1);
        assert_eq!(dispatch_on_array_element_type!(DataType::C128, |Element| size_of::<Element>()), 16);
    }

    #[test]
    #[should_panic(expected = "unsupported element data type c64 for this dispatch")]
    fn test_dispatch_on_array_element_type_rejects_out_of_class_data_types() {
        dispatch_on_array_element_type!(@real DataType::C64, |Element| element_data_type::<Element>());
    }

    #[test]
    #[should_panic(expected = "unsupported element data type token for this dispatch")]
    fn test_dispatch_on_array_element_type_rejects_payload_free_data_types() {
        dispatch_on_array_element_type!(DataType::Token, |Element| element_data_type::<Element>());
    }

    #[test]
    fn test_impl_array_binary_elementwise_operation() -> Result<(), ProgramError> {
        // Promotion and broadcasting happen before the scalar kernel receives its operands.
        let lhs = Array::from_elements(ArrayType::new_static(DataType::I16, [2, 1]), &[2i16, 5])?;
        let rhs = Array::from_elements(ArrayType::new_static(DataType::I32, [1, 3]), &[1i32, 3, 6])?;
        let output = lhs.minimum(&rhs)?;
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::I32, [2, 3]));
        assert_eq!(output.elements::<i32>()?, vec![1, 2, 2, 1, 3, 5]);
        Ok(())
    }

    #[test]
    fn test_impl_array_binary_elementwise_operation_empty_invalid_data_type() -> Result<(), ProgramError> {
        /// Real minimum used to verify input-class validation.
        trait TestRealMinimum: Sized {
            /// Computes the elementwise minimum of real operands.
            fn real_minimum(&self, rhs: &Self) -> Result<Self, ProgramError>;
        }

        impl_array_binary_elementwise_operation!(
            TestRealMinimum, real_minimum,
            operation = "min",
            inputs = @real,
            checks = [@no_unreduced, @same_reduced_axes],
            |lhs, rhs| Ok(ArrayElement::min(&lhs, &rhs)),
        );

        // Empty operands must satisfy the same input contract as nonempty operands.
        let input = Array::from_elements(ArrayType::new_static(DataType::Boolean, [0]), &[] as &[bool])?;
        let output = input.minimum(&input);
        assert!(matches!(
            output,
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`min` does not support input data type bool",
        ));

        let output = input.real_minimum(&input);
        assert!(matches!(
            output,
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`min` does not support input data type bool",
        ));

        let input = Array::from_elements(ArrayType::new_static(DataType::C64, [0]), &[] as &[Complex<f32>])?;
        let output = input.real_minimum(&input);
        assert!(matches!(
            output,
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`min` does not support input data type c64",
        ));
        Ok(())
    }

    #[test]
    fn test_impl_array_binary_elementwise_operation_empty_invalid_metadata() -> Result<(), ProgramError> {
        // Unreduced values are rejected even when their buffers have no elements.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::Replicated])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [0]).with_sharding(sharding).unwrap(),
            &[] as &[f32],
        )?;
        let output = input.minimum(&input);
        assert!(matches!(
            output,
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`min` does not support unreduced operands",
        ));
        Ok(())
    }

    #[test]
    fn test_impl_array_binary_elementwise_operation_empty_output() -> Result<(), ProgramError> {
        /// Operation whose scalar kernel always fails.
        trait TestFailingOperation: Sized {
            /// Reports a scalar error whenever an element is evaluated.
            fn fail(&self, rhs: &Self) -> Result<Self, ProgramError>;
        }

        impl_array_binary_elementwise_operation!(
            TestFailingOperation, fail,
            operation = "min",
            inputs = @real,
            checks = [@no_unreduced, @same_reduced_axes],
            |_lhs, _rhs| Err(ProgramError::InvalidArgument {
                message: "scalar kernel must not run".to_string(),
            }),
        );

        // Broadcasting to an empty shape does not evaluate the scalar kernel.
        let lhs = Array::from_elements(ArrayType::new_static(DataType::F32, [0, 1]), &[] as &[f32])?;
        let rhs = Array::from_elements(ArrayType::new_static(DataType::F32, [1, 2]), &[1f32, 2.0])?;
        let output = lhs.fail(&rhs)?;
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::F32, [0, 2]));
        assert_eq!(output.elements::<f32>()?, Vec::<f32>::new());
        Ok(())
    }

    #[test]
    fn test_impl_array_binary_elementwise_operation_scalar_error() {
        /// Operation whose scalar kernel always fails.
        trait TestFailingOperation: Sized {
            /// Reports a scalar error whenever an element is evaluated.
            fn fail(&self, rhs: &Self) -> Result<Self, ProgramError>;
        }

        impl_array_binary_elementwise_operation!(
            TestFailingOperation, fail,
            operation = "min",
            inputs = @real,
            checks = [@no_unreduced, @same_reduced_axes],
            |_lhs, _rhs| Err(ProgramError::InvalidArgument {
                message: "scalar kernel failed".to_string(),
            }),
        );

        let output = Array::scalar(1i32).fail(&Array::scalar(2i32));
        assert!(matches!(
            output,
            Err(ProgramError::InvalidArgument { message }) if message == "scalar kernel failed",
        ));
    }
}
