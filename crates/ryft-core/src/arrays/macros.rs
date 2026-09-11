/// Dispatches one runtime [`DataType`](crate::arrays::DataType) to its sealed Rust array element type, so that array
/// kernels can run generic element code without storing or allocating another dynamic element representation.
///
/// The macro takes a class selector, a [`DataType`](crate::arrays::DataType) expression, and an `|Element| body`
/// closure-like form, and expands to a `match` that binds the type alias `Element` to the matching element type from
/// [`arrays::elements`](crate::arrays::elements) in every selected arm. The body is instantiated once per selected
/// element type and may therefore use `Element` in any type position, including calls to generic functions bounded
/// by [`ArrayElement`](crate::arrays::elements::ArrayElement) or by kernel capability traits:
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
            (I1, $crate::arrays::elements::i1),
            (I2, $crate::arrays::elements::i2),
            (I4, $crate::arrays::elements::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::elements::u1),
            (U2, $crate::arrays::elements::u2),
            (U4, $crate::arrays::elements::u4),
            (U8, u8),
            (U16, u16),
            (U32, u32),
            (U64, u64),
            (F4E2M1FN, $crate::arrays::elements::f4e2m1fn),
            (F6E2M3FN, $crate::arrays::elements::f6e2m3fn),
            (F6E3M2FN, $crate::arrays::elements::f6e3m2fn),
            (F8E3M4, $crate::arrays::elements::f8e3m4),
            (F8E4M3, $crate::arrays::elements::f8e4m3),
            (F8E4M3FN, $crate::arrays::elements::f8e4m3fn),
            (F8E4M3FNUZ, $crate::arrays::elements::f8e4m3fnuz),
            (F8E4M3B11FNUZ, $crate::arrays::elements::f8e4m3b11fnuz),
            (F8E5M2, $crate::arrays::elements::f8e5m2),
            (F8E5M2FNUZ, $crate::arrays::elements::f8e5m2fnuz),
            (F8E8M0FNU, $crate::arrays::elements::f8e8m0fnu),
            (BF16, $crate::arrays::elements::bf16),
            (F16, $crate::arrays::elements::f16),
            (F32, f32),
            (F64, f64),
            (C64, $crate::arrays::elements::Complex<f32>),
            (C128, $crate::arrays::elements::Complex<f64>),
        ) $data_type, |$element| $body)
    };

    (@numeric $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (I1, $crate::arrays::elements::i1),
            (I2, $crate::arrays::elements::i2),
            (I4, $crate::arrays::elements::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::elements::u1),
            (U2, $crate::arrays::elements::u2),
            (U4, $crate::arrays::elements::u4),
            (U8, u8),
            (U16, u16),
            (U32, u32),
            (U64, u64),
            (F4E2M1FN, $crate::arrays::elements::f4e2m1fn),
            (F6E2M3FN, $crate::arrays::elements::f6e2m3fn),
            (F6E3M2FN, $crate::arrays::elements::f6e3m2fn),
            (F8E3M4, $crate::arrays::elements::f8e3m4),
            (F8E4M3, $crate::arrays::elements::f8e4m3),
            (F8E4M3FN, $crate::arrays::elements::f8e4m3fn),
            (F8E4M3FNUZ, $crate::arrays::elements::f8e4m3fnuz),
            (F8E4M3B11FNUZ, $crate::arrays::elements::f8e4m3b11fnuz),
            (F8E5M2, $crate::arrays::elements::f8e5m2),
            (F8E5M2FNUZ, $crate::arrays::elements::f8e5m2fnuz),
            (F8E8M0FNU, $crate::arrays::elements::f8e8m0fnu),
            (BF16, $crate::arrays::elements::bf16),
            (F16, $crate::arrays::elements::f16),
            (F32, f32),
            (F64, f64),
            (C64, $crate::arrays::elements::Complex<f32>),
            (C128, $crate::arrays::elements::Complex<f64>),
        ) $data_type, |$element| $body)
    };

    (@ordered $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (Boolean, bool),
            (I1, $crate::arrays::elements::i1),
            (I2, $crate::arrays::elements::i2),
            (I4, $crate::arrays::elements::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::elements::u1),
            (U2, $crate::arrays::elements::u2),
            (U4, $crate::arrays::elements::u4),
            (U8, u8),
            (U16, u16),
            (U32, u32),
            (U64, u64),
            (F4E2M1FN, $crate::arrays::elements::f4e2m1fn),
            (F6E2M3FN, $crate::arrays::elements::f6e2m3fn),
            (F6E3M2FN, $crate::arrays::elements::f6e3m2fn),
            (F8E3M4, $crate::arrays::elements::f8e3m4),
            (F8E4M3, $crate::arrays::elements::f8e4m3),
            (F8E4M3FN, $crate::arrays::elements::f8e4m3fn),
            (F8E4M3FNUZ, $crate::arrays::elements::f8e4m3fnuz),
            (F8E4M3B11FNUZ, $crate::arrays::elements::f8e4m3b11fnuz),
            (F8E5M2, $crate::arrays::elements::f8e5m2),
            (F8E5M2FNUZ, $crate::arrays::elements::f8e5m2fnuz),
            (F8E8M0FNU, $crate::arrays::elements::f8e8m0fnu),
            (BF16, $crate::arrays::elements::bf16),
            (F16, $crate::arrays::elements::f16),
            (F32, f32),
            (F64, f64),
        ) $data_type, |$element| $body)
    };

    (@real $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (I1, $crate::arrays::elements::i1),
            (I2, $crate::arrays::elements::i2),
            (I4, $crate::arrays::elements::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::elements::u1),
            (U2, $crate::arrays::elements::u2),
            (U4, $crate::arrays::elements::u4),
            (U8, u8),
            (U16, u16),
            (U32, u32),
            (U64, u64),
            (F4E2M1FN, $crate::arrays::elements::f4e2m1fn),
            (F6E2M3FN, $crate::arrays::elements::f6e2m3fn),
            (F6E3M2FN, $crate::arrays::elements::f6e3m2fn),
            (F8E3M4, $crate::arrays::elements::f8e3m4),
            (F8E4M3, $crate::arrays::elements::f8e4m3),
            (F8E4M3FN, $crate::arrays::elements::f8e4m3fn),
            (F8E4M3FNUZ, $crate::arrays::elements::f8e4m3fnuz),
            (F8E4M3B11FNUZ, $crate::arrays::elements::f8e4m3b11fnuz),
            (F8E5M2, $crate::arrays::elements::f8e5m2),
            (F8E5M2FNUZ, $crate::arrays::elements::f8e5m2fnuz),
            (F8E8M0FNU, $crate::arrays::elements::f8e8m0fnu),
            (BF16, $crate::arrays::elements::bf16),
            (F16, $crate::arrays::elements::f16),
            (F32, f32),
            (F64, f64),
        ) $data_type, |$element| $body)
    };

    (@integer $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (I1, $crate::arrays::elements::i1),
            (I2, $crate::arrays::elements::i2),
            (I4, $crate::arrays::elements::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::elements::u1),
            (U2, $crate::arrays::elements::u2),
            (U4, $crate::arrays::elements::u4),
            (U8, u8),
            (U16, u16),
            (U32, u32),
            (U64, u64),
        ) $data_type, |$element| $body)
    };

    (@signed $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (I1, $crate::arrays::elements::i1),
            (I2, $crate::arrays::elements::i2),
            (I4, $crate::arrays::elements::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
        ) $data_type, |$element| $body)
    };

    (@float $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (F4E2M1FN, $crate::arrays::elements::f4e2m1fn),
            (F6E2M3FN, $crate::arrays::elements::f6e2m3fn),
            (F6E3M2FN, $crate::arrays::elements::f6e3m2fn),
            (F8E3M4, $crate::arrays::elements::f8e3m4),
            (F8E4M3, $crate::arrays::elements::f8e4m3),
            (F8E4M3FN, $crate::arrays::elements::f8e4m3fn),
            (F8E4M3FNUZ, $crate::arrays::elements::f8e4m3fnuz),
            (F8E4M3B11FNUZ, $crate::arrays::elements::f8e4m3b11fnuz),
            (F8E5M2, $crate::arrays::elements::f8e5m2),
            (F8E5M2FNUZ, $crate::arrays::elements::f8e5m2fnuz),
            (F8E8M0FNU, $crate::arrays::elements::f8e8m0fnu),
            (BF16, $crate::arrays::elements::bf16),
            (F16, $crate::arrays::elements::f16),
            (F32, f32),
            (F64, f64),
        ) $data_type, |$element| $body)
    };

    (@complex $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (C64, $crate::arrays::elements::Complex<f32>),
            (C128, $crate::arrays::elements::Complex<f64>),
        ) $data_type, |$element| $body)
    };

    (@boolean_or_integer $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@arms(
            (Boolean, bool),
            (I1, $crate::arrays::elements::i1),
            (I2, $crate::arrays::elements::i2),
            (I4, $crate::arrays::elements::i4),
            (I8, i8),
            (I16, i16),
            (I32, i32),
            (I64, i64),
            (U1, $crate::arrays::elements::u1),
            (U2, $crate::arrays::elements::u2),
            (U4, $crate::arrays::elements::u4),
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
            other => unreachable!("unsupported element data type `{other}` for this dispatch"),
        }
    };
}

/// Implements a unary or binary elementwise capability for [`Array`](crate::Array).
///
/// Generates `impl Capability for Array`. `@unary` implements `fn function(&self) -> Result<Self, ProgramError>` and
/// `@binary` implements `fn function(&self, rhs: &Self) -> Result<Self, ProgramError>`. Both forms validate the input
/// contract before evaluating elements. Unary operations preserve the input type and layout while binary operations
/// determine the common element type and broadcast shape. Inputs already using the common element type remain borrowed.
/// Other inputs are converted before the scalar calculation runs. The calculation receives decoded values of the
/// concrete Rust element type selected by [`dispatch_on_array_element_type!`]. Its result has that same element type,
/// inferred from the generated traversal; no caller-visible type binding is needed.
///
/// Empty results undergo all input and metadata checks, but skip operand conversion and scalar
/// evaluation. The result buffer follows the inferred layout, including zero-initialized padding.
/// Non-empty results use [`Array::map_elements`](crate::Array::map_elements) or
/// [`Array::map_element_pairs`](crate::Array::map_element_pairs) to traverse operands directly
/// through their storage layouts. Scalar errors are propagated without modification.
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
/// # use ryft_core::arrays::macros::impl_array_elementwise_operation;
/// trait Minimum: Sized {
///     fn minimum(&self, rhs: &Self) -> Result<Self, ProgramError>;
/// }
///
/// impl_array_elementwise_operation!(
///     @binary
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
/// Unary operations keep the input type and layout and receive one decoded scalar. This local capability
/// uses the [`ArrayElement`](crate::arrays::ArrayElement) contract for a calculation over real elements:
///
/// ```
/// # use ryft_core::{Array, ArrayElement, ProgramError};
/// # use ryft_core::arrays::macros::impl_array_elementwise_operation;
/// trait Nonnegative: Sized {
///     fn nonnegative(&self) -> Result<Self, ProgramError>;
/// }
///
/// impl_array_elementwise_operation!(
///     @unary
///     Nonnegative, nonnegative,
///     operation = "nonnegative",
///     inputs = @numeric @real,
///     checks = [@no_unreduced],
///     |input| Ok(ArrayElement::max(&input, &ArrayElement::from_signed(0)?)),
/// );
///
/// let result = Array::vector(vec![-2i32, 1, 5]).nonnegative()?;
/// assert_eq!(result.elements::<i32>()?, vec![0, 1, 5]);
/// # Ok::<(), ProgramError>(())
/// ```
///
/// # Parameters
///
///   - `@unary` or `@binary`: Number of operands accepted by the generated function.
///   - `$capability`: Capability trait path.
///   - `$method`: Name of its function to implement.
///   - `operation = $operation`: Name used in diagnostics, such as `"min"`.
///   - `inputs = $(@selector)+`: Composable predicates over numeric elements, using the same intersection rules as
///     [`check_types!`](crate::check_types). `@numeric` accepts all numeric types, `@float` excludes integers, and
///     `@real` excludes complex values. For example, `@float @real` accepts only real floating-point values and
///     `@numeric @real` also accepts integers. Order and repetition do not change the accepted types. This macro's
///     numeric base universe always excludes Booleans and payload-free types, including for empty arrays.
///   - `checks = $checks`: Ordered list of array metadata checks from [`check_types!`](crate::check_types), typically
///     `@no_unreduced`, `@same_unreduced_axes`, or `@same_reduced_axes`. An empty list applies no additional checks.
///   - `|input| body` or `|lhs, rhs| body`: Names for decoded scalar operands, followed by an expression returning
///     their element type wrapped in `Result<_, ProgramError>`. The body must compile for every type in `inputs`.
#[macro_export]
macro_rules! impl_array_elementwise_operation {
    // Intersect selectors for dispatch. Numeric is the base universe; float and real each narrow it independently.
    (@dispatch [$(@$selector:ident)+] $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::impl_array_elementwise_operation!(
            @select [numeric complex] [$(@$selector)+] $data_type, |$element| $body,
        )
    };

    // Numeric does not widen a class already restricted by another selector.
    (@select [$base:ident $kind:ident] [@numeric $($rest:tt)*] $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::impl_array_elementwise_operation!(
            @select [$base $kind] [$($rest)*] $data_type, |$element| $body,
        )
    };

    // Float removes integers while preserving any existing real-only restriction.
    (@select [$base:ident $kind:ident] [@float $($rest:tt)*] $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::impl_array_elementwise_operation!(
            @select [float $kind] [$($rest)*] $data_type, |$element| $body,
        )
    };

    // Real removes complex values without changing whether integers are accepted.
    (@select [$base:ident $kind:ident] [@real $($rest:tt)*] $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::impl_array_elementwise_operation!(
            @select [$base real] [$($rest)*] $data_type, |$element| $body,
        )
    };

    // The numeric class includes integers, real floating-point values, and complex values.
    (@select [numeric complex] [] $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@numeric $data_type, |$element| $body)
    };

    // The real numeric class excludes complex values.
    (@select [numeric real] [] $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@real $data_type, |$element| $body)
    };

    // Float capabilities include complex values, while the storage dispatcher separates those classes.
    (@select [float complex] [] $data_type:expr, |$element:ident| $body:expr $(,)?) => {{
        let data_type = $data_type;
        if data_type.is_complex() {
            $crate::arrays::macros::dispatch_on_array_element_type!(@complex data_type, |$element| $body)
        } else {
            $crate::arrays::macros::dispatch_on_array_element_type!(@float data_type, |$element| $body)
        }
    }};

    // Intersecting float with real maps to the storage dispatcher's real floating-point class.
    (@select [float real] [] $data_type:expr, |$element:ident| $body:expr $(,)?) => {
        $crate::arrays::macros::dispatch_on_array_element_type!(@float $data_type, |$element| $body)
    };

    // Generates a unary capability preserving the input element type, shape, and layout.
    (
        @unary
        $capability:path, $method:ident,
        operation = $operation:expr,
        inputs = $(@$selector:ident)+,
        checks = [$(@$check:ident),* $(,)?],
        |$input_element:ident| $body:expr $(,)?
    ) => {
        impl $capability for $crate::arrays::Array {
            fn $method(&self) -> Result<Self, $crate::programs::ProgramError> {
                use $crate::programs::Typed as _;
                let operation = $operation;
                let input_type = self.r#type();
                let data_type = input_type.data_type();

                // Validate before traversal so empty arrays satisfy the same contract as non-empty arrays.
                $crate::macros::check_types!(@numeric $(@$selector)+, operation, [data_type]);
                $($crate::macros::check_types!(@$check, operation, [input_type.as_ref()]);)*

                // The mapped output retains the input layout. Empty traversal allocates storage without evaluating
                // the scalar calculation. Non-empty traversal decodes and re-encodes each element in its own type.
                $crate::arrays::macros::impl_array_elementwise_operation!(
                    @dispatch [$(@$selector)+] data_type, |Element| {
                        self.map_elements::<Element, Element>(input_type.into_owned(), |$input_element| $body)
                    },
                )
            }
        }
    };

    // Generates a capability implementation with validated promotion, broadcasting, and a fallible scalar kernel.
    (
        @binary
        $capability:path, $method:ident,
        operation = $operation:expr,
        inputs = $(@$selector:ident)+,
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
                $crate::macros::check_types!(
                    @numeric $(@$selector)+, operation, [lhs_type.data_type(), rhs_type.data_type()],
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
                $crate::arrays::macros::impl_array_elementwise_operation!(
                    @dispatch [$(@$selector)+] data_type, |Element| {
                        lhs.map_element_pairs::<Element, Element>(&rhs, output_type, |$lhs_element, $rhs_element| $body)
                    },
                )
            }
        }
    };
}

pub use crate::{dispatch_on_array_element_type, impl_array_elementwise_operation};

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::elements::{ArrayElement, Complex};
    use crate::arrays::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::layouts::{Layout, StridedLayout};
    use crate::programs::{ProgramError, TypeError, Typed};

    /// Numeric identity used to test generated unary implementations.
    trait TestIdentity: Sized {
        /// Returns an array with unchanged element values and metadata.
        fn identity(&self) -> Result<Self, ProgramError>;
    }

    impl_array_elementwise_operation!(
        @unary
        TestIdentity, identity,
        operation = "identity",
        inputs = @numeric,
        checks = [@no_unreduced],
        |input| Ok(input),
    );

    /// Numeric minimum used to test generated array implementations.
    trait TestMinimum: Sized {
        /// Computes the elementwise minimum of the operands.
        fn minimum(&self, rhs: &Self) -> Result<Self, ProgramError>;
    }

    impl_array_elementwise_operation!(
        @binary
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
    #[should_panic(expected = "unsupported element data type `c64` for this dispatch")]
    fn test_dispatch_on_array_element_type_rejects_out_of_class_data_types() {
        dispatch_on_array_element_type!(@real DataType::C64, |Element| element_data_type::<Element>());
    }

    #[test]
    #[should_panic(expected = "unsupported element data type `token` for this dispatch")]
    fn test_dispatch_on_array_element_type_rejects_payload_free_data_types() {
        dispatch_on_array_element_type!(DataType::Token, |Element| element_data_type::<Element>());
    }

    #[test]
    fn test_impl_array_elementwise_operation_unary() -> Result<(), ProgramError> {
        // Logical traversal preserves nondefault layout metadata and element ordering.
        let input_type =
            ArrayType::new_static(DataType::I32, [2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![4, 8])));
        let input = Array::from_elements(input_type.clone(), &[1i32, 2, 3, 4])?;
        let output = input.identity()?;
        assert_eq!(output.r#type().as_ref(), &input_type);
        assert_eq!(output.elements::<i32>()?, vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn test_impl_array_elementwise_operation_unary_float() -> Result<(), ProgramError> {
        /// Identity restricted to floating-point and complex inputs.
        trait TestFloatIdentity: Sized {
            /// Preserves each floating-point or complex element.
            fn float_identity(&self) -> Result<Self, ProgramError>;
        }

        impl_array_elementwise_operation!(
            @unary
            TestFloatIdentity, float_identity,
            operation = "float_identity",
            inputs = @float @numeric,
            checks = [],
            |input| Ok(input),
        );

        // The floating-point class includes complex values without discarding imaginary components.
        let input = Array::from_elements(
            ArrayType::new_static(DataType::C64, [2]),
            &[Complex::new(1f32, -2.0), Complex::new(3.0, 4.0)],
        )?;
        assert_eq!(input.float_identity()?, input);
        let input = Array::scalar(-0f32);
        assert_eq!(input.float_identity()?.elements::<f32>()?[0].to_bits(), (-0f32).to_bits());
        assert!(matches!(
            Array::scalar(1i32).float_identity(),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`float_identity` does not support input data type `i32`",
        ));
        Ok(())
    }

    #[test]
    fn test_impl_array_elementwise_operation_unary_real_float() -> Result<(), ProgramError> {
        /// Identity restricted to real floating-point inputs.
        trait TestRealFloatIdentity: Sized {
            /// Preserves each real floating-point element.
            fn real_float_identity(&self) -> Result<Self, ProgramError>;
        }

        // Selector intersections are independent of order and tolerate repeated refinements.
        impl_array_elementwise_operation!(
            @unary
            TestRealFloatIdentity, real_float_identity,
            operation = "real_float_identity",
            inputs = @real @numeric @float @real,
            checks = [],
            |input| Ok(input),
        );

        assert_eq!(Array::scalar(2f32).real_float_identity()?, Array::scalar(2f32));
        assert!(matches!(
            Array::scalar(Complex::new(1f32, 2.0)).real_float_identity(),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`real_float_identity` does not support input data type `c64`",
        ));

        // Empty operands cannot bypass validation of the declared input class.
        let input = Array::from_elements(ArrayType::new_static(DataType::Boolean, [0]), &[] as &[bool])?;
        assert!(matches!(
            input.real_float_identity(),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`real_float_identity` does not support input data type `bool`",
        ));
        assert!(matches!(
            input.identity(),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`identity` does not support input data type `bool`",
        ));
        Ok(())
    }

    #[test]
    fn test_impl_array_elementwise_operation_unary_empty_invalid_metadata() -> Result<(), ProgramError> {
        // Validate reduction metadata before skipping evaluation of an empty buffer.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::Replicated])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [0]).with_sharding(sharding).unwrap(),
            &[] as &[f32],
        )?;
        assert!(matches!(
            input.identity(),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`identity` does not support unreduced operands",
        ));
        Ok(())
    }

    #[test]
    fn test_impl_array_elementwise_operation_unary_scalar_error() -> Result<(), ProgramError> {
        /// Operation whose scalar kernel always fails.
        trait TestFailingUnaryOperation: Sized {
            /// Reports an error whenever an element is evaluated.
            fn fail(&self) -> Result<Self, ProgramError>;
        }

        impl_array_elementwise_operation!(
            @unary
            TestFailingUnaryOperation, fail,
            operation = "fail",
            inputs = @numeric,
            checks = [],
            |_input| Err(ProgramError::InvalidArgument {
                message: "scalar kernel failed".to_string(),
            }),
        );

        assert!(matches!(
            Array::scalar(1i32).fail(),
            Err(ProgramError::InvalidArgument { message }) if message == "scalar kernel failed",
        ));

        // Empty output construction succeeds without invoking the fallible scalar calculation.
        let input = Array::from_elements(ArrayType::new_static(DataType::I32, [0]), &[] as &[i32])?;
        let output = input.fail()?;
        assert_eq!(output.r#type(), input.r#type());
        assert_eq!(output.elements::<i32>()?, Vec::<i32>::new());
        Ok(())
    }

    #[test]
    fn test_impl_array_elementwise_operation_binary() -> Result<(), ProgramError> {
        // Promotion and broadcasting happen before the scalar kernel receives its operands.
        let lhs = Array::from_elements(ArrayType::new_static(DataType::I16, [2, 1]), &[2i16, 5])?;
        let rhs = Array::from_elements(ArrayType::new_static(DataType::I32, [1, 3]), &[1i32, 3, 6])?;
        let output = lhs.minimum(&rhs)?;
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::I32, [2, 3]));
        assert_eq!(output.elements::<i32>()?, vec![1, 2, 2, 1, 3, 5]);
        Ok(())
    }

    #[test]
    fn test_impl_array_elementwise_operation_binary_empty_invalid_data_type() -> Result<(), ProgramError> {
        /// Real minimum used to verify input-class validation.
        trait TestRealMinimum: Sized {
            /// Computes the elementwise minimum of real operands.
            fn real_minimum(&self, rhs: &Self) -> Result<Self, ProgramError>;
        }

        impl_array_elementwise_operation!(
            @binary
            TestRealMinimum, real_minimum,
            operation = "min",
            inputs = @numeric @real,
            checks = [@no_unreduced, @same_reduced_axes],
            |lhs, rhs| Ok(ArrayElement::min(&lhs, &rhs)),
        );

        // Empty inputs must satisfy the same input contract as non-empty inputs.
        let input = Array::from_elements(ArrayType::new_static(DataType::Boolean, [0]), &[] as &[bool])?;
        let output = input.minimum(&input);
        assert!(matches!(
            output,
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`min` does not support input data type `bool`",
        ));

        let output = input.real_minimum(&input);
        assert!(matches!(
            output,
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`min` does not support input data type `bool`",
        ));

        let input = Array::from_elements(ArrayType::new_static(DataType::C64, [0]), &[] as &[Complex<f32>])?;
        let output = input.real_minimum(&input);
        assert!(matches!(
            output,
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`min` does not support input data type `c64`",
        ));
        Ok(())
    }

    #[test]
    fn test_impl_array_elementwise_operation_binary_empty_invalid_metadata() -> Result<(), ProgramError> {
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
    fn test_impl_array_elementwise_operation_binary_empty_output() -> Result<(), ProgramError> {
        /// Operation whose scalar kernel always fails.
        trait TestFailingOperation: Sized {
            /// Reports a scalar error whenever an element is evaluated.
            fn fail(&self, rhs: &Self) -> Result<Self, ProgramError>;
        }

        impl_array_elementwise_operation!(
            @binary
            TestFailingOperation, fail,
            operation = "min",
            inputs = @numeric @real,
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
    fn test_impl_array_elementwise_operation_binary_scalar_error() {
        /// Operation whose scalar kernel always fails.
        trait TestFailingOperation: Sized {
            /// Reports a scalar error whenever an element is evaluated.
            fn fail(&self, rhs: &Self) -> Result<Self, ProgramError>;
        }

        impl_array_elementwise_operation!(
            @binary
            TestFailingOperation, fail,
            operation = "min",
            inputs = @numeric @real,
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
