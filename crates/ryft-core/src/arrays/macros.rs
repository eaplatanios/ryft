/// Dispatches one runtime [`DataType`](crate::types::DataType) to its sealed Rust array element type, so that array
/// kernels can run generic element code without storing or allocating another dynamic element representation.
///
/// The macro takes a class selector, a [`DataType`](crate::types::DataType) expression, and an `|Element| body`
/// closure-like form, and expands to a `match` that binds the type alias `Element` to the matching element type from
/// [`arrays::encoding`](crate::arrays::encoding) in every selected arm. The body is instantiated once per selected
/// element type and may therefore use `Element` in any type position, including calls to generic functions bounded
/// by [`ArrayElement`](crate::arrays::encoding::ArrayElement) or by kernel capability traits:
///
/// ```
/// # use ryft_core::arrays::macros::dispatch_on_array_element_type;
/// # use ryft_core::types::DataType;
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
/// so a [`DataType`](crate::DataType) outside the selected class (including the payload-free
/// [`Token`](crate::DataType::Token) and [`Zero`](crate::DataType::Zero) types, which no selector includes) is an
/// internal invariant violation and panics with a descriptive message rather than forcing every body to return a
/// [`Result`].
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
                $crate::types::DataType::$variant => {
                    type $element = $element_type;
                    $body
                }
            )+
            other => unreachable!("unsupported element data type {other} for this dispatch"),
        }
    };
}

pub use crate::dispatch_on_array_element_type;

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::encoding::ArrayElement;
    use crate::types::DataType;

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
}
