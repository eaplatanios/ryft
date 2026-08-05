use half::{bf16, f16};
use num_complex::Complex;

use crate::arrays::ArrayAddressing;
use crate::programs::ProgramError;
use crate::programs::types::TypeError;
use crate::types::{ArrayType, DataType};

/// Value type with exactly one portable array-element byte encoding. [`ArrayElement`] pairs each supported Rust type
/// with the one [`DataType`] it represents (e.g., [`f32`] with [`DataType::F32`]) and pins the exact bytes that one
/// value of that type contributes to physical array storage, so that typed Rust slices convert to and from stored array
/// bytes deterministically on every platform:
///
///   - Booleans encode as one byte holding `0` or `1`.
///   - Integers encode as their little-endian two's-complement bytes.
///   - Floating-point values (i.e., [`bf16`], [`f16`](struct@f16), [`f32`], and [`f64`]) encode as their little-endian
///     IEEE bit patterns, preserving signed zeros and NaN payload bits exactly.
///   - [`Complex`] values encode their real component immediately followed by their imaginary component.
///
/// For example, `-2i32` always encodes as `[254, 255, 255, 255]` and `Complex::new(1.5f32, -2.0)` always encodes as
/// `[0, 0, 192, 63, 0, 0, 0, 192]`, regardless of the host's native endianness. The compile-time [`DataType`] pairing
/// makes mismatched conversions fail before any bytes move: decoding an `f32[2]` array as [`i32`] values is an error
/// even though both encodings are four bytes wide.
///
/// The trait is sealed because its encodings define the array storage format itself and a foreign implementation could
/// write bytes that the checked storage boundary rejects, or worse, reinterprets with a different meaning. Sub-byte
/// integers and the low-precision floating-point formats that have no corresponding Rust type deliberately have no
/// [`ArrayElement`] implementation; their values cross the same boundary as validated raw element bytes instead.
pub trait ArrayElement: private::Codec {}

impl<T: private::Codec> ArrayElement for T {}

// TODO(eaplatanios): Review from here onwards.

mod private {
    use crate::types::DataType;

    /// Private implementation contract that seals and implements [`super::ArrayElement`].
    pub trait Codec: Copy {
        /// [`DataType`] represented by this Rust value.
        const DATA_TYPE: DataType;

        /// Number of bytes in this value's portable representation.
        const BYTE_COUNT: usize;

        /// Writes this value's portable representation to an exactly sized byte slice.
        fn encode(self, bytes: &mut [u8]);

        /// Decodes one value from a byte slice whose length matches its representation.
        fn decode(bytes: &[u8]) -> Self;
    }
}

impl private::Codec for bool {
    const DATA_TYPE: DataType = DataType::Boolean;
    const BYTE_COUNT: usize = 1;

    #[inline]
    fn encode(self, bytes: &mut [u8]) {
        bytes[0] = u8::from(self);
    }

    #[inline]
    fn decode(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
}

// Implements the sealed codec for integer primitives using their exact little-endian two's-complement bit patterns.
macro_rules! impl_integer_array_element {
    ($type:ty, $data_type:ident) => {
        impl private::Codec for $type {
            const DATA_TYPE: DataType = DataType::$data_type;
            const BYTE_COUNT: usize = size_of::<Self>();

            #[inline]
            fn encode(self, bytes: &mut [u8]) {
                bytes.copy_from_slice(&self.to_le_bytes());
            }

            #[inline]
            fn decode(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().unwrap())
            }
        }
    };
}

impl_integer_array_element!(i8, I8);
impl_integer_array_element!(i16, I16);
impl_integer_array_element!(i32, I32);
impl_integer_array_element!(i64, I64);
impl_integer_array_element!(u8, U8);
impl_integer_array_element!(u16, U16);
impl_integer_array_element!(u32, U32);
impl_integer_array_element!(u64, U64);

// Implements the sealed codec for real floating-point primitives while preserving their exact payload bits.
macro_rules! impl_float_array_element {
    ($type:ty, $bits:ty, $data_type:ident) => {
        impl private::Codec for $type {
            const DATA_TYPE: DataType = DataType::$data_type;
            const BYTE_COUNT: usize = size_of::<$bits>();

            #[inline]
            fn encode(self, bytes: &mut [u8]) {
                bytes.copy_from_slice(&self.to_bits().to_le_bytes());
            }

            #[inline]
            fn decode(bytes: &[u8]) -> Self {
                Self::from_bits(<$bits>::from_le_bytes(bytes.try_into().unwrap()))
            }
        }
    };
}

impl_float_array_element!(bf16, u16, BF16);
impl_float_array_element!(f16, u16, F16);
impl_float_array_element!(f32, u32, F32);
impl_float_array_element!(f64, u64, F64);

impl private::Codec for Complex<f32> {
    const DATA_TYPE: DataType = DataType::C64;
    const BYTE_COUNT: usize = 8;

    #[inline]
    fn encode(self, bytes: &mut [u8]) {
        self.re.encode(&mut bytes[..4]);
        self.im.encode(&mut bytes[4..]);
    }

    #[inline]
    fn decode(bytes: &[u8]) -> Self {
        Self::new(<f32 as private::Codec>::decode(&bytes[..4]), <f32 as private::Codec>::decode(&bytes[4..]))
    }
}

impl private::Codec for Complex<f64> {
    const DATA_TYPE: DataType = DataType::C128;
    const BYTE_COUNT: usize = 16;

    #[inline]
    fn encode(self, bytes: &mut [u8]) {
        self.re.encode(&mut bytes[..8]);
        self.im.encode(&mut bytes[8..]);
    }

    #[inline]
    fn decode(bytes: &[u8]) -> Self {
        Self::new(<f64 as private::Codec>::decode(&bytes[..8]), <f64 as private::Codec>::decode(&bytes[8..]))
    }
}

/// Encodes typed logical row-major elements into the physical storage declared by `type`. Storage bytes that no
/// logical element occupies, namely layout holes and tile padding, are set to zero.
pub(crate) fn encode_elements<T: ArrayElement>(r#type: &ArrayType, elements: &[T]) -> Result<Vec<u8>, ProgramError> {
    if r#type.data_type() != T::DATA_TYPE {
        return Err(TypeError::invalid(format!(
            "cannot encode {} values as array elements of data type {}",
            T::DATA_TYPE,
            r#type.data_type(),
        ))
        .into());
    }
    let addressing = ArrayAddressing::new(r#type.clone())?;
    if elements.len() != addressing.element_count() {
        return Err(TypeError::invalid(format!(
            "array type {} requires {} logical elements but got {}",
            r#type,
            addressing.element_count(),
            elements.len(),
        ))
        .into());
    }
    debug_assert_eq!(T::BYTE_COUNT, addressing.element_byte_width());
    let mut storage = vec![0; addressing.storage_byte_len()];
    if addressing.is_dense_row_major() {
        for (element, bytes) in elements.iter().zip(storage.chunks_exact_mut(T::BYTE_COUNT)) {
            element.encode(bytes);
        }
    } else {
        for (index, element) in elements.iter().enumerate() {
            element.encode(&mut storage[addressing.byte_range_for_flat_index(index)]);
        }
    }
    Ok(storage)
}

/// Validates already encoded logical row-major element bytes and places them into the physical storage declared by
/// `type`. Storage bytes that no logical element occupies, namely layout holes and tile padding, are set to zero.
pub(crate) fn encode_logical_bytes(r#type: &ArrayType, bytes: &[u8]) -> Result<Vec<u8>, ProgramError> {
    let addressing = ArrayAddressing::new(r#type.clone())?;
    if bytes.len() != addressing.logical_byte_len() {
        return Err(TypeError::invalid(format!(
            "array type {} requires {} logical element bytes but got {}",
            r#type,
            addressing.logical_byte_len(),
            bytes.len(),
        ))
        .into());
    }
    let element_byte_width = addressing.element_byte_width();
    validate_logical_bytes(r#type.data_type(), element_byte_width, bytes)?;
    if addressing.is_dense_row_major() {
        return Ok(bytes.to_vec());
    }
    let mut storage = vec![0; addressing.storage_byte_len()];
    for element in 0..addressing.element_count() {
        let logical_start = element * element_byte_width;
        let logical_range = logical_start..logical_start + element_byte_width;
        storage[addressing.byte_range_for_flat_index(element)].copy_from_slice(&bytes[logical_range]);
    }
    Ok(storage)
}

/// Validates physical storage and decodes its typed elements in logical row-major order.
pub(crate) fn decode_elements<T: ArrayElement>(r#type: &ArrayType, bytes: &[u8]) -> Result<Vec<T>, ProgramError> {
    if r#type.data_type() != T::DATA_TYPE {
        return Err(TypeError::invalid(format!(
            "cannot decode array elements of data type {} as {} values",
            r#type.data_type(),
            T::DATA_TYPE,
        ))
        .into());
    }
    let addressing = ArrayAddressing::new(r#type.clone())?;
    validate_storage_bytes_with_addressing(&addressing, bytes)?;
    debug_assert_eq!(T::BYTE_COUNT, addressing.element_byte_width());
    if addressing.is_dense_row_major() {
        return Ok(bytes.chunks_exact(T::BYTE_COUNT).map(T::decode).collect());
    }
    Ok((0..addressing.element_count())
        .map(|element| T::decode(&bytes[addressing.byte_range_for_flat_index(element)]))
        .collect())
}

/// Validates physical storage and returns its encoded elements in logical row-major order.
///
/// Layout holes and tile padding are not logical elements and must contain zero bytes. They are omitted from the
/// returned portable representation.
pub(crate) fn decode_logical_bytes(r#type: &ArrayType, bytes: &[u8]) -> Result<Vec<u8>, ProgramError> {
    let addressing = ArrayAddressing::new(r#type.clone())?;
    validate_storage_bytes_with_addressing(&addressing, bytes)?;
    if addressing.is_dense_row_major() {
        return Ok(bytes.to_vec());
    }
    let mut logical_bytes = Vec::with_capacity(addressing.logical_byte_len());
    for element in 0..addressing.element_count() {
        logical_bytes.extend_from_slice(&bytes[addressing.byte_range_for_flat_index(element)]);
    }
    Ok(logical_bytes)
}

/// Validates the length and logical element encodings of physical array storage.
pub(crate) fn validate_storage_bytes(r#type: &ArrayType, bytes: &[u8]) -> Result<(), ProgramError> {
    let addressing = ArrayAddressing::new(r#type.clone())?;
    validate_storage_bytes_with_addressing(&addressing, bytes)
}

/// Validates the storage length, every logical element encoding, and that layout holes and tile padding contain only
/// zero bytes, using an already checked addressing descriptor.
fn validate_storage_bytes_with_addressing(addressing: &ArrayAddressing, bytes: &[u8]) -> Result<(), ProgramError> {
    if bytes.len() != addressing.storage_byte_len() {
        return Err(TypeError::invalid(format!(
            "array type {} requires {} physical storage bytes but got {}",
            addressing.r#type(),
            addressing.storage_byte_len(),
            bytes.len(),
        ))
        .into());
    }
    if addressing.is_dense_row_major() {
        return validate_logical_bytes(addressing.r#type().data_type(), addressing.element_byte_width(), bytes);
    }
    let mut logical_nonzero_byte_count = 0usize;
    for element in 0..addressing.element_count() {
        let element_bytes = &bytes[addressing.byte_range_for_flat_index(element)];
        validate_element_bytes(addressing.r#type().data_type(), element, element_bytes)?;
        logical_nonzero_byte_count += element_bytes.iter().filter(|byte| **byte != 0).count();
    }
    if bytes.iter().filter(|byte| **byte != 0).count() != logical_nonzero_byte_count {
        return Err(TypeError::invalid("array layout holes and tile padding must contain zero bytes").into());
    }
    Ok(())
}

/// Validates contiguous logical element encodings.
fn validate_logical_bytes(data_type: DataType, element_byte_width: usize, bytes: &[u8]) -> Result<(), ProgramError> {
    if element_byte_width == 0 {
        return Ok(());
    }
    for (element, bytes) in bytes.chunks_exact(element_byte_width).enumerate() {
        validate_element_bytes(data_type, element, bytes)?;
    }
    Ok(())
}

/// Validates one element's data-type-specific bit representation.
fn validate_element_bytes(data_type: DataType, element: usize, bytes: &[u8]) -> Result<(), ProgramError> {
    let valid = match data_type {
        DataType::Boolean => matches!(bytes, [0 | 1]),
        DataType::I1 | DataType::U1 => bytes[0] & !0b1 == 0,
        DataType::I2 | DataType::U2 => bytes[0] & !0b11 == 0,
        DataType::I4 | DataType::U4 | DataType::F4E2M1FN => bytes[0] & !0b1111 == 0,
        DataType::F6E2M3FN | DataType::F6E3M2FN => bytes[0] & !0b11_1111 == 0,
        _ => true,
    };
    if !valid {
        return Err(TypeError::invalid(format!(
            "array element {element} has invalid {data_type} byte encoding {bytes:?}",
        ))
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::{Dimension, Layout, Shape, StridedLayout, Tile, TileDimension, TiledLayout};

    use super::*;

    #[test]
    fn test_array_element_encoding() {
        let integers = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2)]));
        let integer_bytes = encode_elements(&integers, &[1i32, -2]).unwrap();
        assert_eq!(integer_bytes, [1, 0, 0, 0, 254, 255, 255, 255]);
        assert_eq!(decode_elements::<i32>(&integers, &integer_bytes), Ok(vec![1, -2]));

        let floats = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let float_values = [f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234)];
        let float_bytes = encode_elements(&floats, &float_values).unwrap();
        assert_eq!(float_bytes, [0, 0, 0, 128, 52, 18, 192, 127]);
        let decoded = decode_elements::<f32>(&floats, &float_bytes).unwrap();
        assert_eq!(decoded.into_iter().map(f32::to_bits).collect::<Vec<_>>(), [0x8000_0000, 0x7fc0_1234]);

        let complex = ArrayType::scalar(DataType::C64);
        let complex_bytes = encode_elements(&complex, &[Complex::new(1.5f32, -2.0)]).unwrap();
        assert_eq!(complex_bytes, [0, 0, 192, 63, 0, 0, 0, 192]);
        assert_eq!(decode_elements::<Complex<f32>>(&complex, &complex_bytes), Ok(vec![Complex::new(1.5, -2.0)]));
    }

    #[test]
    fn test_layout_aware_array_encoding() {
        let shape = Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]);
        let values = [1u8, 2, 3, 4, 5, 6];
        let cases = [
            (ArrayType::new(DataType::U8, shape.clone()), vec![1, 2, 3, 4, 5, 6]),
            // Explicit layouts that reproduce dense row-major storage follow the bulk dense encoding paths.
            (
                ArrayType::new(DataType::U8, shape.clone())
                    .with_layout(Layout::Strided(StridedLayout::new(vec![3, 1]))),
                vec![1, 2, 3, 4, 5, 6],
            ),
            (
                ArrayType::new(DataType::U8, shape.clone())
                    .with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()))),
                vec![1, 2, 3, 4, 5, 6],
            ),
            (
                ArrayType::new(DataType::U8, shape.clone())
                    .with_layout(Layout::Strided(StridedLayout::new(vec![4, 1]))),
                vec![1, 2, 3, 0, 4, 5, 6],
            ),
            (
                ArrayType::new(DataType::U8, shape.clone())
                    .with_layout(Layout::Strided(StridedLayout::new(vec![-4, 1]))),
                vec![4, 5, 6, 0, 1, 2, 3],
            ),
            (
                ArrayType::new(DataType::U8, shape.clone())
                    .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
                vec![1, 4, 2, 5, 3, 6],
            ),
            (
                ArrayType::new(DataType::U8, shape).with_layout(Layout::Tiled(TiledLayout::new(
                    vec![1, 0],
                    vec![Tile::new(vec![TileDimension::Sized(2), TileDimension::Sized(2)])],
                ))),
                vec![1, 2, 4, 5, 3, 0, 6, 0],
            ),
        ];

        for (r#type, expected_bytes) in cases {
            let bytes = encode_elements(&r#type, &values).unwrap();
            assert_eq!(bytes, expected_bytes);
            assert_eq!(bytes.len(), ArrayAddressing::new(r#type.clone()).unwrap().storage_byte_len());
            assert_eq!(decode_elements::<u8>(&r#type, &bytes), Ok(values.to_vec()));
            assert_eq!(decode_logical_bytes(&r#type, &bytes), Ok(values.to_vec()));
        }
    }

    #[test]
    fn test_array_byte_validation() {
        let booleans = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)]));
        assert_eq!(encode_logical_bytes(&booleans, &[0, 1]), Ok(vec![0, 1]));
        assert!(matches!(
            encode_logical_bytes(&booleans, &[0, 2]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array element 1 has invalid bool byte encoding [2]",
        ));
        assert!(matches!(
            validate_storage_bytes(&booleans, &[0]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array type bool[2] requires 2 physical storage bytes but got 1",
        ));

        let narrow = ArrayType::scalar(DataType::I2);
        assert_eq!(encode_logical_bytes(&narrow, &[0b11]), Ok(vec![0b11]));
        assert!(matches!(
            encode_logical_bytes(&narrow, &[0b100]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array element 0 has invalid i2 byte encoding [4]",
        ));
        let low_precision = ArrayType::scalar(DataType::F4E2M1FN);
        assert_eq!(encode_logical_bytes(&low_precision, &[0b1111]), Ok(vec![0b1111]));
        assert!(matches!(
            encode_logical_bytes(&low_precision, &[0b1_0000]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array element 0 has invalid f4e2m1fn byte encoding [16]",
        ));

        assert!(matches!(
            encode_elements(&booleans, &[0u8, 1]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot encode u8 values as array elements of data type bool",
        ));
        assert!(matches!(
            encode_elements(&booleans, &[true]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array type bool[2] requires 2 logical elements but got 1",
        ));

        let padded = ArrayType::new(DataType::U8, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_layout(Layout::Tiled(TiledLayout::new(
                vec![1, 0],
                vec![Tile::new(vec![TileDimension::Sized(2), TileDimension::Sized(2)])],
            )));
        let mut bytes = encode_elements(&padded, &[1u8, 2, 3, 4, 5, 6]).unwrap();
        bytes[5] = 255;
        bytes[7] = 254;
        assert!(matches!(
            validate_storage_bytes(&padded, &bytes),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array layout holes and tile padding must contain zero bytes",
        ));

        let payload_free = ArrayType::new(DataType::Token, Shape::new(vec![Dimension::Static(usize::MAX)]));
        assert_eq!(encode_logical_bytes(&payload_free, &[]), Ok(Vec::new()));
        assert_eq!(validate_storage_bytes(&payload_free, &[]), Ok(()));
    }
}
