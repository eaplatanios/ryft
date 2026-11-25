//! MXFP8 codec, which stores every value as an 8-bit `e4m3` element with one power of two scale per block of 32
//! values. Refer to the [module documentation](super) for the shared conventions.

use ryft_core::f8e4m3fn;

use crate::codecs::{CodecError, E4M3_MAXIMUM, E4M3_VALUES};

// TODO(eaplatanios): Review this module.

/// Number of values that share one block scale in the MXFP8 format.
pub const MXFP8_BLOCK_SIZE: usize = 32;

/// Decodes MXFP8 values into `f32` values. Every value is computed as `element · 2^(exponent − 127)` in `f32`.
///
/// # Parameters
///
///   - `elements`: `e4m3` element bytes, one per value.
///   - `scale_exponents`: Biased exponent bytes of the power of two block scales, one per 32 values.
///
/// # Errors
///
/// Returns [`CodecError::InvalidLayout`] if the number of values is not 32 times the number of block scales,
/// and [`CodecError::InvalidEncoding`] if an element is not finite or if a scale exponent byte is `255`, which
/// encodes NaN.
pub fn decode_mxfp8(elements: &[u8], scale_exponents: &[u8]) -> Result<Vec<f32>, CodecError> {
    if elements.len() != scale_exponents.len() * MXFP8_BLOCK_SIZE {
        return Err(CodecError::InvalidLayout {
            message: format!(
                "{} MXFP8 values do not match {} block scales of {MXFP8_BLOCK_SIZE} values",
                elements.len(),
                scale_exponents.len(),
            ),
        });
    }

    let mut values = Vec::with_capacity(elements.len());
    for (block, &exponent) in elements.as_chunks::<MXFP8_BLOCK_SIZE>().0.iter().zip(scale_exponents) {
        if exponent == 255 {
            return Err(CodecError::InvalidEncoding {
                message: "MXFP8 scale exponent byte 0xff encodes NaN".to_string(),
            });
        }
        let scale = (f32::from(exponent) - 127.0).exp2();
        for &byte in block {
            let element = E4M3_VALUES[byte as usize];
            if !element.is_finite() {
                return Err(CodecError::InvalidEncoding {
                    message: format!("MXFP8 element byte {byte:#04x} is not finite"),
                });
            }
            values.push(element * scale);
        }
    }
    Ok(values)
}

/// Encodes `f32` values as MXFP8, returning the `e4m3` element bytes and the biased exponent bytes of the power of two
/// block scales. The scale of a block is the smallest power of two that is at least `amax / 448`, where `amax` is the
/// largest magnitude of the block, and it is clamped to `[2^-127, 2^127]`. Elements round to the nearest `e4m3` value
/// with ties to even after being clamped to `±448`. The arithmetic mirrors the native reference kernels in `f32`.
///
/// # Errors
///
/// Returns [`CodecError::InvalidLayout`] if the number of values is not a multiple of 32, and
/// [`CodecError::InvalidValue`] if a value is NaN.
pub fn encode_mxfp8(values: &[f32]) -> Result<(Vec<u8>, Vec<u8>), CodecError> {
    if !values.len().is_multiple_of(MXFP8_BLOCK_SIZE) {
        return Err(CodecError::InvalidLayout {
            message: format!("{} values do not form MXFP8 blocks of {MXFP8_BLOCK_SIZE} values", values.len()),
        });
    }
    if values.iter().any(|value| value.is_nan()) {
        return Err(CodecError::InvalidValue { message: "MXFP8 cannot encode NaN values".to_string() });
    }

    let minimum_scale = (-127.0f32).exp2();
    let mut elements = Vec::with_capacity(values.len());
    let mut scale_exponents = Vec::with_capacity(values.len() / MXFP8_BLOCK_SIZE);
    for block in values.as_chunks::<MXFP8_BLOCK_SIZE>().0 {
        let amax = block.iter().fold(0.0f32, |amax, value| amax.max(value.abs()));
        let raw_scale = amax * (1.0 / E4M3_MAXIMUM);

        // Adding the mantissa mask to the bits of a positive value carries into the exponent exactly when the
        // mantissa is not zero, which rounds the value up to a power of two.
        let exponent = ((raw_scale.to_bits().wrapping_add(0x007f_ffff)) & 0x7f80_0000) >> 23;
        let exponent = if raw_scale <= minimum_scale { 0 } else { exponent.min(254) };
        let scale = f32::from_bits(exponent << 23).max(minimum_scale);
        let inverse = if exponent == 0 { 127.0f32.exp2() } else { 1.0 / scale };
        scale_exponents.push(exponent as u8);
        elements.extend(
            block.iter().map(|&value| {
                f8e4m3fn::from_f32((value * inverse).clamp(-E4M3_MAXIMUM, E4M3_MAXIMUM)).unwrap().to_bits()
            }),
        );
    }
    Ok((elements, scale_exponents))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    /// Leading two blocks of the elements and scale exponents of one output row of both expert projections of layer
    /// 20, expert 3, of the Inkling 2 Mini release, which are stored as MXFP8.
    const MXFP8_W13_ROW: &[u8] = &[
        0xc0, 0x67, 0xee, 0x6d, 0x67, 0xdc, 0xe8, 0x70, 0x66, 0x6a, 0xea, 0xd8, 0x6c, 0x69, 0x6a, 0x67, 0x63, 0xf4,
        0x68, 0x66, 0x64, 0x7a, 0x70, 0xd3, 0x5d, 0xe8, 0xe4, 0x6d, 0xf4, 0x68, 0xf3, 0x52, 0xf7, 0x66, 0xe3, 0xec,
        0xf9, 0x72, 0x5a, 0xc1, 0xf1, 0x79, 0xc1, 0x71, 0x7d, 0x68, 0x73, 0xe4, 0xe5, 0xf6, 0xf0, 0xf3, 0x77, 0x72,
        0xeb, 0x61, 0x6d, 0xf9, 0x5e, 0xe7, 0xf2, 0x79, 0xfa, 0xe8,
    ];
    const MXFP8_W13_ROW_SCALES: &[u8] = &[0x73, 0x72];
    const MXFP8_W2_ROW: &[u8] = &[
        0x5a, 0xf9, 0xdf, 0xeb, 0xf2, 0xef, 0xf8, 0x6c, 0x68, 0x69, 0xe0, 0xee, 0x5a, 0xfa, 0xf0, 0x6b, 0xbb, 0x74,
        0xd6, 0xf3, 0x4d, 0xdb, 0x70, 0xea, 0xeb, 0x48, 0xd4, 0x6b, 0xda, 0x39, 0x53, 0xde, 0xc1, 0xd2, 0xf1, 0xe8,
        0x50, 0x5a, 0x69, 0xd2, 0x79, 0xf2, 0x68, 0x72, 0xd9, 0x5c, 0xf1, 0xf3, 0x60, 0xdb, 0xe4, 0x70, 0x6c, 0xf0,
        0xe9, 0x73, 0x6f, 0xf5, 0xe9, 0x6b, 0xf2, 0x73, 0xf8, 0x6e,
    ];
    const MXFP8_W2_ROW_SCALES: &[u8] = &[0x73, 0x73];

    #[test]
    fn test_decode_mxfp8() {
        // Element bytes 0x38 (1), 0xc0 (-2), and 0x7e (448) with scale exponent byte 126 (i.e., one half).
        let mut elements = [0u8; 32];
        elements[..3].copy_from_slice(&[0x38, 0xc0, 0x7e]);
        let values = decode_mxfp8(&elements, &[126]).unwrap();
        assert_eq!(values[..4], [0.5, -1.0, 224.0, 0.0]);

        assert_eq!(
            decode_mxfp8(&elements[..31], &[126]),
            Err(CodecError::InvalidLayout {
                message: "31 MXFP8 values do not match 1 block scales of 32 values".to_string(),
            }),
        );
        assert_eq!(
            decode_mxfp8(&elements, &[255]),
            Err(CodecError::InvalidEncoding { message: "MXFP8 scale exponent byte 0xff encodes NaN".to_string() }),
        );
        elements[5] = 0x7f;
        assert_eq!(
            decode_mxfp8(&elements, &[126]),
            Err(CodecError::InvalidEncoding { message: "MXFP8 element byte 0x7f is not finite".to_string() }),
        );
    }

    #[test]
    fn test_decode_mxfp8_released_weights() {
        let values = decode_mxfp8(MXFP8_W13_ROW, MXFP8_W13_ROW_SCALES).unwrap();
        assert_eq!(values.len(), 64);

        // The first element byte is 0xc0 (i.e., -2) and the first scale exponent byte is 0x73 (i.e., 2^-12).
        assert_eq!(MXFP8_W13_ROW[0], 0xc0);
        assert_eq!(MXFP8_W13_ROW_SCALES[0], 0x73);
        assert_eq!(values[0], -2.0 * (-12.0f32).exp2());
    }

    #[test]
    fn test_encode_mxfp8() {
        // A block whose largest magnitude is 448 has a unit scale, and one whose largest magnitude is just above 448
        // needs the next power of two.
        let mut values = [0.0f32; 64];
        values[..3].copy_from_slice(&[448.0, -1.0, 0.0625]);
        values[32..35].copy_from_slice(&[449.0, -2.0, 0.125]);
        let (elements, scale_exponents) = encode_mxfp8(&values).unwrap();
        assert_eq!(scale_exponents, vec![127, 128]);
        assert_eq!(elements[..3], [0x7e, 0xb8, 0x18]);
        assert_eq!(decode_mxfp8(&elements, &scale_exponents).unwrap()[32..35], [448.0, -2.0, 0.125]);

        // An all-zero block uses the smallest scale.
        assert_eq!(encode_mxfp8(&[0.0; 32]), Ok((vec![0x00; 32], vec![0])));

        assert_eq!(
            encode_mxfp8(&values[..33]),
            Err(CodecError::InvalidLayout { message: "33 values do not form MXFP8 blocks of 32 values".to_string() }),
        );
        assert_eq!(
            encode_mxfp8(&[f32::NAN; 32]),
            Err(CodecError::InvalidValue { message: "MXFP8 cannot encode NaN values".to_string() }),
        );
    }

    #[test]
    fn test_encode_mxfp8_released_weights() {
        // Decoding released weights and encoding them again reproduces the released bytes exactly, for both expert
        // projections.
        for (elements, scale_exponents) in [(MXFP8_W13_ROW, MXFP8_W13_ROW_SCALES), (MXFP8_W2_ROW, MXFP8_W2_ROW_SCALES)]
        {
            let values = decode_mxfp8(elements, scale_exponents).unwrap();
            assert_eq!(encode_mxfp8(&values), Ok((elements.to_vec(), scale_exponents.to_vec())));
        }
    }
}
