//! NVFP4 codec, which stores every value as a 4-bit `e2m1` element with one `e4m3` scale per block of 16 values and
//! one `f32` outer scale per group of blocks. Refer to the [module documentation](super) for the shared conventions.

use std::sync::LazyLock;

use ryft_core::{ArrayType, DataType, decode_elements, f4e2m1fn, f8e4m3fn};

use crate::codecs::{CodecError, E4M3_MAXIMUM, E4M3_VALUES};

// TODO(eaplatanios): Review this module.

/// Number of values that share one block scale in the NVFP4 format.
pub const NVFP4_BLOCK_SIZE: usize = 16;

/// Rule that selects the block scale of an NVFP4 block from the largest magnitude of its values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Nvfp4ScaleSelection {
    /// The block scale maps the largest magnitude of the block to `6`, the largest element magnitude. Released
    /// NVFP4 weights use this rule.
    Six,

    /// Both the scale that maps the largest magnitude of the block to `6` and the one that maps it to `4` are tried,
    /// and the latter is kept only when the sum of the absolute reconstruction errors of the block is strictly
    /// smaller with it. This is the "four over six" rule with the mean absolute error criterion that native kernels
    /// apply to activations.
    FourOverSix,
}

/// Outer scale of one NVFP4 group of blocks (e.g., one expert or one token), together with the two derived factors
/// that the encoder multiplies by. The three factors are kept separately because the reference kernels compute each
/// of them with one rounded `f32` operation, and deriving one from another would round differently.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Nvfp4OuterScale {
    /// Factor that decoded values are multiplied by.
    scale: f32,

    /// Factor that maps values into the range of the block scales when encoding (i.e., the reciprocal of the scale).
    encoder_scale: f32,

    /// Factor that maps block scales back to the range of the values when encoding.
    decoder_scale: f32,
}

impl Nvfp4OuterScale {
    /// Creates the outer scale of a weight group whose largest magnitude is `amax`, as released checkpoints store
    /// it. The outer scale is `amax / denominator`, where the released denominator `2688 = 6 · 448` maps `amax` to
    /// the product of the largest element and block scale magnitudes.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidValue`] if `amax` is negative or not finite, or if `denominator` is not
    /// positive and finite.
    pub fn for_weights(amax: f32, denominator: f32) -> Result<Self, CodecError> {
        if !amax.is_finite() || amax < 0.0 || !denominator.is_finite() || denominator <= 0.0 {
            return Err(CodecError::InvalidValue {
                message: format!(
                    "NVFP4 weight outer scales need a finite non-negative `amax` and a finite positive denominator \
                     but got {amax} and {denominator}",
                ),
            });
        }
        let encoder_scale = (denominator / amax).min(f32::MAX);
        Ok(Self { scale: amax / denominator, encoder_scale, decoder_scale: 1.0 / encoder_scale })
    }

    /// Creates the outer scale of an activation group (i.e., one token) from its values, as the native reference
    /// kernels compute it. The outer scale is `amax · (1 / 1536)`, where `amax` is the largest magnitude of `values`
    /// and `1536 = 6 · 256` leaves the upper part of the block scale range unused.
    pub fn for_activations(values: &[f32]) -> Self {
        let amax = values.iter().fold(0.0f32, |amax, value| amax.max(value.abs()));
        let encoder_scale = (1536.0 / amax).min(f32::MAX);
        Self { scale: amax * (1.0 / 1536.0), encoder_scale, decoder_scale: 1.0 / encoder_scale }
    }

    /// Returns the factor that decoded values are multiplied by.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

/// Unpacks NVFP4 element bytes, which hold two 4-bit codes each with the first one in the low nibble, into one code
/// per byte. The result is the host storage of an [`F4E2M1FN`](DataType::F4E2M1FN) array.
pub fn unpack_nvfp4_elements(packed: &[u8]) -> Vec<u8> {
    packed.iter().flat_map(|&byte| [byte & 0x0f, byte >> 4]).collect()
}

/// Packs 4-bit NVFP4 element codes, provided as one code per byte, into bytes that hold two codes each with the first
/// one in the low nibble. This is the inverse of [`unpack_nvfp4_elements`].
///
/// # Errors
///
/// Returns [`CodecError::InvalidLayout`] if the number of codes is odd, and
/// [`CodecError::InvalidEncoding`] if a code does not fit in 4 bits.
pub fn pack_nvfp4_elements(codes: &[u8]) -> Result<Vec<u8>, CodecError> {
    if !codes.len().is_multiple_of(2) {
        return Err(CodecError::InvalidLayout {
            message: format!("cannot pack an odd number of 4-bit codes ({})", codes.len()),
        });
    }
    if let Some(code) = codes.iter().find(|&&code| code > 0x0f) {
        return Err(CodecError::InvalidEncoding { message: format!("code {code:#04x} does not fit in 4 bits") });
    }
    Ok(codes.as_chunks::<2>().0.iter().map(|pair| pair[0] | (pair[1] << 4)).collect())
}

/// Decodes packed NVFP4 values that share one outer scale into `f32` values. Every value is computed as
/// `(element · block_scale) · outer_scale` with `f32` arithmetic in that order, which is the order of the reference
/// checkpoint adapter, so decoding is reproducible bit for bit.
///
/// # Parameters
///
///   - `packed`: Packed element bytes, two values per byte with the first one in the low nibble.
///   - `block_scales`: `e4m3` block scale bytes, one per 16 values.
///   - `outer_scale`: Factor shared by all values (refer to [`Nvfp4OuterScale::scale`]).
///
/// # Errors
///
/// Returns [`CodecError::InvalidLayout`] if the number of values is not 16 times the number of block scales,
/// [`CodecError::InvalidEncoding`] if a block scale is negative or not finite, and
/// [`CodecError::InvalidValue`] if `outer_scale` is negative or not finite.
pub fn decode_nvfp4(packed: &[u8], block_scales: &[u8], outer_scale: f32) -> Result<Vec<f32>, CodecError> {
    if packed.len() * 2 != block_scales.len() * NVFP4_BLOCK_SIZE {
        return Err(CodecError::InvalidLayout {
            message: format!(
                "{} packed NVFP4 bytes hold {} values, which do not match {} block scales of {NVFP4_BLOCK_SIZE} values",
                packed.len(),
                packed.len() * 2,
                block_scales.len(),
            ),
        });
    }
    if !outer_scale.is_finite() || outer_scale < 0.0 {
        return Err(CodecError::InvalidValue {
            message: format!("NVFP4 outer scale must be finite and non-negative but got {outer_scale}"),
        });
    }

    let mut values = Vec::with_capacity(packed.len() * 2);
    for (block, &scale_byte) in packed.as_chunks::<{ NVFP4_BLOCK_SIZE / 2 }>().0.iter().zip(block_scales) {
        let scale = E4M3_VALUES[scale_byte as usize];
        if !scale.is_finite() || scale.is_sign_negative() {
            return Err(CodecError::InvalidEncoding {
                message: format!("NVFP4 block scale byte {scale_byte:#04x} is negative or not finite"),
            });
        }
        for &byte in block {
            values.push(E2M1_VALUES[(byte & 0x0f) as usize] * scale * outer_scale);
            values.push(E2M1_VALUES[(byte >> 4) as usize] * scale * outer_scale);
        }
    }
    Ok(values)
}

/// Encodes `f32` values that share one outer scale as packed NVFP4, returning the packed element bytes and the `e4m3`
/// block scale bytes. Elements round to the nearest `e2m1` value with ties to even and saturate at `±6`, and the sign
/// of zero is kept. The arithmetic mirrors the native reference kernels in `f32` operation by operation, including
/// the pairwise summation order of the reconstruction errors that [`Nvfp4ScaleSelection::FourOverSix`] compares.
///
/// # Parameters
///
///   - `values`: Values to encode, whose number must be a multiple of 16.
///   - `outer_scale`: Outer scale shared by all values.
///   - `scale_selection`: Rule that selects the scale of each block.
///
/// # Errors
///
/// Returns [`CodecError::InvalidLayout`] if the number of values is not a multiple of 16, and
/// [`CodecError::InvalidValue`] if a value is NaN.
pub fn encode_nvfp4(
    values: &[f32],
    outer_scale: Nvfp4OuterScale,
    scale_selection: Nvfp4ScaleSelection,
) -> Result<(Vec<u8>, Vec<u8>), CodecError> {
    if !values.len().is_multiple_of(NVFP4_BLOCK_SIZE) {
        return Err(CodecError::InvalidLayout {
            message: format!("{} values do not form NVFP4 blocks of {NVFP4_BLOCK_SIZE} values", values.len()),
        });
    }
    if values.iter().any(|value| value.is_nan()) {
        return Err(CodecError::InvalidValue { message: "NVFP4 cannot encode NaN values".to_string() });
    }

    // Quantizes one block with the block scale that maps its largest magnitude to `target`, and returns the scale
    // byte, the element codes, and the sum of the absolute reconstruction errors.
    let candidate = |block: &[f32], block_max: f32, target_reciprocal: f32| {
        let scale = f8e4m3fn::from_f32((block_max * (outer_scale.encoder_scale * target_reciprocal)).min(E4M3_MAXIMUM))
            .unwrap();
        let scale_value = scale.to_f32();
        let inverse = (1.0 / (scale_value * outer_scale.decoder_scale)).min(f32::MAX);
        let mut codes = [0u8; NVFP4_BLOCK_SIZE];
        let mut errors = [0.0f32; NVFP4_BLOCK_SIZE];
        for (index, &value) in block.iter().enumerate() {
            let element = f4e2m1fn::from_f32(value * inverse).unwrap();
            codes[index] = element.to_bits();
            errors[index] = ((element.to_f32() * scale_value) * outer_scale.scale - value).abs();
        }
        let mut width = NVFP4_BLOCK_SIZE;
        while width > 1 {
            width /= 2;
            for index in 0..width {
                errors[index] = errors[2 * index] + errors[2 * index + 1];
            }
        }
        (scale.to_bits(), codes, errors[0])
    };

    let mut packed = Vec::with_capacity(values.len() / 2);
    let mut block_scales = Vec::with_capacity(values.len() / NVFP4_BLOCK_SIZE);
    for block in values.as_chunks::<NVFP4_BLOCK_SIZE>().0 {
        let block_max = block.iter().fold(0.0f32, |block_max, value| block_max.max(value.abs()));
        let (mut scale, mut codes, error) = candidate(block, block_max, 1.0 / 6.0);
        if scale_selection == Nvfp4ScaleSelection::FourOverSix {
            let (four_scale, four_codes, four_error) = candidate(block, block_max, 0.25);
            if four_error < error {
                (scale, codes) = (four_scale, four_codes);
            }
        }
        block_scales.push(scale);
        packed.extend(codes.as_chunks::<2>().0.iter().map(|pair| pair[0] | (pair[1] << 4)));
    }
    Ok((packed, block_scales))
}

/// Values of the 16 `e2m1` codes, indexed by code.
static E2M1_VALUES: LazyLock<[f32; 16]> = LazyLock::new(|| {
    let codes = (0..16u8).collect::<Vec<_>>();
    let elements = decode_elements::<f4e2m1fn>(&ArrayType::new_static(DataType::F4E2M1FN, [16]), &codes).unwrap();
    std::array::from_fn(|code| elements[code].to_f32())
});

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    /// Leading two blocks of the packed elements and block scales of one output row of the first expert projection
    /// of layer 0, expert 3, of the Inkling 2 Mini release, which is stored as NVFP4.
    const NVFP4_W13_ROW: &[u8] =
        &[0xba, 0x02, 0x2e, 0x40, 0xf5, 0xbc, 0x4d, 0x54, 0x38, 0x1e, 0xd9, 0x64, 0xf8, 0x66, 0x6f, 0xb9];
    const NVFP4_W13_ROW_SCALES: &[u8] = &[0x72, 0x6f];

    /// The same for the second expert projection.
    const NVFP4_W2_ROW: &[u8] =
        &[0xcd, 0x36, 0xe1, 0x0a, 0x4a, 0x15, 0x7c, 0xb7, 0x74, 0xec, 0xdd, 0xa2, 0x7a, 0x23, 0xe0, 0x8b];
    const NVFP4_W2_ROW_SCALES: &[u8] = &[0x6b, 0x67];

    /// Released per expert `amax` of expert 3 of layer 0 for both projections, and the released outer scale
    /// denominator.
    const NVFP4_W13_AMAX: f32 = f32::from_bits(0x3e28_6dc8);
    const NVFP4_W2_AMAX: f32 = f32::from_bits(0x3e9d_f977);
    const NVFP4_DENOMINATOR: f32 = 2688.0;

    #[test]
    fn test_nvfp4_outer_scale_for_weights() {
        let outer_scale = Nvfp4OuterScale::for_weights(2688.0, 2688.0).unwrap();
        assert_eq!(outer_scale.scale(), 1.0);
        assert_eq!(outer_scale, Nvfp4OuterScale { scale: 1.0, encoder_scale: 1.0, decoder_scale: 1.0 });

        // A zero `amax` describes an all-zero group, whose encoder factor saturates rather than becoming infinite.
        assert_eq!(
            Nvfp4OuterScale::for_weights(0.0, 2688.0),
            Ok(Nvfp4OuterScale { scale: 0.0, encoder_scale: f32::MAX, decoder_scale: 1.0 / f32::MAX }),
        );
        assert_eq!(
            Nvfp4OuterScale::for_weights(-1.0, 2688.0),
            Err(CodecError::InvalidValue {
                message: "NVFP4 weight outer scales need a finite non-negative `amax` and a finite positive \
                          denominator but got -1 and 2688"
                    .to_string(),
            }),
        );
        assert_eq!(
            Nvfp4OuterScale::for_weights(1.0, 0.0),
            Err(CodecError::InvalidValue {
                message: "NVFP4 weight outer scales need a finite non-negative `amax` and a finite positive \
                          denominator but got 1 and 0"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_nvfp4_outer_scale_for_activations() {
        assert_eq!(
            Nvfp4OuterScale::for_activations(&[0.5, -1536.0, 3.0]),
            Nvfp4OuterScale { scale: 1536.0 * (1.0 / 1536.0), encoder_scale: 1.0, decoder_scale: 1.0 },
        );
        assert_eq!(
            Nvfp4OuterScale::for_activations(&[0.0, -0.0]),
            Nvfp4OuterScale { scale: 0.0, encoder_scale: f32::MAX, decoder_scale: 1.0 / f32::MAX },
        );
    }

    #[test]
    fn test_unpack_nvfp4_elements() {
        assert_eq!(unpack_nvfp4_elements(&[0xba, 0x02, 0x2e]), vec![0x0a, 0x0b, 0x02, 0x00, 0x0e, 0x02]);
        assert_eq!(unpack_nvfp4_elements(&[]), Vec::<u8>::new());
    }

    #[test]
    fn test_pack_nvfp4_elements() {
        assert_eq!(pack_nvfp4_elements(&[0x0a, 0x0b, 0x02, 0x00, 0x0e, 0x02]), Ok(vec![0xba, 0x02, 0x2e]));
        assert_eq!(pack_nvfp4_elements(&unpack_nvfp4_elements(NVFP4_W13_ROW)), Ok(NVFP4_W13_ROW.to_vec()));
        assert_eq!(
            pack_nvfp4_elements(&[1, 2, 3]),
            Err(CodecError::InvalidLayout { message: "cannot pack an odd number of 4-bit codes (3)".to_string() }),
        );
        assert_eq!(
            pack_nvfp4_elements(&[1, 0x10]),
            Err(CodecError::InvalidEncoding { message: "code 0x10 does not fit in 4 bits".to_string() }),
        );
    }

    #[test]
    fn test_decode_nvfp4() {
        // Codes 0xa (-1), 0xb (-1.5), 0x2 (1), 0x0 (0), 0xe (-4), 0x7 (6), and 0x8 (-0) with block scale byte 0x38
        // (i.e., 1) and an outer scale of 0.5.
        let packed = [0xba, 0x02, 0x7e, 0x08, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            decode_nvfp4(&packed, &[0x38], 0.5),
            Ok(vec![-0.5, -0.75, 0.5, 0.0, -2.0, 3.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        );

        assert_eq!(
            decode_nvfp4(&packed, &[0x38, 0x38], 0.5),
            Err(CodecError::InvalidLayout {
                message: "8 packed NVFP4 bytes hold 16 values, which do not match 2 block scales of 16 values"
                    .to_string(),
            }),
        );
        assert_eq!(
            decode_nvfp4(&packed, &[0xb8], 0.5),
            Err(CodecError::InvalidEncoding {
                message: "NVFP4 block scale byte 0xb8 is negative or not finite".to_string(),
            }),
        );
        assert_eq!(
            decode_nvfp4(&packed, &[0x7f], 0.5),
            Err(CodecError::InvalidEncoding {
                message: "NVFP4 block scale byte 0x7f is negative or not finite".to_string(),
            }),
        );
        assert_eq!(
            decode_nvfp4(&packed, &[0x38], f32::INFINITY),
            Err(CodecError::InvalidValue {
                message: "NVFP4 outer scale must be finite and non-negative but got inf".to_string(),
            }),
        );
    }

    #[test]
    fn test_decode_nvfp4_released_weights() {
        let outer_scale = Nvfp4OuterScale::for_weights(NVFP4_W13_AMAX, NVFP4_DENOMINATOR).unwrap();
        let values = decode_nvfp4(NVFP4_W13_ROW, NVFP4_W13_ROW_SCALES, outer_scale.scale()).unwrap();
        assert_eq!(values.len(), 32);

        // The first packed byte is 0xba and the first block scale byte is 0x72 (i.e., 160), so the first two values
        // are -1 and -1.5 times the block scale times the outer scale.
        assert_eq!(NVFP4_W13_ROW[0], 0xba);
        assert_eq!(NVFP4_W13_ROW_SCALES[0], 0x72);
        assert_eq!(values[0], -160.0 * (NVFP4_W13_AMAX / 2688.0));
        assert_eq!(values[1], -1.5 * 160.0 * (NVFP4_W13_AMAX / 2688.0));
        assert!(values.iter().all(|value| value.abs() <= NVFP4_W13_AMAX));
    }

    #[test]
    fn test_encode_nvfp4() {
        // With a unit outer scale, a block whose largest magnitude is 6 has scale 1 and encodes its values directly,
        // rounding ties to even, saturating, and keeping the sign of zero.
        let outer_scale = Nvfp4OuterScale::for_weights(2688.0, 2688.0).unwrap();
        let values = [6.0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -5.5, 0.3, -0.2, 2.4, 3.6, 4.9, 0.0];
        assert_eq!(
            encode_nvfp4(&values, outer_scale, Nvfp4ScaleSelection::Six),
            Ok((vec![0x87, 0x20, 0x42, 0x64, 0xf6, 0x81, 0x64, 0x06], vec![0x38])),
        );

        // An all-zero block has a zero scale and zero elements.
        assert_eq!(
            encode_nvfp4(&[0.0; 16], Nvfp4OuterScale::for_activations(&[0.0; 16]), Nvfp4ScaleSelection::FourOverSix),
            Ok((vec![0x00; 8], vec![0x00])),
        );

        assert_eq!(
            encode_nvfp4(&values[..15], outer_scale, Nvfp4ScaleSelection::Six),
            Err(CodecError::InvalidLayout { message: "15 values do not form NVFP4 blocks of 16 values".to_string() }),
        );
        assert_eq!(
            encode_nvfp4(&[f32::NAN; 16], outer_scale, Nvfp4ScaleSelection::Six),
            Err(CodecError::InvalidValue { message: "NVFP4 cannot encode NaN values".to_string() }),
        );
    }

    #[test]
    fn test_encode_nvfp4_four_over_six() {
        // The values of this block are 4 and its multiples of one third, so the scale that maps the largest
        // magnitude to 4 represents the values 4 and 2 exactly while the scale that maps it to 6 cannot.
        let mut values = [0.0f32; 16];
        values[0] = 4.0;
        values[1] = 2.0;
        values[2] = 1.0;
        let outer_scale = Nvfp4OuterScale::for_weights(2688.0, 2688.0).unwrap();
        let (six_packed, six_scales) = encode_nvfp4(&values, outer_scale, Nvfp4ScaleSelection::Six).unwrap();
        let (four_packed, four_scales) = encode_nvfp4(&values, outer_scale, Nvfp4ScaleSelection::FourOverSix).unwrap();
        assert_eq!(decode_nvfp4(&four_packed, &four_scales, 1.0).unwrap()[..3], [4.0, 2.0, 1.0]);
        assert_eq!(four_scales, vec![0x38]);
        assert_ne!(six_scales, four_scales);
        assert_ne!(decode_nvfp4(&six_packed, &six_scales, 1.0).unwrap()[..3], [4.0, 2.0, 1.0]);

        // A block that the larger scale already represents exactly keeps it, because the comparison is strict.
        let mut values = [0.0f32; 16];
        values[0] = 6.0;
        values[1] = 3.0;
        assert_eq!(
            encode_nvfp4(&values, outer_scale, Nvfp4ScaleSelection::FourOverSix),
            encode_nvfp4(&values, outer_scale, Nvfp4ScaleSelection::Six),
        );
    }

    #[test]
    fn test_encode_nvfp4_released_weights() {
        // Decoding released weights and encoding them again with the released outer scale reproduces the released
        // bytes exactly, for both expert projections.
        for (packed, scales, amax) in
            [(NVFP4_W13_ROW, NVFP4_W13_ROW_SCALES, NVFP4_W13_AMAX), (NVFP4_W2_ROW, NVFP4_W2_ROW_SCALES, NVFP4_W2_AMAX)]
        {
            let outer_scale = Nvfp4OuterScale::for_weights(amax, NVFP4_DENOMINATOR).unwrap();
            let values = decode_nvfp4(packed, scales, outer_scale.scale()).unwrap();
            assert_eq!(
                encode_nvfp4(&values, outer_scale, Nvfp4ScaleSelection::Six),
                Ok((packed.to_vec(), scales.to_vec())),
            );
        }
    }
}
