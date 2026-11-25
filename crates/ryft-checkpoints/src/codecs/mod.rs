//! Host codecs for block-scaled low precision arrays as checkpoints store them and as native kernels compute them.
//!
//! Two formats are covered. **NVFP4** stores every value as a 4-bit `e2m1` element (i.e., one of
//! `±{0, 0.5, 1, 1.5, 2, 3, 4, 6}`), every block of 16 consecutive values shares one 8-bit `e4m3` scale, and a group
//! of blocks (e.g., one expert of a weight array or one token of an activation array) shares one `f32` outer scale,
//! so that a value decodes as `element · block_scale · outer_scale`. **MXFP8** stores every value as an 8-bit `e4m3`
//! element, and every block of 32 consecutive values shares one power of two scale that is stored as a biased
//! exponent byte, so that a value decodes as `element · 2^(exponent − 127)`.
//!
//! The functions of this module operate on host byte slices in the layouts of released checkpoints. Packed NVFP4
//! elements hold two 4-bit codes per byte with the first value in the low nibble, and bit 3 of a code is its sign.
//! They serve three purposes. Importers use the decoders to expand packed weights into floating-point parameters.
//! Exporters use the encoders to pack trained weights again. Tests use both as the reference against which the
//! quantizers of `ryft-core` that models stage into compiled programs (i.e., `BlockQuantize` and `block_scaled_dot`)
//! are checked, which is why the encoders mirror the arithmetic of the native reference kernels operation by
//! operation in `f32` rather than computing in a wider type.
//!
//! The encoders implement the rules that were verified to reproduce the bytes of the Inkling 2 Mini release exactly.
//! NVFP4 block scales map the largest magnitude of a block to the largest element magnitude (i.e., `6`), optionally
//! choosing instead the scale that maps it to `4` when that lowers the reconstruction error of the block
//! ([`Nvfp4ScaleSelection::FourOverSix`]). MXFP8 block scales are the smallest power of two that keeps every scaled
//! value of the block within the largest element magnitude (i.e., `448`), which differs from the power of two that
//! keeps only the exponent of the largest value.

use std::sync::LazyLock;

use ryft_core::{ArrayType, DataType, decode_elements, f8e4m3fn};
use thiserror::Error;

pub mod mxfp8;
pub mod nvfp4;

pub use mxfp8::{MXFP8_BLOCK_SIZE, decode_mxfp8, encode_mxfp8};
pub use nvfp4::{
    NVFP4_BLOCK_SIZE, Nvfp4OuterScale, Nvfp4ScaleSelection, decode_nvfp4, encode_nvfp4, pack_nvfp4_elements,
    unpack_nvfp4_elements,
};

// TODO(eaplatanios): Review this module.

/// Error returned when a block-scaled array cannot be decoded or encoded.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum CodecError {
    #[error("invalid block-scaled layout: {message}")]
    InvalidLayout { message: String },

    #[error("invalid block-scaled encoding: {message}")]
    InvalidEncoding { message: String },

    #[error("invalid value for block-scaled quantization: {message}")]
    InvalidValue { message: String },
}

/// Largest finite magnitude of the `e4m3` format, which bounds NVFP4 block scales and MXFP8 elements.
const E4M3_MAXIMUM: f32 = 448.0;

/// Values of the 256 `e4m3` bytes, indexed by byte. The two NaN encodings decode to NaN.
static E4M3_VALUES: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let bytes = (0..=255u8).collect::<Vec<_>>();
    let elements = decode_elements::<f8e4m3fn>(&ArrayType::new_static(DataType::F8E4M3FN, [256]), &bytes).unwrap();
    std::array::from_fn(|byte| elements[byte].to_f32())
});
