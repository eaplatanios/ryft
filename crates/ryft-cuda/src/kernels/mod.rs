//! Optional CUDA kernel compiler adapters over canonical Ryft kernels.

// TODO(eaplatanios): Review this module.

#[cfg(feature = "cutile")]
pub mod cutile;
