//! Direct Triton compilation of verified Ryft kernels.
//!
//! [`kernels`] owns typed TTIR lowering and an explicitly selected native compiler process. Compilation does not
//! create a GPU context. Concrete platform artifacts retain their existing launcher and ownership contracts.

pub mod kernels;
