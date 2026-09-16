//! Direct Triton compilation of verified Ryft kernels.
//!
//! [`kernels`] owns typed TTIR lowering and the linked `ryft-xla-sys` native compiler bridge. Compilation does not
//! create a GPU context. Concrete platform artifacts retain their existing launcher and ownership contracts.

pub mod kernels;
