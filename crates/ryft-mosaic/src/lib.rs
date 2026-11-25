//! Mosaic compiler adapters over backend-independent Ryft kernel definitions.
//!
//! The GPU adapter emits checked compiler input. Execution, buffers, native compilation caches, and executable
//! persistence belong to the execution integration. This crate does not depend on XLA execution or PJRT.

// TODO(eaplatanios): Review this module.

pub mod kernels;
