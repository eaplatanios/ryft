//! The `affine` dialect provides a powerful abstraction for affine [`Operation`](crate::Operation)s and analyses.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/) for more information.

pub mod affine_expressions;
pub mod affine_maps;
pub mod integer_sets;

pub use affine_expressions::*;
pub use affine_maps::*;
pub use integer_sets::*;
