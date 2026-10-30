//! The [`CompositeOperation`] that calls a primal program with custom transform rules and the
//! [`CompositeTransposeOperation`] carrier that applies a retained or attached backward rule during transposition.

mod call;
mod transpose;

pub use call::{COMPOSITE_OPERATION_NAME, CompositeJvpRule, CompositeOperation};
pub use transpose::{COMPOSITE_TRANSPOSE_OPERATION_NAME, CompositeTransposeOperation};
