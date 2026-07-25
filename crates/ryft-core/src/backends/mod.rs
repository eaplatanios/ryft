//! Contains reference backend implementations that exercise the Ryft tracing, transformation, interpretation, partial
//! evaluation, batching, differentiation, and transposition machinery without depending on an optimized backend such
//! as the XLA backend.

pub mod arrays;
pub mod dimensions;
pub mod scalars;

pub use arrays::{Array, ArrayOperation, ArrayTracingContext};
pub use dimensions::{
    DimensionArithmetic, DimensionArithmeticOperation, DimensionOperation, DimensionTracingContext, DimensionValue,
};
pub use scalars::{Scalar, ScalarOperation, ScalarTracingContext};
