//! Contains backend implementations that exercise the Ryft tracing, transformation, interpretation, partial evaluation,
//! batching, differentiation, and transposition machinery for various value types like scalars, arrays (without
//! depending on an optimized backend such as the XLA backend), and dimension values.

pub mod array_programs;
pub mod arrays;
pub mod dimensions;
pub mod scalars;

pub use array_programs::ArrayProgramValue;
pub use arrays::{Array, ArrayOperation, ArrayTracingContext};
pub use dimensions::{DimensionOperation, DimensionTracingContext, DimensionValue};
pub use scalars::{Scalar, ScalarOperation, ScalarTracingContext};
