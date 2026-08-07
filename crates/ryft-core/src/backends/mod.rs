//! Contains backend implementations that exercise the Ryft tracing, transformation, interpretation, partial evaluation,
//! batching, differentiation, and transposition machinery for arrays and dimension values without depending on an
//! optimized backend such as the XLA backend.

pub mod array_programs;
pub mod arrays;
pub mod dimensions;

pub use array_programs::{ArrayIrOperation, ArrayIrValue};
pub use arrays::{Array, ArrayOperation, ArrayTracingContext};
pub use dimensions::{DimensionOperation, DimensionTracingContext, DimensionValue};
