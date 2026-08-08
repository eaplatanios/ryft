//! Contains backend implementations that exercise the Ryft tracing, transformation, interpretation, partial evaluation,
//! batching, differentiation, and transposition machinery for arrays and dimension values without depending on an
//! optimized backend such as the XLA backend.

pub mod arrays;

pub use arrays::Array;
