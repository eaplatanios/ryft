//! Contains reference backend implementations that exercise the Ryft tracing, transformation, interpretation, partial
//! evaluation, batching, differentiation, and transposition machinery without depending on an optimized backend such
//! as the XLA backend.

pub mod scalars;

pub use scalars::{Scalar, ScalarOperation, ScalarTracingContext};
