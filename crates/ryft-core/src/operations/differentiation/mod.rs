//! Operations that control how values are differentiated rather than what they compute. Each operation is defined by an
//! [`Operation`](crate::Operation) type (e.g., [`StopGradientOperation`]) together with a user-facing function or value
//! capability trait (e.g., [`StopGradient`]) that applies it to eager [`Array`](crate::Array)s and traced values alike,
//! so the same code executes immediately or records into a program depending on the value it runs on. The operations
//! fall into two groups:
//!
//!   - **Linear Maps:** [`LinearCallOperation`] calls a residual-parameterized linear map together with its transpose,
//!     which lets differentiation rules keep a linear map and its handwritten transpose together in tangent programs
//!     (e.g., for shape-dependent maps such as dynamic reshapes).
//!   - **Gradient Barriers:** [`StopGradient`] and [`StopGradients`] return values unchanged while replacing their
//!     tangents with structural zeros, so that no derivative flows through them.
//!
//! Outside of differentiation, all of these operations are transparent. That is, interpretation and backend lowering
//! execute the forward map of a linear call when it has one and pass the inputs of a gradient barrier through
//! unchanged, while under differentiation a gradient barrier produces zero tangents. Functions with custom derivative
//! rules live in the [`custom_functions`](crate::operations::custom_functions) module, and the differentiation
//! transforms themselves live in the [`differentiation`](crate::differentiation) module.
//!
//! # Examples
//!
//! ```rust
//! # use ryft_core::{Array, ProgramError, StopGradient, differentiate_at};
//! # fn main() -> Result<(), ProgramError> {
//! // A gradient barrier treats its input as a constant, so the gradient of `x * stop_gradient(x)` is `x`.
//! let gradient = differentiate_at(Array::scalar(3.0f64)?).gradient(|x| x.clone() * x.stop_gradient().unwrap())?;
//! assert_eq!(gradient, Array::scalar(3.0f64)?);
//! # Ok(())
//! # }
//! ```

pub mod linear_call;
pub mod rematerialize;
pub mod stop_gradient;

pub use linear_call::{LINEAR_CALL_OPERATION_NAME, LinearCallOperation};
pub use rematerialize::{REMATERIALIZE_OPERATION_NAME, RematerializeOperation};
pub use stop_gradient::{STOP_GRADIENT_OPERATION_NAME, StopGradient, StopGradientOperation, StopGradients};
