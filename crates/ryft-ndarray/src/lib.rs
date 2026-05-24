pub mod arrays;
#[cfg(feature = "benchmarking")]
pub mod benchmarking;
pub mod domains;
pub mod operations;

pub use arrays::{Array, ArrayError, NdArrayElement};
pub use domains::NdArrayDomain;
pub use operations::{LinearNdarrayOperation, NdarrayOperation};
