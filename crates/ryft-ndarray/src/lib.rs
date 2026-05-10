pub mod arrays;
#[cfg(feature = "benchmarking")]
pub mod benchmarking;
pub mod domains;
pub mod jacobians;
pub mod operations;

pub use arrays::{Array, ArrayError, NdArrayElement};
pub use domains::NdArrayDomain;
pub use jacobians::DenseJacobianNdArrayExt;
pub use operations::{
    LinearNdarrayElementOperation, LinearNdarrayOperation, NdarrayElementOperation, NdarrayOperation,
};
