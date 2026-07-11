pub mod and;
pub mod not;
pub mod or;
pub mod xor;

pub use and::{AND_OPERATION_NAME, And, AndOperation};
pub use not::{NOT_OPERATION_NAME, Not, NotOperation};
pub use or::{OR_OPERATION_NAME, Or, OrOperation};
pub use xor::{XOR_OPERATION_NAME, Xor, XorOperation};
