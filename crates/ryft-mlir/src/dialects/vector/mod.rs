//! The Vector dialect provides operations for manipulating multidimensional SIMD vectors.
//!
//! Refer to the [official MLIR Vector dialect documentation](https://mlir.llvm.org/docs/Dialects/Vector/)
//! for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__vector__;

use crate::{DialectHandle, Error};

pub mod attributes;
pub mod operations;
pub mod passes;

pub use attributes::*;
pub use operations::*;
pub use passes::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the Vector [`Dialect`](crate::Dialect).
    pub fn vector() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__vector__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_dialect_handle_vector() {
        let handle = DialectHandle::vector().unwrap();
        assert_eq!(handle.namespace().unwrap(), "vector");

        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "vector");
        let dialect_2 = context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
