//! The `shape` dialect is designed for shape computation and manipulation. It provides abstractions for reasoning
//! about and computing the shapes of tensors and arrays, especially when those shapes are not statically known.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/ShapeDialect/) for more information.
use ryft_xla_sys::bindings::mlirGetDialectHandle__shape__;

use crate::{DialectHandle, Error};

pub mod operations;
pub mod types;

pub use operations::*;
pub use types::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `shape` [`Dialect`](crate::Dialect).
    pub fn shape() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__shape__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_shape_dialect() {
        let handle = DialectHandle::shape().unwrap();
        assert_eq!(handle.namespace().unwrap(), "shape");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "shape");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::shape().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
