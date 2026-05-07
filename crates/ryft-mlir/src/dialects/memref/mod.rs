//! The `memref` dialect is designed for operations on memory references. It provides abstractions for working with
//! buffers and memory at a relatively low level.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/MemRef/) for more information.
use ryft_xla_sys::bindings::mlirGetDialectHandle__memref__;

use crate::{DialectHandle, Error};

pub mod operations;

pub use operations::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `memref` [`Dialect`](crate::Dialect).
    pub fn memref() -> Result<Self, Error> {
        unsafe {
            Self::from_c_api(mlirGetDialectHandle__memref__())
                .ok_or_else(|| Error::internal("expected non-null MLIR dialect handle"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_memref_dialect() {
        let handle = DialectHandle::memref().unwrap();
        assert_eq!(handle.namespace().unwrap(), "memref");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "memref");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::memref().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
