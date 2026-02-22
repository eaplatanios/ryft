//! The `linalg` dialect is designed for structured linear algebra operations and serves as a high-level abstraction
//! for expressing computations on dense and sparse arrays.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Linalg/) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__linalg__;

use crate::DialectHandle;

pub mod passes;

pub use passes::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `linalg` [`Dialect`](crate::Dialect).
    pub fn linalg() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__linalg__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_linalg_dialect() {
        let handle = DialectHandle::linalg();
        assert_eq!(handle.namespace().unwrap(), "linalg");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "linalg");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::linalg());
        assert_eq!(dialect_1, dialect_2);
    }
}
