//! The `linalg` dialect is designed for structured linear algebra operations and serves as a high-level abstraction
//! for expressing computations on dense and sparse arrays.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Linalg/) for more information.
use ryft_xla_sys::bindings::mlirGetDialectHandle__linalg__;

use crate::{DialectHandle, Error};

pub mod attributes;
pub mod operations;
pub mod passes;

pub use attributes::*;
pub use operations::*;
pub use passes::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `linalg` [`Dialect`](crate::Dialect).
    pub fn linalg() -> Result<Self, Error> {
        unsafe {
            Self::from_c_api(mlirGetDialectHandle__linalg__())
                .ok_or_else(|| Error::internal("expected non-null MLIR dialect handle"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_linalg_dialect() {
        let handle = DialectHandle::linalg().unwrap();
        assert_eq!(handle.namespace().unwrap(), "linalg");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "linalg");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::linalg().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
