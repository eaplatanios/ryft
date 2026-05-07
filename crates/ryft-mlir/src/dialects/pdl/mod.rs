//! The `pdl` dialect presents a high level abstraction for the rewrite pattern infrastructure available in MLIR.
//! This abstraction allows for representing patterns transforming MLIR, as MLIR. This allows for applying all the
//! benefits that the general MLIR infrastructure provides, to the infrastructure itself. This means that pattern
//! matching can be more easily verified for correctness, targeted by frontends, and optimized.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/) for more information.
use ryft_xla_sys::bindings::mlirGetDialectHandle__pdl__;

use crate::{DialectHandle, Error};

pub mod operations;
pub mod types;

pub use operations::*;
pub use types::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `pdl` [`Dialect`](crate::Dialect).
    pub fn pdl() -> Result<Self, Error> {
        unsafe {
            Self::from_c_api(mlirGetDialectHandle__pdl__())
                .ok_or_else(|| Error::internal("expected non-null MLIR dialect handle"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_pdl_dialect() {
        let handle = DialectHandle::pdl().unwrap();
        assert_eq!(handle.namespace().unwrap(), "pdl");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "pdl");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::pdl().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
