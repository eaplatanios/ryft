//! The `cf` (i.e., control flow) dialect contains low-level (i.e., non-region based) control flow constructs.
//! These constructs generally represent control flow directly on SSA blocks of a control flow graph. For structured
//! control flow constructs like `if` statements and `for` loops refer to the [`scf`](crate::dialects::scf) dialect.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/)
//! for more information.
use ryft_xla_sys::bindings::mlirGetDialectHandle__cf__;

use crate::{DialectHandle, Error};

pub mod operations;

pub use operations::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `cf` [`Dialect`](crate::Dialect).
    pub fn cf() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__cf__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_cf_dialect() {
        let handle = DialectHandle::cf().unwrap();
        assert_eq!(handle.namespace().unwrap(), "cf");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "cf");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::cf().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
