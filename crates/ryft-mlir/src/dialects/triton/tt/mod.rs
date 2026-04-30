//! The Triton `tt` dialect provides the core Triton tensor IR used before target-specific Triton lowering.
//!
//! Refer to the [official Triton dialect documentation](https://triton-lang.org/main/dialects/TritonDialect.html)
//! for more information.

use ryft_xla_sys::mlir::dialects::triton::tt::mlirGetDialectHandle__tt__;

use crate::DialectHandle;

pub mod attributes;
pub mod operations;
pub mod types;

pub use attributes::*;
pub use operations::*;
pub use types::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the Triton `tt` [`Dialect`](crate::Dialect).
    pub fn triton_tt() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__tt__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_triton_tt_dialect() {
        let handle = DialectHandle::triton_tt();
        assert_eq!(handle.namespace().unwrap(), "tt");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "tt");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::triton_tt());
        assert_eq!(dialect_1, dialect_2);
    }
}
