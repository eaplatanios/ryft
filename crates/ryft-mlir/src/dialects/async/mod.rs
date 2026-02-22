//! The `async` dialect contains [`Operation`](crate::Operation)s for modeling asynchronous execution.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/AsyncDialect/) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__async__;

use crate::DialectHandle;

pub mod passes;

pub use passes::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `async` [`Dialect`](crate::Dialect).
    pub fn r#async() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__async__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_async_dialect() {
        let handle = DialectHandle::r#async();
        assert_eq!(handle.namespace().unwrap(), "async");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "async");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::r#async());
        assert_eq!(dialect_1, dialect_2);
    }
}
