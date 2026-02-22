//! The `func` dialect contains operations surrounding high order function abstractions,
//! such as functions, function calls, etc.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Func/)
//! for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__func__;

use crate::DialectHandle;

pub mod operations;

pub use operations::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `func` [`Dialect`](crate::Dialect).
    pub fn func() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__func__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_func_dialect() {
        let handle = DialectHandle::func();
        assert_eq!(handle.namespace().unwrap(), "func");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "func");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::func());
        assert_eq!(dialect_1, dialect_2);
    }
}
