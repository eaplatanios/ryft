//! The `sparse_tensor` dialect supports all the [`Attribute`](crate::Attribute)s, [`Type`](crate::Type)s,
//! [`Operation`](crate::Operation)s, and [`Pass`](crate::Pass)es that are required to make sparse tensor types
//! first class citizens within the MLIR compiler infrastructure. The dialect forms a bridge between high-level
//! operations on sparse tensors types and lower-level operations on the actual sparse storage schemes consisting
//! of positions, coordinates, and values. Lower-level support may consist of fully generated code or may be
//! provided by means of a small sparse runtime support library.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/)
//! for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__sparse_tensor__;

use crate::DialectHandle;

pub mod passes;

pub use passes::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `sparse_tensor` [`Dialect`](crate::Dialect).
    pub fn sparse_tensor() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__sparse_tensor__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_sparse_tensor_dialect() {
        let handle = DialectHandle::sparse_tensor();
        assert_eq!(handle.namespace().unwrap(), "sparse_tensor");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "sparse_tensor");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::sparse_tensor());
        assert_eq!(dialect_1, dialect_2);
    }
}
