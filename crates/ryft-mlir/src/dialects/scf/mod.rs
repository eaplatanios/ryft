//! The `scf` (i.e., structured control flow) dialect contains [`Operation`](crate::Operation)s that represent control
//! flow constructs such as `if` statements and `for` loops. Being structured means that the control flow has a
//! structure unlike, for example, assertions of go-to statements. Unstructured control flow operations are located
//! in the [`cf`](crate::dialects::cf) dialect.
//!
//! Originally, this dialect was developed as a common lowering stage for the [`affine`](crate::dialects::affine)
//! and [`linalg`](crate::dialects::linalg) dialects. Both convert to SCF loops instead of targeting branch-based
//! CFGs directly. Typically, `scf` is lowered to `cf` and then lowered to some final target like LLVM or SPIR-V.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/SCFDialect/)
//! for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__scf__;

use crate::DialectHandle;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `scf` [`Dialect`](crate::Dialect).
    pub fn scf() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__scf__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_scf_dialect() {
        let handle = DialectHandle::scf();
        assert_eq!(handle.namespace().unwrap(), "scf");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "scf");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::scf());
        assert_eq!(dialect_1, dialect_2);
    }
}
