//! The `llvm` dialect maps [LLVM IR](https://llvm.org/docs/LangRef.html) to MLIR by defining the corresponding
//! [`Operation`](crate::Operation)s and [`Type`](crate::Type)s. LLVM IR metadata is usually represented as MLIR
//! attributes, which offer additional structure verification.
//!
//! We use "LLVM IR" to designate the [intermediate representation of LLVM](https://llvm.org/docs/LangRef.html) and
//! "LLVM dialect" or "LLVM IR dialect" to refer to this MLIR dialect.
//!
//! Unless explicitly stated otherwise, the semantics of the LLVM dialect operations must correspond to the semantics
//! of LLVM IR instructions and any divergence is considered a bug. The dialect also contains auxiliary operations that
//! smoothen the differences in the IR structure (e.g., MLIR does not have phi operations and LLVM IR does not have a
//! constant operation). These auxiliary operations are systematically prefixed with `mlir` (e.g., `llvm.mlir.constant`
//! where `llvm.` is the dialect namespace prefix).
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/LLVM/) for more information.

use ryft_xla_sys::bindings::{mlirGetDialectHandle__llvm__, mlirRegisterAllLLVMTranslations};

use crate::{Context, DialectHandle};

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `llvm` [`Dialect`](crate::Dialect).
    pub fn llvm() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__llvm__()).unwrap() }
    }
}

impl<'t> Context<'t> {
    /// Registers all translations to LLVM IR for dialects that can support it to this [`Context`].
    pub fn register_all_llvm_translations(&self) {
        unsafe { mlirRegisterAllLLVMTranslations(*self.handle.borrow_mut()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_llvm_dialect() {
        let handle = DialectHandle::llvm();
        assert_eq!(handle.namespace().unwrap(), "llvm");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "llvm");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::llvm());
        assert_eq!(dialect_1, dialect_2);
    }

    #[test]
    fn test_register_llvm_translations_twice() {
        // We intentionally try to register multiple times to ensure that the operation is idempotent.
        let context = Context::new();
        for _ in 0..100 {
            context.register_all_llvm_translations();
        }
    }
}
