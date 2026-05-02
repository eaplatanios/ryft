//! The `emitc` dialect models C and C++ source-level constructs in MLIR so other dialects can lower to code that can
//! be emitted by MLIR's C++ emitter.
//!
//! Emit-C operations cover C/C++ expressions, calls, declarations, functions, globals, loops, conditionals, fields,
//! and textual escape hatches. The dialect also defines C/C++-oriented types such as arrays, lvalues, pointers, and
//! platform-sized integer types.
//!
//! Refer to the [official MLIR Emit-C documentation](https://mlir.llvm.org/docs/Dialects/emitc/) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__emitc__;

use crate::DialectHandle;

pub mod attributes;
pub mod operations;
pub mod types;

pub use attributes::*;
pub use operations::*;
pub use types::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `emitc` [`Dialect`](crate::Dialect).
    pub fn emit_c() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__emitc__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_emit_c_dialect() {
        let handle = DialectHandle::emit_c();
        assert_eq!(handle.namespace().unwrap(), "emitc");

        // Check that registration works both in the context and in a registry.
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "emitc");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::emit_c());
        assert_eq!(dialect_1, dialect_2);
    }
}
