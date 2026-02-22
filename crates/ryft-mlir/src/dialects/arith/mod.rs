//! The `arith` dialect is intended to hold basic integer and floating point [`Operation`](crate::Operation)s.
//! This includes unary, binary, and ternary arithmetic ops, bitwise and shift ops, cast ops, and compare ops.
//! Operations in this dialect also accept vectors and tensors of integers or floats. The dialect assumes integers
//! are represented by bitvectors with a two's complement representation. Unless otherwise stated, the operations
//! within this dialect propagate poison values (i.e., if any of its inputs are poison, then the output is poison).
//! Unless otherwise stated, operations applied to `vector` and `tensor` values propagate poison elementwise.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/ArithOps/) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__arith__;

use crate::DialectHandle;

pub mod operations;

pub use operations::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `arith` [`Dialect`](crate::Dialect).
    pub fn arith() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__arith__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_arith_dialect() {
        let handle = DialectHandle::arith();
        assert_eq!(handle.namespace().unwrap(), "arith");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "arith");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::arith());
        assert_eq!(dialect_1, dialect_2);
    }
}
