//! The `quant` dialect offers a framework for defining and manipulating quantized values. Central to this framework
//! is the `!quant.uniform` data type, which is used to represent quantized values. This dialect also provides a suite
//! of [`Operation`](crate::Operation)s to handle and convert quantized values between their original floating-point
//! representations and the optimized, lower bit-width integer representations. The `quant` dialect is instrumented with
//! transformation passes to lower these operations into other core MLIR dialects, while also flattening all occurrences
//! of quantized types into their integer counterparts.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/QuantDialect/) for more information.
use ryft_xla_sys::bindings::mlirGetDialectHandle__quant__;

use crate::{DialectHandle, Error};

pub mod operations;
pub mod types;

pub use operations::*;
pub use types::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `quant` [`Dialect`](crate::Dialect).
    pub fn quant() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__quant__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_quant_dialect() {
        let handle = DialectHandle::quant().unwrap();
        assert_eq!(handle.namespace().unwrap(), "quant");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "quant");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::quant().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
