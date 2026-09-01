//! The Complex dialect provides operations and attributes for complex-number arithmetic.
//!
//! Refer to the [official MLIR Complex dialect documentation](https://mlir.llvm.org/docs/Dialects/Complex/) for more
//! information.

use ryft_xla_sys::mlir::dialects::complex::mlirGetDialectHandle__complex__;

use crate::{DialectHandle, Error};

pub mod attributes;

pub use attributes::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the Complex [`Dialect`](crate::Dialect).
    pub fn complex() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__complex__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_complex_dialect() {
        let handle = DialectHandle::complex().unwrap();
        assert_eq!(handle.namespace().unwrap(), "complex");

        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "complex");
        let dialect_2 = context.load_dialect(DialectHandle::complex().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
