//! The Math dialect provides mathematical operations over integer and floating-point values.
//!
//! Refer to the [official MLIR Math dialect documentation](https://mlir.llvm.org/docs/Dialects/MathOps/) for more
//! information.

pub mod operations;
pub mod passes;

pub use operations::*;
pub use passes::*;

use ryft_xla_sys::bindings::mlirGetDialectHandle__math__;

use crate::{DialectHandle, Error};

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the Math [`Dialect`](crate::Dialect).
    pub fn math() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__math__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_math_dialect() {
        let handle = DialectHandle::math().unwrap();
        assert_eq!(handle.namespace().unwrap(), "math");

        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "math");
        let dialect_2 = context.load_dialect(DialectHandle::math().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
