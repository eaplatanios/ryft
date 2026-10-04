//! The UB dialect models undefined behavior through poison values and unreachable control flow.
//!
//! Refer to the [official MLIR UB dialect documentation](https://mlir.llvm.org/docs/Dialects/UB/) for more
//! information.

use ryft_xla_sys::mlir::dialects::ub::mlirGetDialectHandle__ub__;

use crate::{DialectHandle, Error};

pub mod attributes;
pub mod operations;

pub use attributes::*;
pub use operations::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the UB [`Dialect`](crate::Dialect).
    pub fn ub() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__ub__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_ub_dialect() {
        let handle = DialectHandle::ub().unwrap();
        assert_eq!(handle.namespace().unwrap(), "ub");

        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "ub");
        let dialect_2 = context.load_dialect(DialectHandle::ub().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
