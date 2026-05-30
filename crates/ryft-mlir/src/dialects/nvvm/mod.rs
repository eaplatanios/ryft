//! The `nvvm` dialect provides support for NVIDIA PTX and NVVM IR operations and types in MLIR.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/NVVMDialect/) for more information.

pub mod attributes;
pub mod operations;

pub use attributes::*;
pub use operations::*;

use ryft_xla_sys::bindings::mlirGetDialectHandle__nvvm__;

use crate::{DialectHandle, Error};

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `nvvm` [`Dialect`](crate::Dialect).
    pub fn nvvm() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__nvvm__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Context;

    use super::*;

    #[test]
    fn test_nvvm_dialect() {
        let handle = DialectHandle::nvvm().unwrap();
        assert_eq!(handle.namespace().unwrap(), "nvvm");

        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        let dialect_2 = context.load_dialect(DialectHandle::nvvm().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
