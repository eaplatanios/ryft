//! The Bufferization dialect provides operations that bridge tensor values and memory buffers.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/BufferizationOps/)
//! for more information.

use ryft_xla_sys::mlir::dialects::bufferization::mlirGetDialectHandle__bufferization__;

use crate::{DialectHandle, Error};

pub mod operations;

pub use operations::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the Bufferization [`Dialect`](crate::Dialect).
    pub fn bufferization() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__bufferization__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_bufferization_dialect() {
        let handle = DialectHandle::bufferization().unwrap();
        assert_eq!(handle.namespace().unwrap(), "bufferization");

        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "bufferization");
        let dialect_2 = context.load_dialect(DialectHandle::bufferization().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
