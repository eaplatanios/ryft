//! The Mosaic TPU dialect provides JAX Mosaic operations, attributes, and types for TPU kernels.
//!
//! The wrappers in this module target the Mosaic TPU dialect version pinned by the `ryft-xla-sys` JAX dependency.
//! Refer to the [JAX Mosaic TPU API documentation](https://docs.jax.dev/en/latest/jax.experimental.pallas.tpu.html)
//! for more information.
use ryft_xla_sys::mlir::dialects::mosaic::tpu::mlirGetDialectHandle__tpu__;

use crate::{DialectHandle, Error};

pub mod attributes;
pub mod operations;
pub mod types;

pub use attributes::*;
pub use operations::*;
pub use types::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the Mosaic TPU [`Dialect`](crate::Dialect).
    pub fn mosaic_tpu() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__tpu__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_mosaic_tpu_dialect() {
        let handle = DialectHandle::mosaic_tpu().unwrap();
        assert_eq!(handle.namespace().unwrap(), "tpu");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "tpu");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::mosaic_tpu().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
