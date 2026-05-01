//! The Mosaic GPU dialect provides JAX Mosaic operations and layout metadata for GPU kernels.
//!
//! The wrappers in this module target the Mosaic GPU dialect version pinned by the `ryft-xla-sys` JAX dependency.
//! Refer to the [JAX Pallas Mosaic GPU documentation](https://docs.jax.dev/en/latest/pallas/gpu/index.html) for more
//! information.

use ryft_xla_sys::mlir::dialects::mosaic::gpu::mlirGetDialectHandle__mosaic_gpu__;

use crate::DialectHandle;

pub mod attributes;
pub mod operations;
pub mod types;

pub use attributes::*;
pub use operations::*;
pub use types::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the Mosaic GPU [`Dialect`](crate::Dialect).
    pub fn mosaic_gpu() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__mosaic_gpu__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_mosaic_gpu_dialect() {
        let handle = DialectHandle::mosaic_gpu();
        assert_eq!(handle.namespace().unwrap(), "mosaic_gpu");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "mosaic_gpu");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::mosaic_gpu());
        assert_eq!(dialect_1, dialect_2);
    }
}
