//! The `nvgpu` dialect provides NVIDIA-GPU-specific operations and types that bridge target-agnostic GPU/vector IR
//! and the lower-level NVVM dialect. It models PTX-specific instructions while preserving MLIR memref and vector
//! abstractions for memory and register operands.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/NVGPU/) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__nvgpu__;

use crate::{DialectHandle, Error};

pub mod attributes;
pub mod operations;
pub mod types;

pub use attributes::*;
pub use operations::*;
pub use types::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `nvgpu` [`Dialect`](crate::Dialect).
    pub fn nvgpu() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__nvgpu__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_nvgpu_dialect() {
        let handle = DialectHandle::nvgpu().unwrap();
        assert_eq!(handle.namespace().unwrap(), "nvgpu");

        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "nvgpu");

        let dialect_2 = context.load_dialect(DialectHandle::nvgpu().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
