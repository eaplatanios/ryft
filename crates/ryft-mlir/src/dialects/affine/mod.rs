//! The `affine` dialect provides a powerful abstraction for affine [`Operation`](crate::Operation)s and analyses.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Affine/) for more information.
use ryft_xla_sys::mlir::dialects::affine::mlirGetDialectHandle__affine__;

use crate::{DialectHandle, Error};

pub mod affine_expressions;
pub mod affine_maps;
pub mod integer_sets;
pub mod operations;

pub use affine_expressions::*;
pub use affine_maps::*;
pub use integer_sets::*;
pub use operations::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `affine` [`Dialect`](crate::Dialect).
    pub fn affine() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__affine__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_affine_dialect() {
        let handle = DialectHandle::affine().unwrap();
        assert_eq!(handle.namespace().unwrap(), "affine");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "affine");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::affine().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
