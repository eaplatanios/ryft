//! The `transform` dialect represents compiler transformations as MLIR IR, allowing transformation scripts to
//! identify payload IR operations, values, and parameters and apply fine-grained rewrites under explicit control.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Transform/) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__transform__;

use crate::DialectHandle;

pub mod attributes;
pub mod operations;
pub mod types;

pub use attributes::*;
pub use operations::*;
pub use types::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `transform` [`Dialect`](crate::Dialect).
    pub fn transform() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__transform__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_transform_dialect() {
        let handle = DialectHandle::transform();
        assert_eq!(handle.namespace().unwrap(), "transform");

        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "transform");

        let dialect_2 = context.load_dialect(DialectHandle::transform());
        assert_eq!(dialect_1, dialect_2);
    }
}
