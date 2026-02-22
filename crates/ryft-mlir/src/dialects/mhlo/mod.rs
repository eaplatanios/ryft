//! The `mhlo` dialect represents high-level operations from the XLA compiler. It is essentially MLIR's representation
//! of the HLO intermediate representation used in TensorFlow's XLA (Accelerated Linear Algebra) compiler. It should
//! generally be considered internal and frameworks building on top of the XLA compiler should use the
//! [StableHLO](crate::dialects::stable_hlo) dialect instead.
//!
//! Refer to the [official XLA documentation](https://openxla.org/xla) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__mhlo__;

use crate::DialectHandle;

pub mod passes;

pub use passes::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the MHLO [`Dialect`](crate::Dialect).
    pub fn mhlo() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__mhlo__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_mhlo_dialect() {
        let handle = DialectHandle::mhlo();
        assert_eq!(handle.namespace().unwrap(), "mhlo");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "mhlo");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::mhlo());
        assert_eq!(dialect_1, dialect_2);
    }
}
