//! The versioned HLO (i.e., vHLO) dialect is used for serialization and stability. It provides a snapshot of the
//! StableHLO dialect at a given point in time by versioning individual program elements.
//!
//! Refer to the [official StableHLO documentation](https://openxla.org/stablehlo/vhlo) for more information.
use ryft_xla_sys::bindings::mlirGetDialectHandle__vhlo__;

use crate::{DialectHandle, Error};

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the versioned HLO [`Dialect`](crate::Dialect).
    pub fn versioned_hlo() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__vhlo__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_versioned_hlo_dialect() {
        let handle = DialectHandle::versioned_hlo().unwrap();
        assert_eq!(handle.namespace().unwrap(), "vhlo");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "vhlo");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::versioned_hlo().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
