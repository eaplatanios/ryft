//! The `chlo` dialect contains [`Operation`](crate::Operation)s that align closely with the API surface area of the
//! `XlaBuilder` C++ API, where such operations have semantics that go beyond what exists in lower level dialects such
//! as [StableHLO](crate::dialects::stable_hlo). Essentially, whenever the client library uses syntactic sugar or
//! composition of multiple ops for an API call, this dialect tries to model the API call and provide conversion
//! patterns to fully materialize into lower level dialects.
//!
//! Refer to the [official StableHLO documentation](https://openxla.org/stablehlo/generated/chlo) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__chlo__;

use crate::DialectHandle;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `chlo` [`Dialect`](crate::Dialect).
    pub fn chlo() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__chlo__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_chlo_dialect() {
        let handle = DialectHandle::chlo();
        assert_eq!(handle.namespace().unwrap(), "chlo");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "chlo");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::chlo());
        assert_eq!(dialect_1, dialect_2);
    }
}
