//! The StableHLO dialect provides an [`Operation`](crate::Operation) set for high-level operations (HLO) in machine
//! learning (ML) models. StableHLO works as a portability layer between different ML frameworks and ML compilers:
//! ML frameworks that produce StableHLO programs are compatible with ML compilers that consume StableHLO programs.
//!
//! Refer to the [official StableHLO documentation](https://openxla.org/stablehlo/spec) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__stablehlo__;

use crate::DialectHandle;

pub mod attributes;
pub mod operations;
pub mod passes;
pub mod types;

pub use attributes::*;
#[allow(deprecated)]
pub use operations::*;
pub use passes::*;
pub use types::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the StableHLO [`Dialect`](crate::Dialect).
    pub fn stable_hlo() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__stablehlo__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_stable_hlo_dialect() {
        let handle = DialectHandle::stable_hlo();
        assert_eq!(handle.namespace().unwrap(), "stablehlo");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "stablehlo");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::stable_hlo());
        assert_eq!(dialect_1, dialect_2);
    }
}
