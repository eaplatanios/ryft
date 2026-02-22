//! The Shardy dialect is meant to provide extensive user control over tensor partitioning and debuggability features.
//! It includes an axis-based sharding representation, a set of compiler APIs, functionality for sharding propagation,
//! and plans for an SPMD partitioner.
//!
//! Refer to the [official Shardy documentation](https://openxla.org/shardy) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__sdy__;

use crate::DialectHandle;

pub mod attributes;
pub mod operations;
pub mod passes;

pub use attributes::*;
pub use operations::*;
pub use passes::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the Shardy [`Dialect`](crate::Dialect).
    pub fn shardy() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__sdy__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_shardy_dialect() {
        let handle = DialectHandle::shardy();
        assert_eq!(handle.namespace().unwrap(), "sdy");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "sdy");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::shardy());
        assert_eq!(dialect_1, dialect_2);
    }
}
