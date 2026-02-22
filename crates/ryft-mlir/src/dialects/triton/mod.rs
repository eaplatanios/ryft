// TODO(eaplatanios): Clean this up and make sure it is correct.

//! Triton TTIR front-end support.
//!
//! This module exposes the Triton front-end dialect namespace (`tt`) needed to construct TTIR programs that can be
//! compiled by Triton-based PJRT pipelines.
//!
//! The currently exposed C API in `ryft-xla-sys` does not provide dedicated typed helpers for Triton dialect
//! entities, so this module uses dialect-namespace loading plus parsing/opaque wrappers for TTIR-facing types and
//! attributes.
use crate::{Context, Dialect};

pub mod attributes;
pub mod operations;
pub mod types;

pub use attributes::*;
pub use operations::*;
pub use types::*;

// TODO(eaplatanios): Triton dialect support was almost entirely coded up by Codex. It needs a fair amount of work.

/// Enumerates Triton front-end dialect namespaces supported by this module.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TritonDialect {
    /// The core Triton dialect representing TTIR operations (`tt.*`).
    Triton,
}

impl TritonDialect {
    /// All Triton dialect namespaces known to this module.
    pub const ALL: [Self; 1] = [Self::Triton];

    /// Returns the namespace of this [`TritonDialect`].
    pub const fn namespace(&self) -> &'static str {
        match self {
            Self::Triton => "tt",
        }
    }

    /// Returns the [`TritonDialect`] that corresponds to the provided `namespace`.
    pub fn from_namespace<S: AsRef<str>>(namespace: S) -> Option<Self> {
        match namespace.as_ref() {
            "tt" => Some(Self::Triton),
            _ => None,
        }
    }
}

impl<'t> Context<'t> {
    /// Attempts to load the provided Triton [`Dialect`](crate::Dialect) in this [`Context`].
    ///
    /// This relies on [`Context::load_dialect_by_name`] because the currently exposed C API in `ryft-xla-sys` does
    /// not provide dedicated `mlirGetDialectHandle__*` entry points for Triton dialect namespaces.
    ///
    /// Refer to [`Context::load_dialect_by_name`] for registration semantics.
    pub fn load_triton_dialect<'c>(&self, dialect: TritonDialect) -> Option<Dialect<'c, 't>>
    where
        Self: 'c,
    {
        self.load_dialect_by_name(dialect.namespace())
    }

    /// Attempts to load all known Triton front-end dialect namespaces in this [`Context`] and returns those that are
    /// available in the current build/runtime.
    pub fn load_all_triton_dialects<'c>(&self) -> Vec<Dialect<'c, 't>>
    where
        Self: 'c,
    {
        TritonDialect::ALL.into_iter().filter_map(|dialect| self.load_triton_dialect(dialect)).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;

    use super::*;
    use crate::{DialectRegistry, Threading};

    #[test]
    fn test_triton_dialect_namespace_mapping() {
        assert_eq!(TritonDialect::Triton.namespace(), "tt");

        assert_eq!(TritonDialect::from_namespace("tt"), Some(TritonDialect::Triton));
        assert_eq!(TritonDialect::from_namespace("stablehlo"), None);
    }

    #[test]
    fn test_load_triton_dialects() {
        let registry = DialectRegistry::new_with_all_built_in_dialects();
        let context = Context::new_with_registry(&registry, Threading::Disabled);

        for dialect in TritonDialect::ALL {
            if let Some(loaded) = context.load_triton_dialect(dialect) {
                assert_eq!(loaded.namespace().unwrap(), dialect.namespace());
                assert_eq!(context.load_triton_dialect(dialect), Some(loaded));
            }
        }
    }

    #[test]
    fn test_load_all_triton_dialects() {
        let registry = DialectRegistry::new_with_all_built_in_dialects();
        let context = Context::new_with_registry(&registry, Threading::Disabled);

        let loaded_dialects = context.load_all_triton_dialects();
        let loaded_namespaces =
            loaded_dialects.iter().map(|dialect| dialect.namespace().unwrap()).collect::<HashSet<_>>();

        assert_eq!(loaded_namespaces.len(), loaded_dialects.len());
        assert!(loaded_namespaces.iter().all(|namespace| TritonDialect::from_namespace(namespace).is_some()));
    }
}
