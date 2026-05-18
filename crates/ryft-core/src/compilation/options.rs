//! Universal compile-time options consumed by the core jit pipeline.

use std::fmt::Debug;

use super::domain::CompilationDomain;

/// Universal compile-time options passed to
/// [`compile_with_options`](super::compile_with_options).
///
/// `CompilationOptions` is a thin wrapper around the backend-specific options bag. The
/// backend-agnostic parts of cache partitioning (call-site fingerprint, input tree structure)
/// are derived automatically by the core pipeline — see
/// [`compile_with_options`](super::compile_with_options) for details.
pub struct CompilationOptions<E: CompilationDomain> {
    /// Backend-specific options bag. See [`CompilationDomain::Options`] for the contract.
    pub options: E::Options,
}

impl<E: CompilationDomain> Clone for CompilationOptions<E>
where
    E::Options: Clone,
{
    fn clone(&self) -> Self {
        Self { options: self.options.clone() }
    }
}

impl<E: CompilationDomain> Debug for CompilationOptions<E>
where
    E::Options: Debug,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("CompilationOptions").field("options", &self.options).finish()
    }
}

impl<E: CompilationDomain> CompilationOptions<E> {
    /// Creates a [`CompilationOptions`] with the supplied backend options.
    #[inline]
    pub fn new(options: E::Options) -> Self {
        Self { options }
    }
}

impl<E: CompilationDomain> Default for CompilationOptions<E>
where
    E::Options: Default,
{
    #[inline]
    fn default() -> Self {
        Self { options: E::Options::default() }
    }
}
