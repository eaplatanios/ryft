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
pub struct CompilationOptions<D: CompilationDomain> {
    /// Backend-specific options bag. See [`CompilationDomain::Options`] for the contract.
    pub options: D::Options,
}

impl<D: CompilationDomain<Options: Clone>> Clone for CompilationOptions<D> {
    fn clone(&self) -> Self {
        Self { options: self.options.clone() }
    }
}

impl<D: CompilationDomain<Options: Debug>> Debug for CompilationOptions<D> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("CompilationOptions").field("options", &self.options).finish()
    }
}

impl<D: CompilationDomain> CompilationOptions<D> {
    /// Creates a [`CompilationOptions`] with the supplied backend options.
    #[inline]
    pub fn new(options: D::Options) -> Self {
        Self { options }
    }
}

impl<D: CompilationDomain<Options: Default>> Default for CompilationOptions<D> {
    #[inline]
    fn default() -> Self {
        Self { options: D::Options::default() }
    }
}
