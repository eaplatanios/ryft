//! Universal compile-time options consumed by the core jit pipeline.

use std::fmt::Debug;

use super::domain::CompilationDomain;

/// Universal compile-time options passed to
/// [`compile_and_execute_with_options`](super::compile_and_execute_with_options).
///
/// The core wrapper carries only fields that are universal across backends:
///
///   - [`Self::static_args_hash`] — opaque digest of any state the closure captures that
///     should partition the cache. Mixed into the [`FunctionFingerprint::Composite`] used as
///     the cache-key seed.
///   - [`Self::options`] — backend-specific options bag. Backends drive everything from this
///     field (mesh, donation, sharding overrides, ...) through their
///     [`CompilationDomain::Options`] type.
///
/// Backends define their own option types (for example XLA's `XlaOptions`) and the core
/// pipeline passes them through unchanged.
///
/// [`FunctionFingerprint::Composite`]: super::FunctionFingerprint::Composite
#[derive(Clone, Debug)]
pub struct CompilationOptions<E: CompilationDomain> {
    /// Opaque hash of any state captured by the closure passed to
    /// [`compile_and_execute_with_options`](super::compile_and_execute_with_options) that
    /// should partition the cache. Mixed into the call-site
    /// [`FunctionFingerprint`](super::FunctionFingerprint) so that repeat invocations at the
    /// same source line with different captured state still get distinct cache entries.
    /// Defaults to `0` (no contribution).
    ///
    /// JAX's `static_argnums` keys the cache on each static argument's value identity. Rust's
    /// closures capture state implicitly with no language-level "value identity"; the caller
    /// hashes the relevant state themselves and passes the digest here. Any stable hasher
    /// works.
    pub static_args_hash: u64,

    /// Backend-specific options bag. See [`CompilationDomain::Options`] for the contract.
    pub options: E::Options,
}

impl<E: CompilationDomain> CompilationOptions<E> {
    /// Creates a [`CompilationOptions`] with the supplied backend options and a zero
    /// `static_args_hash`.
    #[inline]
    pub fn new(options: E::Options) -> Self {
        Self { static_args_hash: 0, options }
    }

    /// Returns a new [`CompilationOptions`] with the supplied `static_args_hash`.
    #[inline]
    pub fn with_static_args_hash(mut self, static_args_hash: u64) -> Self {
        self.static_args_hash = static_args_hash;
        self
    }
}

impl<E: CompilationDomain> Default for CompilationOptions<E>
where
    E::Options: Default,
{
    #[inline]
    fn default() -> Self {
        Self { static_args_hash: 0, options: E::Options::default() }
    }
}
