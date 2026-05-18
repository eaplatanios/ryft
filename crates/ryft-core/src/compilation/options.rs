//! Universal compile-time options consumed by the core jit pipeline.

use std::fmt::Debug;

use super::domain::CompilationDomain;

/// Universal compile-time options passed to
/// [`compile_and_execute_with_options`](super::compile_and_execute_with_options).
///
/// `CompilationOptions` carries the truly cross-cutting fields that apply to every backend
/// plus a backend-specific options bag:
///
///   - [`Self::static_args_hash`] — opaque digest of any state the closure captures that
///     should partition the cache. Typically populated by
///     [`compile_and_execute_with_statics`](super::compile_and_execute_with_statics) from a
///     typed `static_args: S` parameter; callers using
///     [`compile_and_execute_with_options`](super::compile_and_execute_with_options) directly
///     can set it themselves.
///   - [`Self::options`] — backend-specific options bag (e.g. XLA's mesh, sharding overrides,
///     and buffer donation flags).
pub struct CompilationOptions<E: CompilationDomain> {
    // TODO(eaplatanios): This seems very brittle. Can we do something better here about making captured state part
    //  of the compilation key? This feels prone to mistakes and we should ideally have a way of not depending on user
    //  mistakes for this sort of thing but rather handling it correctly internally in our library.
    /// Opaque hash of any state captured by the closure that should partition the cache.
    /// Mixed into the call-site
    /// [`FunctionFingerprint`](super::FunctionFingerprint) so that repeat invocations at the
    /// same source line with different captured state still get distinct cache entries.
    /// Defaults to `0` (no contribution).
    ///
    /// Most callers populate this indirectly via
    /// [`compile_and_execute_with_statics`](super::compile_and_execute_with_statics), which
    /// auto-hashes a typed `static_args: S` parameter.
    pub static_args_hash: u64,

    /// Backend-specific options bag. See [`CompilationDomain::Options`] for the contract.
    pub options: E::Options,
}

impl<E: CompilationDomain> Clone for CompilationOptions<E>
where
    E::Options: Clone,
{
    fn clone(&self) -> Self {
        Self { static_args_hash: self.static_args_hash, options: self.options.clone() }
    }
}

impl<E: CompilationDomain> Debug for CompilationOptions<E>
where
    E::Options: Debug,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompilationOptions")
            .field("static_args_hash", &self.static_args_hash)
            .field("options", &self.options)
            .finish()
    }
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
