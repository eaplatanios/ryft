//! The [`CompilationDomain`] trait.

use std::fmt::Debug;

use crate::parameters::Parameterized;
use crate::tracing::TracingError;
use crate::tracing::domains::TracingDomain;
use crate::tracing::programs::Program;
use crate::types::Typed;

use super::context::CompilationContext;
use super::fingerprint::FunctionFingerprint;

/// Backend interface for the backend-agnostic compilation pipeline.
///
/// A [`CompilationDomain`] is a [`TracingDomain`] that can additionally lower a traced
/// [`Program`] into a backend-specific compiled artifact and execute it against runtime values.
/// The trait is intentionally minimal: everything backend-specific (mesh, device placement,
/// donation flags, layout overrides, platform identity for disk caching, ...) flows through
/// the [`Self::Options`] associated type and is interpreted entirely by the engine.
///
/// # Composition with transforms
///
/// Because a [`CompilationDomain`] *is a* [`TracingDomain`], the existing `ryft-core` transforms
/// (`grad`, `value_and_grad`, `jvp`, `vjp`, `linearize`, `jacrev`, `jacfwd`, `hessian`, `vmap`)
/// compose naturally inside the closure passed to
/// [`compile_and_execute_with_options`](super::compile_and_execute_with_options). The transform
/// is traced as part of the staged program, so the resulting executable computes the
/// transformed function directly.
pub trait CompilationDomain: TracingDomain
where
    Self::Value: Typed<Self::Type>,
{
    /// Backend's compiled artifact. Carries everything needed to execute, baked in by the
    /// engine's [`Self::compile`] step (output types, donation flags, expected layouts,
    /// mesh, etc.). `Clone` is required so the in-memory cache can hand the same artifact
    /// to multiple [`CompiledFunction`](super::CompiledFunction) handles.
    type CompiledProgram: Clone;

    /// Backend-specific compile options. For XLA: mesh, donation, sharding overrides. For
    /// backends without per-call options, set to `()`. The cache uses the engine's
    /// [`Self::fingerprint`] method to derive its key, so [`Self::Options`] does not need
    /// to implement [`Hash`].
    type Options: Clone + Debug;

    /// Backend-specific error type. Must absorb [`TracingError`] so that errors raised inside
    /// the trace path can flow through the engine's own error channel.
    type Error: std::error::Error + From<TracingError>;

    /// Computes the cache key for a given compilation. The engine has full control over what
    /// goes in: the call-site [`FunctionFingerprint`], the input type signatures, the
    /// [`Self::Options`], and any backend-specific state (compile options, platform identity
    /// for cross-machine disk caches, etc.). The cache treats the returned `u64` opaquely.
    fn fingerprint(&self, function: &FunctionFingerprint, inputs: &[Self::Type], options: &Self::Options) -> u64;

    /// Lowers a traced [`Program`] into a [`Self::CompiledProgram`]. The engine reads its own
    /// backend-specific per-call state (donation, mesh, shardings, etc.) out of `options` and
    /// bakes whatever [`Self::execute`] needs into the artifact.
    fn compile<Input, Output>(
        &self,
        program: &Program<Self::Type, Self::Value, Self::OperationCarrier, Input, Output>,
        options: &Self::Options,
    ) -> Result<Self::CompiledProgram, Self::Error>
    where
        Input: Parameterized<Self::Value>,
        Output: Parameterized<Self::Value>;

    /// Executes a compiled program. Every piece of per-call state is already in the artifact;
    /// the caller hands over runtime values in flat input order and gets runtime values back
    /// in flat output order.
    fn execute(
        &self,
        program: &Self::CompiledProgram,
        inputs: Vec<Self::Value>,
    ) -> Result<Vec<Self::Value>, Self::Error>;

    /// Serializes a compiled program for the disk-cache tier. Backends that don't support
    /// persistent caching return an "unsupported" error variant; the cache treats any error
    /// as a signal to skip the disk tier for that entry.
    fn serialize_program(&self, program: &Self::CompiledProgram) -> Result<Vec<u8>, Self::Error>;

    /// Deserializes a compiled program previously emitted by [`Self::serialize_program`].
    /// Backends should validate platform compatibility (e.g. PJRT platform name + version) and
    /// return an error on mismatch; the cache treats any error as a signal to skip the disk
    /// tier for that entry.
    fn deserialize_program(&self, bytes: &[u8]) -> Result<Self::CompiledProgram, Self::Error>;

    /// Returns the [`CompilationContext`] this engine uses to memoize compiled programs, if any.
    ///
    /// The default returns `None`, in which case the core entry points
    /// ([`compile_and_execute_with_options`](super::compile_and_execute_with_options),
    /// [`compile_and_execute`](super::compile_and_execute)) compile fresh on every call. Engines
    /// that want caching override this to return a reference to their internal cache — typically
    /// stored as an [`Arc<CompilationContext<Self>>`](std::sync::Arc) field so engine clones
    /// share the same cache.
    fn cache(&self) -> Option<&CompilationContext<Self>> {
        None
    }
}
