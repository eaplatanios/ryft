//! XLA domains, sessions, compilation, and execution through the shared domain interfaces.

use std::sync::{Arc, LazyLock};

use ryft_core::{
    ArrayIrType, CompilationContext, DeviceMesh, Domain as CoreDomain, DomainTracer, SpecializationCacheStatistics,
};
use ryft_pjrt::Client;
use ryft_pjrt::protos::CompilationOptions;

use crate::domains::effects::EffectScope;
use crate::experimental::ops::{Constant, Operation};
use crate::{ArrayError, Error, Value};

pub mod analysis;
pub mod compilation;
pub mod eager;
pub mod effects;
pub mod execution;
pub mod options;
pub mod persistence;
pub mod programs;
pub mod sessions;
pub mod target;

pub use analysis::{AnalysisValue, CompilationAnalysis, MemoryAnalysis, OptimizedProgram};
pub use compilation::{CompilationError, CompilationKey, DispatchKey};
pub use execution::ExecutionError;
pub use options::{FeedbackDirectedProfile, InputBoundBucketing, Options};
pub use programs::{CompiledProgram, LoweredProgram};
pub use sessions::Session;
pub use target::{Target, TargetDevice};

pub(crate) use compilation::with_input_shardings;
pub(crate) use effects::EffectCallbackGuard;
#[cfg(test)]
pub(crate) use execution::StatefulFailureInjection;
pub(crate) use execution::{validate_input_type, validate_runtime_outputs};

/// Stateful backend that materializes, lowers, compiles, and executes traced XLA programs
/// against a live PJRT [`Client`].
///
/// Domains retain a [`Session`] for the client and compilation cache, a selected effect scope, optional
/// [`DeviceMesh`], and shared default [`CompilationOptions`]. Clones share their scope; [`Self::fork_effect_scope`]
/// starts an independent sequence while retaining the cache and configuration. Cached programs are session-free;
/// execution always uses the invoking domain's scope, including cache hits and retained compiled functions.
///
/// Ordered I/O dependencies gate the successor computation's entire launch, including its pure work. The backend
/// may execute inline or asynchronously. Use unordered printing for throughput-sensitive debugging where ordering
/// does not matter. [`Self::effects_barrier`] also observes unordered I/O and local assertion failures, so it is
/// broader than flushing prints. Call it before dropping the client when its effects must be observed.
///
/// Custom calls and prints declaring `DeviceOrderedIo` may execute per device inside manual computation regions on
/// CPU, while `OrderedIo` occurrences keep the program on one device. Tokens retain per-device dependencies without a
/// Ryft-wide device rendezvous; the backend can impose additional scheduling
/// dependencies, including serialization of multi-device launches. Opaque registered handlers must not reenter
/// dispatch or barriers. Managed print/assertion handlers reject such reentrancy before acquiring scope locks.
///
/// The same domain type covers both staged tracing and concrete execution. Every domain belongs to a [`Session`]
/// and is therefore backed by a live PJRT client: tracing only uses the domain as a type witness and never needs a
/// clientless instance, while compiling ahead of time without devices requires a compile-target abstraction rather
/// than an execution domain without a client.
///
/// The concrete values of this domain are [`Value`]s, every member of which carries the domain that it belongs to.
/// Array IR values therefore dispatch their capabilities through this domain eagerly, free transform entry points
/// recover it from their inputs, and first-class dimensions can produce arrays (e.g., when compared or converted to
/// scalars) without an explicit context.
pub struct Domain<'c> {
    /// Session that owns the client, compilation cache, default effect scope, and kernel runtimes of this domain.
    session: Arc<Session<'c>>,

    /// Concrete device mesh that eager binds prefer when deriving their execution mesh and that the
    /// constant-materialization fast path requires.
    mesh: Option<DeviceMesh>,

    /// Default compilation options forwarded to [`Client::compile`]. Shared because the template is immutable and
    /// [`Domain`] values are cloned into every transform tracer that executes through this domain.
    compilation_options: Arc<CompilationOptions>,

    /// Selected ordering scope, which may be a fork of the session default.
    effect_scope: Arc<EffectScope<'c>>,
}

/// Tracer shape used while staging XLA programs directly from types.
pub(crate) type Tracer<'context> = DomainTracer<Domain<'context>>;

impl<'c> Clone for Domain<'c> {
    fn clone(&self) -> Self {
        Self {
            session: Arc::clone(&self.session),
            mesh: self.mesh.clone(),
            compilation_options: Arc::clone(&self.compilation_options),
            effect_scope: Arc::clone(&self.effect_scope),
        }
    }
}

impl<'c> Domain<'c> {
    /// Returns the shared default [`CompilationOptions`] template.
    #[inline]
    fn default_compilation_options() -> Arc<CompilationOptions> {
        static DEFAULT: LazyLock<Arc<CompilationOptions>> = LazyLock::new(|| Arc::new(CompilationOptions::default()));
        Arc::clone(&DEFAULT)
    }

    /// Creates a domain over `session` using the session's default effect scope and the supplied execution
    /// configuration.
    fn from_session_configuration(
        session: Arc<Session<'c>>,
        mesh: Option<DeviceMesh>,
        compilation_options: Arc<CompilationOptions>,
    ) -> Self {
        Self { effect_scope: Arc::clone(&session.default_effect_scope), session, mesh, compilation_options }
    }

    /// Returns a sibling of this domain that shares its session, effect scope, and mesh but forwards
    /// `compilation_options` to the compile path. Arrays produced through the returned domain inherit it.
    pub fn with_compilation_options(&self, compilation_options: CompilationOptions) -> Self {
        Self { compilation_options: Arc::new(compilation_options), ..self.clone() }
    }

    /// Returns a sibling of this domain that places mesh-less eager outputs on `mesh`. Used to keep arrays derived from
    /// a first-class dimension on the mesh of the array that the dimension was read from, and by XLA-internal unit
    /// tests that exercise the [`Self::constant`] materialization helper.
    pub(crate) fn with_mesh(&self, mesh: DeviceMesh) -> Self {
        Self { mesh: Some(mesh), ..self.clone() }
    }

    /// Forks ordering and completion state while retaining the session, compilation cache, and configuration.
    /// Use forks at independent task boundaries. References retain their own identity-based coordination.
    pub fn fork_effect_scope(&self) -> Self {
        Self { effect_scope: Arc::new(EffectScope::default()), ..self.clone() }
    }

    /// Returns the PJRT [`Client`] of this domain's session.
    #[inline]
    pub fn client(&self) -> &'c Client<'c> {
        self.session.client()
    }

    /// Returns the session shared by this domain.
    #[inline]
    pub fn session(&self) -> &Arc<Session<'c>> {
        &self.session
    }

    /// Returns the [`Target`] of this domain's session. Refer to the documentation of [`Session::target`] for
    /// more information.
    #[inline]
    pub fn target(&self) -> Result<&Target, Error> {
        self.session.target()
    }

    /// Returns the [`DeviceMesh`] this domain resolves shard placement against, or [`ArrayError::MissingMesh`] when the
    /// domain was constructed without a mesh; eager binds derive their mesh from the inputs in that case.
    #[inline]
    pub fn mesh(&self) -> Result<&DeviceMesh, Error> {
        self.mesh.as_ref().ok_or_else(|| Error::from(ArrayError::MissingMesh))
    }

    /// Returns the base [`CompilationOptions`] template that the compile path forwards to PJRT.
    #[inline]
    pub fn compilation_options(&self) -> &CompilationOptions {
        self.compilation_options.as_ref()
    }

    /// Returns the number of compiled programs currently cached in the in-memory tier.
    #[inline]
    pub fn program_count(&self) -> usize {
        self.compilation_context().program_count()
    }

    /// Returns this domain's shared compilation context, which is owned by its session.
    #[inline]
    pub fn compilation_context(&self) -> &CompilationContext<'c, Self> {
        self.session.compilation_context()
    }

    /// Returns a snapshot of the activity of the session's eager dispatch cache, which maps eager operation
    /// applications directly to compiled programs, in front of tracing and lowering.
    #[inline]
    pub fn eager_dispatch_statistics(&self) -> SpecializationCacheStatistics {
        self.session.eager_dispatch_cache.statistics()
    }

    /// Removes every entry from the in-memory compilation cache and the eager dispatch cache of this domain's session.
    /// Mirrors JAX's `clear_caches()`, which also clears the per-primitive eager dispatch cache.
    ///
    /// Eager dispatch cache hits do not re-read the `XLA_FLAGS` environment variable (the miss path still records it
    /// in the compilation key), so call this function after changing it if the change must apply to eager operations
    /// that already ran. XLA itself reads the variable when it initializes its debug options.
    #[inline]
    pub fn clear_cache(&self) {
        self.compilation_context().clear_cache();
        self.session.eager_dispatch_cache.clear();
    }

    /// Waits for this scope's previously reserved effects, including unordered I/O and local assertions.
    /// Backend-selected inline execution is allowed; this function waits only for work still pending. Later
    /// reservations and other scopes are excluded. Call before client destruction to ensure effects are observed.
    pub fn effects_barrier(&self) -> Result<(), Error> {
        self.effect_scope.barrier()
    }

    /// Acknowledges barrier-reported failures after affected work has completed, without waiting or repairing
    /// references. If work is pending or unreported, call `effects_barrier()` and retry this function.
    pub fn acknowledge_effect_errors(&self) -> Result<(), Error> {
        self.effect_scope.acknowledge()
    }
}

impl<'c> CoreDomain for Domain<'c> {
    type Type = ArrayIrType;
    type Value = Value<'c>;
    type Constant = Constant;
    type Operation = Operation;
}

impl ryft_core::NamedAxes for Domain<'_> {
    fn named_axis(&self, _name: &str) -> Option<ryft_core::NamedAxis> {
        None
    }

    fn named_axes(&self) -> Vec<(String, ryft_core::NamedAxis)> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests;
