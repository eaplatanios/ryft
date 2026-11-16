//! XLA sessions and the shared resources used to stage, compile, and execute array programs. A [`Session`] implements
//! [`Domain`] over [`ArrayIrType`], with [`Value`]s that hold device arrays, checked dimensions, or mutable references.
//! Values retain their session, so eager operations and transforms recover the execution context from their inputs.
//! [`Constant`] and [`Operation`] provide the corresponding families of stored constants and operations for traced
//! programs.
//!
//! # Sessions
//!
//! A [`Session`] executes on one PJRT [`Client`] and is a cheap handle to the resources that it creates for
//! that client, which are its compilation cache, runtime extension resources, and default compilation options.
//! [`Session::new`] creates a session on a client that it creates for the best available [`Platform`], and
//! [`Session::builder`] allows configuring everything in detail. Clone a session wherever another handle is
//! needed. Clones share its resources, while creating another session gives an independent cache and effect
//! scope, even on the same client. Configure built-in or downstream providers through [`RuntimeExtensions`] and
//! [`SessionBuilder::runtime_extensions`]. Each session initializes its own runtime instances lazily when an executable
//! requires them.
//!
//! [`Options`](crate::Options) supplies the execution [`DeviceMesh`], sharding overrides, donation flags, and backend
//! compilation policies for a staged function. The session's default [`CompilationOptions`] supply the underlying PJRT
//! compiler settings. [`Target`] describes the client's platform and devices without retaining live device handles.
//!
//! Traces whose capture tables hold constants rather than runtime values refer to runtime arrays through
//! [`SessionConstant`] handles. [`ThroughSession`] imports runtime [`Value`]s into such traces by retaining their
//! arrays in the session, which keeps each array alive for as long as any handle to it exists.
//!
//! # Compilation and Execution
//!
//! The [`crate::compilation`] module implements compilation for a session and [`crate::execution`] implements
//! execution. Use [`compile`](crate::compile) for a compiled function or [`jit`](crate::jit) for a retained
//! dispatcher that specializes on its runtime inputs. The backend also implements the generic lifecycle in
//! [`ryft_core::compilation`]. When staging directly through [`ryft_core::StagedFunction::stage`], apply
//! [`with_input_shardings`](crate::with_input_shardings) to the input types first if the options specify
//! input sharding overrides.
//!
//! [`LoweredProgram`](crate::LoweredProgram) contains StableHLO and the metadata needed for compilation.
//! [`CompiledProgram`](crate::CompiledProgram) contains the loaded executable and its invocation contract. These flat
//! artifacts are distinct from the typed function handles in [`crate::compilation::functions`], which also retain input
//! and output container structures. The session's [`CompilationContext`] coordinates artifact reuse, and
//! [`CompilationKey`](crate::CompilationKey) records the exact computation and compilation configuration.
//! [`CompilationAnalysis`](crate::CompilationAnalysis) and [`OptimizedProgram`](crate::OptimizedProgram)
//! expose compiler diagnostics for a compiled program.
//!
//! # Effects and Completion
//!
//! Cloned sessions share effect ordering. [`Session::fork_effect_scope`] starts an independent sequence while retaining
//! the shared resources of the session. Cached programs contain no session or effect scope, so execution uses the
//! invoking session's scope. Mutable references also retain their own coordination across calls, based on their
//! identities.
//!
//! Array readiness observes completion of the work that produced that array. [`Session::effects_barrier`] observes
//! previously submitted effects, including work whose outputs were discarded. Call it before dropping the client when
//! those effects must be observed. Sessions and their values borrow the client and cannot outlive it. Clients that
//! `ryft-xla` creates for sessions live for the rest of the process.

use std::borrow::Cow;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use ryft_core::{
    ArrayIrType, ArrayType, CapturingContext, CompilationContext, Context, DeviceMesh, DiskCache, Domain, DomainTracer,
    ImportingContext, LogicalMesh, MeshAxis, MeshContext, NamedAxes, NamedAxis, NestedTracingContext, Parameter,
    ProgramError, SpecializationCache, SpecializationCacheStatistics, TracingContext, Type, TypeError,
    TypeIdentityRenaming, Typed,
};
use ryft_macros::Parameter;
use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Client, ClientOptions, Plugin};

use crate::arrays::RetainedArray;
use crate::compilation::keys::EagerDispatchKey;
use crate::sessions::eager::{EAGER_DISPATCH_CACHE_CAPACITY, EagerDispatchProgram};
use crate::sessions::effects::EffectScope;
use crate::sessions::platforms::process_client;
use crate::sessions::runtimes::RuntimeRegistry;
use crate::{Array, ArrayError, Constant, Error, Operation, Value};

pub mod eager;
pub mod effects;
pub mod platforms;
pub mod runtimes;
pub mod targets;

pub use platforms::Platform;
pub use runtimes::{
    ExecutionResources, MAX_RUNTIME_REQUIREMENTS_BYTES, RUNTIME_REQUIREMENTS_ATTRIBUTE, RuntimeError, RuntimeExtension,
    RuntimeExtensions, RuntimeProvider, RuntimeRequirement, named_ffi_type_id, normalize_runtime_requirements,
    runtime_requirements, with_runtime_requirements,
};
pub use targets::{Target, TargetDevice};

// TODO(eaplatanios): Review from here onwards.

/// Source of the identifiers of sessions, which are unique within the process.
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(0);

/// Stateful backend that materializes, lowers, compiles, and executes traced XLA programs against a live PJRT
/// [`Client`].
///
/// A session is the handle through which `ryft-xla` executes on a client. Every [`Array`] carries the session that it
/// belongs to, so eager operations and transforms recover their execution context from their inputs, and first-class
/// dimensions can produce arrays (e.g., when compared or converted to scalars) without an explicit context. Handles are
/// cheap to clone. Build one session per client and derive every other handle from it (i.e., through [`Clone`] or
/// [`Self::fork_effect_scope`]) so that all arrays on that client share one compilation cache and one set of runtime
/// extensions, much like JAX's process-wide caches. Separate sessions remain independent, even on the same client, so
/// building a second session is an explicit request for an isolated cache and effect ordering. Cached
/// executables retain neither sessions nor arrays, so the ownership graph has no cycle between caches and sessions.
///
/// The same type covers both staging and concrete execution. Every session is backed by a live PJRT client. Tracing
/// only uses the session as a type witness and never needs an instance without a client, while compiling ahead of time
/// without devices requires an abstraction of compilation targets rather than a session without a client.
///
/// # Construction
///
/// [`Self::new`] creates a session with the default configuration, which runs on a client that the session creates
/// for the best available [`Platform`]. [`Self::builder`] returns a [`SessionBuilder`] for everything else.
/// [`SessionBuilder::platform`], [`SessionBuilder::plugin`], and [`SessionBuilder::client_options`] select and
/// configure the client that the session creates. Clients that are created this way live for the rest of the process
/// and are shared by the sessions that request the same configuration, so their sessions can be `Session<'static>`.
/// [`SessionBuilder::client`] instead takes a client that the caller created and owns, which is the extension point
/// for clients that `ryft-xla` cannot create and for programs that must destroy their client. [`Self::new_mesh`]
/// creates device meshes over the devices of the client.
///
/// # Shared and Selected State
///
/// All handles of one session share its compilation cache, its eager dispatch cache, its runtime extensions, its
/// retained constants, and its default [`CompilationOptions`]. Each handle additionally selects an effect scope and an
/// optional [`DeviceMesh`]. Clones share their scope, while [`Self::fork_effect_scope`] starts an independent sequence
/// and retains everything else. Cached programs retain no session, so execution always uses the invoking session's
/// scope, including for cache hits and retained compiled functions.
///
/// # Compilation Caching
///
/// Each session owns one [`CompilationContext`], which its builder configures. By default, it keeps compiled programs
/// only in memory. [`SessionBuilder::disk_cache`] adds a persistent [`DiskCache`] so that later processes restart
/// warm, and [`SessionBuilder::disk_cache_from_env`] adds the one that the [`DiskCache::ENV_VAR`] environment variable
/// configures, which sessions never read otherwise. [`SessionBuilder::compilation_context`] supplies a whole context
/// (e.g., one with a bounded in-memory tier or, for multi-process programs, one with an exchange from
/// [`DistributedRuntime::compilation_artifact_exchange`](crate::DistributedRuntime::compilation_artifact_exchange)).
/// Refer to [`CompilationContext`] for a configuration example.
///
/// # Effects and Completion
///
/// Ordered I/O dependencies gate the successor computation's entire launch, including its pure work. The backend
/// may execute inline or asynchronously. Use unordered printing when debugging code whose throughput matters and
/// whose ordering does not. [`Self::effects_barrier`] also observes unordered I/O and local assertion failures, so it
/// is broader than flushing prints. Call it before dropping the client when its effects must be observed. Destruction
/// of the last handle waits for launches borrowing extension resources before releasing those resources. Ordinary
/// execution adds no implicit destruction barrier, and backend behavior with unfinished device work is specific to
/// each plugin.
///
/// Custom calls and prints declaring [`EffectClass::DeviceOrderedIo`](ryft_core::EffectClass::DeviceOrderedIo) may
/// execute separately on each device inside manual computation regions on CPU, while occurrences of
/// [`EffectClass::OrderedIo`](ryft_core::EffectClass::OrderedIo) keep the program on one device. Tokens retain the
/// dependencies of each device without a device rendezvous across Ryft. The backend can impose additional scheduling
/// dependencies, including serialization of multi-device launches. Opaque registered handlers must not reenter
/// dispatch or barriers. Managed print and assertion handlers reject such reentrancy before acquiring scope locks.
///
/// # Example
///
/// ```
/// use ryft_core::{ArrayType, DataType, MeshAxis, MeshAxisType, Print};
/// use ryft_xla::{Platform, Session};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let session = Session::builder().platform(Platform::Cpu).build()?;
/// let mesh = session.new_mesh([MeshAxis::new("x", 1, MeshAxisType::Auto)?])?;
/// let value = session.array(ArrayType::scalar(DataType::F64), mesh, 1.0f64.to_ne_bytes())?;
/// value.print("value");
/// session.effects_barrier()?;
/// # Ok(())
/// # }
/// ```
///
/// A session over a client that the caller owns cannot outlive that client.
///
/// ```compile_fail
/// use ryft_xla::Session;
/// fn escaped(client: ryft_pjrt::Client<'_>) {
///     let session = Session::builder().client(&client).build().unwrap();
///     drop(client);
///     session.effects_barrier().unwrap();
/// }
/// ```
pub struct Session<'c> {
    /// Resources shared by every handle of this session.
    state: Arc<SessionState<'c>>,

    /// Concrete device mesh that eager binds prefer when deriving their execution mesh and that the constant
    /// materialization fast path requires.
    mesh: Option<DeviceMesh>,

    /// Selected ordering and completion scope, which clones of this handle share and which
    /// [`Self::fork_effect_scope`] replaces.
    effect_scope: Arc<EffectScope<'c>>,
}

/// Resources that all handles of one [`Session`] share. Its caches, runtime instances, and constant table are mutated
/// through interior mutability, while its configuration is immutable after construction.
struct SessionState<'c> {
    /// Identifier of the session, unique within the process, which [`SessionConstant`] handles use to refer to it.
    id: u64,

    /// Client of the session, which outlives cached executables and effect carriers. It is either owned by the caller
    /// or one of the clients that `ryft-xla` created and that live for the rest of the process.
    client: &'c Client<'c>,

    /// Compilation and execution target snapshot of [`Self::client`], read lazily and retained for the lifetime of the
    /// session, including a failed read.
    target: OnceLock<Result<Target, Error>>,

    /// Compilation cache of the session, which [`Session::compilation_context`] exposes. Cached artifacts contain no
    /// runtime execution association.
    cache: CompilationContext<'c, Session<'c>>,

    /// Compiled programs of eager operation applications, keyed ahead of program construction and lowering. Refer to
    /// the documentation of [`EagerDispatchKey`] for information on what the key captures.
    eager_dispatch_cache: SpecializationCache<EagerDispatchKey, EagerDispatchProgram<'c>>,

    /// Template of the options forwarded to [`Client::compile`], which every compilation of the session starts from.
    /// It is immutable because the eager dispatch cache does not key its entries on it.
    default_compilation_options: CompilationOptions,

    /// Runtime extensions associated with the client, and ownership of launches borrowing their resources.
    runtimes: RuntimeRegistry<'c>,

    /// Arrays that [`SessionConstant`] handles refer to, indexed by [`SessionConstant::index`]. A [`None`] entry was
    /// reclaimed and may be reused. Refer to [`Session::retain_constant`] for information on how arrays enter and
    /// leave this table.
    constants: Mutex<Vec<Option<RetainedConstant<'c>>>>,
}

impl<'c> Clone for Session<'c> {
    fn clone(&self) -> Self {
        Self { state: Arc::clone(&self.state), mesh: self.mesh.clone(), effect_scope: Arc::clone(&self.effect_scope) }
    }
}

impl<'c> Session<'c> {
    /// Creates an independent session with the default configuration, which runs on a client that the session
    /// creates for the best available [`Platform`] and has an empty in-memory compilation cache, the built-in runtime
    /// extensions, and the default compilation options. This is a shorthand for a [`SessionBuilder`] that is built
    /// without any configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if no client can be created, which requires even the built-in CPU plugin to fail.
    pub fn new() -> Result<Self, Error> {
        Self::builder().build()
    }

    /// Returns a [`SessionBuilder`] with the default configuration, which builds a session on the best available
    /// [`Platform`] with an empty in-memory compilation cache, the built-in runtime extensions, and the default
    /// compilation options.
    pub fn builder() -> SessionBuilder<'c> {
        SessionBuilder {
            client_source: ClientSource::Platform(Platform::Auto),
            client_options: None,
            compilation_context: CompilationContext::new(),
            disk_cache: None,
            runtime_extensions: RuntimeExtensions::new(),
            default_compilation_options: CompilationOptions::default(),
        }
    }

    /// Creates an independent session over `client`, which the caller owns, with an empty in-memory compilation cache
    /// and a fresh effect scope, for the unit tests of this crate. This is what a [`SessionBuilder`] that is only given
    /// [`SessionBuilder::client`] builds, without the [`Result`] of [`SessionBuilder::build`], which cannot be an
    /// error for that configuration.
    #[cfg(test)]
    pub(crate) fn with_client(client: &'c Client<'c>) -> Self {
        Self::from_parts(client, CompilationContext::new(), RuntimeExtensions::new(), CompilationOptions::default())
    }

    /// Creates an independent session with a fresh effect scope from its client and the configuration of the
    /// resources that its handles share.
    fn from_parts(
        client: &'c Client<'c>,
        cache: CompilationContext<'c, Self>,
        extensions: RuntimeExtensions<'c>,
        default_compilation_options: CompilationOptions,
    ) -> Self {
        Self {
            state: Arc::new(SessionState {
                id: NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed),
                client,
                target: OnceLock::new(),
                cache,
                eager_dispatch_cache: SpecializationCache::new(EAGER_DISPATCH_CACHE_CAPACITY),
                default_compilation_options,
                runtimes: RuntimeRegistry::new(client, extensions),
                constants: Mutex::new(Vec::new()),
            }),
            mesh: None,
            effect_scope: Arc::new(EffectScope::default()),
        }
    }

    /// Returns a handle of this session that places eager outputs on `mesh` when they would otherwise have no mesh.
    /// Used to keep arrays derived from a first-class dimension on the mesh of the array that the dimension was read
    /// from, and by unit tests of this crate that exercise the [`Self::constant`] materialization helper.
    pub(crate) fn with_mesh(&self, mesh: DeviceMesh) -> Self {
        Self { mesh: Some(mesh), ..self.clone() }
    }

    /// Forks ordering and completion state while retaining the resources that all handles of this session share and
    /// the mesh of this handle. Use forks at independent task boundaries. References retain their own coordination,
    /// based on their identities.
    pub fn fork_effect_scope(&self) -> Self {
        Self { effect_scope: Arc::new(EffectScope::default()), ..self.clone() }
    }

    /// Returns the PJRT client of this session.
    #[inline]
    pub fn client(&self) -> &'c Client<'c> {
        self.state.client
    }

    /// Returns the [`Target`] of this session's client, reading it from the client on first use.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if querying the platform, plugin, or device facts from the client failed. The
    /// failure is retained, so every later call returns the same error.
    pub fn target(&self) -> Result<&Target, Error> {
        self.state
            .target
            .get_or_init(|| Target::from_client(self.state.client))
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Returns the [`DeviceMesh`] that this handle resolves shard placement against, or [`ArrayError::MissingMesh`]
    /// when the handle has no mesh. Eager binds derive their mesh from the inputs in that case.
    #[inline]
    pub fn mesh(&self) -> Result<&DeviceMesh, Error> {
        self.mesh.as_ref().ok_or_else(|| Error::from(ArrayError::MissingMesh))
    }

    /// Returns the template of the options that every compilation of this session forwards to [`Client::compile`].
    /// Refer to [`SessionBuilder::default_compilation_options`] for more information.
    #[inline]
    pub fn default_compilation_options(&self) -> &CompilationOptions {
        &self.state.default_compilation_options
    }

    /// Returns the compilation cache that all handles of this session share.
    #[inline]
    pub fn compilation_context(&self) -> &CompilationContext<'c, Self> {
        &self.state.cache
    }

    /// Returns whether `other` is a handle of the same session as this one (i.e., whether the two share their
    /// compilation cache, runtime extensions, and retained constants), irrespective of their meshes and effect scopes.
    #[inline]
    pub(crate) fn is_same_session(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    /// Returns the effect ordering scope that executions through this handle reserve their submissions in.
    #[inline]
    pub(crate) fn effect_scope(&self) -> &EffectScope<'c> {
        &self.effect_scope
    }

    /// Returns the runtime extension instances of this session and the launches that still borrow their resources.
    #[inline]
    pub(crate) fn runtimes(&self) -> &RuntimeRegistry<'c> {
        &self.state.runtimes
    }

    /// Returns the compiled programs of eager operation applications, keyed ahead of program construction and lowering.
    /// Refer to the documentation of [`EagerDispatchKey`] for information on what the key captures.
    #[inline]
    fn eager_dispatch_cache(&self) -> &SpecializationCache<EagerDispatchKey, EagerDispatchProgram<'c>> {
        &self.state.eager_dispatch_cache
    }

    /// Returns the number of compiled programs currently cached in the in-memory tier.
    #[inline]
    pub fn program_count(&self) -> usize {
        self.compilation_context().program_count()
    }

    /// Returns a snapshot of the activity of this session's eager dispatch cache, which maps eager operation
    /// applications directly to compiled programs, ahead of program construction and lowering.
    #[inline]
    pub fn eager_dispatch_statistics(&self) -> SpecializationCacheStatistics {
        self.state.eager_dispatch_cache.statistics()
    }

    /// Removes every entry from the in-memory compilation cache and the eager dispatch cache of this session.
    /// Mirrors JAX's `clear_caches()`, which also clears the eager dispatch cache of each primitive.
    ///
    /// Eager dispatch cache hits do not re-read the `XLA_FLAGS` environment variable, although the miss path still
    /// records it in the compilation key, so call this function after changing it if the change must apply to eager
    /// operations that already ran. XLA itself reads the variable when it initializes its debug options.
    #[inline]
    pub fn clear_cache(&self) {
        self.compilation_context().clear_cache();
        self.state.eager_dispatch_cache.clear();
    }

    /// Creates a [`DeviceMesh`] with the provided `axes` over the devices of this session's client, like
    /// `jax.make_mesh`. The mesh takes as many devices as its axes require, in the order in which the client reports
    /// them. In a program with multiple processes, these include devices that other processes own.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if the devices of the client cannot be queried, and a
    /// [`ShardingError`](ryft_core::ShardingError) if `axes` do not form a valid mesh or require more devices than
    /// the client has.
    pub fn new_mesh<A: IntoIterator<Item = MeshAxis>>(&self, axes: A) -> Result<DeviceMesh, Error> {
        let logical_mesh = LogicalMesh::new(axes.into_iter().collect())?;
        let devices = self
            .target()?
            .devices()
            .iter()
            .take(logical_mesh.device_count())
            .map(TargetDevice::device)
            .collect();
        Ok(DeviceMesh::new(logical_mesh, devices)?)
    }

    /// Uploads host bytes into an array that belongs to this session. Refer to the documentation of
    /// [`Array::from_host_buffer`] for more information.
    pub fn array<B: AsRef<[u8]>>(&self, r#type: ArrayType, mesh: DeviceMesh, buffer: B) -> Result<Array<'c>, Error> {
        Array::from_host_buffer(self, r#type, mesh, buffer)
    }

    /// Waits for the previously reserved effects of this handle's scope, including unordered I/O and local
    /// assertions. The backend may execute work inline, and this function waits only for work that is still pending.
    /// Later reservations and other scopes (e.g., those of forked handles) are excluded. Failures are retained until
    /// explicitly acknowledged. Call this function before client destruction to ensure that effects are observed.
    pub fn effects_barrier(&self) -> Result<(), Error> {
        self.effect_scope.barrier()
    }

    /// Acknowledges failures that a barrier reported after the affected work has completed, without waiting or
    /// repairing references. If work is pending or unreported, call [`Self::effects_barrier`] and retry this function.
    /// This never changes errors retained by existing arrays or executions.
    pub fn acknowledge_effect_errors(&self) -> Result<(), Error> {
        self.effect_scope.acknowledge()
    }

    /// Returns a new [`Array`] that belongs to this handle and shares the storage that `constant` refers to.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Type`] error if `constant` was not retained by this session or is no longer in the
    /// constant table of this session.
    pub(crate) fn retained_constant(&self, constant: &SessionConstant) -> Result<Array<'c>, ProgramError> {
        if constant.session() != self.state.id {
            return Err(TypeError::invalid(format!(
                "session constant `{constant}` belongs to session #{} but was resolved by session #{}",
                constant.session(),
                self.state.id,
            ))
            .into());
        }
        let constants = self.state.constants.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let retained = constants
            .get(constant.index())
            .and_then(Option::as_ref)
            .filter(|retained| std::ptr::eq(retained.owner.as_ptr(), Arc::as_ptr(constant.owner())))
            .ok_or_else(|| {
                TypeError::invalid(format!("session constant `{constant}` is not in the constant table of its session"))
            })?;
        Ok(retained.storage.to_array(self))
    }

    /// Retains the storage of `array` and returns a [`SessionConstant`] handle that refers to it. The storage stays
    /// retained for as long as any handle to it exists, as the ownership section of the [`SessionConstant`]
    /// documentation describes. Retaining an array whose storage a live handle already refers to returns a handle to
    /// the same entry, so that repeatedly importing the same captured array (e.g., the weights that a compiled function
    /// captures) does not grow the table. Entries without handles are reclaimed here, which releases their storage and
    /// makes their positions reusable.
    fn retain_constant(&self, array: &Array<'c>) -> SessionConstant {
        let mut constants = self.state.constants.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for constant in constants.iter_mut() {
            if constant.as_ref().is_some_and(|constant| constant.owner.strong_count() == 0) {
                *constant = None;
            }
        }
        let existing = constants.iter().enumerate().find_map(|(index, constant)| {
            let constant = constant.as_ref().filter(|constant| constant.storage.retains(array))?;
            Some((index, constant.owner.upgrade()?, constant.storage.r#type().clone()))
        });
        if let Some((index, owner, r#type)) = existing {
            return SessionConstant::new(self.state.id, index, r#type, owner);
        }
        let owner = Arc::new(SessionConstantOwner);
        let constant = RetainedConstant { owner: Arc::downgrade(&owner), storage: array.retain() };
        let r#type = constant.storage.r#type().clone();
        let index = match constants.iter().position(Option::is_none) {
            Some(index) => {
                constants[index] = Some(constant);
                index
            }
            None => {
                constants.push(Some(constant));
                constants.len() - 1
            }
        };
        SessionConstant::new(self.state.id, index, r#type, owner)
    }
}

impl<'c> Domain for Session<'c> {
    type Type = ArrayIrType;
    type Value = Value<'c>;
    type Constant = Constant;
    type Operation = Operation;
}

impl NamedAxes for Session<'_> {
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        let _ = name;
        None
    }

    fn named_axes(&self) -> Vec<(String, NamedAxis)> {
        Vec::new()
    }
}

impl MeshContext for Session<'_> {
    fn default_mesh(&self) -> Option<LogicalMesh> {
        // A session handle that was given a device mesh through `Session::with_mesh` owns the logical view of that
        // mesh as its default, so eager code bound in it can omit meshes (e.g., `shard_map(.., None, ..)`), as under
        // JAX's `jax.set_mesh`. A handle without a mesh has no default.
        self.mesh.as_ref().map(|mesh| mesh.logical_mesh().clone())
    }
}

/// Tracer shape used while staging XLA programs directly from types, which is the [`DomainTracer`] of a [`Session`].
pub type Tracer<'c> = DomainTracer<Session<'c>>;

/// Invalid configuration of a [`SessionBuilder`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SessionError {
    /// The Cargo feature of `ryft-xla` that provides the plugin of the requested platform is not enabled.
    #[error("platform `{platform}` is unavailable because `ryft-xla` was built without its `{feature}` feature")]
    UnavailablePlatform { platform: Platform, feature: &'static str },

    /// Client options were supplied for a client that they cannot configure.
    #[error("invalid client options: {reason}")]
    InvalidClientOptions { reason: String },
}

/// Source of the PJRT client of a session that is being built.
enum ClientSource<'c> {
    /// Process-wide client that `ryft-xla` creates for a platform.
    Platform(Platform),

    /// Process-wide client that `ryft-xla` creates through a plugin that the caller loaded.
    Plugin(Plugin),

    /// Client that the caller created and owns.
    Client(&'c Client<'c>),
}

/// Persistent tier that a session under construction adds to its compilation cache.
enum DiskCacheSource {
    /// Disk cache that the caller supplied.
    Explicit(DiskCache),

    /// Disk cache that the [`DiskCache::ENV_VAR`] environment variable configures, if it is set.
    Environment,
}

/// Builder of [`Session`]s, which [`Session::builder`] returns and which is the one interface for configuring their
/// construction. [`Session::new`] is a shorthand for building a session without any configuration.
///
/// A builder first selects the PJRT client of the session. By default, the session creates its own client for
/// [`Platform::Auto`]. [`Self::platform`] requests a specific platform, [`Self::plugin`] uses a PJRT plugin that
/// `ryft-xla` does not know about, and [`Self::client_options`] configures the client that either of them creates.
/// These clients live for the rest of the process, and sessions that request the same plugin with equal client
/// options share one client, as the [`platforms`] module documents. [`Self::client`]
/// instead supplies a client that the caller created and owns. Each of [`Self::platform`], [`Self::plugin`], and
/// [`Self::client`] replaces the selection of the others.
///
/// The remaining functions configure the resources that all handles of the session share. The order in which a
/// builder is configured does not matter otherwise.
///
/// # Example
///
/// ```no_run
/// use ryft_core::DiskCache;
/// use ryft_pjrt::{ClientOptions, CpuClientOptions};
/// use ryft_xla::{Platform, Session};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // A session on the best available platform, whose compiled programs persist across processes.
/// let session = Session::builder().disk_cache(DiskCache::open("/tmp/ryft-cache")?).build()?;
///
/// // A session on a CPU client with four devices, e.g., for testing sharded programs.
/// let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() });
/// let session = Session::builder().platform(Platform::Cpu).client_options(options).build()?;
/// # Ok(())
/// # }
/// ```
pub struct SessionBuilder<'c> {
    /// Source of the client of the session.
    client_source: ClientSource<'c>,

    /// Options of the client that `ryft-xla` creates for the session, if the caller supplied any.
    client_options: Option<ClientOptions>,

    /// Compilation cache of the session, before [`Self::disk_cache`] is applied to it.
    compilation_context: CompilationContext<'c, Session<'c>>,

    /// Persistent tier that replaces the one of [`Self::compilation_context`], if the caller requested one.
    disk_cache: Option<DiskCacheSource>,

    /// Runtime provider configuration of the session.
    runtime_extensions: RuntimeExtensions<'c>,

    /// Template of the options that the session forwards to [`Client::compile`].
    default_compilation_options: CompilationOptions,
}

impl<'c> SessionBuilder<'c> {
    /// Requests a client that `ryft-xla` creates for `platform`. An explicit platform never falls back to another
    /// one, so building the session fails if `platform` is unavailable.
    pub fn platform(mut self, platform: Platform) -> Self {
        self.client_source = ClientSource::Platform(platform);
        self
    }

    /// Requests a client that `ryft-xla` creates through `plugin`, for PJRT plugins that [`Platform`] does not name
    /// (e.g., one loaded with [`load_plugin`](ryft_pjrt::load_plugin)). The client is created with the options of
    /// [`Self::client_options`], or with the default [`ClientOptions`] if none are supplied.
    pub fn plugin(mut self, plugin: Plugin) -> Self {
        self.client_source = ClientSource::Plugin(plugin);
        self
    }

    /// Supplies the options of the client that `ryft-xla` creates for the session (e.g., the number of CPU devices,
    /// or the visible devices and the memory allocator of a GPU client) in place of the defaults of its platform. The
    /// options must fit the platform or plugin of the session, so they require an explicit one.
    pub fn client_options(mut self, client_options: ClientOptions) -> Self {
        self.client_options = Some(client_options);
        self
    }

    /// Supplies a client that the caller created and owns, which the session borrows and never destroys. This is the
    /// extension point for clients that `ryft-xla` cannot create (e.g., a client of a
    /// [`DistributedRuntime`](crate::DistributedRuntime)) and for programs that must destroy their client.
    pub fn client(mut self, client: &'c Client<'c>) -> Self {
        self.client_source = ClientSource::Client(client);
        self
    }

    /// Supplies the compilation cache of the session in place of the default one, which keeps up to its default
    /// number of compiled programs in memory. Refer to [`CompilationContext`] for information on its tiers, its
    /// artifact exchange, and its event reporting.
    pub fn compilation_context(mut self, compilation_context: CompilationContext<'c, Session<'c>>) -> Self {
        self.compilation_context = compilation_context;
        self
    }

    /// Makes the compilation cache of the session also write through to `disk_cache`, with its size cap and write
    /// thresholds, so that later processes restart warm. This replaces the disk cache of a context that was supplied
    /// through [`Self::compilation_context`].
    pub fn disk_cache(mut self, disk_cache: DiskCache) -> Self {
        self.disk_cache = Some(DiskCacheSource::Explicit(disk_cache));
        self
    }

    /// Makes the compilation cache of the session also write through to the [`DiskCache`] that the
    /// [`DiskCache::ENV_VAR`] environment variable configures, if it is set. An absent variable leaves the cache as
    /// it is otherwise configured. The variable is read when the session is built.
    pub fn disk_cache_from_env(mut self) -> Self {
        self.disk_cache = Some(DiskCacheSource::Environment);
        self
    }

    /// Supplies the immutable runtime provider configuration of the session in place of the built-in providers.
    ///
    /// Initialized instances belong to the session, even when `runtime_extensions` was cloned from another
    /// configuration. Providers initialize lazily when an executable requires them, so building the session does not
    /// load GPU drivers on their behalf.
    pub fn runtime_extensions(mut self, runtime_extensions: RuntimeExtensions<'c>) -> Self {
        self.runtime_extensions = runtime_extensions;
        self
    }

    /// Supplies the template of the options that the session forwards to [`Client::compile`].
    ///
    /// Every compilation of the session, including those of eager operations, starts from
    /// `default_compilation_options`, unless its [`Options`](crate::Options) supply their own through
    /// [`Options::with_compilation_options`](crate::Options::with_compilation_options). Lowering then overlays the
    /// settings that it derives for each program (i.e., the replica and partition counts, the partitioner selection,
    /// and the feedback-directed profile of its options). The template therefore suits settings that hold for every
    /// program of the session (e.g., operand precisions and environment option overrides) rather than settings of one
    /// program signature (e.g., argument layouts). It is immutable once the session is built.
    pub fn default_compilation_options(mut self, default_compilation_options: CompilationOptions) -> Self {
        self.default_compilation_options = default_compilation_options;
        self
    }

    /// Builds an independent session with a fresh effect scope.
    ///
    /// # Errors
    ///
    /// Returns a [`SessionError`] if the requested platform is unavailable in this build or if client options were
    /// supplied for [`Platform::Auto`] or for a client that the caller owns, [`Error::Pjrt`] if the plugin of the
    /// session cannot be loaded or initialized or its client cannot be created, and [`Error::Io`] if the disk cache
    /// that the environment configures is invalid. A session that is only given [`Self::client`] and resources that
    /// do not read the environment cannot fail.
    pub fn build(self) -> Result<Session<'c>, Error> {
        let client: &'c Client<'c> = match (self.client_source, self.client_options) {
            (ClientSource::Platform(platform), client_options) => platform.client(client_options)?,
            (ClientSource::Plugin(plugin), client_options) => {
                process_client(&plugin, client_options.unwrap_or_default())?
            }
            (ClientSource::Client(client), None) => client,
            (ClientSource::Client(_), Some(_)) => {
                return Err(SessionError::InvalidClientOptions {
                    reason: "client options only configure a client that the session creates, but the session was \
                             given an existing client"
                        .to_string(),
                }
                .into());
            }
        };
        let compilation_context = match self.disk_cache {
            None => self.compilation_context,
            Some(DiskCacheSource::Explicit(disk_cache)) => self.compilation_context.with_disk_cache(disk_cache),
            Some(DiskCacheSource::Environment) => self.compilation_context.with_disk_cache_from_env()?,
        };
        Ok(Session::from_parts(client, compilation_context, self.runtime_extensions, self.default_compilation_options))
    }
}

/// Handle without a lifetime to an array that a [`Session`] retains on behalf of a trace whose capture table cannot
/// hold runtime arrays. Such a trace (e.g., the body of a [`rematerialize`](ryft_core::rematerialize) call or of a
/// [`composite`](ryft_core::composite) and its rules, which are traced over constants alone) still needs to refer to
/// the runtime arrays that a compiled function it calls has captured. Importing such an array through
/// [`ThroughSession`] registers it in the session and stages this handle as an immediate constant, which can appear in
/// any region of any program, including lazily traced derivative rules.
///
/// The handle resolves to the retained array through the session that registered it. An eager session resolves it when
/// lifting it, and lowering turns every distinct handle of a program into a hidden executable argument that follows the
/// captures of the program, so that compiled executables depend only on the array's type and never on its data.
///
/// # Ownership
///
/// Every handle co-owns its table entry through a shared ownership token, and so the programs, regions, derivative
/// rules, and caches that hold a handle, or a clone of it, keep the array alive. The session reclaims an entry once no
/// handle to it remains, and only then may it reuse its position for another array. Handles compare and hash by their
/// session, position, and type, which therefore identify one array for as long as any handle exists.
#[derive(Clone, Debug, Parameter)]
pub struct SessionConstant {
    /// Identifier of the [`Session`] that retains the array.
    session: u64,

    /// Position of the array in the constant table of that session.
    index: usize,

    /// [`ArrayType`] of the retained array.
    r#type: ArrayType,

    /// Token shared by every handle to the retained array, whose last release lets the session reclaim it.
    owner: Arc<SessionConstantOwner>,
}

impl SessionConstant {
    /// Creates a new [`SessionConstant`] for the array at `index` in the constant table of session `session`, which
    /// co-owns that entry through `owner`.
    #[inline]
    fn new(session: u64, index: usize, r#type: ArrayType, owner: Arc<SessionConstantOwner>) -> Self {
        Self { session, index, r#type, owner }
    }

    /// Returns the identifier of the session that retains the array.
    #[inline]
    pub fn session(&self) -> u64 {
        self.session
    }

    /// Returns the position of the array in the constant table of its session.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the ownership token that this handle shares with every other handle to the same retained array.
    #[inline]
    fn owner(&self) -> &Arc<SessionConstantOwner> {
        &self.owner
    }

    /// Returns this handle with the identities of its type renamed by `renaming`.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self {
            session: self.session,
            index: self.index,
            r#type: self.r#type.rename_identities(renaming)?,
            owner: Arc::clone(&self.owner),
        })
    }
}

// Handles are identified by their session, position, and type. The ownership token is shared by every handle to one
// array and carries no identity of its own.
impl PartialEq for SessionConstant {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.session == other.session && self.index == other.index && self.r#type == other.r#type
    }
}

impl Eq for SessionConstant {}

impl Hash for SessionConstant {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.session.hash(state);
        self.index.hash(state);
        self.r#type.hash(state);
    }
}

impl Display for SessionConstant {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "session#{}:{}", self.index, self.r#type)
    }
}

impl Typed for SessionConstant {
    type Type = ArrayType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

/// Import path of [`ImportingContext`] through which a trace whose capture table holds constants rather than runtime
/// values (i.e., one whose [`CapturingContext::Capture`] is [`Constant`], such as the body of a
/// [`rematerialize`](ryft_core::rematerialize) or [`composite`](ryft_core::composite) call) imports runtime XLA
/// [`Value`]s. Arrays are retained by their [`Session`], once for each distinct storage, for as long as a handle to
/// them exists and enter the trace as [`Constant::Session`] handles. Dimensions enter as immediate
/// [`Constant::Dimension`]s, and references are rejected because their state cannot be shared with a trace that
/// outlives the current transform.
pub struct ThroughSession;

impl<'c> ImportingContext<Session<'c>, ThroughSession> for TracingContext<Constant, Operation, Constant> {
    #[inline]
    fn import(&self, value: Value<'c>) -> Result<Self::Value, ProgramError> {
        self.lift(session_constant(value)?)
    }
}

impl<'c, C: CapturingContext<Capture = Constant>> ImportingContext<Session<'c>, ThroughSession>
    for NestedTracingContext<C>
where
    Self: Context<Constant = Constant>,
{
    #[inline]
    fn import(&self, value: Value<'c>) -> Result<Self::Value, ProgramError> {
        self.lift(session_constant(value)?)
    }
}

/// Ownership token that every [`SessionConstant`] handle to one retained array shares. The session keeps only a weak
/// reference to it, so the token is released together with the last handle.
#[derive(Debug)]
struct SessionConstantOwner;

/// Entry of the constant table of a [`Session`].
struct RetainedConstant<'c> {
    /// Weak reference to the token that every [`SessionConstant`] handle to this entry shares. The entry may be
    /// reclaimed once the token is released.
    owner: Weak<SessionConstantOwner>,

    /// Retained storage of the array.
    storage: RetainedArray<'c>,
}

/// Converts `value` into the immediate [`Constant`] that a trace capturing constants stages for it.
fn session_constant(value: Value) -> Result<Constant, ProgramError> {
    match value {
        Value::Array(array) => Ok(Constant::Session(array.session().retain_constant(&array))),
        Value::Dimension(dimension) => Ok(Constant::Dimension(dimension.into_value())),
        Value::Reference(_) => {
            Err(TypeError::invalid("cannot import an xla reference into a trace whose capture table holds constants")
                .into())
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;
    #[cfg(feature = "cuda-13")]
    use std::convert::Infallible;

    use pretty_assertions::assert_eq;
    #[cfg(feature = "cuda-13")]
    use ryft_core::{Add, EffectClass, OneLike, ProjectedValue, ReferenceRead, ReferenceWrite, Tracer as CoreTracer};
    use ryft_core::{
        ArrayReference, DataType, Device, Dimension as CoreDimension, DimensionValue, Error as CoreError, MeshAxisType,
        Neg, Placeholder, ProjectedContext, ReferenceReadOperation, ReferenceType, Shape, Sharding, ShardingError,
    };
    #[cfg(feature = "cuda-13")]
    use ryft_cuda::{
        CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaKernelParameterType,
    };
    #[cfg(feature = "cuda-13")]
    use ryft_kernels::{KernelCompilationError, KernelCompiler, KernelDefinition, KernelSchedule, VerifiedKernel};
    use ryft_pjrt::protos::Precision;
    use ryft_pjrt::{CpuClientOptions, load_cpu_plugin};
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform, load_cuda_13_plugin};

    use crate::sessions::runtimes::builtin_requirement;
    use crate::tests::{assert_send_sync, cpu_client, execution_client, hash_of, values_from_bytes, values_to_bytes};
    #[cfg(feature = "cuda-13")]
    use crate::{CUDA_KERNEL_CUSTOM_CALL_TARGET, CudaKernelEmbedding, KernelCompilerBinding, Options, stage_kernel};
    use crate::{Dimension, ExecutionError, FlatProgram, FromPjrt, ProgramBuilder, Reference};

    use super::*;

    /// Builds a one-axis mesh from the client's addressable devices for domain tests.
    pub(crate) fn domain_mesh(client: &Client<'_>, axis: &str, axis_size: usize) -> DeviceMesh {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(axis, axis_size, MeshAxisType::Auto).unwrap()]).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        DeviceMesh::new(logical_mesh, devices).unwrap()
    }

    /// Creates a mesh of two devices with the axis interpretation required by the test.
    pub(crate) fn two_device_mesh(client: &Client<'_>, axis_type: MeshAxisType) -> DeviceMesh {
        let devices: Vec<Device> = client
            .addressable_devices()
            .unwrap()
            .iter()
            .take(2)
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect();
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap(), devices).unwrap()
    }

    /// Builds a flat program that reads and returns one externally supplied reference.
    pub(crate) fn external_reference_read_program(referent_type: ArrayType) -> FlatProgram {
        let mut builder = ProgramBuilder::new();
        let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(referent_type)));
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    /// Creates an independent session and projects it to ordinary array values.
    pub(crate) fn array_domain<'c>(client: &'c Client<'c>) -> ProjectedContext<Session<'c>, ArrayType> {
        ProjectedContext::new(Session::with_client(client))
    }

    /// Borrows the array carried by a test value, requiring the array variant.
    pub(crate) fn program_array<'o, 'c>(value: &'o Value<'c>) -> &'o Array<'c> {
        let Value::Array(array) = value else {
            panic!("expected an array IR value");
        };
        array
    }

    /// Constructs a rank-1 `f32` type replicated over `mesh`.
    pub(crate) fn replicated_vector_type(mesh: &DeviceMesh, size: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(size)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap()
    }

    /// Constructs a scalar of `data_type` replicated over `mesh`.
    pub(crate) fn replicated_scalar_type(mesh: &DeviceMesh, data_type: DataType) -> ArrayType {
        ArrayType::scalar(data_type)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap()
    }

    /// Uploads a replicated rank-1 `f32` test array.
    pub(crate) fn f32_vector<'c>(session: &Session<'c>, mesh: &DeviceMesh, values: &[f32]) -> Array<'c> {
        let r#type = replicated_vector_type(mesh, values.len());
        Array::from_host_buffer(session, r#type, mesh.clone(), values_to_bytes::<f32>(values).as_slice()).unwrap()
    }

    /// Uploads a replicated `f32` test scalar.
    pub(crate) fn f32_scalar<'c>(session: &Session<'c>, mesh: &DeviceMesh, value: f32) -> Array<'c> {
        let r#type = replicated_scalar_type(mesh, DataType::F32);
        Array::from_host_buffer(session, r#type, mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
    }

    /// Uploads a replicated rank-1 `f64` test array.
    pub(crate) fn f64_vector<'c>(session: &Session<'c>, mesh: &DeviceMesh, values: &[f64]) -> Array<'c> {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        Array::from_host_buffer(session, r#type, mesh.clone(), values_to_bytes::<f64>(values).as_slice()).unwrap()
    }

    /// Uploads a replicated rank-1 Boolean test array using one byte per element.
    pub(crate) fn boolean_vector<'c>(session: &Session<'c>, mesh: &DeviceMesh, values: &[bool]) -> Array<'c> {
        let r#type = ArrayType::new(DataType::Boolean, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let values = values.iter().copied().map(u8::from).collect::<Vec<_>>();
        Array::from_host_buffer(session, r#type, mesh.clone(), values.as_slice()).unwrap()
    }

    /// Uploads a replicated Boolean test scalar using one byte per element.
    pub(crate) fn boolean_scalar<'c>(session: &Session<'c>, mesh: &DeviceMesh, value: bool) -> Array<'c> {
        let r#type = replicated_scalar_type(mesh, DataType::Boolean);
        Array::from_host_buffer(session, r#type, mesh.clone(), [u8::from(value)]).unwrap()
    }

    /// Reads the first addressable shard as native-endian `f32` values.
    pub(crate) fn read_f32s(client: &Client<'_>, array: &Array<'_>) -> Vec<f32> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<f32>(bytes.as_slice())
    }

    /// Reads every device shard of `array` in mesh device order and concatenates their native-endian `f32` values.
    pub(crate) fn read_sharded_f32_array(array: &Array<'_>) -> Vec<f32> {
        let mut values = Vec::new();
        for device in array.mesh().devices() {
            let shard_bytes = array
                .device_shard(device.id())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            values.extend(values_from_bytes::<f32>(shard_bytes.as_slice()));
        }
        values
    }

    /// Reads the first addressable shard as native-endian `f64` values.
    pub(crate) fn read_f64s(client: &Client<'_>, array: &Array<'_>) -> Vec<f64> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<f64>(bytes.as_slice())
    }

    /// Reads the first addressable shard as native-endian unsigned 64-bit integers.
    pub(crate) fn read_u64s(client: &Client<'_>, array: &Array<'_>) -> Vec<u64> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<u64>(bytes.as_slice())
    }

    /// Reads the first addressable shard as native-endian signed 64-bit integers.
    pub(crate) fn read_i64s(client: &Client<'_>, array: &Array<'_>) -> Vec<i64> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<i64>(bytes.as_slice())
    }

    /// Reads the first addressable shard, interpreting each nonzero byte as `true`.
    pub(crate) fn read_booleans(client: &Client<'_>, array: &Array<'_>) -> Vec<bool> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap()
            .into_iter()
            .map(|value| value != 0)
            .collect()
    }

    #[test]
    fn test_session_is_send_and_sync() {
        assert_send_sync::<Session<'static>>();
    }

    #[test]
    fn test_session_clone() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client).with_mesh(mesh.clone());
        let clone = session.clone();
        assert!(clone.is_same_session(&session));
        assert!(std::ptr::eq(clone.effect_scope(), session.effect_scope()));
        assert_eq!(clone.mesh().unwrap(), &mesh);

        // The shared state of a session lives for as long as any of its handles.
        let state = Arc::downgrade(&session.state);
        drop(session);
        assert!(state.upgrade().is_some());
        drop(clone);
        assert!(state.upgrade().is_none());
    }

    #[test]
    fn test_session_new() {
        // Sessions that are created with the default configuration run on the process-wide client of the selected
        // platform, and they share nothing else.
        let session = Session::new().unwrap();
        let other = Session::new().unwrap();
        assert!(!session.target().unwrap().devices().is_empty());
        assert!(std::ptr::eq(session.client(), other.client()));
        assert!(!session.is_same_session(&other));
        assert!(!std::ptr::eq(session.effect_scope(), other.effect_scope()));
    }

    #[test]
    fn test_session_builder() {
        // A builder without any configuration builds what `Session::new` creates.
        let session: Session<'static> = Session::builder().build().unwrap();
        assert!(std::ptr::eq(session.client(), Session::new().unwrap().client()));
        assert_eq!(session.default_compilation_options(), &CompilationOptions::default());
        assert!(session.compilation_context().disk_cache().is_none());
        assert_eq!(session.program_count(), 0);
        assert!(matches!(session.mesh(), Err(Error::Array(error)) if matches!(*error, ArrayError::MissingMesh)));
    }

    #[test]
    fn test_session_with_mesh() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let meshed = session.with_mesh(mesh.clone());
        assert_eq!(meshed.mesh().unwrap(), &mesh);
        assert!(meshed.is_same_session(&session));
        assert!(std::ptr::eq(meshed.effect_scope(), session.effect_scope()));

        // The handle that the mesh was added to is unaffected.
        assert!(matches!(session.mesh(), Err(Error::Array(error)) if matches!(*error, ArrayError::MissingMesh)));
    }

    #[test]
    fn test_session_fork_effect_scope() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client).with_mesh(mesh.clone());
        let fork = session.fork_effect_scope();
        assert!(fork.is_same_session(&session));
        assert!(!std::ptr::eq(fork.effect_scope(), session.effect_scope()));
        assert_eq!(fork.mesh().unwrap(), &mesh);

        // A barrier of the fork does not wait for effects that are pending in the scope that it was forked from.
        let pending = session.effect_scope().reserve(None).unwrap();
        fork.effects_barrier().unwrap();
        drop(pending);
        session.effects_barrier().unwrap();
    }

    #[test]
    fn test_session_client() {
        let client = execution_client();
        let session = Session::with_client(&client);
        assert!(std::ptr::eq(session.client(), &client));
        assert!(std::ptr::eq(session.fork_effect_scope().client(), &client));
    }

    #[test]
    fn test_session_target() {
        let client = cpu_client(2);
        let session = Session::with_client(&client);
        let target = session.target().unwrap();
        assert_eq!(target.platform_name(), client.platform_name().unwrap());
        assert_eq!(target.devices().len(), 2);

        // The target is read once and shared by all handles of the session.
        assert!(std::ptr::eq(session.fork_effect_scope().target().unwrap(), target));
    }

    #[test]
    fn test_session_mesh() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        assert!(matches!(session.mesh(), Err(Error::Array(error)) if matches!(*error, ArrayError::MissingMesh)));
        assert_eq!(session.with_mesh(mesh.clone()).mesh().unwrap(), &mesh);
    }

    #[test]
    fn test_session_default_compilation_options() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let compilation_options =
            CompilationOptions { matrix_unit_operand_precision: Precision::Highest as i32, ..Default::default() };
        let session = Session::builder()
            .client(&client)
            .default_compilation_options(compilation_options.clone())
            .build()
            .unwrap();
        assert_eq!(Session::with_client(&client).default_compilation_options(), &CompilationOptions::default());

        // Every handle of a session returns the template that the session was built with.
        assert_eq!(session.default_compilation_options(), &compilation_options);
        assert_eq!(session.fork_effect_scope().default_compilation_options(), &compilation_options);
        assert_eq!(session.with_mesh(mesh).default_compilation_options(), &compilation_options);
    }

    #[test]
    fn test_session_compilation_context() {
        let client = execution_client();
        let session = Session::with_client(&client);
        assert!(std::ptr::eq(session.clone().compilation_context(), session.compilation_context()));
        assert!(std::ptr::eq(session.fork_effect_scope().compilation_context(), session.compilation_context()));
        assert!(!std::ptr::eq(Session::with_client(&client).compilation_context(), session.compilation_context()));
    }

    #[test]
    fn test_session_is_same_session() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        assert!(session.is_same_session(&session));
        assert!(session.is_same_session(&session.clone()));
        assert!(session.is_same_session(&session.fork_effect_scope()));
        assert!(session.is_same_session(&session.with_mesh(mesh)));
        assert!(!session.is_same_session(&Session::with_client(&client)));
    }

    #[test]
    fn test_session_program_count() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let vector = f32_vector(&session, &mesh, &[1.0, 2.0]);
        assert_eq!(session.program_count(), 0);
        Neg::neg(&vector).unwrap();
        assert_eq!(session.program_count(), 1);

        // Repeating an eager operation compiles nothing, while an input of another type compiles another program.
        Neg::neg(&vector).unwrap();
        assert_eq!(session.program_count(), 1);
        Neg::neg(&f32_vector(&session, &mesh, &[1.0, 2.0, 3.0])).unwrap();
        assert_eq!(session.program_count(), 2);
    }

    #[test]
    fn test_session_eager_dispatch_statistics() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let vector = f32_vector(&session, &mesh, &[1.0, 2.0]);
        let statistics = session.eager_dispatch_statistics();
        assert_eq!((statistics.hits, statistics.misses), (0, 0));
        Neg::neg(&vector).unwrap();
        Neg::neg(&vector).unwrap();
        let statistics = session.eager_dispatch_statistics();
        assert_eq!((statistics.hits, statistics.misses), (1, 1));
    }

    #[test]
    fn test_session_clear_cache() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let vector = f32_vector(&session, &mesh, &[1.0, 2.0]);
        Neg::neg(&vector).unwrap();
        assert_eq!((session.eager_dispatch_cache().len(), session.program_count()), (1, 1));

        // Clearing empties both the eager dispatch cache and the compilation cache, so the same operation misses again.
        session.clear_cache();
        assert_eq!((session.eager_dispatch_cache().len(), session.program_count()), (0, 0));
        Neg::neg(&vector).unwrap();
        assert_eq!(session.eager_dispatch_statistics().misses, 2);
        assert_eq!((session.eager_dispatch_cache().len(), session.program_count()), (1, 1));
    }

    #[test]
    fn test_session_new_mesh() {
        let client = cpu_client(4);
        let session = Session::with_client(&client);
        let devices = session.target().unwrap().devices().iter().map(TargetDevice::device).collect::<Vec<_>>();
        let x = MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap();
        let y = MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap();

        // A mesh takes the first devices of the client that its axes require, in the order of the client.
        let mesh = session.new_mesh([x.clone(), y.clone()]).unwrap();
        assert_eq!(mesh.logical_mesh(), &LogicalMesh::new(vec![x.clone(), y]).unwrap());
        assert_eq!(mesh.devices(), devices.as_slice());
        let mesh = session.new_mesh([x.clone()]).unwrap();
        assert_eq!(mesh.devices(), &devices[..2]);

        // Creating a mesh does not make it the mesh of the session.
        assert!(matches!(session.mesh(), Err(Error::Array(error)) if matches!(*error, ArrayError::MissingMesh)));

        // Invalid axes and meshes that need more devices than the client has are rejected.
        assert!(matches!(
            session.new_mesh([x.clone(), x]),
            Err(Error::Core(error))
                if *error == CoreError::Sharding(ShardingError::DuplicateMeshAxisName { name: "x".to_string() }),
        ));
        assert!(matches!(
            session.new_mesh([MeshAxis::new("x", 8, MeshAxisType::Auto).unwrap()]),
            Err(Error::Core(error))
                if *error == CoreError::Sharding(ShardingError::DeviceCountMismatch { expected: 8, actual: 4 }),
        ));
    }

    #[test]
    fn test_session_array() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let r#type = replicated_vector_type(&mesh, 2);
        let array = session.array(r#type.clone(), mesh.clone(), values_to_bytes::<f32>(&[1.0, 2.0])).unwrap();
        assert_eq!(array.r#type().into_owned(), r#type);
        assert_eq!(array.mesh(), &mesh);
        assert!(array.session().is_same_session(&session));
        assert_eq!(values_from_bytes::<f32>(&array.copy_to_host().unwrap()), vec![1.0, 2.0]);
    }

    #[test]
    fn test_session_effects_barrier() {
        let client = execution_client();
        let session = Session::with_client(&client);
        session.effects_barrier().unwrap();

        // A barrier reports the failures of earlier submissions, and keeps reporting them until they are acknowledged.
        let mut reservation = session.effect_scope().reserve(None).unwrap();
        let completion = reservation.publish(Ok(None));
        reservation.commit();
        completion.complete(Err(Arc::from("effect failed")));
        assert!(matches!(
            session.effects_barrier(),
            Err(Error::Execution(ExecutionError::EffectSubmissionFailures { failures }))
                if failures == vec![(0, Arc::from("effect failed"))],
        ));
        assert!(matches!(
            session.effects_barrier(),
            Err(Error::Execution(ExecutionError::EffectSubmissionFailures { failures }))
                if failures == vec![(0, Arc::from("effect failed"))],
        ));
    }

    #[test]
    fn test_session_acknowledge_effect_errors() {
        let client = execution_client();
        let session = Session::with_client(&client);
        let mut reservation = session.effect_scope().reserve(None).unwrap();
        let completion = reservation.publish(Ok(None));
        reservation.commit();
        completion.complete(Err(Arc::from("effect failed")));

        // Failures can only be acknowledged after a barrier has reported them.
        assert!(matches!(
            session.acknowledge_effect_errors(),
            Err(Error::Execution(ExecutionError::PrematureAcknowledgment)),
        ));
        assert!(matches!(
            session.effects_barrier(),
            Err(Error::Execution(ExecutionError::EffectSubmissionFailures { failures }))
                if failures == vec![(0, Arc::from("effect failed"))],
        ));
        session.acknowledge_effect_errors().unwrap();
        session.effects_barrier().unwrap();
    }

    #[test]
    fn test_session_retained_constant() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let array = session.array(ArrayType::scalar(DataType::F32), mesh, 1f32.to_ne_bytes()).unwrap();
        let handle = session.retain_constant(&array);

        // The resolved array shares the retained storage and belongs to the resolving handle.
        let fork = session.fork_effect_scope();
        let resolved = fork.retained_constant(&handle).unwrap();
        assert_eq!(resolved, array);
        assert!(std::ptr::eq(resolved.session().effect_scope(), fork.effect_scope()));

        // Handles resolve only through the session that retains their arrays.
        let other_session = Session::with_client(&client);
        assert_eq!(
            other_session.retained_constant(&handle),
            Err(TypeError::invalid(format!(
                "session constant `{handle}` belongs to session #{} but was resolved by session #{}",
                session.state.id, other_session.state.id,
            ))
            .into()),
        );

        // Handles whose entries were reclaimed and reused by another array no longer resolve.
        let stale =
            SessionConstant::new(session.state.id, 0, handle.r#type().into_owned(), Arc::new(SessionConstantOwner));
        assert_eq!(
            session.retained_constant(&stale),
            Err(TypeError::invalid(format!("session constant `{stale}` is not in the constant table of its session"))
                .into()),
        );
    }

    #[test]
    fn test_session_retain_constant() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let array = |value: f32| session.array(ArrayType::scalar(DataType::F32), mesh.clone(), value.to_ne_bytes());
        let retained_constant_count = || session.state.constants.lock().unwrap().iter().flatten().count();

        // Retaining one array twice shares one entry and one ownership token, and the handles keep the array alive
        // after the array itself is dropped.
        let first = array(1.0).unwrap();
        let handle = session.retain_constant(&first);
        let duplicate = session.retain_constant(&first);
        assert_eq!(duplicate, handle);
        assert!(Arc::ptr_eq(duplicate.owner(), handle.owner()));
        drop((first, duplicate));
        assert_eq!(retained_constant_count(), 1);
        assert_eq!(session.retained_constant(&handle).unwrap(), session.retained_constant(&handle).unwrap());

        // Once every handle is released, the next retention reclaims the entry and reuses its position.
        drop(handle);
        let second = array(2.0).unwrap();
        let handle = session.retain_constant(&second);
        assert_eq!(retained_constant_count(), 1);
        assert_eq!(handle.index(), 0);
        assert_eq!(session.retained_constant(&handle).unwrap(), second);
    }

    #[test]
    fn test_session_named_axis() {
        let client = execution_client();
        assert!(Session::with_client(&client).named_axis("x").is_none());
    }

    #[test]
    fn test_session_named_axes() {
        let client = execution_client();
        assert!(Session::with_client(&client).named_axes().is_empty());
    }

    #[test]
    fn test_session_default_mesh() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        assert_eq!(session.default_mesh(), None);
        assert_eq!(session.with_mesh(mesh.clone()).default_mesh(), Some(mesh.logical_mesh().clone()));
    }

    #[test]
    fn test_session_error_display() {
        assert_eq!(
            SessionError::UnavailablePlatform { platform: Platform::Tpu, feature: "tpu" }.to_string(),
            "platform `tpu` is unavailable because `ryft-xla` was built without its `tpu` feature",
        );
        assert_eq!(
            SessionError::InvalidClientOptions { reason: "reason".to_string() }.to_string(),
            "invalid client options: reason",
        );
    }

    #[test]
    fn test_session_builder_platform() {
        // Sessions that `ryft-xla` creates clients for can be `'static`, and they share the client of their platform
        // without sharing any session state.
        let session: Session<'static> = Session::builder().platform(Platform::Cpu).build().unwrap();
        let other: Session<'static> = Session::builder().platform(Platform::Cpu).build().unwrap();
        assert!(session.target().unwrap().platform_name().eq_ignore_ascii_case("cpu"));
        assert!(std::ptr::eq(session.client(), other.client()));
        assert!(!session.is_same_session(&other));
    }

    #[cfg(not(feature = "tpu"))]
    #[test]
    fn test_session_builder_platform_rejects_platforms_without_their_feature() {
        assert!(matches!(
            Session::builder().platform(Platform::Tpu).build(),
            Err(Error::Session(SessionError::UnavailablePlatform { platform: Platform::Tpu, feature: "tpu" })),
        ));
    }

    #[test]
    fn test_session_builder_plugin() {
        // A plugin that the caller loaded creates a process-wide client like a platform does, and it shares the client
        // of its platform when their client options are equal.
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() });
        let session = Session::builder()
            .plugin(load_cpu_plugin().unwrap())
            .client_options(options.clone())
            .build()
            .unwrap();
        let other = Session::builder().platform(Platform::Cpu).client_options(options).build().unwrap();
        assert_eq!(session.target().unwrap().devices().len(), 2);
        assert!(std::ptr::eq(session.client(), other.client()));
    }

    #[test]
    fn test_session_builder_client_options() {
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() });
        let session = Session::builder().platform(Platform::Cpu).client_options(options.clone()).build().unwrap();
        let default_session = Session::builder().platform(Platform::Cpu).build().unwrap();
        assert_eq!(session.target().unwrap().devices().len(), 4);
        assert!(!std::ptr::eq(session.client(), default_session.client()));

        // Client options cannot configure a client that already exists, and they require an explicit platform.
        let client = cpu_client(1);
        assert!(matches!(
            Session::builder().client(&client).client_options(options.clone()).build(),
            Err(Error::Session(SessionError::InvalidClientOptions { reason }))
                if reason == "client options only configure a client that the session creates, but the session was \
                              given an existing client",
        ));
        assert!(matches!(
            Session::builder().client_options(options).build(),
            Err(Error::Session(SessionError::InvalidClientOptions { reason }))
                if reason == "client options require an explicit platform or plugin, because the platforms that \
                              `Platform::Auto` selects among take different options",
        ));
    }

    #[test]
    fn test_session_builder_client() {
        // A session over a client of the caller borrows it, and the last source that a builder is given wins.
        let client = cpu_client(1);
        let session = Session::builder().platform(Platform::Cpu).client(&client).build().unwrap();
        assert!(std::ptr::eq(session.client(), &client));
        let session = Session::builder().client(&client).platform(Platform::Cpu).build().unwrap();
        assert!(!std::ptr::eq(session.client(), &client));
    }

    #[test]
    fn test_session_builder_compilation_context() {
        // The session compiles through the supplied context, whose in-memory tier here retains a single program.
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::builder()
            .client(&client)
            .compilation_context(CompilationContext::with_capacity(1))
            .build()
            .unwrap();
        Neg::neg(&f32_vector(&session, &mesh, &[1.0, 2.0])).unwrap();
        Neg::neg(&f32_vector(&session, &mesh, &[1.0, 2.0, 3.0])).unwrap();
        assert_eq!(session.program_count(), 1);
        assert_eq!(session.compilation_context().statistics().compilations, 2);
    }

    #[test]
    fn test_session_builder_disk_cache() {
        let client = execution_client();
        let directory = tempfile::tempdir().unwrap();
        let session = Session::builder()
            .client(&client)
            .disk_cache(DiskCache::open(directory.path()).unwrap())
            .build()
            .unwrap();
        let expected_directory = DiskCache::open(directory.path()).unwrap().directory().to_path_buf();
        assert_eq!(session.compilation_context().disk_cache().unwrap().directory(), expected_directory);

        // An explicit disk cache replaces the one of a supplied compilation context, irrespective of their order.
        let context_directory = tempfile::tempdir().unwrap();
        let session = Session::builder()
            .client(&client)
            .disk_cache(DiskCache::open(directory.path()).unwrap())
            .compilation_context(
                CompilationContext::new().with_disk_cache(DiskCache::open(context_directory.path()).unwrap()),
            )
            .build()
            .unwrap();
        assert_eq!(session.compilation_context().disk_cache().unwrap().directory(), expected_directory);
    }

    #[test]
    fn test_session_builder_disk_cache_from_env() {
        // The session adds the disk cache that the environment configures, if any, when it is built. This test does not
        // modify the environment of the test process, and so it compares against whatever the environment configures.
        let client = execution_client();
        let session = Session::builder().client(&client).disk_cache_from_env().build().unwrap();
        assert_eq!(
            session.compilation_context().disk_cache().map(|disk_cache| disk_cache.directory().to_path_buf()),
            DiskCache::from_env().unwrap().map(|disk_cache| disk_cache.directory().to_path_buf()),
        );
    }

    #[test]
    fn test_session_builder_runtime_extensions() {
        // The supplied configuration replaces the built-in providers, so a session with an empty configuration cannot
        // satisfy a requirement that the default configuration satisfies.
        let client = execution_client();
        let requirement = builtin_requirement("ryft.ordered-io");
        let session = Session::with_client(&client);
        session.runtimes().ensure(std::slice::from_ref(&requirement), session.target().unwrap()).unwrap();
        let session =
            Session::builder().client(&client).runtime_extensions(RuntimeExtensions::empty()).build().unwrap();
        assert!(matches!(
            session.runtimes().ensure(&[requirement], session.target().unwrap()),
            Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == "ryft.ordered-io",
        ));
    }

    #[test]
    fn test_session_builder_default_compilation_options() {
        let client = execution_client();
        let compilation_options = CompilationOptions { compile_portable_executable: true, ..Default::default() };
        let session = Session::builder()
            .client(&client)
            .default_compilation_options(compilation_options.clone())
            .build()
            .unwrap();
        assert_eq!(session.default_compilation_options(), &compilation_options);
    }

    #[test]
    fn test_session_builder_build() {
        // Every build creates an independent session with a fresh effect scope and empty caches, even on one client.
        let client = execution_client();
        let session = Session::builder().client(&client).build().unwrap();
        let other = Session::builder().client(&client).build().unwrap();
        assert!(std::ptr::eq(session.client(), other.client()));
        assert!(!session.is_same_session(&other));
        assert!(!std::ptr::eq(session.effect_scope(), other.effect_scope()));
        assert_eq!((session.program_count(), session.eager_dispatch_cache().len()), (0, 0));
    }

    #[test]
    fn test_session_constant_session() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let array = session.array(ArrayType::scalar(DataType::F32), mesh, 1f32.to_ne_bytes()).unwrap();
        assert_eq!(session.retain_constant(&array).session(), session.state.id);
    }

    #[test]
    fn test_session_constant_index() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let first = session.array(ArrayType::scalar(DataType::F32), mesh.clone(), 1f32.to_ne_bytes()).unwrap();
        let second = session.array(ArrayType::scalar(DataType::F32), mesh, 2f32.to_ne_bytes()).unwrap();
        let handles = (session.retain_constant(&first), session.retain_constant(&second));
        assert_eq!((handles.0.index(), handles.1.index()), (0, 1));
    }

    #[test]
    fn test_session_constant_rename_type_identities() {
        // Renaming keeps the session, the position, and the ownership token of a handle, and only renames its type.
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let array = session.array(ArrayType::scalar(DataType::F32), mesh, 1f32.to_ne_bytes()).unwrap();
        let handle = session.retain_constant(&array);
        let renamed = handle.rename_type_identities(&TypeIdentityRenaming::new()).unwrap();
        assert_eq!(renamed, handle);
        assert!(Arc::ptr_eq(renamed.owner(), handle.owner()));
    }

    #[test]
    fn test_session_constant_eq() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let other_session = Session::with_client(&client);
        let first = session.array(ArrayType::scalar(DataType::F32), mesh.clone(), 1f32.to_ne_bytes()).unwrap();
        let second = session.array(ArrayType::scalar(DataType::F32), mesh.clone(), 2f32.to_ne_bytes()).unwrap();
        let other = other_session.array(ArrayType::scalar(DataType::F32), mesh, 1f32.to_ne_bytes()).unwrap();
        let handle = session.retain_constant(&first);

        // Handles are equal when they refer to the same position of the same session with the same type.
        assert_eq!(handle, handle.clone());
        assert_eq!(handle, session.retain_constant(&first));
        assert_ne!(handle, session.retain_constant(&second));
        assert_ne!(handle, other_session.retain_constant(&other));
    }

    #[test]
    fn test_session_constant_hash() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let first = session.array(ArrayType::scalar(DataType::F32), mesh.clone(), 1f32.to_ne_bytes()).unwrap();
        let second = session.array(ArrayType::scalar(DataType::F32), mesh, 2f32.to_ne_bytes()).unwrap();
        let handle = session.retain_constant(&first);
        let other = session.retain_constant(&second);
        assert_eq!(hash_of(&handle), hash_of(&session.retain_constant(&first)));
        assert_ne!(hash_of(&handle), hash_of(&other));
        let constants = HashMap::from([(handle.clone(), "first"), (other.clone(), "second")]);
        assert_eq!(constants.get(&session.retain_constant(&first)), Some(&"first"));
        assert_eq!(constants.get(&other), Some(&"second"));
    }

    #[test]
    fn test_session_constant_display() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let array = session.array(ArrayType::scalar(DataType::F32), mesh, 1f32.to_ne_bytes()).unwrap();
        let handle = session.retain_constant(&array);
        assert_eq!(handle.to_string(), "session#0:f32[][sharding={mesh<['x'=1:auto]>, []}]");
    }

    #[test]
    fn test_session_constant_type() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let array = session
            .array(replicated_scalar_type(&mesh, DataType::F32), mesh.clone(), 1f32.to_ne_bytes())
            .unwrap();
        assert_eq!(session.retain_constant(&array).r#type().into_owned(), replicated_scalar_type(&mesh, DataType::F32));
    }

    #[test]
    fn test_session_constant() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let array = session.array(ArrayType::scalar(DataType::F32), mesh, 1f32.to_ne_bytes()).unwrap();

        // Arrays are retained by their session and become handles to their retained storage.
        let Ok(Constant::Session(handle)) = session_constant(Value::Array(array.clone())) else {
            panic!("an array was not converted into a session constant");
        };
        assert_eq!(session.retained_constant(&handle).unwrap(), array);

        // Dimensions become immediate constants, and references are rejected.
        let extent = DimensionValue::constant(2).unwrap();
        assert_eq!(
            session_constant(Value::Dimension(Dimension::new(extent.clone(), session.clone()))),
            Ok(Constant::Dimension(extent)),
        );
        assert_eq!(
            session_constant(Value::Reference(Reference::new(ArrayReference::new(array), session))),
            Err(TypeError::invalid("cannot import an xla reference into a trace whose capture table holds constants")
                .into()),
        );
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_session_cuda_kernel_execution_and_persistence() {
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client);
        let definition: KernelDefinition =
            KernelDefinition::trace(crate::kernels::tests::definition().operation().clone(), |(references, _)| {
                let value = ProjectedValue::new(references[0].read()?, ArrayType::scalar(DataType::I32));
                references[0].write(value.add(&value.one_like()?)?.value())?;
                Ok(())
            })
            .unwrap();
        // The observable scalar increment must execute through the handler. Keeping inputs live verifies that XLA
        // preserves functional array semantics while supplying an initialized aliased writable result.
        let artifact = CudaKernelArtifact::new(
            CudaArtifactFormat::Ptx,
            b".version 7.0\n.target sm_80\n.address_size 64\n.visible .entry increment(.param .u64 data) {\n\
              .reg .b64 pointer;\n.reg .b32 value;\nld.param.u64 pointer, [data];\nld.global.u32 value, [pointer];\n\
              add.u32 value, value, 1;\nst.global.u32 [pointer], value;\nret;\n}\n"
                .to_vec(),
            "increment",
            "compute_80",
            CudaKernelLaunchDimensions::new([1; 3], [1; 3], 0).unwrap(),
            CudaKernelAbi::new("ryft.kernel.pointer", 1, vec![CudaKernelParameterType::DevicePointer]).unwrap(),
        )
        .unwrap();

        /// Hand-authored artifact fixture admitted only for the exact verified scalar increment body.
        struct IncrementCompiler {
            artifact: CudaKernelArtifact,
            semantic: String,
        }

        /// PTX fixture target requiring the actual CUDA platform, FFI ABI, and supported device architecture.
        struct IncrementTarget;

        impl crate::kernels::KernelTarget for IncrementTarget {
            fn admit_execution(
                &self,
                facts: &crate::kernels::KernelExecutionFacts,
            ) -> Result<(), crate::kernels::KernelEmbeddingError> {
                let supported_devices = !facts.devices().is_empty()
                    && facts.devices().iter().all(|device| match device.attributes().get("compute_capability") {
                        Some(ryft_pjrt::Value::String(capability)) => capability
                            .split('.')
                            .next()
                            .and_then(|major| major.parse::<usize>().ok())
                            .is_some_and(|major| major >= 8),
                        _ => false,
                    });
                if !facts.platform_name().eq_ignore_ascii_case("cuda")
                    || !facts.has_ffi_extension()
                    || facts.pjrt_version().major != ryft_pjrt::VERSION.major
                    || facts.pjrt_version().minor < ryft_pjrt::VERSION.minor
                    || !supported_devices
                {
                    return Err(crate::kernels::KernelEmbeddingError::Invalid {
                        message:
                            "increment fixture requires CUDA compute capability 8.0 or newer and compatible PJRT FFI"
                                .to_owned(),
                    });
                }
                Ok(())
            }
        }

        impl KernelCompiler for IncrementCompiler {
            type Target = IncrementTarget;
            type Options = ();
            type Output = CudaKernelArtifact;
            type Error = Infallible;

            fn admit(
                &self,
                kernel: &VerifiedKernel<'_>,
                target: &IncrementTarget,
                options: &(),
                schedule: &KernelSchedule,
            ) -> Result<(), KernelCompilationError<Infallible>> {
                let _ = (target, options, schedule);
                if kernel.definition().semantic_key().unwrap() != self.semantic {
                    return Err(KernelCompilationError::Incompatible {
                        message: "increment fixture body changed".to_owned(),
                    });
                }
                Ok(())
            }

            fn configuration_key(
                &self,
                target: &IncrementTarget,
                options: &(),
                schedule: &KernelSchedule,
            ) -> Result<Vec<u8>, KernelCompilationError<Infallible>> {
                let _ = (target, options, schedule);
                let mut key = b"increment-fixture-v1;compute_80;pointer-abi-v1;grid1;block1;shared0".to_vec();
                key.extend_from_slice(self.artifact.bytes());
                Ok(key)
            }

            fn compile(
                &self,
                kernel: &VerifiedKernel<'_>,
                target: &IncrementTarget,
                options: &(),
                schedule: &KernelSchedule,
            ) -> Result<CudaKernelArtifact, KernelCompilationError<Infallible>> {
                let _ = (kernel, target, options, schedule);
                Ok(self.artifact.clone())
            }
        }
        let compiler = KernelCompilerBinding::new(
            IncrementCompiler { artifact, semantic: definition.semantic_key().unwrap() },
            IncrementTarget,
            (),
            KernelSchedule::default(),
            CudaKernelEmbedding::new(CUDA_KERNEL_CUSTOM_CALL_TARGET.to_owned(), vec![0]),
            1,
        )
        .unwrap();
        let scalar = ArrayType::scalar(DataType::I32);
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel(inputs[0].context(), &definition, &inputs)
            },
            definition.operation().input_types(),
        )
        .unwrap();
        assert!(program.effects().classes().contains(EffectClass::OrderedState));
        let options = Options::new(mesh.clone()).with_kernel_compiler(compiler);
        let lowered = session.lower_program(&program, 0, &options).unwrap();
        assert!(
            lowered
                .runtime_requirements()
                .iter()
                .any(|requirement| requirement.identifier() == "ryft.kernel.cuda")
        );
        assert!(lowered.kernel_execution_facts().is_some());
        assert!(!lowered.signature().has_effects());
        let compiled = session.compile_program(&lowered).unwrap();
        let input = session.array(scalar, mesh, 42i32.to_ne_bytes()).unwrap();
        let outputs = session.execute_program(&compiled, vec![input.clone()]).unwrap();
        let bytes = compiled.serialize().unwrap().unwrap();
        let restored_session = Session::with_client(&client);
        let restored = restored_session.deserialize_compiled_program(&bytes).unwrap().unwrap();
        assert!(
            restored
                .runtime_requirements()
                .iter()
                .any(|requirement| requirement.identifier() == "ryft.kernel.cuda")
        );
        let restored_outputs = restored_session.execute_program(&restored, vec![outputs[0].clone()]).unwrap();
        for (array, expected) in [(&input, 42), (&outputs[0], 43), (&restored_outputs[0], 44)] {
            assert_eq!(values_from_bytes::<i32>(&array.copy_to_host().unwrap()), vec![expected]);
        }
        drop(outputs);
        drop(restored_outputs);
        // Both sessions retain their native launchers until the final queued execution completes, even though
        // neither this call's outputs nor the session's public handles are retained by the caller.
        drop(session.execute_program(&compiled, vec![input.clone()]).unwrap());
        drop(restored_session.execute_program(&restored, vec![input.clone()]).unwrap());
        drop(input);
        drop(restored_session);
        drop(session);
    }
}
