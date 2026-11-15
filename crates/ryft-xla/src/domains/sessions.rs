//! Sessions, which own the resources that domains share when they execute on one PJRT client.
//!
//! A [`Session`] borrows one externally owned PJRT client and owns its compilation cache, its eager dispatch cache,
//! its default effect scope, its runtime extension instances, and a table of retained constant arrays. Every
//! [`Domain`] derived from a session through [`Session::domain`] shares these resources, while separate sessions stay
//! independent even when they borrow the same client.
//!
//! Traces whose capture tables hold constants rather than runtime values refer to runtime arrays through
//! [`SessionConstant`] handles. [`ThroughSession`] imports runtime [`Value`]s into such traces by retaining their
//! arrays in the session, which keeps each array alive for as long as any handle to it exists.

use std::borrow::Cow;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use ryft_core::{
    ArrayType, CapturingContext, CompilationContext, Context, DeviceMesh, DiskCache, ImportingContext,
    NestedTracingContext, Parameter, ProgramError, SpecializationCache, TracingContext, Type, TypeError,
    TypeIdentityRenaming, Typed,
};
use ryft_macros::Parameter;
use ryft_pjrt::Client;

use crate::arrays::RetainedArray;
use crate::compilation::keys::EagerDispatchKey;
use crate::domains::Domain;
use crate::domains::eager::{EAGER_DISPATCH_CACHE_CAPACITY, EagerDispatchProgram};
use crate::domains::effects::EffectScope;
use crate::domains::runtimes::{RuntimeExtensions, RuntimeRegistry};
use crate::domains::targets::Target;
use crate::{Array, Constant, Error, Operation, Value};

/// Source of the [`Session::id`]s, which are unique within the process.
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(0);

// TODO(eaplatanios): Review this module.

/// Shared compilation and effect resources for execution on one externally owned PJRT [`Client`].
///
/// A session is the handle through which `ryft-xla` executes on a client. Every [`Domain`] belongs to exactly one
/// session, and every [`Array`] belongs to exactly one domain. Create one session per client and derive every domain
/// from it (i.e., through [`Self::domain`], [`Domain::with_compilation_options`], or [`Domain::fork_effect_scope`]) so
/// that all arrays and domains on that client share one compilation cache, one default effect scope, and one set of
/// runtime extensions, much like JAX's process-wide caches. Separate sessions remain independent, even on the same
/// client, so creating a second session is an explicit request for an isolated cache and effect ordering. Cached
/// executables retain neither sessions nor arrays, so the ownership graph has no cycle between caches and sessions.
/// Call [`Self::effects_barrier`] before dropping the client when its effects must be observed. Destruction waits for
/// launches borrowing extension resources before releasing those resources. Ordinary execution adds no implicit
/// destruction barrier, and backend behavior with unfinished device work is specific to each plugin.
///
/// # Compilation Caching
///
/// Each session owns one [`CompilationContext`], which its constructors configure. [`Self::new`] keeps compiled
/// programs only in memory, and [`Self::with_cache_capacity`] bounds that in-memory tier. [`Self::with_disk_cache`]
/// adds a persistent [`DiskCache`] so that later processes restart warm, and [`Self::with_disk_cache_from_env`] adds
/// the one that the [`DiskCache::ENV_VAR`] environment variable configures, which sessions never read otherwise.
/// Multi-process programs share compiled artifacts through a context with an exchange from
/// [`DistributedRuntime::compilation_artifact_exchange`](crate::DistributedRuntime::compilation_artifact_exchange),
/// which they pass to [`Self::with_compilation_context`]. Refer to [`CompilationContext`] for a configuration
/// example.
///
/// # Example
///
/// ```no_run
/// use ryft_core::{ArrayType, DataType, DeviceMesh, Print};
/// use ryft_pjrt::Client;
/// use ryft_xla::Session;
///
/// fn print_values<'c>(client: &'c Client<'c>, mesh: DeviceMesh) -> Result<(), Box<dyn std::error::Error>> {
///     let session = Session::new(client);
///     let value = session.array(ArrayType::scalar(DataType::F64), mesh, 1.0f64.to_ne_bytes())?;
///     value.print("value");
///     session.effects_barrier()?;
///     Ok(())
/// }
/// ```
///
/// A session cannot outlive its client.
///
/// ```compile_fail
/// use ryft_xla::Session;
/// fn escaped(client: ryft_pjrt::Client<'_>) {
///     let session = Session::new(&client);
///     drop(client);
///     session.effects_barrier().unwrap();
/// }
/// ```
pub struct Session<'c> {
    /// Identifier of this session, unique within the process, which [`SessionConstant`] handles use to refer to it.
    id: u64,

    /// Externally owned client, which outlives cached executables and effect carriers.
    client: &'c Client<'c>,

    /// Compilation and execution target snapshot of [`Self::client`], read lazily and retained for the lifetime of the
    /// session, including a failed read.
    target: OnceLock<Result<Target, Error>>,

    /// Compilation cache of this session, which [`Self::compilation_context`] exposes. Cached artifacts contain no
    /// runtime execution association.
    cache: CompilationContext<'c, Domain<'c>>,

    /// Compiled programs of eager operation applications, keyed ahead of program construction and lowering. Refer to
    /// the documentation of [`EagerDispatchKey`] for information on what the key captures.
    eager_dispatch_cache: SpecializationCache<EagerDispatchKey, EagerDispatchProgram<'c>>,

    /// Default ordering and completion scope shared by domains and arrays created through this session.
    default_effect_scope: Arc<EffectScope<'c>>,

    /// Runtime extensions associated with the client, and ownership of launches borrowing their resources.
    runtimes: RuntimeRegistry<'c>,

    /// Arrays that [`SessionConstant`] handles refer to, indexed by [`SessionConstant::index`]. A [`None`] entry was
    /// reclaimed and may be reused. Refer to [`Self::retain_constant`] for information on how arrays enter and leave
    /// this table.
    constants: Mutex<Vec<Option<RetainedConstant<'c>>>>,
}

impl<'c> Session<'c> {
    /// Creates an independent session with an empty in-memory compilation cache and a fresh default effect scope.
    pub fn new(client: &'c Client<'c>) -> Arc<Self> {
        Self::with_compilation_context(client, CompilationContext::new())
    }

    /// Creates a session with an explicitly configured compilation cache.
    pub fn with_compilation_context(client: &'c Client<'c>, cache: CompilationContext<'c, Domain<'c>>) -> Arc<Self> {
        Self::with_runtime_extensions(client, cache, RuntimeExtensions::new())
    }

    /// Creates a session with an explicit compilation cache and immutable runtime provider configuration.
    ///
    /// Initialized instances belong to this session, even when `extensions` was cloned from another configuration.
    /// Providers initialize lazily when an executable requires them, so this constructor does not load GPU drivers.
    pub fn with_runtime_extensions(
        client: &'c Client<'c>,
        cache: CompilationContext<'c, Domain<'c>>,
        extensions: RuntimeExtensions<'c>,
    ) -> Arc<Self> {
        Arc::new(Self {
            id: NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed),
            client,
            target: OnceLock::new(),
            cache,
            eager_dispatch_cache: SpecializationCache::new(EAGER_DISPATCH_CACHE_CAPACITY),
            default_effect_scope: Arc::new(EffectScope::default()),
            runtimes: RuntimeRegistry::new(client, extensions),
            constants: Mutex::new(Vec::new()),
        })
    }

    /// Creates a session whose in-memory compilation cache retains at most `capacity` compiled programs. A `capacity`
    /// of zero is clamped to one entry.
    pub fn with_cache_capacity(client: &'c Client<'c>, capacity: usize) -> Arc<Self> {
        Self::with_compilation_context(client, CompilationContext::with_capacity(capacity))
    }

    /// Creates a session whose compilation cache also writes through to `disk_cache`, with its size cap and write
    /// thresholds.
    pub fn with_disk_cache(client: &'c Client<'c>, disk_cache: DiskCache) -> Arc<Self> {
        Self::with_compilation_context(client, CompilationContext::new().with_disk_cache(disk_cache))
    }

    /// Creates a session whose compilation cache also writes through to a [`DiskCache`] configured via the
    /// [`DiskCache::ENV_VAR`](ryft_core::compilation::DiskCache::ENV_VAR) environment variable, if it is set. An absent
    /// variable produces a cache that only lives in memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the configured cache directory is invalid.
    pub fn with_disk_cache_from_env(client: &'c Client<'c>) -> Result<Arc<Self>, Error> {
        Ok(Self::with_compilation_context(client, CompilationContext::new().with_disk_cache_from_env()?))
    }

    /// Returns the externally owned PJRT client.
    pub fn client(&self) -> &'c Client<'c> {
        self.client
    }

    /// Returns the [`Target`] of this session's client, reading it from the client on first use.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if querying the platform, plugin, or device facts from the client failed. The
    /// failure is retained, so every later call returns the same error.
    pub fn target(&self) -> Result<&Target, Error> {
        self.target.get_or_init(|| Target::from_client(self.client)).as_ref().map_err(Clone::clone)
    }

    /// Returns the shared compilation cache.
    pub fn compilation_context(&self) -> &CompilationContext<'c, Domain<'c>> {
        &self.cache
    }

    /// Returns the runtime extension instances of this session and the launches that still borrow their resources.
    pub(crate) fn runtimes(&self) -> &RuntimeRegistry<'c> {
        &self.runtimes
    }

    /// Returns the compiled programs of eager operation applications, keyed ahead of program construction and lowering.
    /// Refer to the documentation of [`EagerDispatchKey`] for information on what the key captures.
    #[inline]
    pub(super) fn eager_dispatch_cache(&self) -> &SpecializationCache<EagerDispatchKey, EagerDispatchProgram<'c>> {
        &self.eager_dispatch_cache
    }

    /// Returns the default ordering and completion scope shared by domains and arrays created through this session.
    #[inline]
    pub(super) fn default_effect_scope(&self) -> &Arc<EffectScope<'c>> {
        &self.default_effect_scope
    }

    /// Creates a domain sharing this session's cache and default scope, with the default
    /// [`CompilationOptions`](ryft_pjrt::protos::CompilationOptions) and no domain mesh.
    pub fn domain(self: &Arc<Self>) -> Domain<'c> {
        Domain::from_session_configuration(Arc::clone(self), None, Domain::default_compilation_options())
    }

    /// Uploads host bytes into an array that belongs to [`Self::domain`]. Refer to the documentation of
    /// [`Array::from_host_buffer`] for more information.
    pub fn array<B: AsRef<[u8]>>(
        self: &Arc<Self>,
        r#type: ArrayType,
        mesh: DeviceMesh,
        buffer: B,
    ) -> Result<Array<'c>, Error> {
        Array::from_host_buffer(&self.domain(), r#type, mesh, buffer)
    }

    /// Waits for the default scope's previously reserved effects, including unordered I/O and local assertions.
    /// Later submissions and forked scopes are excluded. Failures are retained until explicitly acknowledged.
    pub fn effects_barrier(&self) -> Result<(), Error> {
        self.default_effect_scope.barrier()
    }

    /// Acknowledges failures reported by a completed barrier once their affected ordering chain is quiescent.
    /// This never waits, repairs reference state, or changes errors retained by existing arrays or executions.
    pub fn acknowledge_effect_errors(&self) -> Result<(), Error> {
        self.default_effect_scope.acknowledge()
    }

    /// Returns a new [`Array`] that belongs to `domain` and shares the storage that `constant` refers to.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Type`] error if `constant` was not retained by this session or is no longer in the
    /// constant table of this session.
    pub(crate) fn constant(&self, domain: &Domain<'c>, constant: &SessionConstant) -> Result<Array<'c>, ProgramError> {
        if constant.session() != self.id {
            return Err(TypeError::invalid(format!(
                "session constant `{constant}` belongs to session #{} but was resolved by session #{}",
                constant.session(),
                self.id,
            ))
            .into());
        }
        let constants = self.constants.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let retained = constants
            .get(constant.index())
            .and_then(Option::as_ref)
            .filter(|retained| std::ptr::eq(retained.owner.as_ptr(), Arc::as_ptr(constant.owner())))
            .ok_or_else(|| {
                TypeError::invalid(format!("session constant `{constant}` is not in the constant table of its session"))
            })?;
        Ok(retained.storage.to_array(domain))
    }

    /// Retains the storage of `array` and returns a [`SessionConstant`] handle that refers to it. The storage stays
    /// retained for as long as any handle to it exists, as the ownership section of the [`SessionConstant`]
    /// documentation describes. Retaining an array whose storage a live handle already refers to returns a handle to
    /// the same entry, so that repeatedly importing the same captured array (e.g., the weights that a compiled function
    /// captures) does not grow the table. Entries without handles are reclaimed here, which releases their storage and
    /// makes their positions reusable.
    fn retain_constant(&self, array: &Array<'c>) -> SessionConstant {
        let mut constants = self.constants.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
            return SessionConstant::new(self.id, index, r#type, owner);
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
        SessionConstant::new(self.id, index, r#type, owner)
    }
}

/// Handle without a lifetime to an array that a [`Session`] retains on behalf of a trace whose capture table cannot
/// hold runtime arrays. Such a trace (e.g., the body of a [`rematerialize`](ryft_core::rematerialize) call or of a
/// [`composite`](ryft_core::composite) and its rules, which are traced over constants alone) still needs to refer to
/// the runtime arrays that a compiled function it calls has captured. Importing such an array through
/// [`ThroughSession`] registers it in the session and stages this handle as an immediate constant, which can appear in
/// any region of any program, including lazily traced derivative rules.
///
/// The handle resolves to the retained array through the session that registered it. An eager domain resolves it when
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

impl<'c> ImportingContext<Domain<'c>, ThroughSession> for TracingContext<Constant, Operation, Constant> {
    #[inline]
    fn import(&self, value: Value<'c>) -> Result<Self::Value, ProgramError> {
        self.lift(session_constant(value)?)
    }
}

impl<'c, C: CapturingContext<Capture = Constant>> ImportingContext<Domain<'c>, ThroughSession>
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
fn session_constant<'c>(value: Value<'c>) -> Result<Constant, ProgramError> {
    match value {
        Value::Array(array) => Ok(Constant::Session(array.backend_domain().session().retain_constant(&array))),
        Value::Dimension(dimension) => Ok(Constant::Dimension(dimension.into_value())),
        Value::Reference(_) => {
            Err(TypeError::invalid("cannot import an xla reference into a trace whose capture table holds constants")
                .into())
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda-13")]
    use std::convert::Infallible;

    use pretty_assertions::assert_eq;
    #[cfg(feature = "cuda-13")]
    use ryft_core::{
        Add, EffectClass, OneLike, ProjectedValue, ReferenceRead, ReferenceWrite, Tracer as CoreTracer, TracingContext,
    };
    use ryft_core::{
        ArrayReference, ArrayType, DataType, Device, DeviceMesh, Dimension as CoreDimension, DiskCache, Shape,
        Sharding, ShardingDimension,
    };
    #[cfg(feature = "cuda-13")]
    use ryft_cuda::{
        CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaKernelParameterType,
    };
    #[cfg(feature = "cuda-13")]
    use ryft_kernels::{KernelCompilationError, KernelCompiler, KernelDefinition, KernelSchedule, VerifiedKernel};
    use ryft_pjrt::protos::{CompilationOptions, Precision};
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{ClientOptions, GpuClientOptions, GpuMemoryAllocator, GpuPlatform, load_cuda_13_plugin};

    use crate::domains::tests::domain_mesh;
    #[cfg(feature = "cuda-13")]
    use crate::kernels::{CUDA_KERNEL_CUSTOM_CALL_TARGET, CudaKernelEmbedding, KernelCompilerBinding, stage_kernel};
    #[cfg(feature = "cuda-13")]
    use crate::tests::values_from_bytes;
    use crate::tests::{
        assert_send_sync, cpu_client, execution_client, four_device_mesh_x, logical_mesh_2x2, values_to_bytes,
    };
    use crate::{Array, FromPjrt, Reference, Session};
    #[cfg(feature = "cuda-13")]
    use crate::{Constant, Operation, Options};

    use super::*;

    #[test]
    fn test_session_is_send_and_sync() {
        assert_send_sync::<Session<'static>>();
    }

    #[test]
    fn test_session_with_cache_capacity_evicts_least_recently_used_entry() {
        let client = cpu_client(4);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh_2x2(), devices).unwrap();
        // Capacity = 2 so that the third distinct reshard evicts the first.
        let engine = Session::with_cache_capacity(&client, 2).domain();

        // Replicated source on the 2x2 mesh. Three reshards to three different non-replicated shardings all force the
        // compiled path, because the fast path returns `None` when no destination shard matches a source shard exactly.
        let values = (0..16).map(|index| index as f32).collect::<Vec<_>>();
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let source_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4), CoreDimension::Static(4)]))
                .with_sharding(replicated_sharding)
                .unwrap();
        let make_source = || {
            Array::from_host_buffer(
                &engine,
                source_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&values).as_slice(),
            )
            .unwrap()
        };
        let sharded_along_x = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let sharded_along_y = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let sharded_along_both = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();

        assert_eq!(engine.program_count(), 0);
        let _ = make_source()
            .to_placement(
                &engine,
                crate::arrays::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_along_x },
            )
            .unwrap();
        assert_eq!(engine.program_count(), 1);
        let _ = make_source()
            .to_placement(
                &engine,
                crate::arrays::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_along_y },
            )
            .unwrap();
        assert_eq!(engine.program_count(), 2);
        let _ = make_source()
            .to_placement(
                &engine,
                crate::arrays::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_along_both },
            )
            .unwrap();
        assert_eq!(engine.program_count(), 2, "third distinct reshard should evict the LRU entry");
    }

    #[test]
    fn test_session_with_disk_cache_warm_starts_a_fresh_session() {
        let client = cpu_client(4);
        let mesh = four_device_mesh_x(&client);
        let cache_dir = tempfile::tempdir().unwrap();

        let values = [200.0f32, 201.0, 202.0, 203.0];
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let target_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // The first context compiles cold, and the disk cache picks up the serialized executable.
        let engine_one = Session::with_disk_cache(&client, DiskCache::open(cache_dir.path()).unwrap()).domain();
        let source = Array::from_host_buffer(
            &engine_one,
            source_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let _ = source
            .to_placement(
                &engine_one,
                crate::arrays::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding.clone() },
            )
            .unwrap();
        assert_eq!(engine_one.program_count(), 1, "first reshard should populate the in-memory cache");
        drop(source);
        drop(engine_one);

        // Second context starts with an empty in-memory cache but loads from the disk cache.
        let engine_two = Session::with_disk_cache(&client, DiskCache::open(cache_dir.path()).unwrap()).domain();
        assert_eq!(engine_two.program_count(), 0, "fresh context starts empty in-memory");
        let source_two =
            Array::from_host_buffer(&engine_two, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let _ = source_two
            .to_placement(&engine_two, crate::arrays::DevicePutTarget::Placement { mesh, sharding: target_sharding })
            .unwrap();
        assert_eq!(
            engine_two.program_count(),
            1,
            "second context should populate its in-memory cache via the disk hit (no recompile expected)",
        );
    }

    #[test]
    fn test_session_compilation_context_separates_custom_base_options() {
        let client = cpu_client(4);
        let mesh = four_device_mesh_x(&client);

        // Construct sibling domains of one session with two distinct base option templates. Different `Debug`
        // representations should produce different cache keys, so the same MLIR program compiles separately under each
        // domain even though both domains share the session's compilation cache.
        let custom_options =
            CompilationOptions { matrix_unit_operand_precision: Precision::Highest as i32, ..Default::default() };
        let default_engine = Session::new(&client).domain();
        let custom_engine = default_engine.with_compilation_options(custom_options);

        let values = [200.0f32, 201.0, 202.0, 203.0];
        let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();

        // Reshard once per domain with identical inputs. Each domain compiles its own executable.
        let sharded_target =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source_for_default = Array::from_host_buffer(
            &default_engine,
            source_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let _ = source_for_default
            .to_placement(
                &default_engine,
                crate::arrays::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_target.clone() },
            )
            .unwrap();
        let source_for_custom = Array::from_host_buffer(
            &custom_engine,
            source_type,
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let _ = source_for_custom
            .to_placement(&custom_engine, crate::arrays::DevicePutTarget::Placement { mesh, sharding: sharded_target })
            .unwrap();

        // Both domains observe the shared session cache, which holds one executable per option template.
        assert_eq!(default_engine.program_count(), 2);
        assert_eq!(custom_engine.program_count(), 2);
        assert_eq!(
            custom_engine.compilation_options().matrix_unit_operand_precision,
            Precision::Highest as i32,
            "custom CompilationContext should retain the requested base options",
        );
    }

    #[test]
    fn test_session_domain() {
        let client = execution_client();
        let session = Session::new(&client);
        let retained = Arc::downgrade(&session);
        let domain = session.domain();
        let another = session.domain();
        let fork = domain.fork_effect_scope();
        assert!(std::ptr::eq(session.client(), &client));
        assert!(std::ptr::eq(domain.compilation_context(), session.compilation_context()));
        assert!(std::ptr::eq(fork.compilation_context(), session.compilation_context()));
        assert!(Arc::ptr_eq(&domain.effect_scope, &another.effect_scope));
        assert!(!Arc::ptr_eq(&domain.effect_scope, &fork.effect_scope));
        let pending = domain.effect_scope.reserve(None).unwrap();
        fork.effects_barrier().unwrap();
        drop(pending);
        session.effects_barrier().unwrap();
        drop(session);
        assert!(retained.upgrade().is_some());
        drop(domain);
        drop(another);
        drop(fork);
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn test_session_constant() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::new(&client);
        let domain = session.domain();
        let array = session.array(ArrayType::scalar(DataType::F32), mesh, 1f32.to_ne_bytes()).unwrap();
        let handle = session.retain_constant(&array);

        // The resolved array shares the retained storage and belongs to the resolving domain.
        let resolved = session.constant(&domain.fork_effect_scope(), &handle).unwrap();
        assert_eq!(resolved, array);
        assert!(Arc::ptr_eq(resolved.backend_domain().session(), &session));

        // Handles resolve only through the session that retains their arrays.
        let other_session = Session::new(&client);
        assert_eq!(
            other_session.constant(&other_session.domain(), &handle),
            Err(TypeError::invalid(format!(
                "session constant `{handle}` belongs to session #{} but was resolved by session #{}",
                session.id, other_session.id,
            ))
            .into()),
        );
    }

    #[test]
    fn test_session_retain_constant() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::new(&client);
        let domain = session.domain();
        let array = |value: f32| session.array(ArrayType::scalar(DataType::F32), mesh.clone(), value.to_ne_bytes());
        let retained_constant_count = || session.constants.lock().unwrap().iter().flatten().count();

        // Retaining one array twice shares one entry and one ownership token, and the handles keep the array alive
        // after the array itself is dropped.
        let first = array(1.0).unwrap();
        let handle = session.retain_constant(&first);
        let duplicate = session.retain_constant(&first);
        assert_eq!(duplicate, handle);
        assert!(Arc::ptr_eq(duplicate.owner(), handle.owner()));
        drop((first, duplicate));
        assert_eq!(retained_constant_count(), 1);
        assert_eq!(session.constant(&domain, &handle).unwrap(), session.constant(&domain, &handle).unwrap());

        // Once every handle is released, the next retention reclaims the entry and reuses its position.
        drop(handle);
        let second = array(2.0).unwrap();
        let handle = session.retain_constant(&second);
        assert_eq!(retained_constant_count(), 1);
        assert_eq!(handle.index(), 0);
        assert_eq!(session.constant(&domain, &handle).unwrap(), second);
    }

    #[test]
    fn test_session_constant_display() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::new(&client);
        let array = session.array(ArrayType::scalar(DataType::F32), mesh, 1f32.to_ne_bytes()).unwrap();
        let handle = session.retain_constant(&array);
        assert_eq!(handle.to_string(), "session#0:f32[][sharding={mesh<['x'=1:auto]>, []}]");
    }

    #[test]
    fn test_session_constant_conversion_rejects_references() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::new(&client);
        let domain = session.domain();
        let array = session.array(ArrayType::scalar(DataType::F32), mesh, 1f32.to_ne_bytes()).unwrap();
        assert_eq!(
            session_constant(Value::Reference(Reference::new(ArrayReference::new(array), domain))),
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
        let session = Session::new(&client);
        let domain = session.domain();
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
        let lowered = domain.lower_program(&program, 0, &options).unwrap();
        assert!(
            lowered
                .runtime_requirements()
                .iter()
                .any(|requirement| requirement.identifier() == "ryft.kernel.cuda")
        );
        assert!(lowered.kernel_execution_facts().is_some());
        assert!(!lowered.signature().has_effects());
        let compiled = domain.compile_program(&lowered).unwrap();
        let input = session.array(scalar, mesh, 42i32.to_ne_bytes()).unwrap();
        let outputs = domain.execute_program(&compiled, vec![input.clone()]).unwrap();
        let bytes = compiled.serialize().unwrap().unwrap();
        let restored_session = Session::new(&client);
        let restored_domain = restored_session.domain();
        let restored = restored_domain.deserialize_compiled_program(&bytes).unwrap().unwrap();
        assert!(
            restored
                .runtime_requirements()
                .iter()
                .any(|requirement| requirement.identifier() == "ryft.kernel.cuda")
        );
        let restored_outputs = restored_domain.execute_program(&restored, vec![outputs[0].clone()]).unwrap();
        for (array, expected) in [(&input, 42), (&outputs[0], 43), (&restored_outputs[0], 44)] {
            assert_eq!(values_from_bytes::<i32>(&array.copy_to_host().unwrap()), vec![expected]);
        }
        drop(outputs);
        drop(restored_outputs);
        // Both sessions retain their native launchers until the final queued execution completes, even though
        // neither this call's outputs nor the session's public handles are retained by the caller.
        drop(domain.execute_program(&compiled, vec![input.clone()]).unwrap());
        drop(restored_domain.execute_program(&restored, vec![input.clone()]).unwrap());
        drop(input);
        drop(restored_domain);
        drop(restored_session);
        drop(domain);
        drop(session);
    }
}
