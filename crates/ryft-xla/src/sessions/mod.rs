//! XLA sessions and the shared resources used to stage, compile, and execute array programs. A [`Session`] implements
//! [`Domain`] over [`Type`], with [`Value`]s that hold device arrays, checked dimensions, or mutable references.
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
//! scope, even on the same client. Refer to [`RuntimeExtensions`] for how sessions configure providers and
//! initialize the runtime extensions required by compiled programs.
//!
//! [`Options`] supplies the execution [`DeviceMesh`], sharding overrides, donation flags, and backend compilation
//! policies for a staged function. The session's default [`CompilationOptions`] supply the underlying PJRT compiler
//! settings. [`Target`] describes the client's platform and devices without retaining live device handles.
//!
//! Traces whose capture tables hold constants rather than runtime values refer to runtime arrays through
//! [`RetainedArray`] handles. [`ThroughSession`] imports runtime [`Value`]s into such traces as handles
//! that retain their arrays, which keeps each array alive for as long as any handle to it exists.
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
//! [`CompiledProgram`] contains the loaded executable and its invocation contract. These flat artifacts are
//! distinct from the typed function handles in [`crate::compilation::functions`], which also retain input
//! and output container structures. The session's [`CompilationCache`] coordinates artifact reuse, and
//! [`CompilationKey`](crate::CompilationKey) records the exact computation and compilation configuration.
//! [`CompilationAnalysis`](crate::CompilationAnalysis) and [`OptimizedProgram`](crate::OptimizedProgram)
//! expose compiler diagnostics for a compiled program.
//!
//! # Eager Execution
//!
//! [`Session`] implements [`Context`] by executing operations as they are bound, in the style of JAX's op-by-op
//! dispatch. Operations that need no device computation never reach the device. [`Context::bind`] executes them on the
//! host instead, without compiling a program. These are the reference operations, which act on shared handles, and the
//! dimension operations, dimension casts and comparisons, and conversions between dimensions and scalars, which are
//! evaluated directly and upload only their array results. A nullary additive or multiplicative identity (i.e., a
//! [`ZeroOperation`] or a [`OneOperation`](ryft_core::OneOperation)) on a session with a concrete mesh materializes
//! its constant directly through the PJRT client. Every other operation is built into a flat program with a single
//! instruction over the physical [`ArrayType`]s of its inputs, which is then lowered, compiled through the compilation
//! cache of the session, and executed on the client of the session.
//!
//! Each session retains the compiled programs of eager applications in its eager dispatch cache, keyed by the
//! operation, the structure of its attached regions, its inputs, the execution mesh, and the compilation options.
//! A cache hit only revalidates the placement of its inputs, so a repeated application skips program construction,
//! lowering, and compilation key construction. [`Session::eager_dispatch_statistics`] reports the activity of this
//! cache and [`Session::clear_cache`] clears it together with the compilation cache.
//!
//! Outputs belong to the session that produced them, so chained eager operations and transforms over them keep the same
//! session, effect scope, and compilation options.
//!
//! # Diagnostics
//!
//! A session counts the logical arrays that are created through it, which [`Session::array_statistics`] reports, and
//! a session built with [`SessionBuilder::live_array_tracking`] also describes each of its live arrays through
//! [`Session::live_arrays`] and totals their device bytes through [`Session::live_array_bytes_by_device`].
//! [`Session::device_memory_statistics`] reports the allocator statistics of the devices of the session's client for
//! PJRT plugins that provide them. Refer to the documentation of [`ArrayStatistics`] and [`LiveArray`] for what these
//! represent.
//!
//! # Effects and Completion
//!
//! Cloned sessions share effect ordering. [`Session::with_new_effect_scope`] starts an independent sequence while
//! retaining the shared resources of the session. Cached programs contain no session or effect scope, so execution
//! uses the invoking session's scope. Mutable references also retain their own coordination across calls, based on
//! their identities.
//!
//! Array readiness observes completion of the work that produced that array. [`Session::synchronize`] waits for
//! computations, their effects, and transfers across all handles of the session, even if their outputs were discarded.
//! [`Session::effects_barrier`] waits for effects and reports failures in the calling handle's scope for recovery with
//! [`Session::acknowledge_effect_errors`]. The client of a session lives for the rest of the process, as in JAX,
//! so sessions and their values carry no lifetime and can be stored and moved across threads freely.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use ryft_core::macros::check_count;
use ryft_core::{
    Array as CoreArray, ArrayIrValue, ArrayOperation, ArrayType, AttachedRegions, COMPOSITE_OPERATION_NAME,
    CapturingContext, CastType, Compare, CompilationCache, Concretizable, Constant as CoreConstant, ConstantOperation,
    Context, DataType, Device, DeviceId, DeviceMesh, DimensionFromScalar, DimensionOperation, DimensionSize,
    DimensionType, DimensionValue, DiskCache, Domain, DomainTracer, EagerContext, EffectClass, EmptyRegionDriver,
    ImportingContext, InterpretableOperation, LogicalMesh, MeshAxis, MeshAxisType, MeshContext, NamedAxes, NamedAxis,
    NestedTracingContext, Operation as CoreOperation, OperationProvider, Placeholder, ProgramError, Provenance,
    ProvenanceScope, RegionDriver, RegionRef, SpecializationCache, SpecializationCacheEntry,
    SpecializationCacheStatistics, TracingContext, Type as CoreType, TypeError, Typed, TypedProgram, ValueProjection,
    Zero, ZeroOperation,
};
use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Client, ClientOptions, Error as PjrtError, MemoryStatistics, Plugin};

use crate::arrays::statistics::ArrayTracker;
use crate::compilation::keys::{EagerDispatchKey, EagerDispatchKeyView, EagerDispatchQuery};
use crate::operations::composites::NamedCompositeImplementation;
use crate::sessions::effects::{EffectReservation, EffectScope, ensure_effect_dispatch_is_allowed};
use crate::sessions::platforms::client_for_plugin;
use crate::sessions::runtimes::RuntimeRegistry;
use crate::sessions::synchronization::{SessionWorkTracker, WorkReservation};
use crate::{
    Array, ArrayStatistics, CompiledProgram, Constant, Dimension, Error, FlatProgram, FromPjrt, LiveArray,
    LowerableOperation, Operation, Options, ProgramBuilder, RetainedArray, Type, Value,
};

pub mod effects;
pub mod platforms;
pub mod runtimes;
pub mod synchronization;
pub mod targets;

pub use platforms::Platform;
pub use runtimes::{
    CustomCallOperationExtension, ExecutionResources, MAX_RUNTIME_REQUIREMENTS_BYTES, RUNTIME_REQUIREMENTS_ATTRIBUTE,
    RuntimeError, RuntimeExtension, RuntimeExtensions, RuntimeProvider, RuntimeRequirement,
    normalize_runtime_requirements,
};
pub use targets::{Target, TargetDevice};

/// Represents errors produced while selecting a [`Platform`] or configuring the PJRT client for a [`Session`],
/// or while using a session that lacks something that an operation needs.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SessionError {
    #[error("platform `{platform}` is unavailable because `ryft-xla` was built without its `{feature}` feature")]
    UnavailablePlatform { platform: Platform, feature: &'static str },

    #[error("invalid client options: {reason}")]
    InvalidClientOptions { reason: String },

    #[error("session has no device mesh")]
    MissingMesh,
}

/// Default maximum number of compiled operation applications retained in a session's eager dispatch cache, which
/// [`SessionBuilder::eager_dispatch_cache_capacity`] overrides.
const DEFAULT_EAGER_DISPATCH_CACHE_CAPACITY: usize = 4096;

/// Stateful backend that materializes, lowers, compiles, and executes traced XLA programs against a live PJRT
/// [`Client`]. A session is the handle through which Ryft's XLA backend executes on a client. Every [`Array`] carries
/// the session that it belongs to, so eager operations and transforms recover their execution context from their
/// inputs, and first-class dimensions can produce arrays (e.g., when compared or converted to scalars) without an
/// explicit context. Handles are cheap to clone. Build one session per client and derive every other handle from it
/// (i.e., through [`Clone`], [`Self::with_device_mesh`], or [`Self::with_new_effect_scope`]) so that all arrays on
/// that client share one compilation cache and one set of runtime extensions. Separate sessions remain independent,
/// even on the same client, so building a second session is an explicit request for an isolated cache and effect
/// ordering. Cached executables retain neither sessions nor arrays, so the ownership graph has no cycle between
/// caches and sessions.
///
/// The same type covers both staging and eager execution. Every session is backed by a live PJRT client. Tracing only
/// uses the session as a type witness and never needs an instance without a client, while compiling ahead of time
/// without devices requires an abstraction of compilation targets rather than a session without a client.
///
/// # Construction
///
/// [`Self::new`] creates a session with the default configuration, which runs on a client that the session
/// creates for the best available [`Platform`]. [`Self::builder`] returns a [`SessionBuilder`] for everything else.
/// [`SessionBuilder::platform`], [`SessionBuilder::plugin`], and [`SessionBuilder::client_options`] select and
/// configure the client that the session creates. Clients that are created this way are shared by the sessions that
/// request the same configuration. [`SessionBuilder::client`] instead takes a client that the caller created, which
/// is the extension point for clients that Ryft's XLA backend cannot directly create. Every client of a session lives
/// for the rest of the process and is never destroyed, which is why sessions and their values carry no lifetime for
/// their underlying clients. [`Self::mesh`] creates device meshes over the devices of the client.
///
/// # Shared and Selected State
///
/// All handles of one session share its compilation cache, its eager dispatch cache, its runtime extensions, its
/// retained constants, and its default [`CompilationOptions`]. Each handle additionally selects an effect scope and
/// an optional [`DeviceMesh`]. Clones share their scope, while [`Self::with_new_effect_scope`] starts an independent
/// sequence and retains everything else. Cached programs retain no session, so execution always uses the invoking
/// session's scope, including for cache hits and retained compiled functions.
///
/// # Compilation Caching
///
/// Each session owns one [`CompilationCache`], which its builder configures. By default, it keeps compiled programs
/// only in memory. [`SessionBuilder::disk_cache`] adds a persistent [`DiskCache`] so that later processes restart
/// warm, and [`SessionBuilder::disk_cache_from_env`] adds the one that the [`DiskCache::ENV_VAR`] environment variable
/// configures, which sessions never read otherwise. [`SessionBuilder::compilation_cache`] supplies a whole context
/// (e.g., one with a bounded in-memory tier or, for multi-process programs, one with an exchange from
/// [`DistributedRuntime::compilation_artifact_exchange`](crate::DistributedRuntime::compilation_artifact_exchange)).
/// Refer to the documentation of [`CompilationCache`] for more information.
///
/// # Effects and Completion
///
/// Ordered I/O dependencies gate the successor computation's entire launch, including its pure work. The backend may
/// execute inline or asynchronously. Use unordered printing when debugging code whose throughput matters and whose
/// ordering does not. Before the process exits, finish submitting work and call [`Self::synchronize`] to wait for
/// registered work, including effects across all handles of the session. Use [`Self::effects_barrier`] to wait for
/// effects in one scope and inspect its failures. Destruction of the last handle waits for launches borrowing
/// extension resources before releasing those resources. Ordinary execution adds no implicit destruction barrier,
/// and backend behavior with unfinished device work is specific to each plugin.
///
/// Custom calls and prints declaring [`EffectClass::DeviceOrderedIo`] may execute separately on each device inside
/// manual computation regions on CPU, while occurrences of [`EffectClass::OrderedIo`] keep the program on one device.
/// Tokens retain the dependencies of each device without a device rendezvous across Ryft. The backend can impose
/// additional scheduling dependencies, including serialization of multi-device launches. Opaque registered handlers
/// must not reenter dispatch or barriers. Managed print and assertion handlers reject such reentrancy before acquiring
/// scope locks.
///
/// # Example
///
/// ```
/// # use ryft_core::{ArrayType, DataType, Print};
/// # use ryft_xla::{Array, Platform, Session};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let session = Session::builder().platform(Platform::Cpu).build()?;
/// let value = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F64), None, None, 1.0f64.to_ne_bytes())?;
/// value.print("value");
/// session.synchronize()?;
/// # Ok(())
/// # }
/// ```
///
/// The client of a session must live for the rest of the process, so a client that is dropped cannot be supplied:
///
/// ```compile_fail
/// # use ryft_xla::Session;
/// fn escaped(client: ryft_pjrt::Client<'_>) {
///     let session = Session::builder().client(&client).build().unwrap();
///     drop(client);
///     session.synchronize().unwrap();
/// }
/// ```
pub struct Session {
    /// State of this session that contains resources shared by every handle of this session.
    state: Arc<SessionState>,

    /// Default device mesh of this handle, which eager binds without array inputs execute on.
    mesh: Option<DeviceMesh>,

    /// Selected ordering and completion scope, which clones of this handle share and which
    /// [`Self::with_new_effect_scope`] replaces.
    effect_scope: Arc<EffectScope>,
}

impl Session {
    /// Returns a [`SessionBuilder`] with the default configuration, which, by default, builds a session on the best
    /// available [`Platform`], with an empty in-memory compilation cache, the built-in runtime extensions, and the
    /// default compilation options.
    #[inline]
    pub fn builder() -> SessionBuilder {
        SessionBuilder {
            client_source: ClientSource::Platform(Platform::Auto),
            client_options: None,
            compilation_options: CompilationOptions::default(),
            compilation_cache: CompilationCache::new(),
            disk_cache: None,
            eager_dispatch_cache_capacity: DEFAULT_EAGER_DISPATCH_CACHE_CAPACITY,
            live_array_tracking: false,
            runtime_extensions: RuntimeExtensions::new(),
        }
    }

    /// Creates a new independent [`Session`] with the default configuration, which runs on a client that the session
    /// creates for the best available [`Platform`] and has an empty in-memory compilation cache, the built-in runtime
    /// extensions, and the default compilation options. This is a shorthand for a [`SessionBuilder`] that is built
    /// without any configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if no client can be created, which requires even the built-in CPU plugin to fail.
    #[inline]
    pub fn new() -> Result<Self, Error> {
        Self::builder().build()
    }

    /// Returns a handle of this session whose default mesh is `mesh`. Host uploads through [`Array::from_host_buffer`]
    /// and eager operations without array inputs, such as creating constants or zeros, use this mesh when none is
    /// specified. The handle also exposes it through [`MeshContext::default_mesh`] for calls like
    /// `shard_map(.., None, ..)`. The handle shares the resources and the effect scope of this session.
    #[inline]
    pub fn with_device_mesh(&self, mesh: DeviceMesh) -> Self {
        Self { mesh: Some(mesh), ..self.clone() }
    }

    /// Returns a handle of this session with a new and empty effect scope, which shares the resources of this session
    /// and keeps the mesh of this handle. Handles that share a scope run their ordered effects (e.g., prints) one after
    /// another in submission order, a failed ordered effect blocks further ones until it is observed and acknowledged,
    /// and a scope stays on the devices that its first ordered effect ran on. A new scope lifts all three for
    /// independent work, such as another thread or request, or work on other devices. References retain their
    /// own coordination, based on their identities.
    #[inline]
    pub fn with_new_effect_scope(&self) -> Self {
        Self { effect_scope: Arc::new(EffectScope::default()), ..self.clone() }
    }

    /// Returns the PJRT [`Client`] of this [`Session`].
    #[inline]
    pub fn client(&self) -> &'static Client<'static> {
        self.state.client
    }

    /// Returns the default [`CompilationOptions`] that every compilation of this [`Session`] forwards to
    /// [`Client::compile`]. Every compilation of the session, including those of eager operations, starts with these
    /// options, unless its [`Options`] supply their own using [`Options::with_compilation_options`]. Lowering then
    /// overlays the settings that it derives for each program (i.e., the replica and partition counts, the partitioner
    /// selection, the assignment of the program to the devices of its mesh, and the feedback-directed profile of its
    /// options). The default options therefore suit settings that hold for every program of the session (e.g., operand
    /// precisions and environment option overrides) rather than settings of one specific program signature (e.g., input
    /// layouts).
    #[inline]
    pub fn compilation_options(&self) -> &CompilationOptions {
        &self.state.compilation_options
    }

    /// Returns the [`CompilationCache`] that all handles of this [`Session`] share.
    #[inline]
    pub fn compilation_cache(&self) -> &CompilationCache<'static, Self> {
        &self.state.compilation_cache
    }

    /// Returns the number of compiled programs currently cached in the in-memory tier.
    #[inline]
    pub fn cached_program_count(&self) -> usize {
        self.state.compilation_cache.program_count()
    }

    /// Returns the [`RuntimeRegistry`] of this [`Session`].
    #[inline]
    pub(crate) fn runtimes(&self) -> &RuntimeRegistry {
        &self.state.runtimes
    }

    /// Returns the [`ArrayTracker`] that counts the logical arrays created through this [`Session`]. Array storage
    /// registers with it when it is created.
    #[inline]
    pub(crate) fn array_tracker(&self) -> &Arc<ArrayTracker> {
        &self.state.array_tracker
    }

    /// Returns the [`Target`] of this [`Session`]'s [`Client`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if querying the platform, plugin, or device facts from the client failed.
    /// The failure is retained, so every later call returns the same error.
    #[inline]
    pub fn target(&self) -> Result<&Target, Error> {
        self.state
            .target
            .get_or_init(|| Target::from_client(self.state.client))
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Returns the default [`DeviceMesh`] of this [`Session`], or [`SessionError::MissingMesh`] if this session has no
    /// default mesh. Eager binds execute on the mesh of their array inputs and only use this mesh when they have none.
    /// Refer to [`Self::with_device_mesh`] for more information.
    #[inline]
    pub fn device_mesh(&self) -> Result<&DeviceMesh, Error> {
        self.mesh.as_ref().ok_or_else(|| Error::from(SessionError::MissingMesh))
    }

    /// Resolves the concrete [`DeviceMesh`] that a value is placed on when its caller names none. A value that is
    /// computed from `inputs` is placed where they live, so the first input's mesh takes precedence over the mesh
    /// configured on this handle. Without either, `logical_mesh` is laid out over addressable devices in enumeration
    /// order. If all three are absent, the first addressable device forms a mesh with a single device. Eager operations
    /// derive `logical_mesh` from the first sharded output. This lets transforms place nullary constants through
    /// [`Zero`], [`One`](ryft_core::One), [`Fill`](ryft_core::Fill), and [`Iota`](ryft_core::Iota) on sessions without
    /// a configured mesh. [`Array::from_host_buffer`] uses the effective sharding when the caller omits the mesh.
    pub(crate) fn device_mesh_for(
        &self,
        inputs: &[Array],
        logical_mesh: Option<&LogicalMesh>,
    ) -> Result<DeviceMesh, ProgramError> {
        // The eager dispatch cache keys its entries on this same choice between the first input's mesh and the mesh of
        // this handle, in `Context::bind`. The two must stay in agreement.
        if let Some(mesh) = inputs.first().map(Array::mesh).or(self.mesh.as_ref()) {
            return Ok(mesh.clone());
        }

        let devices = self
            .client()
            .addressable_devices()
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        if let Some(logical_mesh) = logical_mesh {
            let device_count = logical_mesh.device_count();
            if devices.len() < device_count {
                return Err(ProgramError::InvalidArgument {
                    message: format!(
                        "eager XLA placement of a value without array inputs over mesh {:?} requires \
                         {} addressable device(s) but the client only has {}",
                        logical_mesh,
                        device_count,
                        devices.len(),
                    ),
                });
            }

            let devices = devices
                .iter()
                .take(device_count)
                .map(|device| {
                    Device::from_pjrt(device)
                        .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })
                })
                .collect::<Result<Vec<_>, _>>()?;

            return DeviceMesh::new(logical_mesh.clone(), devices)
                .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() });
        }

        let device = devices.first().ok_or_else(|| ProgramError::InvalidArgument {
            message: "eager XLA execution requires at least one addressable device".to_string(),
        })?;
        let device =
            Device::from_pjrt(device).map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        let axis = MeshAxis::new("x", 1, MeshAxisType::Auto)
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        let logical_mesh = LogicalMesh::new(vec![axis])
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        DeviceMesh::new(logical_mesh, vec![device])
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })
    }

    /// Returns whether `other` is a handle of the same underlying [`Session`] as this one (i.e., whether the two share
    /// their compilation cache, runtime extensions, and retained constants), irrespective of their meshes and effect
    /// scopes.
    #[inline]
    pub(crate) fn is_same_session(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    /// Returns a snapshot of the activity of this [`Session`]'s eager dispatch cache, which maps eager operation
    /// applications directly to compiled programs, ahead of program construction and lowering.
    #[inline]
    pub fn eager_dispatch_statistics(&self) -> SpecializationCacheStatistics {
        self.state.eager_dispatch_cache.statistics()
    }

    /// Removes every entry from the in-memory compilation cache and the eager dispatch cache of this [`Session`]. Eager
    /// dispatch cache hits do not re-read the `XLA_FLAGS` environment variable, although the miss path still records it
    /// in the compilation key, so call this function after changing it if the change must apply to eager operations
    /// that already ran. XLA itself reads the variable when it initializes its debug options.
    #[inline]
    pub fn clear_cache(&self) {
        self.compilation_cache().clear_cache();
        self.state.eager_dispatch_cache.clear();
    }

    /// Returns a snapshot of the number of logical arrays that were created through this [`Session`] and of how many of
    /// them are still alive. A logical array is the storage that an [`Array`] and all of its clones share, so cloning
    /// and dropping array handles changes nothing until the last handle of an array is dropped. All handles of the
    /// session share the counts, and independently built sessions count separately, even on the same PJRT client.
    ///
    /// An array counts towards the session that created its storage, even when another session dispatches operations
    /// on it. Each call to [`RetainedArray::to_array`] creates a logical array that counts towards the receiving
    /// session and shares the retained device buffers.
    ///
    /// The counts are maintained without locks and cost nothing when arrays are cloned or dropped, so they are always
    /// available. A count of live arrays that keeps growing across the steps of a loop indicates retained arrays, and
    /// the growth of the count of created arrays over one step is the number of arrays that the step allocated.
    #[inline]
    pub fn array_statistics(&self) -> ArrayStatistics {
        self.state.array_tracker.statistics()
    }

    /// Returns descriptions of the logical arrays that were created through this [`Session`] and are still alive,
    /// in the order in which they were created, or [`None`] if the session was built without
    /// [`SessionBuilder::live_array_tracking`]. Each [`LiveArray`] records the type and placement of an array and the
    /// sizes of its local device buffers. This corresponds to `jax.live_arrays()` and is the way to see which arrays
    /// hold device memory on PJRT plugins that provide no [`Self::device_memory_statistics`].
    ///
    /// The descriptions are metadata that is copied out of the session, so the result does not keep any array or
    /// device buffer alive. Which arrays count towards the session is described in [`Self::array_statistics`].
    #[inline]
    pub fn live_arrays(&self) -> Option<Vec<LiveArray>> {
        self.state.array_tracker.live_arrays()
    }

    /// Returns the number of logical payload bytes that the live arrays of this [`Session`] hold on each
    /// device of the current process, keyed by [`DeviceId`], or [`None`] if the session was built without
    /// [`SessionBuilder::live_array_tracking`]. The live arrays are the ones that [`Self::live_arrays`] describes.
    /// A device buffer that several of them share (e.g., an array and the array that [`RetainedArray::to_array`]
    /// creates from it) is counted once, like in the device memory profile of JAX, so the result is not the sum of
    /// [`LiveArray::addressable_size_in_bytes`] over the live arrays. Devices on which no live array holds a buffer
    /// are absent from the result.
    ///
    /// These are the bytes of the array payloads, which is a lower bound on what the arrays cost. The allocator of
    /// a device may round buffers up and holds memory for other purposes, which [`Self::device_memory_statistics`]
    /// reports where it is supported.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if the element type of a live array has no dense storage size and
    /// [`Error::SizeLimitExceeded`] if a total does not fit in a [`usize`].
    #[inline]
    pub fn live_array_bytes_by_device(&self) -> Result<Option<HashMap<DeviceId, usize>>, Error> {
        self.state.array_tracker.live_array_bytes_by_device()
    }

    /// Returns the allocator statistics of every device that the client of this [`Session`] can address, in the order
    /// in which the client reports the devices and paired with their [`DeviceId`]s. The statistics of a device are
    /// [`None`] if its PJRT plugin or its memory allocator does not provide any, which is the case for the built-in
    /// CPU plugin.
    ///
    /// The statistics describe everything that the allocator of a device has handed out, including temporaries of
    /// running programs and buffers of other sessions on the same client, so they answer whether device memory is
    /// growing rather than which arrays hold it. Reading them queries the allocators and adds no cost to any other
    /// operation of the session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if the devices of the client cannot be queried or if a device fails to report its
    /// statistics for a reason other than not supporting them.
    pub fn device_memory_statistics(&self) -> Result<Vec<(DeviceId, Option<MemoryStatistics>)>, Error> {
        self.client()
            .addressable_devices()?
            .iter()
            .map(|device| {
                let statistics = match device.memory_statistics() {
                    Ok(statistics) => Some(statistics),
                    Err(PjrtError::MissingFunction { .. } | PjrtError::Unimplemented { .. }) => None,
                    Err(error) => return Err(error.into()),
                };
                Ok((device.id()?, statistics))
            })
            .collect()
    }

    /// Resets the allocator statistics of every device that the client of this [`Session`] can
    /// address, so that the peaks that [`Self::device_memory_statistics`] reports afterward (e.g.,
    /// [`MemoryStatistics::peak_bytes_in_use`]) cover only what happens after this call. This makes it possible to
    /// measure the peak memory of one step of a program. Devices that provide no statistics are skipped. The reset
    /// applies to the devices themselves, so it also affects other sessions on the same client.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if the devices of the client cannot be queried or if a device fails to reset its
    /// statistics for a reason other than not supporting them.
    pub fn clear_device_memory_statistics(&self) -> Result<(), Error> {
        for device in self.client().addressable_devices()? {
            match device.clear_memory_statistics() {
                Ok(()) | Err(PjrtError::MissingFunction { .. } | PjrtError::Unimplemented { .. }) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    /// Creates a [`DeviceMesh`] with the provided `axes` over the devices of this session's client. The mesh takes as
    /// many devices as its axes require, in the order in which the client reports them. In a program with multiple
    /// processes, these include devices that other processes own.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if the devices of the client cannot be queried, and a
    /// [`ShardingError`](ryft_core::ShardingError) if `axes` do not form a valid mesh or require more devices than
    /// the client has.
    #[inline]
    pub fn mesh<A: IntoIterator<Item = MeshAxis>>(&self, axes: A) -> Result<DeviceMesh, Error> {
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

    /// Blocks until the work registered with this session when the call takes its snapshot has finished. This includes
    /// compiled executions, their effects, and array transfers that Ryft issues on the calling process. It covers all
    /// handles of the session, including handles with independent effect scopes. Later registrations are handled by a
    /// subsequent synchronization. Cross-host transfers may still need another process to issue a matching operation.
    /// Independently built sessions have their own tracker even when they use the same PJRT client. Direct PJRT
    /// submissions, imported buffer readiness, and standalone shard readbacks use their own completion APIs.
    ///
    /// Use this function to wait for a batch of submissions. At process exit, finish submitting work and call it
    /// once per independently built session to wait for all of its registered work. To inspect failures in one
    /// effect scope, use [`Self::effects_barrier`] on that scope. Its report is required before recovery with
    /// [`Self::acknowledge_effect_errors`].
    ///
    /// # Errors
    ///
    /// Returns [`SynchronizationFailures`](crate::ExecutionError::SynchronizationFailures) in registration order
    /// after waiting for the snapshot. A failed launch or batch of copies contributes one error, including a failure
    /// to obtain a transfer's ready event. Synchronization reports each recorded failure once across the session's
    /// handles. Effect barriers and execution outputs retain their own results. Returns
    /// [`ReentrantDispatch`](crate::ExecutionError::ReentrantDispatch) from within a managed effect callback.
    #[inline]
    pub fn synchronize(&self) -> Result<(), Error> {
        ensure_effect_dispatch_is_allowed()?;
        self.state.work.synchronize()
    }

    /// Waits for the executions already reserved in this handle's effect scope to finish or be canceled. This includes
    /// ordered and unordered I/O, local assertions, and submissions still being prepared. Use this function when
    /// the host needs those effects to finish, even if there are no output arrays to wait on. A subsequent barrier
    /// covers later reservations. Handles created with [`Self::with_new_effect_scope`] have independent barriers.
    /// For process exit, [`Self::synchronize`] waits for all registered session work and its effects after submissions
    /// have stopped.
    ///
    /// If ordered I/O fails, inspect the failures reported by this function, then call
    /// [`Self::acknowledge_effect_errors`] on the same scope before resuming ordered submissions.
    ///
    /// # Errors
    ///
    /// Returns [`EffectSubmissionFailures`](crate::ExecutionError::EffectSubmissionFailures)
    /// with failures in reservation order. These failures remain in the scope until acknowledged. Returns
    /// [`ReentrantDispatch`](crate::ExecutionError::ReentrantDispatch) from within a managed effect callback.
    #[inline]
    pub fn effects_barrier(&self) -> Result<(), Error> {
        self.effect_scope.barrier()
    }

    /// Clears completed failures from this handle's effect scope after [`Self::effects_barrier`] has reported them.
    /// To recover from an ordered I/O failure, first call the barrier on the same scope and inspect the failures it
    /// returns. Then call this function to clear them and reset the failed ordered chain. After acknowledgment
    /// succeeds, the next ordered submission starts a new chain with fresh tokens.
    ///
    /// Existing arrays and executions retain their errors. Account for any effects or reference mutations that already
    /// occurred before retrying the failed execution.
    ///
    /// # Errors
    ///
    /// Returns [`PrematureAcknowledgment`](crate::ExecutionError::PrematureAcknowledgment) if a recorded failure has
    /// yet to be reported or if the scope has both a recorded ordered failure and pending ordered work. This can happen
    /// even after [`Self::effects_barrier`] returns. The barrier waits for the reservations in its snapshot. Work
    /// reserved afterward can fail before acknowledgment and still need to be reported, or later ordered work can still
    /// be pending when the barrier reports an earlier ordered failure. Call the barrier again on the same scope and
    /// retry acknowledgment. Returns [`ReentrantDispatch`](crate::ExecutionError::ReentrantDispatch) from within a
    /// managed effect callback.
    #[inline]
    pub fn acknowledge_effect_errors(&self) -> Result<(), Error> {
        self.effect_scope.acknowledge()
    }

    /// Reserves a place in this handle's effect scope before submitting a program with effects to PJRT. The reservation
    /// is visible to [`Self::effects_barrier`] during submission setup. Refer to [`EffectScope::reserve`] for device
    /// ordering and errors. Before submitting ordered I/O, check for input tokens with
    /// [`EffectReservation::wait_for_predecessor_tokens`]. Once PJRT accepts the submission, finish the
    /// handoff with [`EffectReservation::publish`]. Dropping the reservation before handoff cancels it.
    pub(crate) fn reserve_effects(&self, devices: Option<&[DeviceId]>) -> Result<EffectReservation, Error> {
        self.effect_scope.reserve(devices)
    }

    /// Registers a PJRT execution or batch of transfers with the session and returns a [`WorkReservation`]. Call this
    /// before the first PJRT enqueue so a concurrent [`Self::synchronize`] includes the submission in its snapshot
    /// even if the submitting thread is still collecting completion fences or events. Attach each fence or event as
    /// soon as its work is enqueued. Dropping the reservation hands off completion tracking to the session, including
    /// when the submitting function returns early. The submission stays pending until the attached work finishes.
    /// If no work was attached, the reservation completes as soon as it is dropped.
    pub(crate) fn reserve_work(&self) -> WorkReservation {
        self.state.work.reserve()
    }

    /// Validates that every input lives on this session's PJRT client and that all inputs share one device placement,
    /// mirroring JAX's incompatible devices diagnostic for compiled computations on the eager path. Every [`Array`]
    /// belongs to a session whose client owns its addressable shard buffers, which is validated at construction, so
    /// client identity is exact and also rejects arrays with the same device IDs that a _different_ client owns.
    fn validate_input_placement(&self, inputs: &[Array]) -> Result<(), ProgramError> {
        if inputs.is_empty() {
            return Ok(());
        }

        let invalid_argument = |message: String| ProgramError::InvalidArgument { message };
        for (index, input) in inputs.iter().enumerate() {
            if !std::ptr::eq(input.client(), self.client()) {
                return Err(invalid_argument(format!(
                    "received incompatible devices for eager XLA execution: input #{index} is owned by a different \
                     PJRT client than this session's client",
                )));
            }
        }

        // Arrays produced by the same program, or cloned from one array, share their mesh allocation, so pointer
        // equality settles the common case without comparing devices.
        let first_mesh = inputs[0].mesh();
        for (index, input) in inputs.iter().enumerate().skip(1) {
            let mesh = input.mesh();
            if !std::ptr::eq(mesh, first_mesh)
                && !mesh.devices().iter().map(Device::id).eq(first_mesh.devices().iter().map(Device::id))
            {
                return Err(invalid_argument(format!(
                    "received incompatible devices for eager XLA execution: input #{} is placed on devices {:?} \
                     but input #0 is placed on devices {:?}",
                    index,
                    mesh.devices().iter().map(Device::id).collect::<Vec<_>>(),
                    first_mesh.devices().iter().map(Device::id).collect::<Vec<_>>(),
                )));
            }
        }

        Ok(())
    }
}

impl Clone for Session {
    #[inline]
    fn clone(&self) -> Self {
        Self { state: Arc::clone(&self.state), mesh: self.mesh.clone(), effect_scope: Arc::clone(&self.effect_scope) }
    }
}

impl Domain for Session {
    type Type = Type;
    type Value = Value;
    type Constant = Constant;
    type Operation = Operation;
}

impl NamedAxes for Session {
    // A session is the root eager context. Its default mesh supplies placement, and transforms such as
    // `batch` and `shard_map` introduce named-axis bindings in their own scopes.

    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        let _ = name;
        None
    }

    #[inline]
    fn named_axes(&self) -> Vec<(String, NamedAxis)> {
        Vec::new()
    }
}

impl MeshContext for Session {
    #[inline]
    fn default_mesh(&self) -> Option<LogicalMesh> {
        // A session handle that was given a device mesh through `Session::with_device_mesh` owns the logical view of
        // that mesh as its default, so eager code bound in it can omit meshes (e.g., `shard_map(.., None, ..)`), same
        // as under JAX's `jax.set_mesh`. A handle without a mesh has no default.
        self.mesh.as_ref().map(|mesh| mesh.logical_mesh().clone())
    }
}

impl Context for Session {
    fn lift(&self, constant: Constant) -> Result<Value, ProgramError> {
        // Immediate dimension extents materialize directly, Boolean literals upload a replicated scalar, and a
        // `Constant::Session` handle resolves to the array that it retains. A `Constant::Captured` payload is a
        // symbolic index into a compiled function's capture table carrying only a type and no data, so there is
        // nothing to materialize without the surrounding capture table and lifting it is always rejected.
        match constant {
            Constant::Captured(constant) => Err(TypeError::invalid(format!(
                "XLA captured constant `{constant}` requires a captured program capture table",
            ))
            .into()),
            Constant::Session(constant) => Ok(Value::Array(constant.to_array(self)?)),
            Constant::Dimension(value) => Ok(Value::Dimension(Dimension::new(value, self.clone()))),
            Constant::Boolean(value) => {
                // A Boolean literal is uploaded as a replicated scalar with eager placement.
                Array::from_host_buffer(self, ArrayType::scalar(DataType::Boolean), None, None, [u8::from(value)])
                    .map(Value::Array)
                    .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })
            }
        }
    }

    fn bind<'r, P, R: Into<AttachedRegions<'r, Self::Constant, Self::Operation>>>(
        &self,
        operation: P,
        regions: R,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError>
    where
        P: Into<Self::Operation>,
        Self::Constant: 'r,
        Self::Operation: 'r,
    {
        // Eagerly executes `operation` on the provided concrete inputs. An application is first looked up in the
        // session's eager dispatch cache, keyed by the operation, the structure of its attached regions, its inputs,
        // and the lowering mesh, which together form an `EagerDispatchKey`. A hit validates the input placement and
        // executes the cached program directly, skipping program construction, lowering, and compilation key
        // construction. On a miss, the operation is built into a flat program with a single instruction over the
        // physical array types of the inputs, including their shardings, compiled through this session's compilation
        // cache, and executed on this session's PJRT client. The compilation cache key contains the complete lowered
        // computation, its effective compilation options, and the derived mesh, so repeated eager binds of the same
        // lowering reuse one compiled executable without conflating distinct computations. Higher-order operations
        // (e.g., `condition`, `while`, `scan`, `jit_call`, and `shard_map`) receive their nested programs as attached
        // regions and flow through this same path, and the compiler handles their control flow, so no host interpreter
        // loops are needed.
        //
        // Work that involves no device computation never reaches the compilation cache. Reference operations act on
        // handles on the host, while dimension operations, dimension casts and comparisons, and conversions between
        // dimensions and scalars are decided on the host. Only their array results are uploaded, as replicated scalars
        // with eager placement. A literal constant is host data already, so it is uploaded as well. A placement may
        // move its input to the devices of another mesh, so it is carried out like `Array::to_placement`, which
        // compiles a program only when the shards themselves change. Reference operands of any other operation, and
        // reference state carried by attached regions, are rejected because XLA eager execution lowers each operation.

        ensure_effect_dispatch_is_allowed()
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;

        let operation = operation.into();
        let regions = regions.into();
        operation.validate_region_interfaces(&regions.region_interfaces())?;

        // Reference operations act on shared handles rather than on device computations, so they run on the host
        // through the generic `ArrayIrValue` reference implementations in `ryft-core`, which own handle validation
        // and locking. Those implementations take values without sessions, which are only built once the operation
        // is known to be a reference operation, so every other operation pays for nothing but this match.
        let context = EagerContext::<ArrayIrValue<Array>>::new();
        let host_inputs = || inputs.iter().cloned().map(Value::into_array_ir_value).collect::<Vec<_>>();
        let host_outputs = match &operation {
            Operation::ReferenceNew(operation) => {
                Some(operation.interpret(&context, &EmptyRegionDriver, &host_inputs()))
            }
            Operation::ReferenceRead(operation) => {
                Some(operation.interpret(&context, &EmptyRegionDriver, &host_inputs()))
            }
            Operation::ReferenceWrite(operation) => {
                Some(operation.interpret(&context, &EmptyRegionDriver, &host_inputs()))
            }
            Operation::ReferenceSwap(operation) => {
                Some(operation.interpret(&context, &EmptyRegionDriver, &host_inputs()))
            }
            Operation::ReferenceAddUpdate(operation) => {
                Some(operation.interpret(&context, &EmptyRegionDriver, &host_inputs()))
            }
            Operation::ReferenceAtomicAddUpdate(operation) => {
                Some(operation.interpret(&context, &EmptyRegionDriver, &host_inputs()))
            }
            Operation::ReferenceFreeze(operation) => {
                Some(operation.interpret(&context, &EmptyRegionDriver, &host_inputs()))
            }
            _ => None,
        };

        if let Some(outputs) = host_outputs {
            // The outputs are associated with this session, including the references that the operation created.
            return outputs?.into_iter().map(|output| Ok(Value::from_array_ir_value(output, self)?)).collect();
        }

        // Every other operation is lowered, and lowering cannot bind reference operands.
        if inputs.iter().any(|input| matches!(input, Value::Reference(_))) {
            return Err(ProgramError::UnsupportedOperation {
                message: "eager XLA execution lowers each operation and cannot bind a reference operand; \
                          discharge references before lowering"
                    .to_string(),
            });
        }

        // The intrinsic operation effects alone would miss state carried by regions, while effect scans alone would
        // miss a pure pass-through or capture of a reference type. The borrowed closure scan covers dormant rule
        // regions too without cloning their program graphs. Although those rules do not execute in this eager bind,
        // ordinary XLA rejects unresolved references across the whole artifact, so accepting one here would only
        // defer failure to compilation with a less targeted diagnostic.
        if operation.effects().classes().contains(EffectClass::OrderedState)
            || regions.regions().any(|region| {
                region.contains_effect_in_closure(EffectClass::OrderedState)
                    || region.contains_atom_type_in_closure(CoreType::is_reference)
            })
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{}` carries reference state that XLA eager execution cannot lower; \
                     discharge references before lowering",
                    operation.name(),
                ),
            });
        }

        let array_inputs = inputs
            .iter()
            .filter_map(|input| match input {
                Value::Array(array) => Some(array.clone()),
                Value::Dimension(_) | Value::Reference(_) => None,
            })
            .collect::<Vec<_>>();

        // These operations are decided on the host and return without compiling a program. `Context::bind` has already
        // rejected attached regions for the operations that declare no region slots, which includes all of them. Every
        // other operation falls through to the compiled path below.
        match &operation {
            Operation::Dimension(operation) => {
                let inputs = inputs
                    .iter()
                    .map(|input| {
                        <Value as ValueProjection<DimensionType>>::projected(input)
                            .map(|dimension| dimension.value().clone())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::new().bind(
                    operation.clone(),
                    Vec::new(),
                    inputs.as_slice(),
                )?;
                return Ok(outputs
                    .into_iter()
                    .map(|output| Value::Dimension(Dimension::new(output, self.clone())))
                    .collect());
            }
            Operation::DimensionSize(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                operation.infer_output_types(&[inputs[0].r#type().into_owned()], &[])?;
                let array = <Value as ValueProjection<ArrayType>>::projected(&inputs[0])?;
                let extent = array.dimension_size(operation.axis())?;
                let value = DimensionValue::new(operation.output_type().clone(), extent)?;
                return Ok(vec![Value::Dimension(Dimension::new(value, self.with_device_mesh(array.mesh().clone())))]);
            }
            Operation::DimensionFromScalar(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                operation.infer_output_types(&[inputs[0].r#type().into_owned()], &[])?;
                let array = <Value as ValueProjection<ArrayType>>::projected(&inputs[0])?;
                let value = array.to_dimension(operation.output_type().to_variable())?;
                return Ok(vec![Value::Dimension(Dimension::new(value, self.with_device_mesh(array.mesh().clone())))]);
            }
            Operation::DimensionToScalar(_) => {
                check_count!("input", inputs, 1, ProgramError);
                let dimension = <Value as ValueProjection<DimensionType>>::projected(&inputs[0])?;
                let extent = i64::try_from(dimension.value().extent()).unwrap();
                let output =
                    Array::from_host_buffer(self, ArrayType::scalar(DataType::I64), None, None, extent.to_ne_bytes())
                        .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                return Ok(vec![Value::Array(output)]);
            }
            Operation::Compare(operation) if let [Value::Dimension(left), Value::Dimension(right)] = inputs => {
                // Both extents are concrete, so the predicate is decided on the host by the comparison of first-class
                // dimensions, which proves it from their types and extents, instead of compiling a program for each
                // pair of extents. Only the Boolean result is materialized in this session.
                let output =
                    Compare::<ArrayIrValue<CoreArray>>::compare(left.value(), right.value(), operation.direction())?;
                let output = Concretizable::<bool>::concretize(&output)?;
                let output =
                    Array::from_host_buffer(self, ArrayType::scalar(DataType::Boolean), None, None, [u8::from(output)])
                        .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                return Ok(vec![Value::Array(output)]);
            }
            Operation::Array(ArrayOperation::Constant(operation))
                if cfg!(target_endian = "little") && operation.value().r#type().data_type() != DataType::Token =>
            {
                // Core arrays store little-endian element encodings and PJRT reads native ones, so the two only
                // coincide on little-endian targets. The other targets take the compiled path, whose lowering converts
                // between them, and so do token literals, which that lowering rejects. Also, a literal is host data
                // already, so it is uploaded like any other host buffer. Embedding it in a compiled program instead
                // would compile and retain one program for every distinct literal (e.g., for every value that `Fill`
                // is given).
                check_count!("input", inputs, 0, ProgramError);
                operation.infer_output_types(&[], &[])?;
                let literal = operation.value();
                let r#type = literal.r#type().into_owned();

                // The upload takes dense row-major bytes, which is how a literal without a layout is stored.
                let bytes = match r#type.layout() {
                    Some(_) => Cow::Owned(literal.logical_bytes()),
                    None => Cow::Borrowed(literal.storage_bytes()),
                };

                let output = Array::from_host_buffer(self, r#type, None, None, bytes)
                    .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                return Ok(vec![Value::Array(output)]);
            }
            Operation::Array(ArrayOperation::Place(operation)) => {
                // A placement can move a value to the devices of another mesh, which no single compiled program can
                // do, because a program runs on the devices of one mesh. The move is therefore decided on the host and
                // carried out like `Array::to_placement`, which shares the buffers of whole shards that stay on their
                // devices, copies whole shards between devices, and compiles a program only when the shards themselves
                // change. The binding never donates its input, because other handles may share it. The result has the
                // type that type inference returns, because a memory or layout that the target omits keeps the memory
                // or layout of the array.
                check_count!("input", inputs, 1, ProgramError);
                let array = <Value as ValueProjection<ArrayType>>::into_projected(inputs[0].clone())?;
                operation.infer_output_types(&[array.r#type().into_owned()], &[])?;

                // The result belongs to this session, so its buffers must belong to the client of this session. An
                // array of another client is therefore rejected like the inputs of any other operation, even when it
                // lives on devices with the same IDs, rather than returned as it is or mixed with buffers of this
                // client.
                self.validate_input_placement(std::slice::from_ref(&array))?;

                // The source of the operation says where the value was expected to live when the operation was staged,
                // which transposition needs. It is not compared with the placement of the array, which is known exactly
                // here and can legitimately differ from it. For example, a gradient seed is created on the mesh of this
                // session and not on the mesh of the output that it seeds, and a cotangent computed on the target mesh
                // can be placed along auto mesh axes differently from what the source of the transposed placement says.
                // Type inference has already checked the sharding that types do track. A placement that must not alias
                // its input copies whatever buffers the result still shares with any other array (e.g., the input
                // itself, which the caller still holds, when the placement leaves it where it is),
                // as `Session::place` does for an `Array` leaf.
                let output = array
                    .run_placement_dispatch(self, operation.target(), false)
                    .and_then(|output| if operation.may_alias() { Ok(output) } else { output.into_unaliased(self) })
                    .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                return Ok(vec![Value::Array(output)]);
            }
            Operation::Cast(operation) if matches!(operation.output_type(), Type::Dimension(_)) => {
                // `Context::bind` rejects attached regions. Validate the original declaration before observing
                // every array witness producer, because the host dimension has no buffer that could retain those
                // dependencies. Eager values have exact types, so interpreted inference proves the relationship
                // or rejects it and never leaves a check to a backend.
                let values = inputs.iter().cloned().map(Value::into_array_ir_value).collect::<Vec<_>>();
                let input_types = values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
                let (_, interpreted) = Type::infer_cast_output_type(operation, &input_types)?;
                if interpreted.is_none() {
                    return Err(ProgramError::Concretization {
                        message: format!(
                            "`{}` cannot check symbolic type relationships during interpretation; \
                             bind the operation through its context",
                            operation.name(),
                        ),
                    });
                }
                self.validate_input_placement(&array_inputs)?;
                for input in &array_inputs {
                    input
                        .block_until_ready()
                        .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                }
                return Ok(vec![Value::from_array_ir_value(values[0].clone(), self)?]);
            }
            _ => {}
        }

        // A composite call whose concrete inputs refine the types that its primal region was traced over (e.g., a named
        // call staged in a bounded-dynamic program) replays its primal region, exactly as the interpretation rule of
        // composites does, because that region describes the call at its traced types rather than at the exact types
        // of its inputs. Calls at their traced types compile as one program, so backend implementations of named
        // composites (e.g., fused kernels) still apply to them. A named call that explicitly requests a backend
        // implementation reports that it cannot use it instead of silently replaying its body.
        if operation.name() == COMPOSITE_OPERATION_NAME
            && let Some(primal) = regions.regions().next()
            && primal
                .interface()
                .input_types()
                .iter()
                .zip(inputs)
                .any(|(declared, input)| *declared != *input.r#type())
        {
            if let Some(key) = operation.composite_key() {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();

                // A session whose target cannot be queried cannot compile the call either, so the failure is reported
                // here instead of silently selecting the portable body.
                let target =
                    self.target().map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                let target_platform = Some(target.platform_name());
                let implementation = NamedCompositeImplementation::select(
                    key,
                    input_types.as_slice(),
                    primal.interface().output_types(),
                    target_platform,
                );

                if let Some(error) = implementation.and_then(|implementation| implementation.body_substitution_error())
                {
                    return Err(ProgramError::UnsupportedOperation { message: error.to_string() });
                }
            }

            return primal.interpret_in_context(self, inputs.to_vec());
        }

        // Applications go through the session's eager dispatch cache, which maps the operation, its attached
        // regions, and its inputs directly to a compiled program, so a repeated application skips program construction,
        // lowering, and compilation key construction and only revalidates its input placement. Hits are looked up
        // through a borrowed query, and the owned key is only built on a miss.
        let region_keys = regions.regions().map(RegionRef::key).collect::<Vec<_>>();
        let query = EagerDispatchQuery::new(
            &operation,
            region_keys.as_slice(),
            inputs,
            // This is the mesh that `Self::device_mesh_for` resolves for these inputs when it does not have to fall
            // back to the devices of the client, and it must stay in agreement with that function. The fallbacks depend
            // only on the operation and the client, which the key already covers, and leaving them out keeps cache hits
            // free of device queries.
            array_inputs.first().map(Array::mesh).or(self.mesh.as_ref()),
        );

        let compiled = match self.state.eager_dispatch_cache.get(&query as &dyn EagerDispatchKeyView) {
            Some(compiled) => {
                self.validate_input_placement(array_inputs.as_slice())?;
                compiled
            }
            None => {
                // The query borrows `region_keys`, so the key takes them only after its last use of the query.
                let key_inputs = (0..inputs.len()).map(|index| query.input(index).into_owned()).collect();
                let key_mesh = query.mesh().cloned();
                let key = EagerDispatchKey::new(operation.clone(), region_keys.into(), key_inputs, key_mesh);
                let entry = self
                    .state
                    .eager_dispatch_cache
                    .try_entry(key)
                    .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                match entry {
                    SpecializationCacheEntry::Occupied(compiled) => {
                        self.validate_input_placement(array_inputs.as_slice())?;
                        compiled
                    }
                    SpecializationCacheEntry::Vacant(producer) => {
                        // This application has not been compiled before. Build a program with a single instruction
                        // for it, attaching the provided region bodies to that instruction.
                        let region_input_types = vec![None; regions.region_count()];
                        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
                        let region_ids = regions.import_into(&builder, &region_input_types)?;
                        let output_atoms = {
                            let mut builder = builder.borrow_mut();

                            // Array inputs become inputs of the program, with their physical types and shardings.
                            // Dimension inputs are baked in as constants instead, which is why their extents are
                            // part of the dispatch key. Eager dimension values have static types, which define no
                            // identities, so equal dimension inputs share one constant instruction.
                            let mut dimension_atoms = HashMap::new();
                            let input_atoms = inputs
                                .iter()
                                .map(|input| match input {
                                    Value::Array(array) => {
                                        Ok(builder.add_input(Type::Array(array.r#type().into_owned())))
                                    }
                                    Value::Dimension(dimension) => {
                                        let dimension = dimension.value();
                                        if let Some(&atom) = dimension_atoms.get(dimension) {
                                            return Ok(atom);
                                        }
                                        let operation =
                                            DimensionOperation::Constant(ConstantOperation::new(dimension.clone()));
                                        let atom = builder.add_instruction(
                                            Operation::Dimension(operation),
                                            Vec::new(),
                                            Vec::new(),
                                            None,
                                        )?[0];
                                        dimension_atoms.insert(dimension, atom);
                                        Ok(atom)
                                    }
                                    Value::Reference(_) => {
                                        unreachable!("reference inputs are rejected before the dispatch cache lookup")
                                    }
                                })
                                .collect::<Result<Vec<_>, ProgramError>>()?;
                            builder.add_instruction(operation, region_ids, input_atoms, None)?.to_vec()
                        };

                        // Adding the instruction has inferred the output types, so type errors are reported before
                        // placement errors.
                        self.validate_input_placement(array_inputs.as_slice())?;
                        let builder =
                            Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();

                        // An output that is a dimension with a static type is already known, so it is recorded as a
                        // host value and left out of the program. Every other output is an array that the compiled
                        // program returns.
                        let outputs = output_atoms
                            .iter()
                            .map(|atom| -> Result<_, ProgramError> {
                                Ok(match builder.atoms()[atom.index()].r#type().as_ref() {
                                    Type::Dimension(r#type) if let Some(extent) = r#type.extent() => {
                                        EagerDispatchOutput::Dimension(
                                            DimensionValue::constant(extent).map_err(TypeError::from)?,
                                        )
                                    }
                                    _ => EagerDispatchOutput::Array,
                                })
                            })
                            .collect::<Result<Arc<[_]>, _>>()?;
                        let array_output_atoms = output_atoms
                            .into_iter()
                            .zip(outputs.iter())
                            .filter_map(|(atom, output)| matches!(output, EagerDispatchOutput::Array).then_some(atom))
                            .collect::<Vec<_>>();
                        let output_count = array_output_atoms.len();
                        let program: FlatProgram = builder.build(
                            array_output_atoms,
                            vec![Placeholder; array_inputs.len()],
                            vec![Placeholder; output_count],
                        )?;

                        // The mesh is derived after the program is built, so that an operation without array inputs
                        // can fall back to the logical mesh of its inferred output shardings.
                        let output_types = program
                            .output_types()
                            .iter()
                            .map(<&ArrayType>::try_from)
                            .map(|result| result.cloned())
                            .collect::<Result<Vec<_>, _>>()?;
                        let logical_mesh =
                            output_types.iter().find_map(|r#type| r#type.sharding().map(|sharding| sharding.mesh()));
                        let mesh = self.device_mesh_for(array_inputs.as_slice(), logical_mesh)?;
                        let mut options = Options::new(mesh);

                        // Arguments are placed as their runtime arrays are, including over the auto mesh axes that
                        // types never record.
                        if array_inputs.iter().any(|array| array.sharding().references_auto_axis()) {
                            options = options.with_input_placements(
                                array_inputs.iter().map(|array| Some(array.sharding().clone())).collect(),
                            );
                        }

                        // Lower the program and compile it through the compilation cache of the session, so that
                        // equal lowerings share one compiled program even when their dispatch keys differ.
                        let lowered = self
                            .lower_program(&program, 0, &options)
                            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                        let program = self
                            .compilation_cache()
                            .compile(self, &lowered)
                            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                        producer.insert(EagerDispatchEntry {
                            program,
                            constants: Arc::clone(lowered.session_constants()),
                            outputs,
                        })
                    }
                }
            }
        };

        // Execution produces outputs that belong to this session, so chained eager operations and transforms over them
        // keep executing in the same session and effect scope. Arrays of session constants precede the inputs.
        let mut arguments =
            compiled.constants.iter().map(|constant| constant.to_array(self)).collect::<Result<Vec<_>, _>>()?;
        arguments.extend(array_inputs);
        let mut arrays = self
            .execute_program(&compiled.program, arguments)
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?
            .into_iter();
        compiled
            .outputs
            .iter()
            .map(|output| match output {
                EagerDispatchOutput::Array => arrays.next().map(Value::Array).ok_or_else(|| {
                    ProgramError::MalformedProgram("eager program returned fewer arrays than its outputs".to_string())
                }),
                EagerDispatchOutput::Dimension(value) => {
                    Ok(Value::Dimension(Dimension::new(value.clone(), self.clone())))
                }
            })
            .collect()
    }

    fn is_eager(&self) -> bool {
        // Every `Session` is backed by a PJRT client, so it executes every bound operation for real and its concrete
        // arrays support host readback through `Concretizable<bool>` and `WhilePredicate`. Strategies that fold
        // data-dependent work through values visible on the host (i.e., the eager data-dependent `while` rules
        // and the unrolling of concretizable `while` loops) therefore always apply.
        true
    }

    #[inline]
    fn provenance(&self) -> Provenance {
        // `Session` is a terminal execution domain. It dispatches each operation to its own compiled program instead
        // of recording instructions, so there is nothing to attach provenance to and scopes are a deliberate no-op.
        // Programs traced _over_ this session attach provenance through their own tracing contexts.
        Provenance::unknown()
    }

    #[inline]
    fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, origin: Provenance, function: F) -> R {
        let _ = origin;
        function()
    }

    #[inline]
    fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, scope: ProvenanceScope, function: F) -> R {
        let _ = scope;
        function()
    }
}

impl Zero<Value> for Session {
    fn zero(&self, r#type: &Type) -> Result<Value, ProgramError> {
        // Transform machinery over this session (e.g., the batching rules of nullary constant operations and the
        // accumulator seeding of recursive higher-order rules) synthesizes constants through the `Zero`, `One`, `Fill`,
        // and `Iota` leaves of the active context, which are driven by types alone. The bind below goes through the
        // eager dispatch cache like any other operation, so a program is compiled for the first zero of each type and
        // reused afterwards. Dynamic array zeros are intentionally not available through this capability, because it
        // receives only a type and they require explicit first-class extent operands.
        let mut outputs = self.bind(Operation::provide(ZeroOperation::new(r#type.clone()), &[])?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl CoreConstant<Value, Constant> for Session {
    #[inline]
    fn constant(&self, value: Constant) -> Result<Value, ProgramError> {
        // Materialization delegates to `Context::lift` and therefore shares its semantics. An immediate dimension
        // extent materializes directly and a Boolean literal uploads a replicated scalar, while a `Constant::Captured`
        // payload carries only a type and no data and is always rejected outside a surrounding capture table. The
        // implementation exists because operation families that support interpretation and batching require a
        // `ryft_core::Constant` leaf on their contexts. Programs whose constants were compiled into capture tables
        // never take the captured path.
        self.lift(value)
    }
}

/// Resources that all handles of one [`Session`] share. Its caches and runtime instances are mutated through interior
/// mutability, while its configuration is immutable after construction.
struct SessionState {
    /// Client of the session, which lives for the rest of the process. It is either a client that the caller created
    /// or one of the clients that `ryft-xla` created.
    client: &'static Client<'static>,

    /// Compilation and execution target snapshot of [`Self::client`], read lazily and retained for the lifetime of the
    /// session, including a failed read.
    target: OnceLock<Result<Target, Error>>,

    /// Template of the options forwarded to [`Client::compile`], which every compilation of the session starts from.
    /// It is immutable because the eager dispatch cache does not key its entries on it.
    compilation_options: CompilationOptions,

    /// Compilation cache of the session, which [`Session::compilation_cache`] exposes. Cached artifacts contain no
    /// runtime execution association.
    compilation_cache: CompilationCache<'static, Session>,

    /// Compiled programs of eager operation applications, keyed ahead of program construction and lowering.
    /// Refer to the documentation of [`EagerDispatchKey`] for information on what the key captures.
    eager_dispatch_cache: SpecializationCache<EagerDispatchKey, EagerDispatchEntry>,

    /// Runtime extensions associated with the client, and ownership of launches borrowing their resources.
    runtimes: RuntimeRegistry,

    /// Pending work and synchronization failures shared by all handles of the session.
    work: SessionWorkTracker,

    /// Counts of the logical arrays created through the session, and descriptions of the live ones if the session
    /// tracks them. Array storage holds its own handle, which lets it update the counts when it is dropped.
    array_tracker: Arc<ArrayTracker>,
}

/// Tracer shape used while staging XLA programs directly from types, which is the [`DomainTracer`] of a [`Session`].
pub type Tracer = DomainTracer<Session>;

/// Builder of [`Session`]s, which [`Session::builder`] returns and which is the one interface for configuring their
/// construction. [`Session::new`] is a shorthand for building a session without any configuration. A builder first
/// selects the PJRT [`Client`] of the session. By default, the session creates its own client for [`Platform::Auto`].
/// [`Self::platform`] requests a specific platform, [`Self::plugin`] uses a PJRT plugin that the Ryft XLA backend does
/// not know about, and [`Self::client_options`] configures the client that either of them creates. These clients live
/// for the rest of the process, and sessions that request the same plugin with equal client options share one client,
/// as the [`platforms`] module documents. [`Self::client`] instead supplies a client that the caller created. Each of
/// [`Self::platform`], [`Self::plugin`], and [`Self::client`] replaces the selection of the others. The remaining
/// functions configure the resources that all handles of the session share. The order in which a builder is configured
/// does not matter otherwise.
///
/// # Example
///
/// ```no_run
/// # use ryft_core::DiskCache;
/// # use ryft_pjrt::{ClientOptions, CpuClientOptions};
/// # use ryft_xla::{Platform, Session};
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
pub struct SessionBuilder {
    /// Source of the [`Client`] of the session.
    client_source: ClientSource,

    /// Options of the [`Client`] that the Ryft XLA backend creates for the session, if the caller supplied any.
    client_options: Option<ClientOptions>,

    /// Template of the options that the session forwards to [`Client::compile`].
    compilation_options: CompilationOptions,

    /// Compilation cache of the session, before [`Self::disk_cache`] is applied to it.
    compilation_cache: CompilationCache<'static, Session>,

    /// Persistent tier that replaces the one of [`Self::compilation_cache`], if the caller requested one.
    disk_cache: Option<DiskCacheSource>,

    /// Maximum number of compiled operation applications that the eager dispatch cache of the session retains.
    eager_dispatch_cache_capacity: usize,

    /// Whether the session keeps descriptions of its live arrays for [`Session::live_arrays`] and
    /// [`Session::live_array_bytes_by_device`].
    live_array_tracking: bool,

    /// [`RuntimeExtensions`] of the session.
    runtime_extensions: RuntimeExtensions,
}

impl SessionBuilder {
    /// Requests a client that the Ryft XLA backend creates for `platform`. An explicit platform never falls back to
    /// another one, so building the session fails if `platform` is unavailable.
    #[inline]
    pub fn platform(mut self, platform: Platform) -> Self {
        self.client_source = ClientSource::Platform(platform);
        self
    }

    /// Requests a client that the Ryft XLA backend creates through `plugin`, for PJRT plugins that [`Platform`] does
    /// not name (e.g., one loaded with [`load_plugin`](ryft_pjrt::load_plugin)). The client is created with the options
    /// of [`Self::client_options`], or with the default [`ClientOptions`] if none are supplied.
    #[inline]
    pub fn plugin(mut self, plugin: Plugin) -> Self {
        self.client_source = ClientSource::Plugin(plugin);
        self
    }

    /// Supplies the options of the [`Client`] that the Ryft XLA backend creates for the session (e.g., the number of
    /// CPU devices, or the visible devices and the memory allocator of a GPU client) in place of the defaults of its
    /// platform. The options must fit the platform or plugin of the session, so they require an explicit one.
    #[inline]
    pub fn client_options(mut self, client_options: ClientOptions) -> Self {
        self.client_options = Some(client_options);
        self
    }

    /// Supplies a [`Client`] that the caller created, which must live for the rest of the process (e.g., a leaked
    /// client or one that is stored in a static) and which several sessions may share. This is the extension point
    /// for clients that the Ryft XLA backend cannot create (e.g., a client of a
    /// [`DistributedRuntime`](crate::DistributedRuntime), whose runtime must
    /// then live for the rest of the process as well).
    #[inline]
    pub fn client(mut self, client: &'static Client<'static>) -> Self {
        self.client_source = ClientSource::Client(client);
        self
    }

    /// Supplies the template of the options that the session forwards to [`Client::compile`].
    /// Refer to [`Session::compilation_options`] for more information.
    #[inline]
    pub fn compilation_options(mut self, compilation_options: CompilationOptions) -> Self {
        self.compilation_options = compilation_options;
        self
    }

    /// Supplies the compilation cache of the session in place of the default one, which keeps up to its default
    /// number of compiled programs in memory. Refer to [`CompilationCache`] for information on its tiers, its
    /// artifact exchange, and its event reporting.
    #[inline]
    pub fn compilation_cache(mut self, compilation_cache: CompilationCache<'static, Session>) -> Self {
        self.compilation_cache = compilation_cache;
        self
    }

    /// Makes the compilation cache of the session also write through to `disk_cache`, with its size cap and write
    /// thresholds, so that later processes restart warm. This replaces the disk cache of a context that was supplied
    /// through [`Self::compilation_cache`].
    #[inline]
    pub fn disk_cache(mut self, disk_cache: DiskCache) -> Self {
        self.disk_cache = Some(DiskCacheSource::Explicit(disk_cache));
        self
    }

    /// Opts the session into the persistent [`DiskCache`] that the environment configures. Sessions never read the
    /// environment unless this function is called, so that a variable in the shell of a user cannot change how a
    /// program that did not ask for it compiles.
    ///
    /// When the session is built, it reads [`DiskCache::ENV_VAR`] (i.e., `RYFT_COMPILATION_CACHE_DIR`):
    ///
    ///   - If the variable is set, the compilation cache of the session also writes through to a disk cache in that
    ///     directory, exactly as if it had been passed to [`Self::disk_cache`]. The cache is capped when
    ///     [`DiskCache::MAX_BYTES_ENV_VAR`] is set as well.
    ///   - If the variable is not set, this function has no effect, and the cache stays as it is otherwise
    ///     configured (i.e., in memory only, unless [`Self::compilation_cache`] supplied a disk cache of its own).
    ///
    /// This function and [`Self::disk_cache`] replace each other, so the one that is called last applies.
    ///
    /// # Errors
    ///
    /// This function itself cannot fail, but it makes [`Self::build`] return [`Error::Io`] if the variable is set
    /// and its directory cannot be opened or the size cap is not a valid number of bytes.
    #[inline]
    pub fn disk_cache_from_env(mut self) -> Self {
        self.disk_cache = Some(DiskCacheSource::Environment);
        self
    }

    /// Sets the maximum number of compiled operation applications that the eager dispatch cache of the session retains,
    /// in place of its default of `4,096`. This cache maps eager operation applications directly to their compiled
    /// programs, in front of the compilation cache, whose own capacity is that of the session's [`CompilationCache`]
    /// (i.e., [`CompilationCache::with_capacity`], with a default of `8,192`). The two caches are bounded
    /// independently, like the corresponding caches of JAX, and a program stays alive for as long as
    /// either cache retains it.
    #[inline]
    pub fn eager_dispatch_cache_capacity(mut self, capacity: usize) -> Self {
        self.eager_dispatch_cache_capacity = capacity;
        self
    }

    /// Sets whether the session keeps a description of each of its live arrays, which [`Session::live_arrays`]
    /// then returns and [`Session::live_array_bytes_by_device`] totals. Tracking is disabled by default, in which
    /// case both functions return [`None`] and creating an array costs only the lock-free counting behind
    /// [`Session::array_statistics`]. An enabled session additionally takes a lock and allocates a small description
    /// whenever an array is created, and takes the lock again when the last clone of an array is dropped. Cloning and
    /// dropping other array handles stays free of bookkeeping either way. Enable tracking to find out which arrays
    /// hold device memory.
    #[inline]
    pub fn live_array_tracking(mut self, enabled: bool) -> Self {
        self.live_array_tracking = enabled;
        self
    }

    /// Sets the runtime extensions that programs of the session can use. A runtime extension is native support that
    /// a compiled program needs beyond what the PJRT plugin provides (e.g., the host callback behind `print` or the
    /// launcher of precompiled CUDA kernels), and `runtime_extensions` holds one [`RuntimeProvider`] for each
    /// extension that the session knows how to set up. Refer to [`RuntimeExtensions`] for more information.
    ///
    /// The default is [`RuntimeExtensions::new`], which contains the built-in providers. Call this function to
    /// add providers of your own to them, or to start from [`RuntimeExtensions::empty`] and choose every provider
    /// yourself. A program that needs an extension whose provider is missing fails to compile, load, or execute
    /// in the session with [`RuntimeError::MissingProvider`].
    ///
    /// Building the session sets up none of the extensions. The session sets each one up the first time that a
    /// program needs it and then keeps it for later programs, so a provider for an accelerator that is never used,
    /// never loads its drivers. Sessions that are built with the same `runtime_extensions` share the providers but
    /// each set up extensions of their own.
    #[inline]
    pub fn runtime_extensions(mut self, runtime_extensions: RuntimeExtensions) -> Self {
        self.runtime_extensions = runtime_extensions;
        self
    }

    /// Builds a [`Session`] with a fresh effect scope.
    ///
    /// # Errors
    ///
    /// Returns a [`SessionError`] if the requested platform is unavailable in this build or if client options were
    /// supplied for [`Platform::Auto`] or for a client that the caller created, [`Error::Pjrt`] if the plugin of the
    /// session cannot be loaded or initialized or its client cannot be created, and [`Error::Io`] if the disk cache
    /// that the environment configures is invalid. A session that is only given [`Self::client`] and resources that
    /// do not read the environment cannot fail.
    pub fn build(self) -> Result<Session, Error> {
        let client: &'static Client<'static> = match (self.client_source, self.client_options) {
            (ClientSource::Platform(platform), client_options) => platform.client(client_options)?,
            (ClientSource::Plugin(plugin), client_options) => {
                client_for_plugin(&plugin, client_options.unwrap_or_default())?
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

        let compilation_cache = match self.disk_cache {
            None => self.compilation_cache,
            Some(DiskCacheSource::Explicit(disk_cache)) => self.compilation_cache.with_disk_cache(disk_cache),
            Some(DiskCacheSource::Environment) => self.compilation_cache.with_disk_cache_from_env()?,
        };

        Ok(Session {
            state: Arc::new(SessionState {
                client,
                target: OnceLock::new(),
                compilation_options: self.compilation_options,
                compilation_cache,
                eager_dispatch_cache: SpecializationCache::new(self.eager_dispatch_cache_capacity),
                runtimes: RuntimeRegistry::new(client, self.runtime_extensions),
                work: SessionWorkTracker::default(),
                array_tracker: Arc::new(ArrayTracker::new(self.live_array_tracking)),
            }),
            mesh: None,
            effect_scope: Arc::new(EffectScope::default()),
        })
    }
}

/// Import path of [`ImportingContext`] through which a trace whose capture table holds constants rather
/// than runtime values (i.e., one whose [`CapturingContext::Capture`] is [`Constant`], such as the body of a
/// [`rematerialize`](ryft_core::rematerialize) or [`composite`](ryft_core::composite) call) imports runtime XLA
/// [`Value`]s. Arrays enter the trace as [`Constant::Session`] handles, which retain them for as long as a handle
/// exists and which are equal for one array. Dimensions enter as immediate [`Constant::Dimension`]s, and references
/// are rejected because their state cannot be shared with a trace that outlives the current transform.
pub struct ThroughSession;

impl ImportingContext<Session, ThroughSession> for TracingContext<Constant, Operation, Constant> {
    #[inline]
    fn import(&self, value: Value) -> Result<Self::Value, ProgramError> {
        self.lift(value.constant()?)
    }
}

impl<C: CapturingContext<Capture = Constant>> ImportingContext<Session, ThroughSession> for NestedTracingContext<C>
where
    Self: Context<Constant = Constant>,
{
    #[inline]
    fn import(&self, value: Value) -> Result<Self::Value, ProgramError> {
        self.lift(value.constant()?)
    }
}

/// How one output of an eager operation application is obtained after its compiled program has run. Most outputs
/// are arrays that the program returns. An output whose value is already known when the application is compiled is
/// recorded here instead, and it is handed back on every execution without being computed on a device.
#[derive(Clone)]
enum EagerDispatchOutput {
    /// Array that the compiled program returns.
    Array,

    /// First-class dimension with a static type. Eager dimensions are host values with exact types, so such an output
    /// (e.g., the batch extent that a batched loop threads through its carry) is rebuilt from its type on the host
    /// rather than returned by the compiled program, which can only return arrays.
    Dimension(DimensionValue),
}

/// Entry of a session's eager dispatch cache for one [`EagerDispatchKey`]. It holds everything that is needed to
/// execute an eager operation application again without rebuilding or lowering its program (i.e., the compiled
/// program, the retained arrays that it takes as hidden arguments, and how its outputs are assembled).
#[derive(Clone)]
struct EagerDispatchEntry {
    /// Compiled program of the eager application.
    program: Arc<CompiledProgram>,

    /// [`RetainedArray`]s whose arrays the compiled program receives as hidden arguments ahead of the array inputs
    /// of the application. Their handles are part of the key, through its regions.
    constants: Arc<[RetainedArray]>,

    /// Outputs of the application, in order. The compiled program returns the [`EagerDispatchOutput::Array`]
    /// outputs, in order.
    outputs: Arc<[EagerDispatchOutput]>,
}

/// Where a [`SessionBuilder`] gets the PJRT client of its session from, as selected by the last call of
/// [`SessionBuilder::platform`], [`SessionBuilder::plugin`], or [`SessionBuilder::client`]. The builder only
/// records the choice, and [`SessionBuilder::build`] creates or looks up the client.
enum ClientSource {
    /// Process-wide client that `ryft-xla` creates for a platform.
    Platform(Platform),

    /// Process-wide client that `ryft-xla` creates through a plugin that the caller loaded.
    Plugin(Plugin),

    /// Client that the caller created.
    Client(&'static Client<'static>),
}

/// Where a [`SessionBuilder`] gets the disk cache of its session from, as selected by the last call of
/// [`SessionBuilder::disk_cache`] or [`SessionBuilder::disk_cache_from_env`]. The choice is recorded rather than
/// resolved at once because the environment is only read by [`SessionBuilder::build`].
enum DiskCacheSource {
    /// Disk cache that the caller supplied.
    Explicit(DiskCache),

    /// Disk cache that the [`DiskCache::ENV_VAR`] environment variable configures, if it is set.
    Environment,
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    #[cfg(feature = "cuda-13")]
    use std::convert::Infallible;

    use half::{bf16, f16};
    use indoc::formatdoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use ryft_core::{
        AbsOperation, Add, AddOperation, AndOperation, Array as CoreArray, ArrayOperation, ArrayReference,
        Atan2Operation, AttachedRegions, BatchAxis, CaptureReference, Cast, CastOperation, CompareOperation,
        ComparisonDirection, CompilationCacheDomain, CompositeJvpRule, CompositeOperation, ConditionOperation,
        ConstantOperation, DataType, DenseDifferentiableType, Device, Dimension as CoreDimension,
        DimensionAddOperation, DimensionBounds, DimensionDivOperation, DimensionError, DimensionFromScalarOperation,
        DimensionSizeOperation, DimensionToScalarOperation, DimensionValue, DimensionVariable, DivOperation,
        DomainTracer, DynamicBroadcastOperation, DynamicReshapeOperation, Error as CoreError, Fill, Layout, Memory,
        MeshAxisType, Mul, MulOperation, Neg, NegOperation, OneOperation, ParallelReduceOperation, Place,
        PlaceDispatch, PlaceOperation, Placeholder, Placement, ProjectedContext, ProjectedValue, Reduce, ReductionKind,
        ReferenceNewOperation, ReferenceReadOperation, ReferenceType, ScanOperation, SelectOperation, Shape,
        ShardMapBoundary, ShardMapError, ShardMapOperation, ShardMapOptions, Sharding, ShardingDimension,
        ShardingError, StaticShape, SubOperation, TiledLayout, Tracer as CoreTracer, Typed, Value as CoreValue,
        WhileOperation, ZeroOperation, batch, differentiate_at, i4, shard_map, shard_map_along,
        shard_map_with_inferred_input_shardings,
    };
    use ryft_pjrt::protos::Precision;
    use ryft_pjrt::{CpuClientOptions, ExecutionFence, load_cpu_plugin};

    #[cfg(feature = "cuda-13")]
    use ryft_core::{EffectClass, OneLike, ReferenceRead, ReferenceWrite};
    #[cfg(feature = "cuda-13")]
    use ryft_cuda::{
        CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaKernelParameterType,
    };
    #[cfg(feature = "cuda-13")]
    use ryft_kernels::{KernelCompilationError, KernelCompiler, KernelDefinition, KernelSchedule, VerifiedKernel};
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform};

    use crate::lowering::tests::test_manual_mesh;
    use crate::operations::sharding::shard_map::tests::read_device_f32s;
    use crate::sessions::effects::EffectCallbackGuard;
    use crate::tests::{
        assert_send_sync, cpu_client, execution_client, other_cpu_client, replicated_array, values_from_bytes,
        values_to_bytes,
    };
    use crate::{
        ArrayPlacement, CompileTracer, CompiledFunctionDispatcher, Dimension, ExecutionError, FlatProgram, FromPjrt,
        JitCallOperation, ProgramBuilder, Reference, jit,
    };

    #[cfg(feature = "cuda-13")]
    use crate::{CUDA_KERNEL_CUSTOM_CALL_TARGET, CudaKernelEmbedding, KernelCompilerBinding, Options, stage_kernel};

    use super::*;

    /// Returns whether the session handles use the same effect scope.
    pub(crate) fn shares_effect_scope(first: &Session, second: &Session) -> bool {
        Arc::ptr_eq(&first.effect_scope, &second.effect_scope)
    }

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

    /// Borrows the array carried by a test value, requiring the array variant.
    pub(crate) fn program_array(value: &Value) -> &Array {
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
    pub(crate) fn f32_vector(session: &Session, mesh: &DeviceMesh, values: &[f32]) -> Array {
        let r#type = replicated_vector_type(mesh, values.len());
        Array::from_host_buffer(session, r#type, mesh.clone(), None, values_to_bytes::<f32>(values).as_slice()).unwrap()
    }

    /// Uploads a replicated `f32` test scalar.
    pub(crate) fn f32_scalar(session: &Session, mesh: &DeviceMesh, value: f32) -> Array {
        let r#type = replicated_scalar_type(mesh, DataType::F32);
        Array::from_host_buffer(session, r#type, mesh.clone(), None, value.to_ne_bytes().as_slice()).unwrap()
    }

    /// Uploads a replicated rank-1 `f64` test array.
    pub(crate) fn f64_vector(session: &Session, mesh: &DeviceMesh, values: &[f64]) -> Array {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        Array::from_host_buffer(session, r#type, mesh.clone(), None, values_to_bytes::<f64>(values).as_slice()).unwrap()
    }

    /// Uploads a replicated rank-1 Boolean test array using one byte per element.
    pub(crate) fn boolean_vector(session: &Session, mesh: &DeviceMesh, values: &[bool]) -> Array {
        let r#type = ArrayType::new(DataType::Boolean, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let values = values.iter().copied().map(u8::from).collect::<Vec<_>>();
        Array::from_host_buffer(session, r#type, mesh.clone(), None, values.as_slice()).unwrap()
    }

    /// Uploads a replicated Boolean test scalar using one byte per element.
    pub(crate) fn boolean_scalar(session: &Session, mesh: &DeviceMesh, value: bool) -> Array {
        let r#type = replicated_scalar_type(mesh, DataType::Boolean);
        Array::from_host_buffer(session, r#type, mesh.clone(), None, [u8::from(value)]).unwrap()
    }

    /// Reads the first addressable shard of `array` as native-endian values of type `V`.
    pub(crate) fn read_first_shard<V: Copy>(client: &Client<'_>, array: &Array) -> Vec<V> {
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
        values_from_bytes::<V>(bytes.as_slice())
    }

    /// Reads every device shard of `array` in mesh device order and concatenates their native-endian `f32` values.
    pub(crate) fn read_sharded_f32_array(array: &Array) -> Vec<f32> {
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

    /// Reads the first addressable shard, interpreting each nonzero byte as `true`.
    pub(crate) fn read_booleans(client: &Client<'_>, array: &Array) -> Vec<bool> {
        read_first_shard::<u8>(client, array).into_iter().map(|value| value != 0).collect()
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
    fn test_session_is_send_and_sync() {
        assert_send_sync::<Session>();
    }

    #[test]
    fn test_session_builder() {
        // A builder without any configuration builds what `Session::new` creates.
        let session: Session = Session::builder().build().unwrap();
        assert!(std::ptr::eq(session.client(), Session::new().unwrap().client()));
        assert_eq!(session.compilation_options(), &CompilationOptions::default());
        assert!(session.compilation_cache().disk_cache().is_none());
        assert_eq!(session.cached_program_count(), 0);
        assert!(matches!(session.device_mesh(), Err(Error::Session(SessionError::MissingMesh))));
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
        assert!(!shares_effect_scope(&session, &other));
    }

    #[test]
    fn test_session_with_device_mesh() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let meshed = session.with_device_mesh(mesh.clone());
        assert_eq!(meshed.device_mesh().unwrap(), &mesh);
        assert!(meshed.is_same_session(&session));
        assert!(shares_effect_scope(&meshed, &session));

        // The handle that the mesh was added to is unaffected.
        assert!(matches!(session.device_mesh(), Err(Error::Session(SessionError::MissingMesh))));
    }

    #[test]
    fn test_session_with_new_effect_scope() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let fork = session.with_new_effect_scope();
        assert!(fork.is_same_session(&session));
        assert!(!shares_effect_scope(&fork, &session));
        assert_eq!(fork.device_mesh().unwrap(), &mesh);

        // A barrier of the new scope does not wait for effects that are pending in the scope of the original handle.
        let _pending = session.reserve_effects(None).unwrap();
        fork.effects_barrier().unwrap();
    }

    #[test]
    fn test_session_client() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        assert!(std::ptr::eq(session.client(), client));
        assert!(std::ptr::eq(session.with_new_effect_scope().client(), client));
    }

    #[test]
    fn test_session_compilation_options() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let compilation_options =
            CompilationOptions { matrix_unit_operand_precision: Precision::Highest as i32, ..Default::default() };
        let session =
            Session::builder().client(client).compilation_options(compilation_options.clone()).build().unwrap();
        assert_eq!(
            Session::builder().client(client).build().unwrap().compilation_options(),
            &CompilationOptions::default()
        );

        // Every handle of a session returns the template that the session was built with.
        assert_eq!(session.compilation_options(), &compilation_options);
        assert_eq!(session.with_new_effect_scope().compilation_options(), &compilation_options);
        assert_eq!(session.with_device_mesh(mesh).compilation_options(), &compilation_options);
    }

    #[test]
    fn test_session_compilation_cache() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        assert!(std::ptr::eq(session.clone().compilation_cache(), session.compilation_cache()));
        assert!(std::ptr::eq(session.with_new_effect_scope().compilation_cache(), session.compilation_cache()));
        assert!(!std::ptr::eq(
            Session::builder().client(client).build().unwrap().compilation_cache(),
            session.compilation_cache()
        ));
    }

    #[test]
    fn test_session_cached_program_count() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector = f32_vector(&session, &mesh, &[1.0, 2.0]);
        assert_eq!(session.cached_program_count(), 0);
        Neg::neg(&vector).unwrap();
        assert_eq!(session.cached_program_count(), 1);

        // Repeating an eager operation compiles nothing, while an input of another type compiles another program.
        Neg::neg(&vector).unwrap();
        assert_eq!(session.cached_program_count(), 1);
        Neg::neg(&f32_vector(&session, &mesh, &[1.0, 2.0, 3.0])).unwrap();
        assert_eq!(session.cached_program_count(), 2);
    }

    #[test]
    fn test_session_target() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let target = session.target().unwrap();
        assert_eq!(target.platform_name(), client.platform_name().unwrap());
        assert_eq!(target.devices().len(), 2);

        // The target is read once and shared by all handles of the session.
        assert!(std::ptr::eq(session.with_new_effect_scope().target().unwrap(), target));
    }

    #[test]
    fn test_session_device_mesh() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        assert!(matches!(session.device_mesh(), Err(Error::Session(SessionError::MissingMesh))));
        assert_eq!(session.with_device_mesh(mesh.clone()).device_mesh().unwrap(), &mesh);
    }

    #[test]
    fn test_session_device_mesh_for() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let wide_mesh = two_device_mesh(client, MeshAxisType::Auto);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let first_mesh = DeviceMesh::new(logical_mesh.clone(), vec![wide_mesh.devices()[0]]).unwrap();
        let second_mesh = DeviceMesh::new(logical_mesh, vec![wide_mesh.devices()[1]]).unwrap();

        // Without inputs, a mesh of the handle, or a logical mesh, the first addressable device forms a mesh.
        assert_eq!(session.device_mesh_for(&[], None), Ok(first_mesh.clone()));

        // A logical mesh is laid out over the addressable devices, in the order in which the client reports them.
        assert_eq!(session.device_mesh_for(&[], Some(wide_mesh.logical_mesh())), Ok(wide_mesh.clone()));
        let large_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Auto).unwrap()]).unwrap();
        assert_eq!(
            session.device_mesh_for(&[], Some(&large_mesh)),
            Err(ProgramError::InvalidArgument {
                message: format!(
                    "eager XLA placement of a value without array inputs over mesh {large_mesh:?} requires 4 \
                     addressable device(s) but the client only has 2",
                ),
            }),
        );

        // The mesh of the handle takes precedence over a logical mesh, and the mesh of the first input over both.
        let meshed = session.with_device_mesh(second_mesh.clone());
        assert_eq!(meshed.device_mesh_for(&[], Some(wide_mesh.logical_mesh())), Ok(second_mesh));
        let input = f32_vector(&session, &first_mesh, &[1.0]);
        assert_eq!(meshed.device_mesh_for(&[input], Some(wide_mesh.logical_mesh())), Ok(first_mesh));
    }

    #[test]
    fn test_session_is_same_session() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        assert!(session.is_same_session(&session));
        assert!(session.is_same_session(&session.clone()));
        assert!(session.is_same_session(&session.with_new_effect_scope()));
        assert!(session.is_same_session(&session.with_device_mesh(mesh)));
        assert!(!session.is_same_session(&Session::builder().client(client).build().unwrap()));
    }

    #[test]
    fn test_session_eager_dispatch_statistics() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
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
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector = f32_vector(&session, &mesh, &[1.0, 2.0]);
        Neg::neg(&vector).unwrap();
        assert_eq!((session.state.eager_dispatch_cache.len(), session.cached_program_count()), (1, 1));

        // Clearing empties both the eager dispatch cache and the compilation cache, so the same operation misses again.
        session.clear_cache();
        assert_eq!((session.state.eager_dispatch_cache.len(), session.cached_program_count()), (0, 0));
        Neg::neg(&vector).unwrap();
        assert_eq!(session.eager_dispatch_statistics().misses, 2);
        assert_eq!((session.state.eager_dispatch_cache.len(), session.cached_program_count()), (1, 1));
    }

    #[test]
    fn test_session_array_statistics() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        assert_eq!(session.array_statistics(), ArrayStatistics { live_arrays: 0, created_arrays: 0 });

        // Creating an array counts one logical array, which its clones share.
        let vector = f32_vector(&session, &mesh, &[1.0, 2.0]);
        assert_eq!(session.array_statistics(), ArrayStatistics { live_arrays: 1, created_arrays: 1 });
        let clone = vector.clone();
        assert_eq!(session.array_statistics(), ArrayStatistics { live_arrays: 1, created_arrays: 1 });

        // The output of an eager operation is another logical array.
        let negated = Neg::neg(&vector).unwrap();
        assert_eq!(session.array_statistics(), ArrayStatistics { live_arrays: 2, created_arrays: 2 });

        // An array stays alive until its last clone is dropped, and dropped arrays stay counted as created.
        drop(vector);
        assert_eq!(session.array_statistics(), ArrayStatistics { live_arrays: 2, created_arrays: 2 });
        drop(clone);
        assert_eq!(session.array_statistics(), ArrayStatistics { live_arrays: 1, created_arrays: 2 });
        drop(negated);
        assert_eq!(session.array_statistics(), ArrayStatistics { live_arrays: 0, created_arrays: 2 });

        // All handles of a session share its counts, while another session on the same client counts separately. An
        // array keeps counting towards its original session after changing the session used for dispatch.
        let other_session = Session::builder().client(client).build().unwrap();
        let array = f32_vector(&session.with_new_effect_scope(), &mesh, &[1.0]);
        let retained = RetainedArray::new(&array);
        let array = array.with_session(&other_session).unwrap();
        assert_eq!(session.array_statistics(), ArrayStatistics { live_arrays: 1, created_arrays: 3 });
        assert_eq!(other_session.array_statistics(), ArrayStatistics { live_arrays: 0, created_arrays: 0 });

        // An array that is created from a retained array is a new logical array of the session that it is created in.
        // The retained array itself is not a logical array, even though it keeps the device buffers alive.
        let restored = retained.to_array(&other_session).unwrap();
        assert_eq!(other_session.array_statistics(), ArrayStatistics { live_arrays: 1, created_arrays: 1 });
        drop((array, restored));
        assert_eq!(session.array_statistics(), ArrayStatistics { live_arrays: 0, created_arrays: 3 });
        assert_eq!(other_session.array_statistics(), ArrayStatistics { live_arrays: 0, created_arrays: 1 });
        drop(retained);
    }

    #[test]
    fn test_session_live_arrays() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);

        // A session describes its live arrays only if it was built to track them.
        let session = Session::builder().client(client).build().unwrap();
        let vector = f32_vector(&session, &mesh, &[1.0, 2.0]);
        assert!(session.live_arrays().is_none());
        assert_eq!(session.array_statistics().live_arrays, 1);
        drop(vector);

        // All handles of a tracking session describe the same arrays.
        let session = Session::builder().client(client).live_array_tracking(true).build().unwrap();
        let vector = f32_vector(&session, &mesh, &[1.0, 2.0]);
        let live_arrays = session.with_new_effect_scope().live_arrays().unwrap();
        assert_eq!(live_arrays.len(), 1);
        assert_eq!(live_arrays[0].r#type(), vector.r#type());
        assert_eq!(live_arrays[0].addressable_size_in_bytes().unwrap(), 8);

        // The descriptions hold neither the array nor its buffers, so the array can still donate its buffers while a
        // snapshot exists, and dropping the array releases it.
        assert!(vector.has_unique_shard_buffers());
        drop(vector);
        assert!(session.live_arrays().unwrap().is_empty());
        assert_eq!(live_arrays.len(), 1);
    }

    #[test]
    fn test_session_live_array_bytes_by_device() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let device_ids = mesh.devices().iter().map(|device| device.id()).collect::<Vec<_>>();

        // A session totals the bytes of its live arrays only if it was built to track them.
        let session = Session::builder().client(client).build().unwrap();
        let vector = f32_vector(&session, &mesh, &[1.0, 2.0]);
        assert_eq!(session.live_array_bytes_by_device().unwrap(), None);
        drop(vector);

        // A replicated array holds its payload on every device, and buffers that two live arrays share count once.
        let session = Session::builder().client(client).live_array_tracking(true).build().unwrap();
        assert_eq!(session.live_array_bytes_by_device().unwrap(), Some(HashMap::new()));
        let vector = f32_vector(&session, &mesh, &[1.0, 2.0]);
        let restored = RetainedArray::new(&vector).to_array(&session).unwrap();
        assert_eq!(session.live_arrays().unwrap().len(), 2);
        assert_eq!(
            session.with_new_effect_scope().live_array_bytes_by_device().unwrap(),
            Some(HashMap::from([(device_ids[0], 8), (device_ids[1], 8)])),
        );
        drop((vector, restored));
        assert_eq!(session.live_array_bytes_by_device().unwrap(), Some(HashMap::new()));
    }

    #[test]
    fn test_session_device_memory_statistics() {
        // The built-in CPU plugin provides no allocator statistics, so every addressable device reports none.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device_ids =
            client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        assert_eq!(session.device_memory_statistics().unwrap(), vec![(device_ids[0], None), (device_ids[1], None)]);
    }

    #[test]
    fn test_session_clear_device_memory_statistics() {
        // Devices that provide no allocator statistics are skipped, so the reset succeeds on the built-in CPU plugin.
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        assert!(session.clear_device_memory_statistics().is_ok());
    }

    #[test]
    fn test_session_mesh() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.target().unwrap().devices().iter().map(TargetDevice::device).collect::<Vec<_>>();
        let x = MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap();
        let y = MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap();

        // A mesh takes the first devices of the client that its axes require, in the order of the client.
        let mesh = session.mesh([x.clone(), y.clone()]).unwrap();
        assert_eq!(mesh.logical_mesh(), &LogicalMesh::new(vec![x.clone(), y]).unwrap());
        assert_eq!(mesh.devices(), devices.as_slice());
        let mesh = session.mesh([x.clone()]).unwrap();
        assert_eq!(mesh.devices(), &devices[..2]);

        // Creating a mesh does not make it the mesh of the session.
        assert!(matches!(session.device_mesh(), Err(Error::Session(SessionError::MissingMesh))));

        // Invalid axes and meshes that need more devices than the client has are rejected.
        assert!(matches!(
            session.mesh([x.clone(), x]),
            Err(Error::Core(error))
                if *error == CoreError::Sharding(ShardingError::DuplicateMeshAxisName { name: "x".to_string() }),
        ));
        assert!(matches!(
            session.mesh([MeshAxis::new("x", 8, MeshAxisType::Auto).unwrap()]),
            Err(Error::Core(error))
                if *error == CoreError::Sharding(ShardingError::DeviceCountMismatch { expected: 8, actual: 4 }),
        ));
    }

    #[test]
    fn test_session_synchronize() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let other = Session::builder().client(client).build().unwrap();
        let scope = session.with_new_effect_scope();
        let ready = ExecutionFence::new(Vec::new());
        let mut work = scope.reserve_work();
        let reservation = scope.reserve_effects(Some(&[0])).unwrap();
        work.track_fence(ready.clone());
        work.track_error(ryft_pjrt::Error::invalid_argument("effect outputs invalid").into());
        reservation.publish(Err(Arc::from("effect outputs invalid")), &ready);
        drop(work);

        // Synchronization spans all handles of one session, even when another session uses the same PJRT client.
        other.synchronize().unwrap();
        assert!(matches!(
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(0, Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. }))]
                    if message == "effect outputs invalid"),
        ));
        scope.synchronize().unwrap();

        // The original scope's barrier succeeds because the failure belongs to the new scope. The new scope must
        // report the failure through its own barrier before acknowledgment.
        session.effects_barrier().unwrap();
        assert!(matches!(
            scope.acknowledge_effect_errors(),
            Err(Error::Execution(ExecutionError::PrematureAcknowledgment)),
        ));
        assert!(
            matches!(scope.reserve_effects(Some(&[0])), Err(Error::Execution(ExecutionError::ScopeNeedsRecovery)),)
        );
        assert!(matches!(
            scope.effects_barrier(),
            Err(Error::Execution(ExecutionError::EffectSubmissionFailures { failures }))
                if failures == vec![(0, Arc::from("effect outputs invalid"))],
        ));
        scope.acknowledge_effect_errors().unwrap();
        assert!(scope.reserve_effects(Some(&[1])).unwrap().wait_for_predecessor_tokens().unwrap().is_none());

        let _callback = EffectCallbackGuard::enter();
        assert!(matches!(session.synchronize(), Err(Error::Execution(ExecutionError::ReentrantDispatch))));
    }

    #[test]
    fn test_session_synchronize_tracks_execution_and_transfers() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let source_mesh = ArrayPlacement::device(mesh.devices()[0]).resolve(&session, 1).unwrap().mesh().clone();
        let scoped = session.with_new_effect_scope().with_device_mesh(source_mesh);

        let input = Array::from_host_buffer(
            &scoped,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1.0f32, 2.0]),
        )
        .unwrap();
        assert_eq!(session.state.work.snapshot(), 1);

        let output = Neg::neg(&input).unwrap();
        assert_eq!(session.state.work.snapshot(), 2);

        let placed = output.into_placement(&scoped, ArrayPlacement::device(mesh.devices()[1])).unwrap();
        assert_eq!(session.state.work.snapshot(), 3);

        let bytes = placed.copy_to_host_bytes().unwrap();
        assert_eq!(session.state.work.snapshot(), 4);
        drop((input, placed));

        // The original handle can wait for the submissions made through the scoped handle.
        session.synchronize().unwrap();
        assert!(!session.state.work.has_pending_work());
        assert_eq!(bytes, values_to_bytes(&[-1.0f32, -2.0]));
    }

    #[test]
    fn test_session_effects_barrier() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();

        // A barrier reports the failures of earlier submissions, and keeps reporting them until they are acknowledged.
        let reservation = session.reserve_effects(None).unwrap();
        reservation.publish(Err(Arc::from("effect failed")), &ExecutionFence::new(Vec::new()));
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
        let session = Session::builder().client(client).build().unwrap();
        let reservation = session.reserve_effects(None).unwrap();
        reservation.publish(Err(Arc::from("effect failed")), &ExecutionFence::new(Vec::new()));

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
    fn test_session_clone() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let clone = session.clone();
        assert!(clone.is_same_session(&session));
        assert!(shares_effect_scope(&clone, &session));
        assert_eq!(clone.device_mesh().unwrap(), &mesh);

        // The shared state of a session lives for as long as any of its handles.
        let state = Arc::downgrade(&session.state);
        drop(session);
        assert!(state.upgrade().is_some());
        drop(clone);
        assert!(state.upgrade().is_none());
    }

    #[test]
    fn test_session_named_axis() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        assert!(session.named_axis("x").is_none());
        assert!(session.with_device_mesh(mesh).named_axis("x").is_none());
    }

    #[test]
    fn test_session_named_axes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        assert!(session.named_axes().is_empty());
        assert!(session.with_device_mesh(mesh).named_axes().is_empty());
    }

    #[test]
    fn test_session_default_mesh() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        assert_eq!(session.default_mesh(), None);
        assert_eq!(session.with_device_mesh(mesh.clone()).default_mesh(), Some(mesh.logical_mesh().clone()));
    }

    #[test]
    fn test_session_lift() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();

        // Immediate dimension extents materialize directly on the host, and Boolean literals upload a scalar.
        let extent = DimensionValue::constant(3).unwrap();
        assert_eq!(
            session.lift(Constant::Dimension(extent.clone())),
            Ok(Value::Dimension(Dimension::new(extent, session.clone()))),
        );
        let predicate = session.lift(Constant::Boolean(true)).unwrap();
        assert_eq!(program_array(&predicate).data_type(), DataType::Boolean);
        assert_eq!(program_array(&predicate).shape(), StaticShape::new(Vec::new()));
        assert_eq!(read_booleans(client, program_array(&predicate)), vec![true]);
        assert_eq!(session.cached_program_count(), 0);

        // A captured constant carries only a type, so it cannot be lifted without its capture table.
        let capture = CaptureReference::new(0, Type::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            session.lift(Constant::Captured(capture)),
            Err(TypeError::invalid(
                "XLA captured constant `capture#0:f32[]` requires a captured program capture table"
            )
            .into()),
        );
    }

    #[test]
    fn test_session_bind() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let domain = ProjectedContext::<Session, ArrayType>::new(Session::builder().client(client).build().unwrap());

        // A unary operation executes on its input.
        let input = f32_vector(domain.parent(), &mesh, &[1.0, -2.0, 3.5, 0.0]);
        let outputs = domain.bind(NegOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![-1.0, 2.0, -3.5, 0.0]);

        // A binary operation executes on both of its inputs.
        let left = f32_vector(domain.parent(), &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let right = f32_vector(domain.parent(), &mesh, &[10.0, 20.0, 30.0, 40.0]);
        let outputs = domain.bind(AddOperation::new(), Vec::new(), &[left, right]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_session_bind_rejects_managed_callbacks() {
        let session = Session::builder().client(execution_client()).build().unwrap();
        let _guard = EffectCallbackGuard::enter();
        assert!(matches!(
            session.bind(OneOperation::new(ArrayType::scalar(DataType::F32)), Vec::new(), &[]),
            Err(ProgramError::InvalidArgument { message })
                if message == "execution and effect barriers cannot be entered from a managed callback",
        ));
    }

    #[test]
    fn test_session_bind_identity_rejects_unrepresentable_type() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let array_type = ArrayType::scalar(DataType::Token);

        assert!(matches!(
            session.bind(OneOperation::new(array_type.clone()), Vec::new(), &[]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "data type `token` cannot represent one"
        ));
    }

    #[test]
    fn test_session_bind_identity_rejects_attached_regions() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh);
        let array_type = ArrayType::scalar(DataType::F32);
        let operations = [
            Operation::provide(ZeroOperation::new(Type::Array(array_type.clone())), &[]).unwrap(),
            OneOperation::new(array_type).into(),
        ];

        for operation in operations {
            let name = operation.name();
            let attached_region = ProgramBuilder::new()
                .build::<Vec<Constant>, Vec<Constant>>(Vec::new(), Vec::new(), Vec::new())
                .unwrap();
            assert_eq!(
                session.bind(operation, vec![attached_region], &[]),
                Err(ProgramError::MalformedProgram(format!(
                    "operation `{name}` declares no region slots but 1 regions were attached",
                ))),
            );
        }
    }

    #[test]
    fn test_session_bind_rejects_unresolved_references_before_special_cases_or_program_construction() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();

        // Top-level reference operations execute eagerly on their handles on the host, so they validate their inputs
        // like the core reference implementations do. The `values` tests cover their full lifecycle.
        assert_eq!(
            session.bind(Operation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        let reference =
            Value::Reference(Reference::new(ArrayReference::new(f32_vector(&session, &mesh, &[1.0])), session.clone()));
        assert_eq!(
            session.bind(AddOperation::new(), Vec::new(), &[reference]),
            Err(ProgramError::UnsupportedOperation {
                message: "eager XLA execution lowers each operation and cannot bind a reference operand; discharge \
                          references before lowering"
                    .to_string(),
            }),
        );

        // State carried by regions must be rejected too. Control-flow operations report pure intrinsic effects, so the
        // guard also scans the closures of their attached regions to see a reference operation inside a loop body.
        let condition = {
            let mut builder = ProgramBuilder::new();
            let predicate = builder
                .add_instruction(ZeroOperation::new(ArrayType::scalar(DataType::Boolean)), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![predicate], Vec::new(), vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::new();
            let state = builder
                .add_instruction(ZeroOperation::new(ArrayType::scalar(DataType::F32)), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap();
            builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), Vec::new(), Vec::new()).unwrap()
        };
        assert_eq!(
            session.bind(Operation::While(WhileOperation::new()), vec![condition, body], &[]),
            Err(ProgramError::UnsupportedOperation {
                message: "`while` carries reference state that XLA eager execution cannot lower; discharge \
                          references before lowering"
                    .to_string(),
            }),
        );

        // Reference state confined to a dormant custom derivative rule region is rejected too. Ordinary XLA rejects
        // unresolved reference artifacts across the whole program, so eager binding matches that policy instead of
        // deferring the failure to compilation with a less targeted diagnostic.
        let primal = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(ArrayType::scalar(DataType::F32).into());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![state], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let rule = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let state_tangent = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![output, state_tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        assert_eq!(
            session.bind(
                Operation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false)),
                vec![primal.clone(), rule],
                &[],
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "`composite` carries reference state that XLA eager execution cannot lower; \
                          discharge references before lowering"
                    .to_string(),
            }),
        );

        // A pure capture of a reference type in the dormant rule has no `OrderedState` effect, so only the type scan
        // over the complete arena can keep the eager and compiled XLA rejection policies aligned.
        let pure_reference_rule = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let state_tangent = builder.add_input(ArrayType::scalar(DataType::F32).into());
            builder.add_constant(Constant::Captured(CaptureReference::new(
                0,
                Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            )));
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![state, state_tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        assert_eq!(
            session.bind(
                Operation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false)),
                vec![primal, pure_reference_rule],
                &[],
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "`composite` carries reference state that XLA eager execution cannot lower; \
                          discharge references before lowering"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_session_bind_dimension_operations_run_on_the_host_without_compilation() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh);
        let left = DimensionValue::constant(2).unwrap();
        let right = DimensionValue::constant(3).unwrap();
        let add = DimensionAddOperation::new(left.r#type().as_ref(), right.r#type().as_ref()).unwrap();

        let output = session
            .bind(
                Operation::Dimension(DimensionOperation::Add(add)),
                Vec::new(),
                &[
                    Value::Dimension(Dimension::new(left, session.clone())),
                    Value::Dimension(Dimension::new(right, session.clone())),
                ],
            )
            .unwrap()
            .remove(0);
        let Value::Dimension(output) = output else {
            panic!("dimension arithmetic must produce a dimension value");
        };
        assert_eq!(output.value().extent(), 5);
        assert_eq!(session.cached_program_count(), 0);

        let scalar = session
            .bind(DimensionToScalarOperation, Vec::new(), &[Value::Dimension(output.clone())])
            .unwrap()
            .remove(0);
        let Value::Array(scalar) = scalar else {
            panic!("dimension_to_scalar must produce an array");
        };
        assert_eq!(read_first_shard::<i64>(client, &scalar), vec![5]);
        assert_eq!(session.cached_program_count(), 0);

        let from_scalar = session
            .bind(DimensionFromScalarOperation::new(output.r#type().into_owned()), Vec::new(), &[Value::Array(scalar)])
            .unwrap()
            .remove(0);
        let Value::Dimension(from_scalar) = from_scalar else {
            panic!("dimension_from_scalar must produce a dimension value");
        };
        assert_eq!(from_scalar.value().extent(), 5);
        assert_eq!(session.cached_program_count(), 0);
    }

    #[test]
    fn test_session_bind_uploads_literal_constants_without_compilation() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();

        // A literal is uploaded from its host bytes, which hold one element per byte for the sub-byte data types.
        let outputs = session
            .bind(ConstantOperation::new(CoreArray::vector(vec![1.5f32, -2.0, 0.25]).unwrap()), Vec::new(), &[])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![1.5, -2.0, 0.25]);
        let predicates = session
            .bind(ConstantOperation::new(CoreArray::vector(vec![true, false]).unwrap()), Vec::new(), &[])
            .unwrap();
        assert_eq!(read_booleans(client, program_array(&predicates[0])), vec![true, false]);
        let nibbles = CoreArray::vector(vec![i4::new(-8).unwrap(), i4::new(7).unwrap()]).unwrap();
        let nibbles = session.bind(ConstantOperation::new(nibbles), Vec::new(), &[]).unwrap();
        assert_eq!(program_array(&nibbles[0]).data_type(), DataType::I4);
        assert_eq!(program_array(&nibbles[0]).copy_to_host_bytes().unwrap(), vec![0x08, 0x07]);

        // A literal with a layout is stored in the order of that layout and is uploaded in logical row-major order.
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let r#type = ArrayType::new_static(DataType::F64, [2, 3]).with_layout(layout.clone());
        let literal = CoreArray::from_elements(r#type, &[1f64, 2., 3., 4., 5., 6.]).unwrap();
        assert_ne!(literal.storage_bytes(), literal.logical_bytes().as_slice());
        let outputs = session.bind(ConstantOperation::new(literal), Vec::new(), &[]).unwrap();
        assert_eq!(program_array(&outputs[0]).layout(), Some(&layout));
        assert_eq!(
            program_array(&outputs[0]).copy_to_host().unwrap().elements::<f64>().unwrap(),
            vec![1., 2., 3., 4., 5., 6.],
        );

        // None of the literals above compiled a program, and a literal takes no inputs.
        assert_eq!((session.state.eager_dispatch_cache.len(), session.cached_program_count()), (0, 0));
        assert_eq!(
            session.bind(ConstantOperation::new(CoreArray::scalar(1f32).unwrap()), Vec::new(), &outputs),
            Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
        );

        // Every value that `Fill` is given therefore shares the one program that broadcasts its literal.
        let context = ProjectedContext::<Session, ArrayType>::new(session.clone());
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        assert_eq!(read_first_shard::<f32>(client, &context.fill(&r#type, 2.5f32).unwrap()), vec![2.5, 2.5]);
        assert_eq!(read_first_shard::<f32>(client, &context.fill(&r#type, 3.5f32).unwrap()), vec![3.5, 3.5]);
        assert_eq!(session.cached_program_count(), 1);
    }

    #[test]
    fn test_session_bind_place() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let first_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let second_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
        let input = f32_vector(&session, &first_mesh, &[1.0, 2.0, 3.0, 4.0]);
        let source = Placement::new(first_mesh.clone(), replicated.clone()).unwrap();
        let output_type = |operation: PlaceOperation| operation.infer_output_types(&[input.r#type().into_owned()], &[]);

        // The array moves to the device of the target mesh, and its type is the one that type inference returns.
        let target = Placement::new(second_mesh.clone(), replicated.clone()).unwrap();
        let moved = input.place(&target, None).unwrap();
        assert_eq!(moved.mesh(), &second_mesh);
        assert_eq!(moved.sharding(), &replicated);
        assert_eq!(
            vec![moved.r#type().into_owned()],
            output_type(PlaceOperation::new(target.clone(), Some(source.clone()))).unwrap(),
        );
        assert_eq!(moved.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        // A target memory is recorded in the type of the result and selects the memory of its buffers.
        let memory = Memory::Host { pinned: true };
        let pinned = input.place(&target.clone().with_memory(memory), None).unwrap();
        assert_eq!(pinned.r#type().memory(), memory);
        assert_eq!(pinned.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        assert_eq!(pinned.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        // A target over both devices replicates the array on them.
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let target = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let replicated = input.place(&target, None).unwrap();
        assert_eq!(replicated.mesh(), &mesh);
        assert_eq!(read_first_shard::<f32>(client, &replicated), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(replicated.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        // Whole shards moved in all of the placements above, so none of them compiled a program.
        assert_eq!((session.state.eager_dispatch_cache.len(), session.cached_program_count()), (0, 0));

        // The array is moved from where it actually lives, whatever devices the source names.
        let other_source = Placement::new(second_mesh.clone(), source.sharding().clone()).unwrap();
        assert_eq!(input.place(&target, Some(&other_source)).unwrap().mesh(), &mesh);
        assert_eq!(input.place(&target, Some(&source)).unwrap().mesh(), &mesh);

        // A layout that the target omits is kept, also when the shards themselves change.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let sharded = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]).with_layout(layout.clone()),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        let target =
            Placement::new(first_mesh.clone(), Sharding::replicated(first_mesh.logical_mesh().clone(), 1)).unwrap();
        let gathered = sharded.place(&target, None).unwrap();
        assert_eq!(gathered.mesh(), &first_mesh);
        assert_eq!(gathered.layout(), Some(&layout));
        assert_eq!(
            vec![gathered.r#type().into_owned()],
            PlaceOperation::new(target, Some(Placement::new(mesh, sharding).unwrap().with_layout(layout)))
                .infer_output_types(&[sharded.r#type().into_owned()], &[])
                .unwrap(),
        );
        assert_eq!(gathered.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_session_bind_place_may_alias() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let first_mesh = DeviceMesh::new(logical_mesh.clone(), vec![mesh.devices()[0]]).unwrap();
        let input = f32_vector(&session, &first_mesh, &[1.0, 2.0, 3.0, 4.0]);
        let place = |target: &Placement, may_alias: bool| {
            <Array as CoreValue>::Dispatch::place(&input, target, None, may_alias).unwrap()
        };
        let shares_input_buffer = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| Arc::ptr_eq(shard.buffer().unwrap(), input.shards()[0].buffer().unwrap()))
                .collect::<Vec<_>>()
        };

        // A placement that leaves the array where it is returns the array itself unless it must not alias its input,
        // in which case it returns a copy of its buffers with the same type, placement, and values.
        let target = Placement::new(first_mesh.clone(), Sharding::replicated(logical_mesh, 1)).unwrap();
        assert_eq!(shares_input_buffer(&place(&target, true)), vec![true]);
        let copied = place(&target, false);
        assert_eq!(copied.r#type(), input.r#type());
        assert_eq!(copied.mesh(), &first_mesh);
        assert_eq!(shares_input_buffer(&copied), vec![false]);
        assert_eq!(copied.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        // A shard that stays on its device keeps the buffer of the input unless the result must not alias its input.
        let target = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        assert_eq!(shares_input_buffer(&place(&target, true)), vec![true, false]);
        let copied = place(&target, false);
        assert_eq!(copied.mesh(), &mesh);
        assert_eq!(shares_input_buffer(&copied), vec![false, false]);
        assert_eq!(read_first_shard::<f32>(client, &copied), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(copied.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_session_bind_place_rejects_arrays_of_other_clients() {
        // Both clients have a device with ID 0, so only the identity of the client tells the two arrays apart.
        let client = cpu_client(1);
        let other_client = other_cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let other_session = Session::builder().client(other_client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let other_mesh = domain_mesh(other_client, "x", 1);
        assert_eq!(mesh, other_mesh);
        let target = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let input = f32_vector(&other_session, &other_mesh, &[1.0, 2.0]);
        assert_eq!(
            session.bind(PlaceOperation::new(target, None), Vec::new(), &[Value::Array(input)]),
            Err(ProgramError::InvalidArgument {
                message: "received incompatible devices for eager XLA execution: input #0 is owned by a different \
                          PJRT client than this session's client"
                    .to_owned(),
            }),
        );
    }

    #[test]
    fn test_session_bind_mixed_shape_operations_specialize_dimensions_and_share_the_array_kernel_cache() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let input = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let extent = session
            .bind(
                DimensionSizeOperation::new(input.r#type().as_ref(), 0).unwrap(),
                Vec::new(),
                &[Value::Array(input.clone())],
            )
            .unwrap()
            .remove(0);
        let Value::Dimension(extent) = extent else {
            panic!("dimension_size must produce a dimension value");
        };
        assert_eq!(extent.value().extent(), 6);
        assert_eq!(session.cached_program_count(), 0);

        let two = DimensionValue::constant(2).unwrap();
        let division = DimensionDivOperation::new(extent.r#type().as_ref(), two.r#type().as_ref()).unwrap();
        let three = session
            .bind(
                Operation::Dimension(DimensionOperation::Div(division)),
                Vec::new(),
                &[Value::Dimension(extent), Value::Dimension(Dimension::new(two.clone(), session.clone()))],
            )
            .unwrap()
            .remove(0);
        let Value::Dimension(three) = three else {
            panic!("dimension division must produce a dimension value");
        };
        assert_eq!(three.value().extent(), 3);
        assert_eq!(session.cached_program_count(), 0);

        let reshape_inputs = [
            Value::Array(input),
            Value::Dimension(Dimension::new(two.clone(), session.clone())),
            Value::Dimension(three.clone()),
        ];
        let reshaped = session.bind(DynamicReshapeOperation::new(), Vec::new(), &reshape_inputs).unwrap().remove(0);
        assert_eq!(program_array(&reshaped).shape().as_slice(), &[2, 3]);
        assert_eq!(session.cached_program_count(), 1);

        let four = DimensionValue::constant(4).unwrap();
        let broadcast_inputs = [
            reshaped,
            Value::Dimension(Dimension::new(four, session.clone())),
            Value::Dimension(Dimension::new(two, session.clone())),
            Value::Dimension(three),
        ];
        let broadcast = session
            .bind(DynamicBroadcastOperation::new(vec![1, 2]), Vec::new(), &broadcast_inputs)
            .unwrap()
            .remove(0);
        assert_eq!(program_array(&broadcast).shape().as_slice(), &[4, 2, 3]);
        assert_eq!(
            read_first_shard::<f32>(client, program_array(&broadcast)),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].repeat(4)
        );
        assert_eq!(session.cached_program_count(), 2);

        let repeated = session
            .bind(DynamicBroadcastOperation::new(vec![1, 2]), Vec::new(), &broadcast_inputs)
            .unwrap()
            .remove(0);
        assert_eq!(program_array(&repeated).shape().as_slice(), &[4, 2, 3]);
        assert_eq!(session.cached_program_count(), 2);
    }

    #[test]
    fn test_session_bind_executes_dimension_cast() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let selected = session.with_new_effect_scope();
        let value = Value::Dimension(Dimension::new(DimensionValue::constant(2).unwrap(), session.clone()));
        let output_type = Type::Dimension(DimensionType::from(CoreDimension::Static(2)));
        for checked in [true, false] {
            let output = if checked {
                value.cast(output_type.clone(), &[])
            } else {
                value.cast_unchecked(output_type.clone(), &[])
            }
            .unwrap();
            assert_eq!(output, value);
            assert!(output.session().is_same_session(&session));
            assert!(shares_effect_scope(output.session(), &session));

            let operation = if checked {
                CastOperation::checked(output_type.clone(), vec![output_type.clone()]).unwrap()
            } else {
                CastOperation::unchecked(output_type.clone(), vec![output_type.clone()]).unwrap()
            };
            let outputs = selected.bind(operation, Vec::new(), std::slice::from_ref(&value)).unwrap();
            assert_eq!(outputs, vec![value.clone()]);
            assert!(shares_effect_scope(outputs[0].session(), &selected));
        }
        assert_eq!(session.cached_program_count(), 0);
    }

    #[test]
    fn test_session_bind_dimension_cast_validates_declared_relationships() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let value = Value::Dimension(Dimension::new(DimensionValue::constant(2).unwrap(), session.clone()));
        let output_type = Type::Dimension(DimensionType::from(CoreDimension::Static(2)));
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let declared_types = vec![
            DimensionType::from(extent.clone()).into(),
            ArrayType::new(DataType::F32, Shape::new(vec![extent.into()])).into(),
        ];
        let valid = Value::Array(f32_vector(&session, &mesh, &[1.0, 2.0]));
        let invalid = Value::Array(f32_vector(&session, &mesh, &[1.0, 2.0, 3.0]));
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let operation = constructor(output_type.clone(), declared_types.clone()).unwrap();
            assert_eq!(
                session.bind(operation.clone(), Vec::new(), &[value.clone(), valid.clone()]),
                Ok(vec![value.clone()]),
            );
            assert_eq!(
                session.bind(operation.clone(), Vec::new(), &[value.clone(), invalid.clone()]),
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "extent".to_owned(),
                    expected: 2,
                    actual: 3,
                }))),
            );
            assert_eq!(
                session.bind(operation, Vec::new(), std::slice::from_ref(&value)),
                Err(TypeError::invalid("expected 2 inputs but got 1").into()),
            );
            let mismatch = Type::Dimension(DimensionType::from(CoreDimension::Static(3)));
            assert_eq!(
                session.bind(
                    constructor(mismatch, vec![output_type.clone()]).unwrap(),
                    Vec::new(),
                    std::slice::from_ref(&value)
                ),
                Err(TypeError::invalid("`cast` cannot convert type `dimension<2>` to unrelated type `dimension<3>`")
                    .into()),
            );
        }
        let region = ProgramBuilder::new()
            .build::<Vec<Constant>, Vec<Constant>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        assert_eq!(
            session.bind(
                CastOperation::checked(output_type.clone(), vec![output_type]).unwrap(),
                vec![region],
                &[value],
            ),
            Err(ProgramError::MalformedProgram(
                "operation `cast` declares no region slots but 1 regions were attached".to_owned(),
            )),
        );
        assert_eq!(session.cached_program_count(), 0);
    }

    #[test]
    fn test_session_bind_dimension_cast_preserves_witness_errors() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let value = Value::Dimension(Dimension::new(DimensionValue::constant(2).unwrap(), session.clone()));
        let output_type = Type::Dimension(DimensionType::from(CoreDimension::Static(2)));
        for data_type in [DataType::F32, DataType::Zero] {
            let (event, promise) = client.event(()).unwrap();
            promise.set(Some(ryft_pjrt::Error::invalid_argument("cast witness failed"))).unwrap();
            let bytes = if data_type.is_zero() { Vec::new() } else { values_to_bytes::<f32>(&[1.0, 2.0]) };
            let witness = Value::Array(
                Array::from_host_buffer(&session, ArrayType::new_static(data_type, [2]), mesh.clone(), None, bytes)
                    .unwrap()
                    .with_execution_fence(ExecutionFence::new(vec![event])),
            );
            for checked in [true, false] {
                let wrapped = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
                let Value::Array(array) = &witness else { unreachable!() };
                let witnesses = [ArrayIrValue::Array(array.clone())];
                let result = if checked {
                    wrapped.cast(output_type.clone(), &witnesses)
                } else {
                    wrapped.cast_unchecked(output_type.clone(), &witnesses)
                };
                assert_eq!(
                    result,
                    Err(ProgramError::Concretization {
                        message: "host dimension `cast` cannot retain an array witness producer; bind the cast \
                                  through an array IR context"
                            .to_owned(),
                    })
                );
                let result = if checked {
                    value.cast(output_type.clone(), std::slice::from_ref(&witness))
                } else {
                    value.cast_unchecked(output_type.clone(), std::slice::from_ref(&witness))
                };
                assert_eq!(result, Err(ProgramError::InvalidArgument { message: "cast witness failed".to_owned() }),);
            }
        }
        assert_eq!(session.cached_program_count(), 0);
    }

    #[test]
    fn test_session_bind_dispatch_cache_skips_program_construction_and_lowering() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let left = f32_vector(&session, &mesh, &[1.0, 2.0]);
        let right = f32_vector(&session, &mesh, &[3.0, 4.0]);
        let inputs = [Value::Array(left), Value::Array(right)];

        let first = session.bind(Operation::Array(AddOperation::new().into()), Vec::new(), &inputs).unwrap();
        let second = session.bind(Operation::Array(AddOperation::new().into()), Vec::new(), &inputs).unwrap();
        assert_eq!(read_first_shard::<f32>(client, program_array(&first[0])), vec![4.0, 6.0]);
        assert_eq!(read_first_shard::<f32>(client, program_array(&second[0])), vec![4.0, 6.0]);
        let statistics = session.eager_dispatch_statistics();
        assert_eq!((statistics.misses, statistics.hits, statistics.productions), (1, 1, 1));

        // The hit never reached the compilation cache, so it skipped program construction, lowering, and key
        // construction.
        let compilation_statistics = session.compilation_cache().statistics();
        assert_eq!((compilation_statistics.misses, compilation_statistics.memory_hits), (1, 0));
        assert_eq!(compilation_statistics.compilations, 1);
    }

    #[test]
    fn test_session_bind_dispatch_cache_keys_every_lowering_input() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap();
        fn negate(session: &Session, input: Array) {
            let inputs = [Value::Array(input)];
            session.bind(Operation::Array(NegOperation::new().into()), Vec::new(), &inputs).unwrap();
        }
        let misses = |session: &Session| session.eager_dispatch_statistics().misses;

        let vector = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        negate(&session, vector.clone());
        negate(&session, vector.clone());
        assert_eq!(misses(&session), 1);

        // A different shape, element type, or sharding is a different key. That includes runtime placements over auto
        // mesh axes, which array types do not record.
        negate(&session, f32_vector(&session, &mesh, &[1.0, 2.0]));
        assert_eq!(misses(&session), 2);
        let f64_type = ArrayType::new(DataType::F64, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        negate(
            &session,
            Array::from_host_buffer(&session, f64_type, mesh.clone(), None, values_to_bytes(&[1f64; 4])).unwrap(),
        );
        assert_eq!(misses(&session), 3);
        let sharded = Array::from_host_buffer(
            &session,
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)])),
            mesh.clone(),
            Some(Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()),
            values_to_bytes(&[1f32; 4]),
        )
        .unwrap();
        assert_eq!(sharded.r#type(), vector.r#type());
        negate(&session, sharded);
        assert_eq!(misses(&session), 4);

        // Dimension inputs are keyed by their extents, because eager binding bakes them into the built program as
        // constants of exact types. Dynamic reshapes execute on a single device, so this part uses a session with a
        // single device.
        let single_device_client = cpu_client(1);
        let single_device_mesh = domain_mesh(single_device_client, "x", 1);
        let single_device_session = Session::builder().client(single_device_client).build().unwrap();
        let single_device_vector = f32_vector(&single_device_session, &single_device_mesh, &[1.0, 2.0, 3.0, 4.0]);
        let reshape = |rows: &DimensionValue, columns: &DimensionValue| {
            let inputs = [
                Value::Array(single_device_vector.clone()),
                Value::Dimension(Dimension::new(rows.clone(), single_device_session.clone())),
                Value::Dimension(Dimension::new(columns.clone(), single_device_session.clone())),
            ];
            single_device_session.bind(DynamicReshapeOperation::new(), Vec::new(), &inputs).unwrap().remove(0)
        };
        let rows = DimensionValue::constant(2).unwrap();
        let columns = DimensionValue::constant(2).unwrap();
        assert_eq!(program_array(&reshape(&rows, &columns)).shape().as_slice(), &[2, 2]);
        assert_eq!(program_array(&reshape(&rows, &columns)).shape().as_slice(), &[2, 2]);
        assert_eq!(misses(&single_device_session), 1);
        let one = DimensionValue::constant(1).unwrap();
        let four = DimensionValue::constant(4).unwrap();
        assert_eq!(program_array(&reshape(&one, &four)).shape().as_slice(), &[1, 4]);
        assert_eq!(misses(&single_device_session), 2);
        // A separately created dimension value with the same extent, and hence a distinct identity, is the same key.
        let other_rows = DimensionValue::constant(2).unwrap();
        assert_eq!(program_array(&reshape(&other_rows, &columns)).shape().as_slice(), &[2, 2]);
        assert_eq!(misses(&single_device_session), 2);
    }

    #[test]
    fn test_session_bind_dispatch_cache_keys_attached_regions_structurally() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 2);
        let branch = |operation: Operation| {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(operation, Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let inputs = [
            Value::Array(boolean_scalar(&session, &mesh, true)),
            Value::Array(f32_vector(&session, &mesh, &[1.0, 2.0])),
        ];

        // Branches that are rebuilt for every application but are structurally equal share one cache entry.
        for _ in 0..2 {
            let branches = [
                branch(Operation::Array(AddOperation::new().into())),
                branch(Operation::Array(MulOperation::new().into())),
            ];
            let outputs = session.bind(Operation::Condition(ConditionOperation::new()), branches, &inputs).unwrap();
            assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![2.0, 4.0]);
        }
        let statistics = session.eager_dispatch_statistics();
        assert_eq!((statistics.misses, statistics.hits), (1, 1));

        // A structurally different branch is a different key.
        let branches = [
            branch(Operation::Array(SubOperation::new().into())),
            branch(Operation::Array(MulOperation::new().into())),
        ];
        let outputs = session.bind(Operation::Condition(ConditionOperation::new()), branches, &inputs).unwrap();
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![0.0, 0.0]);
        assert_eq!(session.eager_dispatch_statistics().misses, 2);
        assert_eq!(session.compilation_cache().statistics().compilations, 2);
    }

    #[test]
    fn test_session_bind_dispatch_cache_does_not_retain_failures() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let inputs = [
            Value::Array(f32_vector(&session, &mesh, &[1.0, 2.0])),
            Value::Array(f32_vector(&session, &mesh, &[1.0, 2.0, 3.0])),
        ];
        for _ in 0..2 {
            assert_eq!(
                session.bind(Operation::Array(AddOperation::new().into()), Vec::new(), &inputs),
                Err(TypeError::invalid("`add` input types are not broadcast-compatible").into()),
            );
        }
        let statistics = session.eager_dispatch_statistics();
        assert_eq!((statistics.misses, statistics.hits), (2, 0));
        assert_eq!((statistics.productions, statistics.abandoned_productions), (0, 2));
    }

    #[test]
    fn test_session_bind_dispatch_cache_hits_still_validate_placement() {
        let client = execution_client();
        let other_client = other_cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let other_session = Session::builder().client(other_client).build().unwrap();
        let local = f32_vector(&session, &mesh, &[1.0, 2.0]);
        let foreign = f32_vector(&other_session, &mesh, &[3.0, 4.0]);
        let operation = Operation::Array(AddOperation::new().into());
        let local_inputs = [Value::Array(local.clone()), Value::Array(local.clone())];
        session.bind(operation.clone(), Vec::new(), &local_inputs).unwrap();

        // The second input's type and the first input's mesh match the cached entry, so this lookup hits the cache,
        // and placement validation must still reject the foreign input.
        let mixed_inputs = [Value::Array(local), Value::Array(foreign)];
        assert_eq!(
            session.bind(operation, Vec::new(), &mixed_inputs),
            Err(ProgramError::InvalidArgument {
                message: "received incompatible devices for eager XLA execution: input #1 is owned by a different \
                          PJRT client than this session's client"
                    .to_owned(),
            }),
        );
        assert_eq!(session.eager_dispatch_statistics().hits, 1);
    }

    #[test]
    fn test_session_bind_executes_on_the_mesh_of_its_inputs() {
        // An operation executes on the mesh of its array inputs, even when the handle has another default mesh.
        let client = execution_client();
        let default_mesh = domain_mesh(client, "x", 1);
        let input_mesh = domain_mesh(client, "y", 1);
        let domain = ProjectedContext::<Session, ArrayType>::new(
            Session::builder().client(client).build().unwrap().with_device_mesh(default_mesh),
        );
        let input = f32_vector(domain.parent(), &input_mesh, &[1.0, -2.0]);
        let outputs = domain.bind(NegOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), replicated_vector_type(&input_mesh, 2));
        assert_eq!(outputs[0].mesh(), &input_mesh);
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![-1.0, 2.0]);

        // The same holds for an input on fewer devices than the default mesh of the handle spans.
        let client = cpu_client(2);
        let default_mesh = two_device_mesh(client, MeshAxisType::Auto);
        let input_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![default_mesh.devices()[0]],
        )
        .unwrap();
        let domain = ProjectedContext::<Session, ArrayType>::new(
            Session::builder().client(client).build().unwrap().with_device_mesh(default_mesh),
        );
        let input = f32_vector(domain.parent(), &input_mesh, &[1.0, -2.0]);
        let outputs = domain.bind(NegOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].mesh(), &input_mesh);
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![-1.0, 2.0]);

        // An input on a device other than the first one of the client stays on that device.
        let input_mesh = DeviceMesh::new(
            input_mesh.logical_mesh().clone(),
            vec![domain.parent().device_mesh().unwrap().devices()[1]],
        )
        .unwrap();
        let input = f32_vector(domain.parent(), &input_mesh, &[1.0, -2.0]);
        let outputs = domain.bind(NegOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].mesh(), &input_mesh);
        assert_eq!(read_sharded_f32_array(&outputs[0]), vec![-1.0, 2.0]);
    }

    #[test]
    fn test_session_bind_executes_promoted_and_broadcast_elementwise_operations() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let domain = ProjectedContext::<Session, ArrayType>::new(Session::builder().client(client).build().unwrap());

        let scalar = f32_scalar(domain.parent(), &mesh, 2.0);
        let vector = f64_vector(domain.parent(), &mesh, &[1.0, 2.0, 4.0, 8.0]);

        let divide = domain.bind(DivOperation::new(), Vec::new(), &[scalar.clone(), vector.clone()]).unwrap();
        assert_eq!(read_first_shard::<f64>(client, &divide[0]), vec![2.0, 1.0, 0.5, 0.25]);

        let atan2 = domain.bind(Atan2Operation::new(), Vec::new(), &[vector.clone(), scalar.clone()]).unwrap();
        for (actual, expected) in read_first_shard::<f64>(client, &atan2[0]).into_iter().zip([
            1.0f64.atan2(2.0),
            2.0f64.atan2(2.0),
            4.0f64.atan2(2.0),
            8.0f64.atan2(2.0),
        ]) {
            assert!((actual - expected).abs() < 1e-12, "got {actual}, expected {expected}");
        }

        let compare = domain
            .bind(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), &[scalar.clone(), vector.clone()])
            .unwrap();
        assert_eq!(read_booleans(client, &compare[0]), vec![false, false, true, true]);

        let select = domain
            .bind(SelectOperation::new(), Vec::new(), &[boolean_scalar(domain.parent(), &mesh, true), scalar, vector])
            .unwrap();
        assert_eq!(read_first_shard::<f64>(client, &select[0]), vec![2.0, 2.0, 2.0, 2.0]);

        let boolean_vector = boolean_vector(domain.parent(), &mesh, &[true, false, true, false]);
        let and = domain
            .bind(AndOperation::new(), Vec::new(), &[boolean_scalar(domain.parent(), &mesh, true), boolean_vector])
            .unwrap();
        assert_eq!(read_booleans(client, &and[0]), vec![true, false, true, false]);
    }

    #[test]
    fn test_session_bind_executes_elementwise_operation_on_sharded_inputs() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let domain = ProjectedContext::<Session, ArrayType>::new(Session::builder().client(client).build().unwrap());

        // A vector sharded over a mesh with 2 devices executes eagerly through SPMD compilation of the operation, and
        // each device adds its own shard of 2 elements.
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let input = Array::from_host_buffer(
            domain.parent(),
            input_type,
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.bind(AddOperation::new(), Vec::new(), &[input.clone(), input.clone()]).unwrap();

        assert_eq!(outputs.len(), 1);
        let output = &outputs[0];
        // The input placement propagates to the output, so the result stays sharded over the same mesh axis, while
        // its type, like that of the input, records no placement over that auto mesh axis.
        assert_eq!(output.r#type(), input.r#type());
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(output.shape(), StaticShape::new(vec![4]));
        assert_eq!(output.shards().len(), 2);
        let shard_values = output
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.copy_to_host_bytes().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(shard_values, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);

        // An operation that changes the element type, here the complex absolute value, keeps the runtime placement of
        // its input while its type records the new element type.
        let input_type = ArrayType::new(DataType::C64, Shape::new(vec![CoreDimension::Static(4)]));
        let values =
            [Complex::new(3f32, 4.0), Complex::new(5.0, 12.0), Complex::new(8.0, 15.0), Complex::new(7.0, 24.0)];
        let input = Array::from_host_buffer(
            domain.parent(),
            input_type,
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&values).as_slice(),
        )
        .unwrap();
        let outputs = domain.bind(AbsOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        let output = &outputs[0];
        let expected_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(sharding.without_auto_axes())
            .unwrap();
        assert_eq!(output.r#type().as_ref(), &expected_type);
        assert_eq!(output.sharding(), &sharding);

        // XLA computes complex magnitudes with a scaled square root that may differ from the exact value in the last
        // bit (e.g., `|5 + 12i|` evaluates to `12.999999`).
        let magnitudes = output.copy_to_host().unwrap().elements::<f32>().unwrap();
        assert_eq!(magnitudes.len(), 4);
        for (actual, expected) in magnitudes.into_iter().zip([5f32, 13.0, 17.0, 25.0]) {
            assert!((actual - expected).abs() <= 1e-5 * expected, "expected {expected} but got {actual}");
        }
    }

    #[test]
    fn test_session_bind_surfaces_shape_mismatch_as_type_error() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let domain = ProjectedContext::<Session, ArrayType>::new(Session::builder().client(client).build().unwrap());

        // Mismatched operand shapes fail at bind time through type inference on the built program with a single
        // instruction, and never reach PJRT compilation or execution.
        let left = f32_vector(domain.parent(), &mesh, &[1.0, 2.0]);
        let right = f32_vector(domain.parent(), &mesh, &[1.0, 2.0, 3.0]);
        assert!(matches!(
            domain.bind(AddOperation::new(), Vec::new(), &[left, right]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`add` input types are not broadcast-compatible",
        ));
    }

    #[test]
    fn test_session_bind_rejects_collective_outside_a_mapping_context() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let domain = ProjectedContext::<Session, ArrayType>::new(Session::builder().client(client).build().unwrap());

        // A collective on a concrete array outside any `batch` or `shard_map` binder has no axis to resolve against,
        // mirroring JAX's "unbound axis name" error for a top-level `psum`. The `ParallelReduce` capability of values
        // is not even implemented for `Array`, because its session carries no environment of named axes, so this binds
        // the operation directly and asserts the failure to resolve the axis that surfaces at compile time.
        let input = f32_vector(domain.parent(), &mesh, &[1.0, 2.0]);
        let operation = ParallelReduceOperation::new(ReductionKind::Sum, "i".to_string());
        assert!(matches!(
            domain.bind(operation, Vec::new(), &[input]),
            Err(ProgramError::InvalidArgument { message })
                if message == "collective over axis `i` can only be lowered inside a `shard_map` manual region",
        ));
    }

    #[test]
    fn test_session_bind_shares_one_constant_for_repeated_dimension_inputs() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let two = DimensionValue::constant(2).unwrap();
        let inputs = [
            Value::Array(vector),
            Value::Dimension(Dimension::new(two.clone(), session.clone())),
            Value::Dimension(Dimension::new(two, session.clone())),
        ];
        let output = session.bind(DynamicReshapeOperation::new(), Vec::new(), &inputs).unwrap().remove(0);
        let output = program_array(&output);
        assert_eq!(output.shape().as_slice(), &[2, 2]);
        assert_eq!(read_first_shard::<f32>(client, output), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_session_bind_defines_separate_constants_for_distinct_dimension_identities() {
        // Dimension values compare by extent, but each distinct identity still needs its own constant, because a
        // constant defines its identity and the operation's type refers to both identities.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let bounds = DimensionBounds::new(1, Some(5)).unwrap();
        let rows = DimensionVariable::new("rows", bounds);
        let columns = DimensionVariable::new("columns", bounds);
        let operation = OneOperation::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![CoreDimension::Dynamic(rows.clone()), CoreDimension::Dynamic(columns.clone())]),
        ));
        let inputs = [
            Value::Dimension(Dimension::new(
                DimensionValue::new(DimensionType::from(rows), 2).unwrap(),
                session.clone(),
            )),
            Value::Dimension(Dimension::new(
                DimensionValue::new(DimensionType::from(columns), 2).unwrap(),
                session.clone(),
            )),
        ];
        let output = session.bind(operation, Vec::new(), &inputs).unwrap().remove(0);
        let output = program_array(&output);
        assert_eq!(output.shape().as_slice(), &[2, 2]);
        assert_eq!(read_first_shard::<f32>(client, output), vec![1.0; 4]);
    }

    #[test]
    fn test_session_bind_executes_condition_with_concrete_predicate() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 4);

        let doubled = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let squared = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = Operation::Condition(ConditionOperation::new());

        let input = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let true_outputs = session
            .bind(
                operation.clone(),
                [doubled.clone(), squared.clone()],
                &[Value::Array(boolean_scalar(&session, &mesh, true)), Value::Array(input.clone())],
            )
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, program_array(&true_outputs[0])), vec![2.0, 4.0, 6.0, 8.0]);

        let false_outputs = session
            .bind(
                operation,
                [doubled, squared],
                &[Value::Array(boolean_scalar(&session, &mesh, false)), Value::Array(input)],
            )
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, program_array(&false_outputs[0])), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(session.cached_program_count(), 1, "both predicate values must share one compiled executable");
    }

    #[test]
    fn test_session_bind_executes_condition_over_branches_refined_by_its_inputs() {
        // Eager compilation attaches the branches as provided. Their bounded `f32[rows]` input type is refined by the
        // static `f32[4]` input, so the eagerly executed output takes the refined static type rather than the bounded
        // branch output type.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let branch_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(rows)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let branch = |squares: bool| {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(branch_type.clone().into());
            let output = if squares {
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0]
            } else {
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0]
            };
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };

        let outputs = session
            .bind(
                Operation::Condition(ConditionOperation::new()),
                [branch(false), branch(true)],
                &[
                    Value::Array(boolean_scalar(&session, &mesh, false)),
                    Value::Array(f32_vector(&session, &mesh, &[1.0, 2.0, 3.0, 4.0])),
                ],
            )
            .unwrap();
        assert_eq!(program_array(&outputs[0]).shape(), StaticShape::new(vec![4]));
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_session_bind_condition_branches_consume_forwarded_dimension_authority() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let extent = DimensionValue::constant(3).unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        let branch = |negate: bool| {
            let mut builder = ProgramBuilder::new();
            let extent = builder.add_input(extent.r#type().into_owned().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let scalar = if negate {
                builder.add_instruction(NegOperation::new(), Vec::new(), vec![scalar], None).unwrap()[0]
            } else {
                scalar
            };
            let output = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        let operation = Operation::Condition(ConditionOperation::new());
        let scalar = f32_scalar(&session, &mesh, 2.0);

        let true_outputs = session
            .bind(
                operation.clone(),
                [branch(false), branch(true)],
                &[
                    Value::Array(boolean_scalar(&session, &mesh, true)),
                    Value::Dimension(Dimension::new(extent.clone(), session.clone())),
                    Value::Array(scalar.clone()),
                ],
            )
            .unwrap();
        assert_eq!(program_array(&true_outputs[0]).shape(), StaticShape::new(vec![3]));
        assert_eq!(read_first_shard::<f32>(client, program_array(&true_outputs[0])), vec![2.0; 3]);

        let false_outputs = session
            .bind(
                operation,
                [branch(false), branch(true)],
                &[
                    Value::Array(boolean_scalar(&session, &mesh, false)),
                    Value::Dimension(Dimension::new(extent, session.clone())),
                    Value::Array(scalar),
                ],
            )
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, program_array(&false_outputs[0])), vec![-2.0; 3]);
        assert_eq!(session.cached_program_count(), 1, "both branch selections must share one compiled executable");
    }

    #[test]
    fn test_session_bind_executes_bounded_while() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        // Loop `state = state + 1` while `state < 3`, starting from `0`.
        let condition = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(scalar_type.clone().into());
            let literal = CoreArray::from_elements(scalar_type.clone(), &[3.0f32]).unwrap();
            let limit = builder.add_instruction(ConstantOperation::new(literal), Vec::new(), vec![], None).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    Operation::Array(ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan))),
                    Vec::new(),
                    vec![state, limit],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![predicate], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(scalar_type.clone().into());
            let one =
                builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), vec![], None).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![next], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = Operation::While(WhileOperation::new());

        let outputs = session
            .bind(operation, vec![condition, body], &[Value::Array(f32_scalar(&session, &mesh, 0.0))])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![3.0]);
    }

    #[test]
    fn test_session_bind_executes_scan_with_per_step_outputs() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        // Scan body `(index, carry) -> (carry + 1, carry + 1)` with only a carry. The first output is the next carry
        // and the second is the stacked output of each step, so scanning 4 steps from `0` yields the cumulative sums
        // `[1, 2, 3, 4]`.
        let body = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let one =
                builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), vec![], None).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, one], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![next, next], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::new(1, 4);

        let outputs = session
            .bind(Operation::Scan(scan), [body], &[Value::Array(f32_scalar(&session, &mesh, 0.0))])
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![4.0]);
        assert_eq!(program_array(&outputs[1]).shape(), StaticShape::new(vec![4]));
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[1])), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_session_bind_executes_scan_with_explicit_slice_index() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = ArrayType::scalar(DataType::I64);
        let body = {
            let mut builder = ProgramBuilder::new();
            let index = builder.add_input(scalar_type.clone().into());
            builder.add_input(scalar_type.clone().into());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![index, index], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };

        // The carry records the last visited index, and stacked results retain slice order even when visits reverse.
        // Check the same contract for looped lowering, partially unrolled lowering with and without remaining
        // iterations after the loop, and fully unrolled lowering.
        for (reverse, unroll, last_index) in [
            (false, 1, 3),
            (true, 1, 0),
            (false, 2, 3),
            (true, 2, 0),
            (false, 3, 3),
            (true, 3, 0),
            (false, 4, 3),
            (true, 4, 0),
            (false, 8, 3),
            (true, 8, 0),
            (false, usize::MAX, 3),
            (true, usize::MAX, 0),
        ] {
            let initial =
                Array::from_host_buffer(&session, scalar_type.clone(), mesh.clone(), None, (-1i64).to_ne_bytes())
                    .unwrap();
            let scan = ScanOperation::new(1, 4).with_reverse(reverse).with_unroll(unroll).unwrap();
            let outputs = session.bind(Operation::Scan(scan), [body.clone()], &[Value::Array(initial)]).unwrap();
            assert_eq!(read_first_shard::<i64>(client, program_array(&outputs[0])), vec![last_index]);
            assert_eq!(read_first_shard::<i64>(client, program_array(&outputs[1])), vec![0, 1, 2, 3]);
        }
    }

    #[test]
    fn test_session_bind_executes_scan_with_dynamic_scalar_ssa_length() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let length_type = DimensionType::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let length = DimensionValue::new(length_type.clone(), 3).unwrap();

        let body = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let one =
                builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), vec![], None).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, one], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![next], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        // An unroll factor that does not divide the runtime length runs the remaining iterations in a second loop, and
        // factors beyond the length's upper bound are clamped to it.
        for unroll in [1, 2, 4, usize::MAX] {
            let scan = ScanOperation::new(1, length_type.simplified_dimension()).with_unroll(unroll).unwrap();
            let outputs = session
                .bind(
                    Operation::Scan(scan),
                    [body.clone()],
                    &[
                        Value::Array(f32_scalar(&session, &mesh, 0.0)),
                        Value::Dimension(Dimension::new(length.clone(), session.clone())),
                    ],
                )
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![3.0]);
        }
    }

    #[test]
    fn test_session_bind_executes_scan_with_stacked_inputs() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32);

        // Cumulative sum scan body `(index, carry, x) -> (carry + x, carry + x)` over the stacked input `[1, 2, 3, 4]`
        // starting from carry `0`. The final carry is the total `10` and the stacked outputs of the steps are the
        // running sums `[1, 3, 6, 10]`. The declared types of the body carry no metadata and are refined by the
        // concrete input types, which carry normalized shardings, so the scan binds eagerly despite the metadata
        // mismatch.
        let body = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let x = builder.add_input(scalar_type.clone().into());
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![sum, sum], vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::new(1, 4);

        let carry = f32_scalar(&session, &mesh, 0.0);
        let xs = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let outputs =
            session.bind(Operation::Scan(scan), vec![body], &[Value::Array(carry), Value::Array(xs)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![10.0]);
        assert_eq!(program_array(&outputs[1]).shape(), StaticShape::new(vec![4]));
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[1])), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_session_bind_executes_scan_with_sharded_stacked_inputs() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32);

        // The same cumulative sum scan as above, but with the stacked input sharded over the scanned leading axis of a
        // mesh with 2 devices. SPMD compilation of the operation handles the slicing across shards, and the inferred
        // scan output types leave shardings unspecified, so the outputs come back replicated over the mesh.
        let body = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let x = builder.add_input(scalar_type.clone().into());
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![sum, sum], vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::new(1, 4);

        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let xs_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let xs = Array::from_host_buffer(
            &session,
            xs_type,
            mesh.clone(),
            Some(sharding),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let carry = f32_scalar(&session, &mesh, 0.0);
        let outputs =
            session.bind(Operation::Scan(scan), vec![body], &[Value::Array(carry), Value::Array(xs)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![10.0]);
        assert_eq!(program_array(&outputs[1]).shape(), StaticShape::new(vec![4]));
        assert_eq!(program_array(&outputs[1]).sharding(), &Sharding::replicated(mesh.logical_mesh().clone(), 1),);
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[1])), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_session_bind_executes_jit_call_and_reuses_cached_executable() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 4);

        // A `jit_call` operation with the callee `x -> x * x`, bound eagerly on concrete arrays, dispatches straight
        // through the compiled eager path, mirroring JAX calling a JIT-compiled function from eager code.
        let callee = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = Operation::JitCall(JitCallOperation::new(0));
        let callee = Arc::new(callee);

        let input = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let first = session
            .bind(
                operation.clone(),
                AttachedRegions::Callees(std::slice::from_ref(&callee)),
                &[Value::Array(input.clone())],
            )
            .unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(read_first_shard::<f32>(client, program_array(&first[0])), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(session.cached_program_count(), 1);

        // A repeated eager `jit_call` at the same input signature is a dispatch cache hit.
        let second = session.bind(operation, AttachedRegions::Callees(&[callee]), &[Value::Array(input)]).unwrap();
        assert_eq!(read_first_shard::<f32>(client, program_array(&second[0])), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(session.cached_program_count(), 1, "a repeated eager `jit_call` must be a compilation cache hit");
    }

    #[test]
    fn test_session_bind_executes_shard_map_over_sharded_inputs() {
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let session = Session::builder().client(client).build().unwrap();

        // Manual `shard_map` body `local -> local + local` over `f32[4]` sharded across the mesh with 2 devices. Each
        // device doubles its own shard of 2 elements inside the manual region.
        let sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let local_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(sharding.local_sharding(&["x".to_string()]).unwrap())
            .unwrap();
        let body_program = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(local_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = ShardMapOperation::from_program(
            &body_program,
            vec![global_type.into()],
            ShardMapBoundary::new(logical_mesh, vec![sharding.clone()], vec![sharding.clone()], vec!["x".to_string()])
                .unwrap(),
        )
        .unwrap();
        let operation = Operation::ShardMap(Box::new(operation));
        let body_region = body_program;

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let input = Array::from_host_buffer(
            &session,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let outputs = session.bind(operation, vec![body_region], &[Value::Array(input)]).unwrap();

        assert_eq!(outputs.len(), 1);
        let output = program_array(&outputs[0]);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(output.shards().len(), 2);
        let shard_values = output
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.copy_to_host_bytes().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(shard_values, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    }

    #[test]
    fn test_session_bind_executes_shard_map_frontends() {
        // The frontend functions apply to concrete arrays of the XLA backend, which executes each map eagerly. A
        // sharded input meets a replicated one, the input shardings can be inferred from the input types, and a map
        // along `x` places the dimension that its axes select. Repeating a map at the same signature hits the dispatch
        // cache.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = test_manual_mesh("x", 2);
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let upload = |sharding: &Sharding, values: &[f32]| {
            let r#type = ArrayType::new_static(DataType::F32, [values.len()]).with_sharding(sharding.clone()).unwrap();
            Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(values)).unwrap()
        };
        let left = upload(&sharded, &[1.0, 2.0, 3.0, 4.0]);
        let right = upload(&replicated, &[10.0, 20.0]);
        let add = |left, right| {
            shard_map(
                |(left, right): (
                    ProjectedValue<ArrayType, DomainTracer<Session>>,
                    ProjectedValue<ArrayType, DomainTracer<Session>>,
                )| Ok(left + right),
                (left, right),
                logical_mesh.clone(),
                (sharded.clone(), replicated.clone()),
                sharded.clone(),
                ShardMapOptions::new(),
            )
            .unwrap()
        };
        let sum = add(left.clone(), right.clone());
        assert_eq!(sum.r#type().sharding(), Some(&sharded));
        assert_eq!(read_device_f32s(&sum), vec![vec![11.0, 22.0], vec![13.0, 24.0]]);
        let program_count = session.cached_program_count();
        assert_eq!(read_device_f32s(&add(left.clone(), right.clone())), vec![vec![11.0, 22.0], vec![13.0, 24.0]]);
        assert_eq!(session.cached_program_count(), program_count);

        let inferred = shard_map_with_inferred_input_shardings(
            |(left, right): (
                ProjectedValue<ArrayType, DomainTracer<Session>>,
                ProjectedValue<ArrayType, DomainTracer<Session>>,
            )| Ok(left * right),
            (left.clone(), right),
            logical_mesh.clone(),
            sharded.clone(),
            ShardMapOptions::new(),
        )
        .unwrap();
        assert_eq!(read_device_f32s(&inferred), vec![vec![10.0, 40.0], vec![30.0, 80.0]]);

        let doubled = shard_map_along(
            |x: ProjectedValue<ArrayType, DomainTracer<Session>>| Ok(x.clone() + x),
            left,
            logical_mesh,
            "x",
            Some(0),
            Some(0),
        )
        .unwrap();
        assert_eq!(doubled.r#type().sharding(), Some(&sharded));
        assert_eq!(read_device_f32s(&doubled), vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    }

    #[test]
    fn test_session_bind_executes_shard_map_over_the_default_mesh() {
        // An eager map over `x` of a `2x2` mesh runs a nested map without a mesh, which is defined over the mesh of the
        // enclosing map (its context's default mesh) and makes the remaining axis `y` manual, as JAX's
        // `shard_map(mesh=None)` does inside a `shard_map` body. Outside of every manual region, a map needs a mesh.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let input =
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        let doubled = shard_map(
            |x: ProjectedValue<ArrayType, DomainTracer<Session>>| {
                Ok(shard_map(
                    |x: ProjectedValue<ArrayType, DomainTracer<Session>>| Ok(x.clone() + x),
                    x,
                    None,
                    replicated.clone(),
                    replicated.clone(),
                    ShardMapOptions::new(),
                )?)
            },
            input.clone(),
            logical_mesh,
            sharded.clone(),
            sharded.clone(),
            ShardMapOptions::new().with_manual_axes(["x"]),
        )
        .unwrap();
        assert_eq!(read_device_f32s(&doubled), vec![vec![2.0, 4.0], vec![2.0, 4.0], vec![6.0, 8.0], vec![6.0, 8.0]]);
        let missing = shard_map(
            |x: ProjectedValue<ArrayType, DomainTracer<Session>>| Ok(x),
            input,
            None,
            sharded.clone(),
            sharded.clone(),
            ShardMapOptions::new(),
        );
        assert!(matches!(missing, Err(ShardMapError::MissingMesh)));

        // A session that was given a device mesh owns its logical view as the default mesh, so a top-level map in it
        // can omit the mesh, as under JAX's `jax.set_mesh`.
        let meshed_session = session.with_device_mesh(mesh.clone());
        let input =
            Array::from_host_buffer(&meshed_session, r#type, mesh, None, values_to_bytes(&[1f32, 2., 3., 4.])).unwrap();
        let doubled = shard_map(
            |x: ProjectedValue<ArrayType, DomainTracer<Session>>| Ok(x.clone() + x),
            input,
            None,
            sharded.clone(),
            sharded,
            ShardMapOptions::new().with_manual_axes(["x"]),
        )
        .unwrap();
        assert_eq!(read_device_f32s(&doubled), vec![vec![2.0, 4.0], vec![2.0, 4.0], vec![6.0, 8.0], vec![6.0, 8.0]]);
    }

    #[test]
    fn test_session_bind_shares_compilation_cache_across_independent_arrays() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();

        // Arrays constructed independently of each other in one session dispatch through their own recovered sessions,
        // which share that session's compilation cache so that the same eager operation compiles exactly once.
        let first = f32_vector(&session, &mesh, &[1.0, 2.0]);
        let second = f32_vector(&session, &mesh, &[3.0, 4.0]);
        let first_sum = first.add(&first).unwrap();
        let second_sum = second.add(&second).unwrap();
        let first_domain = first_sum.domain();
        let second_domain = second_sum.domain();
        assert!(std::ptr::eq(first_domain.parent().compilation_cache(), second_domain.parent().compilation_cache(),));
        assert_eq!(first_domain.parent().compilation_cache().statistics().compilations, 1);
        assert_eq!(read_first_shard::<f32>(client, &first_sum), vec![2.0, 4.0]);
        assert_eq!(read_first_shard::<f32>(client, &second_sum), vec![6.0, 8.0]);

        // Eager outputs belong to the client and session of their inputs, so they feed further eager operations that
        // dispatch through the same cache.
        assert!(std::ptr::eq(first_sum.client(), client));
        assert!(first_sum.session().is_same_session(&session));
        let product = first_sum.mul(&second).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &product), vec![6.0, 16.0]);
        assert!(std::ptr::eq(product.client(), client));
        assert!(product.session().is_same_session(&session));
        assert_eq!(session.cached_program_count(), 2);
    }

    #[test]
    fn test_session_bind_rejects_inputs_placed_on_a_foreign_device() {
        let domain_client = cpu_client(1);
        let foreign_client = cpu_client(2);
        let foreign_mesh = domain_mesh(foreign_client, "x", 2);
        let domain =
            ProjectedContext::<Session, ArrayType>::new(Session::builder().client(domain_client).build().unwrap());
        let foreign_session = Session::builder().client(foreign_client).build().unwrap();

        // An input that belongs to a session over a different client is rejected by client identity.
        let input = f32_vector(&foreign_session, &foreign_mesh, &[1.0, 2.0]);
        assert!(matches!(
            domain.bind(NegOperation::new(), Vec::new(), std::slice::from_ref(&input)),
            Err(ProgramError::InvalidArgument { message })
                if message == "received incompatible devices for eager XLA execution: input #0 is owned by a \
                    different PJRT client than this session's client",
        ));
        // Casts that produce dimensions execute on the host but still require array witnesses from the selected client.
        let dimension_type = Type::Dimension(DimensionType::from(CoreDimension::Static(2)));
        let dimension = Value::Dimension(Dimension::new(DimensionValue::constant(2).unwrap(), domain.parent().clone()));
        assert_eq!(
            domain.parent().bind(
                CastOperation::checked(
                    dimension_type.clone(),
                    vec![dimension_type, input.r#type().into_owned().into()],
                )
                .unwrap(),
                Vec::new(),
                &[dimension, Value::Array(input)],
            ),
            Err(ProgramError::InvalidArgument {
                message: "received incompatible devices for eager XLA execution: input #0 is owned by a different \
                          PJRT client than this session's client"
                    .to_owned(),
            }),
        );
    }

    #[test]
    fn test_session_bind_rejects_inputs_placed_on_different_devices() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let wide_mesh = two_device_mesh(client, MeshAxisType::Auto);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let first_mesh = DeviceMesh::new(logical_mesh.clone(), vec![wide_mesh.devices()[0]]).unwrap();
        let second_mesh = DeviceMesh::new(logical_mesh, vec![wide_mesh.devices()[1]]).unwrap();
        let inputs = [
            Value::Array(f32_vector(&session, &first_mesh, &[1.0])),
            Value::Array(f32_vector(&session, &second_mesh, &[2.0])),
        ];
        assert_eq!(
            session.bind(Operation::Array(AddOperation::new().into()), Vec::new(), &inputs),
            Err(ProgramError::InvalidArgument {
                message: format!(
                    "received incompatible devices for eager XLA execution: input #1 is placed on devices {:?} but \
                     input #0 is placed on devices {:?}",
                    vec![second_mesh.devices()[0].id()],
                    vec![first_mesh.devices()[0].id()],
                ),
            }),
        );
    }

    #[test]
    fn test_session_bind_executes_jvp() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let x = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0]);
        let tangent = f32_vector(&session, &mesh, &[1.0, 1.0, 1.0]);
        let (value, tangent): (Array, Array) = differentiate_at(x).jvp(tangent, |x| x.mul(&x)).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &value), vec![1.0, 4.0, 9.0]);
        assert_eq!(read_first_shard::<f32>(client, &tangent), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_session_bind_executes_value_and_gradient() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let x = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0]);
        let (value, gradient) = differentiate_at(x.clone())
            .value_and_gradient(|x| x.mul(&x).unwrap().reduce(&[0], ReductionKind::Sum).unwrap())
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &value), vec![14.0]);
        assert_eq!(read_first_shard::<f32>(client, &gradient), vec![2.0, 4.0, 6.0]);

        // An auxiliary output is evaluated at the primal and returned beside the value without being differentiated.
        let ((value, auxiliary), gradient): ((Array, Array), Array) = differentiate_at(x)
            .with_auxiliary_output()
            .value_and_gradient(|x| {
                let auxiliary = x.add(&x).unwrap();
                (x.mul(&x).unwrap().reduce(&[0], ReductionKind::Sum).unwrap(), auxiliary)
            })
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &value), vec![14.0]);
        assert_eq!(read_first_shard::<f32>(client, &auxiliary), vec![2.0, 4.0, 6.0]);
        assert_eq!(read_first_shard::<f32>(client, &gradient), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_session_bind_executes_vjp() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let x = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0]);
        let (value, pullback) = differentiate_at(x).vjp(|x| x.mul(&x)).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &value), vec![1.0, 4.0, 9.0]);

        // The pullback executes in the session that built it and maps the cotangent `w` to `2x · w`.
        let cotangent = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0]);
        let input_cotangent = pullback.apply_materialized(cotangent).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &input_cotangent), vec![2.0, 8.0, 18.0]);
    }

    #[test]
    fn test_session_bind_differentiation_preserves_capture_sharding() {
        // Sharding over two devices requires a CPU client with two devices.
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let primal = Array::from_host_buffer(
            &session,
            r#type.clone(),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let capture = Array::from_host_buffer(
            &session,
            r#type,
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[4f32, 5.0, 6.0, 7.0]).as_slice(),
        )
        .unwrap();

        // The gradient of `sum(x * c)` is the capture `c`, and it keeps the sharding and session of the capture.
        let (value, gradient) = differentiate_at(primal.clone())
            .with_captures(capture.clone())
            .value_and_gradient(|input, capture| input.mul(&capture).unwrap().reduce(&[0], ReductionKind::Sum).unwrap())
            .unwrap();
        assert_eq!(value.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![60.0]);
        assert_eq!(gradient.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(gradient.sharding(), &sharding);
        assert!(gradient.session().is_same_session(&session));

        // Linearization keeps the capture as a residual on its devices, with its sharding and session. Residual order
        // is not part of the contract, so the residual is found by its contents.
        let (_, pushforward) = differentiate_at(primal)
            .with_captures(capture)
            .linearize(|input, capture| input.mul(&capture))
            .unwrap();
        let capture_residual = pushforward
            .residuals()
            .iter()
            .find(|residual| residual.copy_to_host().unwrap().elements::<f32>().unwrap() == vec![4.0, 5.0, 6.0, 7.0])
            .unwrap();
        assert_eq!(capture_residual.sharding(), &sharding);
        assert!(capture_residual.session().is_same_session(&session));
    }

    #[test]
    fn test_session_bind_executes_coordinate_basis() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();

        // Low precision leaves build exact identity bases of their own element type.
        let values = [f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        let array = replicated_array(&session, &mesh, DataType::F16, &[3], values_to_bytes(&values).as_slice());
        let r#type = array.r#type().into_owned();
        let basis =
            <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(&array.domain(), &r#type, &r#type, 0, 3)
                .unwrap()
                .into_value();
        let (one, zero) = (f16::ONE, f16::ZERO);
        assert_eq!(basis.data_type(), DataType::F16);
        assert_eq!(basis.shape(), StaticShape::new(vec![3, 3]));
        assert_eq!(
            basis.copy_to_host().unwrap().elements::<f16>().unwrap(),
            vec![one, zero, zero, zero, one, zero, zero, zero, one],
        );

        let values = [bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0)];
        let array = replicated_array(&session, &mesh, DataType::BF16, &[3], values_to_bytes(&values).as_slice());
        let r#type = array.r#type().into_owned();
        let basis =
            <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(&array.domain(), &r#type, &r#type, 0, 3)
                .unwrap()
                .into_value();
        let (one, zero) = (bf16::ONE, bf16::ZERO);
        assert_eq!(basis.data_type(), DataType::BF16);
        assert_eq!(basis.shape(), StaticShape::new(vec![3, 3]));
        assert_eq!(
            basis.copy_to_host().unwrap().elements::<bf16>().unwrap(),
            vec![one, zero, zero, zero, one, zero, zero, zero, one],
        );

        // A rank-2 leaf at packed offset 1 of 8 basis items places its six coordinates in items 1 through 6, which
        // exercises the rectangular basis construction and its final reshape.
        let values = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = replicated_array(&session, &mesh, DataType::F32, &[2, 3], values_to_bytes(&values).as_slice());
        let r#type = matrix.r#type().into_owned();
        let basis =
            <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(&matrix.domain(), &r#type, &r#type, 1, 8)
                .unwrap()
                .into_value();
        let mut expected = vec![0f32; 48];
        for coordinate in 0..6 {
            expected[(coordinate + 1) * 6 + coordinate] = 1.0;
        }
        assert_eq!(basis.shape(), StaticShape::new(vec![8, 2, 3]));
        assert_eq!(basis.copy_to_host().unwrap().elements::<f32>().unwrap(), expected);

        // Complex leaves select typed complex ones and zeros.
        let values = [Complex::new(2f32, 1.0), Complex::new(-1.0, 3.0)];
        let array = replicated_array(&session, &mesh, DataType::C64, &[2], values_to_bytes(&values).as_slice());
        let r#type = array.r#type().into_owned();
        let basis =
            <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(&array.domain(), &r#type, &r#type, 0, 2)
                .unwrap()
                .into_value();
        let (one, zero) = (Complex::new(1f32, 0.0), Complex::new(0f32, 0.0));
        assert_eq!(basis.data_type(), DataType::C64);
        assert_eq!(values_from_bytes::<Complex<f32>>(&basis.copy_to_host_bytes().unwrap()), vec![one, zero, zero, one]);
    }

    #[test]
    fn test_session_bind_executes_jacobian_forward() {
        // Sharding over two devices requires a CPU client with two devices.
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // The batched basis tangents carry the sharding of the primal, with a replicated batch axis, so that their
        // replay matches the sharded input types of the tangent program. The Jacobian stays in the input session.
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let bytes = values_to_bytes(&[1f32, 2.0, 3.0, 4.0]);
        let x =
            Array::from_host_buffer(&session, r#type, mesh.clone(), Some(sharding.clone()), bytes.as_slice()).unwrap();
        let jacobian = differentiate_at(x).jacobian_forward(|x| x.mul(&x)).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[4]);
        assert!(block.value().session().is_same_session(&session));
        assert_eq!(block.value().shape(), StaticShape::new(vec![4, 4]));
        assert_eq!(
            block.value().copy_to_host().unwrap().elements::<f32>().unwrap(),
            vec![2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 8.0],
        );

        // Half-precision primals produce exact half-precision Jacobians.
        let r#type = ArrayType::new(DataType::F16, Shape::new(vec![CoreDimension::Static(4)]));
        let bytes = values_to_bytes(&[1f32, 2.0, 3.0, 4.0].map(f16::from_f32));
        let x = Array::from_host_buffer(&session, r#type, mesh, Some(sharding), bytes.as_slice()).unwrap();
        let jacobian = differentiate_at(x).jacobian_forward(|x| x.mul(&x)).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().data_type(), DataType::F16);
        assert_eq!(
            block.value().copy_to_host().unwrap().elements::<f16>().unwrap(),
            [2f32, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 8.0]
                .map(f16::from_f32)
                .to_vec(),
        );
    }

    #[test]
    fn test_session_bind_executes_jacobian_reverse() {
        // Sharding over two devices requires a CPU client with two devices.
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let bytes = values_to_bytes(&[1f32, 2.0, 3.0, 4.0]);
        let x = Array::from_host_buffer(&session, r#type, mesh, Some(sharding), bytes.as_slice()).unwrap();

        // The one-hot cotangent basis replays through the pullback on the sharded primal.
        let jacobian = differentiate_at(x).jacobian_reverse(|x| x.mul(&x)).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(
            block.value().copy_to_host().unwrap().elements::<f32>().unwrap(),
            vec![2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 8.0],
        );
    }

    #[test]
    fn test_session_bind_executes_hessian() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let x = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0]);
        let hessian = differentiate_at(x).hessian(|x| x.mul(&x)?.reduce(&[0], ReductionKind::Sum)).unwrap();
        let block = hessian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[] as &[usize]);
        assert_eq!(block.first_input_type().static_shape().unwrap().as_slice(), &[3]);
        assert_eq!(block.second_input_type().static_shape().unwrap().as_slice(), &[3]);
        assert_eq!(
            block.value().copy_to_host().unwrap().elements::<f32>().unwrap(),
            vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
        );
    }

    #[test]
    fn test_session_bind_executes_holomorphic_jacobians() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let z = Complex::new(1f32, 2.0);
        let input = replicated_array(&session, &mesh, DataType::C64, &[], values_to_bytes(&[z]).as_slice());

        // The holomorphic gradient seeds the pullback with a complex one and recovers the derivative `2z` of `z²`.
        let (value, gradient) = differentiate_at(input.clone())
            .holomorphic()
            .value_and_gradient(|input| input.mul(&input).unwrap())
            .unwrap();
        assert_eq!(values_from_bytes::<Complex<f32>>(&value.copy_to_host_bytes().unwrap()), vec![z * z]);
        assert_eq!(values_from_bytes::<Complex<f32>>(&gradient.copy_to_host_bytes().unwrap()), vec![z + z]);

        let forward =
            differentiate_at(input.clone()).holomorphic().jacobian_forward(|input| input.mul(&input)).unwrap();
        let reverse =
            differentiate_at(input.clone()).holomorphic().jacobian_reverse(|input| input.mul(&input)).unwrap();
        let hessian = differentiate_at(input).holomorphic().hessian(|input| input.mul(&input)).unwrap();
        assert_eq!(
            values_from_bytes::<Complex<f32>>(
                &forward.iter_blocks().next().unwrap().value().copy_to_host_bytes().unwrap()
            ),
            vec![z + z],
        );
        assert_eq!(
            values_from_bytes::<Complex<f32>>(
                &reverse.iter_blocks().next().unwrap().value().copy_to_host_bytes().unwrap()
            ),
            vec![z + z],
        );
        assert_eq!(
            values_from_bytes::<Complex<f32>>(
                &hessian.iter_blocks().next().unwrap().value().copy_to_host_bytes().unwrap()
            ),
            vec![Complex::new(2.0, 0.0)],
        );
    }

    #[test]
    fn test_session_bind_executes_batch() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input = f32_vector(&session, &mesh, &[1.0, 2.0, 3.0]);

        // The free `batch` recovers the session from its array input and squares each batch item through eager
        // dispatch.
        let output: Array = batch(|x| Ok(x.clone() * x), input, BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &output), vec![1.0, 4.0, 9.0]);
    }

    // Measures eager dispatch overhead. Compares repeated eager `add` calls against a warm JIT dispatcher over the same
    // input, and then times each step of the eager dispatch path (i.e., program construction, lowering, compilation
    // key construction, compilation cache lookup, and execution) in isolation.
    #[test]
    #[ignore = "benchmark of the eager dispatch overhead"]
    fn test_session_bind_dispatch_overhead() {
        const REPETITIONS: usize = 10_000;
        let measure = |label: &str, function: &mut dyn FnMut()| {
            for _ in 0..100 {
                function();
            }
            let start = Instant::now();
            for _ in 0..REPETITIONS {
                function();
            }
            let mean = start.elapsed().as_secs_f64() * 1e6 / REPETITIONS as f64;
            println!("eager dispatch overhead: {label}: {mean:.2} us/op");
        };

        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input = f32_vector(&session, &mesh, &[1.0; 8]);

        measure("eager add (end to end)", &mut || {
            input.add(&input).unwrap();
        });
        let function: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> =
            jit(|_, value: CompileTracer| Ok(value.add(&value)?), &session, mesh.clone());
        measure("JIT add (warm dispatch)", &mut || {
            function.execute((), input.clone()).unwrap();
        });
        let vector_type = Type::Array(input.r#type().into_owned());
        let branch = |operation: Operation| {
            let mut builder = ProgramBuilder::new();
            let value = builder.add_input(vector_type.clone());
            let output = builder.add_instruction(operation, Vec::new(), vec![value, value], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let condition_inputs = [Value::Array(boolean_scalar(&session, &mesh, true)), Value::Array(input.clone())];
        measure("eager condition with freshly built branches (end to end)", &mut || {
            let branches = [
                branch(Operation::Array(AddOperation::new().into())),
                branch(Operation::Array(MulOperation::new().into())),
            ];
            session.bind(Operation::Condition(ConditionOperation::new()), branches, &condition_inputs).unwrap();
        });

        let array_type = Type::Array(input.r#type().into_owned());
        let build_program = || -> FlatProgram {
            let mut builder = ProgramBuilder::new();
            let left = builder.add_input(array_type.clone());
            let right = builder.add_input(array_type.clone());
            let operation = Operation::Array(AddOperation::new().into());
            let outputs = builder.add_instruction(operation, Vec::new(), vec![left, right], None).unwrap().to_vec();
            builder.build(outputs, vec![Placeholder; 2], vec![Placeholder; 1]).unwrap()
        };
        let program = build_program();
        let options = Options::new(mesh.clone());
        let lowered = session.lower_program(&program, 0, &options).unwrap();
        let compiled = session.compilation_cache().compile(&session, &lowered).unwrap();
        measure("step: build one-instruction program", &mut || {
            build_program();
        });
        measure("step: lower to StableHLO", &mut || {
            session.lower_program(&program, 0, &options).unwrap();
        });
        measure("step: build compilation key", &mut || {
            session.compilation_key(&lowered).unwrap();
        });
        measure("step: cached compilation (hit)", &mut || {
            session.compilation_cache().compile(&session, &lowered).unwrap();
        });
        measure("step: execute", &mut || {
            session.execute_program(&compiled, vec![input.clone(), input.clone()]).unwrap();
        });
    }

    #[test]
    fn test_session_is_eager() {
        assert!(Session::builder().client(execution_client()).build().unwrap().is_eager());
    }

    #[test]
    fn test_session_provenance() {
        // A session records no instructions, so it has no provenance to report.
        assert_eq!(Session::builder().client(execution_client()).build().unwrap().provenance(), Provenance::unknown());
    }

    #[test]
    fn test_session_invoke_with_provenance_origin() {
        // The origin is ignored, so the function runs as it would without one.
        let session = Session::builder().client(execution_client()).build().unwrap();
        assert_eq!(session.invoke_with_provenance_origin(Provenance::unknown(), || 7), 7);
    }

    #[test]
    fn test_session_invoke_with_provenance_scope() {
        // The scope is ignored, so the provenance inside it is still unknown.
        let session = Session::builder().client(execution_client()).build().unwrap();
        assert_eq!(
            session.invoke_with_provenance_scope(ProvenanceScope::new("scope"), || session.provenance()),
            Provenance::unknown(),
        );
    }

    #[test]
    fn test_session_zero() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let r#type = Type::Array(ArrayType::new_static(DataType::F32, [2]));

        // A zero is an ordinary eager operation with or without a mesh, so one program is compiled for each type.
        for session in [
            Session::builder().client(client).build().unwrap().with_device_mesh(mesh),
            Session::builder().client(client).build().unwrap(),
        ] {
            let zero = session.zero(&r#type).unwrap();
            assert_eq!(read_first_shard::<f32>(client, program_array(&zero)), vec![0.0, 0.0]);
            session.zero(&r#type).unwrap();
            assert_eq!(session.cached_program_count(), 1);

            // A zero of the structural zero data type needs no storage.
            let zero = session.zero(&Type::Array(ArrayType::new_static(DataType::Zero, [3]))).unwrap();
            assert!(program_array(&zero).addressable_shards().next().unwrap().buffer().is_none());
        }
    }

    #[test]
    fn test_session_constant() {
        // Materializing a constant lifts it, with the same results and the same errors.
        let session = Session::builder().client(execution_client()).build().unwrap();
        let extent = DimensionValue::constant(3).unwrap();
        assert_eq!(
            CoreConstant::constant(&session, Constant::Dimension(extent.clone())),
            Ok(Value::Dimension(Dimension::new(extent, session.clone()))),
        );
        let capture = CaptureReference::new(0, Type::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            CoreConstant::constant(&session, Constant::Captured(capture)),
            Err(TypeError::invalid(
                "XLA captured constant `capture#0:f32[]` requires a captured program capture table"
            )
            .into()),
        );
    }

    #[test]
    fn test_session_builder_platform() {
        // Sessions that `ryft-xla` creates clients for can be `'static`, and they share the client of their platform
        // without sharing any session state.
        let session: Session = Session::builder().platform(Platform::Cpu).build().unwrap();
        let other: Session = Session::builder().platform(Platform::Cpu).build().unwrap();
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
            Session::builder().client(client).client_options(options.clone()).build(),
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
        let session = Session::builder().platform(Platform::Cpu).client(client).build().unwrap();
        assert!(std::ptr::eq(session.client(), client));
        let session = Session::builder().client(client).platform(Platform::Cpu).build().unwrap();
        assert!(!std::ptr::eq(session.client(), client));
    }

    #[test]
    fn test_session_builder_compilation_options() {
        let client = execution_client();
        let compilation_options = CompilationOptions { compile_portable_executable: true, ..Default::default() };
        let session =
            Session::builder().client(client).compilation_options(compilation_options.clone()).build().unwrap();
        assert_eq!(session.compilation_options(), &compilation_options);
    }

    #[test]
    fn test_session_builder_compilation_cache() {
        // The session compiles through the supplied context, whose in-memory tier here retains a single program.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder()
            .client(client)
            .compilation_cache(CompilationCache::with_capacity(1))
            .build()
            .unwrap();
        Neg::neg(&f32_vector(&session, &mesh, &[1.0, 2.0])).unwrap();
        Neg::neg(&f32_vector(&session, &mesh, &[1.0, 2.0, 3.0])).unwrap();
        assert_eq!(session.cached_program_count(), 1);
        assert_eq!(session.compilation_cache().statistics().compilations, 2);
    }

    #[test]
    fn test_session_builder_disk_cache() {
        let client = execution_client();
        let directory = tempfile::tempdir().unwrap();
        let session = Session::builder()
            .client(client)
            .disk_cache(DiskCache::open(directory.path()).unwrap())
            .build()
            .unwrap();
        let expected_directory = DiskCache::open(directory.path()).unwrap().directory().to_path_buf();
        assert_eq!(session.compilation_cache().disk_cache().unwrap().directory(), expected_directory);

        // An explicit disk cache replaces the one of a supplied compilation cache, irrespective of their order.
        let context_directory = tempfile::tempdir().unwrap();
        let session = Session::builder()
            .client(client)
            .disk_cache(DiskCache::open(directory.path()).unwrap())
            .compilation_cache(
                CompilationCache::new().with_disk_cache(DiskCache::open(context_directory.path()).unwrap()),
            )
            .build()
            .unwrap();
        assert_eq!(session.compilation_cache().disk_cache().unwrap().directory(), expected_directory);
    }

    #[test]
    fn test_session_builder_disk_cache_from_env() {
        // The session adds the disk cache that the environment configures, if any, when it is built. This test does not
        // modify the environment of the test process, and so it compares against whatever the environment configures.
        let client = execution_client();
        let session = Session::builder().client(client).disk_cache_from_env().build().unwrap();
        assert_eq!(
            session.compilation_cache().disk_cache().map(|disk_cache| disk_cache.directory().to_path_buf()),
            DiskCache::from_env().unwrap().map(|disk_cache| disk_cache.directory().to_path_buf()),
        );
    }

    #[test]
    fn test_session_builder_eager_dispatch_cache_capacity() {
        // The eager dispatch cache retains at most the configured number of applications, independently of the
        // compilation cache, which keeps its own default capacity.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).eager_dispatch_cache_capacity(1).build().unwrap();
        Neg::neg(&f32_vector(&session, &mesh, &[1.0, 2.0])).unwrap();
        Neg::neg(&f32_vector(&session, &mesh, &[1.0, 2.0, 3.0])).unwrap();
        assert_eq!((session.state.eager_dispatch_cache.len(), session.cached_program_count()), (1, 2));
    }

    #[test]
    fn test_session_builder_live_array_tracking() {
        // Sessions describe their live arrays only when they are asked to, which is not the default.
        let client = execution_client();
        assert!(Session::builder().client(client).build().unwrap().live_arrays().is_none());
        let session = Session::builder().client(client).live_array_tracking(false).build().unwrap();
        assert!(session.live_arrays().is_none());
        let session = Session::builder().client(client).live_array_tracking(true).build().unwrap();
        assert!(session.live_arrays().unwrap().is_empty());
    }

    #[test]
    fn test_session_builder_runtime_extensions() {
        // The supplied configuration replaces the built-in providers, so a session with an empty configuration cannot
        // satisfy a requirement that the default configuration satisfies.
        let client = execution_client();
        let requirement = RuntimeRequirement::builtin("ryft.ordered-io");
        let session = Session::builder().client(client).build().unwrap();
        session.runtimes().ensure(std::slice::from_ref(&requirement)).unwrap();
        let session = Session::builder().client(client).runtime_extensions(RuntimeExtensions::empty()).build().unwrap();
        assert!(matches!(
            session.runtimes().ensure(&[requirement]),
            Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == "ryft.ordered-io",
        ));
    }

    #[test]
    fn test_session_builder_build() {
        // Every build creates an independent session with a fresh effect scope and empty caches, even on one client.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let other = Session::builder().client(client).build().unwrap();
        assert!(std::ptr::eq(session.client(), other.client()));
        assert!(!session.is_same_session(&other));
        assert!(!shares_effect_scope(&session, &other));
        assert_eq!((session.cached_program_count(), session.state.eager_dispatch_cache.len()), (0, 0));
    }

    #[test]
    fn test_through_session_import() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array = f32_vector(&session, &mesh, &[1.0, 2.0]);
        let array_type = Type::Array(replicated_vector_type(&mesh, 2));
        let dimension = Value::Dimension(Dimension::new(DimensionValue::constant(3).unwrap(), session.clone()));
        let reference_error: ProgramError =
            TypeError::invalid("cannot import an XLA reference into a trace whose capture table holds constants")
                .into();

        // A root trace stages an imported array as a session constant and an imported dimension as an immediate
        // dimension constant, without adding any inputs to the traced program.
        let (output_types, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                let context = inputs[0].context();
                Ok(vec![
                    ImportingContext::<Session, ThroughSession>::import(context, Value::Array(array.clone()))?,
                    ImportingContext::<Session, ThroughSession>::import(context, dimension.clone())?,
                ])
            },
            vec![Type::Array(ArrayType::scalar(DataType::F32))],
        )
        .unwrap();
        assert_eq!(output_types, vec![array_type.clone(), dimension.r#type().into_owned()]);
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:f32[] .
                let %1:{array_type} = const session:{array_type}
                    %2:dimension<3> = const 3
                in (%1, %2)
            "}
            .trim_end(),
        );

        // References are rejected because their state cannot be shared with the trace.
        let root = TracingContext::<Constant, Operation>::new();
        let reference = Reference::new(ArrayReference::new(array.clone()), session.clone());
        assert_eq!(
            ImportingContext::<Session, ThroughSession>::import(&root, Value::Reference(reference.clone())).err(),
            Some(reference_error.clone()),
        );

        // A trace that is nested in such a trace imports values in the same way.
        let nested = NestedTracingContext::new(root);
        let imported =
            ImportingContext::<Session, ThroughSession>::import(&nested, Value::Array(array.clone())).unwrap();
        assert_eq!(imported.r#type().into_owned(), array_type);
        assert_eq!(
            ImportingContext::<Session, ThroughSession>::import(&nested, Value::Reference(reference)).err(),
            Some(reference_error),
        );
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_session_cuda_kernel_execution_and_persistence() {
        let client = Platform::Cuda
            .client(Some(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            })))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::builder().client(&client).build().unwrap();
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
        let input = Array::from_host_buffer(&session, scalar, None, None, 42i32.to_ne_bytes()).unwrap();
        let outputs = session.execute_program(&compiled, vec![input.clone()]).unwrap();
        let bytes = compiled.serialize().unwrap().unwrap();
        let restored_session = Session::builder().client(&client).build().unwrap();
        let restored = restored_session.deserialize_compiled_program(&bytes).unwrap().unwrap();
        assert!(
            restored
                .runtime_requirements()
                .iter()
                .any(|requirement| requirement.identifier() == "ryft.kernel.cuda")
        );

        let restored_outputs = restored_session.execute_program(&restored, vec![outputs[0].clone()]).unwrap();
        for (array, expected) in [(&input, 42), (&outputs[0], 43), (&restored_outputs[0], 44)] {
            assert_eq!(array.copy_to_host().unwrap().elements::<i32>().unwrap(), vec![expected]);
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
