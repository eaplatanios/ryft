//! XLA execution domains and the shared resources used to stage, compile, and execute array programs.
//!
//! A [`Domain`] implements [`ryft_core::Domain`] over [`ArrayIrType`], with [`Value`]s that hold device arrays,
//! checked dimensions, or mutable references. Values retain their domain, so eager operations and transforms recover
//! the execution context from their inputs. [`Constant`] and [`Operation`] provide the corresponding stored-constant
//! and operation families for traced programs.
//!
//! # Sessions and Domains
//!
//! A [`Session`] borrows one PJRT [`Client`] and owns its compilation cache, default effect scope, and kernel runtime
//! resources. Create a session with [`Session::new`] or one of its cache-configuring constructors, then obtain a domain
//! through [`Session::domain`]. Domains derived from the same session share those resources. Creating another session
//! gives an independent cache and effect scope, even when it borrows the same client.
//!
//! [`Options`] supplies the execution [`DeviceMesh`], sharding overrides, donation flags, and backend compilation
//! policies for a staged function. The domain's [`CompilationOptions`] supplies the underlying PJRT compiler settings.
//! [`Target`] describes the client's platform and devices without retaining live device handles.
//!
//! # Compilation and Execution
//!
//! Use [`compile`](crate::compile) for a compiled function or [`jit`](crate::jit) for a retained dispatcher that
//! specializes on its runtime inputs. The backend also implements the generic lifecycle in
//! [`ryft_core::compilation`]. When staging directly through [`ryft_core::StagedFunction::stage`], apply
//! [`with_input_shardings`] to the input types first if the options specify input sharding overrides.
//!
//! [`LoweredProgram`] contains StableHLO and the metadata needed for compilation. [`CompiledProgram`] contains the
//! loaded executable and its invocation contract. These flat artifacts are distinct from the typed function handles
//! in [`crate::compilation`], which also retain input and output container structures. The session's [`CompilationContext`]
//! coordinates artifact reuse, and [`CompilationKey`] records the exact computation and compilation configuration.
//! [`CompilationAnalysis`] and [`OptimizedProgram`] expose compiler diagnostics for a compiled program.
//!
//! # Effects and Completion
//!
//! Cloned domains share effect ordering. [`Domain::fork_effect_scope`] starts an independent sequence while retaining
//! the session and its cache. Cached programs contain no session or effect scope, so execution uses the invoking
//! domain's scope. Mutable references also retain their own identity-based coordination across calls.
//!
//! Array readiness observes completion of the work that produced that array. [`Domain::effects_barrier`] observes
//! previously submitted effects, including work whose outputs were discarded. Call it before dropping the client
//! when those effects must be observed. Sessions, domains, and their values borrow the client and cannot outlive it.

// TODO(eaplatanios): Review this module.

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
pub mod targets;

pub use analysis::{AnalysisValue, CompilationAnalysis, MemoryAnalysis, OptimizedProgram};
pub use compilation::{CompilationError, CompilationKey, DispatchKey, with_input_shardings};
pub use execution::ExecutionError;
pub use options::{FeedbackDirectedProfile, InputBoundBucketing, Options};
pub use programs::{CompiledProgram, LoweredProgram};
pub use sessions::Session;
pub use targets::{Target, TargetDevice};

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
pub(crate) mod tests {
    use std::cell::RefCell;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayIrType, ArrayType, DataType, Device, DeviceMesh, Dimension as CoreDimension, LogicalMesh, MeshAxis,
        MeshAxisType, Placeholder, ProjectedContext, ReferenceCompletion as CoreReferenceCompletion,
        ReferenceReadOperation, ReferenceType, Shape, Sharding,
    };
    use ryft_pjrt::protos::CompilationOptions;
    use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::domains::sessions::Session;
    use crate::experimental::ops::{Constant, FlatProgram, ProgramBuilder};
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{Array, FromPjrt, Value};

    use super::*;

    /// Deterministic stateful-runtime failure point used by transaction protocol regressions.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub(crate) enum StatefulFailureInjection {
        /// Fails immediately before PJRT submission, after argument preparation but before publication.
        BeforeSubmission,

        /// Fails after mutable inputs cross the execution handoff boundary.
        AfterHandoff,

        /// Fails after state publication but before hidden final-state replacements are reconstructed and committed.
        BeforeHiddenReplacementCommit,

        /// Fails after hidden final-state replacements are committed but before public outputs are reconstructed.
        BeforePublicReconstruction,
    }

    std::thread_local! {
        /// One-shot failure injection scoped to the invoking test thread.
        static STATEFUL_FAILURE_INJECTION: std::cell::Cell<Option<StatefulFailureInjection>> = const {
            std::cell::Cell::new(None)
        };

        /// One-shot backend-neutral dependency joined into the next stateful execution on the current test thread.
        static STATEFUL_COMPLETION_INJECTION: RefCell<Option<CoreReferenceCompletion>> = const { RefCell::new(None) };
    }

    /// Consumes the pending failure only when execution reaches its selected injection point.
    pub(super) fn take_stateful_failure_injection(expected: StatefulFailureInjection) -> bool {
        STATEFUL_FAILURE_INJECTION.with(|injection| {
            if injection.get() != Some(expected) {
                return false;
            }
            injection.set(None);
            true
        })
    }

    /// Takes the dependency to join into the next stateful execution on the current test thread.
    pub(super) fn take_stateful_completion_injection() -> Option<CoreReferenceCompletion> {
        STATEFUL_COMPLETION_INJECTION.with(|injection| injection.borrow_mut().take())
    }

    /// Injects one deterministic failure into the next stateful call on the current test thread.
    pub(crate) fn inject_stateful_failure(failure: StatefulFailureInjection) {
        STATEFUL_FAILURE_INJECTION.with(|injection| injection.set(Some(failure)));
    }

    /// Joins one deterministic dependency into the next stateful call on the current test thread.
    pub(crate) fn inject_stateful_completion(completion: CoreReferenceCompletion) {
        STATEFUL_COMPLETION_INJECTION.with(|injection| *injection.borrow_mut() = Some(completion));
    }

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

    pub(super) fn external_reference_read_program(referent_type: ArrayType) -> FlatProgram {
        let mut builder = ProgramBuilder::new();
        let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(referent_type)));
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    pub(super) fn array_domain<'c>(client: &'c Client<'c>) -> ProjectedContext<Domain<'c>, ArrayType> {
        ProjectedContext::new(Session::new(client).domain())
    }

    pub(super) fn program_array<'a, 'c>(value: &'a Value<'c>) -> &'a Array<'c> {
        let Value::Array(array) = value else {
            panic!("expected an array IR value");
        };
        array
    }

    pub(super) fn replicated_vector_type(mesh: &DeviceMesh, size: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(size)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap()
    }

    pub(super) fn replicated_scalar_type(mesh: &DeviceMesh, data_type: DataType) -> ArrayType {
        ArrayType::scalar(data_type)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap()
    }

    pub(super) fn f32_vector<'c>(domain: &Domain<'c>, mesh: &DeviceMesh, values: &[f32]) -> Array<'c> {
        let r#type = replicated_vector_type(mesh, values.len());
        Array::from_host_buffer(domain, r#type, mesh.clone(), values_to_bytes::<f32>(values).as_slice()).unwrap()
    }

    pub(super) fn f32_scalar<'c>(domain: &Domain<'c>, mesh: &DeviceMesh, value: f32) -> Array<'c> {
        let r#type = replicated_scalar_type(mesh, DataType::F32);
        Array::from_host_buffer(domain, r#type, mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
    }

    pub(super) fn f64_vector<'c>(domain: &Domain<'c>, mesh: &DeviceMesh, values: &[f64]) -> Array<'c> {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        Array::from_host_buffer(domain, r#type, mesh.clone(), values_to_bytes::<f64>(values).as_slice()).unwrap()
    }

    pub(super) fn boolean_vector<'c>(domain: &Domain<'c>, mesh: &DeviceMesh, values: &[bool]) -> Array<'c> {
        let r#type = ArrayType::new(DataType::Boolean, Shape::new(vec![CoreDimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let values = values.iter().copied().map(u8::from).collect::<Vec<_>>();
        Array::from_host_buffer(domain, r#type, mesh.clone(), values.as_slice()).unwrap()
    }

    pub(super) fn boolean_scalar<'c>(domain: &Domain<'c>, mesh: &DeviceMesh, value: bool) -> Array<'c> {
        let r#type = replicated_scalar_type(mesh, DataType::Boolean);
        Array::from_host_buffer(domain, r#type, mesh.clone(), &[u8::from(value)]).unwrap()
    }

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

    pub(super) fn read_u64s(client: &Client<'_>, array: &Array<'_>) -> Vec<u64> {
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

    pub(super) fn read_i64s(client: &Client<'_>, array: &Array<'_>) -> Vec<i64> {
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

    pub(super) fn read_booleans(client: &Client<'_>, array: &Array<'_>) -> Vec<bool> {
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
    fn test_domain_accessors_return_constructor_arguments() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let cloned = domain.clone();

        assert_eq!(domain.mesh().unwrap(), &mesh);
        assert_eq!(domain.compilation_options(), &CompilationOptions::default());
        assert!(Arc::ptr_eq(&domain.compilation_options, &cloned.compilation_options));
        assert!(size_of::<Domain<'_>>() < size_of::<CompilationOptions>());
    }

    #[test]
    fn test_domain_with_compilation_options() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions::default())).unwrap();
        let domain = Session::new(&client).domain();
        let mut compilation_options = CompilationOptions::default();
        compilation_options.compile_portable_executable = true;
        let sibling = domain.with_compilation_options(compilation_options.clone());
        assert!(Arc::ptr_eq(domain.session(), sibling.session()));
        assert!(Arc::ptr_eq(&domain.effect_scope, &sibling.effect_scope));
        assert_eq!(sibling.compilation_options(), &compilation_options);
        assert_eq!(domain.compilation_options(), &CompilationOptions::default());
    }
}
