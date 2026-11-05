use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::io::Write;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;

use serde::{Deserialize, Deserializer, Serialize};

use ryft_core::{CustomCallAttribute, CustomCallOperation, DeviceMesh};
use ryft_pjrt::extensions::ffi::{FfiTypeId, FfiUserData};
use ryft_pjrt::{Client, ExecutionContext, ExecutionFence};

use crate::Error;
use crate::domains::targets::Target;

// TODO(eaplatanios): Review this module.

/// Invalid runtime declarations, provider configuration, or extension admission.
#[derive(Clone, Debug, thiserror::Error)]
pub enum RuntimeError {
    /// A requirement identifier or encoded declaration is invalid.
    #[error("invalid runtime requirement: {reason}")]
    InvalidRequirement { reason: String },

    /// One program requests incompatible contracts from the same extension.
    #[error("conflicting requirements for runtime extension `{identifier}`")]
    ConflictingRequirements { identifier: String },

    /// No configured provider owns a required identifier.
    #[error("no provider is configured for runtime extension `{identifier}`")]
    MissingProvider { identifier: String },

    /// A provider was added twice without explicitly requesting replacement.
    #[error("a provider is already configured for runtime extension `{identifier}`")]
    DuplicateProvider { identifier: String },

    /// The provider does not implement the requested ABI.
    #[error("runtime extension `{identifier}` does not support ABI version {version}")]
    UnsupportedVersion { identifier: String, version: u32 },

    /// Provider-specific configuration does not match the selected runtime contract.
    #[error("invalid configuration for runtime extension `{identifier}`: {reason}")]
    InvalidConfiguration { identifier: String, reason: String },

    /// An initializer recursively requested its own unfinished instance.
    #[error("recursive initialization of runtime extension `{identifier}`")]
    RecursiveInitialization { identifier: String },

    /// Two extensions attempted to install data under one native type identifier.
    #[error("execution context already contains user data for FFI type {type_id}")]
    DuplicateUserData { type_id: i64 },
}

/// Reserved source-operation attribute holding canonical runtime requirements. XLA lowering consumes this attribute;
/// it is not forwarded to the native custom-call handler. Other custom-call attributes retain their usual meaning.
pub const RUNTIME_REQUIREMENTS_ATTRIBUTE: &str = "ryft.runtime.requirements";

/// Maximum size of one encoded runtime-requirement annotation, in bytes.
pub const MAX_RUNTIME_REQUIREMENTS_BYTES: usize = 1024 * 1024;

/// Immutable, portable description of one runtime extension contract.
///
/// Identifiers must be nonempty and contain no NUL. Providers own their ABI version and the interpretation of their
/// canonical configuration bytes. Configuration describes runtime setup; per-kernel parameters remain in the kernel
/// payload. Process-local registration IDs, pointers, and Rust type IDs must never appear here.
///
/// Compiled programs retain only these requirements. The invoking [`Session`](crate::Session) resolves them through
/// its configured [`RuntimeProvider`]s, so cached executables retain no producing session or initialized extensions.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct RuntimeRequirement {
    /// Namespaced identity of the runtime contract.
    identifier: String,

    /// Version of the extension's invocation ABI.
    abi_version: u32,

    /// Provider-owned deterministic runtime configuration.
    configuration: Vec<u8>,
}

impl RuntimeRequirement {
    /// Creates a requirement, validating its identifier and configuration size.
    pub fn new(
        identifier: impl Into<String>,
        abi_version: u32,
        configuration: impl Into<Vec<u8>>,
    ) -> Result<Self, Error> {
        let identifier = identifier.into();
        validate_identifier(&identifier)?;
        let configuration = configuration.into();
        if configuration.len() > MAX_RUNTIME_REQUIREMENTS_BYTES {
            return Err(Error::SizeLimitExceeded {
                message: "runtime configuration exceeds the metadata size limit".into(),
            });
        }
        Ok(Self { identifier, abi_version, configuration })
    }

    /// Returns the namespaced runtime contract identifier.
    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    /// Returns the requested invocation ABI version.
    pub fn abi_version(&self) -> u32 {
        self.abi_version
    }

    /// Returns the provider's canonical runtime configuration bytes.
    pub fn configuration(&self) -> &[u8] {
        &self.configuration
    }
}

impl<'de> Deserialize<'de> for RuntimeRequirement {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// Serialized fields validated through the public constructor.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            /// Namespaced identity to validate before publication.
            identifier: String,

            /// Version of the extension's invocation ABI.
            abi_version: u32,

            /// Provider-owned configuration, bounded by the public constructor.
            #[serde(deserialize_with = "deserialize_runtime_configuration")]
            configuration: Vec<u8>,
        }
        let fields = Fields::deserialize(deserializer)?;
        Self::new(fields.identifier, fields.abi_version, fields.configuration).map_err(serde::de::Error::custom)
    }
}

/// Decodes configuration without constructing an oversized byte vector from persisted metadata.
fn deserialize_runtime_configuration<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
    /// Sequence visitor enforcing the same limit as the public requirement constructor.
    struct Configuration;

    impl<'de> serde::de::Visitor<'de> for Configuration {
        type Value = Vec<u8>;

        fn expecting(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "a runtime configuration of at most {MAX_RUNTIME_REQUIREMENTS_BYTES} bytes")
        }

        fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
            let mut configuration = Vec::new();
            while let Some(byte) = sequence.next_element::<u8>()? {
                if configuration.len() == MAX_RUNTIME_REQUIREMENTS_BYTES {
                    return Err(serde::de::Error::custom("runtime configuration exceeds the metadata size limit"));
                }
                configuration.push(byte);
            }
            Ok(configuration)
        }
    }

    deserializer.deserialize_seq(Configuration)
}

/// Sorts and deduplicates requirements, rejecting incompatible declarations for one identifier.
pub fn normalize_runtime_requirements(
    requirements: impl IntoIterator<Item = RuntimeRequirement>,
) -> Result<Vec<RuntimeRequirement>, Error> {
    let mut canonical = BTreeMap::new();
    for requirement in requirements {
        validate_identifier(requirement.identifier())?;
        if let Some(previous) = canonical.get(requirement.identifier()) {
            if previous != &requirement {
                return Err(RuntimeError::ConflictingRequirements { identifier: requirement.identifier.clone() }.into());
            }
        } else {
            canonical.insert(requirement.identifier.clone(), requirement);
        }
    }
    Ok(canonical.into_values().collect())
}

/// Attaches canonical runtime requirements to a custom call for consumption by XLA lowering.
///
/// This function accepts one annotation per operation. Identical requirements deduplicate, while conflicting
/// requirements or a preexisting annotation are errors. The encoded annotation is bounded by
/// [`MAX_RUNTIME_REQUIREMENTS_BYTES`]. Native FFI handlers receive none of this bookkeeping.
pub fn with_runtime_requirements(
    operation: CustomCallOperation,
    requirements: impl IntoIterator<Item = RuntimeRequirement>,
) -> Result<CustomCallOperation, Error> {
    /// Writer that stops serialization before an oversized annotation can allocate its complete encoded form.
    struct Annotation {
        /// Encoded JSON, bounded by the public annotation size limit.
        bytes: Vec<u8>,

        /// Whether serialization encountered the size limit rather than another encoding failure.
        exceeded: bool,
    }

    impl Write for Annotation {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > MAX_RUNTIME_REQUIREMENTS_BYTES - self.bytes.len() {
                self.exceeded = true;
                return Err(std::io::Error::other("runtime requirements exceed the metadata size limit"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    if operation.attributes().iter().any(|(name, _)| name == RUNTIME_REQUIREMENTS_ATTRIBUTE) {
        return Err(
            RuntimeError::InvalidRequirement { reason: "duplicate runtime-requirement annotation".into() }.into()
        );
    }
    let requirements = normalize_runtime_requirements(requirements)?;
    if requirements.is_empty() {
        return Ok(operation);
    }
    let mut encoded = Annotation { bytes: Vec::new(), exceeded: false };
    if let Err(error) = serde_json::to_writer(&mut encoded, &(1u32, requirements)) {
        if encoded.exceeded {
            return Err(Error::SizeLimitExceeded {
                message: "runtime requirements exceed the metadata size limit".into(),
            });
        }
        return Err(RuntimeError::InvalidRequirement { reason: error.to_string() }.into());
    }
    Ok(operation.with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, String::from_utf8(encoded.bytes).unwrap()))
}

/// Reads and validates the reserved requirement annotation of a custom call.
pub fn runtime_requirements(operation: &CustomCallOperation) -> Result<Vec<RuntimeRequirement>, Error> {
    let mut annotations = operation.attributes().iter().filter(|(name, _)| name == RUNTIME_REQUIREMENTS_ATTRIBUTE);
    let Some((_, attribute)) = annotations.next() else {
        return Ok(Vec::new());
    };
    if annotations.next().is_some() {
        return Err(
            RuntimeError::InvalidRequirement { reason: "duplicate runtime-requirement annotation".into() }.into()
        );
    }
    let CustomCallAttribute::String(encoded) = attribute else {
        return Err(RuntimeError::InvalidRequirement {
            reason: "runtime requirements must be encoded as a string".into(),
        }
        .into());
    };
    if encoded.len() > MAX_RUNTIME_REQUIREMENTS_BYTES {
        return Err(Error::SizeLimitExceeded { message: "runtime requirements exceed the metadata size limit".into() });
    }
    let (version, requirements): (u32, Vec<RuntimeRequirement>) = serde_json::from_str(encoded)
        .map_err(|error| RuntimeError::InvalidRequirement { reason: error.to_string() })?;
    if version != 1 {
        return Err(RuntimeError::InvalidRequirement {
            reason: format!("unsupported runtime-requirement encoding version {version}"),
        }
        .into());
    }
    normalize_runtime_requirements(requirements)
}

/// Returns a deterministic native FFI type ID for the provider-owned user-data type called `name`.
///
/// Registering a type with an explicit ID through [`Client::register_ffi_type`] is idempotent for the same name and
/// ID, so providers can register on every initialization, and native handlers can look up their user data with the
/// same constant. XLA assigns its own IDs sequentially starting from one, while every ID returned here lies in
/// `[2^62, 2^63)`, so the two never collide. XLA does not reject one ID registered under two different names, so
/// `name` should be namespaced (e.g., `"my_crate.runtime"`) to keep accidental collisions negligible.
pub const fn named_ffi_type_id(name: &str) -> FfiTypeId {
    // 64-bit FNV-1a, folded into the range that XLA's sequential type ID assignment never reaches.
    let bytes = name.as_bytes();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    FfiTypeId::new(((hash >> 2) | (1 << 62)) as i64)
}

/// Factory for one named runtime contract, configured before a session is constructed.
///
/// Initialization receives the invoking client's actual target. It must reject unsupported ABIs, configurations,
/// and native capabilities before returning an instance. Instances must not retain an owning session handle.
/// Reentry into an unfinished initializer is rejected. A nested initialization also cannot wait for another thread's
/// initializer in the same session, which prevents dependency cycles between concurrent providers.
///
/// Providers register native types with explicit IDs from [`named_ffi_type_id`] and handlers through PJRT during
/// initialization. Type registration is idempotent, and XLA accepts re-registration of an identical handler. Each
/// session's providers can therefore register again without process-global bookkeeping, while native handlers look
/// up their user data with the same constant ID.
///
/// Providers and extensions may borrow the client without requiring a `'static` runtime. For example, a downstream
/// crate can add a registration-only contract alongside the built-in providers and annotate its custom calls with
/// the same requirement.
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
///
/// use ryft_core::{CompilationContext, CustomCallOperation};
/// use ryft_pjrt::Client;
/// use ryft_xla::domains::{
///     RuntimeExtension, RuntimeExtensions, RuntimeProvider, RuntimeRequirement, with_runtime_requirements,
/// };
/// use ryft_xla::{Error, RuntimeError, Session, Target};
///
/// struct Provider;
/// struct Runtime<'c> { client: &'c Client<'c> }
/// impl<'c> RuntimeExtension<'c> for Runtime<'c> {}
///
/// impl<'c> RuntimeProvider<'c> for Provider {
///     fn identifier(&self) -> &str { "example.runtime" }
///
///     fn initialize(
///         &self,
///         client: &'c Client<'c>,
///         target: &Target,
///         requirement: &RuntimeRequirement,
///     ) -> Result<Arc<dyn RuntimeExtension<'c> + 'c>, Error> {
///         if requirement.abi_version() != 1 {
///             return Err(RuntimeError::UnsupportedVersion {
///                 identifier: self.identifier().into(), version: requirement.abi_version(),
///             }.into());
///         }
///         // Validate configuration and target capabilities, then register this contract's native handlers here.
///         Ok(Arc::new(Runtime { client }))
///     }
/// }
///
/// fn configure<'c>(client: &'c Client<'c>) -> Result<Arc<Session<'c>>, Error> {
///     let providers = RuntimeExtensions::new().with_provider(Arc::new(Provider))?;
///     let requirement = RuntimeRequirement::new("example.runtime", 1, [])?;
///     let operation = with_runtime_requirements(CustomCallOperation::new("example.call", vec![]), [requirement])?;
///     // Apply `operation` in the staged program. XLA consumes its reserved runtime annotation during lowering.
///     Ok(Session::with_runtime_extensions(client, CompilationContext::new(), providers))
/// }
/// ```
pub trait RuntimeProvider<'c>: Send + Sync {
    /// Returns the contract identifier this provider implements.
    fn identifier(&self) -> &str;

    /// Validates and initializes one client-associated contract instance. A failed initialization may be retried;
    /// plugin-global registration steps must therefore be idempotent and preserve compatible partial progress.
    fn initialize(
        &self,
        client: &'c Client<'c>,
        target: &Target,
        requirement: &RuntimeRequirement,
    ) -> Result<Arc<dyn RuntimeExtension<'c> + 'c>, Error>;
}

/// Live extension instance shared by calls through one session.
///
/// The session keeps its stable allocation alive until all executions borrowing extension data have completed.
/// Implementations must synchronize native handler access across devices and threads. Registration-only extensions
/// may use the default preparation behavior, which does not create an execution context.
/// A runtime borrowing its client cannot escape that client.
///
/// ```compile_fail
/// use std::sync::Arc;
/// use ryft_pjrt::Client;
/// use ryft_xla::domains::RuntimeExtension;
///
/// struct BorrowedRuntime<'c>(&'c Client<'c>);
/// impl<'c> RuntimeExtension<'c> for BorrowedRuntime<'c> {}
///
/// fn escaped<'c>(client: Client<'c>) -> Arc<dyn RuntimeExtension<'c> + 'c> {
///     Arc::new(BorrowedRuntime(&client))
/// }
/// ```
pub trait RuntimeExtension<'c>: Send + Sync {
    /// Checks invocation-specific device and participant restrictions before native submission.
    fn validate_execution(&self, _client: &Client<'_>, _mesh: &DeviceMesh) -> Result<(), Error> {
        Ok(())
    }

    /// Attaches invocation data and retains any additional owners needed through completion.
    fn prepare(&self, _resources: &mut ExecutionResources<'c>) -> Result<(), Error> {
        Ok(())
    }
}

/// Immutable provider configuration, independent of initialized session runtime instances.
///
/// Configure [`RuntimeProvider`]s before constructing a [`Session`](crate::Session). Each session initializes its own
/// instances lazily when a compiled program requires them.
#[derive(Clone)]
pub struct RuntimeExtensions<'c> {
    /// Providers keyed by their public contract identifiers.
    providers: BTreeMap<String, Arc<dyn RuntimeProvider<'c> + 'c>>,
}

impl<'c> RuntimeExtensions<'c> {
    /// Creates the built-in configuration. Providers are lazy; this does not load accelerator drivers.
    pub fn new() -> Self {
        let mut configuration = Self::empty();
        let mut providers = crate::kernels::runtime_providers();
        providers.push(crate::operations::debugging::runtime_provider());
        providers.push(crate::operations::debugging::ordered_io_runtime_provider());
        providers.push(crate::operations::assertions::runtime_provider());
        for provider in providers {
            configuration = configuration.with_provider(provider).unwrap();
        }
        configuration
    }

    /// Creates an empty provider configuration for explicit integrations.
    pub fn empty() -> Self {
        Self { providers: BTreeMap::new() }
    }

    /// Adds a provider, rejecting a duplicate identifier.
    pub fn with_provider(mut self, provider: Arc<dyn RuntimeProvider<'c> + 'c>) -> Result<Self, Error> {
        validate_identifier(provider.identifier())?;
        let identifier = provider.identifier().to_owned();
        if self.providers.contains_key(&identifier) {
            return Err(RuntimeError::DuplicateProvider { identifier }.into());
        }
        self.providers.insert(identifier, provider);
        Ok(self)
    }

    /// Explicitly replaces a configured provider before session construction. This does not replace an incompatible
    /// plugin-global native registration and does not mutate existing sessions using a cloned configuration.
    pub fn with_replaced_provider(mut self, provider: Arc<dyn RuntimeProvider<'c> + 'c>) -> Result<Self, Error> {
        validate_identifier(provider.identifier())?;
        let identifier = provider.identifier().to_owned();
        if !self.providers.contains_key(&identifier) {
            return Err(RuntimeError::MissingProvider { identifier }.into());
        }
        self.providers.insert(identifier, provider);
        Ok(self)
    }

    /// Returns configured contract identifiers in deterministic order.
    pub fn identifiers(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }
}

impl Default for RuntimeExtensions<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for RuntimeExtensions<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeExtensions")
            .field("identifiers", &self.identifiers().collect::<Vec<_>>())
            .finish()
    }
}

/// Erased ownership with the session's client lifetime, without requiring `Any` or `'static`.
trait RetainedResource: Send + Sync {}
impl<T: Send + Sync> RetainedResource for T {}

/// Invocation data and owners prepared before one native submission.
///
/// Retain stable allocations such as `Arc<T>` before attaching pointers into them. On preparation failure, the context
/// is destroyed before retained owners. After submission, the session retains owners through the joined fence,
/// including when outputs are discarded. Extensions contribute to one lazily created execution context. Ordinary
/// execution and registration-only extensions create no context and introduce no additional destruction barrier.
pub struct ExecutionResources<'c> {
    /// Client receiving this invocation's context.
    client: &'c Client<'c>,

    /// Native context, allocated only when invocation data is first attached.
    context: Option<ExecutionContext>,

    /// Type identifiers already installed in this invocation.
    user_data: HashSet<FfiTypeId>,

    /// Additional owners with the same lifetime as the invoking session's client.
    retained: Vec<Box<dyn RetainedResource + 'c>>,
}

impl<'c> ExecutionResources<'c> {
    /// Returns the client receiving the invocation.
    pub fn client(&self) -> &'c Client<'c> {
        self.client
    }

    /// Retains an owner through whole-execution completion. Use a stable allocation such as `Arc` or `Box` when native
    /// user data points into the retained value; moving a stack value into this function does not preserve its address.
    pub fn retain<T: 'c + Send + Sync>(&mut self, owner: T) {
        self.retained.push(Box::new(owner));
    }

    /// Adds one borrowed native user-data binding to this invocation, creating the context lazily.
    ///
    /// # Safety
    ///
    /// The pointer must have the type registered under its native type ID and remain stable until completion. Its owner
    /// must be this session's runtime instance or be retained through [`Self::retain`]. Native handlers may access it
    /// concurrently on several devices; the pointee must support those accesses. This boundary permits third-party FFI
    /// integrations while keeping native pointer insertion explicit.
    pub unsafe fn add_ffi_user_data(&mut self, data: FfiUserData) -> Result<(), Error> {
        if !self.user_data.insert(data.type_id) {
            return Err(RuntimeError::DuplicateUserData { type_id: data.type_id.into() }.into());
        }
        if self.context.is_none() {
            self.context = Some(self.client.execution_context()?);
        }
        unsafe { self.context.as_ref().unwrap().add_ffi_user_data(data) }?;
        Ok(())
    }

    /// Separates native context ownership from the session-owned resource lease.
    pub(super) fn into_parts(mut self) -> (Option<ExecutionContext>, RetainedExecutionResources<'c>) {
        let context = self.context.take();
        let needed = context.is_some() || !self.retained.is_empty();
        let retained = std::mem::take(&mut self.retained);
        (context, RetainedExecutionResources { needed, _owners: retained })
    }
}

impl Drop for ExecutionResources<'_> {
    fn drop(&mut self) {
        drop(self.context.take());
    }
}

/// Invocation ownership retained by the session only when extensions borrow runtime data.
pub(super) struct RetainedExecutionResources<'c> {
    /// Whether the invocation borrowed data, even if its owner is the session's runtime instance itself.
    needed: bool,

    /// Additional retained owners, destroyed after whole-execution completion.
    _owners: Vec<Box<dyn RetainedResource + 'c>>,
}

/// Initialization state for one session and complete runtime descriptor.
enum Initialization<'c> {
    /// No successful instance has been published.
    Empty,

    /// One thread initializes outside the registry lock.
    Running(ThreadId),

    /// Immutable stable instance shared by later calls.
    Ready(Arc<dyn RuntimeExtension<'c> + 'c>),
}

/// Independent synchronization for one runtime, so unrelated initialization does not serialize.
struct RuntimeInitialization<'c> {
    /// Mutable initialization state.
    state: Mutex<Initialization<'c>>,

    /// Notification after successful initialization or retryable failure.
    ready: Condvar,
}

/// Resets an unfinished initializer on error or unwind, without poisoning the instance slot.
struct InitializationGuard<'o, 'c> {
    /// Slot whose running state must be cleared unless publication succeeds.
    slot: &'o RuntimeInitialization<'c>,

    /// Whether initialization still needs rollback.
    active: bool,
}

impl Drop for InitializationGuard<'_, '_> {
    fn drop(&mut self) {
        if self.active {
            *self.slot.state.lock().expect("runtime initialization mutex poisoned") = Initialization::Empty;
            self.slot.ready.notify_all();
        }
    }
}

/// One extension-using execution whose resources outlive discarded output handles.
struct PendingExecution<'c> {
    /// Completion of every participating device and native context cleanup.
    fence: ExecutionFence,

    /// Borrowed invocation resources retained until the fence is terminal.
    _resources: RetainedExecutionResources<'c>,
}

/// Per-session runtime instances and pending resource ownership.
pub(super) struct RuntimeRegistry<'c> {
    /// Externally owned client associated with all initialized instances.
    client: &'c Client<'c>,

    /// Frozen provider configuration, shared by value rather than sharing initialized instances.
    configuration: RuntimeExtensions<'c>,

    /// Per-requirement initialization slots.
    instances: Mutex<HashMap<RuntimeRequirement, Arc<RuntimeInitialization<'c>>>>,

    /// Only launches that borrow extension resources, not ordinary PJRT work.
    pending: Mutex<Vec<PendingExecution<'c>>>,
}

impl<'c> RuntimeRegistry<'c> {
    /// Attaches immutable provider configuration to a new session.
    pub(super) fn new(client: &'c Client<'c>, configuration: RuntimeExtensions<'c>) -> Self {
        Self { client, configuration, instances: Mutex::new(HashMap::new()), pending: Mutex::new(Vec::new()) }
    }

    /// Initializes all required providers before native compilation or loading.
    pub(super) fn ensure(&self, requirements: &[RuntimeRequirement], target: &Target) -> Result<(), Error> {
        for requirement in normalize_runtime_requirements(requirements.iter().cloned())? {
            self.instance(&requirement, target)?;
        }
        Ok(())
    }

    /// Validates the selected invocation and composes all requested extension state into at most one context.
    pub(super) fn prepare(
        &self,
        requirements: &[RuntimeRequirement],
        target: &Target,
        mesh: &DeviceMesh,
    ) -> Result<ExecutionResources<'c>, Error> {
        let mut resources =
            ExecutionResources { client: self.client, context: None, user_data: HashSet::new(), retained: Vec::new() };
        for requirement in normalize_runtime_requirements(requirements.iter().cloned())? {
            let runtime = self.instance(&requirement, target)?;
            runtime.validate_execution(self.client, mesh)?;
            runtime.prepare(&mut resources)?;
        }
        Ok(resources)
    }

    /// Retains successful submission resources before any fallible result reconstruction.
    pub(super) fn retain(&self, fence: &ExecutionFence, resources: RetainedExecutionResources<'c>) {
        if !resources.needed {
            return;
        }
        let mut completed = Vec::new();
        {
            let mut pending = self.pending.lock().expect("runtime completion mutex poisoned");
            let mut index = 0;
            while index < pending.len() {
                if matches!(pending[index].fence.is_ready(), Ok(false)) {
                    index += 1;
                } else {
                    completed.push(pending.swap_remove(index));
                }
            }
            if matches!(fence.is_ready(), Ok(false)) {
                pending.push(PendingExecution { fence: fence.clone(), _resources: resources });
            } else {
                completed.push(PendingExecution { fence: fence.clone(), _resources: resources });
            }
        }
        drop(completed);
    }

    /// Initializes outside registry locks and publishes one stable successful instance.
    fn instance(
        &self,
        requirement: &RuntimeRequirement,
        target: &Target,
    ) -> Result<Arc<dyn RuntimeExtension<'c> + 'c>, Error> {
        let provider = self
            .configuration
            .providers
            .get(requirement.identifier())
            .ok_or_else(|| RuntimeError::MissingProvider { identifier: requirement.identifier.clone() })?;
        let slot = {
            let mut instances = self.instances.lock().expect("runtime registry mutex poisoned");
            Arc::clone(instances.entry(requirement.clone()).or_insert_with(|| {
                Arc::new(RuntimeInitialization { state: Mutex::new(Initialization::Empty), ready: Condvar::new() })
            }))
        };
        let current = std::thread::current().id();
        let mut state = slot.state.lock().expect("runtime initialization mutex poisoned");
        loop {
            match &*state {
                Initialization::Ready(runtime) => return Ok(Arc::clone(runtime)),
                Initialization::Running(owner) if *owner == current => {
                    return Err(
                        RuntimeError::RecursiveInitialization { identifier: requirement.identifier.clone() }.into()
                    );
                }
                Initialization::Running(_) => {
                    // Do not wait while this thread initializes another descriptor in the same registry. Two
                    // providers could otherwise each wait for the other's thread indefinitely. Release this slot
                    // before inspecting other slots so concurrent dependency checks never invert their lock order.
                    drop(state);
                    let slots = self
                        .instances
                        .lock()
                        .expect("runtime registry mutex poisoned")
                        .values()
                        .cloned()
                        .collect::<Vec<_>>();
                    let initializing = slots.iter().any(|slot| {
                        matches!(
                            *slot.state.lock().expect("runtime initialization mutex poisoned"),
                            Initialization::Running(owner) if owner == current
                        )
                    });
                    state = slot.state.lock().expect("runtime initialization mutex poisoned");
                    if matches!(*state, Initialization::Running(_)) {
                        if initializing {
                            return Err(RuntimeError::RecursiveInitialization {
                                identifier: requirement.identifier.clone(),
                            }
                            .into());
                        }
                        state = slot.ready.wait(state).expect("runtime initialization mutex poisoned while waiting");
                    }
                }
                Initialization::Empty => {
                    *state = Initialization::Running(current);
                    break;
                }
            }
        }
        drop(state);
        let mut rollback = InitializationGuard { slot: &slot, active: true };
        let runtime = provider.initialize(self.client, target, requirement)?;
        *slot.state.lock().expect("runtime initialization mutex poisoned") =
            Initialization::Ready(Arc::clone(&runtime));
        rollback.active = false;
        slot.ready.notify_all();
        Ok(runtime)
    }
}

impl Drop for RuntimeRegistry<'_> {
    fn drop(&mut self) {
        let pending = std::mem::take(self.pending.get_mut().expect("runtime completion mutex poisoned"));
        for execution in pending {
            // Failure is terminal too; diagnostics remain in the caller's fence and effect/reference state.
            match execution.fence.block_until_ready() {
                Ok(()) | Err(_) => {}
            }
            drop(execution);
        }
    }
}

/// Validates an extension identifier at every public configuration boundary.
fn validate_identifier(identifier: &str) -> Result<(), Error> {
    if identifier.len() > MAX_RUNTIME_REQUIREMENTS_BYTES {
        return Err(Error::SizeLimitExceeded { message: "runtime identifier exceeds the metadata size limit".into() });
    }
    if identifier.is_empty() || identifier.contains('\0') {
        return Err(RuntimeError::InvalidRequirement {
            reason: "identifier must be nonempty and contain no NUL".into(),
        }
        .into());
    }
    Ok(())
}

/// Constructs the fixed ABI used by built-in runtime providers.
pub(crate) fn builtin_requirement(identifier: &str) -> RuntimeRequirement {
    RuntimeRequirement::new(identifier, 1, Vec::new()).unwrap()
}

/// Validates the fixed ABI and empty runtime configuration of a built-in extension.
pub(crate) fn validate_builtin_requirement(requirement: &RuntimeRequirement, identifier: &str) -> Result<(), Error> {
    if requirement.identifier() != identifier {
        return Err(
            RuntimeError::InvalidRequirement { reason: format!("expected runtime extension `{identifier}`") }.into()
        );
    }
    if requirement.abi_version() != 1 {
        return Err(RuntimeError::UnsupportedVersion {
            identifier: identifier.into(),
            version: requirement.abi_version(),
        }
        .into());
    }
    if !requirement.configuration().is_empty() {
        return Err(RuntimeError::InvalidConfiguration {
            identifier: identifier.into(),
            reason: "built-in runtime configuration must be empty".into(),
        }
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayType, CompilationContext, DataType};

    use crate::domains::sessions::Session;
    use crate::domains::tests::domain_mesh;
    use crate::tests::execution_client;

    use super::*;

    /// Factory borrowing test counters, demonstrating a non-`'static` provider configuration.
    struct CountingProvider<'c> {
        /// Contract identity used by the fixture.
        identifier: &'static str,

        /// Number of actual initializer calls.
        calls: &'c AtomicUsize,

        /// Whether the next initialization must fail before publishing an instance.
        fail: &'c AtomicBool,

        /// Optional counter for per-invocation resource destruction.
        drops: Option<Arc<AtomicUsize>>,
    }

    /// Client-borrowing instance whose optional resources need whole-launch retention.
    struct CountingRuntime<'c> {
        /// Borrowed client checked during invocation preparation.
        client: &'c Client<'c>,

        /// Optional per-invocation resource destructor counter.
        drops: Option<Arc<AtomicUsize>>,
    }

    /// Resource whose destruction establishes completion of the fixture's retained ownership.
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl<'c> RuntimeProvider<'c> for CountingProvider<'c> {
        fn identifier(&self) -> &str {
            self.identifier
        }

        fn initialize(
            &self,
            client: &'c Client<'c>,
            _target: &Target,
            requirement: &RuntimeRequirement,
        ) -> Result<Arc<dyn RuntimeExtension<'c> + 'c>, Error> {
            validate_builtin_requirement(requirement, self.identifier)?;
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail.swap(false, Ordering::SeqCst) {
                return Err(Error::custom(std::io::Error::other("fixture initialization failed")));
            }
            Ok(Arc::new(CountingRuntime { client, drops: self.drops.clone() }))
        }
    }

    impl<'c> RuntimeExtension<'c> for CountingRuntime<'c> {
        fn prepare(&self, resources: &mut ExecutionResources<'c>) -> Result<(), Error> {
            assert!(std::ptr::eq(self.client, resources.client()));
            if let Some(drops) = &self.drops {
                resources.retain(DropCounter(Arc::clone(drops)));
            }
            Ok(())
        }
    }

    #[test]
    fn test_runtime_requirement_new() {
        let requirement = RuntimeRequirement::new("test.provider", 2, vec![1, 2]).unwrap();
        assert_eq!(requirement.identifier(), "test.provider");
        assert_eq!(requirement.abi_version(), 2);
        assert_eq!(requirement.configuration(), &[1, 2]);
        for identifier in ["", "invalid\0identifier"] {
            assert!(
                matches!(RuntimeRequirement::new(identifier, 1, []), Err(Error::Runtime(RuntimeError::InvalidRequirement { reason })) if reason == "identifier must be nonempty and contain no NUL")
            );
        }
        assert!(
            serde_json::from_str::<RuntimeRequirement>(r#"{"identifier":"","abi_version":1,"configuration":[]}"#)
                .is_err()
        );
        assert!(matches!(
            RuntimeRequirement::new("x".repeat(MAX_RUNTIME_REQUIREMENTS_BYTES + 1), 1, []),
            Err(Error::SizeLimitExceeded { message }) if message == "runtime identifier exceeds the metadata size limit"
        ));
        assert!(matches!(
            RuntimeRequirement::new("test.provider", 1, vec![0; MAX_RUNTIME_REQUIREMENTS_BYTES + 1]),
            Err(Error::SizeLimitExceeded { message }) if message == "runtime configuration exceeds the metadata size limit"
        ));
    }

    #[test]
    fn test_normalize_runtime_requirements() {
        let first = builtin_requirement("test.a");
        let second = builtin_requirement("test.b");
        assert_eq!(
            normalize_runtime_requirements([second.clone(), first.clone(), second.clone()]).unwrap(),
            vec![first.clone(), second]
        );
        assert!(
            matches!(normalize_runtime_requirements([first, RuntimeRequirement::new("test.a", 2, []).unwrap()]), Err(Error::Runtime(RuntimeError::ConflictingRequirements { identifier })) if identifier == "test.a")
        );
    }

    #[test]
    fn test_runtime_requirement_deserialization_bounds_configuration() {
        let encoded = serde_json::json!({
            "identifier": "test.provider", "abi_version": 1,
            "configuration": vec![0; MAX_RUNTIME_REQUIREMENTS_BYTES + 1],
        });
        assert_eq!(
            serde_json::from_value::<RuntimeRequirement>(encoded).unwrap_err().to_string(),
            "runtime configuration exceeds the metadata size limit"
        );
    }

    #[test]
    fn test_with_runtime_requirements() {
        let call = CustomCallOperation::new("test.call", vec![ArrayType::scalar(DataType::F32)]);
        let annotated =
            with_runtime_requirements(call, [builtin_requirement("test.b"), builtin_requirement("test.a")]).unwrap();
        assert_eq!(
            runtime_requirements(&annotated).unwrap(),
            vec![builtin_requirement("test.a"), builtin_requirement("test.b")]
        );
        assert!(
            matches!(with_runtime_requirements(annotated, [builtin_requirement("test.a")]), Err(Error::Runtime(RuntimeError::InvalidRequirement { reason })) if reason == "duplicate runtime-requirement annotation")
        );
    }

    #[test]
    fn test_with_runtime_requirements_bounds_encoded_configuration() {
        let call = CustomCallOperation::new("test.call", vec![ArrayType::scalar(DataType::F32)]);
        let requirement =
            RuntimeRequirement::new("test.provider", 1, vec![255; MAX_RUNTIME_REQUIREMENTS_BYTES]).unwrap();
        assert!(matches!(
            with_runtime_requirements(call, [requirement]),
            Err(Error::SizeLimitExceeded { message }) if message == "runtime requirements exceed the metadata size limit"
        ));
    }

    #[test]
    fn test_runtime_requirements_rejects_invalid_annotations() {
        let call = || CustomCallOperation::new("test.call", vec![ArrayType::scalar(DataType::F32)]);
        assert!(
            matches!(runtime_requirements(&call().with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, true)), Err(Error::Runtime(RuntimeError::InvalidRequirement { reason })) if reason == "runtime requirements must be encoded as a string")
        );
        assert!(
            matches!(runtime_requirements(&call().with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, "[2,[]]")), Err(Error::Runtime(RuntimeError::InvalidRequirement { reason })) if reason == "unsupported runtime-requirement encoding version 2")
        );
        assert!(matches!(
            runtime_requirements(&call().with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, "not json")),
            Err(Error::Runtime(RuntimeError::InvalidRequirement { .. }))
        ));
        let duplicate = call()
            .with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, "[1,[]]")
            .with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, "[1,[]]");
        assert!(
            matches!(runtime_requirements(&duplicate), Err(Error::Runtime(RuntimeError::InvalidRequirement { reason })) if reason == "duplicate runtime-requirement annotation")
        );
        assert!(matches!(
            runtime_requirements(
                &call().with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, " ".repeat(MAX_RUNTIME_REQUIREMENTS_BYTES + 1))
            ),
            Err(Error::SizeLimitExceeded { .. })
        ));
    }

    #[test]
    fn test_named_ffi_type_id() {
        const CUDA: FfiTypeId = named_ffi_type_id("ryft.kernel.cuda.runtime");
        let rocm = named_ffi_type_id("ryft.kernel.rocm.runtime");
        let empty = named_ffi_type_id("");

        // Equal names produce equal IDs, including across compile-time and runtime evaluation.
        assert_eq!(named_ffi_type_id("ryft.kernel.cuda.runtime"), CUDA);
        assert_ne!(CUDA, rocm);

        // Every ID lies above XLA's sequentially assigned IDs and is never the unknown ID.
        for id in [CUDA, rocm, empty] {
            assert!(i64::from(id) >= 1 << 62);
            assert!(!id.is_unknown());
        }
    }

    #[test]
    fn test_runtime_extensions_with_provider_and_replacement() {
        let calls = AtomicUsize::new(0);
        let fail = AtomicBool::new(false);
        let provider = || {
            Arc::new(CountingProvider { identifier: "test.provider", calls: &calls, fail: &fail, drops: None })
                as Arc<dyn RuntimeProvider<'_>>
        };
        let configuration = RuntimeExtensions::empty().with_provider(provider()).unwrap();
        assert_eq!(configuration.identifiers().collect::<Vec<_>>(), vec!["test.provider"]);
        assert!(
            matches!(configuration.clone().with_provider(provider()), Err(Error::Runtime(RuntimeError::DuplicateProvider { identifier })) if identifier == "test.provider")
        );
        assert!(configuration.with_replaced_provider(provider()).is_ok());
        assert!(
            matches!(RuntimeExtensions::empty().with_replaced_provider(provider()), Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == "test.provider")
        );
    }

    #[test]
    fn test_runtime_registry_initializes_once_per_session_and_retries_failure() {
        let client = execution_client();
        let calls = AtomicUsize::new(0);
        let fail = AtomicBool::new(true);
        let configuration = RuntimeExtensions::empty()
            .with_provider(Arc::new(CountingProvider {
                identifier: "test.provider",
                calls: &calls,
                fail: &fail,
                drops: None,
            }))
            .unwrap();
        let first = Session::with_runtime_extensions(&client, CompilationContext::new(), configuration.clone());
        let second = Session::with_runtime_extensions(&client, CompilationContext::new(), configuration);
        let requirement = builtin_requirement("test.provider");
        assert!(first.runtimes.ensure(std::slice::from_ref(&requirement), first.target().unwrap()).is_err());
        first.runtimes.ensure(std::slice::from_ref(&requirement), first.target().unwrap()).unwrap();
        first.runtimes.ensure(std::slice::from_ref(&requirement), first.target().unwrap()).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        second.runtimes.ensure(&[requirement], second.target().unwrap()).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_runtime_registry_concurrent_initialization() {
        let client = execution_client();
        let calls = AtomicUsize::new(0);
        let fail = AtomicBool::new(false);
        let configuration = RuntimeExtensions::empty()
            .with_provider(Arc::new(CountingProvider {
                identifier: "test.provider",
                calls: &calls,
                fail: &fail,
                drops: None,
            }))
            .unwrap();
        let session = Session::with_runtime_extensions(&client, CompilationContext::new(), configuration);
        let target = session.target().unwrap();
        std::thread::scope(|threads| {
            for _ in 0..8 {
                let session = &session;
                threads
                    .spawn(move || session.runtimes.ensure(&[builtin_requirement("test.provider")], target).unwrap());
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_runtime_registry_retries_after_recursion_and_unwind() {
        /// Provider that first reenters the session and then panics before succeeding.
        struct ReentrantProvider<'c> {
            /// Weak association used solely to exercise recursive initialization without an ownership cycle.
            session: Mutex<std::sync::Weak<Session<'c>>>,

            /// Number of attempts, selecting the fixture's successive failure modes.
            calls: AtomicUsize,
        }

        impl<'c> RuntimeProvider<'c> for ReentrantProvider<'c> {
            fn identifier(&self) -> &str {
                "test.recursive"
            }

            fn initialize(
                &self,
                client: &'c Client<'c>,
                target: &Target,
                requirement: &RuntimeRequirement,
            ) -> Result<Arc<dyn RuntimeExtension<'c> + 'c>, Error> {
                match self.calls.fetch_add(1, Ordering::SeqCst) {
                    0 => {
                        let session = self.session.lock().unwrap().upgrade().unwrap();
                        session.runtimes.ensure(std::slice::from_ref(requirement), target)?;
                        unreachable!()
                    }
                    1 => panic!("fixture initializer unwound"),
                    _ => Ok(Arc::new(CountingRuntime { client, drops: None })),
                }
            }
        }

        let client = execution_client();
        let provider =
            Arc::new(ReentrantProvider { session: Mutex::new(std::sync::Weak::new()), calls: AtomicUsize::new(0) });
        let configuration = RuntimeExtensions::empty().with_provider(provider.clone()).unwrap();
        let session = Session::with_runtime_extensions(&client, CompilationContext::new(), configuration);
        *provider.session.lock().unwrap() = Arc::downgrade(&session);
        let requirement = builtin_requirement("test.recursive");
        let ensure = || session.runtimes.ensure(std::slice::from_ref(&requirement), session.target().unwrap());
        assert!(
            matches!(ensure(), Err(Error::Runtime(RuntimeError::RecursiveInitialization { identifier })) if identifier == "test.recursive")
        );
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(ensure)).is_err());
        ensure().unwrap();
        ensure().unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_runtime_registry_rejects_concurrent_initialization_cycles() {
        /// Two providers that attempt to initialize each other on separate threads.
        struct DependentProvider<'c> {
            /// This provider's public contract identifier.
            identifier: &'static str,

            /// The other provider requested during its first initialization.
            dependency: &'static str,

            /// Weak association permitting reentry without retaining the session.
            session: Mutex<std::sync::Weak<Session<'c>>>,

            /// Rendezvous ensuring both outer initializers have started before either reenters.
            started: Arc<std::sync::Barrier>,

            /// Dependencies are requested once so a rejected attempt can be retried independently.
            reenter: AtomicBool,
        }

        impl<'c> RuntimeProvider<'c> for DependentProvider<'c> {
            fn identifier(&self) -> &str {
                self.identifier
            }

            fn initialize(
                &self,
                client: &'c Client<'c>,
                target: &Target,
                _requirement: &RuntimeRequirement,
            ) -> Result<Arc<dyn RuntimeExtension<'c> + 'c>, Error> {
                if self.reenter.swap(false, Ordering::SeqCst) {
                    self.started.wait();
                    let session = self.session.lock().unwrap().upgrade().unwrap();
                    session.runtimes.ensure(&[builtin_requirement(self.dependency)], target)?;
                }
                Ok(Arc::new(CountingRuntime { client, drops: None }))
            }
        }

        let client = execution_client();
        let started = Arc::new(std::sync::Barrier::new(2));
        let left = Arc::new(DependentProvider {
            identifier: "test.left",
            dependency: "test.right",
            session: Mutex::new(std::sync::Weak::new()),
            started: Arc::clone(&started),
            reenter: AtomicBool::new(true),
        });
        let right = Arc::new(DependentProvider {
            identifier: "test.right",
            dependency: "test.left",
            session: Mutex::new(std::sync::Weak::new()),
            started,
            reenter: AtomicBool::new(true),
        });
        let configuration = RuntimeExtensions::empty()
            .with_provider(left.clone())
            .unwrap()
            .with_provider(right.clone())
            .unwrap();
        let session = Session::with_runtime_extensions(&client, CompilationContext::new(), configuration);
        *left.session.lock().unwrap() = Arc::downgrade(&session);
        *right.session.lock().unwrap() = Arc::downgrade(&session);
        std::thread::scope(|threads| {
            let run =
                |identifier| session.runtimes.ensure(&[builtin_requirement(identifier)], session.target().unwrap());
            let left = threads.spawn(move || run("test.left"));
            let right = threads.spawn(move || run("test.right"));
            let results = [left.join().unwrap(), right.join().unwrap()];
            assert!(results.iter().any(|result| matches!(result, Err(Error::Runtime(RuntimeError::RecursiveInitialization { identifier })) if identifier == "test.left" || identifier == "test.right")));
            assert!(results.iter().all(|result| matches!(
                result,
                Ok(()) | Err(Error::Runtime(RuntimeError::RecursiveInitialization { .. }))
            )));
        });
        session
            .runtimes
            .ensure(&[builtin_requirement("test.left"), builtin_requirement("test.right")], session.target().unwrap())
            .unwrap();
    }

    #[test]
    fn test_runtime_registry_rejects_missing_provider_before_initialization() {
        let client = execution_client();
        let session = Session::with_runtime_extensions(&client, CompilationContext::new(), RuntimeExtensions::empty());
        assert!(
            matches!(session.runtimes.ensure(&[builtin_requirement("test.missing")], session.target().unwrap()), Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == "test.missing")
        );
    }

    #[test]
    fn test_runtime_registry_registration_only_creates_no_context_or_retained_launch() {
        let client = execution_client();
        let calls = AtomicUsize::new(0);
        let fail = AtomicBool::new(false);
        let configuration = RuntimeExtensions::empty()
            .with_provider(Arc::new(CountingProvider {
                identifier: "test.provider",
                calls: &calls,
                fail: &fail,
                drops: None,
            }))
            .unwrap();
        let session = Session::with_runtime_extensions(&client, CompilationContext::new(), configuration);
        let mesh = domain_mesh(&client, "x", 1);
        for requirements in [Vec::new(), vec![builtin_requirement("test.provider")]] {
            let resources = session.runtimes.prepare(&requirements, session.target().unwrap(), &mesh).unwrap();
            let (context, lease) = resources.into_parts();
            assert!(context.is_none());
            assert!(!lease.needed);
            let (event, promise) = client.event(()).unwrap();
            let fence = ExecutionFence::new(vec![event]);
            session.runtimes.retain(&fence, lease);
            assert!(session.runtimes.pending.lock().unwrap().is_empty());
            promise.set(None).unwrap();
            fence.block_until_ready().unwrap();
        }
    }

    #[test]
    fn test_runtime_registry_drop_retains_resources_after_dropped_outputs() {
        let client = execution_client();
        let calls = AtomicUsize::new(0);
        let fail = AtomicBool::new(false);
        let drops = Arc::new(AtomicUsize::new(0));
        let configuration = RuntimeExtensions::empty()
            .with_provider(Arc::new(CountingProvider {
                identifier: "test.provider",
                calls: &calls,
                fail: &fail,
                drops: Some(Arc::clone(&drops)),
            }))
            .unwrap();
        let session = Session::with_runtime_extensions(&client, CompilationContext::new(), configuration);
        let resources = session
            .runtimes
            .prepare(&[builtin_requirement("test.provider")], session.target().unwrap(), &domain_mesh(&client, "x", 1))
            .unwrap();
        let (_, lease) = resources.into_parts();
        let (first, first_promise) = client.event(()).unwrap();
        let (second, second_promise) = client.event(()).unwrap();
        let fence = ExecutionFence::new(vec![first, second]);
        session.runtimes.retain(&fence, lease);
        drop(fence);
        std::thread::scope(|threads| {
            let (started, waiting) = std::sync::mpsc::channel();
            let (released, completed) = std::sync::mpsc::channel();
            threads.spawn(move || {
                started.send(()).unwrap();
                drop(session);
                released.send(()).unwrap();
            });
            waiting.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(matches!(
                completed.recv_timeout(Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            first_promise.set(Some(ryft_pjrt::Error::aborted("terminal test failure"))).unwrap();
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            assert!(matches!(
                completed.recv_timeout(Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            second_promise.set(None).unwrap();
            completed.recv_timeout(Duration::from_secs(5)).unwrap();
        });
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_runtime_registry_releases_completed_resources_without_session_drop() {
        let client = execution_client();
        let calls = AtomicUsize::new(0);
        let fail = AtomicBool::new(false);
        let drops = Arc::new(AtomicUsize::new(0));
        let configuration = RuntimeExtensions::empty()
            .with_provider(Arc::new(CountingProvider {
                identifier: "test.provider",
                calls: &calls,
                fail: &fail,
                drops: Some(Arc::clone(&drops)),
            }))
            .unwrap();
        let session = Session::with_runtime_extensions(&client, CompilationContext::new(), configuration);
        let resources = session
            .runtimes
            .prepare(&[builtin_requirement("test.provider")], session.target().unwrap(), &domain_mesh(&client, "x", 1))
            .unwrap();
        let (_, lease) = resources.into_parts();
        let (event, promise) = client.event(()).unwrap();
        let fence = ExecutionFence::new(vec![event]);
        session.runtimes.retain(&fence, lease);
        promise.set(None).unwrap();
        fence.block_until_ready().unwrap();
        let resources = session
            .runtimes
            .prepare(&[builtin_requirement("test.provider")], session.target().unwrap(), &domain_mesh(&client, "x", 1))
            .unwrap();
        let (_, lease) = resources.into_parts();
        session.runtimes.retain(&fence, lease);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert!(session.runtimes.pending.lock().unwrap().is_empty());
    }
}
