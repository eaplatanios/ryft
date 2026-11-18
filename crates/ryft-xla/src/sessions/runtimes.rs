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

/// Represents errors produced while declaring [`RuntimeRequirement`]s, configuring [`RuntimeExtensions`], initializing
/// a [`RuntimeExtension`], or attaching native user data to [`ExecutionResources`]. Runtime APIs return these errors
/// through [`Error::Runtime`].
#[derive(Clone, Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("invalid runtime requirement: {reason}")]
    InvalidRequirement { reason: String },

    #[error("conflicting requirements for runtime extension `{identifier}`")]
    ConflictingRequirements { identifier: String },

    #[error("no provider is configured for runtime extension `{identifier}`")]
    MissingProvider { identifier: String },

    #[error("a provider is already configured for runtime extension `{identifier}`")]
    DuplicateProvider { identifier: String },

    #[error("runtime extension `{identifier}` does not support ABI version {version}")]
    UnsupportedVersion { identifier: String, version: u32 },

    #[error("invalid configuration for runtime extension `{identifier}`: {reason}")]
    InvalidConfiguration { identifier: String, reason: String },

    #[error("recursive initialization of runtime extension `{identifier}`")]
    RecursiveInitialization { identifier: String },

    #[error("execution context already contains user data for FFI type {type_id}")]
    DuplicateUserData { type_id: i64 },
}

/// Reserved attribute of a source operation that holds canonical runtime requirements. XLA lowering consumes this
/// attribute and does not forward it to the native custom call handler. Other custom call attributes retain their
/// usual meaning.
pub const RUNTIME_REQUIREMENTS_ATTRIBUTE: &str = "ryft.runtime.requirements";

/// Maximum size of one encoded runtime requirement annotation, in bytes.
pub const MAX_RUNTIME_REQUIREMENTS_BYTES: usize = 1024 * 1024;

/// Immutable, portable identity of a runtime extension contract. Identifiers must be non-empty and contain no
/// NUL characters. Each provider defines the invocation contract of every ABI version it supports. Compiled programs
/// retain only these requirements. The invoking [`Session`](crate::Session) resolves them through its configured
/// [`RuntimeProvider`]s, so cached executables retain no producing session or initialized extensions.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRequirement {
    /// Namespaced identity of the runtime contract.
    #[serde(deserialize_with = "RuntimeRequirement::deserialize_identifier")]
    identifier: String,

    /// Version of the extension's invocation ABI.
    abi_version: u32,
}

impl RuntimeRequirement {
    /// Creates a [`RuntimeRequirement`] after validating its identifier.
    #[inline]
    pub fn new(identifier: impl Into<String>, abi_version: u32) -> Result<Self, Error> {
        let identifier = identifier.into();
        validate_identifier(&identifier)?;
        Ok(Self { identifier, abi_version })
    }

    /// Returns the namespaced runtime contract identifier.
    #[inline]
    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    /// Returns the requested invocation ABI version.
    #[inline]
    pub fn abi_version(&self) -> u32 {
        self.abi_version
    }

    /// Deserializes a runtime identifier and applies the same validation as [`Self::new`].
    fn deserialize_identifier<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
        let identifier = String::deserialize(deserializer)?;
        validate_identifier(&identifier).map_err(serde::de::Error::custom)?;
        Ok(identifier)
    }
}

/// Sorts and deduplicates the provided [`RuntimeRequirement`]s, rejecting incompatible declarations for one identifier.
pub fn normalize_runtime_requirements<R: IntoIterator<Item = RuntimeRequirement>>(
    requirements: R,
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

/// Encoding version emitted by [`CustomCallOperationExtension::with_runtime_requirements`].
const RUNTIME_REQUIREMENTS_ENCODING_VERSION: u32 = 1;

/// XLA runtime requirement annotations for a [`CustomCallOperation`].
pub trait CustomCallOperationExtension: Sized {
    /// Declares the native runtime contracts a custom call needs by encoding its [`RuntimeRequirement`]s in the
    /// [`RUNTIME_REQUIREMENTS_ATTRIBUTE`] for XLA lowering. Identical requirements deduplicate. An empty list skips
    /// annotation. XLA lowering consumes the annotation and excludes it from the attributes passed to the native
    /// Foreign Function Interface (FFI) handler.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation already has an annotation, if requirements conflict, or if the encoded
    /// annotation exceeds [`MAX_RUNTIME_REQUIREMENTS_BYTES`].
    fn with_runtime_requirements<R: IntoIterator<Item = RuntimeRequirement>>(
        self,
        requirements: R,
    ) -> Result<Self, Error>;

    /// Reads and validates the [`RUNTIME_REQUIREMENTS_ATTRIBUTE`] of a custom call and returns its canonical
    /// [`RuntimeRequirement`]s. Returns an empty list when the attribute is absent.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate annotations, invalid or unsupported encodings, conflicting requirements,
    /// or annotations that exceed [`MAX_RUNTIME_REQUIREMENTS_BYTES`].
    fn runtime_requirements(&self) -> Result<Vec<RuntimeRequirement>, Error>;
}

impl CustomCallOperationExtension for CustomCallOperation {
    fn with_runtime_requirements<R: IntoIterator<Item = RuntimeRequirement>>(
        self,
        requirements: R,
    ) -> Result<Self, Error> {
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

        if self.attributes().iter().any(|(name, _)| name == RUNTIME_REQUIREMENTS_ATTRIBUTE) {
            return Err(
                RuntimeError::InvalidRequirement { reason: "duplicate runtime-requirement annotation".into() }.into()
            );
        }

        let requirements = normalize_runtime_requirements(requirements)?;
        if requirements.is_empty() {
            return Ok(self);
        }

        let mut encoded = Annotation { bytes: Vec::new(), exceeded: false };
        if let Err(error) = serde_json::to_writer(&mut encoded, &(RUNTIME_REQUIREMENTS_ENCODING_VERSION, requirements))
        {
            if encoded.exceeded {
                return Err(Error::SizeLimitExceeded {
                    message: "runtime requirements exceed the metadata size limit".into(),
                });
            }
            return Err(RuntimeError::InvalidRequirement { reason: error.to_string() }.into());
        }

        Ok(self.with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, String::from_utf8(encoded.bytes).unwrap()))
    }

    fn runtime_requirements(&self) -> Result<Vec<RuntimeRequirement>, Error> {
        let mut annotations = self.attributes().iter().filter(|(name, _)| name == RUNTIME_REQUIREMENTS_ATTRIBUTE);
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
            return Err(Error::SizeLimitExceeded {
                message: "runtime requirements exceed the metadata size limit".into(),
            });
        }

        let (version, requirements): (u32, Vec<RuntimeRequirement>) = serde_json::from_str(encoded)
            .map_err(|error| RuntimeError::InvalidRequirement { reason: error.to_string() })?;
        if version != RUNTIME_REQUIREMENTS_ENCODING_VERSION {
            return Err(RuntimeError::InvalidRequirement {
                reason: format!("unsupported runtime-requirement encoding version {version}"),
            }
            .into());
        }

        normalize_runtime_requirements(requirements)
    }
}

/// Factory for one named runtime contract, configured before a session is constructed. Providers use the invoking
/// client to validate their settings, the requested ABI, and the client's native capabilities before returning an
/// instance. Instances must not retain an owning session handle. Reentry into an unfinished initializer is rejected.
/// A nested initialization also cannot wait for another thread's initializer in the same session, which prevents
/// dependency cycles between concurrent providers.
///
/// Providers register native types with explicit IDs from [`FfiTypeId::from_name`] and handlers through PJRT during
/// initialization. Type registration is idempotent, and XLA accepts re-registration of an identical handler. Each
/// session's providers can therefore register again without bookkeeping that is global to the process, while native
/// handlers look up their user data with the same constant ID.
///
/// Providers and extensions own their state, and they may hold the client, which lives for the rest of the process.
/// See [`RuntimeExtensions`] for how a session configures providers and uses its initialized extensions.
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// #
/// # use ryft_core::CustomCallOperation;
/// # use ryft_pjrt::Client;
/// # use ryft_xla::sessions::{
/// #     CustomCallOperationExtension, RuntimeExtension, RuntimeExtensions, RuntimeProvider, RuntimeRequirement,
/// # };
/// # use ryft_xla::{Error, RuntimeError, Session};
///
/// struct Runtime { client: &'static Client<'static> }
///
/// impl RuntimeExtension for Runtime {}
///
/// struct Provider;
///
/// impl RuntimeProvider for Provider {
///     fn identifier(&self) -> &str { "example.runtime" }
///
///     fn initialize(
///         &self,
///         client: &'static Client<'static>,
///         requirement: &RuntimeRequirement,
///     ) -> Result<Arc<dyn RuntimeExtension>, Error> {
///         if requirement.abi_version() != 1 {
///             return Err(RuntimeError::UnsupportedVersion {
///                 identifier: self.identifier().into(), version: requirement.abi_version(),
///             }.into());
///         }
///
///         // Validate provider settings and client capabilities, then register this contract's native handlers here.
///         Ok(Arc::new(Runtime { client }))
///     }
/// }
///
/// fn configure(client: &'static Client<'static>) -> Result<Session, Error> {
///     let providers = RuntimeExtensions::new().with_provider(Arc::new(Provider))?;
///     let requirement = RuntimeRequirement::new("example.runtime", 1)?;
///     let operation = CustomCallOperation::new("example.call", vec![]).with_runtime_requirements([requirement])?;
///
///     // Apply `operation` in the staged program. XLA consumes its reserved runtime annotation during lowering.
///     Session::builder().client(client).runtime_extensions(providers).build()
/// }
/// ```
pub trait RuntimeProvider: Send + Sync {
    /// Returns the contract identifier that this provider implements.
    fn identifier(&self) -> &str;

    /// Validates and initializes one contract instance associated with `client`. A failed initialization may be
    /// retried, so registration steps that are global to the plugin must be idempotent and preserve compatible partial
    /// progress.
    fn initialize(
        &self,
        client: &'static Client<'static>,
        requirement: &RuntimeRequirement,
    ) -> Result<Arc<dyn RuntimeExtension>, Error>;
}

/// Live extension instance shared by calls through one [`Session`](crate::Session). The session initializes one
/// instance for each [`RuntimeRequirement`] before it first compiles, loads, or executes a program that needs the
/// requirement. Later calls reuse the instance. The session keeps its stable allocation alive until all executions
/// borrowing extension data have completed. Implementations must synchronize native handler access across devices and
/// threads. Extensions that only register native handlers may use the default preparation behavior, which does not
/// create an execution context. A runtime cannot borrow a client that does not live for the rest of the process:
///
/// ```compile_fail
/// # use std::sync::Arc;
/// #
/// # use ryft_pjrt::Client;
/// # use ryft_xla::sessions::RuntimeExtension;
///
/// struct BorrowedRuntime<'c>(&'c Client<'c>);
///
/// impl<'c> RuntimeExtension for BorrowedRuntime<'c> {}
///
/// fn escaped<'c>(client: Client<'c>) -> Arc<dyn RuntimeExtension + 'c> {
///     Arc::new(BorrowedRuntime(&client))
/// }
/// ```
pub trait RuntimeExtension: Send + Sync {
    /// Checks device and participant restrictions for the invocation's mesh before native submission.
    /// Extensions that need live device information can retain the client they received during initialization.
    #[inline]
    fn validate(&self, mesh: &DeviceMesh) -> Result<(), Error> {
        let _ = mesh;
        Ok(())
    }

    /// Attaches invocation data before native submission and retains any additional resources needed through completion.
    /// The session calls this method on each required extension with the same [`ExecutionResources`] for the
    /// invocation.
    #[inline]
    fn prepare(&self, resources: &mut ExecutionResources) -> Result<(), Error> {
        let _ = resources;
        Ok(())
    }
}

/// Invocation data and retained resources for one native submission. Retain stable allocations such as `Arc<T>`
/// before attaching pointers into them. If preparation fails, the context is destroyed before the retained resources.
/// After submission, the session keeps the resources alive through the joined execution fence, including when callers
/// discard the outputs. Extensions share one execution context, created on the first native user data attachment.
/// Ordinary execution and extensions that only register native handlers need no extra destruction barrier.
pub struct ExecutionResources {
    /// Client receiving this invocation's context.
    client: &'static Client<'static>,

    /// [`ExecutionContext`], allocated only when invocation data is first attached. Fields drop in declaration order,
    /// so this context is destroyed before the retained resources, including when preparation fails.
    context: Option<ExecutionContext>,

    /// Type identifiers installed for this invocation.
    user_data: HashSet<FfiTypeId>,

    /// Additional resources held through execution.
    retained: RetainedExecutionResources,
}

impl ExecutionResources {
    /// Returns the client receiving the invocation.
    #[inline]
    pub fn client(&self) -> &'static Client<'static> {
        self.client
    }

    /// Retains a resource until the whole execution completes. When native user data points into the resource, use
    /// an allocation such as `Arc` or `Box` to keep the pointee's address stable when moving the resource into this
    /// method.
    #[inline]
    pub fn retain<T: 'static + Send + Sync>(&mut self, resource: T) {
        self.retained.push(Box::new(resource));
    }

    /// Adds one borrowed native user data binding to this invocation, creating the context lazily. `data` pairs a
    /// native [`FfiTypeId`] with a raw pointer, which the native handlers of this invocation look up by that type ID
    /// and dereference during execution.
    ///
    /// # Safety
    ///
    /// Neither the pointer nor its pointee is checked. The pointer must point to a value of the type registered under
    /// its native type ID and remain stable until completion. The session's runtime instance or a resource held through
    /// [`Self::retain`] must keep the pointee alive. Native handlers may access it concurrently on several devices, so
    /// the pointee must support those accesses. The function is public so that third-party FFI integrations can attach
    /// their own data, and its `unsafe` boundary keeps native pointer insertion explicit.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::DuplicateUserData`] if this invocation already holds user data for the same type ID,
    /// and an [`Error`] if the execution context cannot be created or the native insertion fails.
    pub unsafe fn add_ffi_user_data(&mut self, data: FfiUserData) -> Result<(), Error> {
        let type_id = data.type_id;
        if self.user_data.contains(&type_id) {
            return Err(RuntimeError::DuplicateUserData { type_id: type_id.into() }.into());
        }

        if self.context.is_none() {
            self.context = Some(self.client.execution_context()?);
        }

        unsafe { self.context.as_ref().unwrap().add_ffi_user_data(data) }?;

        // The type ID is recorded only after the native insertion succeeds, so that a failed call can be retried.
        self.user_data.insert(type_id);

        Ok(())
    }

    /// Returns the optional [`ExecutionContext`] for PJRT submission and the resources the session keeps alive until
    /// the execution fence is ready. The second value is `Some` when the invocation has a context or additional
    /// resources. A context can borrow from the session's runtime instance, so even a context with no additional
    /// resources requires retaining the fence.
    pub(crate) fn into_parts(self) -> (Option<ExecutionContext>, Option<RetainedExecutionResources>) {
        let Self { context, retained, .. } = self;
        let resources = (context.is_some() || !retained.is_empty()).then_some(retained);
        (context, resources)
    }
}

/// Invocation resources retained by the session through execution.
pub(crate) type RetainedExecutionResources = Vec<Box<dyn Send + Sync>>;

/// Configures the runtime extensions available to a [`Session`](crate::Session), such as native print handlers and
/// precompiled kernel launchers.
///
/// Supply this configuration through [`SessionBuilder::runtime_extensions`](super::SessionBuilder::runtime_extensions)
/// before building a session. [`Self::new`] includes the built-in providers, and [`Self::with_provider`] adds another
/// [`RuntimeProvider`] under its contract identifier. Programs declare the native support they need through
/// [`RuntimeRequirement`]s, each of which names a provider and specifies the required ABI version.
///
/// On first use of a requirement during compilation, loading, or execution, the session calls the matching
/// [`RuntimeProvider::initialize`] with its PJRT [`Client`]. The session reuses the resulting [`RuntimeExtension`]
/// for later calls that need the same requirement. Sessions built from the same configuration each initialize their
/// own instances. Compiled programs carry the requirements so another compatible session can resolve them through
/// its own providers.
///
/// Before each native submission, the session calls [`RuntimeExtension::validate`] for the invocation's mesh and
/// [`RuntimeExtension::prepare`] to attach any invocation data to shared [`ExecutionResources`]. The session keeps
/// those resources alive until execution completes, even if callers discard the outputs.
///
/// The [provider example](RuntimeProvider#example) shows how to implement a provider, add it to a session,
/// and annotate the custom calls that require it.
#[derive(Clone)]
pub struct RuntimeExtensions {
    /// [`RuntimeProvider`]s keyed by their public contract identifiers.
    providers: BTreeMap<String, Arc<dyn RuntimeProvider>>,
}

// TODO(eaplatanios): Review from here onwards.

impl RuntimeExtensions {
    /// Creates the built-in [`RuntimeExtensions`], with providers for assertions (i.e., `ryft.assert`), ordered I/O
    /// (i.e., `ryft.ordered-io`), printing (i.e., `ryft.print`), and CUDA kernels (i.e., `ryft.kernel.cuda`). The
    /// `rocm` feature also adds a provider for ROCm kernels (i.e., `ryft.kernel.rocm`).
    ///
    /// The session initializes runtime extensions only as programs require them. Creating this configuration does
    /// not load accelerator drivers.
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
    pub fn with_provider(mut self, provider: Arc<dyn RuntimeProvider>) -> Result<Self, Error> {
        validate_identifier(provider.identifier())?;
        let identifier = provider.identifier().to_owned();
        if self.providers.contains_key(&identifier) {
            return Err(RuntimeError::DuplicateProvider { identifier }.into());
        }
        self.providers.insert(identifier, provider);
        Ok(self)
    }

    /// Explicitly replaces a configured provider before session construction. This does not replace an incompatible
    /// native registration that is global to the plugin and does not mutate existing sessions using a cloned
    /// configuration.
    pub fn with_replaced_provider(mut self, provider: Arc<dyn RuntimeProvider>) -> Result<Self, Error> {
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

impl Default for RuntimeExtensions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for RuntimeExtensions {
    #[inline]
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeExtensions")
            .field("identifiers", &self.identifiers().collect::<Vec<_>>())
            .finish()
    }
}

/// Runtime extension instances of one [`Session`](crate::Session) and the ownership of the resources of its pending
/// executions.
///
/// The registry manages the session lifecycle described by [`RuntimeExtensions`].
///
///   - **Initialization:** [`Self::ensure`] creates the instance of each requirement on first use and shares it with
///     later calls. For example, lowering a program that prints attaches the `ryft.print` requirement to it. Before
///     the session first compiles, loads, or executes such a program, the registry asks the print provider to
///     initialize, which registers the native print handler with the client. Later programs that print reuse that
///     instance.
///   - **Preparation:** Before each native submission, [`Self::prepare`] lets every required instance validate the
///     invocation and attach its data to one [`ExecutionResources`]. The print extension attaches nothing, while the
///     CUDA kernel extension attaches a pointer to itself, which its native handler reads while it launches kernels.
///   - **Retention:** [`Self::retain`] keeps those resources alive until the execution completes, even when its
///     outputs are dropped. Dropping the registry waits for the executions that still borrow resources.
pub(crate) struct RuntimeRegistry {
    /// Externally owned client associated with all initialized instances.
    client: &'static Client<'static>,

    /// Frozen provider configuration, shared by value rather than sharing initialized instances.
    configuration: RuntimeExtensions,

    /// Initialization slots, one for each requirement.
    instances: Mutex<HashMap<RuntimeRequirement, Arc<RuntimeInitialization>>>,

    /// Only launches that borrow extension resources, not ordinary PJRT work.
    pending: Mutex<Vec<PendingExecution>>,
}

impl RuntimeRegistry {
    /// Attaches immutable provider configuration to a new session.
    pub(super) fn new(client: &'static Client<'static>, configuration: RuntimeExtensions) -> Self {
        Self { client, configuration, instances: Mutex::new(HashMap::new()), pending: Mutex::new(Vec::new()) }
    }

    /// Initializes all required providers before native compilation or loading.
    pub(crate) fn ensure(&self, requirements: &[RuntimeRequirement]) -> Result<(), Error> {
        for requirement in normalize_runtime_requirements(requirements.iter().cloned())? {
            self.instance(&requirement)?;
        }
        Ok(())
    }

    /// Validates the selected invocation and composes all requested extension state into at most one context.
    pub(crate) fn prepare(
        &self,
        requirements: &[RuntimeRequirement],
        mesh: &DeviceMesh,
    ) -> Result<ExecutionResources, Error> {
        let mut resources =
            ExecutionResources { client: self.client, context: None, user_data: HashSet::new(), retained: Vec::new() };
        for requirement in normalize_runtime_requirements(requirements.iter().cloned())? {
            let runtime = self.instance(&requirement)?;
            runtime.validate(mesh)?;
            runtime.prepare(&mut resources)?;
        }
        Ok(resources)
    }

    /// Tracks successful submissions that need retention before any fallible result reconstruction.
    pub(crate) fn retain(&self, fence: &ExecutionFence, resources: Option<RetainedExecutionResources>) {
        let Some(resources) = resources else {
            return;
        };
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
        // Completed executions no longer need their resources. They are dropped after the pending lock is released,
        // so destructors of retained resources never run while that lock is held.
        drop(completed);
    }

    /// Initializes outside registry locks and publishes one stable successful instance.
    fn instance(&self, requirement: &RuntimeRequirement) -> Result<Arc<dyn RuntimeExtension>, Error> {
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
        let runtime = provider.initialize(self.client, requirement)?;
        *slot.state.lock().expect("runtime initialization mutex poisoned") =
            Initialization::Ready(Arc::clone(&runtime));
        rollback.active = false;
        slot.ready.notify_all();
        Ok(runtime)
    }
}

impl Drop for RuntimeRegistry {
    fn drop(&mut self) {
        let pending = std::mem::take(self.pending.get_mut().expect("runtime completion mutex poisoned"));
        // Native handlers may still read the retained resources of pending executions, so each execution must reach a
        // terminal state before its resources are destroyed.
        for execution in pending {
            // Failure is terminal too, and its diagnostics remain in the caller's fence and in effect and reference
            // state.
            match execution.fence.block_until_ready() {
                Ok(()) | Err(_) => {}
            }
            drop(execution);
        }
    }
}

/// Constructs the fixed ABI used by built-in runtime providers.
pub(crate) fn builtin_requirement(identifier: &str) -> RuntimeRequirement {
    RuntimeRequirement::new(identifier, 1).unwrap()
}

/// Validates the identifier and fixed ABI of a built-in extension.
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
    Ok(())
}

/// Initialization state for one session and complete runtime descriptor.
enum Initialization {
    /// No successful instance has been published.
    Empty,

    /// One thread initializes outside the registry lock.
    Running(ThreadId),

    /// Immutable stable instance shared by later calls.
    Ready(Arc<dyn RuntimeExtension>),
}

/// Independent synchronization for one runtime, so unrelated initialization does not serialize.
struct RuntimeInitialization {
    /// Mutable initialization state.
    state: Mutex<Initialization>,

    /// Notification after successful initialization or retryable failure.
    ready: Condvar,
}

/// Resets an unfinished initializer on error or unwind, without poisoning the instance slot.
struct InitializationGuard<'o> {
    /// Slot whose running state must be cleared unless publication succeeds.
    slot: &'o RuntimeInitialization,

    /// Whether initialization still needs rollback.
    active: bool,
}

impl Drop for InitializationGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            *self.slot.state.lock().expect("runtime initialization mutex poisoned") = Initialization::Empty;
            self.slot.ready.notify_all();
        }
    }
}

/// One execution that uses extensions and whose resources outlive discarded output handles.
struct PendingExecution {
    /// Completion of every participating device and native context cleanup.
    fence: ExecutionFence,

    /// Borrowed invocation resources retained until the fence is terminal.
    _resources: RetainedExecutionResources,
}

/// Validates an extension identifier at a public configuration boundary.
fn validate_identifier(identifier: &str) -> Result<(), Error> {
    if identifier.len() > MAX_RUNTIME_REQUIREMENTS_BYTES {
        return Err(Error::SizeLimitExceeded { message: "runtime identifier exceeds the metadata size limit".into() });
    }

    if identifier.is_empty() || identifier.contains('\0') {
        return Err(RuntimeError::InvalidRequirement {
            reason: "identifier must be non-empty and contain no NUL".into(),
        }
        .into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::sync::{Arc, Barrier, Mutex, OnceLock, Weak};
    use std::thread;
    use std::time::Duration;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{CustomCallAttribute, CustomCallOperation, DeviceMesh};
    use ryft_pjrt::extensions::ffi::{FfiTypeId, FfiTypeInformation, FfiUserData};
    use ryft_pjrt::{Client, ExecutionFence};

    use crate::Error;
    use crate::sessions::tests::domain_mesh;
    use crate::tests::execution_client;

    use super::{
        CustomCallOperationExtension, ExecutionResources, MAX_RUNTIME_REQUIREMENTS_BYTES,
        RUNTIME_REQUIREMENTS_ATTRIBUTE, RuntimeError, RuntimeExtension, RuntimeExtensions, RuntimeProvider,
        RuntimeRegistry, RuntimeRequirement, builtin_requirement, normalize_runtime_requirements,
        validate_builtin_requirement,
    };

    /// Provider that counts initialization attempts and can fail the next attempt.
    struct CountingProvider {
        /// Identifier of the contract provided by this fixture.
        identifier: &'static str,

        /// Number of calls to the initializer.
        calls: AtomicUsize,

        /// Whether the next initializer call should return an error.
        fail: AtomicBool,
    }

    impl CountingProvider {
        /// Creates a provider with no initialization attempts or scheduled failures.
        fn new(identifier: &'static str) -> Self {
            Self { identifier, calls: AtomicUsize::new(0), fail: AtomicBool::new(false) }
        }
    }

    impl RuntimeProvider for CountingProvider {
        fn identifier(&self) -> &str {
            self.identifier
        }

        fn initialize(
            &self,
            client: &'static Client<'static>,
            requirement: &RuntimeRequirement,
        ) -> Result<Arc<dyn RuntimeExtension>, Error> {
            let _ = (client, requirement);
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail.swap(false, Ordering::SeqCst) {
                return Err(Error::custom(std::io::Error::other("fixture initialization failed")));
            }
            Ok(Arc::new(RegistrationOnlyRuntime))
        }
    }

    /// Runtime that needs no invocation context or retained resources.
    struct RegistrationOnlyRuntime;

    impl RuntimeExtension for RegistrationOnlyRuntime {}

    /// Resource that increments a counter when the registry releases it.
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Prepares one single-device invocation without runtime requirements.
    fn prepare_without_requirements(registry: &RuntimeRegistry) -> ExecutionResources {
        let client = registry.client;
        registry.prepare(&[], &domain_mesh(client, "x", 1)).unwrap()
    }

    #[test]
    fn test_runtime_requirement_new() {
        let requirement = RuntimeRequirement::new("test.provider", 2).unwrap();
        assert_eq!(requirement.identifier(), "test.provider");
        assert_eq!(requirement.abi_version(), 2);

        // Runtime contract identifiers must be non-empty and contain no NUL.
        assert!(matches!(
            RuntimeRequirement::new("", 1),
            Err(Error::Runtime(RuntimeError::InvalidRequirement { reason }))
                if reason == "identifier must be non-empty and contain no NUL",
        ));
        assert!(matches!(
            RuntimeRequirement::new("invalid\0identifier", 1),
            Err(Error::Runtime(RuntimeError::InvalidRequirement { reason }))
                if reason == "identifier must be non-empty and contain no NUL",
        ));

        assert!(matches!(
            RuntimeRequirement::new("x".repeat(MAX_RUNTIME_REQUIREMENTS_BYTES + 1), 1),
            Err(Error::SizeLimitExceeded { message })
                if message == "runtime identifier exceeds the metadata size limit",
        ));
    }

    #[test]
    fn test_runtime_requirement_deserialize() {
        let encoded = serde_json::json!({"identifier": "test.provider", "abi_version": 2});
        let requirement = RuntimeRequirement::new("test.provider", 2).unwrap();
        assert_eq!(serde_json::from_value::<RuntimeRequirement>(encoded.clone()).unwrap(), requirement);
        assert_eq!(serde_json::to_value(&requirement).unwrap(), encoded);

        // Deserialization applies the same identifier rules as the constructor.
        assert_eq!(
            serde_json::from_value::<RuntimeRequirement>(serde_json::json!({"identifier": "", "abi_version": 1}),)
                .unwrap_err()
                .to_string(),
            "invalid runtime requirement: identifier must be non-empty and contain no NUL",
        );
    }

    #[test]
    fn test_normalize_runtime_requirements() {
        let first = builtin_requirement("test.a");
        let second = builtin_requirement("test.b");
        assert_eq!(
            normalize_runtime_requirements([second.clone(), first.clone(), second.clone()]).unwrap(),
            vec![first.clone(), second],
        );

        // Each provider can supply only one ABI version to a compiled program.
        assert!(matches!(
            normalize_runtime_requirements([first, RuntimeRequirement::new("test.a", 2).unwrap()]),
            Err(Error::Runtime(RuntimeError::ConflictingRequirements { identifier })) if identifier == "test.a",
        ));
    }

    #[test]
    fn test_custom_call_operation_with_runtime_requirements() {
        let call = CustomCallOperation::new("test.call", vec![]).with_attribute("enabled", true);
        let first = builtin_requirement("test.a");
        let second = RuntimeRequirement::new("test.b", 2).unwrap();
        let annotated = call.clone().with_runtime_requirements([second.clone(), first.clone(), second]).unwrap();
        assert_eq!(
            annotated.attributes(),
            &[
                ("enabled".into(), CustomCallAttribute::from(true)),
                (
                    RUNTIME_REQUIREMENTS_ATTRIBUTE.into(),
                    CustomCallAttribute::from(concat!(
                        r#"[1,[{"identifier":"test.a","abi_version":1},"#,
                        r#"{"identifier":"test.b","abi_version":2}]]"#,
                    )),
                ),
            ],
        );

        // Empty requirements leave the original operation intact.
        assert_eq!(call.clone().with_runtime_requirements([]).unwrap(), call);

        // Each operation accepts one runtime-requirement annotation.
        assert!(matches!(
            annotated.with_runtime_requirements([first]),
            Err(Error::Runtime(RuntimeError::InvalidRequirement { reason }))
                if reason == "duplicate runtime-requirement annotation",
        ));

        // The size limit applies to the combined encoding of all requirements.
        let first = RuntimeRequirement::new("a".repeat(MAX_RUNTIME_REQUIREMENTS_BYTES / 2), 1).unwrap();
        let second = RuntimeRequirement::new("b".repeat(MAX_RUNTIME_REQUIREMENTS_BYTES / 2), 1).unwrap();
        assert!(matches!(
            call.with_runtime_requirements([first, second]),
            Err(Error::SizeLimitExceeded { message })
                if message == "runtime requirements exceed the metadata size limit",
        ));
    }

    #[test]
    fn test_custom_call_operation_runtime_requirements() {
        let call = CustomCallOperation::new("test.call", vec![]).with_attribute("enabled", true);
        assert_eq!(call.runtime_requirements().unwrap(), Vec::new());

        // Readers normalize persisted annotations, including ones written in a different order.
        let annotated = call.clone().with_attribute(
            RUNTIME_REQUIREMENTS_ATTRIBUTE,
            indoc! {r#"
                [1, [
                    {"identifier": "test.b", "abi_version": 2},
                    {"identifier": "test.a", "abi_version": 1},
                    {"identifier": "test.b", "abi_version": 2}
                ]]
            "#},
        );
        assert_eq!(
            annotated.runtime_requirements().unwrap(),
            vec![builtin_requirement("test.a"), RuntimeRequirement::new("test.b", 2).unwrap()],
        );

        // The annotation must be a JSON string with a supported encoding version.
        assert!(matches!(
            call.clone().with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, true).runtime_requirements(),
            Err(Error::Runtime(RuntimeError::InvalidRequirement { reason }))
                if reason == "runtime requirements must be encoded as a string",
        ));
        assert!(matches!(
            call.clone().with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, "[2,[]]").runtime_requirements(),
            Err(Error::Runtime(RuntimeError::InvalidRequirement { reason }))
                if reason == "unsupported runtime-requirement encoding version 2",
        ));
        // The parser supplies the reason for malformed JSON.
        assert!(matches!(
            call.clone().with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, "not json").runtime_requirements(),
            Err(Error::Runtime(RuntimeError::InvalidRequirement { .. })),
        ));
        let duplicate = annotated.with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, "[1,[]]");
        assert!(matches!(
            duplicate.runtime_requirements(),
            Err(Error::Runtime(RuntimeError::InvalidRequirement { reason }))
                if reason == "duplicate runtime-requirement annotation",
        ));

        // The size limit takes precedence over JSON parsing.
        assert!(matches!(
            call.with_attribute(RUNTIME_REQUIREMENTS_ATTRIBUTE, " ".repeat(MAX_RUNTIME_REQUIREMENTS_BYTES + 1))
                .runtime_requirements(),
            Err(Error::SizeLimitExceeded { message })
                if message == "runtime requirements exceed the metadata size limit",
        ));
    }

    #[test]
    fn test_runtime_extensions_new() {
        #[cfg(feature = "rocm")]
        let expected = vec!["ryft.assert", "ryft.kernel.cuda", "ryft.kernel.rocm", "ryft.ordered-io", "ryft.print"];
        #[cfg(not(feature = "rocm"))]
        let expected = vec!["ryft.assert", "ryft.kernel.cuda", "ryft.ordered-io", "ryft.print"];
        assert_eq!(RuntimeExtensions::new().identifiers().collect::<Vec<_>>(), expected);
        assert_eq!(RuntimeExtensions::default().identifiers().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_runtime_extensions_empty() {
        assert_eq!(RuntimeExtensions::empty().identifiers().count(), 0);
    }

    #[test]
    fn test_runtime_extensions_with_provider() {
        let provider: Arc<dyn RuntimeProvider> = Arc::new(CountingProvider::new("test.provider"));
        let configuration = RuntimeExtensions::empty().with_provider(Arc::clone(&provider)).unwrap();
        assert!(Arc::ptr_eq(&configuration.providers["test.provider"], &provider));
        assert!(matches!(
            configuration.with_provider(provider),
            Err(Error::Runtime(RuntimeError::DuplicateProvider { identifier })) if identifier == "test.provider",
        ));
    }

    #[test]
    fn test_runtime_extensions_with_replaced_provider() {
        let original: Arc<dyn RuntimeProvider> = Arc::new(CountingProvider::new("test.provider"));
        let replacement: Arc<dyn RuntimeProvider> = Arc::new(CountingProvider::new("test.provider"));
        let configuration = RuntimeExtensions::empty().with_provider(Arc::clone(&original)).unwrap();
        let replaced = configuration.clone().with_replaced_provider(Arc::clone(&replacement)).unwrap();
        assert!(Arc::ptr_eq(&replaced.providers["test.provider"], &replacement));
        assert!(Arc::ptr_eq(&configuration.providers["test.provider"], &original));
        assert!(matches!(
            RuntimeExtensions::empty().with_replaced_provider(replacement),
            Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == "test.provider",
        ));
    }

    #[test]
    fn test_runtime_extensions_identifiers() {
        let configuration = RuntimeExtensions::empty()
            .with_provider(Arc::new(CountingProvider::new("test.b")))
            .unwrap()
            .with_provider(Arc::new(CountingProvider::new("test.a")))
            .unwrap();
        assert_eq!(configuration.identifiers().collect::<Vec<_>>(), vec!["test.a", "test.b"]);
    }

    #[test]
    fn test_runtime_extensions_debug() {
        let configuration =
            RuntimeExtensions::empty().with_provider(Arc::new(CountingProvider::new("test.provider"))).unwrap();
        assert_eq!(format!("{:?}", RuntimeExtensions::empty()), "RuntimeExtensions { identifiers: [] }");
        assert_eq!(format!("{configuration:?}"), r#"RuntimeExtensions { identifiers: ["test.provider"] }"#);
    }

    #[test]
    fn test_execution_resources_client() {
        let client = execution_client();
        let registry = RuntimeRegistry::new(client, RuntimeExtensions::empty());
        let resources = prepare_without_requirements(&registry);
        assert!(std::ptr::eq(resources.client(), client));
    }

    #[test]
    fn test_execution_resources_retain() {
        let registry = RuntimeRegistry::new(execution_client(), RuntimeExtensions::empty());
        let mut resources = prepare_without_requirements(&registry);
        let drops = Arc::new(AtomicUsize::new(0));
        resources.retain(DropCounter(Arc::clone(&drops)));
        let (context, retained) = resources.into_parts();
        assert!(context.is_none());
        assert!(retained.is_some());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(retained);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_execution_resources_add_ffi_user_data() {
        let client = execution_client();
        let registry = RuntimeRegistry::new(client, RuntimeExtensions::empty());
        let mut resources = prepare_without_requirements(&registry);
        let name = "ryft.test.execution_resources";
        let type_id = FfiTypeId::from_name(name);
        assert_eq!(client.register_ffi_type(name, type_id, FfiTypeInformation::new(None)).unwrap(), type_id);
        assert!(resources.context.is_none());

        let value = Arc::new(42i32);
        let pointer = Arc::as_ptr(&value).cast_mut().cast();
        resources.retain(value);
        // SAFETY: The retained Arc keeps the registered value alive at a stable address.
        unsafe { resources.add_ffi_user_data(FfiUserData::new(type_id, pointer)) }.unwrap();
        assert!(resources.context.is_some());
        // SAFETY: The same Arc still owns the value for this duplicate binding attempt.
        assert!(matches!(
            unsafe { resources.add_ffi_user_data(FfiUserData::new(type_id, pointer)) },
            Err(Error::Runtime(RuntimeError::DuplicateUserData { type_id: actual })) if actual == i64::from(type_id),
        ));
    }

    #[test]
    fn test_runtime_registry_ensure() {
        let client = execution_client();
        let provider = Arc::new(CountingProvider::new("test.provider"));
        let configuration = RuntimeExtensions::empty().with_provider(provider.clone()).unwrap();
        let first = RuntimeRegistry::new(client, configuration.clone());
        let second = RuntimeRegistry::new(client, configuration);
        let requirements = [builtin_requirement("test.provider")];
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        first.ensure(&requirements).unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        first.ensure(&requirements).unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        second.ensure(&requirements).unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_runtime_registry_ensure_retries_failed_initialization() {
        let client = execution_client();
        let provider = Arc::new(CountingProvider::new("test.provider"));
        provider.fail.store(true, Ordering::SeqCst);
        let configuration = RuntimeExtensions::empty().with_provider(provider.clone()).unwrap();
        let registry = RuntimeRegistry::new(client, configuration);
        let requirements = [builtin_requirement("test.provider")];
        let error = registry.ensure(&requirements).unwrap_err();
        assert_eq!(error.downcast_custom::<std::io::Error>().unwrap().to_string(), "fixture initialization failed",);
        registry.ensure(&requirements).unwrap();
        registry.ensure(&requirements).unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_runtime_registry_ensure_initializes_once_under_concurrent_calls() {
        /// Provider whose first initializer waits for the test to allow publication.
        struct BlockingProvider {
            /// Number of calls to the initializer.
            calls: AtomicUsize,

            /// Notifies the test that the first initializer has started.
            started: mpsc::Sender<()>,

            /// Blocks the first initializer until the competing calls are in progress.
            publish: Mutex<mpsc::Receiver<()>>,
        }

        impl RuntimeProvider for BlockingProvider {
            fn identifier(&self) -> &str {
                "test.concurrent"
            }

            fn initialize(
                &self,
                client: &'static Client<'static>,
                requirement: &RuntimeRequirement,
            ) -> Result<Arc<dyn RuntimeExtension>, Error> {
                let _ = (client, requirement);
                if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    self.started.send(()).unwrap();
                    self.publish.lock().unwrap().recv_timeout(Duration::from_secs(5)).unwrap();
                }
                Ok(Arc::new(RegistrationOnlyRuntime))
            }
        }

        let client = execution_client();
        let (started, initializing) = mpsc::channel();
        let (publish, permission) = mpsc::channel();
        let provider =
            Arc::new(BlockingProvider { calls: AtomicUsize::new(0), started, publish: Mutex::new(permission) });
        let configuration = RuntimeExtensions::empty().with_provider(provider.clone()).unwrap();
        let registry = Arc::new(RuntimeRegistry::new(client, configuration));
        let start = Arc::new(Barrier::new(2));
        let (finished, completed) = mpsc::channel();
        let spawn = || {
            let (registry, start, finished) = (registry.clone(), start.clone(), finished.clone());
            thread::spawn(move || {
                start.wait();
                finished.send(registry.ensure(&[builtin_requirement("test.concurrent")])).unwrap();
            })
        };
        let first = spawn();
        let second = spawn();
        initializing.recv_timeout(Duration::from_secs(5)).unwrap();
        let early = completed.recv_timeout(Duration::from_millis(100));
        publish.send(()).unwrap();
        assert!(matches!(early, Err(RecvTimeoutError::Timeout)));
        completed.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        completed.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_runtime_registry_ensure_retries_after_recursion_and_unwind() {
        /// Provider that first reenters the registry and then panics before succeeding.
        struct ReentrantProvider {
            /// Registry to reenter during the first initialization attempt.
            registry: OnceLock<Weak<RuntimeRegistry>>,

            /// Number of attempts, selecting the fixture's successive failure modes.
            calls: AtomicUsize,
        }

        impl RuntimeProvider for ReentrantProvider {
            fn identifier(&self) -> &str {
                "test.recursive"
            }

            fn initialize(
                &self,
                client: &'static Client<'static>,
                requirement: &RuntimeRequirement,
            ) -> Result<Arc<dyn RuntimeExtension>, Error> {
                let _ = client;
                match self.calls.fetch_add(1, Ordering::SeqCst) {
                    0 => self.registry.get().unwrap().upgrade().unwrap().ensure(std::slice::from_ref(requirement))?,
                    1 => panic!("fixture initializer unwound"),
                    _ => {}
                }
                Ok(Arc::new(RegistrationOnlyRuntime))
            }
        }

        let client = execution_client();
        let provider = Arc::new(ReentrantProvider { registry: OnceLock::new(), calls: AtomicUsize::new(0) });
        let configuration = RuntimeExtensions::empty().with_provider(provider.clone()).unwrap();
        let registry = Arc::new(RuntimeRegistry::new(client, configuration));
        provider.registry.set(Arc::downgrade(&registry)).unwrap();
        let requirements = [builtin_requirement("test.recursive")];
        let ensure = || registry.ensure(&requirements);
        // Both a returned error and a panic must release the initialization slot for another attempt.
        assert!(matches!(
            ensure(),
            Err(Error::Runtime(RuntimeError::RecursiveInitialization { identifier })) if identifier == "test.recursive",
        ));
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(ensure)).unwrap_err();
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"fixture initializer unwound"));
        ensure().unwrap();
        ensure().unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_runtime_registry_ensure_rejects_concurrent_initialization_cycles() {
        /// Two providers that attempt to initialize each other on separate threads.
        struct DependentProvider {
            /// This provider's public contract identifier.
            identifier: &'static str,

            /// The other provider requested during its first initialization.
            dependency: &'static str,

            /// Registry to reenter during the first initialization attempt.
            registry: OnceLock<Weak<RuntimeRegistry>>,

            /// Rendezvous ensuring both outer initializers have started before either reenters.
            started: Arc<std::sync::Barrier>,

            /// Dependencies are requested once so a rejected attempt can be retried independently.
            reenter: AtomicBool,
        }

        impl RuntimeProvider for DependentProvider {
            fn identifier(&self) -> &str {
                self.identifier
            }

            fn initialize(
                &self,
                client: &'static Client<'static>,
                requirement: &RuntimeRequirement,
            ) -> Result<Arc<dyn RuntimeExtension>, Error> {
                let _ = (client, requirement);
                if self.reenter.swap(false, Ordering::SeqCst) {
                    self.started.wait();
                    let registry = self.registry.get().unwrap().upgrade().unwrap();
                    registry.ensure(&[builtin_requirement(self.dependency)])?;
                }
                Ok(Arc::new(RegistrationOnlyRuntime))
            }
        }

        let client = execution_client();
        let started = Arc::new(Barrier::new(2));
        let left = Arc::new(DependentProvider {
            identifier: "test.left",
            dependency: "test.right",
            registry: OnceLock::new(),
            started: Arc::clone(&started),
            reenter: AtomicBool::new(true),
        });
        let right = Arc::new(DependentProvider {
            identifier: "test.right",
            dependency: "test.left",
            registry: OnceLock::new(),
            started,
            reenter: AtomicBool::new(true),
        });
        let configuration = RuntimeExtensions::empty()
            .with_provider(left.clone())
            .unwrap()
            .with_provider(right.clone())
            .unwrap();
        let registry = Arc::new(RuntimeRegistry::new(client, configuration));
        left.registry.set(Arc::downgrade(&registry)).unwrap();
        right.registry.set(Arc::downgrade(&registry)).unwrap();
        let (finished, completed) = mpsc::channel();
        let spawn = |identifier, dependency| {
            let (registry, finished) = (registry.clone(), finished.clone());
            thread::spawn(move || {
                let result = registry.ensure(&[builtin_requirement(identifier)]);
                finished.send((dependency, result)).unwrap();
            })
        };
        let left = spawn("test.left", "test.right");
        let right = spawn("test.right", "test.left");
        // One or both initializers can observe the cycle, depending on when the rejected attempt rolls back.
        let mut rejected = 0;
        for _ in 0..2 {
            let (dependency, result) = completed.recv_timeout(Duration::from_secs(5)).unwrap();
            if let Err(error) = result {
                assert!(matches!(
                    error,
                    Error::Runtime(RuntimeError::RecursiveInitialization { identifier }) if identifier == dependency,
                ));
                rejected += 1;
            }
        }
        left.join().unwrap();
        right.join().unwrap();
        assert!(rejected > 0);
        registry.ensure(&[builtin_requirement("test.left"), builtin_requirement("test.right")]).unwrap();
    }

    #[test]
    fn test_runtime_registry_ensure_rejects_missing_provider() {
        let client = execution_client();
        let registry = RuntimeRegistry::new(client, RuntimeExtensions::empty());
        assert!(matches!(
            registry.ensure(&[builtin_requirement("test.missing")]),
            Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == "test.missing",
        ));
    }

    #[test]
    fn test_runtime_registry_prepare() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let provider = Arc::new(CountingProvider::new("test.provider"));
        let configuration = RuntimeExtensions::empty().with_provider(provider.clone()).unwrap();
        let registry = RuntimeRegistry::new(client, configuration);
        let requirements = [builtin_requirement("test.provider")];
        let (context, retained) = registry.prepare(&requirements, &mesh).unwrap().into_parts();

        // A registration-only extension still initializes, but its invocation needs no context or retained resources.
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(context.is_none());
        assert!(retained.is_none());
    }

    #[test]
    fn test_runtime_registry_prepare_validates_invocation_mesh() {
        /// Provider for the runtime that checks the invocation's mesh.
        struct MeshCheckingProvider;

        impl RuntimeProvider for MeshCheckingProvider {
            fn identifier(&self) -> &str {
                "test.mesh"
            }

            fn initialize(
                &self,
                client: &'static Client<'static>,
                requirement: &RuntimeRequirement,
            ) -> Result<Arc<dyn RuntimeExtension>, Error> {
                let _ = requirement;
                Ok(Arc::new(MeshCheckingRuntime { client }))
            }
        }

        /// Extension that rejects the invocation mesh before preparing launch resources.
        struct MeshCheckingRuntime {
            /// Client that initialized the runtime.
            client: &'static Client<'static>,
        }

        impl RuntimeExtension for MeshCheckingRuntime {
            fn validate(&self, mesh: &DeviceMesh) -> Result<(), Error> {
                assert_eq!(mesh, &domain_mesh(self.client, "invocation", 1));
                Err(Error::custom(std::io::Error::other("fixture rejected the invocation mesh")))
            }

            fn prepare(&self, resources: &mut ExecutionResources) -> Result<(), Error> {
                let _ = resources;
                panic!("invocation preparation ran after mesh validation failed");
            }
        }

        let client = execution_client();
        let configuration = RuntimeExtensions::empty().with_provider(Arc::new(MeshCheckingProvider)).unwrap();
        let registry = RuntimeRegistry::new(client, configuration);
        let mesh = domain_mesh(client, "invocation", 1);
        assert!(matches!(
            registry.prepare(&[builtin_requirement("test.mesh")], &mesh),
            Err(error) if error.downcast_custom::<std::io::Error>()
                .is_some_and(|error| error.to_string() == "fixture rejected the invocation mesh"),
        ));
    }

    #[test]
    fn test_runtime_registry_retain() {
        let client = execution_client();
        let registry = RuntimeRegistry::new(client, RuntimeExtensions::empty());
        let drops = Arc::new(AtomicUsize::new(0));
        let mut resources = prepare_without_requirements(&registry);
        resources.retain(DropCounter(Arc::clone(&drops)));
        let (_, retained) = resources.into_parts();
        let (event, promise) = client.event(()).unwrap();
        let fence = ExecutionFence::new(vec![event]);
        registry.retain(&fence, retained);
        let pending_before_completion = registry.pending.lock().unwrap().len();
        let drops_before_completion = drops.load(Ordering::SeqCst);
        promise.set(None).unwrap();
        fence.block_until_ready().unwrap();
        assert_eq!(pending_before_completion, 1);
        assert_eq!(drops_before_completion, 0);

        // Retaining a new submission releases completed resources, including the new ones if its fence is ready.
        let mut resources = prepare_without_requirements(&registry);
        resources.retain(DropCounter(Arc::clone(&drops)));
        let (_, retained) = resources.into_parts();
        registry.retain(&ExecutionFence::new(vec![]), retained);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert!(registry.pending.lock().unwrap().is_empty());
    }

    #[test]
    fn test_runtime_registry_retain_tracks_context_without_additional_resources() {
        let client = execution_client();
        let registry = RuntimeRegistry::new(client, RuntimeExtensions::empty());
        let mut resources = prepare_without_requirements(&registry);
        resources.context = Some(client.execution_context().unwrap());
        let (context, retained) = resources.into_parts();
        assert!(context.is_some());

        let (event, promise) = client.event(()).unwrap();
        let fence = ExecutionFence::new(vec![event]);
        registry.retain(&fence, retained);
        let pending_before_completion = registry.pending.lock().unwrap().len();
        promise.set(None).unwrap();
        fence.block_until_ready().unwrap();
        assert_eq!(pending_before_completion, 1);
    }

    #[test]
    fn test_runtime_registry_retain_skips_submissions_without_resources() {
        let client = execution_client();
        let registry = RuntimeRegistry::new(client, RuntimeExtensions::empty());
        let (_, retained) = prepare_without_requirements(&registry).into_parts();
        let (event, promise) = client.event(()).unwrap();
        let fence = ExecutionFence::new(vec![event]);
        registry.retain(&fence, retained);
        let pending_before_completion = registry.pending.lock().unwrap().len();
        promise.set(None).unwrap();
        fence.block_until_ready().unwrap();
        assert_eq!(pending_before_completion, 0);
    }

    #[test]
    fn test_runtime_registry_drop_waits_for_all_events_after_fence_is_dropped() {
        let client = execution_client();
        let registry = RuntimeRegistry::new(client, RuntimeExtensions::empty());
        let drops = Arc::new(AtomicUsize::new(0));
        let mut resources = prepare_without_requirements(&registry);
        resources.retain(DropCounter(Arc::clone(&drops)));
        let (_, retained) = resources.into_parts();
        let (first, first_promise) = client.event(()).unwrap();
        let (second, second_promise) = client.event(()).unwrap();
        let fence = ExecutionFence::new(vec![first, second]);
        registry.retain(&fence, retained);
        drop(fence);
        first_promise.set(Some(ryft_pjrt::Error::aborted("test execution failed"))).unwrap();

        let (started, waiting) = mpsc::channel();
        let (released, completed) = mpsc::channel();
        let teardown = thread::spawn(move || {
            started.send(()).unwrap();
            drop(registry);
            released.send(()).unwrap();
        });
        let started = waiting.recv_timeout(Duration::from_secs(5));
        let early = completed.recv_timeout(Duration::from_millis(100));
        let drops_before_completion = drops.load(Ordering::SeqCst);
        // Release the last event even if teardown finished early, so a failing assertion cannot strand a waiter.
        second_promise.set(None).unwrap();

        started.unwrap();
        assert!(matches!(early, Err(RecvTimeoutError::Timeout)));
        assert_eq!(drops_before_completion, 0);
        completed.recv_timeout(Duration::from_secs(5)).unwrap();
        teardown.join().unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_builtin_requirement() {
        assert_eq!(builtin_requirement("test.builtin"), RuntimeRequirement::new("test.builtin", 1).unwrap());
    }

    #[test]
    fn test_validate_builtin_requirement() {
        validate_builtin_requirement(&builtin_requirement("test.builtin"), "test.builtin").unwrap();
        assert!(matches!(
            validate_builtin_requirement(&builtin_requirement("test.other"), "test.builtin"),
            Err(Error::Runtime(RuntimeError::InvalidRequirement { reason }))
                if reason == "expected runtime extension `test.builtin`",
        ));
        assert!(matches!(
            validate_builtin_requirement(&RuntimeRequirement::new("test.builtin", 2).unwrap(), "test.builtin"),
            Err(Error::Runtime(RuntimeError::UnsupportedVersion { identifier, version }))
                if identifier == "test.builtin" && version == 2,
        ));
    }
}
