//! Exercises third-party runtime providers through public staging, compilation, execution, and persistence APIs.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pretty_assertions::assert_eq;
use ryft_core::{
    ArrayType, CompilationCacheDomain, CompilationDomain, CustomCall, CustomCallOperation, DataType, Device,
    DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, StagedFunction as CoreStagedFunction,
};
use ryft_pjrt::extensions::ffi::{
    FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits, FfiInput, FfiOutput,
    FfiTypeId, FfiTypeInformation, FfiUserData, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Handler,
};
use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};
use ryft_xla::{
    Array, CustomCallOperationExtension, Error, ExecutionResources, FromPjrt, Options, RuntimeError, RuntimeExtension,
    RuntimeExtensions, RuntimeProvider, RuntimeRequirement, Session, Type, Value,
};

/// Native target that consumes both independently installed states.
const TARGET: &str = "ryft.test.runtime_extensions.add_states";

/// First runtime requirement and its native state type name.
const LEFT: &str = "ryft.test.runtime_extensions.left";

/// Second runtime requirement and its native state type name.
const RIGHT: &str = "ryft.test.runtime_extensions.right";

/// Observable lifecycle events shared by a provider configuration and its client-associated instances.
#[derive(Default)]
struct Statistics {
    /// Successful client-associated instances created from this provider configuration.
    initialized: AtomicUsize,

    /// Invocations to which this provider contributed state.
    prepared: AtomicUsize,

    /// Native callbacks that observed this provider's state.
    called: AtomicUsize,
}

/// Configures one independently initialized state contribution to a custom call.
struct StateProvider {
    /// Public requirement identifier resolved by the session.
    identifier: &'static str,

    /// Native type under which this instance contributes its state.
    type_name: &'static str,

    /// Distinguishes the contributions of the two independent providers.
    weight: i32,

    /// Lifecycle observations shared across sessions constructed from cloned configuration.
    statistics: Arc<Statistics>,
}

/// Stable session-owned state borrowed by the native callback for one invocation.
struct State {
    /// Explicit native identifier matching this state representation.
    type_id: FfiTypeId,

    /// Per-session contribution that exposes accidental sharing of initialized instances.
    value: i32,

    /// Lifecycle observations retained for the full native invocation.
    statistics: Arc<Statistics>,
}

impl RuntimeProvider for StateProvider {
    fn identifier(&self) -> &str {
        self.identifier
    }

    fn initialize(
        &self,
        client: &'static Client<'static>,
        requirement: &RuntimeRequirement,
    ) -> Result<Arc<dyn RuntimeExtension>, Error> {
        assert!(client.platform_name()?.eq_ignore_ascii_case("cpu"));
        assert_eq!(requirement, &RuntimeRequirement::new(self.identifier, 1).unwrap());
        // Both named types are part of this handler's ABI. Register them before installing the callback, while
        // keeping their live state in independently initialized providers and session-owned allocations.
        // Repeating these registrations for every session is idempotent because the IDs are explicit.
        for name in [LEFT, RIGHT] {
            client.register_ffi_type(name, FfiTypeId::from_name(name), FfiTypeInformation::new(None))?;
        }
        client.register_ffi_handler(
            TARGET,
            "cpu",
            FfiHandler::from(add_states_handler as XLA_FFI_Handler),
            FfiHandlerTraits::NONE,
        )?;
        let generation = self.statistics.initialized.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(Arc::new(State {
            type_id: FfiTypeId::from_name(self.type_name),
            value: self.weight * generation as i32,
            statistics: Arc::clone(&self.statistics),
        }))
    }
}

impl RuntimeExtension for State {
    fn prepare(&self, resources: &mut ExecutionResources) -> Result<(), Error> {
        self.statistics.prepared.fetch_add(1, Ordering::SeqCst);
        // The session retains this stable runtime allocation until the whole execution completes, including when
        // public outputs are dropped. The named native type belongs to this `State` representation.
        unsafe { resources.add_ffi_user_data(FfiUserData::new(self.type_id, (self as *const Self).cast_mut().cast())) }
    }
}

/// Reads both independently installed states from the same native invocation context.
fn add_states(frame: &FfiCallFrame<'_>) -> Result<(), FfiError> {
    if frame.input_count() != 1 || frame.output_count() != 1 || frame.attributes().next().is_some() {
        return Err(FfiError::invalid_argument("expected one input, one output, and no native attributes"));
    }
    let context = frame.context()?;
    let mut increment = 0;
    for name in [LEFT, RIGHT] {
        let data = context.user_data(FfiTypeId::from_name(name))?.data;
        if data.is_null() {
            return Err(FfiError::invalid_argument("missing runtime state"));
        }
        // `State::prepare` is the sole producer, and its session retains the `State` allocation until completion.
        let state = unsafe { &*data.cast::<State>() };
        state.statistics.called.fetch_add(1, Ordering::SeqCst);
        increment += state.value;
    }
    let FfiInput::Buffer { buffer: input } = frame.input(0)?;
    let FfiOutput::Buffer { buffer: output } = frame.output(0)?;
    if input.element_type() != FfiBufferType::I32
        || output.element_type() != FfiBufferType::I32
        || !input.dimensions().is_empty()
        || !output.dimensions().is_empty()
    {
        return Err(FfiError::invalid_argument("expected scalar `i32` buffers"));
    }
    // The CPU handler receives one live, aligned `i32` in each buffer, and XLA owns readiness and exclusive output
    // access.
    unsafe {
        let input = input.data().cast::<i32>();
        let output = output.data().cast::<i32>();
        if input.is_null() || output.is_null() {
            return Err(FfiError::invalid_argument("missing scalar buffer"));
        }
        *output = *input + increment;
    }
    Ok(())
}

/// Native callback entry point, including metadata discovery during handler registration.
unsafe extern "C" fn add_states_handler(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    let Ok(frame) = (unsafe { FfiCallFrame::from_c_api(frame) }) else {
        return std::ptr::null_mut();
    };
    if frame.register_metadata(FfiTypeId::UNKNOWN) || frame.stage() != FfiExecutionStage::Execution {
        return std::ptr::null_mut();
    }
    match add_states(&frame) {
        Ok(()) => std::ptr::null_mut(),
        Err(error) => match frame.api() {
            Ok(api) => unsafe { error.to_c_api(api) },
            Err(_) => std::ptr::null_mut(),
        },
    }
}

/// Creates the CPU client and one-device mesh shared by a test's isolated sessions.
fn client_and_mesh() -> (&'static Client<'static>, DeviceMesh) {
    let client = load_cpu_plugin()
        .unwrap()
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let client: &'static Client<'static> = Box::leak(Box::new(client));
    let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
    let mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![device],
    )
    .unwrap();
    (client, mesh)
}

/// Creates separate providers, optionally violating the distinct native type binding contract.
fn providers(left: &Arc<Statistics>, right: &Arc<Statistics>, duplicate_binding: bool) -> RuntimeExtensions {
    RuntimeExtensions::empty()
        .with_provider(Arc::new(StateProvider {
            identifier: LEFT,
            type_name: LEFT,
            weight: 3,
            statistics: Arc::clone(left),
        }))
        .unwrap()
        .with_provider(Arc::new(StateProvider {
            identifier: RIGHT,
            type_name: if duplicate_binding { LEFT } else { RIGHT },
            weight: 20,
            statistics: Arc::clone(right),
        }))
        .unwrap()
}

/// Stages a source custom call using only the downstream public compilation API.
fn staged(mesh: DeviceMesh, requirements: Vec<RuntimeRequirement>) -> CoreStagedFunction<Session, Type, Type> {
    let r#type = ArrayType::scalar(DataType::I32);
    let operation = CustomCallOperation::new(TARGET, vec![r#type.clone()])
        .with_runtime_requirements(requirements)
        .unwrap();
    CoreStagedFunction::stage(
        move |_, _, input| Ok(CustomCall::custom_call(&operation, [&input])?.remove(0)),
        Vec::new(),
        Type::Array(r#type),
        Options::new(mesh),
    )
    .unwrap()
}

/// Returns the canonical source declarations required by the combined callback.
fn requirements() -> Vec<RuntimeRequirement> {
    vec![RuntimeRequirement::new(LEFT, 1).unwrap(), RuntimeRequirement::new(RIGHT, 1).unwrap()]
}

/// Waits for the concrete scalar output, observing callback completion before any lifecycle assertions.
fn scalar(value: &Value) -> i32 {
    let Value::Array(array) = value else { panic!("expected an array output") };
    let bytes = array.shards()[0].buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
    i32::from_ne_bytes(bytes.as_slice().try_into().unwrap())
}

#[test]
fn test_runtime_extensions_compose_execute_and_reload() {
    let (client, mesh) = client_and_mesh();
    let left = Arc::new(Statistics::default());
    let right = Arc::new(Statistics::default());
    let configuration = providers(&left, &right, false);
    let first = Session::builder().client(client).runtime_extensions(configuration.clone()).build().unwrap();
    let lowered = staged(mesh.clone(), requirements()).lower(&first).unwrap();
    assert_eq!(lowered.lowered_program().runtime_requirements(), requirements());
    assert_eq!(left.initialized.load(Ordering::SeqCst), 0);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 0);
    let compiled = lowered.compile(&first).unwrap();
    assert_eq!(compiled.compiled_program().runtime_requirements(), requirements());
    assert_eq!(left.initialized.load(Ordering::SeqCst), 1);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 1);

    let input =
        Array::from_host_buffer(&first, ArrayType::scalar(DataType::I32), None, None, 7i32.to_ne_bytes()).unwrap();
    let output = compiled.executable().execute(&first, Value::Array(input.clone())).unwrap();
    assert_eq!(scalar(&output), 30);
    let output = compiled.executable().execute(&first, Value::Array(input)).unwrap();
    assert_eq!(scalar(&output), 30);
    assert_eq!(left.initialized.load(Ordering::SeqCst), 1);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 1);
    assert_eq!(left.prepared.load(Ordering::SeqCst), 2);
    assert_eq!(right.prepared.load(Ordering::SeqCst), 2);
    assert_eq!(left.called.load(Ordering::SeqCst), 2);
    assert_eq!(right.called.load(Ordering::SeqCst), 2);

    let bytes = first.serialize_program(compiled.compiled_program()).unwrap().unwrap();
    let second = Session::builder().client(client).runtime_extensions(configuration).build().unwrap();
    assert_eq!(left.initialized.load(Ordering::SeqCst), 1);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 1);
    let restored = second.deserialize_program(&bytes).unwrap().unwrap();
    assert_eq!(restored.runtime_requirements(), requirements());
    // Loading resolves the receiving session's providers before its first invocation.
    assert_eq!(left.initialized.load(Ordering::SeqCst), 2);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 2);
    let input =
        Array::from_host_buffer(&second, ArrayType::scalar(DataType::I32), None, None, 7i32.to_ne_bytes()).unwrap();
    let output = second.execute(&restored, vec![Value::Array(input)]).unwrap().remove(0);
    assert_eq!(scalar(&output), 53);
    assert_eq!(left.called.load(Ordering::SeqCst), 3);
    assert_eq!(right.called.load(Ordering::SeqCst), 3);

    let missing = Session::builder().client(client).runtime_extensions(RuntimeExtensions::empty()).build().unwrap();
    assert!(matches!(missing.deserialize_program(&bytes),
        Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == LEFT));
}

#[test]
fn test_runtime_extensions_reject_missing_provider_before_compilation_and_cached_submission() {
    let (client, mesh) = client_and_mesh();
    let left = Arc::new(Statistics::default());
    let right = Arc::new(Statistics::default());
    let first = Session::builder()
        .client(client)
        .runtime_extensions(providers(&left, &right, false))
        .build()
        .unwrap();
    let lowered = staged(mesh.clone(), requirements()).lower(&first).unwrap();
    let missing = Session::builder().client(client).runtime_extensions(RuntimeExtensions::empty()).build().unwrap();
    assert!(matches!(lowered.compile(&missing),
        Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == LEFT));
    assert_eq!(left.initialized.load(Ordering::SeqCst), 0);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 0);
    let compiled = lowered.compile(&first).unwrap();
    let input =
        Array::from_host_buffer(&missing, ArrayType::scalar(DataType::I32), None, None, 7i32.to_ne_bytes()).unwrap();
    assert!(matches!(missing.execute(compiled.compiled_program(), vec![Value::Array(input)]),
        Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == LEFT));
    assert_eq!(left.called.load(Ordering::SeqCst), 0);
    assert_eq!(right.called.load(Ordering::SeqCst), 0);
}

#[test]
fn test_runtime_extensions_reject_duplicate_native_binding_before_submission() {
    let (client, mesh) = client_and_mesh();
    let left = Arc::new(Statistics::default());
    let right = Arc::new(Statistics::default());
    let session = Session::builder()
        .client(client)
        .runtime_extensions(providers(&left, &right, true))
        .build()
        .unwrap();
    let compiled = staged(mesh.clone(), requirements()).lower(&session).unwrap().compile(&session).unwrap();
    let input =
        Array::from_host_buffer(&session, ArrayType::scalar(DataType::I32), None, None, 7i32.to_ne_bytes()).unwrap();
    let type_id = FfiTypeId::from_name(LEFT);
    assert!(matches!(compiled.executable().execute(&session, Value::Array(input)),
        Err(Error::Runtime(RuntimeError::DuplicateUserData { type_id: actual })) if actual == i64::from(type_id)));
    assert_eq!(left.called.load(Ordering::SeqCst), 0);
    assert_eq!(right.called.load(Ordering::SeqCst), 0);
}

#[test]
fn test_runtime_extensions_preserve_custom_initializer_error() {
    #[derive(Debug, PartialEq, thiserror::Error)]
    #[error("downstream provider is unavailable")]
    struct ProviderFailure;

    struct FailingProvider;

    impl RuntimeProvider for FailingProvider {
        fn identifier(&self) -> &str {
            "ryft.test.runtime_extensions.failure"
        }

        fn initialize(
            &self,
            client: &'static Client<'static>,
            requirement: &RuntimeRequirement,
        ) -> Result<Arc<dyn RuntimeExtension>, Error> {
            let _ = (client, requirement);
            Err(Error::custom(ProviderFailure))
        }
    }

    let (client, mesh) = client_and_mesh();
    let configuration = RuntimeExtensions::empty().with_provider(Arc::new(FailingProvider)).unwrap();
    let session = Session::builder().client(client).runtime_extensions(configuration).build().unwrap();
    let requirement = RuntimeRequirement::new("ryft.test.runtime_extensions.failure", 1).unwrap();
    let error = staged(mesh, vec![requirement]).lower(&session).unwrap().compile(&session).err().unwrap();
    assert_eq!(error.downcast_custom::<ProviderFailure>(), Some(&ProviderFailure));
    assert_eq!(error.to_string(), "downstream provider is unavailable");
}
