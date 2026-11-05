//! Exercises third-party runtime providers through public staging, compilation, execution, and persistence APIs.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pretty_assertions::assert_eq;
use ryft_core::{
    ArrayIrType, ArrayType, CompilationCacheDomain, CompilationContext, CompilationDomain, CustomCall,
    CustomCallOperation, DataType, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType,
    StagedFunction as CoreStagedFunction,
};
use ryft_pjrt::extensions::ffi::{
    FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits, FfiInput, FfiOutput,
    FfiTypeId, FfiTypeInformation, FfiUserData, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Handler,
};
use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};
use ryft_xla::domains::{
    ExecutionResources, RuntimeExtension, RuntimeExtensions, RuntimeProvider, RuntimeRequirement, named_ffi_type_id,
    with_runtime_requirements,
};
use ryft_xla::{Domain, Error, FromPjrt, Options, RuntimeError, Session, Target, Value};

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

impl<'c> RuntimeProvider<'c> for StateProvider {
    fn identifier(&self) -> &str {
        self.identifier
    }

    fn initialize(
        &self,
        client: &'c Client<'c>,
        target: &Target,
        requirement: &RuntimeRequirement,
    ) -> Result<Arc<dyn RuntimeExtension<'c> + 'c>, Error> {
        assert!(target.platform_name().eq_ignore_ascii_case("cpu"));
        assert_eq!(requirement, &RuntimeRequirement::new(self.identifier, 1, Vec::new()).unwrap());
        // Both named types are part of this handler's ABI. Register them before installing the callback, while
        // keeping their live state in independently initialized providers and session-owned allocations.
        // Repeating these registrations for every session is idempotent because the IDs are explicit.
        for name in [LEFT, RIGHT] {
            client.register_ffi_type(name, named_ffi_type_id(name), FfiTypeInformation::new(None))?;
        }
        client.register_ffi_handler(
            TARGET,
            "cpu",
            FfiHandler::from(add_states_handler as XLA_FFI_Handler),
            FfiHandlerTraits::NONE,
        )?;
        let generation = self.statistics.initialized.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(Arc::new(State {
            type_id: named_ffi_type_id(self.type_name),
            value: self.weight * generation as i32,
            statistics: Arc::clone(&self.statistics),
        }))
    }
}

impl<'c> RuntimeExtension<'c> for State {
    fn prepare(&self, resources: &mut ExecutionResources<'c>) -> Result<(), Error> {
        self.statistics.prepared.fetch_add(1, Ordering::SeqCst);
        // The session retains this stable runtime allocation through whole-execution completion, including when
        // public outputs are dropped. The named native type belongs to this State representation.
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
        let data = context.user_data(named_ffi_type_id(name))?.data;
        if data.is_null() {
            return Err(FfiError::invalid_argument("missing runtime state"));
        }
        // State::prepare is the sole producer; its session retains the State allocation until completion.
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
        return Err(FfiError::invalid_argument("expected scalar i32 buffers"));
    }
    // The CPU handler receives one live, aligned i32 in each buffer; XLA owns readiness and exclusive output access.
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
fn client_and_mesh() -> (Client<'static>, DeviceMesh) {
    let client = load_cpu_plugin()
        .unwrap()
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
    let mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![device],
    )
    .unwrap();
    (client, mesh)
}

/// Creates separate providers, optionally violating the distinct native type binding contract.
fn providers<'c>(left: &Arc<Statistics>, right: &Arc<Statistics>, duplicate_binding: bool) -> RuntimeExtensions<'c> {
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
fn staged<'c>(
    mesh: DeviceMesh,
    requirements: Vec<RuntimeRequirement>,
) -> CoreStagedFunction<Domain<'c>, ArrayIrType, ArrayIrType> {
    let r#type = ArrayType::scalar(DataType::I32);
    let operation =
        with_runtime_requirements(CustomCallOperation::new(TARGET, vec![r#type.clone()]), requirements).unwrap();
    CoreStagedFunction::stage(
        move |_, _, input| Ok(CustomCall::custom_call(&operation, [&input])?.remove(0)),
        Vec::new(),
        ArrayIrType::Array(r#type),
        Options::new(mesh),
    )
    .unwrap()
}

/// Returns the canonical source declarations required by the combined callback.
fn requirements() -> Vec<RuntimeRequirement> {
    vec![RuntimeRequirement::new(LEFT, 1, Vec::new()).unwrap(), RuntimeRequirement::new(RIGHT, 1, Vec::new()).unwrap()]
}

/// Waits for the concrete scalar output, observing callback completion before any lifecycle assertions.
fn scalar(value: &Value<'_>) -> i32 {
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
    let first = Session::with_runtime_extensions(&client, CompilationContext::new(), configuration.clone());
    let domain = first.domain();
    let lowered = staged(mesh.clone(), requirements()).lower(&domain).unwrap();
    assert_eq!(lowered.lowered_program().runtime_requirements(), requirements());
    assert_eq!(left.initialized.load(Ordering::SeqCst), 0);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 0);
    let compiled = lowered.compile(&domain).unwrap();
    assert_eq!(compiled.compiled_program().runtime_requirements(), requirements());
    assert_eq!(left.initialized.load(Ordering::SeqCst), 1);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 1);

    let input = first.array(ArrayType::scalar(DataType::I32), mesh.clone(), 7i32.to_ne_bytes()).unwrap();
    let output = compiled.executable().execute(&domain, Value::Array(input.clone())).unwrap();
    assert_eq!(scalar(&output), 30);
    let output = compiled.executable().execute(&domain, Value::Array(input)).unwrap();
    assert_eq!(scalar(&output), 30);
    assert_eq!(left.initialized.load(Ordering::SeqCst), 1);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 1);
    assert_eq!(left.prepared.load(Ordering::SeqCst), 2);
    assert_eq!(right.prepared.load(Ordering::SeqCst), 2);
    assert_eq!(left.called.load(Ordering::SeqCst), 2);
    assert_eq!(right.called.load(Ordering::SeqCst), 2);

    let bytes = domain
        .serialize_program(compiled.compiled_program())
        .unwrap()
        .expect("the CPU plugin must support serialization of this typed FFI executable");
    let second = Session::with_runtime_extensions(&client, CompilationContext::new(), configuration);
    let restored_domain = second.domain();
    assert_eq!(left.initialized.load(Ordering::SeqCst), 1);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 1);
    let restored = restored_domain
        .deserialize_program(&bytes)
        .unwrap()
        .expect("the serialized executable must be compatible");
    assert_eq!(restored.runtime_requirements(), requirements());
    // Loading resolves the receiving session's providers before its first invocation.
    assert_eq!(left.initialized.load(Ordering::SeqCst), 2);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 2);
    let input = second.array(ArrayType::scalar(DataType::I32), mesh, 7i32.to_ne_bytes()).unwrap();
    let output = restored_domain.execute(&restored, vec![Value::Array(input)]).unwrap().remove(0);
    assert_eq!(scalar(&output), 53);
    assert_eq!(left.called.load(Ordering::SeqCst), 3);
    assert_eq!(right.called.load(Ordering::SeqCst), 3);

    let missing = Session::with_runtime_extensions(&client, CompilationContext::new(), RuntimeExtensions::empty());
    assert!(matches!(missing.domain().deserialize_program(&bytes),
        Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == LEFT));
}

#[test]
fn test_runtime_extensions_reject_missing_provider_before_compilation_and_cached_submission() {
    let (client, mesh) = client_and_mesh();
    let left = Arc::new(Statistics::default());
    let right = Arc::new(Statistics::default());
    let first = Session::with_runtime_extensions(&client, CompilationContext::new(), providers(&left, &right, false));
    let domain = first.domain();
    let lowered = staged(mesh.clone(), requirements()).lower(&domain).unwrap();
    let missing = Session::with_runtime_extensions(&client, CompilationContext::new(), RuntimeExtensions::empty());
    let missing_domain = missing.domain();
    assert!(matches!(lowered.compile(&missing_domain),
        Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == LEFT));
    assert_eq!(left.initialized.load(Ordering::SeqCst), 0);
    assert_eq!(right.initialized.load(Ordering::SeqCst), 0);
    let compiled = lowered.compile(&domain).unwrap();
    let input = missing.array(ArrayType::scalar(DataType::I32), mesh, 7i32.to_ne_bytes()).unwrap();
    assert!(matches!(missing_domain.execute(compiled.compiled_program(), vec![Value::Array(input)]),
        Err(Error::Runtime(RuntimeError::MissingProvider { identifier })) if identifier == LEFT));
    assert_eq!(left.called.load(Ordering::SeqCst), 0);
    assert_eq!(right.called.load(Ordering::SeqCst), 0);
}

#[test]
fn test_runtime_extensions_reject_duplicate_native_binding_before_submission() {
    let (client, mesh) = client_and_mesh();
    let left = Arc::new(Statistics::default());
    let right = Arc::new(Statistics::default());
    let session = Session::with_runtime_extensions(&client, CompilationContext::new(), providers(&left, &right, true));
    let domain = session.domain();
    let compiled = staged(mesh.clone(), requirements()).lower(&domain).unwrap().compile(&domain).unwrap();
    let input = session.array(ArrayType::scalar(DataType::I32), mesh, 7i32.to_ne_bytes()).unwrap();
    let type_id = named_ffi_type_id(LEFT);
    assert!(matches!(compiled.executable().execute(&domain, Value::Array(input)),
        Err(Error::Runtime(RuntimeError::DuplicateUserData { type_id: actual })) if actual == i64::from(type_id)));
    assert_eq!(left.called.load(Ordering::SeqCst), 0);
    assert_eq!(right.called.load(Ordering::SeqCst), 0);
}

#[test]
fn test_runtime_extensions_preserve_custom_initializer_error() {
    #[derive(Debug, thiserror::Error)]
    #[error("downstream provider is unavailable")]
    struct ProviderFailure;

    struct FailingProvider;

    impl<'c> RuntimeProvider<'c> for FailingProvider {
        fn identifier(&self) -> &str {
            "ryft.test.runtime_extensions.failure"
        }

        fn initialize(
            &self,
            _client: &'c Client<'c>,
            _target: &Target,
            _requirement: &RuntimeRequirement,
        ) -> Result<Arc<dyn RuntimeExtension<'c> + 'c>, Error> {
            Err(Error::custom(ProviderFailure))
        }
    }

    let (client, mesh) = client_and_mesh();
    let configuration = RuntimeExtensions::empty().with_provider(Arc::new(FailingProvider)).unwrap();
    let session = Session::with_runtime_extensions(&client, CompilationContext::new(), configuration);
    let domain = session.domain();
    let requirement = RuntimeRequirement::new("ryft.test.runtime_extensions.failure", 1, Vec::new()).unwrap();
    let error = staged(mesh, vec![requirement]).lower(&domain).unwrap().compile(&domain).err().unwrap();
    assert!(matches!(error, Error::Custom(_)));
    assert!(error.downcast_custom::<ProviderFailure>().is_some());
    assert_eq!(error.to_string(), "downstream provider is unavailable");
}
