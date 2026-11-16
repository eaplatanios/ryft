//! XLA sessions and the shared resources used to stage, compile, and execute array programs. A [`Session`] implements
//! [`ryft_core::Domain`] over [`ArrayIrType`](ryft_core::ArrayIrType), with [`Value`](crate::Value)s that hold device
//! arrays, checked dimensions, or mutable references. Values retain their session, so eager operations and transforms
//! recover the execution context from their inputs. [`Constant`](crate::Constant) and [`Operation`](crate::Operation)
//! provide the corresponding families of stored constants and operations for traced programs.
//!
//! # Sessions
//!
//! A [`Session`] executes on one PJRT [`Client`](ryft_pjrt::Client) and is a cheap handle to the resources that it
//! creates for that client, which are its compilation cache, runtime extension resources, and default compilation
//! options. [`Session::new`] creates a session on a client that it creates for the best available [`Platform`], and
//! [`Session::builder`] allows configuring everything in detail. Clone a session wherever another handle is needed.
//! Clones share its resources, while creating another session gives an independent cache and effect scope, even on
//! the same client. Configure built-in or downstream providers through [`RuntimeExtensions`] and
//! [`SessionBuilder::runtime_extensions`]. Each session initializes its own runtime instances
//! lazily when an executable requires them.
//!
//! [`Options`](crate::Options) supplies the execution [`DeviceMesh`](ryft_core::DeviceMesh), sharding
//! overrides, donation flags, and backend compilation policies for a staged function. The session's default
//! [`CompilationOptions`](ryft_pjrt::protos::CompilationOptions) supply the underlying PJRT compiler settings.
//! [`Target`] describes the client's platform and devices without retaining live device handles.
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
//! and output container structures. The session's [`CompilationContext`](ryft_core::CompilationContext) coordinates
//! artifact reuse, and [`CompilationKey`](crate::CompilationKey) records the exact computation and compilation
//! configuration. [`CompilationAnalysis`](crate::CompilationAnalysis) and [`OptimizedProgram`](crate::OptimizedProgram)
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

pub mod eager;
pub mod effects;
pub mod platforms;
pub mod runtimes;
pub mod sessions;
pub mod targets;

pub use platforms::Platform;
pub use runtimes::{
    ExecutionResources, MAX_RUNTIME_REQUIREMENTS_BYTES, RUNTIME_REQUIREMENTS_ATTRIBUTE, RuntimeError, RuntimeExtension,
    RuntimeExtensions, RuntimeProvider, RuntimeRequirement, named_ffi_type_id, normalize_runtime_requirements,
    runtime_requirements, with_runtime_requirements,
};
pub use sessions::{Session, SessionBuilder, SessionConstant, SessionError, ThroughSession, Tracer};
pub use targets::{Target, TargetDevice};

#[cfg(test)]
pub(crate) mod tests {
    use ryft_core::{
        ArrayIrType, ArrayType, DataType, Device, DeviceMesh, Dimension as CoreDimension, LogicalMesh, MeshAxis,
        MeshAxisType, Placeholder, ProjectedContext, ReferenceReadOperation, ReferenceType, Shape, Sharding,
    };
    use ryft_pjrt::Client;

    use crate::domains::sessions::Session;
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{Array, Constant, FlatProgram, FromPjrt, ProgramBuilder, Value};

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
}
