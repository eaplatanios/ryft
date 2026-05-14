use std::marker::PhantomData;
use std::sync::LazyLock;

use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Buffer, Client, LoadedExecutable, Program};

use ryft_core::operations::constants::{ONE_OPERATION_NAME, ZERO_OPERATION_NAME};
use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::sharding::{DeviceMesh, Sharding};
use ryft_core::tracing::TracingError;
use ryft_core::tracing::domains::{Domain, RuntimeDomain, TracingDomain};
use ryft_core::tracing_v2::LinearizableDomain;
use ryft_core::types::{ArrayType, DataType, TypeError};

use super::ops::{LinearXlaOperation, XlaOperation};
use super::shard_map::{ShardMapTensor, ShardMapTraceError, TracedXlaProgram};
use crate::arrays_v0::{Array, ArrayError};

#[cfg(test)]
use crate::arrays_v0::{ShardDescriptor, ShardLayout, device_put_element_size_in_bytes, dynamic_array_shape_error};
#[cfg(test)]
use crate::pjrt::ToPjrt;
#[cfg(test)]
use ryft_core::sharding::DeviceId;
#[cfg(test)]
use ryft_core::types::StaticShape;

/// Error type returned by [`XlaDomain`] orchestration helpers.
#[derive(Debug, thiserror::Error)]
pub enum XlaDomainError {
    /// Error surfaced while lowering a traced XLA program to StableHLO/Shardy MLIR.
    #[error("{0}")]
    Lowering(#[from] ShardMapTraceError),

    /// Error surfaced while materializing or marshalling [`Array`] values.
    #[error("{0}")]
    Array(#[from] ArrayError),

    /// Error surfaced by the underlying PJRT runtime.
    #[error("{0}")]
    Pjrt(#[from] ryft_pjrt::Error),
}

/// Stateful backend that materializes, lowers, compiles, and executes traced XLA programs against a live PJRT
/// [`Client`].
///
/// [`XlaDomain`] holds three pieces of context:
///
/// - a PJRT [`Client`] used to upload `zero`/`one` shards and to compile and execute programs,
/// - a concrete [`DeviceMesh`] used to resolve shard placement for arrays synthesized from
///   [`ArrayType`] metadata, and
/// - default [`CompilationOptions`] that [`XlaDomain::compile`] forwards to PJRT.
///
/// The same domain token covers both staged tracing and concrete execution. Nested traced code can
/// switch to [`XlaDomain::token`] instead of maintaining a separate tracing-only backend token.
///
/// Holding the mesh on the domain keeps [`RuntimeDomain::zero`] and [`RuntimeDomain::one`] well-defined: both
/// methods can rebuild a replicated fallback sharding from `self.mesh.logical_mesh` when the
/// supplied [`ArrayType`] omits one. The trait contract requires [`ArrayType::shape`] to be
/// fully static on the types passed to `zero` / `one`; dynamic shapes return an error.
pub struct XlaDomain<'c> {
    /// PJRT client used by this domain.
    client: Option<&'c Client<'c>>,

    /// Concrete device mesh used when an [`ArrayType`] does not specify a sharding.
    mesh: Option<DeviceMesh>,

    /// Default compilation options forwarded to [`Client::compile`].
    compilation_options: CompilationOptions,

    /// Phantom marker tying the domain lifetime to the concrete PJRT-backed array value type.
    marker: PhantomData<fn() -> Array<'c>>,
}

impl<'c> XlaDomain<'c> {
    /// Creates a new [`XlaDomain`] with default [`CompilationOptions`].
    #[inline]
    pub fn new(client: &'c Client<'c>, mesh: DeviceMesh) -> Self {
        Self::with_compilation_options(client, mesh, CompilationOptions::default())
    }

    /// Creates a new [`XlaDomain`] with explicit [`CompilationOptions`].
    #[inline]
    pub fn with_compilation_options(
        client: &'c Client<'c>,
        mesh: DeviceMesh,
        compilation_options: CompilationOptions,
    ) -> Self {
        Self { client: Some(client), mesh: Some(mesh), compilation_options, marker: PhantomData }
    }

    /// Returns the singleton tracing-only domain token that carries the XLA staged operation
    /// universe but no PJRT execution context.
    ///
    /// This token is sufficient for nested transforms over already-traced XLA values because those
    /// paths only need the backend's operation carriers; they never materialize concrete arrays via
    /// [`RuntimeDomain::zero`] or [`RuntimeDomain::one`].
    #[inline]
    pub fn token() -> &'static Self {
        static TOKEN: LazyLock<XlaDomain<'static>> = LazyLock::new(|| XlaDomain {
            client: None,
            mesh: None,
            compilation_options: CompilationOptions::default(),
            marker: PhantomData,
        });
        &TOKEN
    }

    /// Returns the PJRT [`Client`] this domain was constructed with.
    #[inline]
    pub fn client(&self) -> &'c Client<'c> {
        self.client.expect("execution XlaDomain should always carry a client")
    }

    /// Returns the concrete [`DeviceMesh`] this domain resolves shard placement against.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        self.mesh.as_ref().expect("execution XlaDomain should always carry a device mesh")
    }

    /// Returns the [`CompilationOptions`] that [`XlaDomain::compile`] forwards to PJRT.
    #[inline]
    pub fn compilation_options(&self) -> &CompilationOptions {
        &self.compilation_options
    }
}

impl<'c> Domain for XlaDomain<'c> {
    type Type = ArrayType;
    type Value = ShardMapTensor;
}

impl<'c> RuntimeDomain for XlaDomain<'c> {
    fn zero(&self, array_type: &ArrayType) -> Result<ShardMapTensor, TracingError> {
        validate_identity_synthesis(ZERO_OPERATION_NAME, array_type)?;
        Ok(ShardMapTensor::zero(array_type.clone()))
    }

    fn one(&self, array_type: &ArrayType) -> Result<ShardMapTensor, TracingError> {
        validate_identity_synthesis(ONE_OPERATION_NAME, array_type)?;
        Ok(ShardMapTensor::one(array_type.clone()))
    }
}

impl<'c> TracingDomain for XlaDomain<'c> {
    type OperationCarrier = XlaOperation;
}

/// Stateless linear [`TracingDomain`] for XLA tangent and cotangent programs over abstract tensor leaves.
#[derive(Copy, Clone, Debug, Default)]
pub struct LinearXlaDomain;

impl LinearXlaDomain {
    /// Returns the singleton linear XLA domain.
    #[inline]
    pub fn token() -> &'static Self {
        static TOKEN: LinearXlaDomain = LinearXlaDomain;
        &TOKEN
    }
}

impl Domain for LinearXlaDomain {
    type Type = ArrayType;
    type Value = ShardMapTensor;
}

impl RuntimeDomain for LinearXlaDomain {
    #[inline]
    fn zero(&self, array_type: &ArrayType) -> Result<Self::Value, TracingError> {
        validate_identity_synthesis(ZERO_OPERATION_NAME, array_type)?;
        Ok(ShardMapTensor::zero(array_type.clone()))
    }

    #[inline]
    fn one(&self, array_type: &ArrayType) -> Result<Self::Value, TracingError> {
        validate_identity_synthesis(ONE_OPERATION_NAME, array_type)?;
        Ok(ShardMapTensor::one(array_type.clone()))
    }
}

impl TracingDomain for LinearXlaDomain {
    type OperationCarrier = LinearXlaOperation<ShardMapTensor>;
}

impl<'c> LinearizableDomain for XlaDomain<'c> {
    type LinearDomain = LinearXlaDomain;

    #[inline]
    fn linear_domain(&self) -> &Self::LinearDomain {
        LinearXlaDomain::token()
    }
}

fn validate_identity_synthesis(identity: &'static str, array_type: &ArrayType) -> Result<(), TracingError> {
    match array_type.data_type() {
        DataType::Token | DataType::C64 | DataType::C128 => Err(TypeError {
            message: (format!(
                "xla domain cannot synthesize {identity} value for element type {}",
                array_type.data_type()
            ))
            .into(),
        }
        .into()),
        _ => Ok(()),
    }
}

impl<'c> XlaDomain<'c> {
    /// Materializes a concrete [`Array`] whose addressable shards are filled with a constant.
    #[cfg(test)]
    fn constant(&self, array_type: &ArrayType, kind: ConstantKind) -> Result<Array<'c>, XlaDomainError> {
        let global_shape = static_dimensions_or_panic(array_type);
        let sharding = match array_type.sharding() {
            Some(sharding) => sharding.clone(),
            None => Sharding::replicated(self.mesh().logical_mesh().clone(), global_shape.len()),
        };
        let effective_type = ArrayType::new(
            array_type.data_type(),
            array_type.shape().clone(),
            array_type.layout().cloned(),
            Some(sharding),
        )
        .map_err(ArrayError::from)?;
        let addressable_ids = addressable_device_ids(self.client(), self.mesh())?;
        let element_size_in_bytes = device_put_element_size_in_bytes(array_type.data_type())?;

        let mut addressable_buffers = Vec::with_capacity(addressable_ids.len());
        for shard in shards_for_type(&effective_type, self.mesh())? {
            let shard_device = shard.device();
            if !addressable_ids.contains(&shard_device.id()) {
                continue;
            }
            let shard_shape = shard.shape();
            let element_count = shard_shape.as_slice().iter().copied().product::<usize>();
            let bytes = constant_bytes(array_type.data_type(), kind, element_count, element_size_in_bytes);
            let dimensions = shard_shape.as_slice().iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
            let device = self
                .client()
                .addressable_devices()?
                .into_iter()
                .find(|device| device.id().map(|id| id == shard_device.id()).unwrap_or(false))
                .ok_or(ArrayError::MissingClientDeviceForLocalDevice {
                    device_id: shard_device.id(),
                    process_index: shard_device.process_index(),
                })?;
            let buffer = self.client().buffer(
                bytes.as_slice(),
                array_type.data_type().to_pjrt(),
                dimensions.as_slice(),
                None,
                device,
                None,
            )?;
            addressable_buffers.push(buffer);
        }

        Array::from_addressable_buffers(effective_type, self.mesh().clone(), addressable_buffers).map_err(Into::into)
    }

    /// Renders a traced XLA program as a StableHLO/Shardy MLIR module.
    ///
    /// # Parameters
    ///
    ///   - `traced`: Traced XLA program to lower.
    ///   - `function_name`: Symbol name to use for the outer `func.func` in the emitted module.
    #[allow(private_bounds, private_interfaces)]
    pub fn lower<
        Input: Parameterized<ArrayType, Family: ParameterizedFamily<super::shard_map::ShardMapTensor>>,
        Output: Parameterized<ArrayType, Family: ParameterizedFamily<super::shard_map::ShardMapTensor>>,
        S: AsRef<str>,
    >(
        &self,
        traced: &TracedXlaProgram<Input, Output>,
        function_name: S,
    ) -> Result<String, XlaDomainError> {
        traced.to_mlir_module(function_name).map_err(Into::into)
    }

    /// Compiles a MLIR/StableHLO module using this domain's PJRT client and default
    /// [`CompilationOptions`].
    ///
    /// # Parameters
    ///
    ///   - `mlir_module`: MLIR text for the module to compile.
    pub fn compile(&self, mlir_module: &str) -> Result<LoadedExecutable<'c>, XlaDomainError> {
        let program = Program::Mlir { bytecode: mlir_module.as_bytes().to_vec() };
        self.client().compile(&program, &self.compilation_options).map_err(Into::into)
    }

    /// Executes a compiled program against this domain's device mesh, reassembling per-device
    /// outputs into distributed [`Array`] values.
    ///
    /// # Parameters
    ///
    ///   - `executable`: Loaded executable to run.
    ///   - `inputs`: Global input arrays in the order expected by the executable.
    ///   - `output_types`: One [`ArrayType`] per executable output, used to reassemble output
    ///     buffers back into distributed [`Array`] values.
    pub fn execute(
        &self,
        executable: &LoadedExecutable<'c>,
        inputs: Vec<Array<'c>>,
        output_types: &[ArrayType],
    ) -> Result<Vec<Array<'c>>, XlaDomainError> {
        let addressable_device_ids = executable
            .addressable_devices()?
            .iter()
            .map(|device| device.id().map_err(XlaDomainError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let arguments = Array::into_execute_arguments(inputs, addressable_device_ids.as_slice())?;
        let device_outputs =
            executable.execute(arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)?;

        let output_count = output_types.len();
        for outputs in &device_outputs {
            if outputs.outputs.len() != output_count {
                return Err(XlaDomainError::Pjrt(ryft_pjrt::Error::invalid_argument(format!(
                    "expected {output_count} output(s) per device, but got {}",
                    outputs.outputs.len(),
                ))));
            }
        }

        let mut per_output_buffers: Vec<Vec<Buffer<'c>>> =
            (0..output_count).map(|_| Vec::with_capacity(addressable_device_ids.len())).collect();
        for device_output in device_outputs {
            device_output.done.r#await()?;
            for (output_index, buffer) in device_output.outputs.into_iter().enumerate() {
                per_output_buffers[output_index].push(buffer);
            }
        }

        let mut outputs = Vec::with_capacity(output_count);
        for (output_index, addressable_buffers) in per_output_buffers.into_iter().enumerate() {
            let output_type = output_types[output_index].clone();
            let sharding = match output_type.sharding() {
                Some(sharding) => sharding.clone(),
                None => {
                    let rank = output_type.shape().dimensions().len();
                    Sharding::replicated(self.mesh().logical_mesh().clone(), rank)
                }
            };
            let resolved_type = ArrayType::new(
                output_type.data_type(),
                output_type.shape().clone(),
                output_type.layout().cloned(),
                Some(sharding),
            )
            .map_err(ArrayError::from)?;
            outputs.push(Array::from_addressable_buffers(resolved_type, self.mesh().clone(), addressable_buffers)?);
        }
        Ok(outputs)
    }

    /// Lowers, compiles, and executes a traced XLA program in a single call.
    ///
    /// # Parameters
    ///
    ///   - `traced`: Traced XLA program to run.
    ///   - `function_name`: Symbol name to use for the outer `func.func` in the emitted module.
    ///   - `inputs`: Global input arrays in the order expected by the traced program.
    ///   - `output_types`: One [`ArrayType`] per traced program output.
    #[allow(private_bounds, private_interfaces)]
    pub fn run<
        Input: Parameterized<ArrayType, Family: ParameterizedFamily<super::shard_map::ShardMapTensor>>,
        Output: Parameterized<ArrayType, Family: ParameterizedFamily<super::shard_map::ShardMapTensor>>,
        S: AsRef<str>,
    >(
        &self,
        traced: &TracedXlaProgram<Input, Output>,
        function_name: S,
        inputs: Vec<Array<'c>>,
        output_types: &[ArrayType],
    ) -> Result<Vec<Array<'c>>, XlaDomainError> {
        let mlir_module = self.lower(traced, function_name)?;
        let executable = self.compile(&mlir_module)?;
        self.execute(&executable, inputs, output_types)
    }
}

// ---------------------------------------------------------------------------
// Constant materialization
// ---------------------------------------------------------------------------

/// Kind of constant value materialized by [`XlaDomain::constant`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg(test)]
enum ConstantKind {
    /// Additive identity.
    Zero,

    /// Multiplicative identity.
    One,
}

/// Returns the static dimensions encoded by `array_type`, panicking if any dimension is dynamic.
///
/// Tests use this helper when constructing static-only values and treat dynamic shapes as programmer error.
#[cfg(test)]
fn static_dimensions_or_panic(array_type: &ArrayType) -> Vec<usize> {
    array_type
        .static_shape()
        .unwrap_or_else(|| panic!("XlaDomain requires static ArrayType shapes, but got {}", array_type.shape()))
        .dimensions()
        .to_vec()
}

/// Returns a dense row-major host buffer encoding `element_count` copies of `kind` for
/// `data_type`.
///
/// Booleans are encoded as one byte per element (`0` / `1`). Integers and floating-point numbers
/// are encoded in native-endian byte order matching
/// [`ryft_pjrt::Client::buffer`](ryft_pjrt::Client::buffer)'s expectations. Complex numbers are
/// encoded as a `(real, imaginary)` pair of native-endian floats.
#[cfg(test)]
fn constant_bytes(data_type: DataType, kind: ConstantKind, element_count: usize, element_size: usize) -> Vec<u8> {
    match kind {
        ConstantKind::Zero => vec![0u8; element_count * element_size],
        ConstantKind::One => {
            let pattern = one_pattern_bytes(data_type);
            debug_assert_eq!(pattern.len(), element_size);
            let mut bytes = Vec::with_capacity(element_count * element_size);
            for _ in 0..element_count {
                bytes.extend_from_slice(&pattern);
            }
            bytes
        }
    }
}

/// Returns the native-endian byte pattern for a single `1`-valued element of `data_type`.
#[cfg(test)]
fn one_pattern_bytes(data_type: DataType) -> Vec<u8> {
    match data_type {
        DataType::Boolean => vec![1u8],
        DataType::I8 => 1i8.to_ne_bytes().to_vec(),
        DataType::U8 => 1u8.to_ne_bytes().to_vec(),
        DataType::I16 => 1i16.to_ne_bytes().to_vec(),
        DataType::U16 => 1u16.to_ne_bytes().to_vec(),
        DataType::I32 => 1i32.to_ne_bytes().to_vec(),
        DataType::U32 => 1u32.to_ne_bytes().to_vec(),
        DataType::I64 => 1i64.to_ne_bytes().to_vec(),
        DataType::U64 => 1u64.to_ne_bytes().to_vec(),
        DataType::BF16 => half::bf16::ONE.to_bits().to_ne_bytes().to_vec(),
        DataType::F16 => half::f16::ONE.to_bits().to_ne_bytes().to_vec(),
        DataType::F32 => 1.0f32.to_ne_bytes().to_vec(),
        DataType::F64 => 1.0f64.to_ne_bytes().to_vec(),
        DataType::C64 => {
            let mut bytes = Vec::with_capacity(8);
            bytes.extend_from_slice(&1.0f32.to_ne_bytes());
            bytes.extend_from_slice(&0.0f32.to_ne_bytes());
            bytes
        }
        DataType::C128 => {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&1.0f64.to_ne_bytes());
            bytes.extend_from_slice(&0.0f64.to_ne_bytes());
            bytes
        }
        // 8-bit floating-point types do not have a canonical Rust representation; encoding `1.0`
        // as a raw byte pattern would depend on the exact FP8 variant. These variants are rejected
        // earlier by [`device_put_element_size_in_bytes`] for `XlaDomain::one`, so this arm is only
        // reachable for the supported set above.
        DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E4M3FN
        | DataType::F8E4M3FNUZ
        | DataType::F8E4M3B11FNUZ
        | DataType::F8E5M2
        | DataType::F8E5M2FNUZ
        | DataType::F8E8M0FNU
        | DataType::Token
        | DataType::I1
        | DataType::I2
        | DataType::I4
        | DataType::U1
        | DataType::U2
        | DataType::U4
        | DataType::F4E2M1FN => {
            panic!("XlaDomain::one does not support element type {data_type}")
        }
    }
}

/// Returns the addressable device IDs for `client`, filtered to devices that are both addressable by the client and
/// present in the mesh.
#[cfg(test)]
fn addressable_device_ids(client: &Client<'_>, mesh: &DeviceMesh) -> Result<Vec<DeviceId>, XlaDomainError> {
    let mut addressable = Vec::new();
    for device in client.addressable_devices()? {
        let device_id = device.id()?;
        if mesh.devices().iter().any(|device| device.id() == device_id) {
            addressable.push(device_id);
        }
    }
    Ok(addressable)
}

/// Returns the shard descriptors implied by `array_type` and `mesh`.
#[cfg(test)]
fn shards_for_type(array_type: &ArrayType, mesh: &DeviceMesh) -> Result<Vec<ShardDescriptor>, ArrayError> {
    let sharding = array_type.sharding().ok_or(crate::Error::MissingSharding)?;
    let global_shape = array_type.static_shape().ok_or_else(|| dynamic_array_shape_error(array_type))?;
    let (descriptors, _) = ShardLayout::new(&global_shape, mesh, sharding)?.into_parts();
    Ok(descriptors)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use ryft_core::sharding::{Device, LogicalMesh, MeshAxis, MeshAxisType, ShardingDimension};
    use ryft_core::types::{Shape, Size};

    use super::*;

    fn cpu_domain_mesh(client: &Client<'_>, axis: &str, axis_size: usize) -> DeviceMesh {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(axis, axis_size, MeshAxisType::Auto).unwrap()]).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        DeviceMesh::new(logical_mesh, devices).unwrap()
    }

    fn f32_values_from_bytes(bytes: &[u8]) -> Vec<f32> {
        assert_eq!(bytes.len() % size_of::<f32>(), 0);
        bytes
            .chunks_exact(size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn test_domain_zero_defaults_missing_sharding_to_replicated() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 2);
        let domain = XlaDomain::new(&client, mesh.clone());

        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3), Size::Static(2)]), None, None).unwrap();
        let array = domain.constant(&array_type, ConstantKind::Zero).unwrap();

        assert_eq!(array.shape(), vec![3, 2]);
        assert_eq!(array.shards().len(), 2);
        assert_eq!(array.addressable_shards().count(), 2);
        for shard in array.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            let host_bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
            let values = f32_values_from_bytes(host_bytes.as_slice());
            assert_eq!(values, vec![0.0; 6]);
        }
    }

    #[test]
    fn test_domain_one_fills_sharded_array_with_ones() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(sharding)).unwrap();
        let domain = XlaDomain::new(&client, mesh);

        let array = domain.constant(&array_type, ConstantKind::One).unwrap();

        assert_eq!(array.shape(), vec![4]);
        assert_eq!(array.shards().len(), 2);
        assert_eq!(array.addressable_shards().count(), 2);
        for shard in array.addressable_shards() {
            assert_eq!(shard.shape(), StaticShape::new(vec![2]));
            let buffer = shard.buffer().unwrap();
            let host_bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
            let values = f32_values_from_bytes(host_bytes.as_slice());
            assert_eq!(values, vec![1.0, 1.0]);
        }
    }

    #[test]
    fn test_domain_identity_synthesis_rejects_unsupported_constant_type() {
        let array_type = ArrayType::scalar(DataType::C64);

        assert!(matches!(
            XlaDomain::token().one(&array_type),
            Err(TracingError::Type(error))
                if error.message == "xla domain cannot synthesize one value for element type c64"
        ));
    }

    #[test]
    fn test_domain_accessors_return_constructor_arguments() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client, mesh.clone());

        assert_eq!(domain.mesh(), &mesh);
        assert_eq!(domain.compilation_options(), &CompilationOptions::default());
    }
}
