//! XLA backend token used for both shard-map tracing and PJRT-backed execution.
//!
//! [`XlaEngine`] is the single backend token for traced XLA programs and PJRT-backed execution.
//! It materializes concrete PJRT-backed arrays, lowers traced programs to MLIR, compiles them,
//! and executes them. Type-directed tracing stages XLA graphs directly from [`ArrayType`] metadata
//! and therefore does not require a second tracing-only engine specialization.
//!
//! Cloning [`Array<'c>`] is cheap because every shard buffer lives behind an [`Arc`]; this is what
//! lets [`XlaEngine`] act as the engine value for transforms that require
//! [`Clone`](Engine::Value).

use std::marker::PhantomData;

use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Buffer, Client, LoadedExecutable, Program};

use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::sharding::{DeviceMesh, Sharding};
use ryft_core::tracing_v2::{LinearPrimitiveOp, engine::Engine};
use ryft_core::types::ArrayType;

use super::arrays::{Array, ArrayError};
use super::ops::XlaPrimitiveOp;
use super::shard_map::{ShardMapTensor, ShardMapTraceError, TracedXlaProgram};

#[cfg(test)]
use ryft_core::sharding::MeshDeviceId;
#[cfg(test)]
use ryft_core::types::Size;
#[cfg(test)]
use ryft_core::types::data_types::DataType;

#[cfg(test)]
use crate::pjrt::ToPjrt;

/// Error type returned by [`XlaEngine`] orchestration helpers.
#[derive(Debug, thiserror::Error)]
pub enum XlaEngineError {
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

/// Stateful [`Engine`] that materializes, lowers, compiles, and executes traced XLA programs
/// against a live PJRT [`Client`].
///
/// [`XlaEngine`] holds three pieces of context:
///
/// - a PJRT [`Client`] used to upload `zero`/`one` shards and to compile and execute programs,
/// - a concrete [`DeviceMesh`] used to resolve shard placement for arrays synthesized from
///   [`ArrayType`] metadata, and
/// - default [`CompilationOptions`] that [`XlaEngine::compile`] forwards to PJRT.
///
/// Holding the mesh on the engine keeps [`Engine::zero`] and [`Engine::one`] infallible — both
/// methods can rebuild a replicated fallback sharding from `self.mesh.logical_mesh` when the
/// supplied [`ArrayType`] omits one. The trait contract requires [`ArrayType::shape`] to be
/// fully static on the types passed to `zero` / `one`; dynamic shapes panic.
pub struct XlaEngine<'c> {
    /// PJRT client used by this engine.
    client: Option<&'c Client<'c>>,

    /// Concrete device mesh used when an [`ArrayType`] does not specify a sharding.
    mesh: Option<DeviceMesh>,

    /// Default compilation options forwarded to [`Client::compile`].
    compilation_options: CompilationOptions,

    marker: PhantomData<fn() -> Array<'c>>,
}

impl<'c> XlaEngine<'c> {
    /// Creates a new [`XlaEngine`] with default [`CompilationOptions`].
    #[inline]
    pub fn new(client: &'c Client<'c>, mesh: DeviceMesh) -> Self {
        Self::with_compilation_options(client, mesh, CompilationOptions::default())
    }

    /// Creates a new [`XlaEngine`] with explicit [`CompilationOptions`].
    #[inline]
    pub fn with_compilation_options(
        client: &'c Client<'c>,
        mesh: DeviceMesh,
        compilation_options: CompilationOptions,
    ) -> Self {
        Self { client: Some(client), mesh: Some(mesh), compilation_options, marker: PhantomData }
    }

    /// Creates a tracing-only backend token that carries the XLA staged operation universe but no
    /// PJRT execution context.
    ///
    /// This token is sufficient for nested transforms over already-traced XLA values because those
    /// paths only need the backend's operation carriers; they never materialize concrete arrays via
    /// [`Engine::zero`] or [`Engine::one`].
    #[inline]
    pub fn token() -> Self {
        Self { client: None, mesh: None, compilation_options: CompilationOptions::default(), marker: PhantomData }
    }

    /// Returns the PJRT [`Client`] this engine was constructed with.
    #[inline]
    pub fn client(&self) -> &'c Client<'c> {
        self.client.expect("execution XlaEngine should always carry a client")
    }

    /// Returns the concrete [`DeviceMesh`] this engine resolves shard placement against.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        self.mesh.as_ref().expect("execution XlaEngine should always carry a device mesh")
    }

    /// Returns the [`CompilationOptions`] that [`XlaEngine::compile`] forwards to PJRT.
    #[inline]
    pub fn compilation_options(&self) -> &CompilationOptions {
        &self.compilation_options
    }
}

impl<'c> Engine for XlaEngine<'c> {
    type Type = ArrayType;
    type Value = ShardMapTensor;
    type TracingOperation = XlaPrimitiveOp;
    type LinearOperation = LinearPrimitiveOp<ArrayType, ShardMapTensor>;

    fn zero(&self, array_type: &ArrayType) -> ShardMapTensor {
        ShardMapTensor::zero(array_type.clone())
    }

    fn one(&self, array_type: &ArrayType) -> ShardMapTensor {
        ShardMapTensor::one(array_type.clone())
    }
}

impl<'c> XlaEngine<'c> {
    /// Materializes a concrete [`Array`] whose addressable shards are filled with a constant.
    #[cfg(test)]
    fn constant(&self, array_type: &ArrayType, kind: ConstantKind) -> Result<Array<'c>, XlaEngineError> {
        let global_shape = static_shape_or_panic(array_type);
        let sharding = match &array_type.sharding {
            Some(sharding) => sharding.clone(),
            None => Sharding::replicated(self.mesh().logical_mesh.clone(), global_shape.len()),
        };
        let effective_type =
            ArrayType::new(array_type.data_type, array_type.shape.clone(), array_type.layout.clone(), Some(sharding))
                .map_err(ArrayError::from)?;
        let addressable_device_ids = addressable_mesh_device_ids(self.client(), self.mesh())?;
        let element_size_in_bytes = dense_element_size_in_bytes(array_type.data_type)?;

        let mut addressable_buffers = Vec::with_capacity(addressable_device_ids.len());
        for shard in shards_for_type(&effective_type, self.mesh())? {
            if !addressable_device_ids.contains(&shard.device.id) {
                continue;
            }
            let shard_shape = shard.shape();
            let element_count = shard_shape.iter().copied().product::<usize>();
            let bytes = constant_bytes(array_type.data_type, kind, element_count, element_size_in_bytes);
            let dimensions = shard_shape.iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
            let device = self
                .client()
                .addressable_devices()?
                .into_iter()
                .find(|device| device.id().map(|id| id == shard.device.id).unwrap_or(false))
                .ok_or(ArrayError::MissingClientDeviceForLocalMeshDevice {
                    device_id: shard.device.id,
                    process_index: shard.device.process_index,
                })?;
            let buffer = self.client().buffer(
                bytes.as_slice(),
                array_type.data_type.to_pjrt(),
                dimensions.as_slice(),
                None,
                device,
                None,
            )?;
            addressable_buffers.push(buffer);
        }

        Array::new(effective_type, self.mesh().clone(), addressable_buffers).map_err(Into::into)
    }

    /// Renders a traced XLA program as a StableHLO/Shardy MLIR module.
    ///
    /// # Parameters
    ///
    ///   - `traced`: Traced XLA program to lower.
    ///   - `function_name`: Symbol name to use for the outer `func.func` in the emitted module.
    #[allow(private_bounds, private_interfaces)]
    pub fn lower<Input, Output, S>(
        &self,
        traced: &TracedXlaProgram<Input, Output>,
        function_name: S,
    ) -> Result<String, XlaEngineError>
    where
        Input: Parameterized<ArrayType, ParameterStructure: Clone>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
        Input::Family: ParameterizedFamily<super::shard_map::ShardMapTensor>,
        Output::Family: ParameterizedFamily<super::shard_map::ShardMapTensor>,
        S: AsRef<str>,
    {
        traced.to_mlir_module(function_name).map_err(Into::into)
    }

    /// Compiles a MLIR/StableHLO module using this engine's PJRT client and default
    /// [`CompilationOptions`].
    ///
    /// # Parameters
    ///
    ///   - `mlir_module`: MLIR text for the module to compile.
    pub fn compile(&self, mlir_module: &str) -> Result<LoadedExecutable<'c>, XlaEngineError> {
        let program = Program::Mlir { bytecode: mlir_module.as_bytes().to_vec() };
        self.client().compile(&program, &self.compilation_options).map_err(Into::into)
    }

    /// Executes a compiled program against this engine's device mesh, reassembling per-device
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
    ) -> Result<Vec<Array<'c>>, XlaEngineError> {
        let addressable_device_ids = executable
            .addressable_devices()?
            .iter()
            .map(|device| device.id().map_err(XlaEngineError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let arguments = Array::into_execute_arguments(inputs, addressable_device_ids.as_slice())?;
        let device_outputs =
            executable.execute(arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)?;

        let output_count = output_types.len();
        for outputs in &device_outputs {
            if outputs.outputs.len() != output_count {
                return Err(XlaEngineError::Pjrt(ryft_pjrt::Error::invalid_argument(format!(
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
            let sharding = match &output_type.sharding {
                Some(sharding) => sharding.clone(),
                None => {
                    let rank = output_type.shape.dimensions.len();
                    Sharding::replicated(self.mesh().logical_mesh.clone(), rank)
                }
            };
            let resolved_type =
                ArrayType::new(output_type.data_type, output_type.shape, output_type.layout, Some(sharding))
                    .map_err(ArrayError::from)?;
            outputs.push(Array::new(resolved_type, self.mesh().clone(), addressable_buffers)?);
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
    pub fn run<Input, Output, S>(
        &self,
        traced: &TracedXlaProgram<Input, Output>,
        function_name: S,
        inputs: Vec<Array<'c>>,
        output_types: &[ArrayType],
    ) -> Result<Vec<Array<'c>>, XlaEngineError>
    where
        Input: Parameterized<ArrayType, ParameterStructure: Clone>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
        Input::Family: ParameterizedFamily<super::shard_map::ShardMapTensor>,
        Output::Family: ParameterizedFamily<super::shard_map::ShardMapTensor>,
        S: AsRef<str>,
    {
        let mlir_module = self.lower(traced, function_name)?;
        let executable = self.compile(&mlir_module)?;
        self.execute(&executable, inputs, output_types)
    }
}

// ---------------------------------------------------------------------------
// Constant materialization
// ---------------------------------------------------------------------------

/// Kind of constant value materialized by [`XlaEngine::constant`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(test)]
enum ConstantKind {
    /// Additive identity.
    Zero,

    /// Multiplicative identity.
    One,
}

/// Returns the static shape encoded by `array_type`, panicking if any dimension is dynamic.
///
/// The [`Engine`] trait's `zero` / `one` methods are infallible, so a dynamic shape is treated as
/// a programming error here.
#[cfg(test)]
fn static_shape_or_panic(array_type: &ArrayType) -> Vec<usize> {
    array_type
        .shape
        .dimensions
        .iter()
        .map(|size| match size {
            Size::Static(value) => *value,
            _ => panic!("XlaEngine requires static ArrayType shapes, but got dimension {size:?}"),
        })
        .collect()
}

/// Returns the per-element dense host-buffer size in bytes for `data_type`.
///
/// Types whose host encoding is unambiguous (e.g., sub-byte or opaque types) are rejected; the
/// supported set matches the types accepted by [`Array::from_host_buffer`].
#[cfg(test)]
fn dense_element_size_in_bytes(data_type: DataType) -> Result<usize, ArrayError> {
    match data_type {
        DataType::Boolean
        | DataType::I8
        | DataType::U8
        | DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E4M3FN
        | DataType::F8E4M3FNUZ
        | DataType::F8E4M3B11FNUZ
        | DataType::F8E5M2
        | DataType::F8E5M2FNUZ
        | DataType::F8E8M0FNU => Ok(1),
        DataType::I16 | DataType::U16 | DataType::BF16 | DataType::F16 => Ok(2),
        DataType::I32 | DataType::U32 | DataType::F32 => Ok(4),
        DataType::I64 | DataType::U64 | DataType::F64 | DataType::C64 => Ok(8),
        DataType::C128 => Ok(16),
        DataType::Token
        | DataType::I1
        | DataType::I2
        | DataType::I4
        | DataType::U1
        | DataType::U2
        | DataType::U4
        | DataType::F4E2M1FN => Err(ArrayError::UnsupportedDevicePutElementType { element_type: data_type }),
    }
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
        // earlier by [`dense_element_size_in_bytes`] for `XlaEngine::one`, so this arm is only
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
            panic!("XlaEngine::one does not support element type {data_type}")
        }
    }
}

/// Returns the addressable mesh-device IDs for `client`, filtered to devices that are both
/// addressable by the client and present in the mesh.
#[cfg(test)]
fn addressable_mesh_device_ids(client: &Client<'_>, mesh: &DeviceMesh) -> Result<Vec<MeshDeviceId>, XlaEngineError> {
    let mut addressable = Vec::new();
    for device in client.addressable_devices()? {
        let device_id = device.id()?;
        if mesh.devices.iter().any(|mesh_device| mesh_device.id == device_id) {
            addressable.push(device_id);
        }
    }
    Ok(addressable)
}

/// Returns the shards implied by `array_type` and `mesh`, wrapping any sharding error as an
/// [`ArrayError`].
///
/// This goes through a stub [`Array`] constructed with no addressable buffers so we reuse
/// [`Array::new`]'s existing shard-descriptor bookkeeping and avoid re-exposing
/// `compute_shard_descriptors`.
#[cfg(test)]
fn shards_for_type<'o>(
    array_type: &ArrayType,
    mesh: &DeviceMesh,
) -> Result<Vec<super::arrays::ArrayShard<'o>>, ArrayError> {
    // Using the non-addressable stub trick keeps the shard-descriptor logic centralized inside
    // [`Array::new`] rather than exposing a private helper from `arrays.rs`.
    let stub = Array::new(array_type.clone(), mesh.clone(), Vec::new())?;
    Ok(stub.shards().to_vec())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, MeshDevice, ShardingDimension};
    use ryft_core::types::{Shape, Size};

    use super::*;

    fn cpu_engine_mesh(client: &Client<'_>, axis: &str, axis_size: usize) -> DeviceMesh {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(axis, axis_size, MeshAxisType::Auto).unwrap()]).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
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
    fn test_engine_zero_defaults_missing_sharding_to_replicated() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_engine_mesh(&client, "x", 2);
        let engine = XlaEngine::new(&client, mesh.clone());

        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3), Size::Static(2)]), None, None).unwrap();
        let array = engine.constant(&array_type, ConstantKind::Zero).unwrap();

        assert_eq!(array.shape(), vec![3, 2]);
        assert_eq!(array.shards().len(), 2);
        assert_eq!(array.addressable_shards().count(), 2);
        for shard in array.addressable_shards() {
            let buffer = shard.buffer.as_ref().unwrap();
            let host_bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
            let values = f32_values_from_bytes(host_bytes.as_slice());
            assert_eq!(values, vec![0.0; 6]);
        }
    }

    #[test]
    fn test_engine_one_fills_sharded_array_with_ones() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_engine_mesh(&client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(sharding)).unwrap();
        let engine = XlaEngine::new(&client, mesh);

        let array = engine.constant(&array_type, ConstantKind::One).unwrap();

        assert_eq!(array.shape(), vec![4]);
        assert_eq!(array.shards().len(), 2);
        assert_eq!(array.addressable_shards().count(), 2);
        for shard in array.addressable_shards() {
            assert_eq!(shard.shape(), vec![2]);
            let buffer = shard.buffer.as_ref().unwrap();
            let host_bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
            let values = f32_values_from_bytes(host_bytes.as_slice());
            assert_eq!(values, vec![1.0, 1.0]);
        }
    }

    #[test]
    fn test_engine_accessors_return_constructor_arguments() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_engine_mesh(&client, "x", 1);
        let engine = XlaEngine::new(&client, mesh.clone());

        assert_eq!(engine.mesh(), &mesh);
        assert_eq!(engine.compilation_options(), &CompilationOptions::default());
    }
}
