use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::OnceLock;

use prost::Message;

use crate::protos::CompilationOptions;
use crate::{
    Api, Buffer, BufferType, Chunk, Client, CopyToDeviceStream, Device, DeviceAssignment, Error, Event, Plugin,
    SerializedDeviceAssignment, Topology, Value, hash_map_from_c_api, invoke_pjrt_api_error_fn, slice_from_c_api,
    str_from_c_api,
};

/// Program that can be compiled using a PJRT [`Client`]. Programs can be provided in multiple formats though not all
/// PJRT [`Plugin`]s support all formats. The [`Program::Mlir`] format is the recommended format to use.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Program {
    /// [MLIR](https://mlir.llvm.org/) program represented using its MLIR bytecode. Note that different PJRT
    /// [`Plugin`]s may support different MLIR dialects and so not all programs are necessarily
    /// compatible with all PJRT plugins.
    Mlir {
        /// MLIR bytecode that represents a program.
        bytecode: Vec<u8>,
    },

    /// XLA HLO program.
    Hlo {
        /// Serialized [`HloModuleProto`](https://github.com/openxla/xla/blob/main/xla/service/hlo.proto#L557)
        /// message that represents a program.
        proto: Vec<u8>,
    },

    /// XLA HLO program paired with a configuration.
    HloWithConfig {
        /// Serialized [`HloModuleProtoWithConfig`](https://github.com/openxla/xla/blob/main/xla/xla.proto#L1725)
        /// message that represents a program.
        proto: Vec<u8>,
    },
}

impl Program {
    /// Returns the code of this [`Program`] that can be passed to functions in the PJRT C API.
    #[allow(deprecated)]
    fn code(&self) -> &[u8] {
        match self {
            Self::Mlir { bytecode } => bytecode,
            Self::Hlo { proto } => proto,
            Self::HloWithConfig { proto } => proto,
        }
    }

    /// Returns the format of this [`Program`] that can be passed to functions in the PJRT C API.
    #[allow(deprecated)]
    fn format(&self) -> std::ffi::CString {
        match self {
            Self::Mlir { .. } => std::ffi::CString::new("mlir").unwrap(),
            Self::Hlo { .. } => std::ffi::CString::new("hlo").unwrap(),
            Self::HloWithConfig { .. } => std::ffi::CString::new("hlo_with_config").unwrap(),
        }
    }
}

/// Represents a compiled [`Program`] that can be serialized and deserialized to e.g., cache compilation artifacts.
pub struct Executable {
    /// Handle that represents this [`Executable`] in the PJRT C API.
    handle: *mut ffi::PJRT_Executable,

    /// Underlying PJRT [`Api`].
    api: Api,

    /// Cached [`Executable::cost_analysis`] of this [`Executable`] so that it will only be constructed once.
    cost_analysis: OnceLock<Result<HashMap<String, Value>, Error>>,
}

impl Executable {
    /// Constructs a new [`Executable`] from the provided [`PJRT_Executable`](ffi::PJRT_Executable)
    /// handle that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(handle: *mut ffi::PJRT_Executable, api: Api) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT executable handle is a null pointer"))
        } else {
            Ok(Self { handle, api, cost_analysis: OnceLock::new() })
        }
    }

    /// Returns the [`PJRT_Executable`](ffi::PJRT_Executable) that corresponds to this [`Executable`]
    /// and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_Executable {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }

    /// Returns a string that identifies this [`Executable`].
    pub fn name(&'_ self) -> Result<Cow<'_, str>, Error> {
        use ffi::PJRT_Executable_Name_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_Name,
            { executable = self.to_c_api() },
            { executable_name, executable_name_size },
        )
        .map(|(string, string_len)| str_from_c_api(string, string_len))
    }

    /// Returns the number of replicas of this [`Executable`].
    pub fn replica_count(&self) -> Result<usize, Error> {
        use ffi::PJRT_Executable_NumReplicas_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Executable_NumReplicas, { executable = self.to_c_api() }, {
            num_replicas
        })
    }

    /// Returns the number of computations/partitions of this [`Executable`].
    pub fn computation_count(&self) -> Result<usize, Error> {
        use ffi::PJRT_Executable_NumPartitions_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Executable_NumPartitions, { executable = self.to_c_api() }, {
            num_partitions
        })
    }

    /// Returns the number of outputs of this [`Executable`] per [`Device`].
    pub fn output_count(&self) -> Result<usize, Error> {
        use ffi::PJRT_Executable_NumOutputs_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Executable_NumOutputs, { executable = self.to_c_api() }, {
            num_outputs
        })
    }

    /// Returns the [`BufferType`] for each of the outputs of this [`Executable`].
    pub fn output_element_types(&self) -> Result<Vec<BufferType>, Error> {
        use ffi::PJRT_Executable_OutputElementTypes_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_OutputElementTypes,
            { executable = self.to_c_api() },
            { output_types, num_output_types },
        )
        .map(|(output_types, output_type_count)| {
            unsafe { slice_from_c_api(output_types, output_type_count) }
                .iter()
                .map(|r#type| unsafe { BufferType::from_c_api(*r#type) })
                .collect()
        })
    }

    /// Returns the dimension sizes of each output of this [`Executable`].
    pub fn output_dimensions(&self) -> Result<Vec<Vec<u64>>, Error> {
        use ffi::PJRT_Executable_OutputDimensions_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_OutputDimensions,
            { executable = self.to_c_api() },
            { num_outputs, dims, dim_sizes },
        )
        .map(|(output_count, dimensions, dimension_counts)| unsafe {
            let dimension_counts = slice_from_c_api(dimension_counts, output_count);
            let mut dimensions_offset = 0;
            let mut output_dimensions = Vec::with_capacity(output_count);
            for dimension_count in dimension_counts {
                if dimensions.is_null() || *dimension_count == 0 {
                    output_dimensions.push(Vec::new());
                } else {
                    output_dimensions.push(
                        slice_from_c_api(dimensions.add(dimensions_offset) as *const u64, *dimension_count).to_vec(),
                    );
                }
                dimensions_offset += *dimension_count;
            }
            output_dimensions
        })
    }

    /// Returns the [`Memory`](crate::Memory) kind of each output of this [`Executable`].
    pub fn output_memory_kinds(&self) -> Result<Vec<Cow<'_, str>>, Error> {
        use ffi::PJRT_Executable_OutputMemoryKinds_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_OutputMemoryKinds,
            { executable = self.to_c_api() },
            { num_outputs, memory_kinds, memory_kind_sizes },
        )
        .map(|(output_count, memory_kinds, memory_kind_sizes)| unsafe {
            let memory_kind_sizes = slice_from_c_api(memory_kind_sizes, output_count);
            let mut output_memory_kinds = Vec::with_capacity(output_count);
            for (index, memory_kind_size) in memory_kind_sizes.iter().enumerate() {
                output_memory_kinds.push(str_from_c_api(*(memory_kinds.add(index)), *memory_kind_size));
            }
            output_memory_kinds
        })
    }

    /// Returns the size of the generated code for this [`Executable`] as a number of bytes. Note that,
    /// for [`Executable`]s that are the result of ahead-of-time compilation (e.g., using [`Plugin::compile`],
    /// as opposed to [`Client::compile`] followed by [`LoadedExecutable::executable`]), this function may return
    /// an [`Error::Unavailable`]. That is because the size of the generated code may depend on the number and
    /// type of addressable [`Device`]s after it is loaded, for example.
    pub fn generated_code_size_in_bytes(&self) -> Result<usize, Error> {
        use ffi::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args;
        let size = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_SizeOfGeneratedCodeInBytes,
            { executable = self.to_c_api() },
            { size_in_bytes },
        )?;
        if size <= 0 {
            Err(Error::unavailable("generated code size is unknown".to_string()))
        } else {
            Ok(size as usize)
        }
    }

    /// Returns a unique fingerprint for this [`Executable`]. Two [`Executable`]s that were produced by compiling with
    /// identical inputs (i.e., with the same [`Program`], [`CompilationOptions`], compiler version, etc.) should have
    /// the same fingerprint. Note that this function may not be implemented by all platforms.
    pub fn fingerprint(&self) -> Result<Cow<'_, str>, Error> {
        use ffi::PJRT_Executable_Fingerprint_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_Fingerprint,
            { executable = self.to_c_api() },
            { executable_fingerprint, executable_fingerprint_size },
        )
        .map(|(string, string_len)| str_from_c_api(string, string_len))
    }

    /// Returns the [`CompilationOptions`] that were used to compile this [`Executable`]. Note that the returned
    /// [`CompilationOptions`] may not be identical to the original [`CompilationOptions`] used to compile the
    /// [`Executable`], as the underlying PJRT [`Plugin`] may have added more information to them (e.g., information
    /// about the inferred output/result shapes of this [`Executable`]).
    pub fn compilation_options(&self) -> Result<CompilationOptions, Error> {
        use ffi::PJRT_Executable_GetCompileOptions_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_GetCompileOptions,
            { executable = self.to_c_api() },
            { serialized_bytes, serialized_bytes_size, serialized_compile_options, serialized_compile_options_deleter },
        )
        .and_then(
            |(
                serialized_bytes,
                serialized_bytes_size,
                serialized_compile_options,
                serialized_compile_options_deleter,
            )| {
                SerializedCompilationOptions {
                    handle: serialized_compile_options,
                    deleter: serialized_compile_options_deleter,
                    data: serialized_bytes,
                    data_size: serialized_bytes_size,
                }
                .proto()
            },
        )
    }

    /// Returns the optimized [`Program`] that corresponds to this [`Executable`].
    #[allow(deprecated)]
    pub fn optimized_program(&self) -> Result<Program, Error> {
        use ffi::PJRT_Executable_OptimizedProgram_Args;
        let mut program = ffi::PJRT_Program::new(std::ptr::null_mut(), 0, std::ptr::null(), 0);
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_OptimizedProgram,
            {
                executable = self.to_c_api(),
                program = &mut program as *mut _,
            },
        )?;
        let mut code = Vec::<u8>::with_capacity(program.code_size);
        program.code = code.as_mut_ptr() as *mut _;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_OptimizedProgram,
            {
                executable = self.to_c_api(),
                program = &mut program as *mut _,
            },
        )?;
        unsafe { code.set_len(program.code_size) };
        let format = str::from_utf8(unsafe { slice_from_c_api(program.format as *const _, program.format_size) });
        match format {
            Ok("mlir") => Ok(Program::Mlir { bytecode: code }),
            Ok("hlo") => Ok(Program::Hlo { proto: code }),
            Ok("hlo_with_config") => Ok(Program::HloWithConfig { proto: code }),
            Ok(format) => Err(Error::plugin_version_mismatch(format!("unknown program format: {format}"))),
            _ => Err(Error::plugin_version_mismatch("unknown program format".to_string())),
        }
    }

    /// Returns [`ExecutableMemoryStatistics`] for this [`Executable`] that allow callers to estimate memory usage
    /// for when they will run this [`Executable`]. The returned memory statistics may contain usage information for
    /// multiple [`Memory`](crate::Memory)s (e.g., HBM for GPUs or TPUs and host memory for CPUs).
    pub fn memory_statistics(&self) -> Result<ExecutableMemoryStatistics, Error> {
        use ffi::PJRT_Executable_GetCompiledMemoryStats_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_GetCompiledMemoryStats,
            { executable = self.to_c_api() },
            {
                generated_code_size_in_bytes,
                argument_size_in_bytes,
                output_size_in_bytes,
                alias_size_in_bytes,
                temp_size_in_bytes,
                host_generated_code_size_in_bytes,
                host_argument_size_in_bytes,
                host_output_size_in_bytes,
                host_alias_size_in_bytes,
                host_temp_size_in_bytes,
                peak_memory_in_bytes,
                total_size_in_bytes,
            },
        )
        .map(
            |(
                generated_code_size_in_bytes,
                argument_size_in_bytes,
                output_size_in_bytes,
                alias_size_in_bytes,
                temp_size_in_bytes,
                host_generated_code_size_in_bytes,
                host_argument_size_in_bytes,
                host_output_size_in_bytes,
                host_alias_size_in_bytes,
                host_temp_size_in_bytes,
                peak_memory_in_bytes,
                total_size_in_bytes,
            )| ExecutableMemoryStatistics {
                device_generated_code_size_in_bytes: generated_code_size_in_bytes as usize,
                device_input_size_in_bytes: argument_size_in_bytes as usize,
                device_output_size_in_bytes: output_size_in_bytes as usize,
                device_alias_size_in_bytes: alias_size_in_bytes as usize,
                device_temporary_size_in_bytes: temp_size_in_bytes as usize,
                device_peak_memory_in_bytes: peak_memory_in_bytes as usize,
                device_total_memory_in_bytes: total_size_in_bytes as usize,
                host_generated_code_size_in_bytes: host_generated_code_size_in_bytes as usize,
                host_input_size_in_bytes: host_argument_size_in_bytes as usize,
                host_output_size_in_bytes: host_output_size_in_bytes as usize,
                host_alias_size_in_bytes: host_alias_size_in_bytes as usize,
                host_temporary_size_in_bytes: host_temp_size_in_bytes as usize,
            },
        )
    }

    /// Returns the cost properties for this [`Executable`] after performing a cost analysis. Note that different
    /// platforms may return different properties. For example, some platforms may return the number of operations
    /// or the memory size of the inputs/outputs of the executable, based on performing program analysis. Other
    /// platforms may return different cost properties.
    ///
    /// Note that, for [`Executable`]s that are the result of ahead-of-time compilation (e.g., using
    /// [`Plugin::compile`], as opposed to [`Client::compile`] followed by [`LoadedExecutable::executable`]), this
    /// function may return an [`Error`]. That is because the cost analysis may depend on the number and type of
    /// addressable [`Device`]s after it is loaded, for example.
    pub fn cost_analysis(&self) -> Result<&HashMap<String, Value>, Error> {
        self.cost_analysis
            .get_or_init(|| {
                use ffi::PJRT_Executable_GetCostAnalysis_Args;
                let (properties, property_count) = invoke_pjrt_api_error_fn!(
                    self.api(),
                    PJRT_Executable_GetCostAnalysis,
                    { executable = self.to_c_api() },
                    { properties, num_properties },
                )?;
                Ok(hash_map_from_c_api(properties, property_count))
            })
            .as_ref()
            .map_err(|error| error.clone())
    }

    /// Serializes this [`Executable`] into a string (i.e., byte array).
    pub fn serialize(&self) -> Result<SerializedExecutable, Error> {
        use ffi::PJRT_Executable_Serialize_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_Serialize,
            { executable = self.to_c_api() },
            { serialized_bytes, serialized_bytes_size, serialized_executable, serialized_executable_deleter },
        )
        .map(
            |(serialized_bytes, serialized_bytes_size, serialized_executable, serialized_executable_deleter)| {
                SerializedExecutable {
                    handle: serialized_executable,
                    deleter: serialized_executable_deleter,
                    data: serialized_bytes,
                    data_size: serialized_bytes_size,
                }
            },
        )
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        use ffi::PJRT_Executable_Destroy_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Executable_Destroy, { event = self.to_c_api() })
            .expect("failed to destroy PJRT executable");
    }
}

/// Platform-specific serialized representation of an [`Executable`]. Note that the serialization format is not
/// guaranteed to be stable over time.
pub struct SerializedExecutable {
    /// Handle that represents this [`SerializedExecutable`] in the PJRT C API.
    handle: *mut ffi::PJRT_SerializedExecutable,

    /// Optional function that must be called to free the underlying memory when dropping this instance.
    deleter: Option<unsafe extern "C" fn(executable: *mut ffi::PJRT_SerializedExecutable)>,

    /// Pointer to the underlying bytes of this [`SerializedExecutable`].
    data: *const std::ffi::c_char,

    /// Size (i.e., number of bytes) of this [`SerializedExecutable`].
    data_size: usize,
}

impl SerializedExecutable {
    /// Returns a pointer to the underlying bytes of this [`SerializedExecutable`].
    pub fn data(&self) -> &[u8] {
        unsafe { slice_from_c_api(self.data as *const _, self.data_size) }
    }
}

impl PartialEq for SerializedExecutable {
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl Eq for SerializedExecutable {}

impl Hash for SerializedExecutable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data().hash(state);
    }
}

unsafe impl Send for SerializedExecutable {}
unsafe impl Sync for SerializedExecutable {}

impl Drop for SerializedExecutable {
    fn drop(&mut self) {
        if let Some(deleter) = self.deleter {
            unsafe { deleter(self.handle) };
        }
    }
}

impl TryFrom<LoadedExecutable<'_>> for Executable {
    type Error = Error;

    fn try_from(value: LoadedExecutable<'_>) -> Result<Self, Self::Error> {
        value.executable()
    }
}

/// Serialized representation of a [`CompilationOptions`] instance. The result of [`SerializedCompilationOptions::data`]
/// matches the result of [`CompilationOptions::encode_to_vec`] (as a slice).
pub struct SerializedCompilationOptions {
    /// Handle that represents this [`SerializedCompilationOptions`] in the PJRT C API.
    handle: *mut ffi::PJRT_SerializedCompileOptions,

    /// Optional function that must be called to free the underlying memory when dropping this instance.
    deleter: Option<unsafe extern "C" fn(options: *mut ffi::PJRT_SerializedCompileOptions)>,

    /// Pointer to the underlying bytes of this [`SerializedCompilationOptions`].
    data: *const std::ffi::c_char,

    /// Size (i.e., number of bytes) of this [`SerializedCompilationOptions`].
    data_size: usize,
}

impl SerializedCompilationOptions {
    /// Returns a pointer to the underlying bytes of this [`SerializedCompilationOptions`].
    pub fn data(&self) -> &[u8] {
        unsafe { slice_from_c_api(self.data as *const _, self.data_size) }
    }

    /// Returns the Protobuf message that corresponds to this [`SerializedCompilationOptions`].
    pub fn proto(&self) -> Result<CompilationOptions, Error> {
        CompilationOptions::decode(self.data()).map_err(|error| Error::invalid_argument(error.to_string()))
    }
}

impl PartialEq for SerializedCompilationOptions {
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl Eq for SerializedCompilationOptions {}

impl Hash for SerializedCompilationOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data().hash(state);
    }
}

unsafe impl Send for SerializedCompilationOptions {}
unsafe impl Sync for SerializedCompilationOptions {}

impl Drop for SerializedCompilationOptions {
    fn drop(&mut self) {
        if let Some(deleter) = self.deleter {
            unsafe { deleter(self.handle) };
        }
    }
}

/// Statistics about the [`Memory`](crate::Memory) consumption of an [`Executable`] (i.e., a compiled [`Program`]).
/// The total device memory required to run the [`Executable`] that corresponds to these statistics is:
///
/// ```text
/// device_generated_code_size_in_bytes
///   + device_input_size_in_bytes
///   + device_output_size_in_bytes
///   - device_alias_size_in_bytes
///   + device_temporary_size_in_bytes
/// ```
///
/// The total host memory required to run that same [`Executable`] can be computed similarly using the host-specific
/// fields of [`ExecutableMemoryStatistics`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutableMemoryStatistics {
    /// Number of bytes used for storing the generated code in device memory.
    pub device_generated_code_size_in_bytes: usize,

    /// Number of bytes used for storing the [`Executable`] input buffers in device memory.
    pub device_input_size_in_bytes: usize,

    /// Number of bytes used for storing the [`Executable`] output buffers in device memory.
    pub device_output_size_in_bytes: usize,

    /// Number of _aliased_ (i.e., re-used) bytes in device memory.
    pub device_alias_size_in_bytes: usize,

    /// Number of bytes used for temporary buffers in device memory.
    pub device_temporary_size_in_bytes: usize,

    /// Peak number of bytes used in device memory.
    pub device_peak_memory_in_bytes: usize,

    /// Total number of bytes available in device memory.
    pub device_total_memory_in_bytes: usize,

    /// Number of bytes used for storing the generated code in host memory.
    pub host_generated_code_size_in_bytes: usize,

    /// Number of bytes used for storing the [`Executable`] input buffers in host memory.
    pub host_input_size_in_bytes: usize,

    /// Number of bytes used for storing the [`Executable`] output buffers in host memory.
    pub host_output_size_in_bytes: usize,

    /// Number of _aliased_ (i.e., re-used) bytes in host memory.
    pub host_alias_size_in_bytes: usize,

    /// Number of bytes used for temporary buffers in host memory.
    pub host_temporary_size_in_bytes: usize,
}

/// In-memory [`Executable`] that represents a compiled [`Program`] which has been loaded by a compatible [`Client`] and
/// is ready to be executed. In PJRT, this is a subclass of [`Executable`].
///
/// The lifetime parameter `'c` captures the lifetime of the [`Client`] that owns this [`LoadedExecutable`],
/// ensuring that the client outlives the loaded executable.
pub struct LoadedExecutable<'c> {
    /// Handle that represents this [`LoadedExecutable`] in the PJRT C API.
    handle: *mut ffi::PJRT_LoadedExecutable,

    /// Underlying PJRT [`Api`].
    api: Api,

    /// Handle of the [`Client`] that owns this [`LoadedExecutable`]. Note that it is safe to hold a raw pointer here
    /// because the corresponding [`Client`] is guaranteed to outlive this [`LoadedExecutable`] by design. The reason we
    /// do not hold a reference to the [`Client`] itself is to avoid having to carry around an additional lifetime for
    /// the [`KeyValueStore`](crate::KeyValueStore) that is associated with that [`Client`].
    client: *mut crate::clients::ffi::PJRT_Client,

    /// [`PhantomData`] used to track the lifetime of the [`Client`] that owns this [`LoadedExecutable`].
    owner: PhantomData<&'c ()>,
}

impl<'c> LoadedExecutable<'c> {
    /// Constructs a new [`LoadedExecutable`] from the provided [`PJRT_LoadedExecutable`](ffi::PJRT_LoadedExecutable)
    /// handle that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(
        handle: *mut ffi::PJRT_LoadedExecutable,
        api: Api,
        client: *mut crate::clients::ffi::PJRT_Client,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT loaded executable handle is a null pointer"))
        } else if client.is_null() {
            Err(Error::invalid_argument("the provided PJRT client handle is a null pointer"))
        } else {
            Ok(Self { handle, api, client, owner: PhantomData })
        }
    }

    /// Returns the [`PJRT_LoadedExecutable`](ffi::PJRT_LoadedExecutable) that corresponds to this [`LoadedExecutable`]
    /// and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_LoadedExecutable {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }

    /// Returns the [`Executable`] that corresponds to this [`LoadedExecutable`].
    pub fn executable(&self) -> Result<Executable, Error> {
        use ffi::PJRT_LoadedExecutable_GetExecutable_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_LoadedExecutable_GetExecutable,
            { loaded_executable = self.to_c_api() },
            { executable },
        )
        .and_then(|handle| unsafe { Executable::from_c_api(handle, self.api()) })
    }

    /// Results the _addressable_ [`Device`]s that this [`LoadedExecutable`] will run on.
    pub fn addressable_devices(&'_ self) -> Result<Vec<Device<'_>>, Error> {
        use ffi::PJRT_LoadedExecutable_AddressableDevices_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_LoadedExecutable_AddressableDevices,
            { executable = self.to_c_api() },
            { addressable_devices, num_addressable_devices },
        )
        .and_then(|(devices, devices_count)| {
            unsafe { slice_from_c_api(devices, devices_count) }
                .iter()
                .map(|handle| unsafe { Device::from_c_api(*handle, self.api()) })
                .collect::<Result<Vec<_>, _>>()
        })
    }

    /// Returns the [`DeviceAssignment`] of this [`LoadedExecutable`].
    pub fn device_assignment(&self) -> Result<DeviceAssignment, Error> {
        use ffi::PJRT_LoadedExecutable_GetDeviceAssignment_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_LoadedExecutable_GetDeviceAssignment,
            { executable = self.to_c_api() },
            {
                serialized_bytes,
                serialized_bytes_size,
                serialized_device_assignment,
                serialized_device_assignment_deleter,
            },
        )
        .and_then(
            |(
                serialized_bytes,
                serialized_bytes_size,
                serialized_device_assignment,
                serialized_device_assignment_deleter,
            )| {
                SerializedDeviceAssignment::C {
                    handle: serialized_device_assignment,
                    deleter: serialized_device_assignment_deleter,
                    data: serialized_bytes,
                    data_size: serialized_bytes_size,
                }
                .deserialize()
            },
        )
    }

    /// Executes this [`LoadedExecutable`] on its _addressable_ devices (or on a single _addressable_ device if `device`
    /// is provided) using the provided inputs. Note that the execution of PJRT programs is asynchronous and so the
    /// runtime may not have completed execution by the time this function returns. You can use [`Buffer::ready`] on the
    /// returned [`Buffer`]s of [`Event::await`] on the returned [`Event`]s to wait for the execution to complete.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: [`ExecutionDeviceInputs`]s for each [`Device`] that is _addressable_ by this [`LoadedExecutable`].
    ///     If `device` is not [`None`], this must contain exactly one entry corresponding to that [`Device`].
    ///     Otherwise, the length of this slice must match the length of [`LoadedExecutable::addressable_devices`]
    ///     for this [`LoadedExecutable`].
    ///   - `launch_id`: Identifier for this execution/launch as part of a potentially multi-device launch. This can be
    ///     used to detect scheduling errors (e.g. if multi-host programs are launched in different orders on different
    ///     hosts, the launch IDs may be used by the runtime to detect the mismatch).
    ///   - `context`: Optional execution context to pass through to the runtime.
    ///   - `call_location`: Optional string that can be used to pass down call site location information for
    ///     diagnostics (e.g., `"file.rs:42:71"` describing the location as column 71 in line 42 of the file `file.rs`).
    ///     This field stores the source location of the Rust code that triggered the execution of this
    ///     [`LoadedExecutable`]. This differs from the source location metadata stored in the program's MLIR
    ///     representation, which typically refer to the origin of individual operations within the StableHLO module.
    ///     The PJRT plugin can use `call_location` for debugging and error reporting, allowing users to pinpoint which
    ///     program execution led to an issue.
    ///   - `incarnation_ids`: Optional mapping from task IDs to incarnation IDs for multi-host execution.
    ///     For more information on what task IDs and incarnation IDs are, refer to the documentation of
    ///     [`ProcessInformation`](crate::ProcessInformation).
    ///   - `device`: Optional _addressable_ [`Device`] on which to execute this [`LoadedExecutable`]. When provided,
    ///     the execution is launched only on that device and `inputs` must contain only inputs for this device. This
    ///     argument can be used with a multi-device [`LoadedExecutable`] to launch its execution only on one device.
    ///     In that case, the callers are responsible for separately launching execution on all participating devices
    ///     specified at compile time. Note that not all PJRT [`Plugin`]s or [`LoadedExecutable`]s support overriding
    ///     the execution device.
    pub fn execute<'l>(
        &self,
        inputs: Vec<ExecutionDeviceInputs<'c, 'l>>,
        launch_id: usize,
        context: Option<ExecutionContext>,
        call_location: Option<&str>,
        incarnation_ids: Option<HashMap<usize, usize>>,
        device: Option<&Device<'c>>,
    ) -> Result<Vec<ExecutionDeviceOutputs<'c>>, Error> {
        use ffi::PJRT_LoadedExecutable_Execute_Args;

        let mut inputs = inputs;

        let device_count = if device.is_some() { 1 } else { self.addressable_devices()?.len() };
        let input_count = inputs.first().map(|inputs| inputs.inputs.len()).unwrap_or(0);
        let input_is_donatable = inputs
            .first()
            .map(|inputs| inputs.inputs.iter().map(|input| input.donatable).collect::<Vec<bool>>())
            .unwrap_or_default();
        let send_callback_count = inputs.first().map(|inputs| inputs.send_callbacks.len()).unwrap_or(0);
        let receive_callback_count = inputs.first().map(|inputs| inputs.receive_callbacks.len()).unwrap_or(0);

        if inputs.len() != device_count {
            return Err(Error::invalid_argument(format!(
                "expected inputs for {device_count} device(s) but got inputs for {} device(s)",
                inputs.len(),
            )));
        }

        for (device_index, device_inputs) in inputs.iter().enumerate() {
            if device_inputs.inputs.len() != input_count {
                return Err(Error::invalid_argument(format!(
                    "expected {input_count} input(s) for each device but got {} for device {device_index}",
                    device_inputs.inputs.len(),
                )));
            }

            for (input_index, input) in device_inputs.inputs.iter().enumerate() {
                if input.donatable != input_is_donatable[input_index] {
                    return Err(Error::invalid_argument(format!(
                        "input {input_index} is not marked consistently across all devices \
                            as donatable or non-donatable",
                    )));
                }
            }

            if device_inputs.send_callbacks.len() != send_callback_count {
                return Err(Error::invalid_argument(format!(
                    "expected {send_callback_count} send callback(s) for each device \
                        but got {} for device {device_index}",
                    device_inputs.send_callbacks.len(),
                )));
            }

            if device_inputs.receive_callbacks.len() != receive_callback_count {
                return Err(Error::invalid_argument(format!(
                    "expected {receive_callback_count} receive callback(s) for each device \
                        but got {} for device {device_index}",
                    device_inputs.receive_callbacks.len(),
                )));
            }
        }

        // We need to handle memory related to send and receive callbacks _very_ carefully here. Specifically,
        // [`SendCallback::to_c_api`] returns a data structure which contains a pointer that was allocated by using
        // [`Box::into_raw`]. We shall pass those pointers to the C API [`PJRT_LoadedExecutable_Execute`] function
        // later on, but we need to make sure that we free the underlying memory _after_ the execution completes and
        // also in the case when something goes wrong. For that reason, we take ownership of these pointers using
        // [`Box::from_raw`] in `owned_send_callbacks`. This will ensure that if anything goes wrong later on in this
        // function, the underlying memory will be freed. Furthermore, after the call to
        // [`PJRT_LoadedExecutable_Execute`] and assuming that everything went well, we move ownership of the callbacks
        // to the corresponding [`Event::on_ready`] callbacks so that the underlying memory will be freed once execution
        // completes.
        let mut send_callbacks = unsafe {
            inputs
                .iter_mut()
                .map(|i| i.send_callbacks.drain(..).map(|c| c.to_c_api()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let mut send_callback_pointers = send_callbacks.iter_mut().map(|c| c.as_mut_ptr()).collect::<Vec<_>>();
        let mut send_callbacks = send_callbacks
            .iter()
            .map(|c| c.iter().map(|c| unsafe { Box::from_raw(c.user_arg as *mut SendCallback) }).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        // We handle receive callbacks in exactly the same way as send callbacks.
        let mut receive_callbacks = inputs
            .iter_mut()
            .map(|i| i.receive_callbacks.drain(..).map(|c| unsafe { c.to_c_api() }).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut receive_callback_pointers = receive_callbacks.iter_mut().map(|c| c.as_mut_ptr()).collect::<Vec<_>>();
        let mut receive_callbacks = receive_callbacks
            .iter()
            .map(|c| c.iter().map(|c| unsafe { Box::from_raw(c.user_arg as *mut ReceiveCallback) }).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let non_donatable_input_indices = input_is_donatable
            .into_iter()
            .enumerate()
            .filter_map(|(index, donatable)| if donatable { None } else { Some(index as i64) })
            .collect::<Vec<_>>();

        let call_location = call_location
            .map(std::ffi::CString::new)
            .transpose()
            .map_err(|error| Error::invalid_argument(error.to_string()))?;

        let incarnation_count = incarnation_ids.as_ref().map(|ids| ids.len()).unwrap_or(0);
        let mut task_ids = Vec::with_capacity(incarnation_count);
        let mut incarnation_ids_list = Vec::with_capacity(incarnation_count);
        incarnation_ids.iter().for_each(|ids| {
            ids.iter().for_each(|(task_id, incarnation_id)| {
                task_ids.push(*task_id as std::ffi::c_int);
                incarnation_ids_list.push(*incarnation_id as i64);
            });
        });

        let mut options = ffi::PJRT_ExecuteOptions::new(
            send_callback_pointers.as_mut_ptr(),
            receive_callback_pointers.as_mut_ptr(),
            send_callback_count,
            receive_callback_count,
            launch_id as i32,
            non_donatable_input_indices.as_ptr(),
            non_donatable_input_indices.len(),
            context.as_ref().map(|context| unsafe { context.to_c_api() }).unwrap_or(std::ptr::null_mut()),
            call_location.as_ref().map(|location| location.as_ptr()).unwrap_or(std::ptr::null()),
            task_ids.len(),
            task_ids.as_mut_ptr(),
            incarnation_ids_list.as_mut_ptr(),
        );

        // We prepare the input buffer handles array. This is an array of [`Buffer`] handle arrays where the outer
        // dimension corresponds to devices and the inner dimension corresponds to program inputs.
        let inputs = inputs
            .iter()
            .map(|i| i.inputs.iter().map(|i| unsafe { i.buffer.to_c_api() }).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let input_pointers = inputs.iter().map(|inputs| inputs.as_ptr()).collect::<Vec<_>>();

        // We pre-allocate the backing arrays for the output [`Buffer`] and [`Event`] handles. The output buffer handles
        // array is an array of [`Buffer`] handles arrays where the outer dimension corresponds to devices and the inner
        // dimension corresponds to program outputs. The [`Event`] handles array contains an [`Event`] for each
        // [`Device`], which can be used to track when the computation for this program is completed on each [`Device`].
        let output_count = self.executable()?.output_count()?;
        let mut output_buffers: Vec<*mut crate::buffers::ffi::PJRT_Buffer> =
            vec![std::ptr::null_mut(); device_count * output_count];
        let output_buffer_pointers = (0..device_count)
            .map(|device_index| unsafe { output_buffers.as_mut_ptr().add(device_index * output_count) })
            .collect::<Vec<_>>();
        let mut done_events: Vec<*mut crate::events::ffi::PJRT_Event> = vec![std::ptr::null_mut(); device_count];

        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_LoadedExecutable_Execute,
            {
                executable = self.to_c_api(),
                options = &mut options as *mut _,
                argument_lists = input_pointers.as_ptr(),
                num_devices = device_count,
                num_args = input_count,
                output_lists = output_buffer_pointers.as_ptr(),
                device_complete_events = done_events.as_mut_ptr(),
                execute_device = device.map(|device| device.to_c_api()).unwrap_or(std::ptr::null_mut()),
            },
        )?;

        // Process the outputs and the completion events.
        let mut execution_outputs = Vec::with_capacity(device_count);
        for device_index in 0..device_count {
            let done_event = unsafe { Event::from_c_api(done_events[device_index], self.api(), ()) }?;
            let send_callbacks = std::mem::take(&mut send_callbacks[device_index]);
            let receive_callbacks = std::mem::take(&mut receive_callbacks[device_index]);
            if !send_callbacks.is_empty() || !receive_callbacks.is_empty() {
                // Move the owned callback allocations into the event completion handler so that they
                // are released *only after* the runtime signals the device execution is done.
                done_event.on_ready(move |_| {
                    drop(send_callbacks);
                    drop(receive_callbacks);
                })?
            }

            let mut outputs = Vec::with_capacity(output_count);
            for output_index in 0..output_count {
                let output_handle = output_buffers[device_index * output_count + output_index];
                outputs.push(unsafe { Buffer::from_c_api(output_handle, self.api(), self.client)? });
            }

            execution_outputs.push(ExecutionDeviceOutputs { outputs, done: done_event })
        }

        Ok(execution_outputs)
    }

    /// Returns `true` if and only if this [`LoadedExecutable`] has been deleted using [`LoadedExecutable::delete`].
    pub fn is_deleted(&self) -> Result<bool, Error> {
        use ffi::PJRT_LoadedExecutable_IsDeleted_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_LoadedExecutable_IsDeleted, { executable = self.to_c_api() }, {
            is_deleted
        })
    }

    /// Drops this [`LoadedExecutable`]'s reference to its associated internal runtime object and resources without
    /// dropping this [`LoadedExecutable`] instance itself. After this function is called, this executable should only
    /// be used as a placeholder. The underlying internal runtime object will be freed after the last execution
    /// completes.
    ///
    /// # Safety
    ///
    /// This function is marked as unsafe because it results in eagerly dropping this [`LoadedExecutable`]'s reference
    /// to its associated internal runtime object before the [`LoadedExecutable`] instance is dropped, making it unsafe
    /// to use. Only [`LoadedExecutable::is_deleted`] is considered safe to call on this [`LoadedExecutable`] after this
    /// function has been called.
    pub unsafe fn delete(&self) -> Result<(), Error> {
        use ffi::PJRT_LoadedExecutable_Delete_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_LoadedExecutable_Delete, { executable = self.to_c_api() })
    }
}

impl Drop for LoadedExecutable<'_> {
    fn drop(&mut self) {
        use ffi::PJRT_LoadedExecutable_Destroy_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_LoadedExecutable_Destroy, { event = self.to_c_api() })
            .expect("failed to destroy PJRT loaded executable");
    }
}

/// Represents a [`Buffer`] that is used as input in a [`LoadedExecutable::execute`] invocation.
pub struct ExecutionInput<'o> {
    /// [`Buffer`] to use as the input value.
    pub buffer: Buffer<'o>,

    /// Boolean flag indicating whether `buffer` should be treated as _donatable_, meaning that the runtime would be
    /// allowed to reuse the [`Buffer`]'s storage for output values and force treating the `buffer` as invalid after
    /// the execution completes. Note that, when executing on multiple [`Device`]s, this flag must be set to the same
    /// value for all corresponding [`ExecutionInput`]s on its [`Device`] (i.e., inputs at the same position).
    ///
    /// # Copy Protection
    ///
    /// Note that is a program input was declared as an alias paired with an output (i.e., this would correspond to
    /// a _donatable_ input) but that input has `donatable` set to `false` then _copy-protection_ kicks in. An extra
    /// output buffer will be allocated, and the contents of the input buffer that was meant to be aliased will be
    /// copied into that output buffer. This is so that the program can execute as if that output buffer was donated
    /// at runtime.
    ///
    /// For more information on buffer donation and aliasing, refer to
    /// [the official XLA documentation](https://openxla.org/xla/aliasing).
    pub donatable: bool,
}

impl<'o> From<Buffer<'o>> for ExecutionInput<'o> {
    fn from(buffer: Buffer<'o>) -> Self {
        Self { buffer, donatable: false }
    }
}

/// Represents the input [`Buffer`]s on a single [`Device`] in a call to [`LoadedExecutable::execute`], paired with
/// information on whether they should be treated as _donatable_ or not, as well as with [`SendCallback`]s and
/// [`ReceiveCallback`]s to be used for that execution on that [`Device`].
#[derive(Default)]
pub struct ExecutionDeviceInputs<'o, 'l> {
    /// Slice that contains the [`ExecutionInput`] that corresponds to each input of the [`LoadedExecutable`] that is
    /// being executed. The length of this slice must match the number of inputs of the corresponding executable.
    pub inputs: &'l [ExecutionInput<'o>],

    /// [`SendCallback`]s to use for _send_ operations that involve the host. There must be one [`SendCallback`] per
    /// `stablehlo.send` operation in the corresponding [`LoadedExecutable`]. The order of the callbacks in this [`Vec`]
    /// does not matter because [`SendCallback::channel_id`] is used to match callbacks to their corresponding
    /// `stablehlo.send` operations.
    pub send_callbacks: Vec<SendCallback>,

    /// [`ReceiveCallback`]s to use for _receive_ operations that involve the host. There must be one
    /// [`ReceiveCallback`] per `stablehlo.recv` operation in the corresponding [`LoadedExecutable`]. The order of the
    /// callbacks in this [`Vec`] does not matter because [`ReceiveCallback::channel_id`] is used to match callbacks
    /// to their corresponding `stablehlo.recv` operations.
    pub receive_callbacks: Vec<ReceiveCallback>,
}

impl<'o, 'l> From<&'l [ExecutionInput<'o>]> for ExecutionDeviceInputs<'o, 'l> {
    fn from(inputs: &'l [ExecutionInput<'o>]) -> Self {
        Self { inputs, ..Default::default() }
    }
}

/// Represents the output [`Buffer`]s on a single [`Device`] of a call to [`LoadedExecutable::execute`], paired with
/// an [`Event`] that can be used to track when the computation for this program ηασ completed on that [`Device`].
pub struct ExecutionDeviceOutputs<'o> {
    /// [`Vec`] that contains the output [`Buffer`] that corresponds to each output of the [`LoadedExecutable`]
    /// that was executed.
    pub outputs: Vec<Buffer<'o>>,

    /// [`Event`] that can be used to track when all computation pending on a single [`Device`] for the execution of
    /// a [`LoadedExecutable`] is completed.
    pub done: Event<()>,
}

/// Callback function that is invoked from the runtime when executing _send_ operations in [`Program`]s. The channel ID
/// used in the [`Program`] _send_ operation must match [`SendCallback::channel_id`] for the callback. Note that there
/// is no guarantee that [`SendCallback`]s will be invoked in the same order as their corresponding _send_ operations in
/// the [`Program`].
///
/// [`SendCallback`]s are used to send data from the PJRT runtime to the host while executing programs. In some ways,
/// they are the opposite of [`ReceiveCallback`]s that are used to receive data in the PJRT runtime from the host while
/// executing programs.
///
/// # Safety
///
/// Certain PJRT implementations might **not** signal [`Error`]s returned by this callback to the execution, and the
/// execution will run to completion with _undefined_ data returned by the callback in that case. If there is any
/// potential control flow that depends on the value of the returned data, returning an [`Error`] may be unsafe.
pub struct SendCallback {
    /// Channel ID that is used to identify the [`Program`] _send_ operation for which this callback will be invoked.
    pub channel_id: usize,

    /// Callback function that is invoked when the [`Program`] _send_ operation corresponding to this callback is
    /// executed. This function may be invoked multiple times for each _send_ operation as it is a _streaming_ callback.
    /// Specifically, the data will be sent in [`Chunk`]s and this function will be invoked for each chunk. Along with
    /// the chunk it has two additional inputs: the total number of bytes that is being sent over and a boolean value
    /// indicating whether the chunk in the current invocation is the last one or not for the ongoing _send_ operation.
    /// If this function returns an [`Error`], then PJRT might propagate that error properly, though not all
    /// implementations will do that; there may be cases where an implementation continues executing the program with
    /// _undefined_ data being sent downstream which can be _unsafe_.
    pub function: Box<dyn FnMut(Chunk, usize, bool) -> Result<(), Error>>,
}

impl SendCallback {
    /// Returns the [`PJRT_SendCallbackInfo`](ffi::PJRT_SendCallbackInfo) that corresponds to this [`SendCallback`]
    /// and which can be passed to functions in the PJRT C API.
    ///
    /// # Safety
    ///
    /// This function consumes this callback returning a [`PJRT_SendCallbackInfo`](ffi::PJRT_SendCallbackInfo)
    /// that owns the underlying callback state. The returned [`PJRT_SendCallbackInfo::user_arg`] must be freed after
    /// the execution that uses this callback completes. The [`LoadedExecutable::execute`] implementation handles this
    /// by tying cleanup to the execution completion events.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(self) -> ffi::PJRT_SendCallbackInfo {
        extern "C" fn callback(
            chunk: *mut crate::transfers::ffi::PJRT_Chunk,
            callback_error: *mut crate::errors::ffi::PJRT_CallbackError,
            total_size_in_bytes: usize,
            done: bool,
            user_arg: *mut std::ffi::c_void,
        ) -> *mut crate::errors::ffi::PJRT_Error {
            unsafe {
                let user_arg = &mut *(user_arg as *mut SendCallback);
                match Chunk::from_c_api(chunk).and_then(|chunk| (user_arg.function)(chunk, total_size_in_bytes, done)) {
                    Ok(()) => std::ptr::null_mut(),
                    Err(error) => {
                        let error_message = error.message();
                        (*callback_error)(error.code(), error_message.as_ptr(), error_message.count_bytes())
                    }
                }
            }
        }

        ffi::PJRT_SendCallbackInfo {
            channel_id: self.channel_id as i64,
            user_arg: Box::into_raw(Box::new(self)) as *mut _,
            send_callback: callback,
        }
    }
}

/// Callback function that is invoked from the runtime when executing _receive_ operations in [`Program`]s. The
/// channel ID used in the [`Program`] _receive_ operation must match [`ReceiveCallback::channel_id`] for the callback.
/// Note that there is no guarantee that [`ReceiveCallback`]s will be invoked in the same order as their corresponding
/// _receive_ operations in the [`Program`].
///
/// [`ReceiveCallback`]s are used to receive data in the PJRT runtime from the host while executing programs. In some
/// ways, they are the opposite of [`SendCallback`]s that are used to send data from the PJRT runtime to the host while
/// executing programs.
pub struct ReceiveCallback {
    /// Channel ID that is used to identify the [`Program`] _send_ operation for which this callback will be invoked.
    pub channel_id: usize,

    /// Callback function that is invoked when the [`Program`] _receive_ operation corresponding to this callback
    /// is executed. This function will be invoked once for each _receive_ operation. It receives as input a
    /// [`CopyToDeviceStream`] which must be used to _stream_ data to the PJRT runtime. Specifically, the data
    /// will be sent in [`Chunk`]s.
    pub function: Box<dyn FnMut(CopyToDeviceStream<'_>)>,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl ReceiveCallback {
    /// Returns the [`PJRT_RecvCallbackInfo`](ffi::PJRT_RecvCallbackInfo) that corresponds to this [`ReceiveCallback`]
    /// and which can be passed to functions in the PJRT C API.
    ///
    /// # Safety
    ///
    /// This function consumes this callback returning a [`PJRT_RecvCallbackInfo`](ffi::PJRT_RecvCallbackInfo)
    /// that owns the underlying callback state. The returned [`PJRT_RecvCallbackInfo::user_arg`] must be freed after
    /// the execution that uses this callback completes. The [`LoadedExecutable::execute`] implementation handles this
    /// by tying cleanup to the execution completion events.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(self) -> ffi::PJRT_RecvCallbackInfo {
        extern "C" fn callback(
            stream: *mut crate::transfers::ffi::PJRT_CopyToDeviceStream,
            user_arg: *mut std::ffi::c_void,
        ) {
            unsafe {
                let user_arg = &mut *(user_arg as *mut ReceiveCallback);
                let stream = CopyToDeviceStream::from_c_api(stream, user_arg.api).unwrap();
                (user_arg.function)(stream);
            }
        }

        ffi::PJRT_RecvCallbackInfo {
            channel_id: self.channel_id as i64,
            user_arg: Box::into_raw(Box::new(self)) as *mut _,
            recv_callback: callback,
        }
    }
}

/// Opaque context that can be provided when executing [`LoadedExecutable`]s to supply additional information
/// to the runtime.
pub struct ExecutionContext {
    /// Handle that represents this [`ExecutionContext`] in the PJRT C API.
    handle: *mut ffi::PJRT_ExecuteContext,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl ExecutionContext {
    /// Constructs a new [`ExecutionContext`] from the provided [`PJRT_ExecuteContext`](ffi::PJRT_ExecuteContext)
    /// handle that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(handle: *mut ffi::PJRT_ExecuteContext, api: Api) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT execute context handle is a null pointer"))
        } else {
            Ok(Self { handle, api })
        }
    }

    /// Returns the [`PJRT_ExecuteContext`](ffi::PJRT_ExecuteContext) that corresponds to this [`ExecutionContext`]
    /// and which can be passed to functions in the PJRT C API.
    ///
    /// # Safety
    ///
    /// This [`ExecutionContext`] **must outlive** the entire execution that the resulting
    /// [`PJRT_ExecuteContext`](ffi::PJRT_ExecuteContext) will be passed to. Note that this _really_
    /// means the entire execution and not just the call to the function that kicks off the execution.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_ExecuteContext {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

impl Drop for ExecutionContext {
    fn drop(&mut self) {
        use ffi::PJRT_ExecuteContext_Destroy_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_ExecuteContext_Destroy, { event = self.to_c_api() })
            .expect("failed to destroy PJRT execution context");
    }
}

impl<'s> Client<'s> {
    /// Compiles a [`Program`] turning it into a [`LoadedExecutable`] which can be executed using this [`Client`].
    /// The compilation is aware of the _addressable_ [`Device`]s of this client, its memory configuration, its
    /// [`Topology`], and any other platform-specific attributes, and will thus be optimized accordingly. The resulting
    /// executable program will be compiled _specifically_ for the [`Device`]s managed by this [`Client`], and will be
    /// ready to be executed on those devices.
    ///
    /// This function is typically used for Just-In-Time (JIT) compilation of [`Program`]s in PJRT. If you want to
    /// perform Ahead-Of-Time (AOT) compilation for a specific [`Topology`] and without necessarily having access to
    /// an initialized [`Client`], then you must use [`Plugin::compile`] instead.
    pub fn compile(&'_ self, program: &Program, options: &CompilationOptions) -> Result<LoadedExecutable<'_>, Error> {
        use ffi::{PJRT_Client_Compile_Args, PJRT_Program};
        let code = program.code();
        let format = program.format();
        let program = PJRT_Program::new(code.as_ptr() as *mut _, code.len(), format.as_ptr(), format.count_bytes());
        let options = options.encode_to_vec();
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_Compile,
            {
                client = self.to_c_api(),
                program = &program as *const _,
                compile_options = options.as_ptr() as *const _,
                compile_options_size = options.len(),
            },
            { executable },
        )
        .and_then(|handle| unsafe { LoadedExecutable::from_c_api(handle, self.api(), self.to_c_api()) })
    }

    /// Deserializes the provided data into a [`LoadedExecutable`]. Note that the provided data must be the result of
    /// [`Executable::serialize`] and must have been produced by the same platform and PJRT version that this [`Client`]
    /// is using.
    pub fn deserialize_and_load_executable(
        &'_ self,
        data: &[u8],
        options: Option<&CompilationOptions>,
    ) -> Result<LoadedExecutable<'_>, Error> {
        use ffi::PJRT_Executable_DeserializeAndLoad_Args;
        let options = options.map(|options| options.encode_to_vec());
        let options = options.as_ref();
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Executable_DeserializeAndLoad,
            {
                client = self.to_c_api(),
                serialized_executable = data.as_ptr() as *const _,
                serialized_executable_size = data.len(),
                overridden_serialized_compile_options = options
                    .map(|options| options.as_ptr() as *const _)
                    .unwrap_or(std::ptr::null()),
                overridden_serialized_compile_options_size = options.map(|options| options.len()).unwrap_or(0),
            },
            { loaded_executable },
        )
        .and_then(|handle| unsafe { LoadedExecutable::from_c_api(handle, self.api(), self.to_c_api()) })
    }

    /// Creates a new [`SendCallback`] for the provided channel ID and using the provided callback function.
    /// The channel ID provided here **must match** the corresponding channel ID in a _send_ operation in the
    /// [`Program`] that will be executed.
    pub fn send_callback<F: 'static + FnMut(Chunk, usize, bool) -> Result<(), Error>>(
        &self,
        channel_id: usize,
        function: F,
    ) -> SendCallback {
        self.api().send_callback(channel_id, function)
    }

    /// Creates a new [`ReceiveCallback`] for the provided channel ID and using the provided callback function.
    /// The channel ID provided here **must match** the corresponding channel ID in a _receive_ operation in the
    /// [`Program`] that will be executed.
    pub fn receive_callback<F: 'static + FnMut(CopyToDeviceStream<'_>)>(
        &self,
        channel_id: usize,
        function: F,
    ) -> ReceiveCallback {
        self.api().receive_callback(channel_id, function)
    }

    /// Creates a new [`ExecutionContext`].
    pub fn execution_context(&self) -> Result<ExecutionContext, Error> {
        self.api().execution_context()
    }
}

impl Plugin {
    /// Compiles a [`Program`] turning it into an [`Executable`] such that it can be executed by [`Client`]s with the
    /// specified [`Topology`]. This is more of a "standalone" version of [`Client::compile`] which can be used for
    /// Ahead-Of-Time (AOT) compilation, as opposed to Just-In-Time (JIT) compilation. If you are interested in the
    /// latter (which is the most typical use case), and you have a [`Client`] available, then you should use
    /// [`Client::compile`] instead. This function is useful for situations where you want to compile a program
    /// for a specific hardware target (e.g., TPU v4) without actually having that hardware physically attached
    /// to the machine you are currently running on.
    ///
    /// # Parameters
    ///
    ///   - `program`: [`Program`] to compile.
    ///   - `topology`: [`Topology`] to compile for. This is necessary in this case because the compiler needs
    ///     to know how many devices will be available when executing this program, as well as their performance
    ///     characteristics, to optimize for them.
    ///   - `options`: Compilation options.
    pub fn compile(
        &self,
        program: &Program,
        topology: &Topology,
        options: &CompilationOptions,
    ) -> Result<Executable, Error> {
        self.api().compile(program, topology, options)
    }

    /// Creates a new [`SendCallback`] for the provided channel ID and using the provided callback function.
    /// The channel ID provided here **must match** the corresponding channel ID in a _send_ operation in the
    /// [`Program`] that will be executed.
    pub fn send_callback<F: 'static + FnMut(Chunk, usize, bool) -> Result<(), Error>>(
        &self,
        channel_id: usize,
        function: F,
    ) -> SendCallback {
        self.api().send_callback(channel_id, function)
    }

    /// Creates a new [`ReceiveCallback`] for the provided channel ID and using the provided callback function.
    /// The channel ID provided here **must match** the corresponding channel ID in a _receive_ operation in the
    /// [`Program`] that will be executed.
    pub fn receive_callback<F: 'static + FnMut(CopyToDeviceStream<'_>)>(
        &self,
        channel_id: usize,
        function: F,
    ) -> ReceiveCallback {
        self.api().receive_callback(channel_id, function)
    }

    /// Creates a new [`ExecutionContext`].
    pub fn execution_context(&self) -> Result<ExecutionContext, Error> {
        self.api().execution_context()
    }
}

impl Api {
    /// Compiles a [`Program`] turning it into an [`Executable`] such that it can be executed by [`Client`]s with the
    /// specified [`Topology`]. This is more of a "standalone" version of [`Client::compile`] which can be used for
    /// Ahead-Of-Time (AOT) compilation, as opposed to Just-In-Time (JIT) compilation. If you are interested in the
    /// latter (which is the most typical use case), and you have a [`Client`] available, then you should use
    /// [`Client::compile`] instead. This function is useful for situations where you want to compile a program
    /// for a specific hardware target (e.g., TPU v4) without actually having that hardware physically attached
    /// to the machine you are currently running on.
    ///
    /// # Parameters
    ///
    ///   - `program`: [`Program`] to compile.
    ///   - `topology`: [`Topology`] to compile for. This is necessary in this case because the compiler needs
    ///     to know how many devices will be available when executing this program, as well as their performance
    ///     characteristics, to optimize for them.
    ///   - `options`: Compilation options.
    pub(crate) fn compile(
        &self,
        program: &Program,
        topology: &Topology,
        options: &CompilationOptions,
    ) -> Result<Executable, Error> {
        use ffi::{PJRT_Compile_Args, PJRT_Program};
        let code = program.code();
        let format = program.format();
        let program = PJRT_Program::new(code.as_ptr() as *mut _, code.len(), format.as_ptr(), format.count_bytes());
        let options = options.encode_to_vec();
        invoke_pjrt_api_error_fn!(
            *self,
            PJRT_Compile,
            {
                topology = topology.to_c_api(),
                program = &program as *const _,
                compile_options = options.as_ptr() as *const _,
                compile_options_size = options.len(),
                client = std::ptr::null_mut(),
            },
            { executable },
        )
        .and_then(|handle| unsafe { Executable::from_c_api(handle, *self) })
    }

    /// Creates a new [`SendCallback`] for the provided channel ID and using the provided callback function.
    /// The channel ID provided here **must match** the corresponding channel ID in a _send_ operation in the
    /// [`Program`] that will be executed.
    pub(crate) fn send_callback<F: 'static + FnMut(Chunk, usize, bool) -> Result<(), Error>>(
        &self,
        channel_id: usize,
        function: F,
    ) -> SendCallback {
        SendCallback { channel_id, function: Box::new(function) }
    }

    /// Creates a new [`ReceiveCallback`] for the provided channel ID and using the provided callback function.
    /// The channel ID provided here **must match** the corresponding channel ID in a _receive_ operation in the
    /// [`Program`] that will be executed.
    pub(crate) fn receive_callback<F: 'static + FnMut(CopyToDeviceStream<'_>)>(
        &self,
        channel_id: usize,
        function: F,
    ) -> ReceiveCallback {
        ReceiveCallback { channel_id, function: Box::new(function), api: *self }
    }

    /// Creates a new [`ExecutionContext`].
    pub(crate) fn execution_context(&self) -> Result<ExecutionContext, Error> {
        use ffi::PJRT_ExecuteContext_Create_Args;
        invoke_pjrt_api_error_fn!(*self, PJRT_ExecuteContext_Create, {}, { context })
            .and_then(|handle| unsafe { ExecutionContext::from_c_api(handle, *self) })
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::buffers::ffi::{PJRT_Buffer, PJRT_Buffer_Type};
    use crate::clients::ffi::PJRT_Client;
    use crate::devices::ffi::{PJRT_Device, PJRT_DeviceAssignmentSerialized};
    use crate::errors::ffi::{PJRT_CallbackError, PJRT_Error};
    use crate::events::ffi::PJRT_Event;
    use crate::ffi::PJRT_Extension_Base;
    use crate::topologies::ffi::PJRT_TopologyDescription;
    use crate::transfers::ffi::{PJRT_Chunk, PJRT_CopyToDeviceStream};
    use crate::values::ffi::PJRT_NamedValue;

    #[repr(C)]
    pub struct PJRT_Program {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub code: *mut std::ffi::c_char,
        pub code_size: usize,
        pub format: *const std::ffi::c_char,
        pub format_size: usize,
    }

    impl PJRT_Program {
        pub fn new(
            code: *mut std::ffi::c_char,
            code_size: usize,
            format: *const std::ffi::c_char,
            format_size: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                code,
                code_size,
                format,
                format_size,
            }
        }

        pub fn empty() -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                code: std::ptr::null_mut(),
                code_size: 0,
                format: std::ptr::null(),
                format_size: 0,
            }
        }
    }

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_Executable {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Executable_Name_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub executable_name: *const std::ffi::c_char,
        pub executable_name_size: usize,
    }

    impl PJRT_Executable_Name_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                executable_name: std::ptr::null(),
                executable_name_size: 0,
            }
        }
    }

    pub type PJRT_Executable_Name = unsafe extern "C" fn(args: *mut PJRT_Executable_Name_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_NumReplicas_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub num_replicas: usize,
    }

    impl PJRT_Executable_NumReplicas_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), executable, num_replicas: 0 }
        }
    }

    pub type PJRT_Executable_NumReplicas =
        unsafe extern "C" fn(args: *mut PJRT_Executable_NumReplicas_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_NumPartitions_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub num_partitions: usize,
    }

    impl PJRT_Executable_NumPartitions_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                num_partitions: 0,
            }
        }
    }

    pub type PJRT_Executable_NumPartitions =
        unsafe extern "C" fn(args: *mut PJRT_Executable_NumPartitions_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_NumOutputs_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub num_outputs: usize,
    }

    impl PJRT_Executable_NumOutputs_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), executable, num_outputs: 0 }
        }
    }

    pub type PJRT_Executable_NumOutputs =
        unsafe extern "C" fn(args: *mut PJRT_Executable_NumOutputs_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_OutputElementTypes_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub output_types: *mut PJRT_Buffer_Type,
        pub num_output_types: usize,
    }

    impl PJRT_Executable_OutputElementTypes_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                output_types: std::ptr::null_mut(),
                num_output_types: 0,
            }
        }
    }

    pub type PJRT_Executable_OutputElementTypes =
        unsafe extern "C" fn(args: *mut PJRT_Executable_OutputElementTypes_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_OutputDimensions_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub num_outputs: usize,
        pub dims: *const i64,
        pub dim_sizes: *const usize,
    }

    impl PJRT_Executable_OutputDimensions_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                num_outputs: 0,
                dims: std::ptr::null(),
                dim_sizes: std::ptr::null(),
            }
        }
    }

    pub type PJRT_Executable_OutputDimensions =
        unsafe extern "C" fn(args: *mut PJRT_Executable_OutputDimensions_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_OutputMemoryKinds_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub num_outputs: usize,
        pub memory_kinds: *const *const std::ffi::c_char,
        pub memory_kind_sizes: *const usize,
    }

    impl PJRT_Executable_OutputMemoryKinds_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                num_outputs: 0,
                memory_kinds: std::ptr::null(),
                memory_kind_sizes: std::ptr::null(),
            }
        }
    }

    pub type PJRT_Executable_OutputMemoryKinds =
        unsafe extern "C" fn(args: *mut PJRT_Executable_OutputMemoryKinds_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_SizeOfGeneratedCodeInBytes_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub size_in_bytes: i64,
    }

    impl PJRT_Executable_SizeOfGeneratedCodeInBytes_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), executable, size_in_bytes: 0 }
        }
    }

    pub type PJRT_Executable_SizeOfGeneratedCodeInBytes =
        unsafe extern "C" fn(args: *mut PJRT_Executable_SizeOfGeneratedCodeInBytes_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_Fingerprint_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub executable_fingerprint: *const std::ffi::c_char,
        pub executable_fingerprint_size: usize,
    }

    impl PJRT_Executable_Fingerprint_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                executable_fingerprint: std::ptr::null_mut(),
                executable_fingerprint_size: 0,
            }
        }
    }

    pub type PJRT_Executable_Fingerprint =
        unsafe extern "C" fn(args: *mut PJRT_Executable_Fingerprint_Args) -> *mut PJRT_Error;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_SerializedCompileOptions {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Executable_GetCompileOptions_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *const PJRT_Executable,
        pub serialized_bytes: *const std::ffi::c_char,
        pub serialized_bytes_size: usize,
        pub serialized_compile_options: *mut PJRT_SerializedCompileOptions,
        pub serialized_compile_options_deleter: Option<unsafe extern "C" fn(exec: *mut PJRT_SerializedCompileOptions)>,
    }

    impl PJRT_Executable_GetCompileOptions_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                serialized_bytes: std::ptr::null(),
                serialized_bytes_size: 0,
                serialized_compile_options: std::ptr::null_mut(),
                serialized_compile_options_deleter: None,
            }
        }
    }

    pub type PJRT_Executable_GetCompileOptions =
        unsafe extern "C" fn(args: *mut PJRT_Executable_GetCompileOptions_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_OptimizedProgram_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub program: *mut PJRT_Program,
    }

    impl PJRT_Executable_OptimizedProgram_Args {
        pub fn new(executable: *mut PJRT_Executable, program: *mut PJRT_Program) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), executable, program }
        }
    }

    pub type PJRT_Executable_OptimizedProgram =
        unsafe extern "C" fn(args: *mut PJRT_Executable_OptimizedProgram_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_GetCompiledMemoryStats_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub generated_code_size_in_bytes: i64,
        pub argument_size_in_bytes: i64,
        pub output_size_in_bytes: i64,
        pub alias_size_in_bytes: i64,
        pub temp_size_in_bytes: i64,
        pub host_generated_code_size_in_bytes: i64,
        pub host_argument_size_in_bytes: i64,
        pub host_output_size_in_bytes: i64,
        pub host_alias_size_in_bytes: i64,
        pub host_temp_size_in_bytes: i64,
        pub peak_memory_in_bytes: i64,
        pub total_size_in_bytes: i64,
    }

    impl PJRT_Executable_GetCompiledMemoryStats_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                generated_code_size_in_bytes: 0,
                argument_size_in_bytes: 0,
                output_size_in_bytes: 0,
                alias_size_in_bytes: 0,
                temp_size_in_bytes: 0,
                host_generated_code_size_in_bytes: 0,
                host_argument_size_in_bytes: 0,
                host_output_size_in_bytes: 0,
                host_alias_size_in_bytes: 0,
                host_temp_size_in_bytes: 0,
                peak_memory_in_bytes: 0,
                total_size_in_bytes: 0,
            }
        }
    }

    pub type PJRT_Executable_GetCompiledMemoryStats =
        unsafe extern "C" fn(args: *mut PJRT_Executable_GetCompiledMemoryStats_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_GetCostAnalysis_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
        pub num_properties: usize,
        pub properties: *const PJRT_NamedValue,
    }

    impl PJRT_Executable_GetCostAnalysis_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                num_properties: 0,
                properties: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Executable_GetCostAnalysis =
        unsafe extern "C" fn(args: *mut PJRT_Executable_GetCostAnalysis_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_Executable,
    }

    impl PJRT_Executable_Destroy_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), executable }
        }
    }

    pub type PJRT_Executable_Destroy = unsafe extern "C" fn(args: *mut PJRT_Executable_Destroy_Args) -> *mut PJRT_Error;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_SerializedExecutable {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Executable_Serialize_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *const PJRT_Executable,
        pub serialized_bytes: *const std::ffi::c_char,
        pub serialized_bytes_size: usize,
        pub serialized_executable: *mut PJRT_SerializedExecutable,
        pub serialized_executable_deleter: Option<unsafe extern "C" fn(exec: *mut PJRT_SerializedExecutable)>,
    }

    impl PJRT_Executable_Serialize_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                serialized_bytes: std::ptr::null(),
                serialized_bytes_size: 0,
                serialized_executable: std::ptr::null_mut(),
                serialized_executable_deleter: None,
            }
        }
    }

    pub type PJRT_Executable_Serialize =
        unsafe extern "C" fn(args: *mut PJRT_Executable_Serialize_Args) -> *mut PJRT_Error;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_LoadedExecutable {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_LoadedExecutable_Fingerprint_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_LoadedExecutable,
        pub executable_fingerprint: *const std::ffi::c_char,
        pub executable_fingerprint_size: usize,
    }

    impl PJRT_LoadedExecutable_Fingerprint_Args {
        pub fn new(executable: *mut PJRT_LoadedExecutable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                executable_fingerprint: std::ptr::null_mut(),
                executable_fingerprint_size: 0,
            }
        }
    }

    pub type PJRT_LoadedExecutable_Fingerprint =
        unsafe extern "C" fn(args: *mut PJRT_LoadedExecutable_Fingerprint_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_LoadedExecutable_GetExecutable_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub loaded_executable: *mut PJRT_LoadedExecutable,
        pub executable: *mut PJRT_Executable,
    }

    impl PJRT_LoadedExecutable_GetExecutable_Args {
        pub fn new(loaded_executable: *mut PJRT_LoadedExecutable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                loaded_executable,
                executable: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_LoadedExecutable_GetExecutable =
        unsafe extern "C" fn(args: *mut PJRT_LoadedExecutable_GetExecutable_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_LoadedExecutable_AddressableDevices_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_LoadedExecutable,
        pub addressable_devices: *const *mut PJRT_Device,
        pub num_addressable_devices: usize,
    }

    impl PJRT_LoadedExecutable_AddressableDevices_Args {
        pub fn new(executable: *mut PJRT_LoadedExecutable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                addressable_devices: std::ptr::null(),
                num_addressable_devices: 0,
            }
        }
    }

    pub type PJRT_LoadedExecutable_AddressableDevices =
        unsafe extern "C" fn(args: *mut PJRT_LoadedExecutable_AddressableDevices_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_LoadedExecutable_GetDeviceAssignment_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_LoadedExecutable,
        pub serialized_bytes: *const std::ffi::c_char,
        pub serialized_bytes_size: usize,
        pub serialized_device_assignment: *mut PJRT_DeviceAssignmentSerialized,
        pub serialized_device_assignment_deleter:
            Option<unsafe extern "C" fn(da: *mut PJRT_DeviceAssignmentSerialized)>,
    }

    impl PJRT_LoadedExecutable_GetDeviceAssignment_Args {
        pub fn new(executable: *mut PJRT_LoadedExecutable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                serialized_bytes: std::ptr::null(),
                serialized_bytes_size: 0,
                serialized_device_assignment: std::ptr::null_mut(),
                serialized_device_assignment_deleter: None,
            }
        }
    }

    pub type PJRT_LoadedExecutable_GetDeviceAssignment =
        unsafe extern "C" fn(args: *mut PJRT_LoadedExecutable_GetDeviceAssignment_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_LoadedExecutable_IsDeleted_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_LoadedExecutable,
        pub is_deleted: bool,
    }

    impl PJRT_LoadedExecutable_IsDeleted_Args {
        pub fn new(executable: *mut PJRT_LoadedExecutable) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                is_deleted: false,
            }
        }
    }

    pub type PJRT_LoadedExecutable_IsDeleted =
        unsafe extern "C" fn(args: *mut PJRT_LoadedExecutable_IsDeleted_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_LoadedExecutable_Delete_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_LoadedExecutable,
    }

    impl PJRT_LoadedExecutable_Delete_Args {
        pub fn new(executable: *mut PJRT_LoadedExecutable) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), executable }
        }
    }

    pub type PJRT_LoadedExecutable_Delete =
        unsafe extern "C" fn(args: *mut PJRT_LoadedExecutable_Delete_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_LoadedExecutable_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_LoadedExecutable,
    }

    impl PJRT_LoadedExecutable_Destroy_Args {
        pub fn new(executable: *mut PJRT_LoadedExecutable) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), executable }
        }
    }

    pub type PJRT_LoadedExecutable_Destroy =
        unsafe extern "C" fn(args: *mut PJRT_LoadedExecutable_Destroy_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Compile_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology: *const PJRT_TopologyDescription,
        pub program: *const PJRT_Program,
        pub compile_options: *const std::ffi::c_char,
        pub compile_options_size: usize,
        pub client: *mut PJRT_Client,
        pub executable: *mut PJRT_Executable,
    }

    impl PJRT_Compile_Args {
        pub fn new(
            topology: *const PJRT_TopologyDescription,
            program: *const PJRT_Program,
            compile_options: *const std::ffi::c_char,
            compile_options_size: usize,
            client: *mut PJRT_Client,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                topology,
                program,
                compile_options,
                compile_options_size,
                client,
                executable: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Compile = unsafe extern "C" fn(args: *mut PJRT_Compile_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_Compile_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub program: *const PJRT_Program,
        pub compile_options: *const std::ffi::c_char,
        pub compile_options_size: usize,
        pub executable: *mut PJRT_LoadedExecutable,
    }

    impl PJRT_Client_Compile_Args {
        pub fn new(
            client: *mut PJRT_Client,
            program: *const PJRT_Program,
            compile_options: *const std::ffi::c_char,
            compile_options_size: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                program,
                compile_options,
                compile_options_size,
                executable: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Client_Compile = unsafe extern "C" fn(args: *mut PJRT_Client_Compile_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_DeserializeAndLoad_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub serialized_executable: *const std::ffi::c_char,
        pub serialized_executable_size: usize,
        pub loaded_executable: *mut PJRT_LoadedExecutable,
        pub overridden_serialized_compile_options: *const std::ffi::c_char,
        pub overridden_serialized_compile_options_size: usize,
    }

    impl PJRT_Executable_DeserializeAndLoad_Args {
        pub fn new(
            client: *mut PJRT_Client,
            serialized_executable: *const std::ffi::c_char,
            serialized_executable_size: usize,
            overridden_serialized_compile_options: *const std::ffi::c_char,
            overridden_serialized_compile_options_size: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                serialized_executable,
                serialized_executable_size,
                loaded_executable: std::ptr::null_mut(),
                overridden_serialized_compile_options,
                overridden_serialized_compile_options_size,
            }
        }
    }

    pub type PJRT_Executable_DeserializeAndLoad =
        unsafe extern "C" fn(args: *mut PJRT_Executable_DeserializeAndLoad_Args) -> *mut PJRT_Error;

    pub type PJRT_SendCallback = unsafe extern "C" fn(
        chunk: *mut PJRT_Chunk,
        callback_error: *mut PJRT_CallbackError,
        total_size_in_bytes: usize,
        done: bool,
        user_arg: *mut std::ffi::c_void,
    ) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_SendCallbackInfo {
        pub channel_id: i64,
        pub user_arg: *mut std::ffi::c_void,
        pub send_callback: PJRT_SendCallback,
    }

    pub type PJRT_RecvCallback =
        unsafe extern "C" fn(stream: *mut PJRT_CopyToDeviceStream, user_arg: *mut std::ffi::c_void);

    #[repr(C)]
    pub struct PJRT_RecvCallbackInfo {
        pub channel_id: i64,
        pub user_arg: *mut std::ffi::c_void,
        pub recv_callback: PJRT_RecvCallback,
    }

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_ExecuteContext {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_ExecuteContext_Create_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub context: *mut PJRT_ExecuteContext,
    }

    impl PJRT_ExecuteContext_Create_Args {
        pub fn new() -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                context: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_ExecuteContext_Create =
        unsafe extern "C" fn(args: *mut PJRT_ExecuteContext_Create_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_ExecuteContext_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub context: *mut PJRT_ExecuteContext,
    }

    impl PJRT_ExecuteContext_Destroy_Args {
        pub fn new(context: *mut PJRT_ExecuteContext) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), context }
        }
    }

    pub type PJRT_ExecuteContext_Destroy =
        unsafe extern "C" fn(args: *mut PJRT_ExecuteContext_Destroy_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_ExecuteOptions {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub send_callbacks: *mut *mut PJRT_SendCallbackInfo,
        pub recv_callbacks: *mut *mut PJRT_RecvCallbackInfo,
        pub num_send_ops: usize,
        pub num_recv_ops: usize,
        pub launch_id: std::ffi::c_int,
        pub non_donatable_input_indices: *const i64,
        pub num_non_donatable_input_indices: usize,
        pub context: *mut PJRT_ExecuteContext,
        pub call_location: *const std::ffi::c_char,
        pub num_tasks: usize,
        pub task_ids: *mut std::ffi::c_int,
        pub incarnation_ids: *mut i64,
    }

    impl PJRT_ExecuteOptions {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            send_callbacks: *mut *mut PJRT_SendCallbackInfo,
            recv_callbacks: *mut *mut PJRT_RecvCallbackInfo,
            num_send_ops: usize,
            num_recv_ops: usize,
            launch_id: std::ffi::c_int,
            non_donatable_input_indices: *const i64,
            num_non_donatable_input_indices: usize,
            context: *mut PJRT_ExecuteContext,
            call_location: *const std::ffi::c_char,
            num_tasks: usize,
            task_ids: *mut std::ffi::c_int,
            incarnation_ids: *mut i64,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                send_callbacks,
                recv_callbacks,
                num_send_ops,
                num_recv_ops,
                launch_id,
                non_donatable_input_indices,
                num_non_donatable_input_indices,
                context,
                call_location,
                num_tasks,
                task_ids,
                incarnation_ids,
            }
        }
    }

    #[repr(C)]
    pub struct PJRT_LoadedExecutable_Execute_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub executable: *mut PJRT_LoadedExecutable,
        pub options: *mut PJRT_ExecuteOptions,
        pub argument_lists: *const *const *mut PJRT_Buffer,
        pub num_devices: usize,
        pub num_args: usize,
        pub output_lists: *const *mut *mut PJRT_Buffer,
        pub device_complete_events: *mut *mut PJRT_Event,
        pub execute_device: *mut PJRT_Device,
    }

    impl PJRT_LoadedExecutable_Execute_Args {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            executable: *mut PJRT_LoadedExecutable,
            options: *mut PJRT_ExecuteOptions,
            argument_lists: *const *const *mut PJRT_Buffer,
            num_devices: usize,
            num_args: usize,
            output_lists: *const *mut *mut PJRT_Buffer,
            device_complete_events: *mut *mut PJRT_Event,
            execute_device: *mut PJRT_Device,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                executable,
                options,
                argument_lists,
                num_devices,
                num_args,
                output_lists,
                device_complete_events,
                execute_device,
            }
        }
    }

    pub type PJRT_LoadedExecutable_Execute =
        unsafe extern "C" fn(args: *mut PJRT_LoadedExecutable_Execute_Args) -> *mut PJRT_Error;
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use indoc::indoc;

    use crate::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use crate::tests::{TestPlatform, test_cpu_plugin, test_for_each_platform};
    use crate::{
        BufferType, Chunk, ClientOptions, CpuClientOptions, DeviceAssignment, Error, Executable, ExecutionContext,
        ExecutionDeviceInputs, ExecutionInput, LoadedExecutable, Program,
    };

    #[test]
    fn test_null_pointer_handling() {
        let api = test_cpu_plugin().api();
        let client = std::ptr::NonNull::<crate::clients::ffi::PJRT_Client>::dangling().as_ptr();
        let loaded_executable = std::ptr::NonNull::<crate::programs::ffi::PJRT_LoadedExecutable>::dangling().as_ptr();
        assert!(matches!(
            unsafe { Executable::from_c_api(std::ptr::null_mut(), api) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT executable handle is a null pointer",
        ));
        assert!(matches!(
            unsafe { LoadedExecutable::from_c_api(std::ptr::null_mut(), api, client) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT loaded executable handle is a null pointer",
        ));
        assert!(matches!(
            unsafe { LoadedExecutable::from_c_api(loaded_executable, api, std::ptr::null_mut()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT client handle is a null pointer",
        ));
        assert!(matches!(
            unsafe { ExecutionContext::from_c_api(std::ptr::null_mut(), api) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT execute context handle is a null pointer",
        ));
    }

    fn test_program(include_send_operation: bool, include_receive_operation: bool) -> Program {
        let module = match (include_send_operation, include_receive_operation) {
            (false, false) => indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x1xi32>, %arg1: tensor<2x1xi32>) -> tensor<2x1xi32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<2x1xi32>
                    return %0 : tensor<2x1xi32>
                  }
                }
            "},
            (true, false) => indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x1xi32>, %arg1: tensor<2x1xi32>) -> tensor<2x1xi32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<2x1xi32>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = \"stablehlo.send\"(%0, %1) <{\
                      channel_handle = #stablehlo.channel_handle<handle = 2, type = 2>, \
                      is_host_transfer = true\
                    }> : (tensor<2x1xi32>, !stablehlo.token) -> !stablehlo.token
                    return %0 : tensor<2x1xi32>
                  }
                }
            "},
            (false, true) => indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x1xi32>) -> tensor<2x1xi32> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:2 = \"stablehlo.recv\"(%0) <{\
                      channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>, \
                      is_host_transfer = true\
                    }> : (!stablehlo.token) -> (tensor<2x1xi32>, !stablehlo.token)
                    %2 = stablehlo.add %arg0, %1#0 : tensor<2x1xi32>
                    return %2 : tensor<2x1xi32>
                  }
                }
            "},
            (true, true) => indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x1xi32>) -> tensor<2x1xi32> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:2 = \"stablehlo.recv\"(%0) <{\
                      channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>, \
                      is_host_transfer = true\
                    }> : (!stablehlo.token) -> (tensor<2x1xi32>, !stablehlo.token)
                    %2 = stablehlo.add %arg0, %1#0 : tensor<2x1xi32>
                    %3 = stablehlo.after_all  : !stablehlo.token
                    %4 = \"stablehlo.send\"(%2, %3) <{\
                      channel_handle = #stablehlo.channel_handle<handle = 2, type = 2>, \
                      is_host_transfer = true\
                    }> : (tensor<2x1xi32>, !stablehlo.token) -> !stablehlo.token
                    return %2 : tensor<2x1xi32>
                  }
                }
            "},
        };
        Program::Mlir { bytecode: module.as_bytes().to_vec() }
    }

    fn test_compilation_options() -> CompilationOptions {
        CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 1,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        }
    }

    #[test]
    fn test_program_code_and_format() {
        let program = test_program(false, false);
        assert_eq!(program.format().to_str().unwrap(), "mlir");
        assert!(!program.code().is_empty());
    }

    #[test]
    fn test_client_compile() {
        test_for_each_platform!(|_plugin, client, platform| {
            let client_addressable_devices = client.addressable_devices().unwrap();
            let program = test_program(false, false);
            let options = test_compilation_options();
            let loaded_executable = client.compile(&program, &options).unwrap();
            let loaded_executable_addressable_devices = loaded_executable.addressable_devices().unwrap();
            assert_eq!(loaded_executable_addressable_devices, client_addressable_devices[..1]);
            assert_eq!(
                loaded_executable.device_assignment(),
                Ok(DeviceAssignment { replica_count: 1, computation_count: 1, assignment: vec![0] }),
            );
            let executable = loaded_executable.executable().unwrap();
            assert_eq!(executable.name().unwrap(), "main");
            assert_eq!(executable.replica_count(), Ok(1));
            assert_eq!(executable.computation_count(), Ok(1));
            assert_eq!(executable.output_count(), Ok(1));
            assert_eq!(executable.output_element_types(), Ok(vec![BufferType::I32]));
            assert_eq!(executable.output_dimensions(), Ok(vec![vec![2, 1]]));

            // The CPU plugin does not implement `output_memory_kinds` and `generated_code_size_in_bytes`.
            let output_memory_kinds = executable.output_memory_kinds();
            match platform {
                TestPlatform::Cpu => {
                    assert!(matches!(executable.output_memory_kinds(), Err(Error::Unimplemented { .. })));
                    assert!(matches!(executable.generated_code_size_in_bytes(), Err(Error::Unavailable { .. })));
                    assert!(matches!(executable.cost_analysis(), Err(Error::Unimplemented { .. })));
                }
                _ => {
                    assert_eq!(output_memory_kinds.unwrap(), vec!["device"]);
                    assert!(executable.generated_code_size_in_bytes().is_ok());
                    assert!(executable.cost_analysis().is_ok());
                }
            };

            assert!(executable.fingerprint().is_ok());
            assert!(executable.compilation_options().is_ok());
            assert!(matches!(executable.optimized_program(), Ok(Program::HloWithConfig { .. })));
            assert!(executable.memory_statistics().is_ok());
            assert!(executable.serialize().is_ok());
        });
    }

    #[test]
    fn test_client_deserialize_and_load_executable() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let program = test_program(false, false);
            let options = test_compilation_options();
            let loaded_executable = client.compile(&program, &options).unwrap();
            let executable = loaded_executable.executable().unwrap();
            let serialized_executable = executable.serialize().unwrap();
            let serialized_executable = serialized_executable.data();
            let deserialized_loaded_executable =
                client.deserialize_and_load_executable(serialized_executable, Some(&options)).unwrap();
            assert_eq!(deserialized_loaded_executable.addressable_devices(), loaded_executable.addressable_devices());
            assert_eq!(deserialized_loaded_executable.device_assignment(), loaded_executable.device_assignment());
            let deserialized_executable = deserialized_loaded_executable.executable().unwrap();
            assert_eq!(deserialized_executable.name(), executable.name());
            assert_eq!(deserialized_executable.replica_count(), executable.replica_count());
            assert_eq!(deserialized_executable.computation_count(), executable.computation_count());
            assert_eq!(deserialized_executable.output_count(), executable.output_count());
            assert_eq!(deserialized_executable.output_element_types(), executable.output_element_types());
            assert_eq!(deserialized_executable.output_dimensions(), executable.output_dimensions());
            assert_eq!(deserialized_executable.output_memory_kinds(), executable.output_memory_kinds());
            assert_eq!(
                deserialized_executable.generated_code_size_in_bytes(),
                executable.generated_code_size_in_bytes(),
            );
            assert_eq!(deserialized_executable.fingerprint(), executable.fingerprint());
            assert_eq!(deserialized_executable.compilation_options(), Ok(options));
            // The deserialized executable optimized program and memory statistics
            // are not guaranteed to match the original.
            assert!(deserialized_executable.optimized_program().is_ok());
            assert!(deserialized_executable.memory_statistics().is_ok());
            assert_eq!(deserialized_executable.cost_analysis(), executable.cost_analysis());
        });
    }

    #[test]
    fn test_plugin_compile() {
        test_for_each_platform!(|plugin, client, platform| {
            let topology = client.topology().unwrap();
            let program = test_program(false, false);
            let options = test_compilation_options();
            let executable = plugin.compile(&program, &topology, &options);
            match platform {
                TestPlatform::Cpu => assert!(executable.is_err()),
                _ => {
                    let executable = executable.unwrap();
                    assert_eq!(executable.name().unwrap(), "main");
                    assert_eq!(executable.replica_count(), Ok(1));
                    assert_eq!(executable.computation_count(), Ok(1));
                    assert_eq!(executable.output_count(), Ok(1));
                    assert_eq!(executable.output_element_types(), Ok(vec![BufferType::I32]));
                    assert_eq!(executable.output_dimensions(), Ok(vec![vec![2, 1]]));
                    let output_memory_kinds = executable.output_memory_kinds();
                    assert_eq!(output_memory_kinds.unwrap(), vec!["device"]);
                    assert!(executable.generated_code_size_in_bytes().is_err());
                    assert!(executable.fingerprint().is_ok());
                    assert_eq!(executable.compilation_options(), Ok(options));
                    assert!(matches!(executable.optimized_program(), Ok(Program::HloWithConfig { .. })));
                    assert!(executable.memory_statistics().is_ok());
                    assert!(executable.cost_analysis().is_err());
                    assert!(executable.serialize().is_ok());
                }
            };
        });
    }

    #[test]
    fn test_loaded_executable_delete() {
        test_for_each_platform!(|_plugin, client, platform| {
            let program = test_program(false, false);
            let options = test_compilation_options();
            let loaded_executable = client.compile(&program, &options).unwrap();

            // TODO(eaplatanios): Is this just broken or are we doing something wrong here?

            // [`LoadedExecutable::is_deleted`] does not appear to work correctly for the GPU plugins.
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    assert_eq!(loaded_executable.is_deleted(), Ok(true));
                }
                _ => assert_eq!(loaded_executable.is_deleted(), Ok(false)),
            };

            assert!(unsafe { loaded_executable.delete() }.is_ok());

            // [`LoadedExecutable::is_deleted`] does not appear to work correctly for any plugin.
            assert_eq!(loaded_executable.is_deleted(), Ok(false));
        });
    }

    #[test]
    fn test_loaded_executable_execute() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let program = test_program(false, false);
            let options = test_compilation_options();
            let executable = client.compile(&program, &options).unwrap();
            let device = executable.addressable_devices().unwrap()[0].clone();

            // Create the first input tensor: `[7, -1]`.
            let mut lhs_bytes = Vec::with_capacity(8);
            lhs_bytes.extend_from_slice(&7i32.to_ne_bytes());
            lhs_bytes.extend_from_slice(&(-1i32).to_ne_bytes());
            let lhs_bytes = lhs_bytes.as_slice();
            let lhs_buffer = client.buffer(lhs_bytes, BufferType::I32, &[2, 1], None, device.clone(), None).unwrap();

            // Create the second input tensor: `[35, -41]`.
            let mut rhs_bytes = Vec::with_capacity(8);
            rhs_bytes.extend_from_slice(&35i32.to_ne_bytes());
            rhs_bytes.extend_from_slice(&(-41i32).to_ne_bytes());
            let rhs_bytes = rhs_bytes.as_slice();
            let rhs_buffer = client.buffer(rhs_bytes, BufferType::I32, &[2, 1], None, device.clone(), None).unwrap();

            // Construct the execution device inputs that consist of our two tensors.
            let inputs = ExecutionDeviceInputs {
                inputs: &[
                    ExecutionInput { buffer: lhs_buffer, donatable: false },
                    ExecutionInput { buffer: rhs_buffer, donatable: false },
                ],
                ..Default::default()
            };

            // Execute the test program using our two input tensors.
            let mut outputs = executable.execute(vec![inputs], 0, None, None, None, None).unwrap();
            assert_eq!(outputs.len(), 1);
            let mut outputs = outputs.remove(0);

            // Wait for the asynchronous execution to complete.
            outputs.done.r#await().unwrap();
            let output = outputs.outputs.remove(0);

            // Copy the contents of the output buffer to the host.
            let output_bytes = output.copy_to_host(None).unwrap().r#await().unwrap();

            // Assert that the output buffer contains the expected value.
            let mut expected_output_bytes = Vec::with_capacity(8);
            expected_output_bytes.extend_from_slice(&42i32.to_ne_bytes());
            expected_output_bytes.extend_from_slice(&(-42i32).to_ne_bytes());
            assert_eq!(output_bytes, expected_output_bytes);
        });
    }

    #[test]
    fn test_loaded_executable_execute_validation_errors() {
        let plugin = test_cpu_plugin();
        let client = plugin.api().client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let program = test_program(false, false);
        let options = CompilationOptions {
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 2,
                partition_count: 1,
                ..Default::default()
            }),
            ..test_compilation_options()
        };
        let executable = client.compile(&program, &options).unwrap();
        let devices = executable.addressable_devices().unwrap();
        assert_eq!(devices.len(), 2);

        // [`LoadedExecutable::execute`] expects a [`Vec`] of inputs per device.
        assert!(matches!(
            executable.execute(Vec::new(), 0, None, None, None, None),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected inputs for 2 device(s) but got inputs for 0 device(s)",
        ));

        let device_0_inputs: Vec<ExecutionInput> = vec![
            client.buffer(&[0u8; 4], BufferType::I32, &[], None, devices[0].clone(), None).unwrap().into(),
            client.buffer(&[0u8; 4], BufferType::I32, &[], None, devices[0].clone(), None).unwrap().into(),
        ];

        let device_1_inputs: Vec<ExecutionInput> = vec![
            client.buffer(&[0u8; 4], BufferType::I32, &[], None, devices[1].clone(), None).unwrap().into(),
            client.buffer(&[0u8; 4], BufferType::I32, &[], None, devices[1].clone(), None).unwrap().into(),
        ];

        // [`LoadedExecutable::execute`] expects a [`Vec`] of inputs per device, where each [`Vec`] contains two inputs.
        let inputs = vec![
            ExecutionDeviceInputs { inputs: &device_0_inputs, ..Default::default() },
            ExecutionDeviceInputs { inputs: &device_1_inputs[..1], ..Default::default() },
        ];
        assert!(matches!(
            executable.execute(inputs, 0, None, None, None, None),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected 2 input(s) for each device but got 1 for device 1",
        ));

        // [`LoadedExecutable::execute`] expects a [`Vec`] of _send_ callbacks per device, where each [`Vec`]
        // contains the same number of _send_ callbacks across all devices.
        let inputs = vec![
            ExecutionDeviceInputs {
                inputs: &device_0_inputs,
                send_callbacks: vec![client.send_callback(1, |_, _, _| Ok(()))],
                ..Default::default()
            },
            ExecutionDeviceInputs { inputs: &device_1_inputs, ..Default::default() },
        ];
        assert!(matches!(
            executable.execute(inputs, 0, None, None, None, None),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected 1 send callback(s) for each device but got 0 for device 1",
        ));

        // [`LoadedExecutable::execute`] expects a [`Vec`] of _receive_ callbacks per device, where each [`Vec`]
        // contains the same number of _receive_ callbacks across all devices.
        let inputs = vec![
            ExecutionDeviceInputs {
                inputs: &device_0_inputs,
                receive_callbacks: vec![client.receive_callback(1, |_| {})],
                ..Default::default()
            },
            ExecutionDeviceInputs { inputs: &device_1_inputs, ..Default::default() },
        ];
        assert!(matches!(
            executable.execute(inputs, 0, None, None, None, None),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected 1 receive callback(s) for each device but got 0 for device 1",
        ));

        // [`LoadedExecutable::execute`] expects the provided call location to not contain the `nul` byte/character.
        let inputs = vec![
            ExecutionDeviceInputs { inputs: &device_0_inputs, ..Default::default() },
            ExecutionDeviceInputs { inputs: &device_1_inputs, ..Default::default() },
        ];
        assert!(matches!(
            executable.execute(inputs, 0, None, Some("invalid\0location"), None, None),
            Err(Error::InvalidArgument { message, .. }) if message == "nul byte found in provided data at position: 7",
        ));

        // [`LoadedExecutable::execute`] expects a [`Vec`] of inputs per device, where each [`Vec`] contains two inputs,
        // and the `donatable` flag for each input at the same index must be the same across all devices.
        let device_1_inputs: Vec<ExecutionInput> = vec![
            ExecutionInput {
                buffer: client.buffer(&[0u8; 4], BufferType::I32, &[], None, devices[1].clone(), None).unwrap(),
                donatable: true,
            },
            client.buffer(&[0u8; 4], BufferType::I32, &[], None, devices[1].clone(), None).unwrap().into(),
        ];
        let inputs = vec![
            ExecutionDeviceInputs { inputs: &device_0_inputs, ..Default::default() },
            ExecutionDeviceInputs { inputs: &device_1_inputs, ..Default::default() },
        ];
        assert!(matches!(
            executable.execute(inputs, 0, None, None, None, None),
            Err(Error::InvalidArgument { message, .. })
                if message == "input 0 is not marked consistently across all devices as donatable or non-donatable",
        ));
    }

    #[test]
    fn test_loaded_executable_execute_with_send_operation() {
        test_for_each_platform!(|_plugin, client, platform| {
            let program = test_program(true, false);
            let options = test_compilation_options();
            let executable = client.compile(&program, &options);
            match platform {
                TestPlatform::Cpu => assert!(executable.is_err()),
                _ => {
                    let executable = executable.unwrap();
                    let device = client.addressable_devices().unwrap()[0].clone();

                    // Create the first input tensor: `[7, -1]`.
                    let mut lhs_bytes = Vec::with_capacity(8);
                    lhs_bytes.extend_from_slice(&7i32.to_ne_bytes());
                    lhs_bytes.extend_from_slice(&(-1i32).to_ne_bytes());
                    let lhs_bytes = lhs_bytes.as_slice();
                    let lhs_buffer =
                        client.buffer(lhs_bytes, BufferType::I32, &[2, 1], None, device.clone(), None).unwrap();

                    // Create the second input tensor: `[35, -41]`.
                    let mut rhs_bytes = Vec::with_capacity(8);
                    rhs_bytes.extend_from_slice(&35i32.to_ne_bytes());
                    rhs_bytes.extend_from_slice(&(-41i32).to_ne_bytes());
                    let rhs_bytes = rhs_bytes.as_slice();
                    let rhs_buffer =
                        client.buffer(rhs_bytes, BufferType::I32, &[2, 1], None, device.clone(), None).unwrap();

                    // Create a _send_ callback that simply records the values it receives.
                    let observed_value = Arc::new(Mutex::new(None));
                    let observed_value_clone = observed_value.clone();
                    let mut collected_bytes = Vec::new();
                    let send_callback = client.send_callback(2, move |chunk, total_size, done| {
                        if collected_bytes.is_empty() && total_size > 0 {
                            collected_bytes.reserve(total_size);
                        }
                        collected_bytes.extend_from_slice(chunk.data());
                        if done && collected_bytes.len() >= 8 {
                            let value = [
                                i32::from_ne_bytes(collected_bytes[0..4].try_into().unwrap()),
                                i32::from_ne_bytes(collected_bytes[4..8].try_into().unwrap()),
                            ];
                            *observed_value_clone.lock().unwrap() = Some(value);
                        }
                        Ok(())
                    });

                    // Construct the execution device inputs that consist of our two tensors and send callback.
                    let inputs = ExecutionDeviceInputs {
                        inputs: &[
                            ExecutionInput { buffer: lhs_buffer, donatable: false },
                            ExecutionInput { buffer: rhs_buffer, donatable: false },
                        ],
                        send_callbacks: vec![send_callback],
                        ..Default::default()
                    };

                    // Execute the test program using our two input tensors.
                    let mut outputs = executable.execute(vec![inputs], 0, None, None, None, None).unwrap();
                    assert_eq!(outputs.len(), 1);
                    let mut outputs = outputs.remove(0);

                    // Wait for the asynchronous execution to complete.
                    outputs.done.r#await().unwrap();
                    let output = outputs.outputs.remove(0);

                    // Copy the contents of the output buffer to the host.
                    let output_bytes = output.copy_to_host(None).unwrap().r#await().unwrap();

                    // Assert that the output buffer contains the expected values.
                    let mut expected_output_bytes = Vec::with_capacity(8);
                    expected_output_bytes.extend_from_slice(&42i32.to_ne_bytes());
                    expected_output_bytes.extend_from_slice(&(-42i32).to_ne_bytes());
                    assert_eq!(output_bytes, expected_output_bytes);

                    // Assert that the _send_ callback observed the expected values.
                    assert_eq!(*observed_value.lock().unwrap(), Some([42i32, -42i32]));
                }
            };
        });
    }

    #[test]
    fn test_loaded_executable_execute_with_receive_operation() {
        test_for_each_platform!(|_plugin, client, platform| {
            let program = test_program(false, true);
            let options = test_compilation_options();
            let executable = client.compile(&program, &options);
            match platform {
                TestPlatform::Cpu => assert!(executable.is_err()),
                _ => {
                    let executable = executable.unwrap();
                    let device = client.addressable_devices().unwrap()[0].clone();

                    // Create the input tensor: `[7, -1]`.
                    let mut lhs_bytes = Vec::with_capacity(8);
                    lhs_bytes.extend_from_slice(&7i32.to_ne_bytes());
                    lhs_bytes.extend_from_slice(&(-1i32).to_ne_bytes());
                    let lhs_bytes = lhs_bytes.as_slice();
                    let lhs_buffer =
                        client.buffer(lhs_bytes, BufferType::I32, &[2, 1], None, device.clone(), None).unwrap();

                    // Create a _receive_ callback that streams the second input tensor to the device in two chunks.
                    let receive_callback = client.receive_callback(1, move |stream| {
                        let total_byte_count = stream.total_byte_count().unwrap();
                        let granule_byte_count = stream.granule_byte_count().unwrap();
                        assert_eq!(total_byte_count, 8);
                        assert_eq!(total_byte_count % granule_byte_count, 0);

                        // Create the second input tensor: `[35, -41]`.
                        let first_chunk_bytes = 35i32.to_ne_bytes().to_vec();
                        let second_chunk_bytes = (-41i32).to_ne_bytes().to_vec();

                        unsafe extern "C" fn drop_chunk(_: *mut std::ffi::c_void, arg: *mut std::ffi::c_void) {
                            unsafe {
                                drop(Box::from_raw(arg as *mut [u8; 4]));
                            }
                        }

                        let mut first_chunk_bytes = Box::new(first_chunk_bytes);
                        let mut first_chunk = crate::transfers::ffi::PJRT_Chunk {
                            data: first_chunk_bytes.as_mut_ptr().cast(),
                            size: first_chunk_bytes.len(),
                            deleter: Some(drop_chunk),
                            deleter_arg: Box::into_raw(first_chunk_bytes).cast(),
                        };
                        let transfer_complete_event =
                            stream.add_chunk(unsafe { Chunk::from_c_api(&mut first_chunk) }.unwrap());
                        assert!(transfer_complete_event.is_ok());

                        let mut second_chunk_bytes = Box::new(second_chunk_bytes);
                        let mut second_chunk = crate::transfers::ffi::PJRT_Chunk {
                            data: second_chunk_bytes.as_mut_ptr().cast(),
                            size: second_chunk_bytes.len(),
                            deleter: Some(drop_chunk),
                            deleter_arg: Box::into_raw(second_chunk_bytes).cast(),
                        };
                        let transfer_complete =
                            stream.add_chunk(unsafe { Chunk::from_c_api(&mut second_chunk) }.unwrap()).unwrap();
                        transfer_complete.r#await().unwrap();
                    });

                    // Construct the execution device inputs that consist of our input tensor and receive callback.
                    let inputs = ExecutionDeviceInputs {
                        inputs: &[ExecutionInput { buffer: lhs_buffer, donatable: false }],
                        receive_callbacks: vec![receive_callback],
                        ..Default::default()
                    };

                    // Execute the test program using our input tensor.
                    let mut outputs = executable.execute(vec![inputs], 0, None, None, None, None).unwrap();
                    assert_eq!(outputs.len(), 1);
                    let mut outputs = outputs.remove(0);

                    // Wait for the asynchronous execution to complete.
                    outputs.done.r#await().unwrap();
                    let output = outputs.outputs.remove(0);

                    // Copy the contents of the output buffer to the host.
                    let output_bytes = output.copy_to_host(None).unwrap().r#await().unwrap();

                    // Assert that the output buffer contains the expected values.
                    let mut expected_output_bytes = Vec::with_capacity(8);
                    expected_output_bytes.extend_from_slice(&42i32.to_ne_bytes());
                    expected_output_bytes.extend_from_slice(&(-42i32).to_ne_bytes());
                    assert_eq!(output_bytes, expected_output_bytes);
                }
            };
        });
    }

    #[test]
    fn test_loaded_executable_execute_with_send_and_receive_operation() {
        test_for_each_platform!(|_plugin, client, platform| {
            let program = test_program(true, true);
            let options = test_compilation_options();
            let executable = client.compile(&program, &options);
            match platform {
                TestPlatform::Cpu => assert!(executable.is_err()),
                _ => {
                    let executable = executable.unwrap();
                    let device = client.addressable_devices().unwrap()[0].clone();

                    // Create the input tensor: `[7, -1]`.
                    let mut lhs_bytes = Vec::with_capacity(8);
                    lhs_bytes.extend_from_slice(&7i32.to_ne_bytes());
                    lhs_bytes.extend_from_slice(&(-1i32).to_ne_bytes());
                    let lhs_bytes = lhs_bytes.as_slice();
                    let lhs_buffer =
                        client.buffer(lhs_bytes, BufferType::I32, &[2, 1], None, device.clone(), None).unwrap();

                    // Create a _send_ callback that records the values observed from the device.
                    let observed_value = Arc::new(Mutex::new(None));
                    let observed_value_clone = observed_value.clone();
                    let mut collected_bytes = Vec::new();
                    let send_callback = client.send_callback(2, move |chunk, total_size, done| {
                        if collected_bytes.is_empty() && total_size > 0 {
                            collected_bytes.reserve(total_size);
                        }
                        collected_bytes.extend_from_slice(chunk.data());
                        if done && collected_bytes.len() >= 8 {
                            let value = [
                                i32::from_ne_bytes(collected_bytes[0..4].try_into().unwrap()),
                                i32::from_ne_bytes(collected_bytes[4..8].try_into().unwrap()),
                            ];
                            *observed_value_clone.lock().unwrap() = Some(value);
                        }
                        Ok(())
                    });

                    // Create a _receive_ callback that streams `[35, -41]` to the device in two chunks.
                    let receive_callback = client.receive_callback(1, move |stream| {
                        let total_byte_count = stream.total_byte_count().unwrap();
                        let granule_byte_count = stream.granule_byte_count().unwrap();
                        assert_eq!(total_byte_count, 8);
                        assert_eq!(total_byte_count % granule_byte_count, 0);

                        unsafe extern "C" fn drop_chunk(_: *mut std::ffi::c_void, arg: *mut std::ffi::c_void) {
                            unsafe {
                                drop(Box::from_raw(arg as *mut [u8; 4]));
                            }
                        }

                        let mut first_chunk_bytes = Box::new(35i32.to_ne_bytes());
                        let mut first_chunk = crate::transfers::ffi::PJRT_Chunk {
                            data: first_chunk_bytes.as_mut_ptr().cast(),
                            size: first_chunk_bytes.len(),
                            deleter: Some(drop_chunk),
                            deleter_arg: Box::into_raw(first_chunk_bytes).cast(),
                        };
                        let transfer_complete_event =
                            stream.add_chunk(unsafe { Chunk::from_c_api(&mut first_chunk) }.unwrap());
                        assert!(transfer_complete_event.is_ok());

                        let mut second_chunk_bytes = Box::new((-41i32).to_ne_bytes());
                        let mut second_chunk = crate::transfers::ffi::PJRT_Chunk {
                            data: second_chunk_bytes.as_mut_ptr().cast(),
                            size: second_chunk_bytes.len(),
                            deleter: Some(drop_chunk),
                            deleter_arg: Box::into_raw(second_chunk_bytes).cast(),
                        };
                        let transfer_complete =
                            stream.add_chunk(unsafe { Chunk::from_c_api(&mut second_chunk) }.unwrap()).unwrap();
                        transfer_complete.r#await().unwrap();
                    });

                    // Construct the execution device inputs that consist of our input tensor and callbacks.
                    let inputs = ExecutionDeviceInputs {
                        inputs: &[ExecutionInput { buffer: lhs_buffer, donatable: false }],
                        send_callbacks: vec![send_callback],
                        receive_callbacks: vec![receive_callback],
                        ..Default::default()
                    };

                    // Execute the test program.
                    let mut outputs = executable.execute(vec![inputs], 0, None, None, None, None).unwrap();
                    assert_eq!(outputs.len(), 1);
                    let mut outputs = outputs.remove(0);

                    // Wait for the asynchronous execution to complete.
                    outputs.done.r#await().unwrap();
                    let output = outputs.outputs.remove(0);

                    // Copy the contents of the output buffer to the host.
                    let output_bytes = output.copy_to_host(None).unwrap().r#await().unwrap();

                    // Assert that the output buffer contains the expected values.
                    let mut expected_output_bytes = Vec::with_capacity(8);
                    expected_output_bytes.extend_from_slice(&42i32.to_ne_bytes());
                    expected_output_bytes.extend_from_slice(&(-42i32).to_ne_bytes());
                    assert_eq!(output_bytes, expected_output_bytes);

                    // Assert that the _send_ callback observed the expected values.
                    assert_eq!(*observed_value.lock().unwrap(), Some([42i32, -42i32]));
                }
            };
        });
    }
}
