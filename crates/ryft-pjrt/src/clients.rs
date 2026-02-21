use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, OnceLock};
use std::time::Duration;

use crate::{
    Api, Device, DeviceAssignment, DeviceId, Error, KeyValueStore, LocalHardwareId, Memory, NamedValue, Value,
    invoke_pjrt_api_error_fn, slice_from_c_api, str_from_c_api,
};

/// Serializes PJRT client lifecycle operations (i.e., creation and destruction) across threads. This is necessary
/// because some PJRT plugin implementations (e.g., the Metal plugin) can fail fatally when client creation and/or
/// destruction race during backend registration and/or teardown.
static PJRT_CLIENT_LIFECYCLE_GUARD: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// PJRT [`Client`]s represent a connection to an accelerator platform. They hold the topology of the system,
/// managing a list of [`Device`]s and their associated [`Memory`]s (a single device may have multiple memory spaces
/// like _High Bandwidth Memory_ and slower _Capacity Memory_, for example). Furthermore, while [`Buffer`]s are
/// associated with (and placed on) [`Device`]s, their lifecycle management is orchestrated through [`Client`]s to
/// ensure thread safety and correct resource allocation.
///
/// Note that a client can optionally use a [`KeyValueStore`] to support multi-process and/or multi-host platforms
/// (and it has a lifetime parameter, `'s`, that corresponds to that [`KeyValueStore`] since it must outlive the
/// client).
pub struct Client<'s> {
    /// Handle that represents this [`Client`] in the PJRT C API.
    handle: *mut ffi::PJRT_Client,

    /// Underlying PJRT [`Api`].
    api: Api,

    /// Cached attributes of the underlying [`Api`] so that they will only be constructed once.
    attributes: OnceLock<Result<HashMap<String, Value>, Error>>,

    /// Underlying [`KeyValueStore`]. Note that if this is [`None`] then this [`Client`] does not have a direct way
    /// to interact with other [`Client`]s and can thus not be used in a multi-process and/or multi-host platform.
    key_value_store: Option<&'s dyn KeyValueStore>,
}

impl<'s> Client<'s> {
    /// Constructs a new [`Client`] from the provided [`PJRT_Client`](ffi::PJRT_Client) handle that came
    /// from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(
        handle: *mut ffi::PJRT_Client,
        api: Api,
        key_value_store: Option<&'s dyn KeyValueStore>,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT client handle is a null pointer"))
        } else {
            Ok(Self { handle, api, attributes: OnceLock::new(), key_value_store })
        }
    }

    /// Returns the [`PJRT_Client`](ffi::PJRT_Client) that corresponds to this [`Client`] and which can
    /// be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_Client {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }

    /// [`Value`] of the attribute with the provided `name` attached to this PJRT [`Client`],
    /// or [`Error::NotFound`] if no such attribute is attached to this [`Client`].
    pub fn attribute<N: AsRef<str>>(&self, name: N) -> Result<Value, Error> {
        let name = name.as_ref();
        self.attributes()?
            .get(name)
            .cloned()
            .ok_or_else(|| Error::not_found(format!("no attribute named '{name}' found in this PJRT client")))
    }

    /// Returns the PJRT attributes associated with this [`Client`] (e.g., the version of the XLA compiler that it was
    /// compiled against). The specific attributes returned depend on the PJRT [`Plugin`](crate::Plugin) implementation.
    pub fn attributes(&self) -> Result<&HashMap<String, Value>, Error> {
        self.attributes.get_or_init(|| self.api().attributes()).as_ref().map_err(|error| error.clone())
    }

    /// Returns the [`KeyValueStore`] that this [`Client`] has access to.
    pub fn key_value_store(&self) -> Option<&dyn KeyValueStore> {
        self.key_value_store
    }

    /// Returns a string that identifies the platform of this [`Client`] (e.g., `"cpu"`, `"gpu"`, `"tpu"`, etc.).
    pub fn platform_name(&'_ self) -> Result<Cow<'_, str>, Error> {
        use ffi::PJRT_Client_PlatformName_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_PlatformName,
            { client = self.to_c_api() },
            { platform_name, platform_name_size },
        )
        .map(|(string, string_len)| str_from_c_api(string, string_len))
    }

    /// Returns a string that contains human-readable, platform-specific, version information for this [`Client`]
    /// (e.g., the CUDA version for GPU clients or the `libtpu` version for TPU clients).
    pub fn platform_version(&'_ self) -> Result<Cow<'_, str>, Error> {
        use ffi::PJRT_Client_PlatformVersion_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_PlatformVersion,
            { client = self.to_c_api() },
            { platform_version, platform_version_size },
        )
        .map(|(string, string_len)| str_from_c_api(string, string_len))
    }

    /// Process index of this [`Client`]. This is always `0` in single-process settings.
    pub fn process_index(&self) -> Result<ProcessIndex, Error> {
        use ffi::PJRT_Client_ProcessIndex_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Client_ProcessIndex, { client = self.to_c_api() }, { process_index })
            .map(|id| id as usize)
    }

    /// Updates this [`Client`] with information about all processes participating in a distributed program. In a
    /// distributed setup (e.g., a multi-node GPU cluster or a multi-host TPU slice), this function is the _handshake_
    /// mechanism that allows a local PJRT [`Client`] to understand its place within the larger cluster. The "right" way
    /// to use it involves coordination with your cluster manager (e.g., Slurm, Kubernetes, etc.). Specifically, when
    /// you initialize a [`Client`], it often only knows about the hardware physically attached to the current machine.
    /// However, for [collective operations](https://en.wikipedia.org/wiki/Collective_operation) the client needs to
    /// know how many other processes exist in the distributed program, what the ID of this process is among them,
    /// and what [`ProcessState`] each process is in. This function is used to provide that information.
    ///
    /// Note that this function **is not** used to communicate information about the IP address and the port each
    /// process listens to. That is because that information is potentially backend-dependent. Instead, the
    /// [`KeyValueStore`] associated with this [`Client`] is that is used to communicate that information when the
    /// processes are initialized.
    pub fn update_global_process_information(&self, processes: &[ProcessInformation]) -> Result<(), Error> {
        use ffi::PJRT_Client_UpdateGlobalProcessInfo_Args;

        // We extract the process information that we need, but we also pull out the error messages and collect them
        // into a [`Vec`]. That is because they need to remain alive for the duration of the lifetime of the process
        // information as they represent the backing storage for the error messages that are referenced from there.
        let (process_information, _): (Vec<_>, Vec<_>) = processes
            .iter()
            .map(|information| {
                let ProcessInformationHandle { process_information, error_message } = information.to_c_api();
                (process_information, error_message)
            })
            .unzip();

        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_UpdateGlobalProcessInfo,
            {
                client = self.to_c_api(),
                process_infos = process_information.as_ptr() as *mut _,
                num_process_infos = process_information.len(),
            },
        )
    }

    /// Results a [`Vec`] containing all [`Device`]s that are visible to this [`Client`],
    /// including both _addressable_ and _non-addressable_ devices.
    pub fn devices(&'_ self) -> Result<Vec<Device<'_>>, Error> {
        use ffi::PJRT_Client_Devices_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_Devices,
            { client = self.to_c_api() },
            { devices, num_devices },
        )
        .and_then(|(devices, devices_count)| {
            unsafe { slice_from_c_api(devices, devices_count) }
                .iter()
                .map(|handle| unsafe { Device::from_c_api(*handle, self.api()) })
                .collect::<Result<Vec<_>, _>>()
        })
    }

    /// Returns a [`Device`] with the provided [`DeviceId`] if it is visible by this [`Client`]
    /// and an [`Error::NotFound`] otherwise.
    pub fn lookup_device(&'_ self, id: DeviceId) -> Result<Device<'_>, Error> {
        use ffi::PJRT_Client_LookupDevice_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_LookupDevice,
            { client = self.to_c_api(), id = id as std::ffi::c_int },
            { device },
        )
        .and_then(|handle| {
            if handle.is_null() {
                Err(Error::not_found(format!("device with ID '{id}' not found")))
            } else {
                unsafe { Device::from_c_api(handle, self.api()) }
            }
        })
    }

    /// Results a [`Vec`] containing all [`Device`]s that are _addressable_ from this [`Client`]
    /// (i.e., devices that this client can issue commands to). Note that all visible devices are
    /// addressable in a single-process environment.
    pub fn addressable_devices(&'_ self) -> Result<Vec<Device<'_>>, Error> {
        use ffi::PJRT_Client_AddressableDevices_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_AddressableDevices,
            { client = self.to_c_api() },
            { addressable_devices, num_addressable_devices },
        )
        .and_then(|(devices, devices_count)| {
            unsafe { slice_from_c_api(devices, devices_count) }
                .iter()
                .map(|handle| unsafe { Device::from_c_api(*handle, self.api()) })
                .collect::<Result<Vec<_>, _>>()
        })
    }

    /// Returns a [`Device`] with the provided [`LocalHardwareId`] if it is addressable by this [`Client`]
    /// and an [`Error::NotFound`] otherwise.
    pub fn lookup_addressable_device(&'_ self, id: LocalHardwareId) -> Result<Device<'_>, Error> {
        use ffi::PJRT_Client_LookupAddressableDevice_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_LookupAddressableDevice,
            { client = self.to_c_api(), id = id as std::ffi::c_int },
            { addressable_device },
        )
        .and_then(|handle| {
            if handle.is_null() {
                Err(Error::not_found(format!("device with ID '{id}' not found in addressable devices")))
            } else {
                unsafe { Device::from_c_api(handle, self.api()) }
            }
        })
    }

    /// Returns a [`Vec`] containing all [`Memory`]s that are _addressable_ from this [`Client`]. Addressable memories
    /// are those that the client can directly transfer data to and from. Note that all visible memories are addressable
    /// in a single-process environment.
    pub fn addressable_memories(&'_ self) -> Result<Vec<Memory<'_>>, Error> {
        use ffi::PJRT_Client_AddressableMemories_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_AddressableMemories,
            { client = self.to_c_api() },
            { addressable_memories, num_addressable_memories },
        )
        .and_then(|(memories, memories_count)| {
            unsafe { slice_from_c_api(memories, memories_count) }
                .iter()
                .map(|handle| unsafe { Memory::from_c_api(*handle, self.api()) })
                .collect::<Result<Vec<_>, _>>()
        })
    }

    /// Returns the default [`DeviceAssignment`] that should be used for this [`Client`],
    /// given the provided number of replicas and computations.
    pub fn default_device_assignment(
        &self,
        replica_count: usize,
        computation_count: usize,
    ) -> Result<DeviceAssignment, Error> {
        use ffi::PJRT_Client_DefaultDeviceAssignment_Args;
        let mut assignment = Vec::with_capacity(replica_count * computation_count);
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_DefaultDeviceAssignment,
            {
                client = self.to_c_api(),
                num_replicas = replica_count as std::ffi::c_int,
                num_partitions = computation_count as std::ffi::c_int,
                default_assignment_size = replica_count * computation_count,
                default_assignment = assignment.as_mut_ptr(),
            },
        )?;
        unsafe { assignment.set_len(replica_count * computation_count) }
        Ok(DeviceAssignment { replica_count, computation_count, assignment })
    }
}

unsafe impl Send for Client<'_> {}
unsafe impl Sync for Client<'_> {}

impl Drop for Client<'_> {
    fn drop(&mut self) {
        let _guard = PJRT_CLIENT_LIFECYCLE_GUARD.lock().unwrap();
        use ffi::PJRT_Client_Destroy_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Client_Destroy, { client = self.to_c_api() })
            .expect("failed to destroy PJRT client");
    }
}

/// Type alias used to represent process indices (i.e., in a multi-process or multi-host platform).
pub type ProcessIndex = usize;

/// Represents the state of a process in a distributed program.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ProcessState {
    /// Default, "zero" state of a process before it has been successfully configured or connected to the distributed
    /// runtime environment. In the context of distributed workloads (e.g., multi-host TPU or GPU clusters), this state
    /// is critical for lifecycle management, distinguishing between a process that has failed and one that simply has
    /// not started yet. The difference between this state and [`ProcessState::Disconnected`] is that the latter
    /// represents a state where a process was connected (i.e., initialized at some point) and got disconnected,
    /// indicating potentially a failure (as opposed to simply slow initialization, for example).
    ///
    /// Specifically, when a PJRT [`Client`] initializes, it typically performs a blocking wait or a "barrier"
    /// operation. It queries the global process state and:
    ///   - If it sees peers in the [`ProcessState::Uninitialized`] state, the client logic executes
    ///     a _backoff-and-retry_ loop. It understands that the cluster is forming.
    ///   - If it sees peers in the [`ProcessState::Disconnected`] state, the client logic triggers
    ///     a failure exception, potentially aborting the run.
    Uninitialized,

    /// State of a process which was previously valid is no longer reachable by the coordination service. A process can
    /// transition to this state via several pathways:
    ///   - **Heartbeat Timeout:** The coordination service relies on active keep-alive signals (heartbeats).
    ///     If a registered process fails to send a heartbeat within the configured heartbeat timeout, the coordinator
    ///     unilaterally demotes it to the [`ProcessState::Disconnected`] state.
    ///   - **Socket Closure:** If the TCP connection to the worker is reset (`RST` packet) or closed (`FIN` packet)
    ///     unexpectedly, the coordinator will demote the process to the [`ProcessState::Disconnected`] state.
    ///   - **Explicit Preemption:** In cloud environments (e.g., Google Cloud TPUs or AWS Trainium), the orchestration
    ///     layer (e.g., Kubernetes or Borg) may preempt a node, demoting the corresponding process to the
    ///     [`ProcessState::Disconnected`] state.
    Disconnected,

    /// Nominal operating state which signifies that the process has successfully completed the handshake with the
    /// coordination service and is actively maintaining its session.
    Connected,

    /// Represents a state where the process is technically reachable (i.e., the communication channel is open),
    /// but the process itself has reported an internal, non-recoverable failure that is represented by the [`Error`]
    /// stored in this state.
    Error { error: Error },
}

/// Contains information about a process participating in a distributed program.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProcessInformation {
    /// Global ID of the process among the list of all processes participating in a distributed program. This
    /// ID persists across restarts. For example, if process `1` crashes and the orchestration system spins up a
    /// replacement node, that replacement node must assume the identity `process_id = 1` to resume the workload.
    /// In this case, [`ProcessInformation::incarnation_id`] provides information about the specific _incarnation_
    /// of a process, separate from its global, persistent, ID.
    ///
    /// This is sometimes referred to as a _task_ ID.
    pub process_id: usize,

    /// Unique ID of the specific _incarnation_ of the process. This ID is typically generated upon process startup
    /// using a random number generator or a high-precision timestamp (e.g., nanoseconds since epoch). This ID enables
    /// the runtime to be aware of process restarts, separate from their identities.
    pub incarnation_id: usize,

    /// [`ProcessState`] of the process.
    pub state: ProcessState,
}

impl ProcessInformation {
    /// Returns the [`ProcessInformationHandle`] that corresponds to this [`ProcessInformation`] and which contains
    /// [`ProcessInformationHandle::process_information`] that can be passed to functions in the PJRT C API.
    pub(crate) fn to_c_api(&self) -> ProcessInformationHandle {
        let state = match &self.state {
            ProcessState::Uninitialized => ffi::PJRT_ProcessState_kUninitialized,
            ProcessState::Disconnected => ffi::PJRT_ProcessState_kDisconnected,
            ProcessState::Connected => ffi::PJRT_ProcessState_kConnected,
            ProcessState::Error { .. } => ffi::PJRT_ProcessState_kError,
        };
        let error_code = match &self.state {
            ProcessState::Error { error } => error.code(),
            _ => crate::errors::ffi::PJRT_Error_Code_OK,
        };
        let error_message = match &self.state {
            ProcessState::Error { error } => error.message().clone(),
            _ => std::ffi::CString::new("").unwrap(),
        };
        ProcessInformationHandle {
            process_information: ffi::PJRT_ProcessInfo {
                struct_size: size_of::<ffi::PJRT_ProcessInfo>(),
                task_id: self.process_id as std::ffi::c_int,
                incarnation_id: self.incarnation_id as u64,
                state,
                error_code: error_code as std::ffi::c_int,
                error_message: error_message.as_ptr(),
                error_message_size: error_message.count_bytes(),
            },
            error_message,
        }
    }
}

/// Wrapper around [`PJRT_ProcessInfo`](ffi::PJRT_ProcessInfo) that owns the backing string storage referenced by the
/// underlying C struct for storing the error message it contains (if it contains one).
pub(crate) struct ProcessInformationHandle {
    /// Backing [`PJRT_ProcessInfo`](ffi::PJRT_ProcessInfo) that can be passed to function the PJRT C API.
    process_information: ffi::PJRT_ProcessInfo,

    /// Backing storage for [`PJRT_ProcessInfo::error_message`](ffi::PJRT_ProcessInfo::error_message).
    error_message: std::ffi::CString,
}

/// Options that can be passed to [`Plugin::client`], and [`Plugin::client_with_key_value_store`]
/// to configure a [`Client`].
#[derive(Clone, Debug, PartialEq)]
pub enum ClientOptions {
    /// Options for configuring a CPU [`Client`] backed by [XLA](https://openxla.org/xla).
    CPU(CpuClientOptions),

    /// Options for configuring a GPU [`Client`] backed by [XLA](https://openxla.org/xla).
    GPU(GpuClientOptions),

    /// Options for configuring a [`Client`]. This represents a default and unstructured set of options that can be
    /// used to support arbitrary PJRT [`Plugin`]s.
    Other(HashMap<String, Value>),
}

impl ClientOptions {
    /// Returns a collection of [`NamedValue`]s that correspond to this [`ClientOptions`] instance.
    pub(crate) fn to_named_values(&self) -> Vec<NamedValue> {
        match self {
            Self::CPU(options) => options.to_named_values(),
            Self::GPU(options) => options.to_named_values(),
            Self::Other(options) => options.iter().map(|(name, value)| NamedValue::new(name, value.clone())).collect(),
        }
    }
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self::Other(HashMap::new())
    }
}

/// Options that can be used to configure a CPU [`Client`] backed by [XLA](https://openxla.org/xla). This is a
/// conveniently typed wrapper over a collection of [`NamedValue`]s to make configuring such CPU [`Client`]s
/// easier and type safe.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CpuClientOptions {
    /// Number of CPU devices to use that defaults to the number of logical CPUs on the host.
    /// This is intended for making debugging or testing multi-device functionality easier.
    pub device_count: Option<usize>,
}

impl CpuClientOptions {
    /// Returns a collection of [`NamedValue`]s that correspond to this [`CpuClientOptions`] instance.
    pub(crate) fn to_named_values(&self) -> Vec<NamedValue> {
        let mut values = Vec::new();
        if let Some(device_count) = self.device_count {
            values.push(NamedValue::new("cpu_device_count", device_count as i64));
        }
        values
    }
}

/// Options that can be used to configure a GPU [`Client`] backed by [XLA](https://openxla.org/xla). This is a
/// conveniently typed wrapper over a collection of [`NamedValue`]s to make configuring such GPU [`Client`]s
/// easier and type safe.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct GpuClientOptions {
    /// Optional GPU platform that the [`Client`] will use. If not provided, it will default to the default
    /// [`GpuPlatform`] of the [`Plugin`](crate::Plugin) that is being used.
    pub platform: Option<GpuPlatform>,

    /// Optional list of GPU IDs for the GPUs that will be visible to the [`Client`]. If not specified, then
    /// all GPUs on the current machine will be visible to the [`Client`]. This is equivalent to what e.g.,
    /// the `CUDA_VISIBLE_DEVICES` environment variable controls in CUDA environments.
    pub visible_devices: Option<Vec<usize>>,

    /// Node/machine ID in a distributed program. This is used to identify the current machine among all machines
    /// participating in the distributed program.
    pub node_id: Option<usize>,

    /// Total number of nodes/machines participating in a distributed program.
    pub node_count: Option<usize>,

    /// Partition index of the current process in a distributed program. This is useful for custom logical device
    /// groupings that differ from the physical process layout. Devices with the same partition index are connected
    /// by fast networking (e.g., [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) on NVIDIA GPUs). Note that
    /// fast-interconnect can also be cross-host, meaning that a partition may include devices on multiple hosts.
    pub partition_index: Option<usize>,

    /// Boolean value indicating whether to stage host-to-device transfers via pinned memory. When `true`, data
    /// transfers from the host (CPU) to the device (GPU) are performed asynchronously using a staging buffer.
    /// This is the default behavior on GPUs where pinned memory transfers are preferred.
    pub should_stage_host_to_device_transfers: bool,

    /// Memory allocator to use for GPU memory management. Refer to the documentation of [`GpuMemoryAllocator`]
    /// for the available options and their trade-offs.
    pub allocator: GpuMemoryAllocator,

    /// Optional size (in bytes) of the memory buffer reserved for [collective operations](
    /// https://en.wikipedia.org/wiki/Collective_operation) (e.g., all-reduce and all-gather operations)
    /// within the [XLA](https://openxla.org/xla) client. This helps manage GPU memory for distributed programs.
    pub collective_memory_size: Option<usize>,

    /// Boolean value indicating whether to cancel [collective operations](
    /// https://en.wikipedia.org/wiki/Collective_operation) when a participant fails. When `true`,
    /// collectives with a failed participant will be canceled to avoid getting stuck.
    pub abort_collectives_on_failure: bool,

    /// Boolean value indicating whether to use the [TFRT GPU client](
    /// https://github.com/openxla/xla/blob/5af63fc1c5ae4033172f6599135d948d9337812d/xla/pjrt/gpu/tfrt/tfrt_gpu_client.h#L115)
    /// instead of the [Stream Executor GPU client](
    /// https://github.com/openxla/xla/blob/5af63fc1c5ae4033172f6599135d948d9337812d/xla/pjrt/gpu/se_gpu_pjrt_client.h#L106)
    /// in [XLA](https://openxla.org/xla).
    pub use_tfrt_gpu_client: bool,

    /// Optional mock GPU topology configuration to use for simulating distributed programs on systems
    /// without the necessary hardware. If provided, this will also enable the simulation of e.g.,
    /// [NCCL](https://developer.nvidia.com/nccl) operations on NVIDIA platforms. This enables developers
    /// to test multi-GPU/multi-node communication logic locally without needing a full cluster setup.
    pub mock_gpu_topology: Option<MockGpuTopology>,
}

impl GpuClientOptions {
    /// Returns a collection of [`NamedValue`]s that correspond to this [`CpuClientOptions`] instance.
    pub(crate) fn to_named_values(&self) -> Vec<NamedValue> {
        let mut values = Vec::new();
        match self.platform {
            Some(GpuPlatform::CUDA) => values.push(NamedValue::new("platform_name", "cuda")),
            Some(GpuPlatform::ROCm) => values.push(NamedValue::new("platform_name", "ROCM")),
            None => {}
        }
        if let Some(visible_devices) = &self.visible_devices {
            values.push(NamedValue::new(
                "visible_devices",
                visible_devices.iter().map(|id| *id as i64).collect::<Vec<_>>(),
            ));
        }
        if let Some(node_id) = self.node_id {
            values.push(NamedValue::new("node_id", node_id as i64));
        }
        if let Some(node_count) = self.node_count {
            values.push(NamedValue::new("node_count", node_count as i64));
        }
        if let Some(partition_index) = self.partition_index {
            values.push(NamedValue::new("partition_index", partition_index as i64));
        }
        values
            .push(NamedValue::new("should_stage_host_to_device_transfers", self.should_stage_host_to_device_transfers));
        match self.allocator {
            GpuMemoryAllocator::Platform => {
                values.push(NamedValue::new("preallocate", false));
                values.push(NamedValue::new("allocator", "platform"));
            }
            GpuMemoryAllocator::BFC { memory_fraction_to_preallocate } => {
                values.push(NamedValue::new("preallocate", true));
                values.push(NamedValue::new("memory_fraction", memory_fraction_to_preallocate));
                values.push(NamedValue::new("allocator", "bfc"));
            }
            GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None } => {
                values.push(NamedValue::new("preallocate", false));
                values.push(NamedValue::new("allocator", "cuda_async"));
            }
            GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: Some(memory_fraction_to_preallocate) } => {
                values.push(NamedValue::new("preallocate", true));
                values.push(NamedValue::new("memory_fraction", memory_fraction_to_preallocate));
                values.push(NamedValue::new("allocator", "cuda_async"));
            }
        }
        if let Some(collective_memory_size) = self.collective_memory_size {
            values.push(NamedValue::new("collective_memory_size", collective_memory_size as i64));
        }
        values.push(NamedValue::new("abort_collectives_on_failure", self.abort_collectives_on_failure));
        values.push(NamedValue::new("use_tfrt_gpu_client", self.use_tfrt_gpu_client));
        if let Some(mock_gpu_topology) = &self.mock_gpu_topology {
            values.push(NamedValue::new("enable_mock_nccl", true));
            values.push(NamedValue::new(
                "mock_gpu_topology",
                format!(
                    "{}x{}x{}",
                    mock_gpu_topology.partition_count,
                    mock_gpu_topology.host_count_per_partition,
                    mock_gpu_topology.device_count_per_host
                ),
            ));
        }
        values
    }
}

impl Default for GpuClientOptions {
    fn default() -> Self {
        Self {
            platform: None,
            visible_devices: None,
            node_id: None,
            node_count: None,
            partition_index: None,
            should_stage_host_to_device_transfers: true,
            allocator: GpuMemoryAllocator::default(),
            collective_memory_size: None,
            abort_collectives_on_failure: false,
            use_tfrt_gpu_client: false,
            mock_gpu_topology: None,
        }
    }
}

/// Represents a specific GPU platform.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuPlatform {
    /// [NVIDIA CUDA](https://developer.nvidia.com/cuda/toolkit) platform.
    CUDA,

    /// [AMD ROCm](https://rocmdocs.amd.com/en/latest/index.html) platform.
    ROCm,
}

/// Memory allocator for GPU [`Client`]s that are backed by [XLA](https://openxla.org/xla). Each allocator offers
/// different trade-offs between memory efficiency, allocation speed, and fragmentation behavior.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum GpuMemoryAllocator {
    /// Platform allocator that allocates exactly what is needed on demand and deallocates memory that is no longer
    /// needed. This is the only [`GpuMemoryAllocator`] that will deallocate GPU memory instead of reusing it. While
    /// this provides the minimal possible GPU memory footprint, it is very slow and not recommended for general use.
    /// It can be useful for debugging OOM failures.
    Platform,

    /// Best-Fit with Coalescing (BFC) allocator that uses a best-fit allocation strategy with memory coalescing to
    /// reduce fragmentation.
    BFC {
        /// Fraction of total GPU memory to preallocate when the first operation is run. Preallocating minimizes
        /// allocation overhead and memory fragmentation, but can sometimes cause out-of-memory (OOM) errors.
        /// Lowering this value can help avoid out-of-memory (OOM) errors that occur when the first operation is run.
        memory_fraction_to_preallocate: f32,
    },

    /// [`GpuMemoryAllocator`] that uses [`cudaMallocAsync`](
    /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html) under the hood.
    /// This removes the large fixed preallocation and instead uses a memory pool that grows dynamically.
    /// Note that this [`GpuMemoryAllocator`] is only available on the [`GpuPlatform::CUDA`] platform.
    ///
    /// Compared to [`GpuMemoryAllocator::BFC`], this [`GpuMemoryAllocator`] differs in the following ways:
    ///
    ///   - Memory fragmentation patterns differ and so out-of-memory (OOM) behavior near memory limits may change.
    ///   - Allocation time is not paid upfront but incurred when the memory pool needs to grow, potentially
    ///     causing less speed stability when the first operation is run. When benchmarking, it is important
    ///     to ignore the first few iterations.
    ///
    /// The risks can be mitigated by preallocating a significant chunk and still getting the benefit of having
    /// a growing memory pool.
    CudaAsync {
        /// Optional fraction of total GPU memory to preallocate when the first operation is run. Preallocating
        /// minimizes allocation overhead and memory fragmentation, but can sometimes cause out-of-memory (OOM) errors.
        /// Lowering this value can help avoid out-of-memory (OOM) errors that occur when the first operation is run.
        memory_fraction_to_preallocate: Option<f32>,
    },
}

impl Default for GpuMemoryAllocator {
    fn default() -> Self {
        Self::BFC { memory_fraction_to_preallocate: 0.75 }
    }
}

/// Configuration for a mock GPU topology to use for simulating distributed programs on systems
/// without the necessary hardware.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MockGpuTopology {
    /// Number of partitions in the simulated topology. Partitions represent groups of hosts connected
    /// by fast interconnect (e.g., [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) on NVIDIA GPUs).
    pub partition_count: usize,

    /// Number of hosts (i.e., machines) within each partition.
    pub host_count_per_partition: usize,

    /// Number of GPU devices on each host (i.e., machine).
    pub device_count_per_host: usize,
}

impl Api {
    /// Constructs a new PJRT [`Client`] using the provided (optional) platform-specific [`ClientOptions`].
    ///
    /// Note that the resulting [`Client`] will not have access to a [`KeyValueStore`] and thus will have no direct way
    /// to interact with other [`Client`]s. Refer to [`Api::client_with_key_value_store`] for more information.
    pub(crate) fn client(&self, options: ClientOptions) -> Result<Client<'static>, Error> {
        let _guard = PJRT_CLIENT_LIFECYCLE_GUARD.lock().unwrap();
        use ffi::PJRT_Client_Create_Args;
        let options = options.to_named_values();
        let options = options.iter().map(|option| unsafe { option.to_c_api() }).collect::<Vec<_>>();
        invoke_pjrt_api_error_fn!(
            *self,
            PJRT_Client_Create,
            {
                create_options = options.as_slice().as_ptr(),
                num_options = options.len(),
                kv_get_callback = None,
                kv_get_user_arg = std::ptr::null_mut(),
                kv_put_callback = None,
                kv_put_user_arg = std::ptr::null_mut(),
                kv_try_get_callback = None,
                kv_try_get_user_arg = std::ptr::null_mut(),
            },
            { client },
        )
        .and_then(|handle| unsafe { Client::from_c_api(handle, *self, None) })
    }

    /// Constructs a new PJRT [`Client`] using the provided (optional) platform-specific [`ClientOptions`] and
    /// [`KeyValueStore`]. The provided [`KeyValueStore`] must be accessible across multiple hosts and/or processes.
    /// Access to this [`KeyValueStore`] may be necessary to create certain kinds of multi-process or multi-host
    /// environments as it enables [`Client`]s (potentially on different machines) to communicate with each other.
    pub(crate) fn client_with_key_value_store<'s, Store: KeyValueStore>(
        &self,
        options: ClientOptions,
        key_value_store: &'s Store,
    ) -> Result<Client<'s>, Error> {
        let _guard = PJRT_CLIENT_LIFECYCLE_GUARD.lock().unwrap();
        use crate::errors::ffi::PJRT_Error;
        use ffi::PJRT_Client_Create_Args;

        /// Opaque value payload that can be passed to PJRT C API functions, paired with a function that can be used
        /// to drop it. Note that the memory layout of `value` is a little weird. Specifically, there is a `usize`
        /// immediately preceding the point at which `value` is pointing that contains `value_size`. That is used
        /// by `drop_fn` to determine how big of a memory region to drop. That is because the PJRT C API does not
        /// allow us to pass a second argument to `drop_fn` with additional information.
        struct CApiValue {
            value: *mut std::ffi::c_char,
            value_size: usize,
            drop_fn: unsafe extern "C" fn(value: *mut std::ffi::c_char),
        }

        impl CApiValue {
            /// Converts a Rust-owned `Vec<u8>` into a [`CApiValue`] that can be passed to PJRT C API functions.
            /// The PJRT C API callback ABI expects a pointer to the value bytes, the number of bytes in the value,
            /// and a deleter callback that only receives the value pointer. To make deallocation possible from that
            /// pointer alone, we allocate a single boxed byte slice with this layout:
            /// `[value_size (native-endian usize)][value bytes...]`
            /// and return a pointer to the payload region. The allocation is intentionally leaked here and reclaimed by
            /// the deleter callback (i.e., [`CApiValue::drop_fn`]).
            pub(crate) fn new(value: Vec<u8>) -> Self {
                unsafe extern "C" fn delete_value(value: *mut std::ffi::c_char) {
                    unsafe {
                        let header_size = size_of::<usize>();
                        let allocation_ptr = (value as *mut u8).sub(header_size);
                        let mut value_size_bytes = [0u8; size_of::<usize>()];
                        std::ptr::copy_nonoverlapping(
                            allocation_ptr as *const u8,
                            value_size_bytes.as_mut_ptr(),
                            header_size,
                        );
                        let value_size = usize::from_ne_bytes(value_size_bytes);
                        let allocation_ptr =
                            std::ptr::slice_from_raw_parts_mut(allocation_ptr, header_size + value_size);
                        drop(Box::from_raw(allocation_ptr));
                    }
                }

                let header_size = size_of::<usize>();
                let value_size = value.len();
                let mut allocation = vec![0u8; header_size + value_size].into_boxed_slice();
                allocation[..header_size].copy_from_slice(&value_size.to_ne_bytes());
                allocation[header_size..].copy_from_slice(&value);
                let allocation_ptr = allocation.as_mut_ptr();
                std::mem::forget(allocation);
                Self {
                    value: unsafe { allocation_ptr.add(header_size) as *mut std::ffi::c_char },
                    value_size,
                    drop_fn: delete_value,
                }
            }
        }

        unsafe extern "C" fn put<KVS: KeyValueStore>(args: *mut ffi::PJRT_KeyValuePutCallback_Args) -> *mut PJRT_Error {
            unsafe {
                let store = ((*args).user_arg as *const KVS).as_ref().expect("invalid PJRT key-value store");
                let key = slice_from_c_api((*args).key as *const u8, (*args).key_size);
                let value = slice_from_c_api((*args).value as *const u8, (*args).value_size);
                match store.put(key, value) {
                    Ok(_) => std::ptr::null_mut(),
                    Err(error) => error.to_c_api((*args).callback_error) as *mut _,
                }
            }
        }

        unsafe extern "C" fn get<KVS: KeyValueStore>(args: *mut ffi::PJRT_KeyValueGetCallback_Args) -> *mut PJRT_Error {
            unsafe {
                let store = ((*args).user_arg as *const KVS).as_ref().expect("invalid PJRT key-value store");
                let key = slice_from_c_api((*args).key as *const u8, (*args).key_size);
                match store.get(key, Duration::from_millis((*args).timeout_in_ms as u64)) {
                    Ok(value) => {
                        let value = CApiValue::new(value);
                        (*args).value = value.value;
                        (*args).value_size = value.value_size;
                        (*args).value_deleter_callback = value.drop_fn;
                        std::ptr::null_mut()
                    }
                    Err(error) => error.to_c_api((*args).callback_error) as *mut _,
                }
            }
        }

        unsafe extern "C" fn try_get<KVS: KeyValueStore>(
            args: *mut ffi::PJRT_KeyValueTryGetCallback_Args,
        ) -> *mut PJRT_Error {
            unsafe {
                let store = ((*args).user_arg as *const KVS).as_ref().expect("invalid PJRT key-value store");
                let key = slice_from_c_api((*args).key as *const u8, (*args).key_size);
                match store.try_get(key) {
                    Ok(value) => {
                        let value = CApiValue::new(value);
                        (*args).value = value.value;
                        (*args).value_size = value.value_size;
                        (*args).value_deleter_callback = value.drop_fn;
                        std::ptr::null_mut()
                    }
                    Err(error) => error.to_c_api((*args).callback_error) as *mut _,
                }
            }
        }

        let options = options.to_named_values();
        let options = options.iter().map(|option| unsafe { option.to_c_api() }).collect::<Vec<_>>();
        invoke_pjrt_api_error_fn!(
            *self,
            PJRT_Client_Create,
            {
                create_options = options.as_slice().as_ptr(),
                num_options = options.len(),
                kv_get_callback = Some(get::<Store>),
                kv_get_user_arg = key_value_store as *const _ as *mut _,
                kv_put_callback = Some(put::<Store>),
                kv_put_user_arg = key_value_store as *const _ as *mut _,
                kv_try_get_callback = Some(try_get::<Store>),
                kv_try_get_user_arg = key_value_store as *const _ as *mut _,
            },
            { client },
        )
        .and_then(|handle| unsafe { Client::from_c_api(handle, *self, Some(key_value_store)) })
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::devices::ffi::PJRT_Device;
    use crate::errors::ffi::{PJRT_CallbackError, PJRT_Error};
    use crate::ffi::PJRT_Extension_Base;
    use crate::memories::ffi::PJRT_Memory;
    use crate::values::ffi::PJRT_NamedValue;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_Client {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    pub type PJRT_KeyValueGetCallback_ValueDeleter = unsafe extern "C" fn(value: *mut std::ffi::c_char);

    #[repr(C)]
    pub struct PJRT_KeyValueGetCallback_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub key: *const std::ffi::c_char,
        pub key_size: usize,
        pub timeout_in_ms: std::ffi::c_int,
        pub callback_error: *mut PJRT_CallbackError,
        pub user_arg: *mut std::ffi::c_void,
        pub value: *mut std::ffi::c_char,
        pub value_size: usize,
        pub value_deleter_callback: PJRT_KeyValueGetCallback_ValueDeleter,
    }

    impl PJRT_KeyValueGetCallback_Args {
        pub fn new(
            key: *const std::ffi::c_char,
            key_size: usize,
            timeout_in_ms: std::ffi::c_int,
            callback_error: *mut PJRT_CallbackError,
            user_arg: *mut std::ffi::c_void,
            value_deleter_callback: PJRT_KeyValueGetCallback_ValueDeleter,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                key,
                key_size,
                timeout_in_ms,
                callback_error,
                user_arg,
                value: std::ptr::null_mut(),
                value_size: 0,
                value_deleter_callback,
            }
        }
    }

    pub type PJRT_KeyValueGetCallback =
        unsafe extern "C" fn(args: *mut PJRT_KeyValueGetCallback_Args) -> *mut PJRT_Error;

    pub type PJRT_KeyValueTryGetCallback_ValueDeleter = unsafe extern "C" fn(value: *mut std::ffi::c_char);

    #[repr(C)]
    pub struct PJRT_KeyValueTryGetCallback_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub key: *const std::ffi::c_char,
        pub key_size: usize,
        pub callback_error: *mut PJRT_CallbackError,
        pub user_arg: *mut std::ffi::c_void,
        pub value: *mut std::ffi::c_char,
        pub value_size: usize,
        pub value_deleter_callback: PJRT_KeyValueTryGetCallback_ValueDeleter,
    }

    impl PJRT_KeyValueTryGetCallback_Args {
        pub fn new(
            key: *const std::ffi::c_char,
            key_size: usize,
            callback_error: *mut PJRT_CallbackError,
            user_arg: *mut std::ffi::c_void,
            value_deleter_callback: PJRT_KeyValueTryGetCallback_ValueDeleter,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                key,
                key_size,
                callback_error,
                user_arg,
                value: std::ptr::null_mut(),
                value_size: 0,
                value_deleter_callback,
            }
        }
    }

    pub type PJRT_KeyValueTryGetCallback =
        unsafe extern "C" fn(args: *mut PJRT_KeyValueTryGetCallback_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_KeyValuePutCallback_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub key: *const std::ffi::c_char,
        pub key_size: usize,
        pub value: *const std::ffi::c_char,
        pub value_size: usize,
        pub callback_error: *mut PJRT_CallbackError,
        pub user_arg: *mut std::ffi::c_void,
    }

    impl PJRT_KeyValuePutCallback_Args {
        pub fn new(
            key: *const std::ffi::c_char,
            key_size: usize,
            value: *const std::ffi::c_char,
            value_size: usize,
            callback_error: *mut PJRT_CallbackError,
            user_arg: *mut std::ffi::c_void,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                key,
                key_size,
                value,
                value_size,
                callback_error,
                user_arg,
            }
        }
    }

    pub type PJRT_KeyValuePutCallback =
        unsafe extern "C" fn(args: *mut PJRT_KeyValuePutCallback_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_Create_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub create_options: *const PJRT_NamedValue,
        pub num_options: usize,
        pub kv_get_callback: Option<PJRT_KeyValueGetCallback>,
        pub kv_get_user_arg: *mut std::ffi::c_void,
        pub kv_put_callback: Option<PJRT_KeyValuePutCallback>,
        pub kv_put_user_arg: *mut std::ffi::c_void,
        pub client: *mut PJRT_Client,
        pub kv_try_get_callback: Option<PJRT_KeyValueTryGetCallback>,
        pub kv_try_get_user_arg: *mut std::ffi::c_void,
    }

    impl PJRT_Client_Create_Args {
        pub fn new(
            create_options: *const PJRT_NamedValue,
            num_options: usize,
            kv_get_callback: Option<PJRT_KeyValueGetCallback>,
            kv_get_user_arg: *mut std::ffi::c_void,
            kv_put_callback: Option<PJRT_KeyValuePutCallback>,
            kv_put_user_arg: *mut std::ffi::c_void,
            kv_try_get_callback: Option<PJRT_KeyValueTryGetCallback>,
            kv_try_get_user_arg: *mut std::ffi::c_void,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                create_options,
                num_options,
                kv_get_callback,
                kv_get_user_arg,
                kv_put_callback,
                kv_put_user_arg,
                client: std::ptr::null_mut(),
                kv_try_get_callback,
                kv_try_get_user_arg,
            }
        }
    }

    pub type PJRT_Client_Create = unsafe extern "C" fn(args: *mut PJRT_Client_Create_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_PlatformName_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub platform_name: *const std::ffi::c_char,
        pub platform_name_size: usize,
    }

    impl PJRT_Client_PlatformName_Args {
        pub fn new(client: *mut PJRT_Client) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                platform_name: std::ptr::null_mut(),
                platform_name_size: 0,
            }
        }
    }

    pub type PJRT_Client_PlatformName =
        unsafe extern "C" fn(args: *mut PJRT_Client_PlatformName_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_PlatformVersion_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub platform_version: *const std::ffi::c_char,
        pub platform_version_size: usize,
    }

    impl PJRT_Client_PlatformVersion_Args {
        pub fn new(client: *mut PJRT_Client) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                platform_version: std::ptr::null_mut(),
                platform_version_size: 0,
            }
        }
    }

    pub type PJRT_Client_PlatformVersion =
        unsafe extern "C" fn(args: *mut PJRT_Client_PlatformVersion_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_ProcessIndex_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub process_index: std::ffi::c_int,
    }

    impl PJRT_Client_ProcessIndex_Args {
        pub fn new(client: *mut PJRT_Client) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), client, process_index: 0 }
        }
    }

    pub type PJRT_Client_ProcessIndex =
        unsafe extern "C" fn(args: *mut PJRT_Client_ProcessIndex_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
    }

    impl PJRT_Client_Destroy_Args {
        pub fn new(client: *mut PJRT_Client) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), client }
        }
    }

    pub type PJRT_Client_Destroy = unsafe extern "C" fn(args: *mut PJRT_Client_Destroy_Args) -> *mut PJRT_Error;

    pub type PJRT_ProcessState = std::ffi::c_uint;
    pub const PJRT_ProcessState_kUnspecified: PJRT_ProcessState = 0;
    pub const PJRT_ProcessState_kUninitialized: PJRT_ProcessState = 1;
    pub const PJRT_ProcessState_kDisconnected: PJRT_ProcessState = 2;
    pub const PJRT_ProcessState_kConnected: PJRT_ProcessState = 3;
    pub const PJRT_ProcessState_kError: PJRT_ProcessState = 4;

    #[repr(C)]
    pub struct PJRT_ProcessInfo {
        pub struct_size: usize,
        pub task_id: std::ffi::c_int,
        pub incarnation_id: u64,
        pub state: PJRT_ProcessState,
        pub error_code: std::ffi::c_int,
        pub error_message: *const std::ffi::c_char,
        pub error_message_size: usize,
    }

    #[repr(C)]
    pub struct PJRT_Client_UpdateGlobalProcessInfo_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub process_infos: *mut PJRT_ProcessInfo,
        pub num_process_infos: usize,
    }

    impl PJRT_Client_UpdateGlobalProcessInfo_Args {
        pub fn new(client: *mut PJRT_Client, process_infos: *mut PJRT_ProcessInfo, num_process_infos: usize) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                process_infos,
                num_process_infos,
            }
        }
    }

    pub type PJRT_Client_UpdateGlobalProcessInfo =
        unsafe extern "C" fn(args: *mut PJRT_Client_UpdateGlobalProcessInfo_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_Devices_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub devices: *const *mut PJRT_Device,
        pub num_devices: usize,
    }

    impl PJRT_Client_Devices_Args {
        pub fn new(client: *mut PJRT_Client) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                devices: std::ptr::null_mut(),
                num_devices: 0,
            }
        }
    }

    pub type PJRT_Client_Devices = unsafe extern "C" fn(args: *mut PJRT_Client_Devices_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_LookupDevice_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub id: std::ffi::c_int,
        pub device: *mut PJRT_Device,
    }

    impl PJRT_Client_LookupDevice_Args {
        pub fn new(client: *mut PJRT_Client, id: std::ffi::c_int) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                id,
                device: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Client_LookupDevice =
        unsafe extern "C" fn(args: *mut PJRT_Client_LookupDevice_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_AddressableDevices_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub addressable_devices: *const *mut PJRT_Device,
        pub num_addressable_devices: usize,
    }

    impl PJRT_Client_AddressableDevices_Args {
        pub fn new(client: *mut PJRT_Client) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                addressable_devices: std::ptr::null_mut(),
                num_addressable_devices: 0,
            }
        }
    }

    pub type PJRT_Client_AddressableDevices =
        unsafe extern "C" fn(args: *mut PJRT_Client_AddressableDevices_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_LookupAddressableDevice_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub local_hardware_id: std::ffi::c_int,
        pub addressable_device: *mut PJRT_Device,
    }

    impl PJRT_Client_LookupAddressableDevice_Args {
        pub fn new(client: *mut PJRT_Client, local_hardware_id: std::ffi::c_int) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                local_hardware_id,
                addressable_device: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Client_LookupAddressableDevice =
        unsafe extern "C" fn(args: *mut PJRT_Client_LookupAddressableDevice_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_AddressableMemories_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub addressable_memories: *const *mut PJRT_Memory,
        pub num_addressable_memories: usize,
    }

    impl PJRT_Client_AddressableMemories_Args {
        pub fn new(client: *mut PJRT_Client) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                addressable_memories: std::ptr::null(),
                num_addressable_memories: 0,
            }
        }
    }

    pub type PJRT_Client_AddressableMemories =
        unsafe extern "C" fn(args: *mut PJRT_Client_AddressableMemories_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_DefaultDeviceAssignment_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub num_replicas: std::ffi::c_int,
        pub num_partitions: std::ffi::c_int,
        pub default_assignment_size: usize,
        pub default_assignment: *mut std::ffi::c_int,
    }

    impl PJRT_Client_DefaultDeviceAssignment_Args {
        pub fn new(
            client: *mut PJRT_Client,
            num_replicas: std::ffi::c_int,
            num_partitions: std::ffi::c_int,
            default_assignment_size: usize,
            default_assignment: *mut std::ffi::c_int,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                num_replicas,
                num_partitions,
                default_assignment_size,
                default_assignment,
            }
        }
    }

    pub type PJRT_Client_DefaultDeviceAssignment =
        unsafe extern "C" fn(args: *mut PJRT_Client_DefaultDeviceAssignment_Args) -> *mut PJRT_Error;
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::Duration;

    use crate::tests::{TestPlatform, test_cpu_client, test_cpu_plugin, test_for_each_platform};
    use crate::{
        Client, ClientOptions, CpuClientOptions, DeviceAssignment, Error, GpuClientOptions, GpuMemoryAllocator,
        GpuPlatform, KeyValueStore, MockGpuTopology, NamedValue, ProcessInformation, ProcessState, Value,
    };

    #[derive(Default)]
    struct TestKeyValueStore {
        values: Mutex<HashMap<Vec<u8>, Vec<u8>>>,
    }

    impl KeyValueStore for TestKeyValueStore {
        fn put(&self, key: &[u8], value: &[u8]) -> Result<(), Error> {
            self.values.lock().unwrap().insert(key.to_vec(), value.to_vec());
            Ok(())
        }

        fn get(&self, key: &[u8], _timeout: Duration) -> Result<Vec<u8>, Error> {
            self.try_get(key)
        }

        fn try_get(&self, key: &[u8]) -> Result<Vec<u8>, Error> {
            self.values
                .lock()
                .unwrap()
                .get(key)
                .cloned()
                .ok_or_else(|| Error::not_found(format!("key '{}' not found", String::from_utf8_lossy(key))))
        }
    }

    #[test]
    fn test_client() {
        let plugin = test_cpu_plugin();
        let client = test_cpu_client();
        assert_eq!(client.attribute("stablehlo_current_version"), Ok(Value::i64_list([1, 13, 7])));
        assert_eq!(client.attribute("stablehlo_minimum_version"), Ok(Value::i64_list([0, 9, 0])));
        assert_eq!(client.attribute("xla_version"), Ok(Value::i64(2)));
        assert!(matches!(
            client.attribute("__missing__"),
            Err(Error::NotFound { message, .. }) if message.contains("__missing__")));
        let attributes = client.attributes().unwrap();
        assert_eq!(attributes.get("stablehlo_current_version"), Some(&Value::i64_list([1, 13, 7])));
        assert_eq!(attributes.get("stablehlo_minimum_version"), Some(&Value::i64_list([0, 9, 0])));
        assert_eq!(attributes.get("xla_version"), Some(&Value::i64(2)));
        assert_eq!(attributes.get("__missing__"), None);
        assert!(client.key_value_store().is_none());
        assert_eq!(client.platform_name().unwrap(), "cpu");
        assert_eq!(client.platform_version().unwrap(), "cpu");
        assert_eq!(client.process_index(), Ok(0));
        assert_eq!(client.devices().unwrap().len(), 8);
        assert!(client.lookup_device(0).is_ok());
        assert!(matches!(client.lookup_device(8), Err(Error::InvalidArgument { .. })));
        assert_eq!(client.addressable_devices().unwrap().len(), 8);
        assert!(client.lookup_addressable_device(0).is_ok());
        assert!(matches!(client.lookup_addressable_device(8), Err(Error::InvalidArgument { .. })));
        let addressable_memories = client.addressable_memories().unwrap();
        assert_eq!(addressable_memories.len(), 24);
        assert_eq!(format!("{}", addressable_memories[0]), "CPU_DEVICE_0");
        assert_eq!(format!("{}", addressable_memories[1]), "PINNED_HOST_1");
        assert_eq!(format!("{}", addressable_memories[2]), "UNPINNED_HOST_2");
        assert_eq!(format!("{}", addressable_memories[3]), "CPU_DEVICE_3");
        assert_eq!(
            client.default_device_assignment(2, 4),
            Ok(DeviceAssignment { replica_count: 2, computation_count: 4, assignment: vec![0, 1, 2, 3, 4, 5, 6, 7] }),
        );

        // Check that the platform name and version are properly set for all enabled platforms.
        test_for_each_platform!(|_plugin, client, platform| {
            match platform {
                TestPlatform::Cpu => {
                    assert_eq!(client.platform_name().unwrap(), "cpu");
                    assert_eq!(client.platform_version().unwrap(), "cpu");
                }
                TestPlatform::Cuda12 => {
                    assert_eq!(client.platform_name().unwrap(), "cuda");
                    assert_eq!(client.platform_version().unwrap(), "cuda 12090");
                }
                TestPlatform::Cuda13 => {
                    assert_eq!(client.platform_name().unwrap(), "cuda");
                    assert_eq!(client.platform_version().unwrap(), "cuda 13000");
                }
                TestPlatform::Rocm7 => {
                    assert_eq!(client.platform_name().unwrap(), "rocm");
                    assert_eq!(client.platform_version().unwrap(), "7.0");
                }
                TestPlatform::Tpu => {
                    assert_eq!(client.platform_name().unwrap(), "tpu");
                    assert_eq!(client.platform_version().unwrap(), "0.0.34");
                }
                TestPlatform::Neuron => {
                    assert_eq!(client.platform_name().unwrap(), "neuron");
                    assert_eq!(client.platform_version().unwrap(), "2.2.14584.0");
                }
                TestPlatform::Metal => {
                    assert_eq!(client.platform_name().unwrap(), "METAL");
                    assert_eq!(client.platform_version().unwrap(), "metal_0.5.1");
                }
            }
        });

        // Test a client with a key-value store.
        let key_value_store = TestKeyValueStore::default();
        let client = plugin.client_with_key_value_store(ClientOptions::default(), &key_value_store).unwrap();
        let store = client.key_value_store().unwrap();
        assert!(store.put(b"key", b"value").is_ok());
        assert_eq!(store.try_get(b"key"), Ok(b"value".to_vec()));
        assert_eq!(store.get(b"key", Duration::from_millis(10)), Ok(b"value".to_vec()));
        assert!(matches!(store.try_get(b"missing"), Err(Error::NotFound { .. })));

        // Test creating a client from a null pointer.
        assert!(matches!(
            unsafe { Client::from_c_api(std::ptr::null_mut(), plugin.api(), None) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT client handle is a null pointer",
        ));
    }

    #[test]
    fn test_client_update_global_process_information() {
        let client = test_cpu_client();
        assert!(
            client
                .update_global_process_information(&[
                    ProcessInformation { process_id: 0, incarnation_id: 1, state: ProcessState::Uninitialized },
                    ProcessInformation { process_id: 1, incarnation_id: 0, state: ProcessState::Connected },
                    ProcessInformation { process_id: 2, incarnation_id: 0, state: ProcessState::Disconnected },
                    ProcessInformation {
                        process_id: 3,
                        incarnation_id: 42,
                        state: ProcessState::Error { error: Error::internal("test error") },
                    },
                ])
                .is_ok()
        );
    }

    #[test]
    fn test_client_options() {
        let cpu_client_options_0 = ClientOptions::CPU(CpuClientOptions::default());
        assert_eq!(cpu_client_options_0.to_named_values(), Vec::new());

        let cpu_client_options_1 = ClientOptions::CPU(CpuClientOptions { device_count: Some(4) });
        assert_eq!(cpu_client_options_1.to_named_values(), vec![NamedValue::new("cpu_device_count", 4i64)]);

        let gpu_client_options_0 = ClientOptions::GPU(GpuClientOptions {
            platform: Some(GpuPlatform::CUDA),
            allocator: GpuMemoryAllocator::Platform,
            ..GpuClientOptions::default()
        });
        let mut gpu_client_options_0_named_values = gpu_client_options_0.to_named_values();
        gpu_client_options_0_named_values.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(
            gpu_client_options_0_named_values,
            vec![
                NamedValue::new("abort_collectives_on_failure", false),
                NamedValue::new("allocator", "platform"),
                NamedValue::new("platform_name", "cuda"),
                NamedValue::new("preallocate", false),
                NamedValue::new("should_stage_host_to_device_transfers", true),
                NamedValue::new("use_tfrt_gpu_client", false),
            ],
        );

        let gpu_client_options_1 = ClientOptions::GPU(GpuClientOptions {
            platform: Some(GpuPlatform::CUDA),
            visible_devices: Some(vec![1, 3, 5]),
            node_id: Some(2),
            node_count: Some(4),
            partition_index: Some(1),
            should_stage_host_to_device_transfers: false,
            allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: Some(0.5) },
            collective_memory_size: Some(1024),
            abort_collectives_on_failure: true,
            use_tfrt_gpu_client: true,
            mock_gpu_topology: Some(MockGpuTopology {
                partition_count: 2,
                host_count_per_partition: 3,
                device_count_per_host: 4,
            }),
        });
        let mut gpu_client_options_1_named_values = gpu_client_options_1.to_named_values();
        gpu_client_options_1_named_values.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(
            gpu_client_options_1_named_values,
            vec![
                NamedValue::new("abort_collectives_on_failure", true),
                NamedValue::new("allocator", "cuda_async"),
                NamedValue::new("collective_memory_size", 1024),
                NamedValue::new("enable_mock_nccl", true),
                NamedValue::new("memory_fraction", 0.5),
                NamedValue::new("mock_gpu_topology", "2x3x4"),
                NamedValue::new("node_count", 4),
                NamedValue::new("node_id", 2),
                NamedValue::new("partition_index", 1),
                NamedValue::new("platform_name", "cuda"),
                NamedValue::new("preallocate", true),
                NamedValue::new("should_stage_host_to_device_transfers", false),
                NamedValue::new("use_tfrt_gpu_client", true),
                NamedValue::new("visible_devices", [1, 3, 5]),
            ]
        );

        let other_client_options = ClientOptions::Other(HashMap::from([
            ("bool".to_string(), Value::from(true)),
            ("integer".to_string(), Value::from(7)),
            ("string".to_string(), Value::from("cpu")),
        ]));
        let mut other_client_options_named_values = other_client_options.to_named_values();
        other_client_options_named_values.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(
            other_client_options_named_values,
            vec![NamedValue::new("bool", true), NamedValue::new("integer", 7i64), NamedValue::new("string", "cpu")],
        );

        assert_eq!(cpu_client_options_0, cpu_client_options_0);
        assert_eq!(cpu_client_options_1, cpu_client_options_1);
        assert_ne!(cpu_client_options_0, cpu_client_options_1);
        assert_ne!(cpu_client_options_1, gpu_client_options_1);
        assert_ne!(cpu_client_options_1, other_client_options);
        assert_eq!(gpu_client_options_0, gpu_client_options_0);
        assert_eq!(gpu_client_options_1, gpu_client_options_1);
        assert_ne!(gpu_client_options_0, gpu_client_options_1);
        assert_ne!(gpu_client_options_1, cpu_client_options_1);
        assert_ne!(gpu_client_options_1, other_client_options);
        assert_eq!(other_client_options, other_client_options);
        assert_ne!(other_client_options, cpu_client_options_0);
        assert_ne!(other_client_options, gpu_client_options_1);
    }
}
