use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::OnceLock;

use prost::Message;

use crate::{
    Api, Client, Error, HasDefaultMemory, Memory, MemoryStatistics, Plugin, ProcessIndex, Value, hash_map_from_c_api,
    invoke_pjrt_api_error_fn, slice_from_c_api, str_from_c_api,
};

/// Type alias used to represent [`Device`] IDs, which are unique among devices of the same type (e.g., CPUs, GPUs)
/// and, on multi-host environments, are also unique across all devices and all hosts.
pub type DeviceId = usize;

/// Type alias used to represent the opaque local hardware IDs of [`Device`]s (e.g., a CUDA device number).
pub type LocalHardwareId = usize;

/// Type alias used to represent replica IDs in [`DeviceAssignment`]s.
pub type ReplicaId = usize;

/// Type alias used to represent computation IDs in [`DeviceAssignment`]s.
pub type ComputationId = usize;

/// Device managed by a PJRT [`Plugin`] (e.g., a specific CPU, GPU, TPU, etc.). Each device has a [`DeviceDescription`]
/// that helps identify its kind and a location within a grid of devices both locally and globally. Devices also know
/// their associated [`Memory`]s and the [`Client`] that they are owned by. Note that a device does not necessarily
/// know the buffers of actual data associated with it, but it can figure that out by looking through its associated
/// [`Memory`]s.
///
/// The lifetime parameter `'o` captures the lifetime of the owner of this [`Device`] (e.g., a [`Client`] or a
/// [`Memory`]), ensuring that the owner outlives the device.
#[derive(Clone)]
pub struct Device<'o> {
    /// Handle that represents this [`Device`] in the PJRT C API.
    handle: *mut ffi::PJRT_Device,

    /// Underlying PJRT [`Api`].
    api: Api,

    /// Cached [`Device::description`] of this [`Device`] so that it will only be constructed once.
    description: OnceLock<Result<DeviceDescription<'o>, Error>>,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`Device`].
    owner: PhantomData<&'o ()>,
}

impl Device<'_> {
    /// Constructs a new [`Device`] from the provided [`PJRT_Device`](ffi::PJRT_Device) handle that came
    /// from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(handle: *mut ffi::PJRT_Device, api: Api) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT device handle is a null pointer"))
        } else {
            Ok(Self { handle, api, description: OnceLock::new(), owner: PhantomData })
        }
    }

    /// Returns the [`PJRT_Device`](ffi::PJRT_Device) that corresponds to this [`Device`] and which can
    /// be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_Device {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }

    /// ID of this [`Device`]. IDs are unique among devices of the same type (e.g., CPUs, GPUs) and, in multi-host
    /// environments, they are also unique across all devices and all hosts.
    pub fn id(&self) -> Result<DeviceId, Error> {
        self.description()?.id()
    }

    /// Vendor-dependent string that uniquely identifies the kind of this [`Device`] (e.g., "Tesla V100-SXM2-16GB").
    pub fn kind(&'_ self) -> Result<Cow<'_, str>, Error> {
        self.description()?.kind()
    }

    /// Index of the process that this [`Device`] belongs to (i.e., is _addressable_ from). Note that this is not
    /// always identical to the process index of the corresponding [`Client`] in a multi-process setting, where each
    /// client can see devices from all processes, but only a subset of them are addressable and have the same process
    /// index as the client.
    pub fn process_index(&self) -> Result<ProcessIndex, Error> {
        self.description()?.process_index()
    }

    /// [`Value`] of the attribute with the provided name attached to this [`Device`], or [`Error::NotFound`]
    /// if no such attribute is attached to this [`Device`].
    pub fn attribute<N: AsRef<str>>(&self, name: N) -> Result<&Value, Error> {
        let name = name.as_ref();
        self.attributes()?
            .get(&name.to_string())
            .ok_or_else(|| Error::not_found(format!("no attribute named '{name}' in this PJRT device")))
    }

    /// Collection of [`Device`]-specific named attributes that are attached to this [`Device`].
    pub fn attributes(&self) -> Result<&HashMap<String, Value>, Error> {
        self.description()?.attributes()
    }

    /// [`DeviceDescription`] associated with this [`Device`].
    pub fn description(&'_ self) -> Result<&DeviceDescription<'_>, Error> {
        self.description
            .get_or_init(|| {
                use ffi::PJRT_Device_GetDescription_Args;
                invoke_pjrt_api_error_fn!(self.api(), PJRT_Device_GetDescription, { device = self.to_c_api() }, {
                    device_description
                })
                .and_then(|handle| unsafe { DeviceDescription::from_c_api(handle, self.api()) })
            })
            .as_ref()
            .map_err(|error| error.clone())
    }

    /// Opaque local hardware ID of this [`Device`] (e.g., its CUDA device number). In general, local hardware IDs are
    /// not guaranteed to be dense/contiguous and are also not always defined.
    pub fn local_hardware_id(&self) -> Result<Option<LocalHardwareId>, Error> {
        use ffi::PJRT_Device_LocalHardwareId_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Device_LocalHardwareId, { device = self.to_c_api() }, {
            local_hardware_id
        })
        .map(|id| if id >= 0 { Some(id as usize) } else { None })
    }

    /// Returns `true` if this [`Device`] is _addressable_ by the owning PJRT [`Client`] (i.e., if that client
    /// can issue commands to this device).
    pub fn is_addressable(&self) -> Result<bool, Error> {
        use ffi::PJRT_Device_IsAddressable_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Device_IsAddressable, { device = self.to_c_api() }, {
            is_addressable
        })
    }

    /// [`Memory`]s that this [`Device`] can address.
    pub fn addressable_memories(&'_ self) -> Result<Vec<Memory<'_>>, Error> {
        use ffi::PJRT_Device_AddressableMemories_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Device_AddressableMemories,
            { device = self.to_c_api() },
            { memories, num_memories },
        )
        .and_then(|(memories, memories_count)| {
            unsafe { slice_from_c_api(memories, memories_count) }
                .iter()
                .map(|handle| unsafe { Memory::from_c_api(*handle, self.api()) })
                .collect::<Result<Vec<_>, _>>()
        })
    }

    /// Default [`Memory`] of this [`Device`] (i.e., the memory in which data processed by this device
    /// should be stored in by default).
    pub fn default_memory(&'_ self) -> Result<Memory<'_>, Error> {
        use ffi::PJRT_Device_DefaultMemory_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Device_DefaultMemory, { device = self.to_c_api() }, { memory })
            .and_then(|handle| unsafe { Memory::from_c_api(handle, self.api()) })
    }

    /// Returns memory/allocator statistics for this [`Device`] (intended for diagnostic purposes). Note that not all
    /// PJRT [`Plugin`]s support this functionality, and this function may return [`Error::Unimplemented`] for plugins
    /// where it is not supported.
    pub fn memory_statistics(&self) -> Result<MemoryStatistics, Error> {
        use ffi::PJRT_Device_MemoryStats_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Device_MemoryStats, { device = self.to_c_api() }, {
            bytes_in_use,
            peak_bytes_in_use,
            peak_bytes_in_use_is_set,
            num_allocs,
            num_allocs_is_set,
            largest_alloc_size,
            largest_alloc_size_is_set,
            bytes_limit,
            bytes_limit_is_set,
            bytes_reserved,
            bytes_reserved_is_set,
            peak_bytes_reserved,
            peak_bytes_reserved_is_set,
            bytes_reservable_limit,
            bytes_reservable_limit_is_set,
            largest_free_block_bytes,
            largest_free_block_bytes_is_set,
            pool_bytes,
            pool_bytes_is_set,
            peak_pool_bytes,
            peak_pool_bytes_is_set,
        })
        .map(
            |(
                bytes_in_use,
                peak_bytes_in_use,
                peak_bytes_in_use_is_set,
                num_allocs,
                num_allocs_is_set,
                largest_alloc_size,
                largest_alloc_size_is_set,
                bytes_limit,
                bytes_limit_is_set,
                bytes_reserved,
                bytes_reserved_is_set,
                peak_bytes_reserved,
                peak_bytes_reserved_is_set,
                bytes_reservable_limit,
                bytes_reservable_limit_is_set,
                largest_free_block_bytes,
                largest_free_block_bytes_is_set,
                pool_bytes,
                pool_bytes_is_set,
                peak_pool_bytes,
                peak_pool_bytes_is_set,
            )| MemoryStatistics {
                bytes_in_use: bytes_in_use.cast_unsigned(),
                peak_bytes_in_use: peak_bytes_in_use_is_set.then_some(peak_bytes_in_use.cast_unsigned()),
                allocation_count: num_allocs_is_set.then_some(num_allocs.cast_unsigned()),
                largest_allocation_size: largest_alloc_size_is_set.then_some(largest_alloc_size.cast_unsigned()),
                bytes_limit: bytes_limit_is_set.then_some(bytes_limit.cast_unsigned()),
                reserved_bytes: bytes_reserved_is_set.then_some(bytes_reserved.cast_unsigned()),
                peak_reserved_bytes: peak_bytes_reserved_is_set.then_some(peak_bytes_reserved.cast_unsigned()),
                reservable_bytes_limit: bytes_reservable_limit_is_set.then_some(bytes_reservable_limit.cast_unsigned()),
                largest_free_block_bytes: largest_free_block_bytes_is_set
                    .then_some(largest_free_block_bytes.cast_unsigned()),
                pool_bytes: pool_bytes_is_set.then_some(pool_bytes.cast_unsigned()),
                peak_pool_bytes: peak_pool_bytes_is_set.then_some(peak_pool_bytes.cast_unsigned()),
            },
        )
    }

    /// _Poisons_ the earliest execution on this [`Device`] with the provided launch ID if it is not finished
    /// yet (i.e., sets the resulting [`Buffer`](crate::Buffer) to an error buffer; refer to the documentation of
    /// [`Client::error_buffer`] for more information on buffer _poisoning_). Returns `true` if the execution was
    /// poisoned successfully and `false` if it had already finished executing.
    pub fn poison_execution(&self, launch_id: i32, error: Error) -> Result<bool, Error> {
        use ffi::PJRT_Device_PoisonExecution_Args;
        let error_message = error.message();
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Device_PoisonExecution,
            {
                device = self.to_c_api(),
                launch_id = launch_id,
                error_code = error.code(),
                error_message = error_message.as_ptr(),
                error_message_size = error_message.count_bytes(),
            },
            { poisoned },
        )
    }
}

impl Display for Device<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.description() {
            Ok(description) => write!(formatter, "{}", description),
            Err(error) => write!(formatter, "<failed to render PJRT device as string; {:?}>", error),
        }
    }
}

impl Debug for Device<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.description() {
            Ok(description) => {
                use ffi::PJRT_DeviceDescription_DebugString_Args;
                match invoke_pjrt_api_error_fn!(
                    self.api(),
                    PJRT_DeviceDescription_DebugString,
                    { device_description = description.handle },
                    { debug_string, debug_string_size },
                ) {
                    Ok((string, string_len)) => write!(formatter, "Device[{}]", str_from_c_api(string, string_len)),
                    Err(error) => write!(formatter, "<failed to render PJRT device as debug string; {:?}>", error),
                }
            }
            Err(error) => write!(formatter, "<failed to render PJRT device as debug string; {:?}>", error),
        }
    }
}

impl PartialEq for Device<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.id().is_ok()
            && other.id().is_ok()
            && self.id() == other.id()
            && self.kind().is_ok()
            && other.kind().is_ok()
            && self.kind() == other.kind()
    }
}

impl Eq for Device<'_> {}

impl HasDefaultMemory for Device<'_> {
    fn default_memory(&self) -> Memory<'_> {
        self.default_memory().expect(format!("default memory not set for device '{self}'").as_str())
    }
}

/// Description of a [`Device`] which may be associated with an actual [`Device`] instance (i.e., obtained via
/// [`Device::description`]) or that is used to describe a device that is not available to the current PJRT plugin.
/// This is useful for compiling executables without having the target hardware available, resulting in executables
/// that can be serialized and persisted ahead of time such that they can be loaded and executed on the actual target
/// hardware later.
///
/// Note that the `'o` lifetime parameter captures the fact that [`DeviceDescription`]s are always owned by some other
/// object (e.g., a [`Client`] or a [`Device`]) and makes sure that that other object stays alive for at least as long
/// as all associated [`DeviceDescription`]s are alive.
#[derive(Clone)]
pub struct DeviceDescription<'o> {
    /// Handle that represents this [`DeviceDescription`] in the PJRT C API.
    handle: *mut ffi::PJRT_DeviceDescription,

    /// Underlying PJRT [`Api`].
    api: Api,

    /// Cached [`DeviceDescription::attributes`] of this [`DeviceDescription`] so that it will only be constructed once.
    attributes: OnceLock<Result<HashMap<String, Value>, Error>>,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`DeviceDescription`].
    owner: PhantomData<&'o ()>,
}

impl DeviceDescription<'_> {
    /// Constructs a new [`DeviceDescription`] from the provided [`PJRT_DeviceDescription`](ffi::PJRT_DeviceDescription)
    /// handle that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(handle: *mut ffi::PJRT_DeviceDescription, api: Api) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT device description handle is a null pointer"))
        } else {
            Ok(Self { handle, api, attributes: OnceLock::new(), owner: PhantomData })
        }
    }

    /// Returns the [`PJRT_DeviceDescription`](ffi::PJRT_DeviceDescription) that corresponds to this
    /// [`DeviceDescription`] and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_DeviceDescription {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }

    /// [`Device`] ID that corresponds to this [`DeviceDescription`]. IDs are unique among devices of the same type
    /// (e.g., CPUs, GPUs) and, in multi-host environments, they are also unique across all devices and all hosts.
    pub fn id(&self) -> Result<DeviceId, Error> {
        use ffi::PJRT_DeviceDescription_Id_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_DeviceDescription_Id, { device_description = self.to_c_api() }, {
            id
        })
        .map(|id| id as usize)
    }

    /// Vendor-dependent string that uniquely identifies the kind of the underlying [`Device`]
    /// (e.g., "Tesla V100-SXM2-16GB").
    pub fn kind(&'_ self) -> Result<Cow<'_, str>, Error> {
        use ffi::PJRT_DeviceDescription_Kind_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_DeviceDescription_Kind,
            { device_description = self.to_c_api() },
            { device_kind, device_kind_size },
        )
        .map(|(string, string_len)| str_from_c_api(string, string_len))
    }

    /// Index of the process that the underlying [`Device`] belongs to (i.e., is _addressable_ from). Note that this is
    /// not always identical to the process index of the corresponding [`Client`] in a multi-process setting, where each
    /// client can see devices from all processes, but only a subset of them are addressable and have the same process
    /// index as the client.
    pub fn process_index(&self) -> Result<ProcessIndex, Error> {
        use ffi::PJRT_DeviceDescription_ProcessIndex_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_DeviceDescription_ProcessIndex,
            { device_description = self.to_c_api() },
            { process_index },
        )
        .map(|id| id as usize)
    }

    /// [`Value`] of the attribute with the provided name attached to this [`DeviceDescription`], or [`Error::NotFound`]
    /// if no such attribute is attached to this [`DeviceDescription`].
    pub fn attribute<N: AsRef<str>>(&self, name: N) -> Result<&Value, Error> {
        let name = name.as_ref();
        self.attributes()?
            .get(&name.to_string())
            .ok_or_else(|| Error::not_found(format!("no attribute named '{name}' in this PJRT device description")))
    }

    /// Collection of [`Device`]-specific named attributes that are attached to this [`DeviceDescription`].
    pub fn attributes(&self) -> Result<&HashMap<String, Value>, Error> {
        self.attributes
            .get_or_init(|| {
                use ffi::PJRT_DeviceDescription_Attributes_Args;
                let (attributes, attribute_count) = invoke_pjrt_api_error_fn!(
                    self.api(),
                    PJRT_DeviceDescription_Attributes,
                    { device_description = self.to_c_api() },
                    { attributes, num_attributes },
                )?;
                Ok(hash_map_from_c_api(attributes, attribute_count))
            })
            .as_ref()
            .map_err(|error| error.clone())
    }
}

impl Display for DeviceDescription<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ffi::PJRT_DeviceDescription_ToString_Args;
        match invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_DeviceDescription_ToString,
            { device_description = self.to_c_api() },
            { to_string, to_string_size },
        ) {
            Ok((string, string_len)) => write!(formatter, "{}", str_from_c_api(string, string_len)),
            Err(error) => write!(formatter, "<failed to render PJRT device description as string; {}>", error),
        }
    }
}

impl Debug for DeviceDescription<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ffi::PJRT_DeviceDescription_DebugString_Args;
        match invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_DeviceDescription_DebugString,
            { device_description = self.to_c_api() },
            { debug_string, debug_string_size },
        ) {
            Ok((string, string_len)) => write!(formatter, "DeviceDescription[{}]", str_from_c_api(string, string_len)),
            Err(error) => write!(formatter, "<failed to render PJRT device description as debug string; {:?}>", error),
        }
    }
}

impl PartialEq for DeviceDescription<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.id().is_ok()
            && other.id().is_ok()
            && self.id() == other.id()
            && self.kind().is_ok()
            && other.kind().is_ok()
            && self.kind() == other.kind()
    }
}

impl Eq for DeviceDescription<'_> {}

/// Represents the [`Device`] assignment for a set of replicated computations. Specifically, for `R` replicas and `C`
/// computations, `R * C` [`Device`]s are required to execute those computations in parallel. [`DeviceAssignment`]s hold
/// the mapping from `(r, c)`, where `r` is a replica index and `c` is a computation index, to the [`DeviceId`] of the
/// [`Device`] on which the corresponding computation should be executed.
///
/// The default [`DeviceAssignment`] for a given [`Client`] can be obtained using [`Client::default_device_assignment`],
/// which is aware of the set of [`Device`]s that _addressable_ by that [`Client`]. Refer to the documentation of
/// [`Client::addressable_devices`] for information on what is an _addressable_ [`Device`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceAssignment {
    /// Number of replicas that this [`DeviceAssignment`] has been computed for.
    pub(crate) replica_count: usize,

    /// Number of computations that this [`DeviceAssignment`] has been computed for.
    pub(crate) computation_count: usize,

    /// Flattened representation of this [`DeviceAssignment`] with the [`DeviceId`]s for `replica_count`
    /// and `computation_count` stored in row-major format.
    pub(crate) assignment: Vec<std::ffi::c_int>,
}

impl DeviceAssignment {
    /// Number of replicas that this [`DeviceAssignment`] has been computed for.
    pub fn replica_count(&self) -> usize {
        self.replica_count
    }

    /// Number of computations that this [`DeviceAssignment`] has been computed for.
    pub fn computation_count(&self) -> usize {
        self.computation_count
    }

    /// Returns the [`DeviceId`] that replica `replica` of computation `computation` is assigned to or an
    /// [`Error::FailedPrecondition`] if any of the provided indices are out of range (i.e., larger than the number of
    /// replicas and computations that this [`DeviceAssignment`] was constructed for, respectively).
    pub fn device_id(&self, replica_id: ReplicaId, computation_id: ComputationId) -> Result<DeviceId, Error> {
        if replica_id >= self.replica_count {
            Err(Error::failed_precondition("replica ID is out of range"))
        } else if computation_id >= self.computation_count {
            Err(Error::failed_precondition("computation ID is out of range"))
        } else {
            Ok(self.assignment[replica_id * self.computation_count + computation_id] as usize)
        }
    }

    /// Internal helper function that returns the replica ID assigned to the [`Device`] with the provided
    /// [`DeviceId`] in this [`DeviceAssignment`]. If there are multiple computations or replicas assigned to the same
    /// [`Device`] or if the provided [`DeviceId`] is not used by this [`DeviceAssignment`], then this function will
    /// return an [`Error::Internal`].
    pub fn replica_id(&self, device_id: DeviceId) -> Result<ReplicaId, Error> {
        self.logical_id(device_id).map(|(replica_id, _)| replica_id)
    }

    /// Internal helper function that returns the computation ID assigned to the [`Device`] with the provided
    /// [`DeviceId`] in this [`DeviceAssignment`]. If there are multiple computations or replicas assigned to the same
    /// [`Device`] or if the provided [`DeviceId`] is not used by this [`DeviceAssignment`], then this function will
    /// return an [`Error::Internal`].
    pub fn computation_id(&self, device_id: DeviceId) -> Result<ComputationId, Error> {
        self.logical_id(device_id).map(|(_, computation_id)| computation_id)
    }

    /// Returns the logical ID (i.e., the pair of replica ID and computation ID) assigned to the [`Device`] with the
    /// provided [`DeviceId`] in this [`DeviceAssignment`]. If there are multiple logical IDs assigned to the same
    /// [`Device`] or if the provided [`DeviceId`] is not used by this [`DeviceAssignment`], then this function will
    /// return an [`Error::Internal`].
    pub fn logical_id(&self, device_id: DeviceId) -> Result<(ReplicaId, ComputationId), Error> {
        let mut logical_id = None;
        for replica_id in 0..self.replica_count {
            for computation_id in 0..self.computation_count {
                if self.assignment[replica_id * self.computation_count + computation_id] as usize == device_id {
                    if logical_id.is_some() {
                        return Err(Error::internal("duplicate device ID"));
                    } else {
                        logical_id = Some((replica_id, computation_id));
                    }
                }
            }
        }
        logical_id.ok_or_else(|| Error::internal("device ID not found"))
    }

    /// Serializes this [`DeviceAssignment`] to a Protobuf message.
    pub fn proto(&self) -> Result<crate::protos::DeviceAssignment, Error> {
        let mut computation_devices = Vec::with_capacity(self.computation_count);
        for _ in 0..self.computation_count {
            computation_devices.push(crate::protos::ComputationDeviceAssignment {
                replica_device_ids: Vec::with_capacity(self.replica_count),
            });
        }
        for replica_id in 0..self.replica_count {
            for computation_id in 0..self.computation_count {
                computation_devices[computation_id]
                    .replica_device_ids
                    .push(self.assignment[replica_id * self.computation_count + computation_id] as i64);
            }
        }
        Ok(crate::protos::DeviceAssignment {
            replica_count: self.replica_count as i32,
            computation_count: self.computation_count as i32,
            computation_devices,
        })
    }

    /// Serializes this [`DeviceAssignment`] into a string (i.e., byte array).
    pub fn serialize(&self) -> Result<SerializedDeviceAssignment, Error> {
        Ok(SerializedDeviceAssignment::Rust { data: self.proto()?.encode_to_vec() })
    }
}

/// Serialized [`DeviceAssignment`].
pub enum SerializedDeviceAssignment {
    /// Serialized [`DeviceAssignment`] that was allocated/serialized using [`DeviceAssignment::serialize`].
    Rust {
        /// Bytes that correspond to a serialized [`DeviceAssignment`].
        data: Vec<u8>,
    },

    /// Serialized [`DeviceAssignment`] that was allocated/serialized using the PJRT C API.
    C {
        /// Handle that represents this [`SerializedDeviceAssignment`] in the PJRT C API.
        handle: *mut ffi::PJRT_DeviceAssignmentSerialized,

        /// Optional function that must be called to free the underlying memory when dropping this instance.
        deleter: Option<unsafe extern "C" fn(device_assignment: *mut ffi::PJRT_DeviceAssignmentSerialized)>,

        /// Pointer to the underlying bytes of this [`SerializedDeviceAssignment`].
        data: *const std::ffi::c_char,

        /// Size (i.e., number of bytes) of this [`SerializedDeviceAssignment`].
        data_size: usize,
    },
}

impl SerializedDeviceAssignment {
    /// Returns a pointer to the underlying bytes of this [`SerializedDeviceAssignment`].
    pub fn data(&self) -> &[u8] {
        match self {
            Self::Rust { data } => &data,
            Self::C { data, data_size, .. } => unsafe { slice_from_c_api(*data as *const u8, *data_size) },
        }
    }

    /// Returns the Protobuf message that corresponds to this [`SerializedDeviceAssignment`].
    pub fn proto(&self) -> Result<crate::protos::DeviceAssignment, Error> {
        crate::protos::DeviceAssignment::decode(self.data()).map_err(|error| Error::invalid_argument(error.to_string()))
    }

    /// Deserializes this [`SerializedDeviceAssignment`] into a [`DeviceAssignment`].
    pub fn deserialize(&self) -> Result<DeviceAssignment, Error> {
        let message = crate::protos::DeviceAssignment::decode(self.data())
            .map_err(|error| Error::invalid_argument(error.to_string()))?;
        let replica_count = message.replica_count as usize;
        let computation_count = message.computation_count as usize;
        let mut assignment = Vec::<std::ffi::c_int>::with_capacity(replica_count * computation_count);
        for replica_id in 0..replica_count {
            for computation_id in 0..computation_count {
                assignment.push(
                    message.computation_devices[computation_id].replica_device_ids[replica_id] as std::ffi::c_int,
                );
            }
        }
        Ok(DeviceAssignment { replica_count, computation_count, assignment })
    }
}

unsafe impl Send for SerializedDeviceAssignment {}
unsafe impl Sync for SerializedDeviceAssignment {}

impl Drop for SerializedDeviceAssignment {
    fn drop(&mut self) {
        if let Self::C { handle, deleter: Some(deleter), .. } = self {
            unsafe { deleter(*handle) };
        }
    }
}

impl Client<'_> {
    /// Deserializes the provided Protobuf message into a [`DeviceAssignment`].
    pub fn device_assignment_from_proto(
        &self,
        proto: crate::protos::DeviceAssignment,
    ) -> Result<DeviceAssignment, Error> {
        self.api().device_assignment_from_proto(proto)
    }

    /// Deserializes the provided data into a [`DeviceAssignment`].
    pub fn deserialize_device_assignment(&self, data: &[u8]) -> Result<DeviceAssignment, Error> {
        self.api().deserialize_device_assignment(data)
    }
}

impl Plugin {
    /// Deserializes the provided Protobuf message into a [`DeviceAssignment`].
    pub fn device_assignment_from_proto(
        &self,
        proto: crate::protos::DeviceAssignment,
    ) -> Result<DeviceAssignment, Error> {
        self.api().device_assignment_from_proto(proto)
    }

    /// Deserializes the provided data into a [`DeviceAssignment`].
    pub fn deserialize_device_assignment(&self, data: &[u8]) -> Result<DeviceAssignment, Error> {
        self.api().deserialize_device_assignment(data)
    }
}

impl Api {
    /// Deserializes the provided Protobuf message into a [`DeviceAssignment`].
    pub(crate) fn device_assignment_from_proto(
        &self,
        proto: crate::protos::DeviceAssignment,
    ) -> Result<DeviceAssignment, Error> {
        let replica_count = proto.replica_count as usize;
        let computation_count = proto.computation_count as usize;
        let mut assignment = Vec::<std::ffi::c_int>::with_capacity(replica_count * computation_count);
        for replica_id in 0..replica_count {
            for computation_id in 0..computation_count {
                if computation_id >= proto.computation_devices.len() {
                    return Err(Error::invalid_argument("invalid device assignment"));
                }
                let replica_device_ids = &proto.computation_devices[computation_id].replica_device_ids;
                if replica_id >= replica_device_ids.len() {
                    return Err(Error::invalid_argument("invalid device assignment"));
                }
                assignment.push(replica_device_ids[replica_id] as std::ffi::c_int);
            }
        }
        Ok(DeviceAssignment { replica_count, computation_count, assignment })
    }

    /// Deserializes the provided data into a [`DeviceAssignment`].
    pub(crate) fn deserialize_device_assignment(&self, data: &[u8]) -> Result<DeviceAssignment, Error> {
        self.device_assignment_from_proto(
            crate::protos::DeviceAssignment::decode(data)
                .map_err(|error| Error::invalid_argument(error.to_string()))?,
        )
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::errors::ffi::{PJRT_Error, PJRT_Error_Code};
    use crate::ffi::PJRT_Extension_Base;
    use crate::memories::ffi::PJRT_Memory;
    use crate::values::ffi::PJRT_NamedValue;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_Device {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Device_GetDescription_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device: *mut PJRT_Device,
        pub device_description: *mut PJRT_DeviceDescription,
    }

    impl PJRT_Device_GetDescription_Args {
        pub fn new(device: *mut PJRT_Device) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device,
                device_description: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Device_GetDescription =
        unsafe extern "C" fn(args: *mut PJRT_Device_GetDescription_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Device_LocalHardwareId_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device: *mut PJRT_Device,
        pub local_hardware_id: std::ffi::c_int,
    }

    impl PJRT_Device_LocalHardwareId_Args {
        pub fn new(device: *mut PJRT_Device) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), device, local_hardware_id: 0 }
        }
    }

    pub type PJRT_Device_LocalHardwareId =
        unsafe extern "C" fn(args: *mut PJRT_Device_LocalHardwareId_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Device_IsAddressable_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device: *mut PJRT_Device,
        pub is_addressable: bool,
    }

    impl PJRT_Device_IsAddressable_Args {
        pub fn new(device: *mut PJRT_Device) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device,
                is_addressable: false,
            }
        }
    }

    pub type PJRT_Device_IsAddressable =
        unsafe extern "C" fn(args: *mut PJRT_Device_IsAddressable_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Device_AddressableMemories_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device: *mut PJRT_Device,
        pub memories: *const *mut PJRT_Memory,
        pub num_memories: usize,
    }

    impl PJRT_Device_AddressableMemories_Args {
        pub fn new(device: *mut PJRT_Device) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device,
                memories: std::ptr::null(),
                num_memories: 0,
            }
        }
    }

    pub type PJRT_Device_AddressableMemories =
        unsafe extern "C" fn(args: *mut PJRT_Device_AddressableMemories_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Device_DefaultMemory_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device: *mut PJRT_Device,
        pub memory: *mut PJRT_Memory,
    }

    impl PJRT_Device_DefaultMemory_Args {
        pub fn new(device: *mut PJRT_Device) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device,
                memory: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Device_DefaultMemory =
        unsafe extern "C" fn(args: *mut PJRT_Device_DefaultMemory_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Device_MemoryStats_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device: *mut PJRT_Device,
        pub bytes_in_use: i64,
        pub peak_bytes_in_use: i64,
        pub peak_bytes_in_use_is_set: bool,
        pub num_allocs: i64,
        pub num_allocs_is_set: bool,
        pub largest_alloc_size: i64,
        pub largest_alloc_size_is_set: bool,
        pub bytes_limit: i64,
        pub bytes_limit_is_set: bool,
        pub bytes_reserved: i64,
        pub bytes_reserved_is_set: bool,
        pub peak_bytes_reserved: i64,
        pub peak_bytes_reserved_is_set: bool,
        pub bytes_reservable_limit: i64,
        pub bytes_reservable_limit_is_set: bool,
        pub largest_free_block_bytes: i64,
        pub largest_free_block_bytes_is_set: bool,
        pub pool_bytes: i64,
        pub pool_bytes_is_set: bool,
        pub peak_pool_bytes: i64,
        pub peak_pool_bytes_is_set: bool,
    }

    impl PJRT_Device_MemoryStats_Args {
        pub fn new(device: *mut PJRT_Device) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device,
                bytes_in_use: 0,
                peak_bytes_in_use: 0,
                peak_bytes_in_use_is_set: false,
                num_allocs: 0,
                num_allocs_is_set: false,
                largest_alloc_size: 0,
                largest_alloc_size_is_set: false,
                bytes_limit: 0,
                bytes_limit_is_set: false,
                bytes_reserved: 0,
                bytes_reserved_is_set: false,
                peak_bytes_reserved: 0,
                peak_bytes_reserved_is_set: false,
                bytes_reservable_limit: 0,
                bytes_reservable_limit_is_set: false,
                largest_free_block_bytes: 0,
                largest_free_block_bytes_is_set: false,
                pool_bytes: 0,
                pool_bytes_is_set: false,
                peak_pool_bytes: 0,
                peak_pool_bytes_is_set: false,
            }
        }
    }

    pub type PJRT_Device_MemoryStats = unsafe extern "C" fn(args: *mut PJRT_Device_MemoryStats_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Device_PoisonExecution_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device: *mut PJRT_Device,
        pub launch_id: i32,
        pub error_code: PJRT_Error_Code,
        pub error_message: *const std::ffi::c_char,
        pub error_message_size: usize,
        pub poisoned: bool,
    }

    impl PJRT_Device_PoisonExecution_Args {
        pub fn new(
            device: *mut PJRT_Device,
            launch_id: i32,
            error_code: PJRT_Error_Code,
            error_message: *const std::ffi::c_char,
            error_message_size: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device,
                launch_id,
                error_code,
                error_message,
                error_message_size,
                poisoned: false,
            }
        }
    }

    pub type PJRT_Device_PoisonExecution =
        unsafe extern "C" fn(args: *mut PJRT_Device_PoisonExecution_Args) -> *mut PJRT_Error;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_DeviceDescription {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_DeviceDescription_Id_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device_description: *mut PJRT_DeviceDescription,
        pub id: std::ffi::c_int,
    }

    impl PJRT_DeviceDescription_Id_Args {
        pub fn new(device_description: *mut PJRT_DeviceDescription) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), device_description, id: 0 }
        }
    }

    pub type PJRT_DeviceDescription_Id =
        unsafe extern "C" fn(args: *mut PJRT_DeviceDescription_Id_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_DeviceDescription_Kind_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device_description: *mut PJRT_DeviceDescription,
        pub device_kind: *const std::ffi::c_char,
        pub device_kind_size: usize,
    }

    impl PJRT_DeviceDescription_Kind_Args {
        pub fn new(device_description: *mut PJRT_DeviceDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device_description,
                device_kind: std::ptr::null(),
                device_kind_size: 0,
            }
        }
    }

    pub type PJRT_DeviceDescription_Kind =
        unsafe extern "C" fn(args: *mut PJRT_DeviceDescription_Kind_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_DeviceDescription_ProcessIndex_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device_description: *mut PJRT_DeviceDescription,
        pub process_index: std::ffi::c_int,
    }

    impl PJRT_DeviceDescription_ProcessIndex_Args {
        pub fn new(device_description: *mut PJRT_DeviceDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device_description,
                process_index: 0,
            }
        }
    }

    pub type PJRT_DeviceDescription_ProcessIndex =
        unsafe extern "C" fn(args: *mut PJRT_DeviceDescription_ProcessIndex_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_DeviceDescription_Attributes_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device_description: *mut PJRT_DeviceDescription,
        pub num_attributes: usize,
        pub attributes: *const PJRT_NamedValue,
    }

    impl PJRT_DeviceDescription_Attributes_Args {
        pub fn new(device_description: *mut PJRT_DeviceDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device_description,
                num_attributes: 0,
                attributes: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_DeviceDescription_Attributes =
        unsafe extern "C" fn(args: *mut PJRT_DeviceDescription_Attributes_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_DeviceDescription_ToString_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device_description: *mut PJRT_DeviceDescription,
        pub to_string: *const std::ffi::c_char,
        pub to_string_size: usize,
    }

    impl PJRT_DeviceDescription_ToString_Args {
        pub fn new(device_description: *mut PJRT_DeviceDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device_description,
                to_string: std::ptr::null(),
                to_string_size: 0,
            }
        }
    }

    pub type PJRT_DeviceDescription_ToString =
        unsafe extern "C" fn(args: *mut PJRT_DeviceDescription_ToString_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_DeviceDescription_DebugString_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device_description: *mut PJRT_DeviceDescription,
        pub debug_string: *const std::ffi::c_char,
        pub debug_string_size: usize,
    }

    impl PJRT_DeviceDescription_DebugString_Args {
        pub fn new(device_description: *mut PJRT_DeviceDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device_description,
                debug_string: std::ptr::null(),
                debug_string_size: 0,
            }
        }
    }

    pub type PJRT_DeviceDescription_DebugString =
        unsafe extern "C" fn(args: *mut PJRT_DeviceDescription_DebugString_Args) -> *mut PJRT_Error;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_DeviceAssignmentSerialized {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;

    use crate::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use crate::tests::{TestPlatform, test_cpu_client, test_for_each_platform};
    use crate::{
        BufferType, Device, DeviceAssignment, DeviceDescription, Error, ExecutionDeviceInputs, ExecutionInput, Program,
    };

    #[test]
    fn test_device() {
        test_for_each_platform!(|plugin, client, platform| {
            let devices = client.devices().unwrap();
            match platform {
                TestPlatform::Cpu => assert_eq!(devices.len(), 8),
                _ => assert!(!devices.is_empty()),
            };
            for (index, device) in devices.iter().enumerate() {
                assert!(device.id().is_ok());
                assert!(device.kind().is_ok());
                assert_eq!(device.process_index(), Ok(0));
                assert!(device.attribute("__test__").is_err());
                assert!(device.attributes().is_ok());
                assert!(device.description().is_ok());
                assert!(device.local_hardware_id().is_ok());
                assert_eq!(device.is_addressable(), Ok(true));
                assert!(device.addressable_memories().is_ok());
                match platform {
                    TestPlatform::Cpu => {
                        assert_eq!(device.id(), Ok(index));
                        assert_eq!(device.kind().map(|kind| kind.to_string()), Ok("cpu".to_string()));
                        assert_eq!(device.attributes().map(|attributes| attributes.is_empty()), Ok(true));
                        assert_eq!(device.local_hardware_id().unwrap(), Some(index));
                        assert_eq!(device.addressable_memories().map(|memories| memories.len()), Ok(3));
                        assert!(device.default_memory().is_ok());
                        assert!(device.memory_statistics().is_err());
                        assert_eq!(format!("{device}"), format!("CpuDevice(id={})", index));
                        assert_eq!(format!("{device:?}"), format!("Device[TFRT_CPU_{}]", index));
                    }
                    TestPlatform::Metal => {
                        assert!(device.default_memory().is_err());
                        assert!(device.memory_statistics().is_err());
                        assert_eq!(format!("{device}"), format!("METAL(id={})", index));
                        assert_eq!(format!("{device:?}"), format!("Device[METAL:{}]", index));
                    }
                    _ => {
                        assert!(device.default_memory().is_ok());
                        assert!(device.memory_statistics().is_ok());
                        assert!(!format!("{device}").is_empty());
                        assert!(!format!("{device:?}").is_empty());
                    }
                }
            }
            assert_eq!(devices[0], devices[0]);
            if devices.len() > 1 {
                assert_eq!(devices[1], devices[1]);
                assert_ne!(devices[0], devices[1]);
            }
            if devices.len() > 2 {
                assert_ne!(devices[1], devices[2]);
            }

            // Test creating a [`Device`] from a null pointer.
            assert!(matches!(
                unsafe { Device::from_c_api(std::ptr::null_mut(), plugin.api()) },
                Err(Error::InvalidArgument { message, .. })
                    if message == "the provided PJRT device handle is a null pointer",
            ));
        });
    }

    #[test]
    fn test_device_poison_execution() {
        // To test device execution poisoning, we need to create a program that takes a non-trivial amount of time to
        // execute. We use a simple matrix multiplication program over reasonably sized matrices for this purpose.
        // First, we construct and compile the program.
        let client = test_cpu_client();
        let device = client.addressable_devices().unwrap()[0].clone();
        let program = Program::Mlir {
            bytecode: indoc! {"
                module {
                  func.func @main(%arg0: tensor<4096x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
                    %0 = stablehlo.dot_general \
                          %arg0, \
                          %arg1, \
                          batching_dims = [] x [], \
                          contracting_dims = [1] x [0] \
                      : (tensor<4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
                    return %0 : tensor<4096x4096xf32>
                  }
                }
            "}
            .as_bytes()
            .to_vec(),
        };
        let compilation_options = CompilationOptions {
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
        };
        let executable = client.compile(&program, &compilation_options).unwrap();

        // Then, we kick off the execution of this program.
        let launch_id = 17usize;
        let mut outputs = executable
            .execute(
                vec![ExecutionDeviceInputs {
                    inputs: &[
                        ExecutionInput {
                            buffer: client
                                .buffer(
                                    &[0u8; 4096 * 4096 * size_of::<f32>()],
                                    BufferType::F32,
                                    [4096u64, 4096u64],
                                    None,
                                    device.clone(),
                                    None,
                                )
                                .unwrap(),
                            donatable: false,
                        },
                        ExecutionInput {
                            buffer: client
                                .buffer(
                                    &[0u8; 4096 * 4096 * size_of::<f32>()],
                                    BufferType::F32,
                                    [4096u64, 4096u64],
                                    None,
                                    device.clone(),
                                    None,
                                )
                                .unwrap(),
                            donatable: false,
                        },
                    ],
                    ..Default::default()
                }],
                launch_id,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let output = outputs.remove(0);

        // Finally, poison the program execution.
        assert_eq!(device.poison_execution(launch_id as i32, Error::aborted("test poison error")), Ok(true));
        assert!(matches!(
            output.done.r#await(),
            Err(Error::Aborted { message, .. }) if message == "test poison error",
        ));
    }

    #[test]
    fn test_device_description() {
        let client = test_cpu_client();
        let devices = client.devices().unwrap();
        let descriptions = devices.iter().map(|device| device.description().unwrap()).collect::<Vec<_>>();
        assert!(descriptions.len() > 1);
        let description = &descriptions[1];
        assert_eq!(description.id(), Ok(1));
        assert_eq!(description.kind().map(|kind| kind.to_string()), Ok("cpu".to_string()));
        assert_eq!(description.process_index(), Ok(0));
        assert!(description.attribute("__test__").is_err());
        assert_eq!(description.attributes().map(|attributes| attributes.is_empty()), Ok(true));
        assert_eq!(descriptions[0], descriptions[0]);
        assert_eq!(descriptions[1], descriptions[1]);
        assert_ne!(descriptions[0], descriptions[1]);
        assert_ne!(descriptions[1], descriptions[0]);
        assert_eq!(format!("{description}"), "CpuDevice(id=1)");
        assert_eq!(format!("{description:?}"), "DeviceDescription[TFRT_CPU_1]");

        // Test creating a [`DeviceDescription`] from a null pointer.
        assert!(matches!(
            unsafe { DeviceDescription::from_c_api(std::ptr::null_mut(), client.api()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT device description handle is a null pointer",
        ));
    }

    #[test]
    fn test_device_assignment() {
        let client = test_cpu_client();
        let device_assignment = client.default_device_assignment(2, 4).unwrap();
        assert_eq!(
            device_assignment,
            DeviceAssignment { replica_count: 2, computation_count: 4, assignment: vec![0, 1, 2, 3, 4, 5, 6, 7] },
        );
        assert_eq!(device_assignment.device_id(0, 0), Ok(0));
        assert_eq!(device_assignment.device_id(0, 1), Ok(1));
        assert_eq!(device_assignment.device_id(0, 2), Ok(2));
        assert_eq!(device_assignment.device_id(0, 3), Ok(3));
        assert!(device_assignment.device_id(0, 4).is_err());
        assert_eq!(device_assignment.device_id(1, 0), Ok(4));
        assert_eq!(device_assignment.device_id(1, 1), Ok(5));
        assert_eq!(device_assignment.device_id(1, 2), Ok(6));
        assert_eq!(device_assignment.device_id(1, 3), Ok(7));
        assert!(device_assignment.device_id(1, 4).is_err());
        assert!(device_assignment.device_id(2, 0).is_err());
        assert!(device_assignment.device_id(2, 4).is_err());
        assert_eq!(device_assignment.replica_id(0), Ok(0));
        assert_eq!(device_assignment.replica_id(1), Ok(0));
        assert_eq!(device_assignment.replica_id(2), Ok(0));
        assert_eq!(device_assignment.replica_id(3), Ok(0));
        assert_eq!(device_assignment.replica_id(4), Ok(1));
        assert_eq!(device_assignment.replica_id(5), Ok(1));
        assert_eq!(device_assignment.replica_id(6), Ok(1));
        assert_eq!(device_assignment.replica_id(7), Ok(1));
        assert!(device_assignment.replica_id(8).is_err());
        assert_eq!(device_assignment.computation_id(0), Ok(0));
        assert_eq!(device_assignment.computation_id(1), Ok(1));
        assert_eq!(device_assignment.computation_id(2), Ok(2));
        assert_eq!(device_assignment.computation_id(3), Ok(3));
        assert_eq!(device_assignment.computation_id(4), Ok(0));
        assert_eq!(device_assignment.computation_id(5), Ok(1));
        assert_eq!(device_assignment.computation_id(6), Ok(2));
        assert_eq!(device_assignment.computation_id(7), Ok(3));
        assert!(device_assignment.computation_id(8).is_err());
    }

    #[test]
    fn test_device_assignment_serialization() {
        let client = test_cpu_client();
        let device_assignment = client.default_device_assignment(2, 4).unwrap();

        let device_assignment_proto = device_assignment.proto().unwrap();
        let roundtripped_device_assignment = client.device_assignment_from_proto(device_assignment_proto).unwrap();
        assert_eq!(roundtripped_device_assignment, device_assignment);

        let serialized_device_assignment = device_assignment.serialize().unwrap();
        let roundtripped_device_assignment = serialized_device_assignment.deserialize().unwrap();
        assert_eq!(roundtripped_device_assignment, device_assignment);

        let roundtripped_device_assignment =
            client.deserialize_device_assignment(serialized_device_assignment.data()).unwrap();
        assert_eq!(roundtripped_device_assignment, device_assignment);
    }
}
