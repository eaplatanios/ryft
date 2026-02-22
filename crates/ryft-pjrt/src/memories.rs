use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::{Api, Device, Error, invoke_pjrt_api_error_fn, slice_from_c_api, str_from_c_api};

/// Type alias used to represent [`Memory`] IDs, which are unique among all memories of the same type.
pub type MemoryId = usize;

/// Type alias used to represent platform-dependent IDs that uniquely identify the kinds of [`Memory`]s.
pub type MemoryKindId = usize;

/// Memory space managed by a PJRT [`Plugin`](crate::Plugin). Memory spaces can be used to describe locations of memory.
/// These can either be _unpinned_ and free to live anywhere but be accessible from a [`Device`], or they can be
/// _pinned_ and must live on a specific [`Device`]. Memory spaces know their associated buffers of data and the
/// [`Device`]s (note the plural) that they are associated with, as well as the [`Client`](crate::Client) that they
/// are part of.
///
/// Note that the `'o` lifetime parameter captures the fact that [`Memory`]s are always owned by some other object
/// (e.g., a [`Client`](crate::Client) or a [`Device`]) and makes sure that that other object stays alive for at least
/// as long as all associated [`Memory`]s are alive.
#[derive(Copy, Clone)]
pub struct Memory<'o> {
    /// Handle that represents this [`Memory`] in the PJRT C API.
    handle: *mut ffi::PJRT_Memory,

    /// Underlying PJRT [`Api`].
    api: Api,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`Memory`].
    owner: PhantomData<&'o ()>,
}

impl Memory<'_> {
    /// Constructs a new [`Memory`] from the provided [`PJRT_Memory`](ffi::PJRT_Memory) handle that came
    /// from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(handle: *mut ffi::PJRT_Memory, api: Api) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT memory handle is a null pointer"))
        } else {
            Ok(Self { handle, api, owner: PhantomData })
        }
    }

    /// Returns the [`PJRT_Memory`](ffi::PJRT_Memory) that corresponds to this [`Memory`] and which can
    /// be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_Memory {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }

    /// ID of this [`Memory`] that is unique among all memories of the same type.
    pub fn id(&self) -> Result<MemoryId, Error> {
        use ffi::PJRT_Memory_Id_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Memory_Id, { memory = self.to_c_api() }, { id })
            .map(|id| id as usize)
    }

    /// Platform-dependent ID that uniquely identifies the kind of this [`Memory`].
    pub fn kind_id(&self) -> Result<MemoryKindId, Error> {
        use ffi::PJRT_Memory_Kind_Id_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Memory_Kind_Id, { memory = self.to_c_api() }, { kind_id })
            .map(|id| id as usize)
    }

    /// Platform-dependent string that uniquely identifies the kind of this [`Memory`].
    pub fn kind(&'_ self) -> Result<Cow<'_, str>, Error> {
        use ffi::PJRT_Memory_Kind_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Memory_Kind,
            { memory = self.to_c_api() },
            { kind, kind_size },
        )
        .map(|(string, string_len)| str_from_c_api(string, string_len))
    }

    /// [`Device`]s that can address this [`Memory`].
    pub fn addressable_by_devices(&'_ self) -> Result<Vec<Device<'_>>, Error> {
        use ffi::PJRT_Memory_AddressableByDevices_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Memory_AddressableByDevices,
            { memory = self.to_c_api() },
            { devices, num_devices },
        )
        .and_then(|(devices, devices_count)| {
            unsafe { slice_from_c_api(devices, devices_count) }
                .iter()
                .map(|handle| unsafe { Device::from_c_api(*handle, self.api()) })
                .collect::<Result<Vec<_>, _>>()
        })
    }
}

impl Display for Memory<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ffi::PJRT_Memory_ToString_Args;
        match invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Memory_ToString,
            { memory = self.to_c_api() },
            { to_string, to_string_size },
        ) {
            Ok((string, string_len)) => write!(formatter, "{}", str_from_c_api(string, string_len)),
            Err(error) => write!(formatter, "<failed to render PJRT memory as string; {}>", error),
        }
    }
}

impl Debug for Memory<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ffi::PJRT_Memory_DebugString_Args;
        match invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Memory_DebugString,
            { memory = self.to_c_api() },
            { debug_string, debug_string_size },
        ) {
            Ok((string, string_len)) => write!(formatter, "{}", str_from_c_api(string, string_len)),
            Err(error) => write!(formatter, "<failed to render PJRT memory as debug string; {:?}>", error),
        }
    }
}

impl PartialEq for Memory<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.id().is_ok()
            && other.id().is_ok()
            && self.id() == other.id()
            && self.kind_id().is_ok()
            && other.kind_id().is_ok()
            && self.kind_id() == other.kind_id()
    }
}

impl Eq for Memory<'_> {}

/// Represents types that have a default [`Memory`] associated with them.
pub trait HasDefaultMemory {
    /// Returns the [`Memory`] that this object is associated with (e.g., the default memory of a [`Device`]).
    fn default_memory(&self) -> Memory<'_>;
}

impl HasDefaultMemory for Memory<'_> {
    fn default_memory(&self) -> Memory<'_> {
        *self
    }
}

/// Statistics about a [`Memory`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryStatistics {
    /// Number of bytes in use.
    pub bytes_in_use: u64,

    /// Peak number of bytes in use.
    pub peak_bytes_in_use: Option<u64>,

    /// Number of allocations thus far.
    pub allocation_count: Option<u64>,

    /// Size (in bytes) of the largest allocation encountered thus far.
    pub largest_allocation_size: Option<u64>,

    /// Upper limit on the number of user-allocatable bytes, if such a limit is known.
    pub bytes_limit: Option<u64>,

    /// Number of reserved bytes.
    pub reserved_bytes: Option<u64>,

    /// Peak number of reserved bytes.
    pub peak_reserved_bytes: Option<u64>,

    /// Upper limit on the number of reservable bytes, if such a limit is known.
    pub reservable_bytes_limit: Option<u64>,

    /// Size (in bytes) of the largest free memory block.
    pub largest_free_block_bytes: Option<u64>,

    /// Number of bytes held by the allocator. This may be higher than [`MemoryStatistics::bytes_in_use`] if the
    /// allocator holds a pool of memory, like the XLA Best-Fit with Coalescing (BFC) memory allocator.
    pub pool_bytes: Option<u64>,

    /// Peak number of bytes held by the allocator.
    pub peak_pool_bytes: Option<u64>,
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::devices::ffi::PJRT_Device;
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_Memory {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Memory_Id_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub memory: *mut PJRT_Memory,
        pub id: std::ffi::c_int,
    }

    impl PJRT_Memory_Id_Args {
        pub fn new(memory: *mut PJRT_Memory) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), memory, id: 0 }
        }
    }

    pub type PJRT_Memory_Id = unsafe extern "C" fn(args: *mut PJRT_Memory_Id_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Memory_Kind_Id_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub memory: *mut PJRT_Memory,
        pub kind_id: std::ffi::c_int,
    }

    impl PJRT_Memory_Kind_Id_Args {
        pub fn new(memory: *mut PJRT_Memory) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), memory, kind_id: 0 }
        }
    }

    pub type PJRT_Memory_Kind_Id = unsafe extern "C" fn(args: *mut PJRT_Memory_Kind_Id_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Memory_Kind_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub memory: *mut PJRT_Memory,
        pub kind: *const std::ffi::c_char,
        pub kind_size: usize,
    }

    impl PJRT_Memory_Kind_Args {
        pub fn new(memory: *mut PJRT_Memory) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                memory,
                kind: std::ptr::null_mut(),
                kind_size: 0,
            }
        }
    }

    pub type PJRT_Memory_Kind = unsafe extern "C" fn(args: *mut PJRT_Memory_Kind_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Memory_AddressableByDevices_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub memory: *mut PJRT_Memory,
        pub devices: *const *mut PJRT_Device,
        pub num_devices: usize,
    }

    impl PJRT_Memory_AddressableByDevices_Args {
        pub fn new(memory: *mut PJRT_Memory) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                memory,
                devices: std::ptr::null(),
                num_devices: 0,
            }
        }
    }

    pub type PJRT_Memory_AddressableByDevices =
        unsafe extern "C" fn(args: *mut PJRT_Memory_AddressableByDevices_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Memory_ToString_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub memory: *mut PJRT_Memory,
        pub to_string: *const std::ffi::c_char,
        pub to_string_size: usize,
    }

    impl PJRT_Memory_ToString_Args {
        pub fn new(memory: *mut PJRT_Memory) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                memory,
                to_string: std::ptr::null(),
                to_string_size: 0,
            }
        }
    }

    pub type PJRT_Memory_ToString = unsafe extern "C" fn(args: *mut PJRT_Memory_ToString_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Memory_DebugString_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub memory: *mut PJRT_Memory,
        pub debug_string: *const std::ffi::c_char,
        pub debug_string_size: usize,
    }

    impl PJRT_Memory_DebugString_Args {
        pub fn new(memory: *mut PJRT_Memory) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                memory,
                debug_string: std::ptr::null(),
                debug_string_size: 0,
            }
        }
    }

    pub type PJRT_Memory_DebugString = unsafe extern "C" fn(args: *mut PJRT_Memory_DebugString_Args) -> *mut PJRT_Error;
}

#[cfg(test)]
mod tests {
    use crate::tests::test_cpu_client;
    use crate::{Error, Memory};

    #[test]
    fn test_memory() {
        let client = test_cpu_client();
        let addressable_memories = client.addressable_memories().unwrap();
        assert_eq!(addressable_memories.len(), 24);

        let memory_0 = &addressable_memories[0];
        let memory_1 = &addressable_memories[1];
        let memory_2 = &addressable_memories[2];
        let memory_3 = &addressable_memories[3];
        assert_eq!(memory_0.id().unwrap(), 0);
        assert_eq!(memory_1.id().unwrap(), 1);
        assert_eq!(memory_2.id().unwrap(), 2);
        assert_eq!(memory_0.kind_id().unwrap(), 42097224);
        assert_eq!(memory_0.kind_id(), memory_3.kind_id());
        assert_eq!(memory_0.kind().unwrap(), "device");
        assert_eq!(memory_1.kind().unwrap(), "pinned_host");
        assert_eq!(memory_2.kind().unwrap(), "unpinned_host");
        assert_eq!(memory_0, memory_0);
        assert_ne!(memory_0, memory_3);

        let devices_0 = memory_0.addressable_by_devices().unwrap();
        let devices_1 = memory_1.addressable_by_devices().unwrap();
        let devices_2 = memory_2.addressable_by_devices().unwrap();
        let devices_3 = memory_3.addressable_by_devices().unwrap();
        assert_eq!(devices_0.len(), 1);
        assert_eq!(devices_1.len(), 1);
        assert_eq!(devices_2.len(), 1);
        assert_eq!(devices_3.len(), 1);
        assert_eq!(devices_0[0], devices_1[0]);
        assert_eq!(devices_0[0], devices_2[0]);
        assert_ne!(devices_0[0], devices_3[0]);

        assert_eq!(format!("{memory_0}"), "CPU_DEVICE_0");
        assert_eq!(format!("{memory_1}"), "PINNED_HOST_1");
        assert_eq!(format!("{memory_2}"), "UNPINNED_HOST_2");
        assert_eq!(format!("{memory_0:?}"), "CpuDeviceMemory(id=0, process_index=0, client=cpu)");
        assert_eq!(format!("{memory_1:?}"), "PinnedHostMemory(id=1, process_index=0, client=cpu)");
        assert_eq!(format!("{memory_2:?}"), "UnpinnedHostMemorySpace(id=2, process_index=0, client=cpu)");

        // Test creating a [`Memory`] from a null pointer.
        assert!(matches!(
            unsafe { Memory::from_c_api(std::ptr::null_mut(), client.api()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT memory handle is a null pointer",
        ));
    }
}
