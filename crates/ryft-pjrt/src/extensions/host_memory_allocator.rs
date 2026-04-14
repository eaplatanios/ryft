use std::ffi::c_void;

use crate::{Api, Client, Error, Plugin, invoke_pjrt_api_error_fn};

/// The PJRT host memory allocator extension provides access to [XLA](https://openxla.org/xla) host-memory allocation
/// APIs that return owned pointers together with backend-provided deleter callbacks. The extension is optional for
/// PJRT [`Plugin`]s and _experimental_, meaning that incompatible changes may be introduced at any time, including
/// changes that break _Application Binary Interface (ABI)_ compatibility.
#[derive(Copy, Clone)]
pub struct HostMemoryAllocatorExtension {
    /// Handle that represents this [`HostMemoryAllocatorExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_HostMemoryAllocator_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl HostMemoryAllocatorExtension {
    /// Constructs a new [`HostMemoryAllocatorExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle if the type of that PJRT
    /// extension matches the PJRT host-memory allocator extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_HostMemoryAllocator {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_HostMemoryAllocator_Extension`](ffi::PJRT_HostMemoryAllocator_Extension) that corresponds
    /// to this [`HostMemoryAllocatorExtension`] and which can be passed to functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_HostMemoryAllocator_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for HostMemoryAllocatorExtension {}
unsafe impl Sync for HostMemoryAllocatorExtension {}

impl Client<'_> {
    /// Attempts to load the [`HostMemoryAllocatorExtension`] from this [`Client`] and returns
    /// [`Error::Unimplemented`] if it is not provided by the underlying [`Plugin`].
    pub fn host_memory_allocator_extension(&self) -> Result<HostMemoryAllocatorExtension, Error> {
        self.api().host_memory_allocator_extension()
    }

    /// Allocates `size` bytes of backend-managed host memory on the provided Non-Uniform Memory Access (NUMA) node and
    /// returns an owned allocation that will invoke the backend-provided deleter callback on drop. The returned memory
    /// is backend-owned and may be uninitialized.
    pub fn host_memory_allocate(&self, size: usize, numa_node_index: i32) -> Result<HostMemoryAllocation, Error> {
        use ffi::PJRT_HostMemoryAllocator_Allocate_Args;
        let extension = self.host_memory_allocator_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_HostMemoryAllocator_Extension => extension,
            PJRT_HostMemoryAllocator_Allocate,
            {
                client = self.to_c_api(),
                size = size,
                numa_node = numa_node_index,
            },
            { ptr, deleter, deleter_arg },
        )
        .and_then(|(pointer, deleter, deleter_arg)| {
            if size != 0 && pointer.is_null() {
                Err(Error::invalid_argument(
                    "the host memory allocator returned a null pointer for a non-empty allocation",
                ))
            } else {
                Ok(HostMemoryAllocation { pointer, size, deleter, deleter_arg })
            }
        })
    }
}

impl Plugin {
    /// Attempts to load the [`HostMemoryAllocatorExtension`] from this [`Plugin`] and returns
    /// [`Error::Unimplemented`] if it is not provided by this [`Plugin`].
    pub fn host_memory_allocator_extension(&self) -> Result<HostMemoryAllocatorExtension, Error> {
        self.api().host_memory_allocator_extension()
    }
}

impl Api {
    /// Attempts to load the [`HostMemoryAllocatorExtension`] from this [`Api`] and returns
    /// [`Error::Unimplemented`] if it is not provided by the underlying [`Plugin`].
    pub(crate) fn host_memory_allocator_extension(&self) -> Result<HostMemoryAllocatorExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let host_memory_allocator_extension = HostMemoryAllocatorExtension::from_c_api(extension, *self);
                if let Some(host_memory_allocator_extension) = host_memory_allocator_extension {
                    return Ok(host_memory_allocator_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the host memory allocator extension is not provided by the PJRT plugin"))
        }
    }
}

/// Owned host memory allocation returned by [`Client::host_memory_allocate`]. The underlying memory is backend-owned
/// and may be uninitialized. This wrapper only tracks the pointer, size, and backend-provided deleter callback so that
/// the allocation can be released safely when dropped.
pub struct HostMemoryAllocation {
    /// Pointer to the allocated host memory.
    pointer: *mut u8,

    /// Number of bytes in this [`HostMemoryAllocation`].
    size: usize,

    /// Backend-provided deleter callback for this [`HostMemoryAllocation`].
    deleter: Option<unsafe extern "C" fn(ptr: *mut c_void, arg: *mut c_void)>,

    /// Opaque deleter state that must be passed back to the backend-provided deleter callback.
    deleter_arg: *mut c_void,
}

impl HostMemoryAllocation {
    /// Returns a pointer to the start of this [`HostMemoryAllocation`]. This is unsafe because the returned pointer
    /// refers to backend-owned memory whose initialization state, aliasing constraints, and validity for dereferencing
    /// are not enforced by Rust. This API is exposed so callers can pass the backend-owned allocation to external FFI
    /// or backend-specific code.
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.pointer
    }

    /// Returns a mutable pointer to the start of this [`HostMemoryAllocation`]. This is unsafe because the returned
    /// pointer refers to backend-owned memory whose initialization state, aliasing constraints, and validity for
    /// dereferencing are not enforced by Rust. This API is exposed so callers can pass the backend-owned allocation
    /// to external FFI or backend-specific code.
    pub unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.pointer
    }

    /// Returns the number of bytes in this [`HostMemoryAllocation`].
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if this [`HostMemoryAllocation`] contains zero bytes.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

unsafe impl Send for HostMemoryAllocation {}
unsafe impl Sync for HostMemoryAllocation {}

impl Drop for HostMemoryAllocation {
    fn drop(&mut self) {
        if let Some(deleter) = self.deleter {
            unsafe { deleter(self.pointer.cast(), self.deleter_arg) };
        }
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::clients::ffi::PJRT_Client;
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;

    pub const PJRT_API_HOST_MEMORY_ALLOCATOR_EXTENSION_VERSION: usize = 0;

    #[repr(C)]
    pub struct PJRT_HostMemoryAllocator_Allocate_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub size: usize,
        pub numa_node: std::ffi::c_int,
        pub ptr: *mut u8,
        pub deleter: Option<unsafe extern "C" fn(ptr: *mut std::ffi::c_void, arg: *mut std::ffi::c_void)>,
        pub deleter_arg: *mut std::ffi::c_void,
    }

    impl PJRT_HostMemoryAllocator_Allocate_Args {
        pub fn new(client: *mut PJRT_Client, size: usize, numa_node: std::ffi::c_int) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                size,
                numa_node,
                ptr: std::ptr::null_mut(),
                deleter: None,
                deleter_arg: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_HostMemoryAllocator_Allocate =
        unsafe extern "C" fn(args: *mut PJRT_HostMemoryAllocator_Allocate_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_HostMemoryAllocator_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_HostMemoryAllocator_Allocate: Option<PJRT_HostMemoryAllocator_Allocate>,
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::tests::{test_cpu_client, test_cpu_plugin};

    use super::HostMemoryAllocation;

    #[test]
    fn test_host_memory_allocator_extension() {
        assert!(test_cpu_plugin().host_memory_allocator_extension().is_err());
        assert!(test_cpu_client().host_memory_allocator_extension().is_err());
    }

    #[test]
    fn test_host_memory_allocation_drop_calls_deleter() {
        unsafe extern "C" fn test_deleter(pointer: *mut c_void, deleter_arg: *mut c_void) {
            let counter = unsafe { &*(deleter_arg as *const AtomicUsize) };
            counter.fetch_add(1, Ordering::SeqCst);
            drop(unsafe { Box::from_raw(pointer as *mut [u8; 4]) });
        }

        let counter = AtomicUsize::new(0);
        let allocation = HostMemoryAllocation {
            pointer: Box::into_raw(Box::new([0u8; 4])).cast(),
            size: 4,
            deleter: Some(test_deleter),
            deleter_arg: &counter as *const AtomicUsize as *mut c_void,
        };
        drop(allocation);

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
