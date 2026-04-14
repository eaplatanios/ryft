use std::ffi::c_void;

use crate::{Api, Client, Error, Plugin, invoke_pjrt_api_error_fn};

/// The PJRT host memory allocator extension provides access to [XLA](https://openxla.org/xla) host-memory allocation
/// APIs that return an owned pointer together with a backend-provided deleter callback. The extension is optional for
/// PJRT [`Plugin`]s and _experimental_, meaning that incompatible changes may be introduced at any time, including
/// changes that break _Application Binary Interface (ABI)_ compatibility.
///
/// Refer to the upstream
/// [`host_memory_allocator_extension.h`](https://github.com/openxla/xla/blob/main/xla/pjrt/extensions/host_allocator/host_memory_allocator/host_memory_allocator_extension.h)
/// header for the canonical C API surface.
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

    /// Allocates `size` bytes of host memory for `client` on the provided NUMA node and returns an owned allocation
    /// that will invoke the backend-provided deleter callback on drop.
    ///
    /// The returned memory is backend-owned and may be uninitialized.
    pub fn allocate(&self, client: &Client<'_>, size: usize, numa_node: i32) -> Result<HostMemoryAllocation, Error> {
        use ffi::PJRT_HostMemoryAllocator_Allocate_Args;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_HostMemoryAllocator_Extension => self,
            PJRT_HostMemoryAllocator_Allocate,
            {
                client = client.to_c_api(),
                size = size,
                numa_node = numa_node,
            },
            { ptr, deleter, deleter_arg },
        )
        .and_then(|(ptr, deleter, deleter_arg)| HostMemoryAllocation::from_raw_parts(ptr, size, deleter, deleter_arg))
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

    /// Allocates `size` bytes of backend-managed host memory on the provided NUMA node.
    ///
    /// Refer to the documentation of [`HostMemoryAllocatorExtension::allocate`] for more information.
    pub fn host_memory_allocate(&self, size: usize, numa_node: i32) -> Result<HostMemoryAllocation, Error> {
        self.host_memory_allocator_extension()?.allocate(self, size, numa_node)
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

/// An owned allocation returned by [`HostMemoryAllocatorExtension::allocate`] or [`Client::host_memory_allocate`].
///
/// The underlying memory is backend-owned and may be uninitialized. This wrapper only tracks the pointer, size, and
/// backend-provided deleter callback so that the allocation can be released safely when dropped.
pub struct HostMemoryAllocation {
    /// Pointer to the allocated host memory.
    pointer: *mut u8,

    /// Number of bytes in the allocation.
    size: usize,

    /// Backend-provided deleter callback for this allocation.
    deleter: Option<unsafe extern "C" fn(ptr: *mut c_void, arg: *mut c_void)>,

    /// Opaque deleter state that must be passed back to the backend-provided deleter callback.
    deleter_arg: *mut c_void,
}

impl HostMemoryAllocation {
    /// Constructs a new [`HostMemoryAllocation`] from raw parts returned by the PJRT C API.
    fn from_raw_parts(
        pointer: *mut u8,
        size: usize,
        deleter: Option<unsafe extern "C" fn(ptr: *mut c_void, arg: *mut c_void)>,
        deleter_arg: *mut c_void,
    ) -> Result<Self, Error> {
        if size != 0 && pointer.is_null() {
            Err(Error::invalid_argument("the host memory allocator returned a null pointer for a non-empty allocation"))
        } else {
            Ok(Self { pointer, size, deleter, deleter_arg })
        }
    }

    /// Returns a const pointer to the start of this allocation.
    pub fn as_ptr(&self) -> *const u8 {
        self.pointer
    }

    /// Returns a mutable pointer to the start of this allocation.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.pointer
    }

    /// Returns the number of bytes in this allocation.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if this allocation contains zero bytes.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Consumes this allocation and returns its raw parts.
    ///
    /// The caller becomes responsible for eventually invoking the returned deleter callback, if present.
    pub fn into_raw_parts(
        mut self,
    ) -> (*mut u8, usize, Option<unsafe extern "C" fn(ptr: *mut c_void, arg: *mut c_void)>, *mut c_void) {
        let raw_parts = (self.pointer, self.size, self.deleter, self.deleter_arg);
        self.pointer = std::ptr::null_mut();
        self.size = 0;
        self.deleter = None;
        self.deleter_arg = std::ptr::null_mut();
        raw_parts
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

    #[test]
    fn test_host_memory_allocation_into_raw_parts_disarms_drop() {
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
        let (pointer, size, deleter, deleter_arg) = allocation.into_raw_parts();

        assert_eq!(size, 4);
        assert!(!pointer.is_null());
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        unsafe { deleter.unwrap()(pointer.cast(), deleter_arg) };

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
