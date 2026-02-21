use crate::{Api, Client, Error, Plugin, invoke_pjrt_api_error_fn};

/// The PJRT host allocator extension provides capabilities for allocating and freeing host memory through PJRT
/// [`Plugin`]-defined allocators. The extension is both optional for PJRT [`Plugin`]s and _experimental_, meaning
/// that incompatible changes may be introduced at any time, including changes that break _Application Binary
/// Interface (ABI)_ compatibility.
#[derive(Copy, Clone)]
pub struct HostAllocatorExtension {
    /// Handle that represents this [`HostAllocatorExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_HostAllocator_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl HostAllocatorExtension {
    /// Constructs a new [`HostAllocatorExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle if the type of that PJRT
    /// extension matches the PJRT host allocator extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_HostAllocator {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_HostAllocator_Extension`](ffi::PJRT_HostAllocator_Extension) that corresponds
    /// to this [`HostAllocatorExtension`] and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_HostAllocator_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for HostAllocatorExtension {}
unsafe impl Sync for HostAllocatorExtension {}

impl Client<'_> {
    /// Attempts to load the [`HostAllocatorExtension`] from this [`Client`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub fn host_allocator_extension(&self) -> Result<HostAllocatorExtension, Error> {
        self.api().host_allocator_extension()
    }

    /// Returns the preferred alignment of the host allocator of this [`Client`].
    pub fn host_allocator_get_preferred_alignment(&self) -> Result<usize, Error> {
        use ffi::PJRT_HostAllocator_GetPreferredAlignment_Args;
        let extension = self.host_allocator_extension()?;
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_HostAllocator_GetPreferredAlignment,
            { client = self.to_c_api() },
            { preferred_alignment },
        )
    }

    /// Allocates `size` bytes using the host allocator of this [`Client`]
    /// returning a raw pointer to the allocated memory.
    pub fn host_allocator_allocate(&self, size: usize, alignment: usize) -> Result<*mut std::ffi::c_void, Error> {
        use ffi::PJRT_HostAllocator_Allocate_Args;
        let extension = self.host_allocator_extension()?;
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_HostAllocator_Allocate,
            {
                client = self.to_c_api(),
                size = size,
                alignment = alignment,
            },
            { ptr },
        )
    }

    /// Frees the memory pointed to by `ptr` assuming it was allocated using the host allocator of this [`Client`].
    pub unsafe fn host_allocator_free(&self, ptr: *mut std::ffi::c_void) -> Result<(), Error> {
        use ffi::PJRT_HostAllocator_Free_Args;
        let extension = self.host_allocator_extension()?;
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_HostAllocator_Free,
            {
                client = self.to_c_api(),
                ptr = ptr,
            },
        )
    }
}

impl Plugin {
    /// Attempts to load the [`HostAllocatorExtension`] from this [`Plugin`] and returns [`Error::Unimplemented`]
    /// if it is not provided by this [`Plugin`].
    pub fn host_allocator_extension(&self) -> Result<HostAllocatorExtension, Error> {
        self.api().host_allocator_extension()
    }
}

impl Api {
    /// Attempts to load the [`HostAllocatorExtension`] from this [`Api`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub(crate) fn host_allocator_extension(&self) -> Result<HostAllocatorExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let host_allocator_extension = HostAllocatorExtension::from_c_api(extension, *self);
                if let Some(host_allocator_extension) = host_allocator_extension {
                    return Ok(host_allocator_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the host allocator extension is not provided by the PJRT plugin"))
        }
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::clients::ffi::PJRT_Client;
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;

    pub const PJRT_API_HOST_ALLOCATOR_EXTENSION_VERSION: usize = 0;

    #[repr(C)]
    pub struct PJRT_HostAllocator_GetPreferredAlignment_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub preferred_alignment: usize,
    }

    impl PJRT_HostAllocator_GetPreferredAlignment_Args {
        pub fn new(client: *mut PJRT_Client) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                preferred_alignment: 0,
            }
        }
    }

    pub type PJRT_HostAllocator_GetPreferredAlignment =
        unsafe extern "C" fn(args: *mut PJRT_HostAllocator_GetPreferredAlignment_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_HostAllocator_Allocate_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub size: usize,
        pub alignment: usize,
        pub ptr: *mut std::ffi::c_void,
    }

    impl PJRT_HostAllocator_Allocate_Args {
        pub fn new(client: *mut PJRT_Client, size: usize, alignment: usize) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                size,
                alignment,
                ptr: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_HostAllocator_Allocate =
        unsafe extern "C" fn(args: *mut PJRT_HostAllocator_Allocate_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_HostAllocator_Free_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub ptr: *mut std::ffi::c_void,
    }

    impl PJRT_HostAllocator_Free_Args {
        pub fn new(client: *mut PJRT_Client, ptr: *mut std::ffi::c_void) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), client, ptr }
        }
    }

    pub type PJRT_HostAllocator_Free = unsafe extern "C" fn(args: *mut PJRT_HostAllocator_Free_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_HostAllocator_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_HostAllocator_GetPreferredAlignment: Option<PJRT_HostAllocator_GetPreferredAlignment>,
        pub PJRT_HostAllocator_Allocate: Option<PJRT_HostAllocator_Allocate>,
        pub PJRT_HostAllocator_Free: Option<PJRT_HostAllocator_Free>,
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{test_cpu_client, test_cpu_plugin};

    #[test]
    fn test_host_allocator_extension() {
        assert!(test_cpu_plugin().host_allocator_extension().is_err());
        assert!(test_cpu_client().host_allocator_extension().is_err());
    }

    // TODO(eaplatanios): Add more tests once there is a PJRT plugin that provides this extension.
}
