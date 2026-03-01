use prost::Message;

use crate::macros::{invoke_pjrt_api_error_fn, invoke_pjrt_api_void_fn};
use crate::{Api, Client, Error, Executable, Plugin, slice_from_c_api};

/// The PJRT executable metadata extension provides capabilities around querying metadata (e.g., executable fingerprints
/// and backend metadata) for compiled [`Executable`]s. The extension is both optional for PJRT [`Plugin`]s and
/// _experimental_, meaning that incompatible changes may be introduced at any time, including changes that break
/// _Application Binary Interface (ABI)_ compatibility.
#[derive(Copy, Clone)]
pub struct ExecutableMetadataExtension {
    /// Handle that represents this [`ExecutableMetadataExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_ExecutableMetadata_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl ExecutableMetadataExtension {
    /// Constructs a new [`ExecutableMetadataExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle that came from a function in the
    /// PJRT C API if the type of that PJRT extension matches the PJRT executable metadata extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_ExecutableMetadata {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_ExecutableMetadata_Extension`](ffi::PJRT_ExecutableMetadata_Extension)
    /// that corresponds to this [`ExecutableMetadataExtension`] and which can be passed to
    /// functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_ExecutableMetadata_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for ExecutableMetadataExtension {}
unsafe impl Sync for ExecutableMetadataExtension {}

impl Client<'_> {
    /// Attempts to load the [`ExecutableMetadataExtension`] from this [`Client`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub fn executable_metadata_extension(&self) -> Result<ExecutableMetadataExtension, Error> {
        self.api().executable_metadata_extension()
    }
}

impl Plugin {
    /// Attempts to load the [`ExecutableMetadataExtension`] from this [`Plugin`] and returns [`Error::Unimplemented`]
    /// if it is not provided by this [`Plugin`].
    pub fn executable_metadata_extension(&self) -> Result<ExecutableMetadataExtension, Error> {
        self.api().executable_metadata_extension()
    }
}

impl Api {
    /// Attempts to load the [`ExecutableMetadataExtension`] from this [`Api`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub(crate) fn executable_metadata_extension(&self) -> Result<ExecutableMetadataExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let executable_metadata_extension = ExecutableMetadataExtension::from_c_api(extension, *self);
                if let Some(executable_metadata_extension) = executable_metadata_extension {
                    return Ok(executable_metadata_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the executable metadata extension is not provided by the PJRT plugin"))
        }
    }
}

impl Executable {
    /// Returns the [`ExecutableMetadata`](crate::protos::ExecutableMetadata) associated with this [`Executable`].
    pub fn metadata(&self) -> Result<crate::protos::ExecutableMetadata, Error> {
        use ffi::PJRT_ExecutableMetadata_DestroySerializedMetadata_Args;
        use ffi::PJRT_ExecutableMetadata_GetExecutableMetadata_Args;
        let extension = self.api().executable_metadata_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_ExecutableMetadata_Extension => extension,
            PJRT_ExecutableMetadata_GetExecutableMetadata,
            { executable = self.to_c_api() },
            { metadata },
        )
        .and_then(|metadata| {
            let (ptr, len) = unsafe { ((*metadata).serialized_metadata, (*metadata).serialized_metadata_size) };
            let serialized_metadata = unsafe { slice_from_c_api(ptr as *const u8, len) }.to_vec();
            let metadata_proto =
                crate::protos::ExecutableMetadata::decode(serialized_metadata.as_slice()).map_err(|error| {
                    Error::invalid_argument(format!(
                        "failed to deserialize executable metadata protobuf returned by PJRT plugin: {error}",
                    ))
                });
            let _ = invoke_pjrt_api_void_fn!(
                @extension ffi::PJRT_ExecutableMetadata_Extension => extension,
                PJRT_ExecutableMetadata_DestroySerializedMetadata,
                { metadata = metadata },
            );
            metadata_proto
        })
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;
    use crate::programs::ffi::PJRT_Executable;

    pub const PJRT_API_EXECUTABLE_METADATA_EXTENSION_VERSION: usize = 1;

    #[repr(C)]
    pub struct PJRT_ExecutableMetadata {
        pub serialized_metadata: *const std::ffi::c_char,
        pub serialized_metadata_size: usize,
    }

    impl PJRT_ExecutableMetadata {
        pub fn new() -> Self {
            Self { serialized_metadata: std::ptr::null(), serialized_metadata_size: 0 }
        }
    }

    #[repr(C)]
    pub struct PJRT_ExecutableMetadata_GetExecutableMetadata_Args {
        pub executable: *mut PJRT_Executable,
        pub metadata: *mut PJRT_ExecutableMetadata,
    }

    impl PJRT_ExecutableMetadata_GetExecutableMetadata_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self { executable, metadata: std::ptr::null_mut() }
        }
    }

    // Extracts executable metadata serialized as `ExecutableMetadata` protobuf bytes.
    pub type PJRT_ExecutableMetadata_GetExecutableMetadata =
        unsafe extern "C" fn(args: *mut PJRT_ExecutableMetadata_GetExecutableMetadata_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_ExecutableMetadata_DestroySerializedMetadata_Args {
        pub metadata: *mut PJRT_ExecutableMetadata,
    }

    impl PJRT_ExecutableMetadata_DestroySerializedMetadata_Args {
        pub fn new(metadata: *mut PJRT_ExecutableMetadata) -> Self {
            Self { metadata }
        }
    }

    // Destroys serialized metadata and releases memory owned by the plugin.
    pub type PJRT_ExecutableMetadata_DestroySerializedMetadata =
        unsafe extern "C" fn(args: *mut PJRT_ExecutableMetadata_DestroySerializedMetadata_Args);

    #[repr(C)]
    pub struct PJRT_ExecutableMetadata_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_ExecutableMetadata_GetExecutableMetadata: Option<PJRT_ExecutableMetadata_GetExecutableMetadata>,
        pub PJRT_ExecutableMetadata_DestroySerializedMetadata:
            Option<PJRT_ExecutableMetadata_DestroySerializedMetadata>,
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{test_cpu_client, test_cpu_plugin};

    #[test]
    fn test_executable_metadata_extension() {
        assert!(test_cpu_plugin().executable_metadata_extension().is_err());
        assert!(test_cpu_client().executable_metadata_extension().is_err());
    }

    // TODO(eaplatanios): Add more tests once there is a PJRT plugin that provides this extension.
}
