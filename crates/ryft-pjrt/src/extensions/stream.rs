use crate::{Api, Buffer, Client, Device, Error, Plugin, invoke_pjrt_api_error_fn};

/// Type alias for platform-specific stream handles used by [`StreamExtension`]s.
pub type StreamHandle = isize;

/// The PJRT stream extension provides capabilities for integrating externally-managed ready events with
/// backend-specific stream (e.g., CUDA or HIP stream) semantics. The extension is both optional for PJRT [`Plugin`]s
/// and _experimental_, meaning that incompatible changes may be introduced at any time, including changes that break
/// _Application Binary Interface (ABI)_ compatibility.
#[derive(Copy, Clone)]
pub struct StreamExtension {
    /// Handle that represents this [`StreamExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_Stream_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl StreamExtension {
    /// Constructs a new [`StreamExtension`] from the provided [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base)
    /// handle if the type of that PJRT extension matches the PJRT stream extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_Stream {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_Stream_Extension`](ffi::PJRT_Stream_Extension) that corresponds to this [`StreamExtension`]
    /// and which can be passed to functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_Stream_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for StreamExtension {}
unsafe impl Sync for StreamExtension {}

impl Client<'_> {
    /// Attempts to load the [`StreamExtension`] from this [`Client`] and returns
    /// [`Error::Unimplemented`] if it is not provided by the underlying [`Plugin`].
    pub fn stream_extension(&self) -> Result<StreamExtension, Error> {
        self.api().stream_extension()
    }
}

impl Plugin {
    /// Attempts to load the [`StreamExtension`] from this [`Plugin`] and returns
    /// [`Error::Unimplemented`] if it is not provided by this [`Plugin`].
    pub fn stream_extension(&self) -> Result<StreamExtension, Error> {
        self.api().stream_extension()
    }
}

impl Api {
    /// Attempts to load the [`StreamExtension`] from this [`Api`] and returns
    /// [`Error::Unimplemented`] if it is not provided by the underlying [`Plugin`].
    pub(crate) fn stream_extension(&self) -> Result<StreamExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let stream_extension = StreamExtension::from_c_api(extension, *self);
                if let Some(stream_extension) = stream_extension {
                    return Ok(stream_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the stream extension is not provided by the PJRT plugin"))
        }
    }
}

impl Device<'_> {
    /// Returns a platform-specific [`StreamHandle`] that should be used to track when an externally-managed [`Buffer`]
    /// is ready to use on this [`Device`].
    pub fn stream_for_external_ready_events(&self) -> Result<StreamHandle, Error> {
        use ffi::PJRT_Get_Stream_For_External_Ready_Events_Args;
        let extension = self.api().stream_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_Stream_Extension => extension,
            PJRT_Get_Stream_For_External_Ready_Events,
            { device = self.to_c_api() },
            { stream },
        )
    }
}

impl Buffer<'_> {
    /// Waits until this [`Buffer`] is ready on the platform-specific stream that the provided [`StreamHandle`]
    /// is associated with.
    pub fn wait_until_ready_on_stream(&self, stream: StreamHandle) -> Result<(), Error> {
        use ffi::PJRT_Wait_Until_Buffer_Ready_On_Stream_Args;
        let extension = self.api().stream_extension()?;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_Stream_Extension => extension,
            PJRT_Wait_Until_Buffer_Ready_On_Stream,
            {
                stream = stream,
                buffer = self.to_c_api(),
            },
        )
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::buffers::ffi::PJRT_Buffer;
    use crate::devices::ffi::PJRT_Device;
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;

    pub const PJRT_API_STREAM_EXTENSION_VERSION: usize = 0;

    #[repr(C)]
    pub struct PJRT_Get_Stream_For_External_Ready_Events_Args {
        pub struct_size: usize,
        pub device: *mut PJRT_Device,
        pub stream: isize,
    }

    impl PJRT_Get_Stream_For_External_Ready_Events_Args {
        pub fn new(device: *mut PJRT_Device) -> Self {
            Self { struct_size: size_of::<Self>(), device, stream: 0 }
        }
    }

    pub type PJRT_Get_Stream_For_External_Ready_Events =
        unsafe extern "C" fn(args: *mut PJRT_Get_Stream_For_External_Ready_Events_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Wait_Until_Buffer_Ready_On_Stream_Args {
        pub struct_size: usize,
        pub stream: isize,
        pub buffer: *mut PJRT_Buffer,
    }

    impl PJRT_Wait_Until_Buffer_Ready_On_Stream_Args {
        pub fn new(stream: isize, buffer: *mut PJRT_Buffer) -> Self {
            Self { struct_size: size_of::<Self>(), stream, buffer }
        }
    }

    pub type PJRT_Wait_Until_Buffer_Ready_On_Stream =
        unsafe extern "C" fn(args: *mut PJRT_Wait_Until_Buffer_Ready_On_Stream_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Stream_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_Get_Stream_For_External_Ready_Events: Option<PJRT_Get_Stream_For_External_Ready_Events>,
        pub PJRT_Wait_Until_Buffer_Ready_On_Stream: Option<PJRT_Wait_Until_Buffer_Ready_On_Stream>,
    }
}

#[cfg(test)]
mod tests {
    use crate::BufferType;
    use crate::Error;
    use crate::tests::TestPlatform;
    use crate::tests::test_for_each_platform;

    #[test]
    fn test_stream_extension() {
        test_for_each_platform!(|plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    assert!(plugin.stream_extension().is_ok());
                    assert!(client.stream_extension().is_ok());
                }
                _ => {
                    assert!(matches!(plugin.stream_extension(), Err(Error::Unimplemented { .. })));
                    assert!(matches!(client.stream_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }

    #[test]
    fn test_device_stream_for_external_ready_events() {
        test_for_each_platform!(|plugin, client, platform| {
            let device = client.addressable_devices().unwrap()[0].clone();
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    assert!(device.stream_for_external_ready_events().is_ok());
                }
                _ => {
                    assert!(matches!(device.stream_for_external_ready_events(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }

    #[test]
    fn test_buffer_wait_until_ready_on_stream() {
        test_for_each_platform!(|plugin, client, platform| {
            let device = client.addressable_devices().unwrap()[0].clone();
            let buffer = client.buffer(&[1u8, 2u8, 3u8, 4u8], BufferType::U8, [4], None, device.clone(), None).unwrap();
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    let stream = device.stream_for_external_ready_events().unwrap();
                    assert!(buffer.wait_until_ready_on_stream(stream).is_ok());
                }
                _ => {
                    assert!(matches!(buffer.wait_until_ready_on_stream(0), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }
}
