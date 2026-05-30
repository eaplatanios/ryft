use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};

use crate::{Api, Client, Error, Plugin, invoke_pjrt_api_error_fn, slice_from_c_api};

/// The PJRT multi-slice extension provides lifecycle and introspection operations for multi-slice configurations.
/// A [`MultiSliceConfig`] can then be passed to APIs such as [`LoadOptions`](crate::LoadOptions) that need to
/// associate a computation with a multi-slice runtime setup.
#[derive(Copy, Clone)]
pub struct MultiSliceExtension {
    /// Handle that represents this [`MultiSliceExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_MultiSlice_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl MultiSliceExtension {
    /// Constructs a new [`MultiSliceExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle if the type of that PJRT extension matches the
    /// PJRT multi-slice extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_MultiSlice {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_MultiSlice_Extension`](ffi::PJRT_MultiSlice_Extension) that corresponds to this
    /// [`MultiSliceExtension`] and which can be passed to functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_MultiSlice_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for MultiSliceExtension {}
unsafe impl Sync for MultiSliceExtension {}

impl Client<'_> {
    /// Attempts to load the [`MultiSliceExtension`] from this [`Client`] and returns [`Error::Unimplemented`] if it is
    /// not provided by the underlying [`Plugin`].
    pub fn multi_slice_extension(&self) -> Result<MultiSliceExtension, Error> {
        self.api().multi_slice_extension()
    }
}

impl Plugin {
    /// Attempts to load the [`MultiSliceExtension`] from this [`Plugin`] and returns [`Error::Unimplemented`] if it is
    /// not provided by this [`Plugin`].
    pub fn multi_slice_extension(&self) -> Result<MultiSliceExtension, Error> {
        self.api().multi_slice_extension()
    }
}

impl Api {
    /// Attempts to load the [`MultiSliceExtension`] from this [`Api`] and returns [`Error::Unimplemented`] if it is not
    /// provided by the underlying [`Plugin`].
    pub(crate) fn multi_slice_extension(&self) -> Result<MultiSliceExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let multi_slice_extension = MultiSliceExtension::from_c_api(extension, *self);
                if let Some(multi_slice_extension) = multi_slice_extension {
                    return Ok(multi_slice_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the multi-slice extension is not provided by the PJRT plugin"))
        }
    }
}

/// Configuration that describes a multi-slice PJRT setup. This type owns the underlying
/// [`PJRT_MultiSlice_Config`](crate::programs::ffi::PJRT_MultiSlice_Config) handle and releases it through the
/// [`MultiSliceExtension`] that produced it. The handle is borrowed when passed through
/// [`LoadOptions`](crate::LoadOptions).
pub struct MultiSliceConfig {
    /// Handle that represents this [`MultiSliceConfig`] in the PJRT C API.
    handle: *mut crate::programs::ffi::PJRT_MultiSlice_Config,

    /// [`MultiSliceExtension`] associated with this multi-slice configuration handle.
    extension: MultiSliceExtension,
}

impl MultiSliceConfig {
    /// Constructs a new [`MultiSliceConfig`] from the provided
    /// [`PJRT_MultiSlice_Config`](crate::programs::ffi::PJRT_MultiSlice_Config) handle.
    #[allow(dead_code)]
    pub(crate) unsafe fn from_c_api(
        handle: *mut crate::programs::ffi::PJRT_MultiSlice_Config,
        extension: MultiSliceExtension,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT multi-slice config handle is a null pointer"))
        } else {
            Ok(Self { handle, extension })
        }
    }

    /// Returns the [`PJRT_MultiSlice_Config`](crate::programs::ffi::PJRT_MultiSlice_Config) that corresponds to this
    /// [`MultiSliceConfig`] and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut crate::programs::ffi::PJRT_MultiSlice_Config {
        self.handle
    }

    /// Returns the total number of slices described by this [`MultiSliceConfig`].
    pub fn num_slices(&self) -> Result<i32, Error> {
        use ffi::PJRT_MultiSlice_Config_NumSlices_Args;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_MultiSlice_Extension => self.extension,
            PJRT_MultiSlice_Config_NumSlices,
            { config = self.to_c_api() },
            { num_slices },
        )
    }

    /// Returns the local slice ID described by this [`MultiSliceConfig`].
    pub fn slice_id(&self) -> Result<i32, Error> {
        use ffi::PJRT_MultiSlice_Config_SliceId_Args;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_MultiSlice_Extension => self.extension,
            PJRT_MultiSlice_Config_SliceId,
            { config = self.to_c_api() },
            { slice_id },
        )
    }

    /// Returns the device count available in each slice described by this [`MultiSliceConfig`].
    pub fn device_count_per_slice(&self) -> Result<HashMap<i32, i32>, Error> {
        use ffi::PJRT_MultiSlice_Config_NumDevicesPerSlice_Args;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_MultiSlice_Extension => self.extension,
            PJRT_MultiSlice_Config_NumDevicesPerSlice,
            { config = self.to_c_api() },
            { num_devices_per_slice_map, slice_ids, num_devices, devices_per_slice_map, devices_per_slice_map_deleter },
        )
        .map(|(num_devices_per_slice_map, slice_ids, num_devices, devices_per_slice_map, deleter)| {
            let slice_ids = unsafe { slice_from_c_api(slice_ids, num_devices_per_slice_map) };
            let device_counts = unsafe { slice_from_c_api(num_devices, num_devices_per_slice_map) };
            let device_count_per_slice =
                slice_ids.iter().copied().zip(device_counts.iter().copied()).collect::<HashMap<_, _>>();
            if let Some(deleter) = deleter {
                unsafe { deleter(devices_per_slice_map) };
            }
            device_count_per_slice
        })
    }

    /// Serializes this [`MultiSliceConfig`] into a [`SerializedMultiSliceConfig`].
    pub fn serialize(&self) -> Result<SerializedMultiSliceConfig, Error> {
        use ffi::PJRT_MultiSlice_Config_Serialize_Args;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_MultiSlice_Extension => self.extension,
            PJRT_MultiSlice_Config_Serialize,
            { config = self.to_c_api() },
            { serialized, size, serialized_config, serialized_config_deleter },
        )
        .map(|(serialized, size, serialized_config, serialized_config_deleter)| SerializedMultiSliceConfig {
            handle: serialized_config,
            deleter: serialized_config_deleter,
            data: serialized,
            data_size: size,
        })
    }
}

impl Debug for MultiSliceConfig {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("MultiSliceConfig").field("handle", &self.handle).finish_non_exhaustive()
    }
}

unsafe impl Send for MultiSliceConfig {}
unsafe impl Sync for MultiSliceConfig {}

impl Drop for MultiSliceConfig {
    fn drop(&mut self) {
        use ffi::PJRT_MultiSlice_Config_Destroy_Args;
        invoke_pjrt_api_error_fn!(
            @extension ffi::PJRT_MultiSlice_Extension => self.extension,
            PJRT_MultiSlice_Config_Destroy,
            { config = self.to_c_api() },
        )
        .expect("failed to destroy PJRT multi-slice config");
    }
}

/// Serialized representation of a [`MultiSliceConfig`].
pub struct SerializedMultiSliceConfig {
    /// Handle that owns the serialized multi-slice configuration data in the PJRT C API.
    handle: *mut ffi::PJRT_MultiSlice_SerializedConfig,

    /// Deleter for the serialized multi-slice configuration data.
    deleter: Option<unsafe extern "C" fn(serialized_config: *mut ffi::PJRT_MultiSlice_SerializedConfig)>,

    /// Pointer to the serialized bytes.
    data: *const std::ffi::c_char,

    /// Number of serialized bytes.
    data_size: usize,
}

impl SerializedMultiSliceConfig {
    /// Returns the serialized bytes of this [`SerializedMultiSliceConfig`].
    pub fn data(&self) -> &[u8] {
        unsafe { slice_from_c_api(self.data as *const _, self.data_size) }
    }
}

impl PartialEq for SerializedMultiSliceConfig {
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl Eq for SerializedMultiSliceConfig {}

impl Hash for SerializedMultiSliceConfig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data().hash(state);
    }
}

impl Debug for SerializedMultiSliceConfig {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SerializedMultiSliceConfig")
            .field("data_size", &self.data_size)
            .finish_non_exhaustive()
    }
}

unsafe impl Send for SerializedMultiSliceConfig {}
unsafe impl Sync for SerializedMultiSliceConfig {}

impl Drop for SerializedMultiSliceConfig {
    fn drop(&mut self) {
        if let Some(deleter) = self.deleter {
            unsafe { deleter(self.handle) };
        }
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;
    use crate::programs::ffi::PJRT_MultiSlice_Config;

    /// Version of the PJRT multi-slice extension C API.
    pub const PJRT_API_MULTI_SLICE_EXTENSION_VERSION: usize = 1;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_MultiSlice_NumDevicesPerSlice {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_MultiSlice_SerializedConfig {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_MultiSlice_Config_Destroy_Args {
        pub struct_size: usize,
        pub config: *mut PJRT_MultiSlice_Config,
    }

    impl PJRT_MultiSlice_Config_Destroy_Args {
        pub fn new(config: *mut PJRT_MultiSlice_Config) -> Self {
            Self { struct_size: size_of::<Self>(), config }
        }
    }

    pub type PJRT_MultiSlice_Config_Destroy =
        unsafe extern "C" fn(args: *mut PJRT_MultiSlice_Config_Destroy_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_MultiSlice_Config_NumSlices_Args {
        pub struct_size: usize,
        pub config: *mut PJRT_MultiSlice_Config,
        pub num_slices: i32,
    }

    impl PJRT_MultiSlice_Config_NumSlices_Args {
        pub fn new(config: *mut PJRT_MultiSlice_Config) -> Self {
            Self { struct_size: size_of::<Self>(), config, num_slices: 0 }
        }
    }

    pub type PJRT_MultiSlice_Config_NumSlices =
        unsafe extern "C" fn(args: *mut PJRT_MultiSlice_Config_NumSlices_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_MultiSlice_Config_SliceId_Args {
        pub struct_size: usize,
        pub config: *mut PJRT_MultiSlice_Config,
        pub slice_id: i32,
    }

    impl PJRT_MultiSlice_Config_SliceId_Args {
        pub fn new(config: *mut PJRT_MultiSlice_Config) -> Self {
            Self { struct_size: size_of::<Self>(), config, slice_id: 0 }
        }
    }

    pub type PJRT_MultiSlice_Config_SliceId =
        unsafe extern "C" fn(args: *mut PJRT_MultiSlice_Config_SliceId_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_MultiSlice_Config_NumDevicesPerSlice_Args {
        pub struct_size: usize,
        pub config: *mut PJRT_MultiSlice_Config,
        pub num_devices_per_slice_map: usize,
        pub slice_ids: *const i32,
        pub num_devices: *const i32,
        pub devices_per_slice_map: *mut PJRT_MultiSlice_NumDevicesPerSlice,
        pub devices_per_slice_map_deleter: Option<unsafe extern "C" fn(ptr: *mut PJRT_MultiSlice_NumDevicesPerSlice)>,
    }

    impl PJRT_MultiSlice_Config_NumDevicesPerSlice_Args {
        pub fn new(config: *mut PJRT_MultiSlice_Config) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                config,
                num_devices_per_slice_map: 0,
                slice_ids: std::ptr::null(),
                num_devices: std::ptr::null(),
                devices_per_slice_map: std::ptr::null_mut(),
                devices_per_slice_map_deleter: None,
            }
        }
    }

    pub type PJRT_MultiSlice_Config_NumDevicesPerSlice =
        unsafe extern "C" fn(args: *mut PJRT_MultiSlice_Config_NumDevicesPerSlice_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_MultiSlice_Config_Serialize_Args {
        pub struct_size: usize,
        pub config: *mut PJRT_MultiSlice_Config,
        pub serialized: *const std::ffi::c_char,
        pub size: usize,
        pub serialized_config: *mut PJRT_MultiSlice_SerializedConfig,
        pub serialized_config_deleter: Option<unsafe extern "C" fn(ptr: *mut PJRT_MultiSlice_SerializedConfig)>,
    }

    impl PJRT_MultiSlice_Config_Serialize_Args {
        pub fn new(config: *mut PJRT_MultiSlice_Config) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                config,
                serialized: std::ptr::null(),
                size: 0,
                serialized_config: std::ptr::null_mut(),
                serialized_config_deleter: None,
            }
        }
    }

    pub type PJRT_MultiSlice_Config_Serialize =
        unsafe extern "C" fn(args: *mut PJRT_MultiSlice_Config_Serialize_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_MultiSlice_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_MultiSlice_Config_Destroy: Option<PJRT_MultiSlice_Config_Destroy>,
        pub PJRT_MultiSlice_Config_NumSlices: Option<PJRT_MultiSlice_Config_NumSlices>,
        pub PJRT_MultiSlice_Config_SliceId: Option<PJRT_MultiSlice_Config_SliceId>,
        pub PJRT_MultiSlice_Config_NumDevicesPerSlice: Option<PJRT_MultiSlice_Config_NumDevicesPerSlice>,
        pub PJRT_MultiSlice_Config_Serialize: Option<PJRT_MultiSlice_Config_Serialize>,
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{test_cpu_client, test_cpu_plugin};

    #[test]
    fn test_multi_slice_extension() {
        assert!(test_cpu_plugin().multi_slice_extension().is_err());
        assert!(test_cpu_client().multi_slice_extension().is_err());
    }

    // TODO(eaplatanios): Add more tests once there is a PJRT plugin that provides this extension.
}
