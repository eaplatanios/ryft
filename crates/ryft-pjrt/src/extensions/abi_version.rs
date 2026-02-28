use std::hash::{Hash, Hasher};

use prost::Message;

use crate::{invoke_pjrt_api_error_fn, slice_from_c_api, Api, Client, Error, Executable, Plugin};

/// The PJRT ABI version extension provides capabilities for querying runtime and executable ABI versions and checking
/// compatibility between them. The extension is optional for PJRT [`Plugin`]s and _experimental_, meaning that
/// incompatible changes may be introduced at any time, including changes that break _Application Binary Interface
/// (ABI)_ compatibility.
#[derive(Copy, Clone)]
pub struct AbiVersionExtension {
    /// Handle that represents this [`AbiVersionExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_AbiVersion_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl AbiVersionExtension {
    /// Constructs a new [`AbiVersionExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle if the type of that PJRT extension matches
    /// the PJRT ABI version extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_AbiVersion {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_AbiVersion_Extension`](ffi::PJRT_AbiVersion_Extension) that corresponds to this
    /// [`AbiVersionExtension`] and which can be passed to functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_AbiVersion_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for AbiVersionExtension {}
unsafe impl Sync for AbiVersionExtension {}

impl Executable {
    /// Returns the [`ExecutableAbiVersion`] associated with this [`Executable`].
    pub fn abi_version(&self) -> Result<ExecutableAbiVersion, Error> {
        use ffi::PJRT_Executable_GetAbiVersion_Args;
        let extension = self.api().abi_version_extension()?;
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_Executable_GetAbiVersion,
            { executable = self.to_c_api() },
            { abi_version },
        )
        .and_then(|abi_version| unsafe { ExecutableAbiVersion::from_c_api(abi_version, extension) })
    }
}

impl Client<'_> {
    /// Attempts to load the [`AbiVersionExtension`] from this [`Client`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub fn abi_version_extension(&self) -> Result<AbiVersionExtension, Error> {
        self.api().abi_version_extension()
    }

    /// Returns the [`RuntimeAbiVersion`] associated with this [`Client`].
    pub fn abi_version(&self) -> Result<RuntimeAbiVersion, Error> {
        use ffi::PJRT_Client_RuntimeAbiVersion_Args;
        let extension = self.abi_version_extension()?;
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_Client_RuntimeAbiVersion,
            { client = self.to_c_api() },
            { abi_version },
        )
        .and_then(|abi_version| unsafe { RuntimeAbiVersion::from_c_api(abi_version, extension) })
    }

    /// Deserializes a [`RuntimeAbiVersion`] from the provided ABI version Protobuf message.
    pub fn runtime_abi_version_from_proto(
        &self,
        abi_version: crate::protos::RuntimeAbiVersion,
    ) -> Result<RuntimeAbiVersion, Error> {
        self.api().runtime_abi_version_from_proto(abi_version)
    }

    /// Deserializes a [`RuntimeAbiVersion`] from the provided serialized ABI version.
    pub fn deserialize_runtime_abi_version(
        &self,
        serialized_abi_version: &SerializedAbiVersion,
    ) -> Result<RuntimeAbiVersion, Error> {
        self.api().deserialize_runtime_abi_version(serialized_abi_version)
    }

    /// Deserializes an [`ExecutableAbiVersion`] from the provided ABI version Protobuf message.
    pub fn executable_abi_version_from_proto(
        &self,
        abi_version: crate::protos::ExecutableAbiVersion,
    ) -> Result<ExecutableAbiVersion, Error> {
        self.api().executable_abi_version_from_proto(abi_version)
    }

    /// Deserializes an [`ExecutableAbiVersion`] from the provided serialized ABI version.
    pub fn deserialize_executable_abi_version(
        &self,
        serialized_abi_version: &SerializedAbiVersion,
    ) -> Result<ExecutableAbiVersion, Error> {
        self.api().deserialize_executable_abi_version(serialized_abi_version)
    }
}

impl Plugin {
    /// Attempts to load the [`AbiVersionExtension`] from this [`Plugin`] and returns [`Error::Unimplemented`]
    /// if it is not provided by this [`Plugin`].
    pub fn abi_version_extension(&self) -> Result<AbiVersionExtension, Error> {
        self.api().abi_version_extension()
    }

    /// Deserializes a [`RuntimeAbiVersion`] from the provided ABI version Protobuf message.
    pub fn runtime_abi_version_from_proto(
        &self,
        abi_version: crate::protos::RuntimeAbiVersion,
    ) -> Result<RuntimeAbiVersion, Error> {
        self.api().runtime_abi_version_from_proto(abi_version)
    }

    /// Deserializes a [`RuntimeAbiVersion`] from the provided serialized ABI version.
    pub fn deserialize_runtime_abi_version(
        &self,
        serialized_abi_version: &SerializedAbiVersion,
    ) -> Result<RuntimeAbiVersion, Error> {
        self.api().deserialize_runtime_abi_version(serialized_abi_version)
    }

    /// Deserializes an [`ExecutableAbiVersion`] from the provided ABI version Protobuf message.
    pub fn executable_abi_version_from_proto(
        &self,
        abi_version: crate::protos::ExecutableAbiVersion,
    ) -> Result<ExecutableAbiVersion, Error> {
        self.api().executable_abi_version_from_proto(abi_version)
    }

    /// Deserializes an [`ExecutableAbiVersion`] from the provided serialized ABI version.
    pub fn deserialize_executable_abi_version(
        &self,
        serialized_abi_version: &SerializedAbiVersion,
    ) -> Result<ExecutableAbiVersion, Error> {
        self.api().deserialize_executable_abi_version(serialized_abi_version)
    }
}

impl Api {
    /// Attempts to load the [`AbiVersionExtension`] from this [`Api`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub(crate) fn abi_version_extension(&self) -> Result<AbiVersionExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let abi_version_extension = AbiVersionExtension::from_c_api(extension, *self);
                if let Some(abi_version_extension) = abi_version_extension {
                    return Ok(abi_version_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the ABI version extension is not provided by the PJRT plugin"))
        }
    }

    /// Deserializes a [`RuntimeAbiVersion`] from the provided ABI version Protobuf message.
    pub(crate) fn runtime_abi_version_from_proto(
        &self,
        abi_version: crate::protos::RuntimeAbiVersion,
    ) -> Result<RuntimeAbiVersion, Error> {
        let serialized_abi_version = SerializedAbiVersion::from_proto(abi_version);
        self.deserialize_runtime_abi_version(&serialized_abi_version)
    }

    /// Deserializes a [`RuntimeAbiVersion`] from the provided serialized ABI version.
    pub(crate) fn deserialize_runtime_abi_version(
        &self,
        serialized_abi_version: &SerializedAbiVersion,
    ) -> Result<RuntimeAbiVersion, Error> {
        use ffi::PJRT_RuntimeAbiVersion_FromProto_Args;
        let extension = self.abi_version_extension()?;
        let data = serialized_abi_version.data();
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_RuntimeAbiVersion_FromProto,
            {
                serialized_proto = data.as_ptr() as *const _,
                serialized_proto_size = data.len(),
            },
            { abi_version },
        )
        .and_then(|abi_version| unsafe { RuntimeAbiVersion::from_c_api(abi_version, extension) })
    }

    /// Deserializes an [`ExecutableAbiVersion`] from the provided ABI version Protobuf message.
    pub(crate) fn executable_abi_version_from_proto(
        &self,
        abi_version: crate::protos::ExecutableAbiVersion,
    ) -> Result<ExecutableAbiVersion, Error> {
        let serialized_abi_version = SerializedAbiVersion::from_proto(abi_version);
        self.deserialize_executable_abi_version(&serialized_abi_version)
    }

    /// Deserializes an [`ExecutableAbiVersion`] from the provided serialized ABI version.
    pub(crate) fn deserialize_executable_abi_version(
        &self,
        serialized_abi_version: &SerializedAbiVersion,
    ) -> Result<ExecutableAbiVersion, Error> {
        use ffi::PJRT_ExecutableAbiVersion_FromProto_Args;
        let extension = self.abi_version_extension()?;
        let data = serialized_abi_version.data();
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_ExecutableAbiVersion_FromProto,
            {
                serialized_proto = data.as_ptr() as *const _,
                serialized_proto_size = data.len(),
            },
            { abi_version },
        )
        .and_then(|abi_version| unsafe { ExecutableAbiVersion::from_c_api(abi_version, extension) })
    }
}

/// Runtime ABI version descriptor returned by the PJRT [`AbiVersionExtension`].
pub struct RuntimeAbiVersion {
    /// Handle that represents this [`RuntimeAbiVersion`] in the PJRT C API.
    handle: *mut ffi::PJRT_RuntimeAbiVersion,

    /// [`AbiVersionExtension`] associated with this ABI version handle.
    extension: AbiVersionExtension,
}

impl RuntimeAbiVersion {
    /// Constructs a new [`RuntimeAbiVersion`] from the provided
    /// [`PJRT_RuntimeAbiVersion`](ffi::PJRT_RuntimeAbiVersion) handle.
    pub(crate) unsafe fn from_c_api(
        handle: *mut ffi::PJRT_RuntimeAbiVersion,
        extension: AbiVersionExtension,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT runtime ABI version handle is a null pointer"))
        } else {
            Ok(Self { handle, extension })
        }
    }

    /// Returns the [`PJRT_RuntimeAbiVersion`](ffi::PJRT_RuntimeAbiVersion) that corresponds to this
    /// [`RuntimeAbiVersion`] and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_RuntimeAbiVersion {
        self.handle
    }

    /// Returns the platform identifier associated with this [`RuntimeAbiVersion`].
    pub fn platform_id(&self) -> Result<u64, Error> {
        use ffi::PJRT_RuntimeAbiVersion_PlatformId_Args;
        invoke_pjrt_api_error_fn!(
            @unchecked self.extension,
            PJRT_RuntimeAbiVersion_PlatformId,
            { abi_version = self.to_c_api() },
            { platform_id },
        )
    }

    /// Returns whether this [`RuntimeAbiVersion`] is compatible with the provided [`RuntimeAbiVersion`].
    pub fn is_compatible_with_runtime(&self, abi_version: &RuntimeAbiVersion) -> Result<(), Error> {
        use ffi::PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args;
        invoke_pjrt_api_error_fn!(
            @unchecked self.extension,
            PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime,
            {
                abi_version = self.to_c_api(),
                other_abi_version = abi_version.to_c_api(),
            },
        )
    }

    /// Returns whether this [`RuntimeAbiVersion`] is compatible with the provided [`ExecutableAbiVersion`].
    pub fn is_compatible_with_executable(&self, abi_version: &ExecutableAbiVersion) -> Result<(), Error> {
        use ffi::PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args;
        invoke_pjrt_api_error_fn!(
            @unchecked self.extension,
            PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable,
            {
                abi_version = self.to_c_api(),
                executable_abi_version = abi_version.to_c_api(),
            },
        )
    }

    /// Returns the Protobuf message that corresponds to this [`RuntimeAbiVersion`].
    pub fn proto(&self) -> Result<crate::protos::RuntimeAbiVersion, Error> {
        self.serialize()?.proto()
    }

    /// Serializes this [`RuntimeAbiVersion`] into a [`SerializedAbiVersion`].
    pub fn serialize(&self) -> Result<SerializedAbiVersion, Error> {
        use ffi::PJRT_RuntimeAbiVersion_ToProto_Args;
        invoke_pjrt_api_error_fn!(
            @unchecked self.extension,
            PJRT_RuntimeAbiVersion_ToProto,
            { abi_version = self.to_c_api() },
            { serialized_proto, serialized_proto_size, serialized_proto_holder, serialized_proto_deleter },
        )
        .map(|(serialized_proto, serialized_proto_size, serialized_proto_holder, serialized_proto_deleter)| {
            SerializedAbiVersion {
                handle: serialized_proto_holder,
                deleter: serialized_proto_deleter,
                data: serialized_proto,
                data_size: serialized_proto_size,
            }
        })
    }
}

unsafe impl Send for RuntimeAbiVersion {}
unsafe impl Sync for RuntimeAbiVersion {}

impl Drop for RuntimeAbiVersion {
    fn drop(&mut self) {
        use ffi::PJRT_RuntimeAbiVersion_Destroy_Args;
        invoke_pjrt_api_error_fn!(
            @unchecked self.extension,
            PJRT_RuntimeAbiVersion_Destroy,
            { abi_version = self.to_c_api() },
        )
        .expect("failed to destroy PJRT runtime ABI version");
    }
}

/// Executable ABI version descriptor returned by the PJRT [`AbiVersionExtension`].
pub struct ExecutableAbiVersion {
    /// Handle that represents this [`ExecutableAbiVersion`] in the PJRT C API.
    handle: *mut ffi::PJRT_ExecutableAbiVersion,

    /// [`AbiVersionExtension`] associated with this ABI version handle.
    extension: AbiVersionExtension,
}

impl ExecutableAbiVersion {
    /// Constructs a new [`ExecutableAbiVersion`] from the provided
    /// [`PJRT_ExecutableAbiVersion`](ffi::PJRT_ExecutableAbiVersion) handle.
    pub(crate) unsafe fn from_c_api(
        handle: *mut ffi::PJRT_ExecutableAbiVersion,
        extension: AbiVersionExtension,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT executable ABI version handle is a null pointer"))
        } else {
            Ok(Self { handle, extension })
        }
    }

    /// Returns the [`PJRT_ExecutableAbiVersion`](ffi::PJRT_ExecutableAbiVersion) that corresponds to this
    /// [`ExecutableAbiVersion`] and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_ExecutableAbiVersion {
        self.handle
    }

    /// Returns the platform identifier associated with this [`ExecutableAbiVersion`].
    pub fn platform_id(&self) -> Result<u64, Error> {
        use ffi::PJRT_ExecutableAbiVersion_PlatformId_Args;
        invoke_pjrt_api_error_fn!(
            @unchecked self.extension,
            PJRT_ExecutableAbiVersion_PlatformId,
            { abi_version = self.to_c_api() },
            { platform_id },
        )
    }

    /// Returns the Protobuf message that corresponds to this [`ExecutableAbiVersion`].
    pub fn proto(&self) -> Result<crate::protos::ExecutableAbiVersion, Error> {
        self.serialize()?.proto()
    }

    /// Serializes this [`ExecutableAbiVersion`] into a [`SerializedAbiVersion`].
    pub fn serialize(&self) -> Result<SerializedAbiVersion, Error> {
        use ffi::PJRT_ExecutableAbiVersion_ToProto_Args;
        invoke_pjrt_api_error_fn!(
            @unchecked self.extension,
            PJRT_ExecutableAbiVersion_ToProto,
            { abi_version = self.to_c_api() },
            { serialized_proto, serialized_proto_size, serialized_proto_holder, serialized_proto_deleter },
        )
        .map(|(serialized_proto, serialized_proto_size, serialized_proto_holder, serialized_proto_deleter)| {
            SerializedAbiVersion {
                handle: serialized_proto_holder,
                deleter: serialized_proto_deleter,
                data: serialized_proto,
                data_size: serialized_proto_size,
            }
        })
    }
}

unsafe impl Send for ExecutableAbiVersion {}
unsafe impl Sync for ExecutableAbiVersion {}

impl Drop for ExecutableAbiVersion {
    fn drop(&mut self) {
        use ffi::PJRT_ExecutableAbiVersion_Destroy_Args;
        invoke_pjrt_api_error_fn!(
            @unchecked self.extension,
            PJRT_ExecutableAbiVersion_Destroy,
            { abi_version = self.to_c_api() },
        )
        .expect("failed to destroy PJRT executable ABI version");
    }
}

/// Serialized [`RuntimeAbiVersion`] or [`ExecutableAbiVersion`].
pub struct SerializedAbiVersion {
    /// Handle that represents this [`SerializedAbiVersion`] in the PJRT C API.
    handle: *mut ffi::PJRT_SerializedProto,

    /// Optional function that must be called to free the underlying memory when dropping this instance.
    deleter: Option<unsafe extern "C" fn(holder: *mut ffi::PJRT_SerializedProto)>,

    /// Pointer to the underlying bytes of this [`SerializedAbiVersion`].
    data: *const std::ffi::c_char,

    /// Size (i.e., number of bytes) of this [`SerializedAbiVersion`].
    data_size: usize,
}

impl SerializedAbiVersion {
    /// Returns a pointer to the underlying bytes of this [`SerializedAbiVersion`].
    pub fn data(&self) -> &[u8] {
        unsafe { slice_from_c_api(self.data as *const _, self.data_size) }
    }

    /// Constructs a [`SerializedAbiVersion`] from the provided ABI version Protobuf message.
    fn from_proto<M: Message>(abi_version: M) -> Self {
        unsafe extern "C" fn delete_boxed_data(handle: *mut ffi::PJRT_SerializedProto) {
            if !handle.is_null() {
                unsafe { drop(Box::from_raw(handle as *mut Vec<u8>)) };
            }
        }

        let serialized_abi_version = Box::new(abi_version.encode_to_vec());
        let data = serialized_abi_version.as_ptr() as *const std::ffi::c_char;
        let data_size = serialized_abi_version.len();
        let handle = Box::into_raw(serialized_abi_version) as *mut ffi::PJRT_SerializedProto;

        Self { handle, deleter: Some(delete_boxed_data), data, data_size }
    }

    /// Returns the Protobuf message that corresponds to this [`SerializedAbiVersion`].
    fn proto<M: Message + Default>(&self) -> Result<M, Error> {
        M::decode(self.data()).map_err(|error| {
            Error::invalid_argument(format!("serialized PJRT ABI version could not be decoded as Protobuf: {error}"))
        })
    }
}

impl PartialEq for SerializedAbiVersion {
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl Eq for SerializedAbiVersion {}

impl Hash for SerializedAbiVersion {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data().hash(state);
    }
}

unsafe impl Send for SerializedAbiVersion {}
unsafe impl Sync for SerializedAbiVersion {}

impl Drop for SerializedAbiVersion {
    fn drop(&mut self) {
        if let Some(deleter) = self.deleter {
            unsafe { deleter(self.handle) };
        }
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::clients::ffi::PJRT_Client;
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;
    use crate::programs::ffi::PJRT_Executable;

    pub const PJRT_API_ABI_VERSION_EXTENSION_VERSION: usize = 1;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_RuntimeAbiVersion {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_ExecutableAbiVersion {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_SerializedProto {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Client_RuntimeAbiVersion_Args {
        pub struct_size: usize,
        pub client: *mut PJRT_Client,
        pub abi_version: *mut PJRT_RuntimeAbiVersion,
    }

    impl PJRT_Client_RuntimeAbiVersion_Args {
        pub fn new(client: *mut PJRT_Client) -> Self {
            Self { struct_size: size_of::<Self>(), client, abi_version: std::ptr::null_mut() }
        }
    }

    pub type PJRT_Client_RuntimeAbiVersion =
        unsafe extern "C" fn(args: *mut PJRT_Client_RuntimeAbiVersion_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Executable_GetAbiVersion_Args {
        pub struct_size: usize,
        pub executable: *mut PJRT_Executable,
        pub abi_version: *mut PJRT_ExecutableAbiVersion,
    }

    impl PJRT_Executable_GetAbiVersion_Args {
        pub fn new(executable: *mut PJRT_Executable) -> Self {
            Self { struct_size: size_of::<Self>(), executable, abi_version: std::ptr::null_mut() }
        }
    }

    pub type PJRT_Executable_GetAbiVersion =
        unsafe extern "C" fn(args: *mut PJRT_Executable_GetAbiVersion_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_RuntimeAbiVersion_Destroy_Args {
        pub struct_size: usize,
        pub abi_version: *mut PJRT_RuntimeAbiVersion,
    }

    impl PJRT_RuntimeAbiVersion_Destroy_Args {
        pub fn new(abi_version: *mut PJRT_RuntimeAbiVersion) -> Self {
            Self { struct_size: size_of::<Self>(), abi_version }
        }
    }

    pub type PJRT_RuntimeAbiVersion_Destroy =
        unsafe extern "C" fn(args: *mut PJRT_RuntimeAbiVersion_Destroy_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args {
        pub struct_size: usize,
        pub abi_version: *const PJRT_RuntimeAbiVersion,
        pub other_abi_version: *const PJRT_RuntimeAbiVersion,
    }

    impl PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args {
        pub fn new(
            abi_version: *const PJRT_RuntimeAbiVersion,
            other_abi_version: *const PJRT_RuntimeAbiVersion,
        ) -> Self {
            Self { struct_size: size_of::<Self>(), abi_version, other_abi_version }
        }
    }

    pub type PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime =
        unsafe extern "C" fn(args: *mut PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args {
        pub struct_size: usize,
        pub abi_version: *const PJRT_RuntimeAbiVersion,
        pub executable_abi_version: *const PJRT_ExecutableAbiVersion,
    }

    impl PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args {
        pub fn new(
            abi_version: *const PJRT_RuntimeAbiVersion,
            executable_abi_version: *const PJRT_ExecutableAbiVersion,
        ) -> Self {
            Self { struct_size: size_of::<Self>(), abi_version, executable_abi_version }
        }
    }

    pub type PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable =
        unsafe extern "C" fn(args: *mut PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_RuntimeAbiVersion_ToProto_Args {
        pub struct_size: usize,
        pub abi_version: *const PJRT_RuntimeAbiVersion,
        pub serialized_proto: *const std::ffi::c_char,
        pub serialized_proto_size: usize,
        pub serialized_proto_holder: *mut PJRT_SerializedProto,
        pub serialized_proto_deleter: Option<unsafe extern "C" fn(holder: *mut PJRT_SerializedProto)>,
    }

    impl PJRT_RuntimeAbiVersion_ToProto_Args {
        pub fn new(abi_version: *const PJRT_RuntimeAbiVersion) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                abi_version,
                serialized_proto: std::ptr::null(),
                serialized_proto_size: 0,
                serialized_proto_holder: std::ptr::null_mut(),
                serialized_proto_deleter: None,
            }
        }
    }

    pub type PJRT_RuntimeAbiVersion_ToProto =
        unsafe extern "C" fn(args: *mut PJRT_RuntimeAbiVersion_ToProto_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_RuntimeAbiVersion_PlatformId_Args {
        pub struct_size: usize,
        pub abi_version: *const PJRT_RuntimeAbiVersion,
        pub platform_id: u64,
    }

    impl PJRT_RuntimeAbiVersion_PlatformId_Args {
        pub fn new(abi_version: *const PJRT_RuntimeAbiVersion) -> Self {
            Self { struct_size: size_of::<Self>(), abi_version, platform_id: 0 }
        }
    }

    pub type PJRT_RuntimeAbiVersion_PlatformId =
        unsafe extern "C" fn(args: *mut PJRT_RuntimeAbiVersion_PlatformId_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_ExecutableAbiVersion_Destroy_Args {
        pub struct_size: usize,
        pub abi_version: *mut PJRT_ExecutableAbiVersion,
    }

    impl PJRT_ExecutableAbiVersion_Destroy_Args {
        pub fn new(abi_version: *mut PJRT_ExecutableAbiVersion) -> Self {
            Self { struct_size: size_of::<Self>(), abi_version }
        }
    }

    pub type PJRT_ExecutableAbiVersion_Destroy =
        unsafe extern "C" fn(args: *mut PJRT_ExecutableAbiVersion_Destroy_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_ExecutableAbiVersion_ToProto_Args {
        pub struct_size: usize,
        pub abi_version: *const PJRT_ExecutableAbiVersion,
        pub serialized_proto: *const std::ffi::c_char,
        pub serialized_proto_size: usize,
        pub serialized_proto_holder: *mut PJRT_SerializedProto,
        pub serialized_proto_deleter: Option<unsafe extern "C" fn(holder: *mut PJRT_SerializedProto)>,
    }

    impl PJRT_ExecutableAbiVersion_ToProto_Args {
        pub fn new(abi_version: *const PJRT_ExecutableAbiVersion) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                abi_version,
                serialized_proto: std::ptr::null(),
                serialized_proto_size: 0,
                serialized_proto_holder: std::ptr::null_mut(),
                serialized_proto_deleter: None,
            }
        }
    }

    pub type PJRT_ExecutableAbiVersion_ToProto =
        unsafe extern "C" fn(args: *mut PJRT_ExecutableAbiVersion_ToProto_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_ExecutableAbiVersion_PlatformId_Args {
        pub struct_size: usize,
        pub abi_version: *const PJRT_ExecutableAbiVersion,
        pub platform_id: u64,
    }

    impl PJRT_ExecutableAbiVersion_PlatformId_Args {
        pub fn new(abi_version: *const PJRT_ExecutableAbiVersion) -> Self {
            Self { struct_size: size_of::<Self>(), abi_version, platform_id: 0 }
        }
    }

    pub type PJRT_ExecutableAbiVersion_PlatformId =
        unsafe extern "C" fn(args: *mut PJRT_ExecutableAbiVersion_PlatformId_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_RuntimeAbiVersion_FromProto_Args {
        pub struct_size: usize,
        pub serialized_proto: *const std::ffi::c_char,
        pub serialized_proto_size: usize,
        pub abi_version: *mut PJRT_RuntimeAbiVersion,
    }

    impl PJRT_RuntimeAbiVersion_FromProto_Args {
        pub fn new(serialized_proto: *const std::ffi::c_char, serialized_proto_size: usize) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                serialized_proto,
                serialized_proto_size,
                abi_version: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_RuntimeAbiVersion_FromProto =
        unsafe extern "C" fn(args: *mut PJRT_RuntimeAbiVersion_FromProto_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_ExecutableAbiVersion_FromProto_Args {
        pub struct_size: usize,
        pub serialized_proto: *const std::ffi::c_char,
        pub serialized_proto_size: usize,
        pub abi_version: *mut PJRT_ExecutableAbiVersion,
    }

    impl PJRT_ExecutableAbiVersion_FromProto_Args {
        pub fn new(serialized_proto: *const std::ffi::c_char, serialized_proto_size: usize) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                serialized_proto,
                serialized_proto_size,
                abi_version: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_ExecutableAbiVersion_FromProto =
        unsafe extern "C" fn(args: *mut PJRT_ExecutableAbiVersion_FromProto_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_AbiVersion_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_Client_RuntimeAbiVersion: Option<PJRT_Client_RuntimeAbiVersion>,
        pub PJRT_Executable_GetAbiVersion: Option<PJRT_Executable_GetAbiVersion>,
        pub PJRT_RuntimeAbiVersion_Destroy: Option<PJRT_RuntimeAbiVersion_Destroy>,
        pub PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime: Option<PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime>,
        pub PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable:
            Option<PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable>,
        pub PJRT_RuntimeAbiVersion_ToProto: Option<PJRT_RuntimeAbiVersion_ToProto>,
        pub PJRT_RuntimeAbiVersion_PlatformId: Option<PJRT_RuntimeAbiVersion_PlatformId>,
        pub PJRT_ExecutableAbiVersion_Destroy: Option<PJRT_ExecutableAbiVersion_Destroy>,
        pub PJRT_ExecutableAbiVersion_ToProto: Option<PJRT_ExecutableAbiVersion_ToProto>,
        pub PJRT_ExecutableAbiVersion_PlatformId: Option<PJRT_ExecutableAbiVersion_PlatformId>,
        pub PJRT_RuntimeAbiVersion_FromProto: Option<PJRT_RuntimeAbiVersion_FromProto>,
        pub PJRT_ExecutableAbiVersion_FromProto: Option<PJRT_ExecutableAbiVersion_FromProto>,
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{test_cpu_client, test_cpu_plugin};

    #[test]
    fn test_host_allocator_extension() {
        assert!(test_cpu_plugin().abi_version_extension().is_err());
        assert!(test_cpu_client().abi_version_extension().is_err());
    }

    // TODO(eaplatanios): Add more tests once there is a PJRT plugin that provides this extension.
}
