use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::OnceLock;

use prost::Message;

use crate::protos::Shape;
use crate::{
    Api, BufferType, Client, DeviceDescription, Error, Layout, MemoryKindId, NamedValue, Plugin, Value,
    hash_map_from_c_api, invoke_pjrt_api_error_fn, slice_from_c_api, str_from_c_api,
};

/// Represents a PJRT [`Device`](crate::Device) topology.
///
/// The lifetime parameter `'o` represents the lifetime of the owner of this [`Topology`] (e.g., a [`Client`])
/// if it is borrowed. If it is not borrowed, then it will be set to `'static`.
pub struct Topology<'o> {
    /// Handle that represents this [`Topology`] in the PJRT C API.
    handle: *mut ffi::PJRT_TopologyDescription,

    /// Underlying PJRT [`Api`].
    api: Api,

    /// Cached [`Topology::attributes`] of this [`Topology`] so that it will only be constructed once.
    attributes: OnceLock<Result<HashMap<String, Value>, Error>>,

    /// Boolean flag indicating whether this [`Topology`] is borrowed or owned. This influences the behavior
    /// of [`Topology`]'s [`Drop`] implementation as it will only free the underlying memory if the topology
    /// is owned (i.e., `is_borrowed` is set to `false`).
    is_borrowed: bool,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`Topology`], if it is borrowed.
    /// If it is not borrowed, then the lifetime is `'static`.
    owner: PhantomData<&'o ()>,
}

impl Topology<'_> {
    /// Constructs a new [`Topology`] from the provided [`PJRT_TopologyDescription`](ffi::PJRT_TopologyDescription)
    /// handle that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(
        handle: *mut ffi::PJRT_TopologyDescription,
        api: Api,
        is_borrowed: bool,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT topology handle is a null pointer"))
        } else {
            Ok(Self { handle, api, attributes: OnceLock::new(), is_borrowed, owner: PhantomData })
        }
    }

    /// Returns the [`PJRT_TopologyDescription`](ffi::PJRT_TopologyDescription) that corresponds to this [`Topology`]
    /// and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_TopologyDescription {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }

    /// Returns a string that identifies the platform of this [`Topology`] (e.g., `"cpu"`, `"gpu"`, `"tpu"`, etc.).
    pub fn platform_name(&'_ self) -> Result<Cow<'_, str>, Error> {
        use ffi::PJRT_TopologyDescription_PlatformName_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_TopologyDescription_PlatformName,
            { topology = self.to_c_api() },
            { platform_name, platform_name_size },
        )
        .map(|(string, string_len)| str_from_c_api(string, string_len))
    }

    /// Returns a string that contains human-readable, platform-specific, version information for this [`Topology`]
    /// (e.g., the CUDA version for GPU topologies or the `libtpu` version for TPU topologies).
    pub fn platform_version(&'_ self) -> Result<Cow<'_, str>, Error> {
        use ffi::PJRT_TopologyDescription_PlatformVersion_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_TopologyDescription_PlatformVersion,
            { topology = self.to_c_api() },
            { platform_version, platform_version_size },
        )
        .map(|(string, string_len)| str_from_c_api(string, string_len))
    }

    /// Returns [`DeviceDescription`]s for all [`Device`](crate::Device)s in this [`Topology`]. Note that the device
    /// descriptions can be returned in an arbitrary order, but will always be returned in the same order across
    /// multiple calls to this function from within the same process.
    pub fn device_descriptions(&'_ self) -> Result<Vec<DeviceDescription<'_>>, Error> {
        use ffi::PJRT_TopologyDescription_GetDeviceDescriptions_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_TopologyDescription_GetDeviceDescriptions,
            { topology = self.to_c_api() },
            { descriptions, num_descriptions },
        )
        .and_then(|(descriptions, descriptions_count)| {
            unsafe { slice_from_c_api(descriptions, descriptions_count) }
                .iter()
                .map(|handle| unsafe { DeviceDescription::from_c_api(*handle, self.api()) })
                .collect::<Result<Vec<_>, _>>()
        })
    }

    /// [`Value`] of the attribute with the provided name attached to this [`Topology`], or [`Error::NotFound`]
    /// if no such attribute is attached to this [`Topology`].
    pub fn attribute<N: AsRef<str>>(&self, name: N) -> Result<&Value, Error> {
        let name = name.as_ref();
        self.attributes()?
            .get(&name.to_string())
            .ok_or_else(|| Error::not_found(format!("no attribute named `{name}` in this PJRT topology")))
    }

    /// Collection of [`Topology`]-specific named attributes that are attached to this [`Topology`].
    pub fn attributes(&self) -> Result<&HashMap<String, Value>, Error> {
        self.attributes
            .get_or_init(|| {
                use ffi::PJRT_TopologyDescription_Attributes_Args;
                let (attributes, attribute_count) = invoke_pjrt_api_error_fn!(
                    self.api(),
                    PJRT_TopologyDescription_Attributes,
                    { topology = self.to_c_api() },
                    { attributes, num_attributes },
                )?;
                Ok(hash_map_from_c_api(attributes, attribute_count))
            })
            .as_ref()
            .map_err(|error| error.clone())
    }

    /// Returns a stable fingerprint for this [`Topology`]. Topologies with identical structure and device metadata
    /// should return the same fingerprint.
    pub fn fingerprint(&self) -> Result<u64, Error> {
        use ffi::PJRT_TopologyDescription_Fingerprint_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_TopologyDescription_Fingerprint, { topology = self.to_c_api() }, {
            fingerprint
        })
    }

    /// Returns the [`MemoryKindId`]s of all memory spaces in this [`Topology`].
    pub fn memory_kind_ids(&self) -> Result<Vec<MemoryKindId>, Error> {
        use ffi::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_TopologyDescription_GetMemorySpaceKindIds,
            { topology = self.to_c_api() },
            { memory_space_kind_ids, num_memory_space_kind_ids },
        )
        .map(|(memory_space_kind_ids, memory_space_kind_count)| {
            unsafe { slice_from_c_api(memory_space_kind_ids, memory_space_kind_count) }
                .iter()
                .map(|memory_kind_id| *memory_kind_id as MemoryKindId)
                .collect()
        })
    }

    /// Returns the canonical XLA [`Shape`] for the provided logical shape in the specified memory kind.
    pub fn canonical_shape_for_memory(
        &self,
        memory_kind_id: MemoryKindId,
        dimensions: &[i64],
        element_type: BufferType,
        layout: Option<&Layout>,
    ) -> Result<Shape, Error> {
        use ffi::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args;
        let element_type = unsafe { element_type.to_c_api() };
        let layout = layout.map(|layout| unsafe { layout.to_c_api() });
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace,
            {
                topology = self.to_c_api(),
                memory_space_kind_id = memory_kind_id as std::ffi::c_int,
                dims = dimensions.as_ptr(),
                num_dims = dimensions.len(),
                element_type = element_type,
                layout = layout.as_ref().map(|layout| layout as *const _).unwrap_or(std::ptr::null()),
            },
            { serialized_shape, serialized_shape_size, serialized_shape_deleter },
        )
        .and_then(|(serialized_shape, serialized_shape_size, serialized_shape_deleter)| {
            let serialized_shape_bytes =
                unsafe { slice_from_c_api(serialized_shape as *const u8, serialized_shape_size) }.to_vec();
            if let Some(deleter) = serialized_shape_deleter {
                unsafe { deleter(serialized_shape) };
            }
            Shape::decode(serialized_shape_bytes.as_slice())
                .map_err(|error| Error::invalid_argument(format!("failed to decode canonical shape Protobuf: {error}")))
        })
    }

    /// Serializes this [`Topology`] to its complete Protobuf representation, including platform metadata and the
    /// opaque platform-specific `google.protobuf.Any` payload.
    pub fn proto(&self) -> Result<crate::protos::Topology, Error> {
        // It would be nice to be able to get this directly without having to go through `Topology::serialize` first,
        // but the PJRT C API does not provide a direct Protobuf accessor.
        self.serialize()?.proto()
    }

    /// Serializes this [`Topology`] into a string (i.e., byte array).
    pub fn serialize(&self) -> Result<SerializedTopology, Error> {
        use ffi::PJRT_TopologyDescription_Serialize_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_TopologyDescription_Serialize,
            { topology = self.to_c_api() },
            { serialized_bytes, serialized_bytes_size, serialized_topology, serialized_topology_deleter },
        )
        .map(|(serialized_bytes, serialized_bytes_size, serialized_topology, serialized_topology_deleter)| {
            SerializedTopology {
                handle: serialized_topology,
                deleter: serialized_topology_deleter,
                data: serialized_bytes,
                data_size: serialized_bytes_size,
            }
        })
    }
}

impl Hash for Topology<'_> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.serialize().expect("failed to serialize PJRT topology").data().hash(hasher)
    }
}

impl Drop for Topology<'_> {
    fn drop(&mut self) {
        if !self.is_borrowed {
            use ffi::PJRT_TopologyDescription_Destroy_Args;
            invoke_pjrt_api_error_fn!(self.api(), PJRT_TopologyDescription_Destroy, { topology = self.to_c_api() })
                .expect("failed to destroy PJRT topology");
        }
    }
}

impl Client<'_> {
    /// Returns the runtime [`Topology`] of this [`Client`].
    pub fn topology(&'_ self) -> Result<Topology<'_>, Error> {
        use ffi::PJRT_Client_TopologyDescription_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Client_TopologyDescription, { client = self.to_c_api() }, {
            topology
        })
        .and_then(|handle| unsafe { Topology::from_c_api(handle, self.api(), true) })
    }

    /// Deserializes the provided Protobuf message into a [`Topology`].
    pub fn topology_from_proto(&self, proto: crate::protos::Topology) -> Result<Topology<'static>, Error> {
        self.api().topology_from_proto(proto)
    }

    /// Deserializes the provided data into a [`Topology`].
    pub fn deserialize_topology(&self, data: &[u8]) -> Result<Topology<'static>, Error> {
        self.api().deserialize_topology(data)
    }
}

impl Plugin {
    /// Constructs a new PJRT [`Topology`] using the provided name and (optional) platform-specific options.
    pub fn topology<N: AsRef<str>>(
        &self,
        name: N,
        options: HashMap<String, Value>,
    ) -> Result<Topology<'static>, Error> {
        self.api().topology(name, options)
    }

    /// Deserializes the provided Protobuf message into a [`Topology`].
    pub fn topology_from_proto(&self, proto: crate::protos::Topology) -> Result<Topology<'static>, Error> {
        self.api().topology_from_proto(proto)
    }

    /// Deserializes the provided data into a [`Topology`].
    pub fn deserialize_topology(&self, data: &[u8]) -> Result<Topology<'static>, Error> {
        self.api().deserialize_topology(data)
    }
}

impl Api {
    /// Constructs a new PJRT [`Topology`] using the provided name and (optional) platform-specific options.
    pub(crate) fn topology<N: AsRef<str>>(
        &self,
        name: N,
        options: HashMap<String, Value>,
    ) -> Result<Topology<'static>, Error> {
        use ffi::PJRT_TopologyDescription_Create_Args;
        let name = name.as_ref();
        let options = options.into_iter().map(|(name, value)| NamedValue::new(name, value)).collect::<Vec<_>>();
        let options = options.iter().map(|option| unsafe { option.to_c_api() }).collect::<Vec<_>>();
        invoke_pjrt_api_error_fn!(
            *self,
            PJRT_TopologyDescription_Create,
            {
                topology_name = name.as_ptr() as *const _,
                topology_name_size = name.len(),
                create_options = options.as_slice().as_ptr(),
                num_options = options.len(),
            },
            { topology },
        )
        .and_then(|handle| unsafe { Topology::from_c_api(handle, *self, false) })
    }

    /// Deserializes the provided Protobuf message into a [`Topology`].
    pub(crate) fn topology_from_proto(&self, proto: crate::protos::Topology) -> Result<Topology<'static>, Error> {
        self.deserialize_topology(&proto.encode_to_vec())
    }

    /// Deserializes the provided data into a [`Topology`].
    pub(crate) fn deserialize_topology(&self, data: &[u8]) -> Result<Topology<'static>, Error> {
        use ffi::PJRT_TopologyDescription_Deserialize_Args;
        invoke_pjrt_api_error_fn!(
            *self,
            PJRT_TopologyDescription_Deserialize,
            { serialized_topology = data.as_ptr() as *const _, serialized_topology_size = data.len() },
            { topology },
        )
        .and_then(|handle| unsafe { Topology::from_c_api(handle, *self, false) })
    }
}

/// Serialized [`Topology`].
pub struct SerializedTopology {
    /// Handle that represents this [`SerializedTopology`] in the PJRT C API.
    handle: *mut ffi::PJRT_SerializedTopology,

    /// Optional function that must be called to free the underlying memory when dropping this instance.
    deleter: Option<unsafe extern "C" fn(topology: *mut ffi::PJRT_SerializedTopology)>,

    /// Pointer to the underlying bytes of this [`SerializedTopology`].
    data: *const std::ffi::c_char,

    /// Size (i.e., number of bytes) of this [`SerializedTopology`].
    data_size: usize,
}

impl SerializedTopology {
    /// Returns a pointer to the underlying bytes of this [`SerializedTopology`].
    pub fn data(&self) -> &[u8] {
        unsafe { slice_from_c_api(self.data as *const _, self.data_size) }
    }

    /// Returns the complete Protobuf representation of this [`SerializedTopology`], preserving its platform metadata
    /// and opaque platform-specific `google.protobuf.Any` payload. The result can be passed directly to
    /// [`Client::topology_from_proto`] or [`Plugin::topology_from_proto`].
    pub fn proto(&self) -> Result<crate::protos::Topology, Error> {
        crate::protos::Topology::decode(self.data())
            .map_err(|error| Error::invalid_argument(format!("failed to decode topology Protobuf: {error}")))
    }
}

impl PartialEq for SerializedTopology {
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl Eq for SerializedTopology {}

impl Hash for SerializedTopology {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data().hash(state);
    }
}

unsafe impl Send for SerializedTopology {}
unsafe impl Sync for SerializedTopology {}

impl Drop for SerializedTopology {
    fn drop(&mut self) {
        if let Some(deleter) = self.deleter {
            unsafe { deleter(self.handle) };
        }
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::buffers::ffi::{PJRT_Buffer_MemoryLayout, PJRT_Buffer_Type};
    use crate::clients::ffi::PJRT_Client;
    use crate::devices::ffi::PJRT_DeviceDescription;
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;
    use crate::values::ffi::PJRT_NamedValue;

    #[repr(C)]
    pub struct PJRT_Client_TopologyDescription_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub topology: *mut PJRT_TopologyDescription,
    }

    impl PJRT_Client_TopologyDescription_Args {
        pub fn new(client: *mut PJRT_Client) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                topology: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Client_TopologyDescription =
        unsafe extern "C" fn(args: *mut PJRT_Client_TopologyDescription_Args) -> *mut PJRT_Error;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_TopologyDescription {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_TopologyDescription_Create_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology_name: *const std::ffi::c_char,
        pub topology_name_size: usize,
        pub create_options: *const PJRT_NamedValue,
        pub num_options: usize,
        pub topology: *mut PJRT_TopologyDescription,
    }

    impl PJRT_TopologyDescription_Create_Args {
        pub fn new(
            topology_name: *const std::ffi::c_char,
            topology_name_size: usize,
            create_options: *const PJRT_NamedValue,
            num_options: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                topology_name,
                topology_name_size,
                create_options,
                num_options,
                topology: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_TopologyDescription_Create =
        unsafe extern "C" fn(args: *mut PJRT_TopologyDescription_Create_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_TopologyDescription_PlatformName_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology: *const PJRT_TopologyDescription,
        pub platform_name: *const std::ffi::c_char,
        pub platform_name_size: usize,
    }

    impl PJRT_TopologyDescription_PlatformName_Args {
        pub fn new(topology: *mut PJRT_TopologyDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                topology,
                platform_name: std::ptr::null(),
                platform_name_size: 0,
            }
        }
    }

    pub type PJRT_TopologyDescription_PlatformName =
        unsafe extern "C" fn(args: *mut PJRT_TopologyDescription_PlatformName_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_TopologyDescription_PlatformVersion_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology: *mut PJRT_TopologyDescription,
        pub platform_version: *const std::ffi::c_char,
        pub platform_version_size: usize,
    }

    impl PJRT_TopologyDescription_PlatformVersion_Args {
        pub fn new(topology: *mut PJRT_TopologyDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                topology,
                platform_version: std::ptr::null(),
                platform_version_size: 0,
            }
        }
    }

    pub type PJRT_TopologyDescription_PlatformVersion =
        unsafe extern "C" fn(args: *mut PJRT_TopologyDescription_PlatformVersion_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_TopologyDescription_GetDeviceDescriptions_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology: *const PJRT_TopologyDescription,
        pub descriptions: *const *mut PJRT_DeviceDescription,
        pub num_descriptions: usize,
    }

    impl PJRT_TopologyDescription_GetDeviceDescriptions_Args {
        pub fn new(topology: *mut PJRT_TopologyDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                topology,
                descriptions: std::ptr::null(),
                num_descriptions: 0,
            }
        }
    }

    pub type PJRT_TopologyDescription_GetDeviceDescriptions =
        unsafe extern "C" fn(args: *mut PJRT_TopologyDescription_GetDeviceDescriptions_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_TopologyDescription_Attributes_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology: *mut PJRT_TopologyDescription,
        pub attributes: *const PJRT_NamedValue,
        pub num_attributes: usize,
    }

    impl PJRT_TopologyDescription_Attributes_Args {
        pub fn new(topology: *mut PJRT_TopologyDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                topology,
                attributes: std::ptr::null(),
                num_attributes: 0,
            }
        }
    }

    pub type PJRT_TopologyDescription_Attributes =
        unsafe extern "C" fn(args: *mut PJRT_TopologyDescription_Attributes_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_TopologyDescription_Fingerprint_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology: *const PJRT_TopologyDescription,
        pub fingerprint: u64,
    }

    impl PJRT_TopologyDescription_Fingerprint_Args {
        pub fn new(topology: *mut PJRT_TopologyDescription) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), topology, fingerprint: 0 }
        }
    }

    pub type PJRT_TopologyDescription_Fingerprint =
        unsafe extern "C" fn(args: *mut PJRT_TopologyDescription_Fingerprint_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_TopologyDescription_GetMemorySpaceKindIds_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology: *const PJRT_TopologyDescription,
        pub memory_space_kind_ids: *const std::ffi::c_int,
        pub num_memory_space_kind_ids: usize,
    }

    impl PJRT_TopologyDescription_GetMemorySpaceKindIds_Args {
        pub fn new(topology: *mut PJRT_TopologyDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                topology,
                memory_space_kind_ids: std::ptr::null(),
                num_memory_space_kind_ids: 0,
            }
        }
    }

    pub type PJRT_TopologyDescription_GetMemorySpaceKindIds =
        unsafe extern "C" fn(args: *mut PJRT_TopologyDescription_GetMemorySpaceKindIds_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology: *const PJRT_TopologyDescription,
        pub memory_space_kind_id: std::ffi::c_int,
        pub dims: *const i64,
        pub num_dims: usize,
        pub element_type: PJRT_Buffer_Type,
        pub layout: *const PJRT_Buffer_MemoryLayout,
        pub serialized_shape: *const std::ffi::c_char,
        pub serialized_shape_size: usize,
        pub serialized_shape_deleter: Option<unsafe extern "C" fn(serialized_shape: *const std::ffi::c_char)>,
    }

    impl PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args {
        pub fn new(
            topology: *mut PJRT_TopologyDescription,
            memory_space_kind_id: std::ffi::c_int,
            dims: *const i64,
            num_dims: usize,
            element_type: PJRT_Buffer_Type,
            layout: *const PJRT_Buffer_MemoryLayout,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                topology,
                memory_space_kind_id,
                dims,
                num_dims,
                element_type,
                layout,
                serialized_shape: std::ptr::null(),
                serialized_shape_size: 0,
                serialized_shape_deleter: None,
            }
        }
    }

    pub type PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace = unsafe extern "C" fn(
        args: *mut PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args,
    ) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_TopologyDescription_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology: *mut PJRT_TopologyDescription,
    }

    impl PJRT_TopologyDescription_Destroy_Args {
        pub fn new(topology: *mut PJRT_TopologyDescription) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), topology }
        }
    }

    pub type PJRT_TopologyDescription_Destroy =
        unsafe extern "C" fn(args: *mut PJRT_TopologyDescription_Destroy_Args) -> *mut PJRT_Error;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_SerializedTopology {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_TopologyDescription_Serialize_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub topology: *mut PJRT_TopologyDescription,
        pub serialized_bytes: *const std::ffi::c_char,
        pub serialized_bytes_size: usize,
        pub serialized_topology: *mut PJRT_SerializedTopology,
        pub serialized_topology_deleter:
            Option<unsafe extern "C" fn(serialized_topology: *mut PJRT_SerializedTopology)>,
    }

    impl PJRT_TopologyDescription_Serialize_Args {
        pub fn new(topology: *mut PJRT_TopologyDescription) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                topology,
                serialized_bytes: std::ptr::null(),
                serialized_bytes_size: 0,
                serialized_topology: std::ptr::null_mut(),
                serialized_topology_deleter: None,
            }
        }
    }

    pub type PJRT_TopologyDescription_Serialize =
        unsafe extern "C" fn(args: *mut PJRT_TopologyDescription_Serialize_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_TopologyDescription_Deserialize_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub serialized_topology: *const std::ffi::c_char,
        pub serialized_topology_size: usize,
        pub topology: *mut PJRT_TopologyDescription,
    }

    impl PJRT_TopologyDescription_Deserialize_Args {
        pub fn new(serialized_topology: *const std::ffi::c_char, serialized_topology_size: usize) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                serialized_topology,
                serialized_topology_size,
                topology: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_TopologyDescription_Deserialize =
        unsafe extern "C" fn(args: *mut PJRT_TopologyDescription_Deserialize_Args) -> *mut PJRT_Error;
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;
    use prost::Message;

    use crate::protos::{CpuTargetMachineOptions, CpuTopology, CpuTopologyDevice, GpuTopology};
    use crate::tests::{TestPlatform, test_for_each_platform};
    use crate::{Error, Topology};

    use super::SerializedTopology;

    #[test]
    fn test_client_topology() {
        test_for_each_platform!(|plugin, client, platform| {
            if platform == TestPlatform::Metal {
                assert!(matches!(client.topology(), Err(Error::Unimplemented { .. })));
            } else {
                let topology = client.topology().unwrap();
                assert!(topology.platform_name().is_ok());
                assert!(topology.platform_version().is_ok());
                assert!(topology.attribute("__test__").is_err());
                assert!(topology.attributes().is_ok());
                let platform_name = topology.platform_name().unwrap();
                let attributes = topology.attributes().unwrap();
                let fingerprint = topology.fingerprint();
                match platform {
                    TestPlatform::Mps => assert!(matches!(topology.serialize(), Err(Error::Unimplemented { .. }))),
                    _ => {
                        let serialized_topology = topology.serialize().unwrap();
                        assert!(!serialized_topology.data().is_empty());
                        let proto = serialized_topology.proto().unwrap();
                        assert_eq!(proto.platform_name.as_str(), platform_name.as_ref());
                        assert_eq!(proto.platform_version.as_str(), topology.platform_version().unwrap().as_ref());
                        assert_eq!(topology.proto(), Ok(proto.clone()));
                        let deserialized = client.deserialize_topology(serialized_topology.data());
                        match platform {
                            TestPlatform::Cpu | TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm10 => {
                                let deserialized = deserialized.unwrap();
                                assert_eq!(deserialized.platform_name(), topology.platform_name());
                                assert_eq!(deserialized.platform_version(), topology.platform_version());
                                if platform == TestPlatform::Cpu {
                                    assert_eq!(deserialized.attributes(), topology.attributes());
                                } else {
                                    // GPU serialization preserves device metadata, but not the live client's
                                    // target configuration and host compiler options.
                                    assert!(deserialized.attribute("device_memory_bytes_limit").is_ok());
                                }
                                assert_eq!(deserialized.fingerprint(), fingerprint);
                                let deserialized_proto = client.topology_from_proto(proto).unwrap();
                                assert_eq!(deserialized_proto.platform_name(), topology.platform_name());
                                assert_eq!(deserialized_proto.fingerprint(), fingerprint);
                            }
                            _ => assert!(deserialized.is_err()),
                        }
                    }
                }

                match platform {
                    TestPlatform::Cpu => {
                        assert_eq!(platform_name, "cpu");
                        assert!(attributes.is_empty());
                        assert!(fingerprint.is_ok());
                    }
                    TestPlatform::Cuda12 => {
                        assert_eq!(platform_name, "cuda");
                        assert!(!attributes.is_empty());
                        assert!(fingerprint.is_ok());
                    }
                    TestPlatform::Cuda13 => {
                        assert_eq!(platform_name, "cuda");
                        assert!(!attributes.is_empty());
                        assert!(fingerprint.is_ok());
                    }
                    TestPlatform::Rocm10 => {
                        assert_eq!(platform_name, "rocm");
                        assert!(!attributes.is_empty());
                        assert!(fingerprint.is_ok());
                    }
                    TestPlatform::Tpu => {
                        assert_eq!(platform_name, "tpu");
                        assert!(!attributes.is_empty());
                        assert!(fingerprint.is_ok());
                    }
                    TestPlatform::Neuron => {
                        assert_eq!(platform_name, "neuron");
                        assert!(!attributes.is_empty());
                        assert!(fingerprint.is_ok());
                    }
                    TestPlatform::Metal => {
                        assert_eq!(platform_name, "METAL");
                        assert!(!attributes.is_empty());
                        assert!(fingerprint.is_ok());
                    }
                    TestPlatform::Mps => {
                        assert_eq!(platform_name, "mps");
                        assert!(attributes.is_empty());
                        assert!(fingerprint.is_ok());
                    }
                }
            }

            // Test creating a `Topology` from a null pointer.
            assert!(matches!(
                unsafe { Topology::from_c_api(std::ptr::null_mut(), plugin.api(), false) },
                Err(Error::InvalidArgument { message, .. })
                    if message == "the provided PJRT topology handle is a null pointer",
            ));
        });
    }

    #[test]
    fn test_plugin_topology() {
        test_for_each_platform!(|plugin, _client, platform| {
            let topology = plugin.topology("test", HashMap::new());
            match platform {
                TestPlatform::Cpu | TestPlatform::Metal | TestPlatform::Mps => assert!(topology.is_err()),
                _ => assert!(topology.is_ok()),
            }
        });
    }

    #[test]
    fn test_serialized_topology_proto_cpu() {
        // The nested CPU device ID uses a non-UTF-8 varint. Treating the envelope as a GPU topology incorrectly
        // interprets this opaque payload as a string and rejects an otherwise valid topology.
        let data = b"\x08\x80\x01\x12\x03cpu\x1a\x01v\x4a\x3e\
                     \x0a\x28type.googleapis.com/xla.CpuTopologyProto\
                     \x12\x12\x0a\x05\x10\x02\x18\x80\x01\x2a\x09\x0a\x01t\x12\x01c\x1a\x01f";
        let serialized = SerializedTopology {
            handle: std::ptr::null_mut(),
            deleter: None,
            data: data.as_ptr().cast(),
            data_size: data.len(),
        };
        let proto = serialized.proto().unwrap();
        assert_eq!(proto.platform_id, 128);
        assert_eq!(proto.platform_name, "cpu");
        assert_eq!(proto.platform_version, "v");
        assert!(!proto.is_subslice_topology);
        let payload = proto.platform_specific_topology.unwrap();
        assert_eq!(payload.type_url, "type.googleapis.com/xla.CpuTopologyProto");
        assert_eq!(
            CpuTopology::decode(payload.value.as_slice()),
            Ok(CpuTopology {
                cpu_devices: vec![CpuTopologyDevice { process_index: 2, local_hardware_id: 128 }],
                target_machine_options: Some(CpuTargetMachineOptions {
                    triple: "t".to_string(),
                    cpu: "c".to_string(),
                    features: "f".to_string(),
                }),
            }),
        );
    }

    #[test]
    fn test_serialized_topology_proto_gpu() {
        let data = b"\x08\x80\x01\x12\x04cuda\x1a\x01v\x4a\x37\
                     \x0a\x28type.googleapis.com/xla.GpuTopologyProto\
                     \x12\x0b\x1a\x03gpu\x20\x01\x28\x02\x30\x03";
        let serialized = SerializedTopology {
            handle: std::ptr::null_mut(),
            deleter: None,
            data: data.as_ptr().cast(),
            data_size: data.len(),
        };
        let proto = serialized.proto().unwrap();
        assert_eq!(proto.platform_id, 128);
        assert_eq!(proto.platform_name, "cuda");
        assert_eq!(proto.platform_version, "v");
        assert!(!proto.is_subslice_topology);
        let payload = proto.platform_specific_topology.unwrap();
        assert_eq!(payload.type_url, "type.googleapis.com/xla.GpuTopologyProto");
        assert_eq!(
            GpuTopology::decode(payload.value.as_slice()),
            Ok(GpuTopology {
                platform_version: "gpu".to_string(),
                num_partitions: 1,
                num_hosts_per_partition: 2,
                num_devices_per_host: 3,
                ..Default::default()
            }),
        );
    }

    #[test]
    fn test_serialized_topology_proto_unknown_platform() {
        let data = b"\x08\x80\x01\x12\x03tpu\x1a\x01v\x4a\x31\
                     \x0a\x2ctype.googleapis.com/xla.UnknownTopologyProto\x12\x01\xff";
        let serialized = SerializedTopology {
            handle: std::ptr::null_mut(),
            deleter: None,
            data: data.as_ptr().cast(),
            data_size: data.len(),
        };
        let proto = serialized.proto().unwrap();
        assert_eq!(proto.platform_id, 128);
        assert_eq!(proto.platform_name, "tpu");
        assert_eq!(proto.platform_version, "v");
        assert!(!proto.is_subslice_topology);
        let payload = proto.platform_specific_topology.unwrap();
        assert_eq!(payload.type_url, "type.googleapis.com/xla.UnknownTopologyProto");
        assert_eq!(payload.value, vec![0xff]);
    }

    #[test]
    fn test_serialized_topology_proto_invalid_data() {
        let data = [0u8];
        let serialized = SerializedTopology {
            handle: std::ptr::null_mut(),
            deleter: None,
            data: data.as_ptr().cast(),
            data_size: data.len(),
        };
        assert!(matches!(
            serialized.proto(),
            Err(Error::InvalidArgument { message, .. })
                if message == "failed to decode topology Protobuf: \
                               failed to decode Protobuf message: \
                               invalid tag value: 0",
        ));
    }
}
