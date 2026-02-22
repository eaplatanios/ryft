#![allow(clippy::missing_safety_doc)]

use std::borrow::Cow;
use std::collections::HashMap;

pub mod buffers;
pub mod clients;
pub mod devices;
pub mod distributed;
pub mod errors;
pub mod events;
pub mod extensions;
pub mod memories;
pub mod plugins;
pub mod programs;
pub mod protos;
pub mod topologies;
pub mod transfers;
pub mod values;
pub mod versions;

pub use buffers::*;
pub use clients::*;
pub use devices::*;
pub use distributed::*;
pub use errors::*;
pub use events::*;
pub use memories::*;
pub use plugins::*;
pub use programs::*;
pub use topologies::*;
pub use transfers::*;
pub use values::*;
pub use versions::*;

pub(crate) mod macros;

pub(crate) use macros::{
    invoke_distributed_api_error_fn, invoke_distributed_api_fn_helper, invoke_distributed_api_void_fn,
    invoke_pjrt_api_error_fn, invoke_pjrt_api_fn_helper, invoke_pjrt_api_void_fn, invoke_xla_ffi_api_error_fn,
    invoke_xla_ffi_api_fn_helper, invoke_xla_ffi_api_void_fn,
};

/// Wrapper of a [`PJRT_Api`](ffi::PJRT_Api) handle that can be used to interact with the PJRT C API.
#[derive(Copy, Clone)]
pub(crate) struct Api {
    /// Handle that represents this [`Api`] in the PJRT C API.
    handle: *const ffi::PJRT_Api,
}

impl Api {
    /// Constructs a new [`Api`] from the provided [`PJRT_Api`](ffi::PJRT_Api) handle that came
    /// from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(handle: *const ffi::PJRT_Api) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT API handle is a null pointer"))
        } else {
            Ok(Self { handle })
        }
    }

    /// Returns the [`PJRT_Api`](ffi::PJRT_Api) that corresponds to this [`Api`] and which can
    /// be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_Api {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`]. Note that this function simply returns a copy of this [`Api`].
    /// It is only used as a helper for implementing our declarative macros in [`macros`] and being able to support
    /// both core PJRT C API functions and PJRT extension functions using the same macros.
    pub(crate) fn api(&self) -> Api {
        *self
    }

    /// Returns the PJRT version that this [`Api`] supports.
    pub(crate) fn version(&self) -> Version {
        let handle = unsafe { &(*self.to_c_api()).pjrt_api_version };
        Version { major: handle.major_version as usize, minor: handle.minor_version as usize }
    }

    /// [`Value`] of the attribute with the provided `name` attached to this PJRT [`Api`],
    /// or [`Error::NotFound`] if no such attribute is attached to this [`Api`].
    ///
    /// Note that this function is expensive in that it recreates the resulting [`HashMap`] each time it is invoked.
    /// [`Client::attribute`] is much more efficient and should be used instead when possible.
    pub(crate) fn attribute<N: AsRef<str>>(&self, name: N) -> Result<Value, Error> {
        let name = name.as_ref();
        self.attributes()?
            .get(name)
            .cloned()
            .ok_or_else(|| Error::not_found(format!("no attribute named '{name}' found in this PJRT plugin")))
    }

    /// Returns the PJRT attributes associated with this [`Api`] (e.g., the version of the XLA compiler that it was
    /// compiled against). The specific attributes returned depend on the PJRT [`Plugin`] implementation.
    ///
    /// Note that this function is expensive in that it recreates the resulting [`HashMap`] each time it is invoked.
    /// [`Client::attribute`] is much more efficient and should be used instead when possible.
    pub(crate) fn attributes(&self) -> Result<HashMap<String, Value>, Error> {
        use crate::plugins::ffi::PJRT_Plugin_Attributes_Args;
        let result = invoke_pjrt_api_error_fn!(*self, PJRT_Plugin_Attributes, {}, { attributes, num_attributes });
        let (attributes, attribute_count) = result?;
        Ok(hash_map_from_c_api(attributes, attribute_count))
    }
}

unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Client<'_> {
    /// Returns the PJRT version that this [`Client`] supports.
    pub fn version(&self) -> Version {
        self.api().version()
    }
}

impl Plugin {
    /// Returns the PJRT version that this [`Plugin`] supports.
    pub fn version(&self) -> Version {
        self.api().version()
    }

    /// [`Value`] of the attribute with the provided `name` attached to this PJRT [`Plugin`],
    /// or [`Error::NotFound`] if no such attribute is attached to this [`Plugin`].
    ///
    /// Note that this function is expensive in that it recreates the resulting [`HashMap`] each time it is invoked.
    /// [`Client::attribute`] is much more efficient and should be used instead when possible.
    pub fn attribute<N: AsRef<str>>(&self, name: N) -> Result<Value, Error> {
        self.api().attribute(name)
    }

    /// Returns the PJRT attributes associated with this [`Plugin`] (e.g., the version of the XLA compiler that it was
    /// compiled against). The specific attributes returned depend on the PJRT [`Plugin`] implementation.
    ///
    /// Note that this function is expensive in that it recreates the resulting [`HashMap`] each time it is invoked.
    /// [`Client::attribute`] is much more efficient and should be used instead when possible.
    pub fn attributes(&self) -> Result<HashMap<String, Value>, Error> {
        self.api().attributes()
    }
}

/// Returns an [`str`] representation for the provided C string. Note that the returned value is a [`Cow`] because
/// this function will avoid creating a copy of the C string unless it really needs to (e.g., for UTF-8 conversion).
pub(crate) fn str_from_c_api<'a>(ptr: *const std::ffi::c_char, size: usize) -> Cow<'a, str> {
    String::from_utf8_lossy(unsafe { slice_from_c_api(ptr as *const u8, size) })
}

/// Returns a [`HashMap`] representation for the provided [`PJRT_NamedValue`] array.
pub(crate) fn hash_map_from_c_api(ptr: *const values::ffi::PJRT_NamedValue, size: usize) -> HashMap<String, Value> {
    unsafe { slice_from_c_api(ptr, size) }
        .iter()
        .map(|value| unsafe { NamedValue::from_c_api(value) })
        .map(|named_value| (named_value.name, named_value.value))
        .collect::<HashMap<String, Value>>()
}

/// Returns a slice from C API pointer and size pair, treating null pointers and zero sizes as empty slices.
/// The reason we need this helper function is that [`std::slice::from_raw_parts`] results in undefined behavior
/// if the provided pointer is null or the size is zero.
pub(crate) unsafe fn slice_from_c_api<'a, T>(ptr: *const T, size: usize) -> &'a [T] {
    if ptr.is_null() || size == 0 { &[] } else { unsafe { std::slice::from_raw_parts(ptr, size) } }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::buffers::ffi::*;
    use crate::clients::ffi::*;
    use crate::devices::ffi::*;
    use crate::errors::ffi::*;
    use crate::events::ffi::*;
    use crate::memories::ffi::*;
    use crate::plugins::ffi::*;
    use crate::programs::ffi::*;
    use crate::topologies::ffi::*;
    use crate::transfers::ffi::*;
    use crate::versions::ffi::*;

    pub type PJRT_Extension_Type = std::ffi::c_uint;
    pub const PJRT_Extension_Type_Gpu_Custom_Call: PJRT_Extension_Type = 0;
    pub const PJRT_Extension_Type_Profiler: PJRT_Extension_Type = 1;
    pub const PJRT_Extension_Type_Custom_Partitioner: PJRT_Extension_Type = 2;
    pub const PJRT_Extension_Type_Stream: PJRT_Extension_Type = 3;
    pub const PJRT_Extension_Type_Layouts: PJRT_Extension_Type = 4;
    pub const PJRT_Extension_Type_FFI: PJRT_Extension_Type = 5;
    pub const PJRT_Extension_Type_MemoryDescriptions: PJRT_Extension_Type = 6;
    pub const PJRT_Extension_Type_Triton: PJRT_Extension_Type = 7;
    pub const PJRT_Extension_Type_RawBuffer: PJRT_Extension_Type = 8;
    pub const PJRT_Extension_Type_PhaseCompile: PJRT_Extension_Type = 9;
    pub const PJRT_Extension_Type_Example: PJRT_Extension_Type = 10;
    pub const PJRT_Extension_Type_Unknown: PJRT_Extension_Type = 11;
    pub const PJRT_Extension_Type_CrossHostTransfers: PJRT_Extension_Type = 12;
    pub const PJRT_Extension_Type_ExecutableMetadata: PJRT_Extension_Type = 13;
    pub const PJRT_Extension_Type_Callback: PJRT_Extension_Type = 14;
    pub const PJRT_Extension_Type_HostAllocator: PJRT_Extension_Type = 15;
    pub const PJRT_Extension_Type_TpuTopology: PJRT_Extension_Type = 16;
    pub const PJRT_Extension_Type_TpuExecutable: PJRT_Extension_Type = 17;
    pub const PJRT_Extension_Type_Megascale: PJRT_Extension_Type = 18;
    pub const PJRT_Extension_Type_Shardings: PJRT_Extension_Type = 19;

    /// PJRT extension base type. The `extension_type` field must be used to identify the type of the extension
    /// and reinterpret its instance accordingly.
    #[repr(C)]
    pub struct PJRT_Extension_Base {
        pub struct_size: usize,
        pub extension_type: PJRT_Extension_Type,
        pub next: *mut PJRT_Extension_Base,
    }

    #[repr(C)]
    pub struct PJRT_Api {
        // For backwards compatibility, callers must use this value to guard accesses to fields
        // that may have been added after the plugin version they are interacting with was released.
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub pjrt_api_version: PJRT_Api_Version,

        pub PJRT_Error_Destroy: Option<PJRT_Error_Destroy>,
        pub PJRT_Error_Message: Option<PJRT_Error_Message>,
        pub PJRT_Error_GetCode: Option<PJRT_Error_GetCode>,

        pub PJRT_Plugin_Initialize: Option<PJRT_Plugin_Initialize>,
        pub PJRT_Plugin_Attributes: Option<PJRT_Plugin_Attributes>,

        pub PJRT_Event_Destroy: Option<PJRT_Event_Destroy>,
        pub PJRT_Event_IsReady: Option<PJRT_Event_IsReady>,
        pub PJRT_Event_Error: Option<PJRT_Event_Error>,
        pub PJRT_Event_Await: Option<PJRT_Event_Await>,
        pub PJRT_Event_OnReady: Option<PJRT_Event_OnReady>,

        pub PJRT_Client_Create: Option<PJRT_Client_Create>,
        pub PJRT_Client_Destroy: Option<PJRT_Client_Destroy>,
        pub PJRT_Client_PlatformName: Option<PJRT_Client_PlatformName>,
        pub PJRT_Client_ProcessIndex: Option<PJRT_Client_ProcessIndex>,
        pub PJRT_Client_PlatformVersion: Option<PJRT_Client_PlatformVersion>,
        pub PJRT_Client_Devices: Option<PJRT_Client_Devices>,
        pub PJRT_Client_AddressableDevices: Option<PJRT_Client_AddressableDevices>,
        pub PJRT_Client_LookupDevice: Option<PJRT_Client_LookupDevice>,
        pub PJRT_Client_LookupAddressableDevice: Option<PJRT_Client_LookupAddressableDevice>,
        pub PJRT_Client_AddressableMemories: Option<PJRT_Client_AddressableMemories>,
        pub PJRT_Client_Compile: Option<PJRT_Client_Compile>,
        pub PJRT_Client_DefaultDeviceAssignment: Option<PJRT_Client_DefaultDeviceAssignment>,
        pub PJRT_Client_BufferFromHostBuffer: Option<PJRT_Client_BufferFromHostBuffer>,

        pub PJRT_DeviceDescription_Id: Option<PJRT_DeviceDescription_Id>,
        pub PJRT_DeviceDescription_ProcessIndex: Option<PJRT_DeviceDescription_ProcessIndex>,
        pub PJRT_DeviceDescription_Attributes: Option<PJRT_DeviceDescription_Attributes>,
        pub PJRT_DeviceDescription_Kind: Option<PJRT_DeviceDescription_Kind>,
        pub PJRT_DeviceDescription_DebugString: Option<PJRT_DeviceDescription_DebugString>,
        pub PJRT_DeviceDescription_ToString: Option<PJRT_DeviceDescription_ToString>,

        pub PJRT_Device_GetDescription: Option<PJRT_Device_GetDescription>,
        pub PJRT_Device_IsAddressable: Option<PJRT_Device_IsAddressable>,
        pub PJRT_Device_LocalHardwareId: Option<PJRT_Device_LocalHardwareId>,
        pub PJRT_Device_AddressableMemories: Option<PJRT_Device_AddressableMemories>,
        pub PJRT_Device_DefaultMemory: Option<PJRT_Device_DefaultMemory>,
        pub PJRT_Device_MemoryStats: Option<PJRT_Device_MemoryStats>,

        pub PJRT_Memory_Id: Option<PJRT_Memory_Id>,
        pub PJRT_Memory_Kind: Option<PJRT_Memory_Kind>,
        pub PJRT_Memory_DebugString: Option<PJRT_Memory_DebugString>,
        pub PJRT_Memory_ToString: Option<PJRT_Memory_ToString>,
        pub PJRT_Memory_AddressableByDevices: Option<PJRT_Memory_AddressableByDevices>,

        pub PJRT_Executable_Destroy: Option<PJRT_Executable_Destroy>,
        pub PJRT_Executable_Name: Option<PJRT_Executable_Name>,
        pub PJRT_Executable_NumReplicas: Option<PJRT_Executable_NumReplicas>,
        pub PJRT_Executable_NumPartitions: Option<PJRT_Executable_NumPartitions>,
        pub PJRT_Executable_NumOutputs: Option<PJRT_Executable_NumOutputs>,
        pub PJRT_Executable_SizeOfGeneratedCodeInBytes: Option<PJRT_Executable_SizeOfGeneratedCodeInBytes>,
        pub PJRT_Executable_GetCostAnalysis: Option<PJRT_Executable_GetCostAnalysis>,
        pub PJRT_Executable_OutputMemoryKinds: Option<PJRT_Executable_OutputMemoryKinds>,
        pub PJRT_Executable_OptimizedProgram: Option<PJRT_Executable_OptimizedProgram>,
        pub PJRT_Executable_Serialize: Option<PJRT_Executable_Serialize>,

        pub PJRT_LoadedExecutable_Destroy: Option<PJRT_LoadedExecutable_Destroy>,
        pub PJRT_LoadedExecutable_GetExecutable: Option<PJRT_LoadedExecutable_GetExecutable>,
        pub PJRT_LoadedExecutable_AddressableDevices: Option<PJRT_LoadedExecutable_AddressableDevices>,
        pub PJRT_LoadedExecutable_Delete: Option<PJRT_LoadedExecutable_Delete>,
        pub PJRT_LoadedExecutable_IsDeleted: Option<PJRT_LoadedExecutable_IsDeleted>,
        pub PJRT_LoadedExecutable_Execute: Option<PJRT_LoadedExecutable_Execute>,
        pub PJRT_Executable_DeserializeAndLoad: Option<PJRT_Executable_DeserializeAndLoad>,
        pub PJRT_LoadedExecutable_Fingerprint: Option<PJRT_LoadedExecutable_Fingerprint>,

        pub PJRT_Buffer_Destroy: Option<PJRT_Buffer_Destroy>,
        pub PJRT_Buffer_ElementType: Option<PJRT_Buffer_ElementType>,
        pub PJRT_Buffer_Dimensions: Option<PJRT_Buffer_Dimensions>,
        pub PJRT_Buffer_UnpaddedDimensions: Option<PJRT_Buffer_UnpaddedDimensions>,
        pub PJRT_Buffer_DynamicDimensionIndices: Option<PJRT_Buffer_DynamicDimensionIndices>,
        pub PJRT_Buffer_GetMemoryLayout: Option<PJRT_Buffer_GetMemoryLayout>,
        pub PJRT_Buffer_OnDeviceSizeInBytes: Option<PJRT_Buffer_OnDeviceSizeInBytes>,
        pub PJRT_Buffer_Device: Option<PJRT_Buffer_Device>,
        pub PJRT_Buffer_Memory: Option<PJRT_Buffer_Memory>,
        pub PJRT_Buffer_Delete: Option<PJRT_Buffer_Delete>,
        pub PJRT_Buffer_IsDeleted: Option<PJRT_Buffer_IsDeleted>,
        pub PJRT_Buffer_CopyToDevice: Option<PJRT_Buffer_CopyToDevice>,
        pub PJRT_Buffer_ToHostBuffer: Option<PJRT_Buffer_ToHostBuffer>,
        pub PJRT_Buffer_IsOnCpu: Option<PJRT_Buffer_IsOnCpu>,
        pub PJRT_Buffer_ReadyEvent: Option<PJRT_Buffer_ReadyEvent>,
        pub PJRT_Buffer_UnsafePointer: Option<PJRT_Buffer_UnsafePointer>,
        pub PJRT_Buffer_IncreaseExternalReferenceCount: Option<PJRT_Buffer_IncreaseExternalReferenceCount>,
        pub PJRT_Buffer_DecreaseExternalReferenceCount: Option<PJRT_Buffer_DecreaseExternalReferenceCount>,
        pub PJRT_Buffer_OpaqueDeviceMemoryDataPointer: Option<PJRT_Buffer_OpaqueDeviceMemoryDataPointer>,

        pub PJRT_CopyToDeviceStream_Destroy: Option<PJRT_CopyToDeviceStream_Destroy>,
        pub PJRT_CopyToDeviceStream_AddChunk: Option<PJRT_CopyToDeviceStream_AddChunk>,
        pub PJRT_CopyToDeviceStream_TotalBytes: Option<PJRT_CopyToDeviceStream_TotalBytes>,
        pub PJRT_CopyToDeviceStream_GranuleSize: Option<PJRT_CopyToDeviceStream_GranuleSize>,
        pub PJRT_CopyToDeviceStream_CurrentBytes: Option<PJRT_CopyToDeviceStream_CurrentBytes>,

        pub PJRT_TopologyDescription_Create: Option<PJRT_TopologyDescription_Create>,
        pub PJRT_TopologyDescription_Destroy: Option<PJRT_TopologyDescription_Destroy>,
        pub PJRT_TopologyDescription_PlatformName: Option<PJRT_TopologyDescription_PlatformName>,
        pub PJRT_TopologyDescription_PlatformVersion: Option<PJRT_TopologyDescription_PlatformVersion>,
        pub PJRT_TopologyDescription_GetDeviceDescriptions: Option<PJRT_TopologyDescription_GetDeviceDescriptions>,
        pub PJRT_TopologyDescription_Serialize: Option<PJRT_TopologyDescription_Serialize>,
        pub PJRT_TopologyDescription_Attributes: Option<PJRT_TopologyDescription_Attributes>,

        pub PJRT_Compile: Option<PJRT_Compile>,

        pub PJRT_Executable_OutputElementTypes: Option<PJRT_Executable_OutputElementTypes>,
        pub PJRT_Executable_OutputDimensions: Option<PJRT_Executable_OutputDimensions>,

        pub PJRT_Buffer_CopyToMemory: Option<PJRT_Buffer_CopyToMemory>,

        pub PJRT_Client_CreateViewOfDeviceBuffer: Option<PJRT_Client_CreateViewOfDeviceBuffer>,

        pub PJRT_Executable_Fingerprint: Option<PJRT_Executable_Fingerprint>,

        pub PJRT_Client_TopologyDescription: Option<PJRT_Client_TopologyDescription>,

        pub PJRT_Executable_GetCompiledMemoryStats: Option<PJRT_Executable_GetCompiledMemoryStats>,

        pub PJRT_Memory_Kind_Id: Option<PJRT_Memory_Kind_Id>,

        pub PJRT_ExecuteContext_Create: Option<PJRT_ExecuteContext_Create>,
        pub PJRT_ExecuteContext_Destroy: Option<PJRT_ExecuteContext_Destroy>,

        pub PJRT_Buffer_CopyRawToHost: Option<PJRT_Buffer_CopyRawToHost>,

        pub PJRT_AsyncHostToDeviceTransferManager_Destroy: Option<PJRT_AsyncHostToDeviceTransferManager_Destroy>,
        pub PJRT_AsyncHostToDeviceTransferManager_TransferData:
            Option<PJRT_AsyncHostToDeviceTransferManager_TransferData>,
        pub PJRT_Client_CreateBuffersForAsyncHostToDevice: Option<PJRT_Client_CreateBuffersForAsyncHostToDevice>,
        pub PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer:
            Option<PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer>,
        pub PJRT_AsyncHostToDeviceTransferManager_Device: Option<PJRT_AsyncHostToDeviceTransferManager_Device>,
        pub PJRT_AsyncHostToDeviceTransferManager_BufferCount:
            Option<PJRT_AsyncHostToDeviceTransferManager_BufferCount>,
        pub PJRT_AsyncHostToDeviceTransferManager_BufferSize: Option<PJRT_AsyncHostToDeviceTransferManager_BufferSize>,
        pub PJRT_AsyncHostToDeviceTransferManager_SetBufferError:
            Option<PJRT_AsyncHostToDeviceTransferManager_SetBufferError>,
        pub PJRT_AsyncHostToDeviceTransferManager_AddMetadata:
            Option<PJRT_AsyncHostToDeviceTransferManager_AddMetadata>,
        pub PJRT_Client_DmaMap: Option<PJRT_Client_DmaMap>,
        pub PJRT_Client_DmaUnmap: Option<PJRT_Client_DmaUnmap>,

        pub PJRT_Client_CreateUninitializedBuffer: Option<PJRT_Client_CreateUninitializedBuffer>,
        pub PJRT_Client_UpdateGlobalProcessInfo: Option<PJRT_Client_UpdateGlobalProcessInfo>,

        pub PJRT_TopologyDescription_Deserialize: Option<PJRT_TopologyDescription_Deserialize>,
        pub PJRT_Client_CreateAliasBuffer: Option<PJRT_Client_CreateAliasBuffer>,
        pub PJRT_Client_FulfillAliasBuffer: Option<PJRT_Client_FulfillAliasBuffer>,
        pub PJRT_LoadedExecutable_GetDeviceAssignment: Option<PJRT_LoadedExecutable_GetDeviceAssignment>,
        pub PJRT_Client_CreateErrorBuffer: Option<PJRT_Client_CreateErrorBuffer>,
        pub PJRT_AsyncHostToDeviceTransferManager_TransferLiteral:
            Option<PJRT_AsyncHostToDeviceTransferManager_TransferLiteral>,
        pub PJRT_Buffer_CopyRawToHostFuture: Option<PJRT_Buffer_CopyRawToHostFuture>,

        pub PJRT_Device_PoisonExecution: Option<PJRT_Device_PoisonExecution>,
        pub PJRT_Device_CreateAsyncTrackingEvent: Option<PJRT_Device_CreateAsyncTrackingEvent>,
        pub PJRT_AsyncTrackingEvent_Destroy: Option<PJRT_AsyncTrackingEvent_Destroy>,

        pub PJRT_Executable_GetCompileOptions: Option<PJRT_Executable_GetCompileOptions>,

        pub PJRT_Buffer_DonateWithControlDependency: Option<PJRT_Buffer_DonateWithControlDependency>,

        pub PJRT_Event_Create: Option<PJRT_Event_Create>,
        pub PJRT_Event_Set: Option<PJRT_Event_Set>,
    }
}

#[cfg(test)]
mod tests {
    use crate::versions::ffi::{PJRT_API_MAJOR, PJRT_API_MINOR};
    use crate::{
        Api, Client, ClientOptions, CpuClientOptions, Error, NamedValue, Plugin, Value, Version, hash_map_from_c_api,
        load_cpu_plugin, str_from_c_api,
    };

    /// Platform identifier used by [`test_for_each_platform`] to signal which backend is being tested.
    #[allow(dead_code)]
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub(crate) enum TestPlatform {
        Cpu,
        Cuda12,
        Cuda13,
        Rocm7,
        Tpu,
        Neuron,
        Metal,
    }

    /// Executes the provided block once per enabled test platform backend. The CPU backend is always tested, and then
    /// other accelerator backends are included only when their corresponding cargo features are enabled (i.e.,
    /// `cuda-12`, `cuda-13`, `rocm-7`, `tpu`, `neuron`, and `metal`).
    ///
    /// # Parameters
    ///
    ///   - `$plugin`: Identifier bound to the corresponding backend [`Plugin`] value for each iteration.
    ///   - `$client`: Optional identifier bound to the backend-specific [`Client`] value for each iteration.
    ///     If there are only two parameters, then `$client` is not bound and no [`Client`] will be created.
    ///   - `$platform`: Identifier bound to the [`TestPlatform`] enum variant for the current iteration.
    ///   - `$body`: Block executed once per included backend using the bound identifiers above.
    /// ```
    macro_rules! test_for_each_platform {
        (|$plugin:ident, $platform:ident| $body:block) => {{
            {
                let $plugin = $crate::tests::test_cpu_plugin();
                let $platform = $crate::tests::TestPlatform::Cpu;
                $body
            }

            #[cfg(feature = "cuda-12")]
            {
                let $plugin = $crate::load_cuda_12_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT CUDA 12 plugin");
                let $platform = $crate::tests::TestPlatform::Cuda12;
                $body
            }

            #[cfg(feature = "cuda-13")]
            {
                let $plugin = $crate::load_cuda_13_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT CUDA 13 plugin");
                let $platform = $crate::tests::TestPlatform::Cuda13;
                $body
            }

            #[cfg(feature = "rocm-7")]
            {
                let $plugin = $crate::load_rocm_7_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT ROCm 7 plugin");
                let $platform = $crate::tests::TestPlatform::Rocm7;
                $body
            }

            #[cfg(feature = "tpu")]
            {
                let $plugin = $crate::load_tpu_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT TPU plugin");
                let $platform = $crate::tests::TestPlatform::Tpu;
                $body
            }

            #[cfg(feature = "neuron")]
            {
                let $plugin = $crate::load_neuron_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT AWS Neuron plugin");
                let $platform = $crate::tests::TestPlatform::Neuron;
                $body
            }

            #[cfg(feature = "metal")]
            {
                let $plugin = $crate::load_metal_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT Metal plugin");
                let $platform = $crate::tests::TestPlatform::Metal;
                $body
            }
        }};

        (|$plugin:ident, $client:ident, $platform:ident| $body:block) => {{
            {
                let $plugin = $crate::tests::test_cpu_plugin();
                let $client =
                    $plugin.client($crate::ClientOptions::CPU($crate::CpuClientOptions { device_count: Some(8) }));
                let $client = $client.expect("failed to create a PJRT CPU client");
                let $platform = $crate::tests::TestPlatform::Cpu;
                $body
            }

            #[cfg(feature = "cuda-12")]
            {
                let $plugin = $crate::load_cuda_12_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT CUDA 12 plugin");
                let $client = $plugin.client($crate::ClientOptions::GPU($crate::GpuClientOptions {
                    allocator: $crate::GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                    ..Default::default()
                }));
                let $client = $client.expect("failed to create a PJRT CUDA 12 client");
                let $platform = $crate::tests::TestPlatform::Cuda12;
                $body
            }

            #[cfg(feature = "cuda-13")]
            {
                let $plugin = $crate::load_cuda_13_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT CUDA 13 plugin");
                let $client = $plugin.client($crate::ClientOptions::GPU($crate::GpuClientOptions {
                    allocator: $crate::GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                    ..Default::default()
                }));
                let $client = $client.expect("failed to create a PJRT CUDA 13 client");
                let $platform = $crate::tests::TestPlatform::Cuda13;
                $body
            }

            #[cfg(feature = "rocm-7")]
            {
                let $plugin = $crate::load_rocm_7_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT ROCm 7 plugin");
                let $client = $plugin.client($crate::ClientOptions::GPU($crate::GpuClientOptions {
                    allocator: $crate::GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                    ..Default::default()
                }));
                let $client = $client.expect("failed to create a PJRT ROCm 7 client");
                let $platform = $crate::tests::TestPlatform::Rocm7;
                $body
            }

            #[cfg(feature = "tpu")]
            {
                let $plugin = $crate::load_tpu_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT TPU plugin");
                let $client = $plugin.client($crate::ClientOptions::default());
                let $client = $client.expect("failed to create a PJRT TPU client");
                let $platform = $crate::tests::TestPlatform::Tpu;
                $body
            }

            #[cfg(feature = "neuron")]
            {
                let $plugin = $crate::load_neuron_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT AWS Neuron plugin");
                let $client = $plugin.client($crate::ClientOptions::default());
                let $client = $client.expect("failed to create a PJRT AWS Neuron client");
                let $platform = $crate::tests::TestPlatform::Neuron;
                $body
            }

            #[cfg(feature = "metal")]
            {
                let $plugin = $crate::load_metal_plugin();
                let $plugin = $plugin.expect("failed to load the PJRT Metal plugin");
                let $client = $plugin.client($crate::ClientOptions::default());
                let $client = $client.expect("failed to create a PJRT Metal client");
                let $platform = $crate::tests::TestPlatform::Metal;
                $body
            }
        }};
    }

    pub(crate) use test_for_each_platform;

    pub(crate) fn test_cpu_plugin() -> Plugin {
        load_cpu_plugin().expect("failed to load the built-in PJRT CPU plugin")
    }

    pub(crate) fn test_cpu_client() -> Client<'static> {
        test_cpu_plugin()
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(8) }))
            .expect("failed to create a PJRT CPU client")
    }

    #[test]
    fn test_api() {
        // Test creating an [`Api`] from a null pointer.
        assert!(matches!(
            unsafe { Api::from_c_api(std::ptr::null()) },
            Err(Error::InvalidArgument { message, .. }) if message == "the provided PJRT API handle is a null pointer",
        ));

        // Test constructing valid [`Api`]s across all supported platforms.
        test_for_each_platform!(|plugin, client, platform| {
            match platform {
                TestPlatform::Metal => {
                    assert_eq!(plugin.version(), Version { major: 0, minor: 47 });
                    assert_eq!(client.version(), Version { major: 0, minor: 47 });
                    assert_eq!(plugin.api().version(), Version { major: 0, minor: 47 });
                }
                _ => {
                    assert_eq!(
                        plugin.version(),
                        Version { major: PJRT_API_MAJOR as usize, minor: PJRT_API_MINOR as usize },
                    );
                    assert_eq!(
                        client.version(),
                        Version { major: PJRT_API_MAJOR as usize, minor: PJRT_API_MINOR as usize },
                    );
                    assert_eq!(
                        plugin.api().version(),
                        Version { major: PJRT_API_MAJOR as usize, minor: PJRT_API_MINOR as usize },
                    );
                }
            };
        });

        let plugin = test_cpu_plugin();
        let api = plugin.api();
        assert_eq!(plugin.attribute("stablehlo_current_version"), Ok(Value::i64_list([1, 13, 7])));
        assert_eq!(plugin.attribute("stablehlo_minimum_version"), Ok(Value::i64_list([0, 9, 0])));
        assert_eq!(plugin.attribute("xla_version"), Ok(Value::i64(2)));
        assert_eq!(plugin.attribute("xla_version"), api.attribute("xla_version"));
        assert!(matches!(
            plugin.attribute("__missing__"),
            Err(Error::NotFound { message, .. }) if message.contains("__missing__")));
        let attributes = plugin.attributes().unwrap();
        assert_eq!(attributes.get("stablehlo_current_version"), Some(&Value::i64_list([1, 13, 7])));
        assert_eq!(attributes.get("stablehlo_minimum_version"), Some(&Value::i64_list([0, 9, 0])));
        assert_eq!(attributes.get("xla_version"), Some(&Value::i64(2)));
        assert_eq!(attributes.get("__missing__"), None);

        // Check that PJRT extensions can be loaded successfully.
        assert!(plugin.ffi_extension().is_ok());
        assert!(plugin.memory_descriptions_extension().is_ok());
        assert!(plugin.layouts_extension().is_ok());
    }

    #[test]
    fn test_str_from_c_api() {
        // Test using a null pointer.
        let str = str_from_c_api(std::ptr::null(), 7);
        assert!(matches!(str, std::borrow::Cow::Borrowed("")));
        assert_eq!(str, "");

        // Testing using a valid UTF-8 string.
        let string = b"cpu";
        let string = str_from_c_api(string.as_ptr() as *const std::ffi::c_char, string.len());
        assert!(matches!(string, std::borrow::Cow::Borrowed("cpu")));
        assert_eq!(string, "cpu");

        // Test using an invalid UTF-8 string.
        let string = [b'c', b'p', 0x80];
        let string = str_from_c_api(string.as_ptr() as *const std::ffi::c_char, string.len());
        assert!(matches!(string, std::borrow::Cow::Owned(_)));
        assert_eq!(string, "cp\u{fffd}");
    }

    #[test]
    fn test_hash_map_from_c_api() {
        // Test using a null pointer.
        assert!(hash_map_from_c_api(std::ptr::null(), 0).is_empty());

        // Test using a non-empty list of [`NamedValue`]s.
        let values = vec![
            NamedValue::new("boolean", true),
            NamedValue::new("integer", 42_i64),
            NamedValue::new("list", vec![1_i64, 2_i64, 3_i64]),
            NamedValue::new("string", "hello"),
        ];
        let values = values.iter().map(|value| unsafe { value.to_c_api() }).collect::<Vec<_>>();
        let hash_map = hash_map_from_c_api(values.as_ptr(), values.len());
        assert_eq!(hash_map.len(), 4);
        assert_eq!(hash_map.get("boolean"), Some(&Value::r#bool(true)));
        assert_eq!(hash_map.get("integer"), Some(&Value::i64(42)));
        assert_eq!(hash_map.get("list"), Some(&Value::i64_list([1, 2, 3])));
        assert_eq!(hash_map.get("string"), Some(&Value::string("hello")));
    }
}
