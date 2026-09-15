//! Exact HIP 7.13 C ABI used by the concrete launcher.
//!
//! `hip_runtime_api.h` SHA-256: `2b10f6e53712d7d6fdc09ad74f1b8eb7feb7bbb0a71598d60ee408d19446df07`.
//! The versioned properties structure has been checked with Linux/aarch64 and Linux/x86_64 C layout probes.

use std::ffi::{c_char, c_void};

/// Native device properties returned by `hipGetDevicePropertiesR0600`.
///
/// The bitfield-only `hipDeviceArch_t` occupies one native unsigned integer; no bitfields are inspected.
#[repr(C)]
#[allow(non_snake_case)]
pub(super) struct HipDeviceProperties {
    /// Native `name` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) name: [c_char; 256],

    /// Native `uuid` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) uuid: [c_char; 16],

    /// Native `luid` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) luid: [c_char; 8],

    /// Native `luidDeviceNodeMask` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) luidDeviceNodeMask: u32,

    /// Native `totalGlobalMem` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) totalGlobalMem: usize,

    /// Native `sharedMemPerBlock` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) sharedMemPerBlock: usize,

    /// Native `regsPerBlock` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) regsPerBlock: i32,

    /// Native `warpSize` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) warpSize: i32,

    /// Native `memPitch` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) memPitch: usize,

    /// Native `maxThreadsPerBlock` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxThreadsPerBlock: i32,

    /// Native `maxThreadsDim` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxThreadsDim: [i32; 3],

    /// Native `maxGridSize` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxGridSize: [i32; 3],

    /// Native `clockRate` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) clockRate: i32,

    /// Native `totalConstMem` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) totalConstMem: usize,

    /// Native `major` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) major: i32,

    /// Native `minor` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) minor: i32,

    /// Native `textureAlignment` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) textureAlignment: usize,

    /// Native `texturePitchAlignment` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) texturePitchAlignment: usize,

    /// Native `deviceOverlap` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) deviceOverlap: i32,

    /// Native `multiProcessorCount` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) multiProcessorCount: i32,

    /// Native `kernelExecTimeoutEnabled` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) kernelExecTimeoutEnabled: i32,

    /// Native `integrated` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) integrated: i32,

    /// Native `canMapHostMemory` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) canMapHostMemory: i32,

    /// Native `computeMode` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) computeMode: i32,

    /// Native `maxTexture1D` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTexture1D: i32,

    /// Native `maxTexture1DMipmap` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTexture1DMipmap: i32,

    /// Native `maxTexture1DLinear` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTexture1DLinear: i32,

    /// Native `maxTexture2D` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTexture2D: [i32; 2],

    /// Native `maxTexture2DMipmap` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTexture2DMipmap: [i32; 2],

    /// Native `maxTexture2DLinear` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTexture2DLinear: [i32; 3],

    /// Native `maxTexture2DGather` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTexture2DGather: [i32; 2],

    /// Native `maxTexture3D` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTexture3D: [i32; 3],

    /// Native `maxTexture3DAlt` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTexture3DAlt: [i32; 3],

    /// Native `maxTextureCubemap` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTextureCubemap: i32,

    /// Native `maxTexture1DLayered` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTexture1DLayered: [i32; 2],

    /// Native `maxTexture2DLayered` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTexture2DLayered: [i32; 3],

    /// Native `maxTextureCubemapLayered` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxTextureCubemapLayered: [i32; 2],

    /// Native `maxSurface1D` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxSurface1D: i32,

    /// Native `maxSurface2D` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxSurface2D: [i32; 2],

    /// Native `maxSurface3D` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxSurface3D: [i32; 3],

    /// Native `maxSurface1DLayered` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxSurface1DLayered: [i32; 2],

    /// Native `maxSurface2DLayered` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxSurface2DLayered: [i32; 3],

    /// Native `maxSurfaceCubemap` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxSurfaceCubemap: i32,

    /// Native `maxSurfaceCubemapLayered` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxSurfaceCubemapLayered: [i32; 2],

    /// Native `surfaceAlignment` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) surfaceAlignment: usize,

    /// Native `concurrentKernels` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) concurrentKernels: i32,

    /// Native `ECCEnabled` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) ECCEnabled: i32,

    /// Native `pciBusID` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) pciBusID: i32,

    /// Native `pciDeviceID` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) pciDeviceID: i32,

    /// Native `pciDomainID` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) pciDomainID: i32,

    /// Native `tccDriver` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) tccDriver: i32,

    /// Native `asyncEngineCount` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) asyncEngineCount: i32,

    /// Native `unifiedAddressing` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) unifiedAddressing: i32,

    /// Native `memoryClockRate` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) memoryClockRate: i32,

    /// Native `memoryBusWidth` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) memoryBusWidth: i32,

    /// Native `l2CacheSize` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) l2CacheSize: i32,

    /// Native `persistingL2CacheMaxSize` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) persistingL2CacheMaxSize: i32,

    /// Native `maxThreadsPerMultiProcessor` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxThreadsPerMultiProcessor: i32,

    /// Native `streamPrioritiesSupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) streamPrioritiesSupported: i32,

    /// Native `globalL1CacheSupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) globalL1CacheSupported: i32,

    /// Native `localL1CacheSupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) localL1CacheSupported: i32,

    /// Native `sharedMemPerMultiprocessor` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) sharedMemPerMultiprocessor: usize,

    /// Native `regsPerMultiprocessor` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) regsPerMultiprocessor: i32,

    /// Native `managedMemory` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) managedMemory: i32,

    /// Native `isMultiGpuBoard` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) isMultiGpuBoard: i32,

    /// Native `multiGpuBoardGroupID` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) multiGpuBoardGroupID: i32,

    /// Native `hostNativeAtomicSupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) hostNativeAtomicSupported: i32,

    /// Native `singleToDoublePrecisionPerfRatio` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) singleToDoublePrecisionPerfRatio: i32,

    /// Native `pageableMemoryAccess` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) pageableMemoryAccess: i32,

    /// Native `concurrentManagedAccess` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) concurrentManagedAccess: i32,

    /// Native `computePreemptionSupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) computePreemptionSupported: i32,

    /// Native `canUseHostPointerForRegisteredMem` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) canUseHostPointerForRegisteredMem: i32,

    /// Native `cooperativeLaunch` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) cooperativeLaunch: i32,

    /// Native `cooperativeMultiDeviceLaunch` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) cooperativeMultiDeviceLaunch: i32,

    /// Native `sharedMemPerBlockOptin` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) sharedMemPerBlockOptin: usize,

    /// Native `pageableMemoryAccessUsesHostPageTables` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) pageableMemoryAccessUsesHostPageTables: i32,

    /// Native `directManagedMemAccessFromHost` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) directManagedMemAccessFromHost: i32,

    /// Native `maxBlocksPerMultiProcessor` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxBlocksPerMultiProcessor: i32,

    /// Native `accessPolicyMaxWindowSize` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) accessPolicyMaxWindowSize: i32,

    /// Native `reservedSharedMemPerBlock` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) reservedSharedMemPerBlock: usize,

    /// Native `hostRegisterSupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) hostRegisterSupported: i32,

    /// Native `sparseHipArraySupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) sparseHipArraySupported: i32,

    /// Native `hostRegisterReadOnlySupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) hostRegisterReadOnlySupported: i32,

    /// Native `timelineSemaphoreInteropSupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) timelineSemaphoreInteropSupported: i32,

    /// Native `memoryPoolsSupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) memoryPoolsSupported: i32,

    /// Native `gpuDirectRDMASupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) gpuDirectRDMASupported: i32,

    /// Native `gpuDirectRDMAFlushWritesOptions` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) gpuDirectRDMAFlushWritesOptions: u32,

    /// Native `gpuDirectRDMAWritesOrdering` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) gpuDirectRDMAWritesOrdering: i32,

    /// Native `memoryPoolSupportedHandleTypes` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) memoryPoolSupportedHandleTypes: u32,

    /// Native `deferredMappingHipArraySupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) deferredMappingHipArraySupported: i32,

    /// Native `ipcEventSupported` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) ipcEventSupported: i32,

    /// Native `clusterLaunch` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) clusterLaunch: i32,

    /// Native `unifiedFunctionPointers` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) unifiedFunctionPointers: i32,

    /// Native `reserved` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) reserved: [i32; 63],

    /// Native `hipReserved` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) hipReserved: [i32; 32],

    /// Native `gcnArchName` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) gcnArchName: [c_char; 256],

    /// Native `maxSharedMemoryPerMultiProcessor` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) maxSharedMemoryPerMultiProcessor: usize,

    /// Native `clockInstructionRate` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) clockInstructionRate: i32,

    /// Native `arch` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) arch: u32,

    /// Native `hdpMemFlushCntl` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) hdpMemFlushCntl: *mut u32,

    /// Native `hdpRegFlushCntl` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) hdpRegFlushCntl: *mut u32,

    /// Native `cooperativeMultiDeviceUnmatchedFunc` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) cooperativeMultiDeviceUnmatchedFunc: i32,

    /// Native `cooperativeMultiDeviceUnmatchedGridDim` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) cooperativeMultiDeviceUnmatchedGridDim: i32,

    /// Native `cooperativeMultiDeviceUnmatchedBlockDim` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) cooperativeMultiDeviceUnmatchedBlockDim: i32,

    /// Native `cooperativeMultiDeviceUnmatchedSharedMem` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) cooperativeMultiDeviceUnmatchedSharedMem: i32,

    /// Native `isLargeBar` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) isLargeBar: i32,

    /// Native `asicRevision` field from the pinned HIP 7.13 `hipDeviceProp_tR0600`.
    pub(super) asicRevision: i32,
}

/// Native module launch function with pointer-array argument packing.
pub(super) type HipModuleLaunchKernel = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_device_properties_layout() {
        assert_eq!(size_of::<HipDeviceProperties>(), 1472);
        assert_eq!(align_of::<HipDeviceProperties>(), 8);
        assert_eq!(std::mem::offset_of!(HipDeviceProperties, gcnArchName), 1160);
    }
}
