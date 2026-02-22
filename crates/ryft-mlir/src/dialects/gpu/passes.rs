use std::sync::OnceLock;

use ryft_xla_sys::bindings::mlirRegisterGPUPasses;

use crate::{GLOBAL_REGISTRATION_MUTEX, mlir_pass};

/// Registers the MLIR `gpu` [`Dialect`](crate::Dialect) passes with the global registry.
pub fn register_gpu_passes() {
    // Use [`OnceLock`] to ensure that [`register_gpu_passes`] is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterGPUPasses()
    });
}

mlir_pass!(gpu_async_region_pass, GPUGpuAsyncRegionPass);
mlir_pass!(gpu_decompose_memrefs_pass, GPUGpuDecomposeMemrefsPass);
mlir_pass!(gpu_eliminate_barriers_pass, GPUGpuEliminateBarriers);
mlir_pass!(gpu_kernel_outlining_pass, GPUGpuKernelOutliningPass);
mlir_pass!(gpu_launch_sink_index_computations_pass, GPUGpuLaunchSinkIndexComputationsPass);
mlir_pass!(gpu_map_parallel_loops_pass, GPUGpuMapParallelLoopsPass);
mlir_pass!(gpu_module_to_binary_pass, GPUGpuModuleToBinaryPass);
mlir_pass!(gpu_nvvm_attach_target_pass, GPUGpuNVVMAttachTarget);
mlir_pass!(gpu_rocdl_attach_target_pass, GPUGpuROCDLAttachTarget);
mlir_pass!(gpu_spirv_attach_target_pass, GPUGpuSPIRVAttachTarget);
mlir_pass!(gpu_xevm_attach_target_pass, GPUGpuXeVMAttachTarget);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_gpu_passes() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_gpu_passes();
        register_gpu_passes();
        register_gpu_async_region_pass();
        register_gpu_async_region_pass();
        register_gpu_decompose_memrefs_pass();
        register_gpu_decompose_memrefs_pass();
        register_gpu_eliminate_barriers_pass();
        register_gpu_eliminate_barriers_pass();
        register_gpu_kernel_outlining_pass();
        register_gpu_kernel_outlining_pass();
        register_gpu_launch_sink_index_computations_pass();
        register_gpu_launch_sink_index_computations_pass();
        register_gpu_map_parallel_loops_pass();
        register_gpu_map_parallel_loops_pass();
        register_gpu_module_to_binary_pass();
        register_gpu_module_to_binary_pass();
        register_gpu_nvvm_attach_target_pass();
        register_gpu_nvvm_attach_target_pass();
        register_gpu_rocdl_attach_target_pass();
        register_gpu_rocdl_attach_target_pass();
        register_gpu_spirv_attach_target_pass();
        register_gpu_spirv_attach_target_pass();
        register_gpu_xevm_attach_target_pass();
        register_gpu_xevm_attach_target_pass();
    }

    #[test]
    fn test_create_gpu_passes() {
        // Verify that pass creation does not crash for the various `gpu` dialect passes.
        let _ = create_gpu_async_region_pass();
        let _ = create_gpu_decompose_memrefs_pass();
        let _ = create_gpu_eliminate_barriers_pass();
        let _ = create_gpu_kernel_outlining_pass();
        let _ = create_gpu_launch_sink_index_computations_pass();
        let _ = create_gpu_map_parallel_loops_pass();
        let _ = create_gpu_module_to_binary_pass();
        let _ = create_gpu_nvvm_attach_target_pass();
        let _ = create_gpu_rocdl_attach_target_pass();
        let _ = create_gpu_spirv_attach_target_pass();
        let _ = create_gpu_xevm_attach_target_pass();
    }
}
