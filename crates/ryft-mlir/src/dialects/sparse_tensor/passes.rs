use std::sync::OnceLock;

use ryft_xla_sys::bindings::mlirRegisterSparseTensorPasses;

use crate::{GLOBAL_REGISTRATION_MUTEX, mlir_pass};

/// Registers the MLIR `sparse_tensor` [`Dialect`](crate::Dialect) passes with the global registry.
pub fn register_sparse_tensor_passes() {
    // Use [`OnceLock`] to ensure that [`register_sparse_tensor_passes`] is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterSparseTensorPasses()
    });
}

mlir_pass!(sparse_tensor_lower_foreach_to_scf_pass, SparseTensorLowerForeachToSCF);
mlir_pass!(sparse_tensor_lower_sparse_iteration_to_scf_pass, SparseTensorLowerSparseIterationToSCF);
mlir_pass!(sparse_tensor_lower_sparse_ops_to_foreach_pass, SparseTensorLowerSparseOpsToForeach);
mlir_pass!(sparse_tensor_pre_sparsification_rewrite_pass, SparseTensorPreSparsificationRewrite);
mlir_pass!(sparse_tensor_sparse_assembler_pass, SparseTensorSparseAssembler);
mlir_pass!(sparse_tensor_sparse_buffer_rewrite_pass, SparseTensorSparseBufferRewrite);
mlir_pass!(sparse_tensor_sparse_gpu_codegen_pass, SparseTensorSparseGPUCodegen);
mlir_pass!(sparse_tensor_sparse_reinterpret_map_pass, SparseTensorSparseReinterpretMap);
mlir_pass!(sparse_tensor_sparse_collapse_pass, SparseTensorSparseSpaceCollapse);
mlir_pass!(sparse_tensor_sparse_tensor_codegen_pass, SparseTensorSparseTensorCodegen);
mlir_pass!(sparse_tensor_sparse_tensor_conversion_pass, SparseTensorSparseTensorConversionPass);
mlir_pass!(sparse_tensor_sparse_vectorization_pass, SparseTensorSparseVectorization);
mlir_pass!(sparse_tensor_sparsification_and_bufferization_pass, SparseTensorSparsificationAndBufferization);
mlir_pass!(sparse_tensor_sparsification_pass, SparseTensorSparsificationPass);
mlir_pass!(sparse_tensor_stage_sparse_operations_pass, SparseTensorStageSparseOperations);
mlir_pass!(sparse_tensor_storage_specifier_to_llvm_pass, SparseTensorStorageSpecifierToLLVM);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_sparse_tensor_passes() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_sparse_tensor_passes();
        register_sparse_tensor_passes();
        register_sparse_tensor_lower_foreach_to_scf_pass();
        register_sparse_tensor_lower_foreach_to_scf_pass();
        register_sparse_tensor_lower_sparse_iteration_to_scf_pass();
        register_sparse_tensor_lower_sparse_iteration_to_scf_pass();
        register_sparse_tensor_lower_sparse_ops_to_foreach_pass();
        register_sparse_tensor_lower_sparse_ops_to_foreach_pass();
        register_sparse_tensor_pre_sparsification_rewrite_pass();
        register_sparse_tensor_pre_sparsification_rewrite_pass();
        register_sparse_tensor_sparse_assembler_pass();
        register_sparse_tensor_sparse_assembler_pass();
        register_sparse_tensor_sparse_buffer_rewrite_pass();
        register_sparse_tensor_sparse_buffer_rewrite_pass();
        register_sparse_tensor_sparse_gpu_codegen_pass();
        register_sparse_tensor_sparse_gpu_codegen_pass();
        register_sparse_tensor_sparse_reinterpret_map_pass();
        register_sparse_tensor_sparse_reinterpret_map_pass();
        register_sparse_tensor_sparse_collapse_pass();
        register_sparse_tensor_sparse_collapse_pass();
        register_sparse_tensor_sparse_tensor_codegen_pass();
        register_sparse_tensor_sparse_tensor_codegen_pass();
        register_sparse_tensor_sparse_tensor_conversion_pass();
        register_sparse_tensor_sparse_tensor_conversion_pass();
        register_sparse_tensor_sparse_vectorization_pass();
        register_sparse_tensor_sparse_vectorization_pass();
        register_sparse_tensor_sparsification_and_bufferization_pass();
        register_sparse_tensor_sparsification_and_bufferization_pass();
        register_sparse_tensor_sparsification_pass();
        register_sparse_tensor_sparsification_pass();
        register_sparse_tensor_stage_sparse_operations_pass();
        register_sparse_tensor_stage_sparse_operations_pass();
        register_sparse_tensor_storage_specifier_to_llvm_pass();
        register_sparse_tensor_storage_specifier_to_llvm_pass();
    }

    #[test]
    fn test_create_sparse_tensor_passes() {
        // Verify that pass creation does not crash for the various `sparse_tensor` dialect passes.
        let _ = create_sparse_tensor_lower_foreach_to_scf_pass();
        let _ = create_sparse_tensor_lower_sparse_iteration_to_scf_pass();
        let _ = create_sparse_tensor_lower_sparse_ops_to_foreach_pass();
        let _ = create_sparse_tensor_pre_sparsification_rewrite_pass();
        let _ = create_sparse_tensor_sparse_assembler_pass();
        let _ = create_sparse_tensor_sparse_buffer_rewrite_pass();
        let _ = create_sparse_tensor_sparse_gpu_codegen_pass();
        let _ = create_sparse_tensor_sparse_reinterpret_map_pass();
        let _ = create_sparse_tensor_sparse_collapse_pass();
        let _ = create_sparse_tensor_sparse_tensor_codegen_pass();
        let _ = create_sparse_tensor_sparse_tensor_conversion_pass();
        let _ = create_sparse_tensor_sparse_vectorization_pass();
        let _ = create_sparse_tensor_sparsification_and_bufferization_pass();
        let _ = create_sparse_tensor_sparsification_pass();
        let _ = create_sparse_tensor_stage_sparse_operations_pass();
        let _ = create_sparse_tensor_storage_specifier_to_llvm_pass();
    }
}
