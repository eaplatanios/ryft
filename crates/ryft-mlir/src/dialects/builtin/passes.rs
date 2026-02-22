use std::sync::OnceLock;

use ryft_xla_sys::bindings::{mlirRegisterConversionPasses, mlirRegisterTransformsPasses};

use crate::{GLOBAL_REGISTRATION_MUTEX, mlir_pass};

/// Registers the MLIR conversion passes with the global registry.
pub fn register_conversions_passes() {
    // Use [`OnceLock`] to ensure that [`register_conversions_passes`] is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterConversionPasses()
    });
}

mlir_pass!(conversion_arith_to_amdgpu_pass, ConversionArithToAMDGPUConversionPass);
mlir_pass!(conversion_arith_to_arm_sme_pass, ConversionArithToArmSMEConversionPass);
mlir_pass!(conversion_arith_to_llvm_pass, ConversionArithToLLVMConversionPass);
mlir_pass!(conversion_amdgpu_to_rocdl_pass, ConversionConvertAMDGPUToROCDLPass);
mlir_pass!(conversion_affine_for_to_gpu_pass, ConversionConvertAffineForToGPUPass);
mlir_pass!(conversion_arith_to_emit_c_pass, ConversionConvertArithToEmitC);
mlir_pass!(conversion_arith_to_spirv_pass, ConversionConvertArithToSPIRVPass);
mlir_pass!(conversion_arm_neon_2d_to_intr_pass, ConversionConvertArmNeon2dToIntrPass);
mlir_pass!(conversion_arm_sme_to_llvm_pass, ConversionConvertArmSMEToLLVM);
mlir_pass!(conversion_arm_sme_to_scf_pass, ConversionConvertArmSMEToSCFPass);
mlir_pass!(conversion_async_to_llvm_pass, ConversionConvertAsyncToLLVMPass);
mlir_pass!(conversion_bufferization_to_mem_ref_pass, ConversionConvertBufferizationToMemRefPass);
mlir_pass!(conversion_complex_to_llvm_pass, ConversionConvertComplexToLLVMPass);
mlir_pass!(conversion_complex_to_libm_pass, ConversionConvertComplexToLibm);
mlir_pass!(conversion_complex_to_rocdl_library_calls_pass, ConversionConvertComplexToROCDLLibraryCalls);
mlir_pass!(conversion_complex_to_spirv_pass, ConversionConvertComplexToSPIRVPass);
mlir_pass!(conversion_complex_to_standard_pass, ConversionConvertComplexToStandardPass);
mlir_pass!(conversion_control_flow_to_llvm_pass, ConversionConvertControlFlowToLLVMPass);
mlir_pass!(conversion_control_flow_to_spirv_pass, ConversionConvertControlFlowToSPIRVPass);
mlir_pass!(conversion_func_to_emit_c_pass, ConversionConvertFuncToEmitC);
mlir_pass!(conversion_func_to_llvm_pass, ConversionConvertFuncToLLVMPass);
mlir_pass!(conversion_func_to_spirv_pass, ConversionConvertFuncToSPIRVPass);
mlir_pass!(conversion_gpu_to_spirv_pass, ConversionConvertGPUToSPIRV);
mlir_pass!(conversion_gpu_ops_to_llvm_spv_ops_pass, ConversionConvertGpuOpsToLLVMSPVOps);
mlir_pass!(conversion_gpu_ops_to_nvvm_ops_pass, ConversionConvertGpuOpsToNVVMOps);
mlir_pass!(conversion_gpu_ops_to_rocdl_ops_pass, ConversionConvertGpuOpsToROCDLOps);
mlir_pass!(conversion_index_to_llvm_pass, ConversionConvertIndexToLLVMPass);
mlir_pass!(conversion_index_to_spirv_pass, ConversionConvertIndexToSPIRVPass);
mlir_pass!(conversion_linalg_to_standard_pass, ConversionConvertLinalgToStandardPass);
mlir_pass!(conversion_math_to_ap_float_pass, ConversionMathToAPFloatConversionPass);
mlir_pass!(conversion_math_to_emit_c_pass, ConversionConvertMathToEmitC);
mlir_pass!(conversion_math_to_funcs_pass, ConversionConvertMathToFuncs);
mlir_pass!(conversion_math_to_llvm_pass, ConversionConvertMathToLLVMPass);
mlir_pass!(conversion_math_to_libm_pass, ConversionConvertMathToLibmPass);
mlir_pass!(conversion_math_to_rocdl_pass, ConversionConvertMathToROCDL);
mlir_pass!(conversion_math_to_spirv_pass, ConversionConvertMathToSPIRVPass);
mlir_pass!(conversion_math_to_xevm_pass, ConversionConvertMathToXeVM);
mlir_pass!(conversion_mem_ref_to_emit_c_pass, ConversionConvertMemRefToEmitC);
mlir_pass!(conversion_mem_ref_to_spirv_pass, ConversionConvertMemRefToSPIRVPass);
mlir_pass!(conversion_nvgpu_to_nvvm_pass, ConversionConvertNVGPUToNVVMPass);
mlir_pass!(conversion_nvvm_to_llvm_pass, ConversionConvertNVVMToLLVMPass);
mlir_pass!(conversion_open_acc_to_scf_pass, ConversionConvertOpenACCToSCFPass);
mlir_pass!(conversion_open_mp_to_llvm_pass, ConversionConvertOpenMPToLLVMPass);
mlir_pass!(conversion_pdl_to_pdl_interp_pass, ConversionConvertPDLToPDLInterpPass);
mlir_pass!(conversion_parallel_loop_to_gpu_pass, ConversionConvertParallelLoopToGpuPass);
mlir_pass!(conversion_scf_to_open_mp_pass, ConversionConvertSCFToOpenMPPass);
mlir_pass!(conversion_spirv_to_llvm_pass, ConversionConvertSPIRVToLLVMPass);
mlir_pass!(conversion_shape_constraints_pass, ConversionConvertShapeConstraintsPass);
mlir_pass!(conversion_shape_to_standard_pass, ConversionConvertShapeToStandardPass);
mlir_pass!(conversion_shard_to_mpi_pass, ConversionConvertShardToMPIPass);
mlir_pass!(conversion_tensor_to_linalg_pass, ConversionConvertTensorToLinalgPass);
mlir_pass!(conversion_tensor_to_spirv_pass, ConversionConvertTensorToSPIRVPass);
mlir_pass!(conversion_to_emit_c_pass, ConversionConvertToEmitC);
mlir_pass!(conversion_to_llvm_pass, ConversionConvertToLLVMPass);
mlir_pass!(conversion_vector_to_amx_pass, ConversionConvertVectorToAMX);
mlir_pass!(conversion_vector_to_arm_sme_pass, ConversionConvertVectorToArmSMEPass);
mlir_pass!(conversion_vector_to_gpu_pass, ConversionConvertVectorToGPU);
mlir_pass!(conversion_vector_to_llvm_pass, ConversionConvertVectorToLLVMPass);
mlir_pass!(conversion_vector_to_scf_pass, ConversionConvertVectorToSCF);
mlir_pass!(conversion_vector_to_spirv_pass, ConversionConvertVectorToSPIRVPass);
mlir_pass!(conversion_vector_to_xe_gpu_pass, ConversionConvertVectorToXeGPU);
mlir_pass!(conversion_xe_gpu_to_xevm_pass, ConversionConvertXeGPUToXeVMPass);
mlir_pass!(conversion_xevm_to_llvm_pass, ConversionConvertXeVMToLLVMPass);
mlir_pass!(conversion_finalize_mem_ref_to_llvm_pass, ConversionFinalizeMemRefToLLVMConversionPass);
mlir_pass!(conversion_gpu_to_llvm_pass, ConversionGpuToLLVMConversionPass);
mlir_pass!(conversion_lift_control_flow_to_scf_pass, ConversionLiftControlFlowToSCFPass);
mlir_pass!(conversion_lower_affine_pass, ConversionLowerAffinePass);
mlir_pass!(conversion_lower_host_code_to_llvm_pass, ConversionLowerHostCodeToLLVMPass);
mlir_pass!(conversion_map_mem_ref_storage_class_pass, ConversionMapMemRefStorageClass);
mlir_pass!(conversion_reconcile_unrealized_casts_pass, ConversionReconcileUnrealizedCastsPass);
mlir_pass!(conversion_scf_to_control_flow_pass, ConversionSCFToControlFlowPass);
mlir_pass!(conversion_scf_to_emit_c_pass, ConversionSCFToEmitC);
mlir_pass!(conversion_scf_to_spirv_pass, ConversionSCFToSPIRV);
mlir_pass!(conversion_set_llvm_module_data_layout_pass, ConversionSetLLVMModuleDataLayoutPass);
mlir_pass!(conversion_tosa_to_arith_pass, ConversionTosaToArithPass);
mlir_pass!(conversion_tosa_to_linalg_pass, ConversionTosaToLinalg);
mlir_pass!(conversion_tosa_to_linalg_named_pass, ConversionTosaToLinalgNamed);
mlir_pass!(conversion_tosa_to_ml_program_pass, ConversionTosaToMLProgram);
mlir_pass!(conversion_tosa_to_scf_pass, ConversionTosaToSCFPass);
mlir_pass!(conversion_tosa_to_tensor_pass, ConversionTosaToTensorPass);
mlir_pass!(conversion_ub_to_llvm_pass, ConversionUBToLLVMConversionPass);
mlir_pass!(conversion_ub_to_spirv_pass, ConversionUBToSPIRVConversionPass);

/// Registers the MLIR transformation passes with the global registry.
pub fn register_transforms_passes() {
    // Use [`OnceLock`] to ensure that [`register_transforms_passes`] is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterTransformsPasses()
    });
}

mlir_pass!(transforms_bubble_down_memory_space_casts_pass, TransformsBubbleDownMemorySpaceCasts);
mlir_pass!(transforms_cse_pass, TransformsCSE);
mlir_pass!(transforms_canonicalizer_pass, TransformsCanonicalizer);
mlir_pass!(transforms_composite_fixed_point_pass, TransformsCompositeFixedPointPass);
mlir_pass!(transforms_control_flow_sink_pass, TransformsControlFlowSink);
mlir_pass!(transforms_generate_runtime_verification_pass, TransformsGenerateRuntimeVerification);
mlir_pass!(transforms_inliner_pass, TransformsInliner);
mlir_pass!(transforms_location_snapshot_pass, TransformsLocationSnapshot);
mlir_pass!(transforms_loop_invariant_code_motion_pass, TransformsLoopInvariantCodeMotion);
mlir_pass!(transforms_loop_invariant_subset_hoisting_pass, TransformsLoopInvariantSubsetHoisting);
mlir_pass!(transforms_mem2reg_pass, TransformsMem2Reg);
mlir_pass!(transforms_print_ir_pass, TransformsPrintIRPass);
mlir_pass!(transforms_print_op_stats_pass, TransformsPrintOpStats);
mlir_pass!(transforms_remove_dead_values_pass, TransformsRemoveDeadValues);
mlir_pass!(transforms_sccp_pass, TransformsSCCP);
mlir_pass!(transforms_sroa_pass, TransformsSROA);
mlir_pass!(transforms_strip_debug_info_pass, TransformsStripDebugInfo);
mlir_pass!(transforms_symbol_dce_pass, TransformsSymbolDCE);
mlir_pass!(transforms_symbol_privatize_pass, TransformsSymbolPrivatize);
mlir_pass!(transforms_topological_sort_pass, TransformsTopologicalSort);
mlir_pass!(transforms_view_op_graph_pass, TransformsViewOpGraph);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_conversions_passes() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_conversions_passes();
        register_conversions_passes();
    }

    #[test]
    fn test_register_transforms_passes() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_transforms_passes();
        register_transforms_passes();
    }
}
