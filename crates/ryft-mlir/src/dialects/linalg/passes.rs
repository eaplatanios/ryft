use std::sync::OnceLock;

use ryft_xla_sys::bindings::mlirRegisterLinalgPasses;

use crate::{GLOBAL_REGISTRATION_MUTEX, mlir_pass};

/// Registers the MLIR `linalg` [`Dialect`](crate::Dialect) passes with the global registry.
pub fn register_linalg_passes() {
    // Use [`OnceLock`] to ensure that [`register_linalg_passes`] is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterLinalgPasses()
    });
}

mlir_pass!(linalg_convert_elementwise_to_linalg_pass, LinalgConvertElementwiseToLinalgPass);
mlir_pass!(linalg_to_affine_loops_pass, LinalgConvertLinalgToAffineLoopsPass);
mlir_pass!(linalg_to_loops_pass, LinalgConvertLinalgToLoopsPass);
mlir_pass!(linalg_to_parallel_loops_pass, LinalgConvertLinalgToParallelLoopsPass);
mlir_pass!(linalg_block_pack_matmul_pass, LinalgLinalgBlockPackMatmul);
mlir_pass!(linalg_elementwise_op_fusion_pass, LinalgLinalgElementwiseOpFusionPass);
mlir_pass!(linalg_fold_into_elementwise_pass, LinalgLinalgFoldIntoElementwisePass);
mlir_pass!(linalg_fold_unit_extent_dims_pass, LinalgLinalgFoldUnitExtentDimsPass);
mlir_pass!(linalg_generalize_named_ops_pass, LinalgLinalgGeneralizeNamedOpsPass);
mlir_pass!(linalg_inline_scalar_operands_pass, LinalgLinalgInlineScalarOperandsPass);
mlir_pass!(linalg_morph_ops_pass, LinalgLinalgMorphOpsPass);
mlir_pass!(linalg_specialize_generic_ops_pass, LinalgLinalgSpecializeGenericOpsPass);
mlir_pass!(linalg_simplify_depthwise_conv_pass, LinalgSimplifyDepthwiseConvPass);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_linalg_passes() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_linalg_passes();
        register_linalg_passes();
        register_linalg_convert_elementwise_to_linalg_pass();
        register_linalg_convert_elementwise_to_linalg_pass();
        register_linalg_to_affine_loops_pass();
        register_linalg_to_affine_loops_pass();
        register_linalg_to_loops_pass();
        register_linalg_to_loops_pass();
        register_linalg_to_parallel_loops_pass();
        register_linalg_to_parallel_loops_pass();
        register_linalg_block_pack_matmul_pass();
        register_linalg_block_pack_matmul_pass();
        register_linalg_elementwise_op_fusion_pass();
        register_linalg_elementwise_op_fusion_pass();
        register_linalg_fold_into_elementwise_pass();
        register_linalg_fold_into_elementwise_pass();
        register_linalg_fold_unit_extent_dims_pass();
        register_linalg_fold_unit_extent_dims_pass();
        register_linalg_generalize_named_ops_pass();
        register_linalg_generalize_named_ops_pass();
        register_linalg_inline_scalar_operands_pass();
        register_linalg_inline_scalar_operands_pass();
        register_linalg_morph_ops_pass();
        register_linalg_morph_ops_pass();
        register_linalg_specialize_generic_ops_pass();
        register_linalg_specialize_generic_ops_pass();
        register_linalg_simplify_depthwise_conv_pass();
        register_linalg_simplify_depthwise_conv_pass();
    }

    #[test]
    fn test_create_linalg_passes() {
        // Verify that pass creation does not crash for the various `linalg` dialect passes.
        let _ = create_linalg_convert_elementwise_to_linalg_pass();
        let _ = create_linalg_to_affine_loops_pass();
        let _ = create_linalg_to_loops_pass();
        let _ = create_linalg_to_parallel_loops_pass();
        let _ = create_linalg_block_pack_matmul_pass();
        let _ = create_linalg_elementwise_op_fusion_pass();
        let _ = create_linalg_fold_into_elementwise_pass();
        let _ = create_linalg_fold_unit_extent_dims_pass();
        let _ = create_linalg_generalize_named_ops_pass();
        let _ = create_linalg_inline_scalar_operands_pass();
        let _ = create_linalg_morph_ops_pass();
        let _ = create_linalg_specialize_generic_ops_pass();
        let _ = create_linalg_simplify_depthwise_conv_pass();
    }
}
