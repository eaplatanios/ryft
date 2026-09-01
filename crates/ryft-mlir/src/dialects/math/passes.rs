use std::sync::OnceLock;

use crate::mlir_pass;

mlir_pass!(math_expand_ops_pass, MathExpandOpsPass);
mlir_pass!(math_extend_to_supported_types_pass, MathExtendToSupportedTypes);
mlir_pass!(math_sincos_fusion_pass, MathSincosFusionPass);
mlir_pass!(math_uplift_to_fma_pass, MathUpliftToFMA);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_math_passes() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_math_expand_ops_pass();
        register_math_expand_ops_pass();
        register_math_extend_to_supported_types_pass();
        register_math_extend_to_supported_types_pass();
        register_math_sincos_fusion_pass();
        register_math_sincos_fusion_pass();
        register_math_uplift_to_fma_pass();
        register_math_uplift_to_fma_pass();
    }

    #[test]
    fn test_create_math_passes() {
        let _ = create_math_expand_ops_pass().unwrap();
        let _ = create_math_extend_to_supported_types_pass().unwrap();
        let _ = create_math_sincos_fusion_pass().unwrap();
        let _ = create_math_uplift_to_fma_pass().unwrap();
    }
}
