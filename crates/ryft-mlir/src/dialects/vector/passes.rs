use std::sync::OnceLock;

use crate::macros::mlir_pass;

mlir_pass!(
    lower_vector_mask_pass,
    LowerVectorMaskPass,
    "Lowers masked vector operations to explicit predication. \
    Refer to the [official MLIR documentation]\
    (https://mlir.llvm.org/docs/Passes/#-lower-vector-mask).",
);

mlir_pass!(
    lower_vector_multi_reduction_pass,
    LowerVectorMultiReduction,
    "Lowers multidimensional vector reductions to simpler vector operations. \
    Refer to the [official MLIR documentation]\
    (https://mlir.llvm.org/docs/Passes/#-lower-vector-multi-reduction).",
);

mlir_pass!(
    lower_vector_to_from_elements_to_shuffle_tree_pass,
    LowerVectorToFromElementsToShuffleTree,
    "Lowers vector element assembly and decomposition using shuffle trees. \
    Refer to the [official MLIR documentation]\
    (https://mlir.llvm.org/docs/Passes/#-lower-vector-to-from-elements-to-shuffle-tree).",
);

#[cfg(test)]
mod tests {
    use crate::Context;

    use super::*;

    #[test]
    fn test_create_lower_vector_mask_pass() {
        let context = Context::new();
        let mut pass_manager = context.pass_manager().unwrap();
        let pass = create_lower_vector_mask_pass().unwrap();
        pass_manager.add_pass(pass);
    }

    #[test]
    fn test_register_lower_vector_mask_pass() {
        // Registration is intentionally idempotent.
        register_lower_vector_mask_pass();
        register_lower_vector_mask_pass();
    }

    #[test]
    fn test_create_lower_vector_multi_reduction_pass() {
        let context = Context::new();
        let mut pass_manager = context.pass_manager().unwrap();
        let pass = create_lower_vector_multi_reduction_pass().unwrap();
        pass_manager.add_pass(pass);
    }

    #[test]
    fn test_register_lower_vector_multi_reduction_pass() {
        // Registration is intentionally idempotent.
        register_lower_vector_multi_reduction_pass();
        register_lower_vector_multi_reduction_pass();
    }

    #[test]
    fn test_create_lower_vector_to_from_elements_to_shuffle_tree_pass() {
        let context = Context::new();
        let mut pass_manager = context.pass_manager().unwrap();
        let pass = create_lower_vector_to_from_elements_to_shuffle_tree_pass().unwrap();
        pass_manager.add_pass(pass);
    }

    #[test]
    fn test_register_lower_vector_to_from_elements_to_shuffle_tree_pass() {
        // Registration is intentionally idempotent.
        register_lower_vector_to_from_elements_to_shuffle_tree_pass();
        register_lower_vector_to_from_elements_to_shuffle_tree_pass();
    }
}
