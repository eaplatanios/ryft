use std::sync::OnceLock;

use crate::mlir_pass;

mlir_pass!(di_scope_for_llvm_func_op_pass, DIScopeForLLVMFuncOpPass);
mlir_pass!(
    simplify_llvm_control_flow_pass,
    LLVMSimplifyCFGPass,
    "Merges blocks and canonicalizes branches in LLVM functions that permit optimization.",
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_di_scope_for_llvm_func_op_pass() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_di_scope_for_llvm_func_op_pass();
        register_di_scope_for_llvm_func_op_pass();
        let _ = create_di_scope_for_llvm_func_op_pass().unwrap();
    }

    #[test]
    fn test_simplify_llvm_control_flow_pass() {
        register_simplify_llvm_control_flow_pass();
        let _ = create_simplify_llvm_control_flow_pass().unwrap();
    }
}
