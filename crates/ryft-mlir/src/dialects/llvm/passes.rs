use std::sync::OnceLock;

use crate::mlir_pass;

mlir_pass!(di_scope_for_llvm_func_op_pass, DIScopeForLLVMFuncOpPass);

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
}
