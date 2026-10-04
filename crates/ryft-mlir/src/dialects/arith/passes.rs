use std::sync::OnceLock;

use crate::mlir_pass;

mlir_pass!(arith_expand_ops_pass, ArithExpandOpsPass);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arith_expand_ops_pass() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_arith_expand_ops_pass();
        register_arith_expand_ops_pass();
        let _ = create_arith_expand_ops_pass().unwrap();
    }
}
