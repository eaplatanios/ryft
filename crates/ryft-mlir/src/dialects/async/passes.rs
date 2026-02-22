use std::sync::OnceLock;

use ryft_xla_sys::bindings::mlirRegisterAsyncPasses;

use crate::{GLOBAL_REGISTRATION_MUTEX, mlir_pass};

/// Registers the MLIR `async` [`Dialect`](crate::Dialect) passes with the global registry.
pub fn register_async_passes() {
    // Use [`OnceLock`] to ensure that [`register_async_passes`] is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterAsyncPasses()
    });
}

mlir_pass!(async_func_to_async_runtime_pass, AsyncAsyncFuncToAsyncRuntimePass);
mlir_pass!(async_parallel_for_pass, AsyncAsyncParallelForPass);
mlir_pass!(async_runtime_policy_based_ref_counting_pass, AsyncAsyncRuntimePolicyBasedRefCountingPass);
mlir_pass!(async_runtime_ref_counting_pass, AsyncAsyncRuntimeRefCountingPass);
mlir_pass!(async_runtime_ref_counting_opt_pass, AsyncAsyncRuntimeRefCountingOptPass);
mlir_pass!(async_to_async_runtime_pass, AsyncAsyncToAsyncRuntimePass);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_async_passes() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_async_passes();
        register_async_passes();
        register_async_func_to_async_runtime_pass();
        register_async_func_to_async_runtime_pass();
        register_async_parallel_for_pass();
        register_async_parallel_for_pass();
        register_async_runtime_policy_based_ref_counting_pass();
        register_async_runtime_policy_based_ref_counting_pass();
        register_async_runtime_ref_counting_pass();
        register_async_runtime_ref_counting_pass();
        register_async_runtime_ref_counting_opt_pass();
        register_async_runtime_ref_counting_opt_pass();
        register_async_to_async_runtime_pass();
        register_async_to_async_runtime_pass();
    }

    #[test]
    fn test_create_async_passes() {
        // Verify that pass creation does not crash for the various `async` dialect passes.
        let _ = create_async_func_to_async_runtime_pass();
        let _ = create_async_parallel_for_pass();
        let _ = create_async_runtime_policy_based_ref_counting_pass();
        let _ = create_async_runtime_ref_counting_pass();
        let _ = create_async_runtime_ref_counting_opt_pass();
        let _ = create_async_to_async_runtime_pass();
    }
}
