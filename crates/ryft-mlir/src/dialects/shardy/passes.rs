use std::sync::OnceLock;

use ryft_xla_sys::bindings::{mlirRegisterAllSdyPassesAndPipelines, mlirRegisterAllXlaSdyPassesAndPipelines};

use crate::GLOBAL_REGISTRATION_MUTEX;

/// Registers all compiler passes and pipelines of the Shardy [`Dialect`](crate::Dialect).
pub fn register_shardy_passes_and_pipelines() {
    // Use [`OnceLock`] to ensure that [`register_shardy_passes_and_pipelines`] is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterAllSdyPassesAndPipelines()
    });
}

/// Registers all compiler passes and pipelines of the XLA Shardy [`Dialect`](crate::Dialect).
pub fn register_xla_shardy_passes_and_pipelines() {
    // Use [`OnceLock`] to ensure that [`register_xla_shardy_passes_and_pipelines`] is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterAllXlaSdyPassesAndPipelines()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_shardy_passes_and_pipelines() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_shardy_passes_and_pipelines();
        register_shardy_passes_and_pipelines();
        register_xla_shardy_passes_and_pipelines();
        register_xla_shardy_passes_and_pipelines();
    }
}
