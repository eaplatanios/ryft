use std::sync::OnceLock;

use ryft_xla_sys::bindings::mlirRegisterAllStablehloPasses;

use crate::GLOBAL_REGISTRATION_MUTEX;

/// Registers all compiler passes of the StableHLO [`Dialect`](crate::Dialect).
pub fn register_stable_hlo_passes() {
    // Use [`OnceLock`] to ensure that [`register_stable_hlo_passes`] is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterAllStablehloPasses()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_stable_hlo_passes() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_stable_hlo_passes();
        register_stable_hlo_passes();
    }
}
