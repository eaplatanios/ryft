use std::sync::OnceLock;

use ryft_xla_sys::bindings::mlirRegisterAllMhloPasses;

use crate::GLOBAL_REGISTRATION_MUTEX;

/// Registers all compiler passes of the MHLO [`Dialect`](crate::Dialect).
pub fn register_mhlo_passes() {
    // Use [`OnceLock`] to ensure that [`register_mhlo_passes`] is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterAllMhloPasses()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_mhlo_passes() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_mhlo_passes();
        register_mhlo_passes();
    }
}
