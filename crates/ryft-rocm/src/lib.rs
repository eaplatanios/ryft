//! Concrete HIP 7.13 HSACO loading and asynchronous kernel submission.
//!
//! The caller owns device allocations, streams and execution completion. The launcher retains loaded modules and
//! their primary contexts until synchronized eviction or shutdown. It creates neither streams nor outer buffers.
//! Native execution is available only on 64-bit Linux and has not yet been qualified on AMD hardware.

use std::sync::Mutex;

use thiserror::Error;

/// Validation, availability, or native execution failure produced by this crate.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The artifact or launch request violates the supported contract.
    #[error("{message}")]
    InvalidArgument { message: String },

    /// The required platform, runtime version or entry point is unavailable.
    #[error("{message}")]
    Unavailable { message: String },

    /// A HIP function returned a nonzero error code.
    #[error("hip function `{operation}` failed with error code `{code}`")]
    Driver { operation: &'static str, code: i32 },

    /// A successful native call returned an invalid handle or resource fact.
    #[error("{message}")]
    Internal { message: String },
}

mod artifacts;
mod drivers;
mod ffi;
mod launchers;
mod launches;

pub use artifacts::{RocmKernelArtifact, RocmKernelLaunchDimensions};
pub use launchers::RocmKernelLauncher;
pub use launches::{RocmDevicePointer, RocmKernelLaunch, RocmStream};

/// Destructor failures awaiting explicit observation by the application.
static CLEANUP_ERRORS: Mutex<Vec<Error>> = Mutex::new(Vec::new());

impl Error {
    /// Creates an invalid request diagnostic.
    pub(crate) fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument { message: message.into() }
    }

    /// Creates an unavailable runtime diagnostic.
    pub(crate) fn unavailable(message: impl Into<String>) -> Self {
        Self::Unavailable { message: message.into() }
    }

    /// Creates a violated native invariant diagnostic.
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::Internal { message: message.into() }
    }

    /// Takes cleanup failures recorded by destructors since the previous call.
    pub fn take_cleanup_errors() -> Vec<Self> {
        std::mem::take(&mut *CLEANUP_ERRORS.lock().unwrap_or_else(|error| error.into_inner()))
    }

    /// Records a fallible destructor operation without discarding its error.
    pub(crate) fn record_cleanup(self) {
        CLEANUP_ERRORS.lock().unwrap_or_else(|error| error.into_inner()).push(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_take_cleanup_errors() {
        let error = Error::Driver { operation: "hipModuleUnload", code: 400 };
        error.clone().record_cleanup();
        assert_eq!(Error::take_cleanup_errors(), vec![error]);
        assert!(Error::take_cleanup_errors().is_empty());
    }
}
