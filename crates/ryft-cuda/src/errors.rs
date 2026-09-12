use std::backtrace::Backtrace;
use std::sync::Mutex;

use thiserror::Error;

// TODO(eaplatanios): Review this.

/// Error produced while validating, loading, or launching a CUDA kernel artifact.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The caller supplied an invalid artifact or launch request.
    #[error("{message}")]
    InvalidArgument { message: String, backtrace: String },

    /// The requested CUDA service, resource, or launcher operation is unavailable or unsupported.
    #[error("{message}")]
    Unavailable { message: String, backtrace: String },

    /// An external runtime fact required to construct a launch was unavailable.
    #[error("{message}")]
    Integration { message: String, backtrace: String },

    /// An invariant guaranteed by a successful CUDA Driver API call was violated.
    #[error("{message}")]
    Internal { message: String, backtrace: String },

    /// A CUDA Driver API function returned an error.
    #[error("CUDA driver function `{operation}` failed with `{name}` ({code}): {message}")]
    Driver { operation: String, code: i32, name: String, message: String, backtrace: String },
}

/// Cleanup failures that could not be returned because a resource was being dropped.
static CLEANUP_ERRORS: Mutex<Vec<Error>> = Mutex::new(Vec::new());

impl Error {
    /// Takes all unobserved destructor cleanup failures recorded by this crate in the current process.
    ///
    /// Prefer explicit launcher cleanup, which returns failures directly. Destructors record otherwise unreportable
    /// failures here without writing to standard error or panicking. Concurrent destructors may record new failures
    /// after this snapshot; each recorded failure is returned to exactly one caller.
    pub fn take_cleanup_errors() -> Vec<Self> {
        let mut errors = CLEANUP_ERRORS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        std::mem::take(&mut *errors)
    }

    /// Records a destructor failure for later observation without performing fallible output operations.
    pub(crate) fn record_cleanup_error(self) {
        CLEANUP_ERRORS.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(self);
    }

    /// Creates an error for an invalid artifact or launch request.
    pub(super) fn invalid_argument<M: Into<String>>(message: M) -> Self {
        Self::InvalidArgument { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates an error for unavailable CUDA resources or launcher state.
    pub(super) fn unavailable<M: Into<String>>(message: M) -> Self {
        Self::Unavailable { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates an integration error for a framework-owned CUDA resource.
    pub fn integration<M: Into<String>>(message: M) -> Self {
        Self::Integration { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates an error for a violated driver invariant.
    pub(super) fn internal<M: Into<String>>(message: M) -> Self {
        Self::Internal { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a driver error preserving the operation, numeric code, name, and diagnostic.
    pub(super) fn driver<O: Into<String>, N: Into<String>, M: Into<String>>(
        operation: O,
        code: i32,
        name: N,
        message: M,
    ) -> Self {
        Self::Driver {
            operation: operation.into(),
            code,
            name: name.into(),
            message: message.into(),
            backtrace: Backtrace::capture().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_error_take_cleanup_errors() {
        let _guard = crate::tests::CLEANUP_ERROR_TEST_LOCK.lock().unwrap();
        // Other modules may record destructor failures concurrently, so identify this test's unique diagnostic.
        Error::internal("cleanup observation test").record_cleanup_error();
        let errors = Error::take_cleanup_errors();
        assert_eq!(errors.iter().filter(|error| error.to_string() == "cleanup observation test").count(), 1);
        assert!(Error::take_cleanup_errors().iter().all(|error| error.to_string() != "cleanup observation test"));
    }

    #[test]
    fn test_error_invalid_argument() {
        let error = Error::invalid_argument("test diagnostic");
        assert!(matches!(error, Error::InvalidArgument { ref message, .. } if message == "test diagnostic"));
        assert_eq!(error.to_string(), "test diagnostic");
    }

    #[test]
    fn test_error_unavailable() {
        let error = Error::unavailable("test diagnostic");
        assert!(matches!(error, Error::Unavailable { ref message, .. } if message == "test diagnostic"));
        assert_eq!(error.to_string(), "test diagnostic");
    }

    #[test]
    fn test_error_integration() {
        let error = Error::integration("test diagnostic");
        assert!(matches!(error, Error::Integration { ref message, .. } if message == "test diagnostic"));
        assert_eq!(error.to_string(), "test diagnostic");
    }

    #[test]
    fn test_error_internal() {
        let error = Error::internal("test diagnostic");
        assert!(matches!(error, Error::Internal { ref message, .. } if message == "test diagnostic"));
        assert_eq!(error.to_string(), "test diagnostic");
    }

    #[test]
    fn test_error_driver() {
        let error = Error::driver("cuInit", 999, "CUDA_ERROR_UNKNOWN", "test diagnostic");
        assert!(matches!(
            error,
            Error::Driver { operation, code: 999, name, message, .. }
                if operation == "cuInit" && name == "CUDA_ERROR_UNKNOWN" && message == "test diagnostic",
        ));
    }

    #[test]
    fn test_error_display_and_debug() {
        let error = Error::Driver {
            operation: "cuModuleLoadDataEx".to_string(),
            code: 209,
            name: "CUDA_ERROR_NO_BINARY_FOR_GPU".to_string(),
            message: "no kernel image is available for execution on the device".to_string(),
            backtrace: "test backtrace".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "CUDA driver function `cuModuleLoadDataEx` failed with `CUDA_ERROR_NO_BINARY_FOR_GPU` (209): \
             no kernel image is available for execution on the device",
        );
        assert_eq!(
            format!("{error:?}"),
            "Driver { operation: \"cuModuleLoadDataEx\", code: 209, name: \"CUDA_ERROR_NO_BINARY_FOR_GPU\", \
             message: \"no kernel image is available for execution on the device\", backtrace: \"test backtrace\" }",
        );
    }
}
