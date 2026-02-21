use std::backtrace::Backtrace;

use thiserror::Error;

use crate::{Api, invoke_pjrt_api_error_fn, invoke_pjrt_api_void_fn, str_from_c_api};

/// Represents errors that can occur when interacting with the PJRT C API. The error types are based on the
/// [Abseil status codes](https://abseil.io/docs/cpp/guides/status-codes) which PJRT uses internally.
///
/// Each variant includes a `backtrace` field that captures the call stack at the point where the error was created,
/// which is useful for debugging. Note that it is represented as a [`String`] and not as a [`Backtrace`] because using
/// the latter is only currently supported in unstable Rust.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Error {
    #[error("error while loading a PJRT plugin from '{path}': {error}")]
    PluginLoadingError { path: String, error: String, backtrace: String },

    #[error("the loaded PJRT plugin version is not supported by ryft; {message}")]
    PluginVersionMismatch { message: String, backtrace: String },

    #[error("{message}")]
    Cancelled { message: String, backtrace: String },

    #[error("{message}")]
    Unknown { message: String, backtrace: String },

    #[error("{message}")]
    InvalidArgument { message: String, backtrace: String },

    #[error("{message}")]
    DeadlineExceeded { message: String, backtrace: String },

    #[error("{message}")]
    NotFound { message: String, backtrace: String },

    #[error("{message}")]
    AlreadyExists { message: String, backtrace: String },

    #[error("{message}")]
    PermissionDenied { message: String, backtrace: String },

    #[error("{message}")]
    ResourceExhausted { message: String, backtrace: String },

    #[error("{message}")]
    FailedPrecondition { message: String, backtrace: String },

    #[error("{message}")]
    Aborted { message: String, backtrace: String },

    #[error("{message}")]
    OutOfRange { message: String, backtrace: String },

    #[error("{message}")]
    Unimplemented { message: String, backtrace: String },

    #[error("{message}")]
    Internal { message: String, backtrace: String },

    #[error("{message}")]
    Unavailable { message: String, backtrace: String },

    #[error("{message}")]
    DataLoss { message: String, backtrace: String },

    #[error("{message}")]
    Unauthenticated { message: String, backtrace: String },
}

impl Error {
    /// Constructs a new [`Error`] from the provided [`PJRT_Error`](ffi::PJRT_Error) handle that came
    /// from a function in the PJRT C API. Note that this function will return [`None`] if the provided
    /// [`PJRT_Error`](ffi::PJRT_Error) has a status code which represents that nothing went wrong.
    #[allow(non_upper_case_globals)]
    pub(crate) unsafe fn from_c_api(handle: *const ffi::PJRT_Error, api: Api) -> Result<Option<Self>, Self> {
        use ffi::*;

        if handle.is_null() {
            return Ok(None);
        }

        // Helper closure to make sure that the underlying PJRT error is dropped before this function returns.
        let destroy_error = || invoke_pjrt_api_void_fn!(api, PJRT_Error_Destroy, { error = handle as *mut _ });
        let message = invoke_pjrt_api_void_fn!(api, PJRT_Error_Message, { error = handle }, { message, message_size });
        let (message, message_size) = message.inspect_err(|_: &Self| drop::<Result<(), Self>>(destroy_error()))?;
        let message = str_from_c_api(message, message_size).into_owned();
        let code = invoke_pjrt_api_error_fn!(api, PJRT_Error_GetCode, { error = handle }, { code });
        let error = Some(match code.inspect_err(|_| drop(destroy_error()))? {
            PJRT_Error_Code_OK => {
                destroy_error()?;
                return Ok(None);
            }
            PJRT_Error_Code_CANCELLED => Self::cancelled(message),
            PJRT_Error_Code_UNKNOWN => Self::unknown(message),
            PJRT_Error_Code_INVALID_ARGUMENT => Self::invalid_argument(message),
            PJRT_Error_Code_DEADLINE_EXCEEDED => Self::deadline_exceeded(message),
            PJRT_Error_Code_NOT_FOUND => Self::not_found(message),
            PJRT_Error_Code_ALREADY_EXISTS => Self::already_exists(message),
            PJRT_Error_Code_PERMISSION_DENIED => Self::permission_denied(message),
            PJRT_Error_Code_RESOURCE_EXHAUSTED => Self::resource_exhausted(message),
            PJRT_Error_Code_FAILED_PRECONDITION => Self::failed_precondition(message),
            PJRT_Error_Code_ABORTED => Self::aborted(message),
            PJRT_Error_Code_OUT_OF_RANGE => Self::out_of_range(message),
            PJRT_Error_Code_UNIMPLEMENTED => Self::unimplemented(message),
            PJRT_Error_Code_INTERNAL => Self::internal(message),
            PJRT_Error_Code_UNAVAILABLE => Self::unavailable(message),
            PJRT_Error_Code_DATA_LOSS => Self::data_loss(message),
            PJRT_Error_Code_UNAUTHENTICATED => Self::unauthenticated(message),
            _ => Self::plugin_version_mismatch(message),
        });
        destroy_error()?;
        Ok(error)
    }

    /// Returns the [`PJRT_Error`](ffi::PJRT_Error) that corresponds to this [`Error`] and which can
    /// be passed to functions in the PJRT C API.
    pub(crate) fn to_c_api(&self, callback: *mut ffi::PJRT_CallbackError) -> *const ffi::PJRT_Error {
        unsafe {
            if callback.is_null() {
                panic!("no error callback function was provided");
            }
            let error_callback_fn = *callback;
            let code = self.code();
            let message = self.message();
            error_callback_fn(code, message.as_ptr(), message.count_bytes())
        }
    }

    /// Creates a new [`Error::PluginLoadingError`].
    pub fn plugin_loading_error<P: Into<String>, E: Into<String>>(path: P, error: E) -> Self {
        Self::PluginLoadingError { path: path.into(), error: error.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::PluginVersionMismatch`].
    pub fn plugin_version_mismatch<M: Into<String>>(message: M) -> Self {
        Self::PluginVersionMismatch { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::Cancelled`].
    pub fn cancelled<M: Into<String>>(message: M) -> Self {
        Self::Cancelled { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::Unknown`].
    pub fn unknown<M: Into<String>>(message: M) -> Self {
        Self::Unknown { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::InvalidArgument`].
    pub fn invalid_argument<M: Into<String>>(message: M) -> Self {
        Self::InvalidArgument { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::DeadlineExceeded`].
    pub fn deadline_exceeded<M: Into<String>>(message: M) -> Self {
        Self::DeadlineExceeded { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::NotFound`].
    pub fn not_found<M: Into<String>>(message: M) -> Self {
        Self::NotFound { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::AlreadyExists`].
    pub fn already_exists<M: Into<String>>(message: M) -> Self {
        Self::AlreadyExists { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::PermissionDenied`].
    pub fn permission_denied<M: Into<String>>(message: M) -> Self {
        Self::PermissionDenied { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::ResourceExhausted`].
    pub fn resource_exhausted<M: Into<String>>(message: M) -> Self {
        Self::ResourceExhausted { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::FailedPrecondition`].
    pub fn failed_precondition<M: Into<String>>(message: M) -> Self {
        Self::FailedPrecondition { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::Aborted`].
    pub fn aborted<M: Into<String>>(message: M) -> Self {
        Self::Aborted { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::OutOfRange`].
    pub fn out_of_range<M: Into<String>>(message: M) -> Self {
        Self::OutOfRange { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::Unimplemented`].
    pub fn unimplemented<M: Into<String>>(message: M) -> Self {
        Self::Unimplemented { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::Internal`].
    pub fn internal<M: Into<String>>(message: M) -> Self {
        Self::Internal { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::Unavailable`].
    pub fn unavailable<M: Into<String>>(message: M) -> Self {
        Self::Unavailable { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::DataLoss`].
    pub fn data_loss<M: Into<String>>(message: M) -> Self {
        Self::DataLoss { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::Unauthenticated`].
    pub fn unauthenticated<M: Into<String>>(message: M) -> Self {
        Self::Unauthenticated { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Returns the [`PJRT_Error_Code`](ffi::PJRT_Error_Code) that corresponds to this [`Error`].
    pub(crate) fn code(&self) -> ffi::PJRT_Error_Code {
        match self {
            Self::PluginLoadingError { .. } | Self::PluginVersionMismatch { .. } | Self::Unknown { .. } => {
                ffi::PJRT_Error_Code_UNKNOWN
            }
            Self::Cancelled { .. } => ffi::PJRT_Error_Code_CANCELLED,
            Self::InvalidArgument { .. } => ffi::PJRT_Error_Code_INVALID_ARGUMENT,
            Self::DeadlineExceeded { .. } => ffi::PJRT_Error_Code_DEADLINE_EXCEEDED,
            Self::NotFound { .. } => ffi::PJRT_Error_Code_NOT_FOUND,
            Self::AlreadyExists { .. } => ffi::PJRT_Error_Code_ALREADY_EXISTS,
            Self::PermissionDenied { .. } => ffi::PJRT_Error_Code_PERMISSION_DENIED,
            Self::ResourceExhausted { .. } => ffi::PJRT_Error_Code_RESOURCE_EXHAUSTED,
            Self::FailedPrecondition { .. } => ffi::PJRT_Error_Code_FAILED_PRECONDITION,
            Self::Aborted { .. } => ffi::PJRT_Error_Code_ABORTED,
            Self::OutOfRange { .. } => ffi::PJRT_Error_Code_OUT_OF_RANGE,
            Self::Unimplemented { .. } => ffi::PJRT_Error_Code_UNIMPLEMENTED,
            Self::Internal { .. } => ffi::PJRT_Error_Code_INTERNAL,
            Self::Unavailable { .. } => ffi::PJRT_Error_Code_UNAVAILABLE,
            Self::DataLoss { .. } => ffi::PJRT_Error_Code_DATA_LOSS,
            Self::Unauthenticated { .. } => ffi::PJRT_Error_Code_UNAUTHENTICATED,
        }
    }

    /// Returns the message that is stored in this [`Error`] as a [`std::ffi::CString`].
    pub(crate) fn message(&self) -> std::ffi::CString {
        match self {
            Self::PluginLoadingError { path, error, .. } => {
                let message = format!("failed to load plugin from {path:?}; {error}");
                std::ffi::CString::new(message.as_str()).unwrap()
            }
            Self::PluginVersionMismatch { message, .. } => {
                let message = format!("plugin version mismatch; {message}");
                std::ffi::CString::new(message.as_str()).unwrap()
            }
            Self::Cancelled { message, .. }
            | Self::Unknown { message, .. }
            | Self::InvalidArgument { message, .. }
            | Self::DeadlineExceeded { message, .. }
            | Self::NotFound { message, .. }
            | Self::AlreadyExists { message, .. }
            | Self::PermissionDenied { message, .. }
            | Self::ResourceExhausted { message, .. }
            | Self::FailedPrecondition { message, .. }
            | Self::Aborted { message, .. }
            | Self::OutOfRange { message, .. }
            | Self::Unimplemented { message, .. }
            | Self::Internal { message, .. }
            | Self::Unavailable { message, .. }
            | Self::DataLoss { message, .. }
            | Self::Unauthenticated { message, .. } => std::ffi::CString::new(message.as_str()).unwrap(),
        }
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::ffi::PJRT_Extension_Base;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_Error {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Error_Message_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub error: *const PJRT_Error,
        pub message: *const std::ffi::c_char,
        pub message_size: usize,
    }

    impl PJRT_Error_Message_Args {
        pub fn new(error: *const PJRT_Error) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                error,
                message: std::ptr::null_mut(),
                message_size: 0,
            }
        }
    }

    pub type PJRT_Error_Message = unsafe extern "C" fn(args: *mut PJRT_Error_Message_Args);

    pub type PJRT_Error_Code = std::ffi::c_uint;
    pub const PJRT_Error_Code_OK: PJRT_Error_Code = 0;
    pub const PJRT_Error_Code_CANCELLED: PJRT_Error_Code = 1;
    pub const PJRT_Error_Code_UNKNOWN: PJRT_Error_Code = 2;
    pub const PJRT_Error_Code_INVALID_ARGUMENT: PJRT_Error_Code = 3;
    pub const PJRT_Error_Code_DEADLINE_EXCEEDED: PJRT_Error_Code = 4;
    pub const PJRT_Error_Code_NOT_FOUND: PJRT_Error_Code = 5;
    pub const PJRT_Error_Code_ALREADY_EXISTS: PJRT_Error_Code = 6;
    pub const PJRT_Error_Code_PERMISSION_DENIED: PJRT_Error_Code = 7;
    pub const PJRT_Error_Code_RESOURCE_EXHAUSTED: PJRT_Error_Code = 8;
    pub const PJRT_Error_Code_FAILED_PRECONDITION: PJRT_Error_Code = 9;
    pub const PJRT_Error_Code_ABORTED: PJRT_Error_Code = 10;
    pub const PJRT_Error_Code_OUT_OF_RANGE: PJRT_Error_Code = 11;
    pub const PJRT_Error_Code_UNIMPLEMENTED: PJRT_Error_Code = 12;
    pub const PJRT_Error_Code_INTERNAL: PJRT_Error_Code = 13;
    pub const PJRT_Error_Code_UNAVAILABLE: PJRT_Error_Code = 14;
    pub const PJRT_Error_Code_DATA_LOSS: PJRT_Error_Code = 15;
    pub const PJRT_Error_Code_UNAUTHENTICATED: PJRT_Error_Code = 16;

    #[repr(C)]
    pub struct PJRT_Error_GetCode_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub error: *const PJRT_Error,
        pub code: PJRT_Error_Code,
    }

    impl PJRT_Error_GetCode_Args {
        pub fn new(error: *const PJRT_Error) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), error, code: 0 }
        }
    }

    pub type PJRT_Error_GetCode = unsafe extern "C" fn(args: *mut PJRT_Error_GetCode_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Error_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub error: *mut PJRT_Error,
    }

    impl PJRT_Error_Destroy_Args {
        pub fn new(error: *mut PJRT_Error) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), error }
        }
    }

    pub type PJRT_Error_Destroy = unsafe extern "C" fn(args: *mut PJRT_Error_Destroy_Args);

    /// Callback function that can be passed to certain PJRT C API functions and which will be invoked
    /// if and when an error is encountered to enable constructing and returning a [`PJRT_Error`] to PJRT.
    pub type PJRT_CallbackError = unsafe extern "C" fn(
        code: PJRT_Error_Code,
        message: *const std::ffi::c_char,
        message_size: usize,
    ) -> *mut PJRT_Error;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error() {
        let errors = [
            Error::plugin_version_mismatch("version"),
            Error::cancelled("cancelled"),
            Error::unknown("unknown"),
            Error::invalid_argument("invalid argument"),
            Error::deadline_exceeded("deadline exceeded"),
            Error::not_found("not found"),
            Error::already_exists("already exists"),
            Error::permission_denied("permission denied"),
            Error::resource_exhausted("resource exhausted"),
            Error::failed_precondition("failed precondition"),
            Error::aborted("aborted"),
            Error::out_of_range("out of range"),
            Error::unimplemented("unimplemented"),
            Error::internal("internal"),
            Error::unavailable("unavailable"),
            Error::data_loss("data loss"),
            Error::unauthenticated("unauthenticated"),
        ];

        for (i, error_i) in errors.iter().enumerate() {
            for (j, error_j) in errors.iter().enumerate() {
                if i == j {
                    assert_eq!(error_i, error_j);
                    assert_eq!(error_i.clone(), error_j.clone());
                } else {
                    assert_ne!(error_i, error_j);
                }
            }
        }

        assert_eq!(errors[0].code(), ffi::PJRT_Error_Code_UNKNOWN);
        assert_eq!(errors[1].code(), ffi::PJRT_Error_Code_CANCELLED);
        assert_eq!(errors[2].code(), ffi::PJRT_Error_Code_UNKNOWN);
        assert_eq!(errors[3].code(), ffi::PJRT_Error_Code_INVALID_ARGUMENT);
        assert_eq!(errors[4].code(), ffi::PJRT_Error_Code_DEADLINE_EXCEEDED);
        assert_eq!(errors[5].code(), ffi::PJRT_Error_Code_NOT_FOUND);
        assert_eq!(errors[6].code(), ffi::PJRT_Error_Code_ALREADY_EXISTS);
        assert_eq!(errors[7].code(), ffi::PJRT_Error_Code_PERMISSION_DENIED);
        assert_eq!(errors[8].code(), ffi::PJRT_Error_Code_RESOURCE_EXHAUSTED);
        assert_eq!(errors[9].code(), ffi::PJRT_Error_Code_FAILED_PRECONDITION);
        assert_eq!(errors[10].code(), ffi::PJRT_Error_Code_ABORTED);
        assert_eq!(errors[11].code(), ffi::PJRT_Error_Code_OUT_OF_RANGE);
        assert_eq!(errors[12].code(), ffi::PJRT_Error_Code_UNIMPLEMENTED);
        assert_eq!(errors[13].code(), ffi::PJRT_Error_Code_INTERNAL);
        assert_eq!(errors[14].code(), ffi::PJRT_Error_Code_UNAVAILABLE);
        assert_eq!(errors[15].code(), ffi::PJRT_Error_Code_DATA_LOSS);
        assert_eq!(errors[16].code(), ffi::PJRT_Error_Code_UNAUTHENTICATED);

        assert_eq!(errors[0].message().to_str().unwrap(), "plugin version mismatch; version");
        assert_eq!(errors[1].message().to_str().unwrap(), "cancelled");
        assert_eq!(errors[2].message().to_str().unwrap(), "unknown");
        assert_eq!(errors[3].message().to_str().unwrap(), "invalid argument");
        assert_eq!(errors[4].message().to_str().unwrap(), "deadline exceeded");
        assert_eq!(errors[5].message().to_str().unwrap(), "not found");
        assert_eq!(errors[6].message().to_str().unwrap(), "already exists");
        assert_eq!(errors[7].message().to_str().unwrap(), "permission denied");
        assert_eq!(errors[8].message().to_str().unwrap(), "resource exhausted");
        assert_eq!(errors[9].message().to_str().unwrap(), "failed precondition");
        assert_eq!(errors[10].message().to_str().unwrap(), "aborted");
        assert_eq!(errors[11].message().to_str().unwrap(), "out of range");
        assert_eq!(errors[12].message().to_str().unwrap(), "unimplemented");
        assert_eq!(errors[13].message().to_str().unwrap(), "internal");
        assert_eq!(errors[14].message().to_str().unwrap(), "unavailable");
        assert_eq!(errors[15].message().to_str().unwrap(), "data loss");
        assert_eq!(errors[16].message().to_str().unwrap(), "unauthenticated");
    }

    #[test]
    fn test_error_display_and_debug() {
        let error = Error::invalid_argument("bad input");
        assert_eq!(format!("{error}"), "bad input");
        let debug = format!("{error:?}");
        assert!(debug.starts_with("InvalidArgument { message: \"bad input\", backtrace: \""));

        let error = Error::plugin_loading_error("/path", "err");
        assert_eq!(format!("{error}"), "error while loading a PJRT plugin from '/path': err");
        let debug = format!("{error:?}");
        assert!(debug.starts_with("PluginLoadingError { path: \"/path\", error: \"err\", backtrace: \""));
    }
}
