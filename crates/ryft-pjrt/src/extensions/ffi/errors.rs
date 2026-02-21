use std::backtrace::Backtrace;

use thiserror::Error;

use crate::extensions::ffi::handlers::FfiApi;
use crate::macros::invoke_xla_ffi_api_void_fn;

/// Represents errors that can occur when interacting with the XLA FFI C API. The error types are based on the
/// [Abseil status codes](https://abseil.io/docs/cpp/guides/status-codes) which XLA FFI uses internally.
///
/// Each variant includes a `backtrace` field that captures the call stack at the point where the error was created,
/// which is useful for debugging. Note that it is represented as a [`String`] and not as a [`Backtrace`] because using
/// the latter is only currently supported in unstable Rust.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FfiError {
    #[error("the loaded XLA FFI API version is not supported by ryft; {message}")]
    ApiVersionMismatch { message: String, backtrace: String },

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

impl FfiError {
    /// Constructs a new [`FfiError`] from the provided [`XLA_FFI_Error`](ffi::XLA_FFI_Error) handle that came
    /// from a function in the XLA FFI API. Note that this function will return [`None`] if the provided
    /// [`XLA_FFI_Error`](ffi::XLA_FFI_Error) has an empty error message.
    ///
    /// Note that due to limitations in the XLA FFI API, we cannot extract an error code from the provided
    /// [`XLA_FFI_Error`](ffi::XLA_FFI_Error) and thus, if the error message is non-empty, it will always be
    /// converted to an [`FfiError::Unknown`].
    pub unsafe fn from_c_api(handle: *mut ffi::XLA_FFI_Error, api: FfiApi) -> Result<Option<Self>, Self> {
        use ffi::*;

        if handle.is_null() {
            return Ok(None);
        }

        // Helper closure to make sure that the underlying PJRT error is dropped before this function returns.
        let destroy_error = || invoke_xla_ffi_api_void_fn!(api, XLA_FFI_Error_Destroy, { error = handle as *mut _ });
        let message = invoke_xla_ffi_api_void_fn!(api, XLA_FFI_Error_GetMessage, { error = handle }, { message });
        let message = message.inspect_err(|_: &Self| drop::<Result<(), Self>>(destroy_error()))?;
        let message = if message.is_null() {
            return Ok(None);
        } else {
            unsafe { std::ffi::CStr::from_ptr(message) }.to_string_lossy().into_owned()
        };
        Ok(Some(Self::unknown(message)))
    }

    /// Returns the [`XLA_FFI_Error`](ffi::XLA_FFI_Error) that corresponds to this [`FfiError`] and which can
    /// be returned from an [`XLA_FFI_Handler`](ffi::XLA_FFI_Handler).
    pub unsafe fn to_c_api(&self, api: FfiApi) -> *mut ffi::XLA_FFI_Error {
        unsafe {
            let error_create_fn = (*api.to_c_api())
                .XLA_FFI_Error_Create
                .unwrap_or_else(|| panic!("the provided XLA FFI API is missing the `XLA_FFI_Error_Create` function"));
            let code = self.code();
            let message = self.message();
            let mut args = ffi::XLA_FFI_Error_Create_Args::new(message.as_ptr(), code);
            error_create_fn(&mut args as *mut _)
        }
    }

    /// Creates a new [`FfiError::ApiVersionMismatch`].
    pub fn api_version_mismatch<M: Into<String>>(message: M) -> Self {
        Self::ApiVersionMismatch { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::Cancelled`].
    pub fn cancelled<M: Into<String>>(message: M) -> Self {
        Self::Cancelled { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::Unknown`].
    pub fn unknown<M: Into<String>>(message: M) -> Self {
        Self::Unknown { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::InvalidArgument`].
    pub fn invalid_argument<M: Into<String>>(message: M) -> Self {
        Self::InvalidArgument { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::DeadlineExceeded`].
    pub fn deadline_exceeded<M: Into<String>>(message: M) -> Self {
        Self::DeadlineExceeded { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::NotFound`].
    pub fn not_found<M: Into<String>>(message: M) -> Self {
        Self::NotFound { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::AlreadyExists`].
    pub fn already_exists<M: Into<String>>(message: M) -> Self {
        Self::AlreadyExists { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::PermissionDenied`].
    pub fn permission_denied<M: Into<String>>(message: M) -> Self {
        Self::PermissionDenied { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::ResourceExhausted`].
    pub fn resource_exhausted<M: Into<String>>(message: M) -> Self {
        Self::ResourceExhausted { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::FailedPrecondition`].
    pub fn failed_precondition<M: Into<String>>(message: M) -> Self {
        Self::FailedPrecondition { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::Aborted`].
    pub fn aborted<M: Into<String>>(message: M) -> Self {
        Self::Aborted { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::OutOfRange`].
    pub fn out_of_range<M: Into<String>>(message: M) -> Self {
        Self::OutOfRange { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::Unimplemented`].
    pub fn unimplemented<M: Into<String>>(message: M) -> Self {
        Self::Unimplemented { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::Internal`].
    pub fn internal<M: Into<String>>(message: M) -> Self {
        Self::Internal { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::Unavailable`].
    pub fn unavailable<M: Into<String>>(message: M) -> Self {
        Self::Unavailable { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::DataLoss`].
    pub fn data_loss<M: Into<String>>(message: M) -> Self {
        Self::DataLoss { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`FfiError::Unauthenticated`].
    pub fn unauthenticated<M: Into<String>>(message: M) -> Self {
        Self::Unauthenticated { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Returns the [`XLA_FFI_Error_Code`](ffi::XLA_FFI_Error_Code) that corresponds to this [`FfiError`].
    pub(crate) fn code(&self) -> ffi::XLA_FFI_Error_Code {
        match self {
            Self::ApiVersionMismatch { .. } | Self::Unknown { .. } => ffi::XLA_FFI_Error_Code_UNKNOWN,
            Self::Cancelled { .. } => ffi::XLA_FFI_Error_Code_CANCELLED,
            Self::InvalidArgument { .. } => ffi::XLA_FFI_Error_Code_INVALID_ARGUMENT,
            Self::DeadlineExceeded { .. } => ffi::XLA_FFI_Error_Code_DEADLINE_EXCEEDED,
            Self::NotFound { .. } => ffi::XLA_FFI_Error_Code_NOT_FOUND,
            Self::AlreadyExists { .. } => ffi::XLA_FFI_Error_Code_ALREADY_EXISTS,
            Self::PermissionDenied { .. } => ffi::XLA_FFI_Error_Code_PERMISSION_DENIED,
            Self::ResourceExhausted { .. } => ffi::XLA_FFI_Error_Code_RESOURCE_EXHAUSTED,
            Self::FailedPrecondition { .. } => ffi::XLA_FFI_Error_Code_FAILED_PRECONDITION,
            Self::Aborted { .. } => ffi::XLA_FFI_Error_Code_ABORTED,
            Self::OutOfRange { .. } => ffi::XLA_FFI_Error_Code_OUT_OF_RANGE,
            Self::Unimplemented { .. } => ffi::XLA_FFI_Error_Code_UNIMPLEMENTED,
            Self::Internal { .. } => ffi::XLA_FFI_Error_Code_INTERNAL,
            Self::Unavailable { .. } => ffi::XLA_FFI_Error_Code_UNAVAILABLE,
            Self::DataLoss { .. } => ffi::XLA_FFI_Error_Code_DATA_LOSS,
            Self::Unauthenticated { .. } => ffi::XLA_FFI_Error_Code_UNAUTHENTICATED,
        }
    }

    /// Returns the message that is stored in this [`FfiError`] as a [`std::ffi::CString`].
    pub(crate) fn message(&self) -> std::ffi::CString {
        match self {
            Self::ApiVersionMismatch { message, .. } => {
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

    use crate::extensions::ffi::handlers::ffi::XLA_FFI_Extension_Base;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct XLA_FFI_Error {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    pub type XLA_FFI_Error_GetMessage = unsafe extern "C" fn(args: *mut XLA_FFI_Error_GetMessage_Args);

    pub type XLA_FFI_Error_Code = std::ffi::c_uint;
    pub const XLA_FFI_Error_Code_OK: XLA_FFI_Error_Code = 0;
    pub const XLA_FFI_Error_Code_CANCELLED: XLA_FFI_Error_Code = 1;
    pub const XLA_FFI_Error_Code_UNKNOWN: XLA_FFI_Error_Code = 2;
    pub const XLA_FFI_Error_Code_INVALID_ARGUMENT: XLA_FFI_Error_Code = 3;
    pub const XLA_FFI_Error_Code_DEADLINE_EXCEEDED: XLA_FFI_Error_Code = 4;
    pub const XLA_FFI_Error_Code_NOT_FOUND: XLA_FFI_Error_Code = 5;
    pub const XLA_FFI_Error_Code_ALREADY_EXISTS: XLA_FFI_Error_Code = 6;
    pub const XLA_FFI_Error_Code_PERMISSION_DENIED: XLA_FFI_Error_Code = 7;
    pub const XLA_FFI_Error_Code_RESOURCE_EXHAUSTED: XLA_FFI_Error_Code = 8;
    pub const XLA_FFI_Error_Code_FAILED_PRECONDITION: XLA_FFI_Error_Code = 9;
    pub const XLA_FFI_Error_Code_ABORTED: XLA_FFI_Error_Code = 10;
    pub const XLA_FFI_Error_Code_OUT_OF_RANGE: XLA_FFI_Error_Code = 11;
    pub const XLA_FFI_Error_Code_UNIMPLEMENTED: XLA_FFI_Error_Code = 12;
    pub const XLA_FFI_Error_Code_INTERNAL: XLA_FFI_Error_Code = 13;
    pub const XLA_FFI_Error_Code_UNAVAILABLE: XLA_FFI_Error_Code = 14;
    pub const XLA_FFI_Error_Code_DATA_LOSS: XLA_FFI_Error_Code = 15;
    pub const XLA_FFI_Error_Code_UNAUTHENTICATED: XLA_FFI_Error_Code = 16;

    #[repr(C)]
    pub struct XLA_FFI_Error_Create_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub message: *const std::ffi::c_char,
        pub error_code: XLA_FFI_Error_Code,
    }

    impl XLA_FFI_Error_Create_Args {
        pub fn new(message: *const std::ffi::c_char, code: XLA_FFI_Error_Code) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), message, error_code: code }
        }
    }

    pub type XLA_FFI_Error_Create = unsafe extern "C" fn(args: *mut XLA_FFI_Error_Create_Args) -> *mut XLA_FFI_Error;

    #[repr(C)]
    pub struct XLA_FFI_Error_GetMessage_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub error: *mut XLA_FFI_Error,
        pub message: *const std::ffi::c_char,
    }

    impl XLA_FFI_Error_GetMessage_Args {
        pub fn new(error: *mut XLA_FFI_Error) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                error,
                message: std::ptr::null(),
            }
        }
    }

    #[repr(C)]
    pub struct XLA_FFI_Error_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub error: *mut XLA_FFI_Error,
    }

    impl XLA_FFI_Error_Destroy_Args {
        pub fn new(error: *mut XLA_FFI_Error) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), error }
        }
    }

    pub type XLA_FFI_Error_Destroy = unsafe extern "C" fn(args: *mut XLA_FFI_Error_Destroy_Args);
}

#[cfg(test)]
mod tests {
    use super::{FfiError, ffi};

    #[test]
    fn test_error() {
        let errors = [
            FfiError::api_version_mismatch("version"),
            FfiError::cancelled("cancelled"),
            FfiError::unknown("unknown"),
            FfiError::invalid_argument("invalid argument"),
            FfiError::deadline_exceeded("deadline exceeded"),
            FfiError::not_found("not found"),
            FfiError::already_exists("already exists"),
            FfiError::permission_denied("permission denied"),
            FfiError::resource_exhausted("resource exhausted"),
            FfiError::failed_precondition("failed precondition"),
            FfiError::aborted("aborted"),
            FfiError::out_of_range("out of range"),
            FfiError::unimplemented("unimplemented"),
            FfiError::internal("internal"),
            FfiError::unavailable("unavailable"),
            FfiError::data_loss("data loss"),
            FfiError::unauthenticated("unauthenticated"),
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

        assert_eq!(errors[0].code(), ffi::XLA_FFI_Error_Code_UNKNOWN);
        assert_eq!(errors[1].code(), ffi::XLA_FFI_Error_Code_CANCELLED);
        assert_eq!(errors[2].code(), ffi::XLA_FFI_Error_Code_UNKNOWN);
        assert_eq!(errors[3].code(), ffi::XLA_FFI_Error_Code_INVALID_ARGUMENT);
        assert_eq!(errors[4].code(), ffi::XLA_FFI_Error_Code_DEADLINE_EXCEEDED);
        assert_eq!(errors[5].code(), ffi::XLA_FFI_Error_Code_NOT_FOUND);
        assert_eq!(errors[6].code(), ffi::XLA_FFI_Error_Code_ALREADY_EXISTS);
        assert_eq!(errors[7].code(), ffi::XLA_FFI_Error_Code_PERMISSION_DENIED);
        assert_eq!(errors[8].code(), ffi::XLA_FFI_Error_Code_RESOURCE_EXHAUSTED);
        assert_eq!(errors[9].code(), ffi::XLA_FFI_Error_Code_FAILED_PRECONDITION);
        assert_eq!(errors[10].code(), ffi::XLA_FFI_Error_Code_ABORTED);
        assert_eq!(errors[11].code(), ffi::XLA_FFI_Error_Code_OUT_OF_RANGE);
        assert_eq!(errors[12].code(), ffi::XLA_FFI_Error_Code_UNIMPLEMENTED);
        assert_eq!(errors[13].code(), ffi::XLA_FFI_Error_Code_INTERNAL);
        assert_eq!(errors[14].code(), ffi::XLA_FFI_Error_Code_UNAVAILABLE);
        assert_eq!(errors[15].code(), ffi::XLA_FFI_Error_Code_DATA_LOSS);
        assert_eq!(errors[16].code(), ffi::XLA_FFI_Error_Code_UNAUTHENTICATED);

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
        let error = FfiError::invalid_argument("bad input");
        assert_eq!(format!("{error}"), "bad input");
        let debug = format!("{error:?}");
        assert!(debug.starts_with("InvalidArgument { message: \"bad input\", backtrace: \""));
    }
}
