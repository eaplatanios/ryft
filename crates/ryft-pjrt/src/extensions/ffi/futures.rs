use crate::{invoke_xla_ffi_api_error_fn, invoke_xla_ffi_api_void_fn};

use crate::extensions::ffi::errors::FfiError;
use crate::extensions::ffi::handlers::FfiApi;

/// [`FfiFuture`]s provide a mechanism to signal the completion of asynchronous computations
/// in XLA FFI handlers back to the XLA runtime.
pub struct FfiFuture {
    /// Handle that represents this [`FfiFuture`] in the XLA FFI API.
    handle: *mut ffi::XLA_FFI_Future,

    /// Underlying XLA [`FfiApi`].
    api: FfiApi,
}

impl FfiFuture {
    /// Constructs a new [`FfiFuture`] from the provided [`XLA_FFI_Future`](ffi::XLA_FFI_Future)
    /// that came from a function in the XLA FFI API.
    pub unsafe fn from_c_api(handle: *mut ffi::XLA_FFI_Future, api: FfiApi) -> Result<Self, FfiError> {
        if handle.is_null() {
            Err(FfiError::invalid_argument("the provided XLA FFI future handle is a null pointer"))
        } else {
            Ok(Self { handle, api })
        }
    }

    /// Returns the [`XLA_FFI_Future`](ffi::XLA_FFI_Future) that corresponds to this [`FfiFuture`]
    /// and which can be passed to functions in the XLA FFI API.
    pub unsafe fn to_c_api(&self) -> *mut ffi::XLA_FFI_Future {
        self.handle
    }

    /// Marks this [`FfiFuture`] as having completed successfully.
    pub fn set_available(&self) -> Result<(), FfiError> {
        use ffi::XLA_FFI_Future_SetAvailable_Args;
        invoke_xla_ffi_api_error_fn!(self.api, XLA_FFI_Future_SetAvailable, { future = self.handle })
    }

    /// Marks this [`FfiFuture`] as having failed with the provided [`FfiError`].
    pub fn set_error(&self, error: FfiError) -> Result<(), FfiError> {
        use ffi::XLA_FFI_Future_SetError_Args;
        let error = unsafe { error.to_c_api(self.api) };
        invoke_xla_ffi_api_error_fn!(self.api, XLA_FFI_Future_SetError, { future = self.handle, error = error })
            .inspect_err(|_| {
                use crate::extensions::ffi::errors::ffi::XLA_FFI_Error_Destroy_Args;
                let _ = invoke_xla_ffi_api_void_fn!(self.api, XLA_FFI_Error_Destroy, { error = error });
            })
    }
}

impl FfiApi {
    /// Creates a new [`FfiFuture`].
    pub fn future(&self) -> Result<FfiFuture, FfiError> {
        use ffi::XLA_FFI_Future_Create_Args;
        let future = invoke_xla_ffi_api_error_fn!(*self, XLA_FFI_Future_Create, {}, { future })?;
        unsafe { FfiFuture::from_c_api(future, *self) }
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::extensions::ffi::errors::ffi::XLA_FFI_Error;
    use crate::extensions::ffi::handlers::ffi::XLA_FFI_Extension_Base;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct XLA_FFI_Future {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct XLA_FFI_Future_Create_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub future: *mut XLA_FFI_Future,
    }

    impl XLA_FFI_Future_Create_Args {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), future: std::ptr::null_mut() }
        }
    }

    pub type XLA_FFI_Future_Create = unsafe extern "C" fn(args: *mut XLA_FFI_Future_Create_Args) -> *mut XLA_FFI_Error;

    #[repr(C)]
    pub struct XLA_FFI_Future_SetAvailable_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub future: *mut XLA_FFI_Future,
    }

    impl XLA_FFI_Future_SetAvailable_Args {
        pub fn new(future: *mut XLA_FFI_Future) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), future }
        }
    }

    pub type XLA_FFI_Future_SetAvailable =
        unsafe extern "C" fn(args: *mut XLA_FFI_Future_SetAvailable_Args) -> *mut XLA_FFI_Error;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct XLA_FFI_Future_SetError_Args {
        pub struct_size: usize,
        pub extension_start: *mut XLA_FFI_Extension_Base,
        pub future: *mut XLA_FFI_Future,
        pub error: *mut XLA_FFI_Error,
    }

    impl XLA_FFI_Future_SetError_Args {
        pub fn new(future: *mut XLA_FFI_Future, error: *mut XLA_FFI_Error) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), future, error }
        }
    }

    pub type XLA_FFI_Future_SetError =
        unsafe extern "C" fn(args: *mut XLA_FFI_Future_SetError_Args) -> *mut XLA_FFI_Error;
}

#[cfg(test)]
mod tests {
    use crate::extensions::ffi::errors::FfiError;
    use crate::extensions::ffi::tests::test_ffi_api;

    use super::FfiFuture;

    #[test]
    fn test_ffi_future() {
        let api = test_ffi_api();

        let future = api.future().unwrap();
        assert!(future.set_available().is_ok());
        assert!(future.set_error(FfiError::invalid_argument("boom")).is_ok());
        assert!(!unsafe { future.to_c_api() }.is_null());

        let future = unsafe { FfiFuture::from_c_api(std::ptr::null_mut(), api) };
        assert!(matches!(future, Err(FfiError::InvalidArgument { .. })));
    }
}
