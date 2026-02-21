/// Low-level helper macro for invoking PJRT C API functions. This macro handles the boilerplate of looking up a
/// function pointer in the [`PJRT_Api`](crate::ffi::PJRT_Api) struct, constructing the appropriate `*_Args` struct,
/// invoking the function, and extracting any output values. It also takes care of error handling by checking the C
/// API error returned by the function (if that function returned an error pointer) and converting it to a Rust
/// [`Error`](crate::Error) value if something went wrong. Note that if the requested function is not available in
/// the loaded plugin, this macro will generate code that returns an
/// [`Error::Unimplemented`](crate::Error::Unimplemented) error.
///
/// This macro is not intended to be used directly. Instead, use [`invoke_pjrt_api_void_fn!`] for functions that
/// do not return errors, or [`invoke_pjrt_api_error_fn!`] for functions that may return errors.
///
/// # Parameters
///
///   - `$api`: API instance that provides access to PJRT C API function pointers. The type that this expression
///      evaluates to must provide an `api()` function that returns an [`Api`](crate::Api) instance. This is typically
///      either an [`Api`](crate::Api) instance itself or a PJRT extension instance. Note that you can also optionally
///      use the `@unchecked` keyword prefix if you want to skip checking that `$fn` exists based on the underlying PJRT
///      API struct size. This check is based on how PJRT handles versioning, though it is not currently supported for
///      PJRT extensions and that is why we support the optional `@unchecked` keyword prefix.
///   - `$fn`: Name of the PJRT C API function to invoke (e.g., `PJRT_Client_Create`).
///   - `$input_name = $input_value`: Zero or more input argument assignments that correspond
///     to fields in the corresponding `<$fn>_Args` struct in the PJRT C API.
///   - `$output_name`: Zero or more output field names to extract from the `<$fn>_Args` struct after the
///     PJRT C API function invocation.
macro_rules! invoke_pjrt_api_fn_helper {
    (
        $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {
        paste::paste! {
            {
                // TODO(eaplatanios): If there are any extensions that support this kind of checking,
                //  we will need to figure out how to support them here.
                let api_handle = unsafe { $api.to_c_api() };
                let api_fn_offset = std::mem::offset_of!(crate::ffi::PJRT_Api, $fn);
                let api_struct_size = unsafe { (*api_handle).struct_size } as usize;
                if api_struct_size <= api_fn_offset {
                    Err(crate::errors::Error::unimplemented(format!(
                        "`{}` is not available in the loaded PJRT plugin (version {})",
                        stringify!($fn).to_owned(),
                        $api.api().version(),
                    )))
                } else {
                    $crate::invoke_pjrt_api_fn_helper!(
                        @unchecked $api,
                        $fn,
                        { $($input_name = $input_value),* },
                        { $($output_name),* },
                    )
                }
            }
        }
    };
    (
        @unchecked $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {
        paste::paste! {
            unsafe {
                let api_fn = (*$api.to_c_api()).$fn.ok_or_else(|| crate::errors::Error::unimplemented(format!(
                    "`{}` is not implemented in the loaded PJRT plugin (version {})",
                    stringify!($fn).to_owned(),
                    $api.api().version(),
                )));
                match api_fn {
                    Ok(api_fn) => {
                        let mut args = [<$fn _Args>]::new($($input_value),*);
                        let error = api_fn(&mut args as *mut _);
                        Ok((($(args.$output_name),*), error))
                    },
                    Err(error) => Err(error),
                }
            }
        }
    };
}

/// Helper used for invoking PJRT C API functions that cannot return errors. Use this macro for PJRT C API functions
/// that have a `void` return type. For functions that have a `PJRT_Error*` return type and require error handling,
/// use the [`invoke_pjrt_api_error_fn!`] macro instead.
///
/// This macro is a wrapper over [`invoke_pjrt_api_fn_helper!`].
///
/// # Parameters
///
///   - `$api`: API instance that provides access to PJRT C API function pointers. The type that this expression
///      evaluates to must provide an `api()` function that returns an [`Api`](crate::Api) instance. This is typically
///      either an [`Api`](crate::Api) instance itself or a PJRT extension instance.Note that you can also optionally
///      use the `@unchecked` keyword prefix if you want to skip checking that `$fn` exists based on the underlying PJRT
///      API struct size. This check is based on how PJRT handles versioning, though it is not currently supported for
///      PJRT extensions and that is why we support the optional `@unchecked` keyword prefix.
///   - `$fn`: Name of the PJRT C API function to invoke (e.g., `PJRT_Client_Create`).
///   - `$input_name = $input_value`: Zero or more input argument assignments that correspond
///     to fields in the corresponding `<$fn>_Args` struct in the PJRT C API.
///   - `$output_name`: Zero or more output field names to extract from the `<$fn>_Args` struct after the
///     PJRT C API function invocation.
macro_rules! invoke_pjrt_api_void_fn {
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident $(,)?
    ) => {
        $crate::invoke_pjrt_api_void_fn!(
            $(@$unchecked)? $api,
            $fn,
            {},
            {},
        )
    };
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? } $(,)?
    ) => {
        $crate::invoke_pjrt_api_void_fn!(
            $(@$unchecked)? $api,
            $fn,
            { $($input_name = $input_value),* },
            {},
        )
    };
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {
        $crate::invoke_pjrt_api_fn_helper!(
            $(@$unchecked)? $api,
            $fn,
            { $($input_name = $input_value),* },
            { $($output_name),* },
        ).map(|(outputs, _)| outputs)
    };
}

/// Helper used for invoking PJRT C API functions that may return errors. Use this macro for PJRT C API functions
/// that have a `PJRT_Error*` return type and require error handling. For functions that have a `void` return type,
/// use the [`invoke_pjrt_api_void_fn!`] macro instead.
///
/// This macro is a wrapper over [`invoke_pjrt_api_fn_helper!`].
///
/// # Parameters
///
///   - `$api`: API instance that provides access to PJRT C API function pointers. The type that this expression
///      evaluates to must provide an `api()` function that returns an [`Api`](crate::Api) instance. This is typically
///      either an [`Api`](crate::Api) instance itself or a PJRT extension instance. Note that you can also optionally
///      use the `@unchecked` keyword prefix if you want to skip checking that `$fn` exists based on the underlying PJRT
///      API struct size. This check is based on how PJRT handles versioning, though it is not currently supported for
///      PJRT extensions and that is why we support the optional `@unchecked` keyword prefix.
///   - `$fn`: Name of the PJRT C API function to invoke (e.g., `PJRT_Client_Create`).
///   - `$input_name = $input_value`: Zero or more input argument assignments that correspond
///     to fields in the corresponding `<$fn>_Args` struct in the PJRT C API.
///   - `$output_name`: Zero or more output field names to extract from the `<$fn>_Args` struct after the
///     PJRT C API function invocation.
macro_rules! invoke_pjrt_api_error_fn {
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident $(,)?
    ) => {
        $crate::invoke_pjrt_api_error_fn!(
            $(@$unchecked)? $api,
            $fn,
            {},
            {},
        )
    };
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? } $(,)?
    ) => {
        $crate::invoke_pjrt_api_error_fn!(
            $(@$unchecked)? $api,
            $fn,
            { $($input_name = $input_value),* },
            {},
        )
    };
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {{
        $crate::invoke_pjrt_api_fn_helper!(
            $(@$unchecked)? $api,
            $fn,
            { $($input_name = $input_value),* },
            { $($output_name),* },
        ).and_then(|(outputs, error)| {
            if error.is_null() {
                Ok(outputs)
            } else {
                unsafe {
                    match $crate::Error::from_c_api(error, $api.api()) {
                        Ok(None) => Ok(outputs),
                        Ok(Some(error)) => Err(error),
                        Err(error) => Err(error),
                    }
                }
            }
        })
    }};
}

/// Low-level helper macro for invoking XLA distributed runtime C API functions.
///
/// This macro is not intended to be used directly. Instead, use [`invoke_distributed_api_void_fn!`] for functions
/// with a `void` return type or [`invoke_distributed_api_error_fn!`] for functions that return `PJRT_Error*`.
///
/// # Parameters
///
///   - `$fn`: Name of the XLA distributed runtime C API function to invoke
///      (e.g., `PJRT_Distributed_Runtime_Client_Connect`).
///   - `$input_name = $input_value`: Zero or more input argument assignments that correspond to fields in the
///     corresponding `<$fn>_Args` struct in the XLA distributed runtime C API.
///   - `$output_name`: Zero or more output field names to extract from the `<$fn>_Args` struct after the XLA
///     distributed runtime C API function invocation.
macro_rules! invoke_distributed_api_fn_helper {
    (
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {{
        paste::paste! {
            unsafe {
                let mut args = ryft_xla_sys::distributed::[<$fn _Args>]::new($($input_value),*);
                let result = ryft_xla_sys::distributed::$fn(&mut args as *mut _);
                (($(args.$output_name),*), result)
            }
        }
    }};
}

/// Helper used for invoking XLA distributed runtime C API functions that cannot return errors. Use this macro for
/// XLA distributed runtime C API functions that have a `void` return type. For functions that have a `PJRT_Error*`
/// return type and require error handling, use the [`invoke_distributed_api_error_fn!`] macro instead.
///
/// This macro is a wrapper over [`invoke_distributed_api_fn_helper!`].
///
/// # Parameters
///
///   - `$api`: API instance associated with the distributed runtime object being used. This parameter exists for API
///      symmetry with [`invoke_distributed_api_error_fn!`].
///   - `$fn`: Name of the XLA distributed runtime C API function to invoke
///      (e.g., `PJRT_Distributed_Runtime_Service_Shutdown`).
///   - `$input_name = $input_value`: Zero or more input argument assignments that correspond to fields in the
///     corresponding `<$fn>_Args` struct in the XLA distributed runtime C API.
///   - `$output_name`: Zero or more output field names to extract from the `<$fn>_Args` struct after the XLA
///     distributed runtime C API function invocation.
macro_rules! invoke_distributed_api_void_fn {
    (
        $api:expr,
        $fn:ident $(,)?
    ) => {
        $crate::invoke_distributed_api_void_fn!(
            $api,
            $fn,
            {},
            {},
        )
    };
    (
        $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? } $(,)?
    ) => {
        $crate::invoke_distributed_api_void_fn!(
            $api,
            $fn,
            { $($input_name = $input_value),* },
            {},
        )
    };
    (
        $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {{
        let _ = &$api;
        $crate::invoke_distributed_api_fn_helper!(
            $fn,
            { $($input_name = $input_value),* },
            { $($output_name),* },
        ).0
    }};
}

/// Helper used for invoking XLA distributed runtime C API functions that may return errors. Use this macro for
/// XLA distributed runtime C API functions that have a `PJRT_Error*` return type and require error handling.
/// For functions that have a `void` return type, use the [`invoke_distributed_api_void_fn!`] macro instead.
///
/// This macro is a wrapper over [`invoke_distributed_api_fn_helper!`].
///
/// # Parameters
///
///   - `$api`: API instance that provides access to PJRT C API function pointers. The type that this expression
///      evaluates to must provide an `api()` function that returns an [`Api`](crate::Api) instance. This is typically
///      either an [`Api`](crate::Api) instance itself or a PJRT extension instance.
///   - `$fn`: Name of the XLA distributed runtime C API function to invoke
///      (e.g., `PJRT_Distributed_Runtime_Client_Connect`).
///   - `$input_name = $input_value`: Zero or more input argument assignments that correspond to fields in the
///     corresponding `<$fn>_Args` struct in the XLA distributed runtime C API.
///   - `$output_name`: Zero or more output field names to extract from the `<$fn>_Args` struct after the XLA
///     distributed runtime C API function invocation.
macro_rules! invoke_distributed_api_error_fn {
    (
        $api:expr,
        $fn:ident $(,)?
    ) => {
        $crate::invoke_distributed_api_error_fn!(
            $api,
            $fn,
            {},
            {},
        )
    };
    (
        $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? } $(,)?
    ) => {
        $crate::invoke_distributed_api_error_fn!(
            $api,
            $fn,
            { $($input_name = $input_value),* },
            {},
        )
    };
    (
        $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {{
        let (outputs, error) = $crate::invoke_distributed_api_fn_helper!(
            $fn,
            { $($input_name = $input_value),* },
            { $($output_name),* },
        );
        let error = error as *mut $crate::errors::ffi::PJRT_Error;
        if error.is_null() {
            Ok(outputs)
        } else {
            unsafe {
                match $crate::Error::from_c_api(error as *const _, $api) {
                    Ok(None) => Ok(outputs),
                    Ok(Some(error)) => Err(error),
                    Err(error) => Err(error),
                }
            }
        }
    }};
}

/// Low-level helper macro for invoking XLA FFI functions. This macro handles the boilerplate of looking up a
/// function pointer in the [`XLA_FFI_Api`](crate::extensions::ffi::ffi::XLA_FFI_Api) struct, constructing the
/// appropriate `*_Args` struct, invoking the function, and extracting any output values. It also takes care of error
/// handling by checking the C API error returned by the function (if that function returned an error pointer) and
/// converting it to a Rust [`FfiError`](crate::extensions::ffi::FfiError) value if something went wrong. Note that if
/// the requested function is not available in this XLA FFI API, this macro will generate code that returns an
/// [`FfiError::Unimplemented`](crate::extensions::ffi::FfiError::Unimplemented) error.
///
/// This macro is not intended to be used directly. Instead, use [`invoke_xla_ffi_api_void_fn!`] for functions that
/// do not return errors, or [`invoke_xla_ffi_api_error_fn!`] for functions that may return errors.
///
/// # Parameters
///
///   - `$api`: XLA FFI API instance that provides access to XLA FFI API function pointers.
///     The type that this expression evaluates to must provide an `api()` function that returns
///     an [`FfiApi`](crate::extensions::ffi::FfiApi) instance. Note that you can also optionally use the `@unchecked`
///     keyword prefix if you want to skip checking that `$fn` exists based on the underlying XLA FFI API struct size.
///     This check is based on how XLA FFI handles versioning.
///   - `$fn`: Name of the XLA FFI API function to invoke (e.g., `XLA_FFI_Type_Register`).
///   - `$input_name = $input_value`: Zero or more input argument assignments that correspond
///     to fields in the corresponding `<$fn>_Args` struct in the XLA FFI API.
///   - `$output_name`: Zero or more output field names to extract from the `<$fn>_Args` struct after the
///     PJRT C API function invocation.
macro_rules! invoke_xla_ffi_api_fn_helper {
    (
        $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {
        paste::paste! {
            {
                let api_handle = unsafe { $api.to_c_api() };
                let api_fn_offset = std::mem::offset_of!(crate::extensions::ffi::ffi::XLA_FFI_Api, $fn);
                let api_struct_size = unsafe { (*api_handle).struct_size } as usize;
                if api_struct_size <= api_fn_offset {
                    Err(crate::extensions::ffi::FfiError::unimplemented(format!(
                        "`{}` is not available in the loaded XLA FFI API (version {})",
                        stringify!($fn).to_owned(),
                        $api.version(),
                    )))
                } else {
                    $crate::invoke_xla_ffi_api_fn_helper!(
                        @unchecked $api,
                        $fn,
                        { $($input_name = $input_value),* },
                        { $($output_name),* },
                    )
                }
            }
        }
    };
    (
        @unchecked $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {
        paste::paste! {
            unsafe {
                let api_fn = (*$api.to_c_api()).$fn.ok_or_else(|| crate::extensions::ffi::FfiError::unimplemented(
                    format!(
                        "`{}` is not implemented in the loaded XLA FFI API (version {})",
                        stringify!($fn).to_owned(),
                        $api.version(),
                    ),
                ));
                match api_fn {
                    Ok(api_fn) => {
                        let mut args = [<$fn _Args>]::new($($input_value),*);
                        let error = api_fn(&mut args as *mut _);
                        Ok((($(args.$output_name),*), error))
                    },
                    Err(error) => Err(error),
                }
            }
        }
    };
}

/// Helper used for invoking XLA FFI API functions that cannot return errors. Use this macro for XLA FFI API functions
/// that have a `void` return type. For functions that have an `XLA_FFI_Error*` return type and require error handling,
/// use the [`invoke_xla_ffi_api_error_fn!`] macro instead.
///
/// This macro is a wrapper over [`invoke_xla_ffi_api_fn_helper!`].
///
/// # Parameters
///
///   - `$api`: XLA FFI API instance that provides access to XLA FFI API function pointers.
///     The type that this expression evaluates to must provide an `api()` function that returns
///     an [`FfiApi`](crate::extensions::ffi::FfiApi) instance. Note that you can also optionally use the `@unchecked`
///     keyword prefix if you want to skip checking that `$fn` exists based on the underlying XLA FFI API struct size.
///     This check is based on how XLA FFI handles versioning.
///   - `$fn`: Name of the XLA FFI API function to invoke (e.g., `XLA_FFI_Type_Register`).
///   - `$input_name = $input_value`: Zero or more input argument assignments that correspond
///     to fields in the corresponding `<$fn>_Args` struct in the XLA FFI API.
///   - `$output_name`: Zero or more output field names to extract from the `<$fn>_Args` struct after the
///     PJRT C API function invocation.
macro_rules! invoke_xla_ffi_api_void_fn {
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident $(,)?
    ) => {
        $crate::invoke_xla_ffi_api_void_fn!(
            $(@$unchecked)? $api,
            $fn,
            {},
            {},
        )
    };
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? } $(,)?
    ) => {
        $crate::invoke_xla_ffi_api_void_fn!(
            $(@$unchecked)? $api,
            $fn,
            { $($input_name = $input_value),* },
            {},
        )
    };
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {
        $crate::invoke_xla_ffi_api_fn_helper!(
            $(@$unchecked)? $api,
            $fn,
            { $($input_name = $input_value),* },
            { $($output_name),* },
        ).map(|(outputs, _)| outputs)
    };
}

/// Helper used for invoking XLA FFI API functions that may return errors. Use this macro for XLA FFI API functions
/// that have an `XLA_FFI_Error*` return type and require error handling.
///
/// This macro is a wrapper over [`invoke_xla_ffi_api_fn_helper!`].
///
/// # Parameters
///
///   - `$api`: XLA FFI API instance that provides access to XLA FFI API function pointers.
///     The type that this expression evaluates to must provide an `api()` function that returns
///     an [`FfiApi`](crate::extensions::ffi::FfiApi) instance. Note that you can also optionally use the `@unchecked`
///     keyword prefix if you want to skip checking that `$fn` exists based on the underlying XLA FFI API struct size.
///     This check is based on how XLA FFI handles versioning.
///   - `$fn`: Name of the XLA FFI API function to invoke (e.g., `XLA_FFI_Type_Register`).
///   - `$input_name = $input_value`: Zero or more input argument assignments that correspond
///     to fields in the corresponding `<$fn>_Args` struct in the XLA FFI API.
///   - `$output_name`: Zero or more output field names to extract from the `<$fn>_Args` struct after the
///     PJRT C API function invocation.
macro_rules! invoke_xla_ffi_api_error_fn {
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident $(,)?
    ) => {
        $crate::invoke_xla_ffi_api_error_fn!(
            $(@$unchecked)? $api,
            $fn,
            {},
            {},
        )
    };
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? } $(,)?
    ) => {
        $crate::invoke_xla_ffi_api_error_fn!(
            $(@$unchecked)? $api,
            $fn,
            { $($input_name = $input_value),* },
            {},
        )
    };
    (
        $(@$unchecked:tt)? $api:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {{
        $crate::invoke_xla_ffi_api_fn_helper!(
            $(@$unchecked)? $api,
            $fn,
            { $($input_name = $input_value),* },
            { $($output_name),* },
        ).and_then(|(outputs, error)| {
            if error.is_null() {
                Ok(outputs)
            } else {
                unsafe {
                    match $crate::extensions::ffi::FfiError::from_c_api(error, $api) {
                        Ok(None) => Ok(outputs),
                        Ok(Some(error)) => Err(error),
                        Err(error) => Err(error),
                    }
                }
            }
        })
    }};
}

pub(crate) use {
    invoke_distributed_api_error_fn, invoke_distributed_api_fn_helper, invoke_distributed_api_void_fn,
    invoke_pjrt_api_error_fn, invoke_pjrt_api_fn_helper, invoke_pjrt_api_void_fn, invoke_xla_ffi_api_error_fn,
    invoke_xla_ffi_api_fn_helper, invoke_xla_ffi_api_void_fn,
};
