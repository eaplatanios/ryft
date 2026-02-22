use prost::Message;

use crate::protos::{ProfileOptions, XSpace};
use crate::{Api, Client, Error, Plugin, slice_from_c_api, str_from_c_api};

/// Helper macro for invoking PJRT profiler plugin API functions that may return
/// [`PLUGIN_Profiler_Error`](ffi::PLUGIN_Profiler_Error) errors. This is analogous to
/// [`invoke_pjrt_api_error_fn!`](crate::invoke_pjrt_api_error_fn) but handles the profiler
/// extension's separate error type.
///
/// # Parameters
///
///   - `$extension`: Expression that evaluates to the [`ProfilerExtension`] that must be used to invoke the desired
///     PJRT profiler C API function.
///   - `$fn`: Name of the PJRT profiler C API function to invoke (e.g., `PLUGIN_Profiler_Create`).
///   - `$input_name = $input_value`: Zero or more input argument assignments that correspond
///     to fields in the corresponding `<$fn>_Args` struct in the PJRT profiler C API.
///   - `$output_name`: Zero or more output field names to extract from the `<$fn>_Args` struct after the
///     PJRT C API function invocation.
macro_rules! invoke_profiler_api_error_fn {
    (
        $extension:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? } $(,)?
    ) => {
        invoke_profiler_api_error_fn!(
            $extension,
            $fn,
            { $($input_name = $input_value),* },
            {},
        )
    };
    (
        $extension:expr,
        $fn:ident,
        { $($input_name:ident = $input_value:expr),* $(,)? },
        { $($output_name:ident),* $(,)? } $(,)?
    ) => {{
        paste::paste! {
            unsafe {
                let profiler_api = (*$extension.to_c_api()).profiler_api;
                let api_fn = (*profiler_api).$fn.ok_or_else(|| crate::errors::Error::unimplemented(format!(
                    "`{}` is not implemented in the loaded PJRT plugin (version {})",
                    stringify!($fn).to_owned(),
                    $extension.api().version(),
                )));
                match api_fn {
                    Ok(api_fn) => {
                        let mut args = ffi::[<$fn _Args>]::new($($input_value),*);
                        let error = api_fn(&mut args as *mut _);
                        match profiler_error_to_error(error, profiler_api, stringify!($fn)) {
                            Ok(None) => Ok(($(args.$output_name),*)),
                            Ok(Some(error)) => Err(error),
                            Err(error) => Err(error),
                        }
                    },
                    Err(error) => Err(error),
                }
            }
        }
    }};
}

/// The PJRT profiler extension provides capabilities for backend-specific profiling of PJRT operations. The extension
/// is both optional for PJRT [`Plugin`]s and _experimental_, meaning that incompatible changes may be introduced at any
/// time, including changes that break _Application Binary Interface (ABI)_ compatibility.
#[derive(Copy, Clone)]
pub struct ProfilerExtension {
    /// Handle that represents this [`ProfilerExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_Profiler_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl ProfilerExtension {
    /// Constructs a new [`ProfilerExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle if the type of that PJRT
    /// extension matches the PJRT profiler extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_Profiler {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_Profiler_Extension`](ffi::PJRT_Profiler_Extension) that corresponds
    /// to this [`ProfilerExtension`] and which can be passed to functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_Profiler_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for ProfilerExtension {}
unsafe impl Sync for ProfilerExtension {}

impl Client<'_> {
    /// Attempts to load the [`ProfilerExtension`] from this [`Client`] and returns
    /// [`Error::Unimplemented`] if it is not provided by the underlying [`Plugin`].
    pub fn profiler_extension(&self) -> Result<ProfilerExtension, Error> {
        self.api().profiler_extension()
    }

    /// Creates a new [`Profiler`] that can be used to profile PJRT operations. Refer to the documentation of
    /// [`Profiler`] for information on how to use the returned [`Profiler`].
    pub fn profiler(&self, options: &ProfileOptions) -> Result<Profiler, Error> {
        self.api().profiler(options)
    }
}

impl Plugin {
    /// Attempts to load the [`ProfilerExtension`] from this [`Plugin`] and returns
    /// [`Error::Unimplemented`] if it is not provided by this [`Plugin`].
    pub fn profiler_extension(&self) -> Result<ProfilerExtension, Error> {
        self.api().profiler_extension()
    }

    /// Creates a new [`Profiler`] that can be used to profile PJRT operations. Refer to the documentation of
    /// [`Profiler`] for information on how to use the returned [`Profiler`].
    pub fn profiler(&self, options: &ProfileOptions) -> Result<Profiler, Error> {
        self.api().profiler(options)
    }
}

impl Api {
    /// Attempts to load the [`ProfilerExtension`] from this [`Api`] and returns
    /// [`Error::Unimplemented`] if it is not provided by the underlying [`Plugin`].
    pub(crate) fn profiler_extension(&self) -> Result<ProfilerExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let profiler_extension = ProfilerExtension::from_c_api(extension, *self);
                if let Some(profiler_extension) = profiler_extension {
                    return Ok(profiler_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the profiler extension is not provided by the PJRT plugin"))
        }
    }

    /// Creates a new [`Profiler`] that can be used to profile PJRT operations. Refer to the documentation of
    /// [`Profiler`] for information on how to use the returned [`Profiler`].
    pub(crate) fn profiler(&self, options: &ProfileOptions) -> Result<Profiler, Error> {
        use prost::Message;
        let extension = self.profiler_extension()?;
        let options = options.encode_to_vec();
        invoke_profiler_api_error_fn!(
            extension,
            PLUGIN_Profiler_Create,
            {
                options = options.as_ptr() as *const _,
                options_size = options.len(),
            },
            { profiler },
        )
        .and_then(|handle| unsafe { Profiler::from_c_api(handle, extension) })
    }
}

/// Profiler that can be used to profile PJRT program execution. A [`Profiler`] captures trace data from PJRT
/// [`Plugin`]s during program execution. The typical profiling workflow is as follows:
///
///   1. **Create** a [`Profiler`] via [`Client::profiler`] or [`Plugin::profiler`], providing [`ProfileOptions`]
///      to configure the profiling session (e.g., host tracing level, device tracing level, etc.).
///   2. **Start** profiling via [`Profiler::start`]. All PJRT operations performed after this call (and before
///      calling [`Profiler::stop`]) will be captured by registered host and device tracers.
///   3. **Perform** PJRT operations (e.g., compile and execute programs, transfer buffers, etc.).
///   4. **Stop** profiling via [`Profiler::stop`].
///   5. **Collect** the resulting trace data via [`Profiler::results`], which returns an [`XSpace`] Protocol buffer
///      containing the profiling results.
///
/// The collected `XSpace` data can be saved as `.xplane.pb` files for visualization in
/// [XProf](https://github.com/openxla/xprof) and [TensorBoard](https://www.tensorflow.org/tensorboard),
/// or analyzed programmatically.
pub struct Profiler {
    /// Handle that represents this [`Profiler`] in the PJRT C API.
    handle: *mut ffi::PLUGIN_Profiler,

    /// [`ProfilerExtension`] used to create this [`Profiler`].
    extension: ProfilerExtension,
}

impl Profiler {
    /// Constructs a new [`Profiler`] from the provided [`PLUGIN_Profiler`](ffi::PLUGIN_Profiler)
    /// handle that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(
        handle: *mut ffi::PLUGIN_Profiler,
        extension: ProfilerExtension,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT profiler handle is a null pointer"))
        } else {
            Ok(Self { handle, extension })
        }
    }

    /// Returns the [`PLUGIN_Profiler`](ffi::PLUGIN_Profiler) that corresponds to this [`Profiler`]
    /// and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PLUGIN_Profiler {
        self.handle
    }

    /// Starts profiling capture for this [`Profiler`]. All PJRT operations performed after this call (and before
    /// calling [`Profiler::stop`]) will be recorded by the plugin's registered host and device tracers (e.g., CUPTI for
    /// NVIDIA GPUs, ROCProfiler for AMD GPUs). Refer to the documentation of [`Profiler`] for information on what a
    /// typical profiling workflow looks like.
    pub fn start(&self) -> Result<(), Error> {
        invoke_profiler_api_error_fn!(self.extension, PLUGIN_Profiler_Start, { profiler = self.to_c_api() })
    }

    /// Stops profiling capture for this [`Profiler`]. After this call, no further operations will be recorded.
    /// The captured trace data can then be retrieved via [`Profiler::results`]. Refer to the documentation of
    /// [`Profiler`] for information on what a typical profiling workflow looks like.
    pub fn stop(&self) -> Result<(), Error> {
        invoke_profiler_api_error_fn!(self.extension, PLUGIN_Profiler_Stop, { profiler = self.to_c_api() })
    }

    /// Returns the profiling results captured by this [`Profiler`] as an [`XSpace`].
    pub fn results(&self) -> Result<XSpace, Error> {
        // The C API header documents a two-pass protocol (first call with null buffer to get the size and then call
        // with the allocated buffer to fill), but the actual XLA implementation handles everything in a single call
        // when `args->buffer` is null: it allocates an internal buffer of `serialized_size + 1` bytes, serializes
        // the [`XSpace`] Protobuf message, and sets `args.buffer` and `args.buffer_size_in_bytes` to point to this
        // internal storage. A second call with a non-null buffer is a no-op. The internal buffer remains valid until
        // the profiler is destroyed but here we decode the resulting [`XSpace`] anyway and so we do not need to worry
        // about its lifetime.
        invoke_profiler_api_error_fn!(
            self.extension,
            PLUGIN_Profiler_CollectData,
            { profiler = self.to_c_api() },
            { buffer, buffer_size_in_bytes },
        )
        .and_then(|(buffer, buffer_size_in_bytes)| {
            XSpace::decode(unsafe { slice_from_c_api(buffer, buffer_size_in_bytes.saturating_sub(1)) }).map_err(
                |error| {
                    Error::invalid_argument(format!(
                        "failed to deserialize the profiling trace Protobuf returned by PJRT plugin: {error}",
                    ))
                },
            )
        })
    }
}

unsafe impl Send for Profiler {}
unsafe impl Sync for Profiler {}

impl Drop for Profiler {
    fn drop(&mut self) {
        invoke_profiler_api_error_fn!(self.extension, PLUGIN_Profiler_Destroy, { profiler = self.to_c_api() },)
            .expect("failed to destroy PJRT profiler");
    }
}

/// Internal helper function that converts PJRT profiler errors to [`Error`]s.
#[allow(non_upper_case_globals)]
unsafe fn profiler_error_to_error(
    error: *mut ffi::PLUGIN_Profiler_Error,
    api: *mut ffi::PLUGIN_Profiler_Api,
    function_name: &str,
) -> Result<Option<Error>, Error> {
    use crate::errors::ffi::*;

    if error.is_null() {
        return Ok(None);
    }

    if api.is_null() {
        return Err(Error::invalid_argument("the provided PJRT profiler API handle is a null pointer"));
    }

    let api = unsafe { &*api };

    // Helper closure to make sure that the underlying PJRT profiler error is dropped before this function returns.
    let destroy_error = || {
        if let Some(destroy_fn) = api.PLUGIN_Profiler_Error_Destroy {
            let mut args = ffi::PLUGIN_Profiler_Error_Destroy_Args::new(error);
            args.state = api.state;
            unsafe { destroy_fn(&mut args as *mut _) };
        }
    };

    let error_message = if let Some(error_message_fn) = api.PLUGIN_Profiler_Error_Message {
        let mut args = ffi::PLUGIN_Profiler_Error_Message_Args::new(error);
        args.state = api.state;
        unsafe { error_message_fn(&mut args as *mut _) };
        let message = str_from_c_api(args.message, args.message_size).into_owned();
        if message.is_empty() { "unknown PJRT profiler error".to_string() } else { message }
    } else {
        "unknown PJRT profiler error".to_string()
    };

    let error_code = if let Some(error_code_fn) = api.PLUGIN_Profiler_Error_GetCode {
        let mut args = ffi::PLUGIN_Profiler_Error_GetCode_Args::new(error);
        args.state = api.state;
        match unsafe {
            profiler_error_to_error(error_code_fn(&mut args as *mut _), api as *const _ as *mut _, function_name)
        } {
            Ok(None) => Ok(if args.code >= 0 { args.code as u32 } else { PJRT_Error_Code_UNKNOWN }),
            Ok(Some(error)) | Err(error) => Err(error),
        }
    } else {
        Ok(PJRT_Error_Code_UNKNOWN)
    };

    let error_message = format!("failed to invoke `{function_name}`; {error_message}");
    let error = Some(match error_code.inspect_err(|_| destroy_error())? {
        PJRT_Error_Code_OK => {
            destroy_error();
            return Ok(None);
        }
        PJRT_Error_Code_CANCELLED => Error::cancelled(error_message),
        PJRT_Error_Code_UNKNOWN => Error::unknown(error_message),
        PJRT_Error_Code_INVALID_ARGUMENT => Error::invalid_argument(error_message),
        PJRT_Error_Code_DEADLINE_EXCEEDED => Error::deadline_exceeded(error_message),
        PJRT_Error_Code_NOT_FOUND => Error::not_found(error_message),
        PJRT_Error_Code_ALREADY_EXISTS => Error::already_exists(error_message),
        PJRT_Error_Code_PERMISSION_DENIED => Error::permission_denied(error_message),
        PJRT_Error_Code_RESOURCE_EXHAUSTED => Error::resource_exhausted(error_message),
        PJRT_Error_Code_FAILED_PRECONDITION => Error::failed_precondition(error_message),
        PJRT_Error_Code_ABORTED => Error::aborted(error_message),
        PJRT_Error_Code_OUT_OF_RANGE => Error::out_of_range(error_message),
        PJRT_Error_Code_UNIMPLEMENTED => Error::unimplemented(error_message),
        PJRT_Error_Code_INTERNAL => Error::internal(error_message),
        PJRT_Error_Code_UNAVAILABLE => Error::unavailable(error_message),
        PJRT_Error_Code_DATA_LOSS => Error::data_loss(error_message),
        PJRT_Error_Code_UNAUTHENTICATED => Error::unauthenticated(error_message),
        _ => Error::plugin_version_mismatch(error_message),
    });
    destroy_error();
    Ok(error)
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::ffi::PJRT_Extension_Base;

    pub const PJRT_API_PROFILER_EXTENSION_VERSION: usize = 1;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PLUGIN_Profiler_Error {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PLUGIN_Profiler_Error_Message_Args {
        pub struct_size: usize,
        pub state: *mut std::ffi::c_void,
        pub error: *const PLUGIN_Profiler_Error,
        pub message: *const std::ffi::c_char,
        pub message_size: usize,
    }

    impl PLUGIN_Profiler_Error_Message_Args {
        pub fn new(error: *const PLUGIN_Profiler_Error) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                state: std::ptr::null_mut(),
                error,
                message: std::ptr::null(),
                message_size: 0,
            }
        }
    }

    pub type PLUGIN_Profiler_Error_Message = unsafe extern "C" fn(args: *mut PLUGIN_Profiler_Error_Message_Args);

    #[repr(C)]
    pub struct PLUGIN_Profiler_Error_GetCode_Args {
        pub struct_size: usize,
        pub state: *mut std::ffi::c_void,
        pub error: *const PLUGIN_Profiler_Error,
        pub code: std::ffi::c_int,
    }

    impl PLUGIN_Profiler_Error_GetCode_Args {
        pub fn new(error: *const PLUGIN_Profiler_Error) -> Self {
            Self { struct_size: size_of::<Self>(), state: std::ptr::null_mut(), error, code: 0 }
        }
    }

    pub type PLUGIN_Profiler_Error_GetCode =
        unsafe extern "C" fn(args: *mut PLUGIN_Profiler_Error_GetCode_Args) -> *mut PLUGIN_Profiler_Error;

    #[repr(C)]
    pub struct PLUGIN_Profiler_Error_Destroy_Args {
        pub struct_size: usize,
        pub state: *mut std::ffi::c_void,
        pub error: *mut PLUGIN_Profiler_Error,
    }

    impl PLUGIN_Profiler_Error_Destroy_Args {
        pub fn new(error: *mut PLUGIN_Profiler_Error) -> Self {
            Self { struct_size: size_of::<Self>(), state: std::ptr::null_mut(), error }
        }
    }

    pub type PLUGIN_Profiler_Error_Destroy = unsafe extern "C" fn(args: *mut PLUGIN_Profiler_Error_Destroy_Args);

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PLUGIN_Profiler {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PLUGIN_Profiler_Create_Args {
        pub struct_size: usize,
        pub options: *const std::ffi::c_char,
        pub options_size: usize,
        pub profiler: *mut PLUGIN_Profiler,
    }

    impl PLUGIN_Profiler_Create_Args {
        pub fn new(options: *const std::ffi::c_char, options_size: usize) -> Self {
            Self { struct_size: size_of::<Self>(), options, options_size, profiler: std::ptr::null_mut() }
        }
    }

    pub type PLUGIN_Profiler_Create =
        unsafe extern "C" fn(args: *mut PLUGIN_Profiler_Create_Args) -> *mut PLUGIN_Profiler_Error;

    #[repr(C)]
    pub struct PLUGIN_Profiler_Start_Args {
        pub struct_size: usize,
        pub profiler: *mut PLUGIN_Profiler,
    }

    impl PLUGIN_Profiler_Start_Args {
        pub fn new(profiler: *mut PLUGIN_Profiler) -> Self {
            Self { struct_size: size_of::<Self>(), profiler }
        }
    }

    pub type PLUGIN_Profiler_Start =
        unsafe extern "C" fn(args: *mut PLUGIN_Profiler_Start_Args) -> *mut PLUGIN_Profiler_Error;

    #[repr(C)]
    pub struct PLUGIN_Profiler_Stop_Args {
        pub struct_size: usize,
        pub profiler: *mut PLUGIN_Profiler,
    }

    impl PLUGIN_Profiler_Stop_Args {
        pub fn new(profiler: *mut PLUGIN_Profiler) -> Self {
            Self { struct_size: size_of::<Self>(), profiler }
        }
    }

    pub type PLUGIN_Profiler_Stop =
        unsafe extern "C" fn(args: *mut PLUGIN_Profiler_Stop_Args) -> *mut PLUGIN_Profiler_Error;

    #[repr(C)]
    pub struct PLUGIN_Profiler_CollectData_Args {
        pub struct_size: usize,
        pub profiler: *mut PLUGIN_Profiler,
        pub buffer: *mut u8,
        pub buffer_size_in_bytes: usize,
    }

    impl PLUGIN_Profiler_CollectData_Args {
        pub fn new(profiler: *mut PLUGIN_Profiler) -> Self {
            Self { struct_size: size_of::<Self>(), profiler, buffer: std::ptr::null_mut(), buffer_size_in_bytes: 0 }
        }
    }

    pub type PLUGIN_Profiler_CollectData =
        unsafe extern "C" fn(args: *mut PLUGIN_Profiler_CollectData_Args) -> *mut PLUGIN_Profiler_Error;

    #[repr(C)]
    pub struct PLUGIN_Profiler_Destroy_Args {
        pub struct_size: usize,
        pub profiler: *mut PLUGIN_Profiler,
    }

    impl PLUGIN_Profiler_Destroy_Args {
        pub fn new(profiler: *mut PLUGIN_Profiler) -> Self {
            Self { struct_size: size_of::<Self>(), profiler }
        }
    }

    pub type PLUGIN_Profiler_Destroy =
        unsafe extern "C" fn(args: *mut PLUGIN_Profiler_Destroy_Args) -> *mut PLUGIN_Profiler_Error;

    #[repr(C)]
    pub struct PLUGIN_Profiler_Api {
        pub struct_size: usize,
        pub state: *mut std::ffi::c_void,
        pub PLUGIN_Profiler_Error_Destroy: Option<PLUGIN_Profiler_Error_Destroy>,
        pub PLUGIN_Profiler_Error_Message: Option<PLUGIN_Profiler_Error_Message>,
        pub PLUGIN_Profiler_Error_GetCode: Option<PLUGIN_Profiler_Error_GetCode>,
        pub PLUGIN_Profiler_Create: Option<PLUGIN_Profiler_Create>,
        pub PLUGIN_Profiler_Destroy: Option<PLUGIN_Profiler_Destroy>,
        pub PLUGIN_Profiler_Start: Option<PLUGIN_Profiler_Start>,
        pub PLUGIN_Profiler_Stop: Option<PLUGIN_Profiler_Stop>,
        pub PLUGIN_Profiler_CollectData: Option<PLUGIN_Profiler_CollectData>,
    }

    #[repr(C)]
    pub struct PJRT_Profiler_Extension {
        pub base: PJRT_Extension_Base,
        pub profiler_api: *mut PLUGIN_Profiler_Api,
        pub trace_me_context_id: i64,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;

    use crate::protos::{
        CompilationOptions, ExecutableCompilationOptions, Precision, ProfileDeviceType, ProfileOptions,
    };
    use crate::tests::{TestPlatform, test_for_each_platform};
    use crate::{BufferType, Error, ExecutionDeviceInputs, ExecutionInput, Program};

    #[test]
    fn test_profiler_extension() {
        test_for_each_platform!(|plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    assert!(plugin.profiler_extension().is_ok());
                    assert!(client.profiler_extension().is_ok());
                }
                _ => {
                    assert!(matches!(plugin.profiler_extension(), Err(Error::Unimplemented { .. })));
                    assert!(matches!(client.profiler_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }

    #[test]
    fn test_profiler() {
        test_for_each_platform!(|plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    // Create a profiler and start profiling.
                    let options = ProfileOptions {
                        version: 1,
                        device_type: ProfileDeviceType::Cpu as i32,
                        host_tracing_level: 2,
                        device_tracing_level: 1,
                        ..Default::default()
                    };
                    let profiler = client.profiler(&options).expect("failed to create PJRT profiler");
                    assert!(profiler.start().is_ok());

                    // Create, compile, and execute a simple StableHLO addition program, while profiling everything.
                    let program = Program::Mlir {
                        bytecode: indoc! {"
                            module {
                              func.func @main(%arg0: tensor<2x1xi32>, %arg1: tensor<2x1xi32>) -> tensor<2x1xi32> {
                                %0 = stablehlo.add %arg0, %arg1 : tensor<2x1xi32>
                                return %0 : tensor<2x1xi32>
                              }
                            }
                        "}
                        .as_bytes()
                        .to_vec(),
                    };
                    let options = CompilationOptions {
                        argument_layouts: Vec::new(),
                        parameter_is_tupled_arguments: false,
                        executable_build_options: Some(ExecutableCompilationOptions {
                            device_ordinal: -1,
                            replica_count: 1,
                            partition_count: 1,
                            ..Default::default()
                        }),
                        compile_portable_executable: false,
                        profile_version: 0,
                        serialized_multi_slice_configuration: Vec::new(),
                        environment_option_overrides: HashMap::new(),
                        target_config: None,
                        allow_in_place_mlir_modification: false,
                        matrix_unit_operand_precision: Precision::Default as i32,
                    };
                    let executable = client.compile(&program, &options).unwrap();
                    let device = executable.addressable_devices().unwrap()[0].clone();
                    let mut lhs_bytes = Vec::with_capacity(8);
                    lhs_bytes.extend_from_slice(&7i32.to_ne_bytes());
                    lhs_bytes.extend_from_slice(&(-1i32).to_ne_bytes());
                    let lhs_buffer = client
                        .buffer(lhs_bytes.as_slice(), BufferType::I32, &[2u64, 1], None, device.clone(), None)
                        .unwrap();
                    let mut rhs_bytes = Vec::with_capacity(8);
                    rhs_bytes.extend_from_slice(&35i32.to_ne_bytes());
                    rhs_bytes.extend_from_slice(&(-41i32).to_ne_bytes());
                    let rhs_buffer = client
                        .buffer(rhs_bytes.as_slice(), BufferType::I32, &[2u64, 1], None, device.clone(), None)
                        .unwrap();
                    let inputs = ExecutionDeviceInputs {
                        inputs: &[
                            ExecutionInput { buffer: lhs_buffer, donatable: false },
                            ExecutionInput { buffer: rhs_buffer, donatable: false },
                        ],
                        ..Default::default()
                    };
                    let mut outputs = executable.execute(vec![inputs], 0, None, None, None, None).unwrap();
                    assert_eq!(outputs.len(), 1);
                    let mut outputs = outputs.remove(0);
                    outputs.done.r#await().unwrap();
                    let output = outputs.outputs.remove(0);
                    let output_bytes = output.copy_to_host(None).unwrap().r#await().unwrap();
                    let mut expected_output_bytes = Vec::with_capacity(8);
                    expected_output_bytes.extend_from_slice(&42i32.to_ne_bytes());
                    expected_output_bytes.extend_from_slice(&(-42i32).to_ne_bytes());
                    assert_eq!(output_bytes, expected_output_bytes);

                    // Stop the profiling and verify that there were some profiling data that were collected.
                    assert!(profiler.stop().is_ok());
                    let results = profiler.results().expect("failed to collect profiling results");
                    assert!(results.errors.is_empty());
                }
                _ => {
                    assert!(matches!(plugin.profiler_extension(), Err(Error::Unimplemented { .. })));
                    assert!(matches!(client.profiler_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }
}
